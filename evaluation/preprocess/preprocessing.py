#!/usr/bin/env python3
"""CLI for converting external benchmarks into fixture JSON.

New evaluation code should import from ``evaluation.utils``. This module is
also available as a command-line tool for fixture-generation scripts and parser
regression tests.

Supports three input formats:

  1. ONNX file (+ optional VNNLib spec) — used by `safenlp/*` and
     VNN-COMP LunarLander. We extract MatMul + Add + ReLU
     chains, fold any constant `Sub` (input shift) into the first
     linear layer's bias, parse the VNNLib input box, and synthesize
     a property matrix that rules out the VNNLib output assertions.
     VNNLib files in VNN-COMP style typically encode counterexample
     regions; PANDA proves those regions are unreachable.

  2. Fairproof JSON triplet (`weights.json`, `layer_sizes.json`,
     `inputpoint.json`) — single-file emit with an `epsilon` box
     around the input point and an identity-output property.

  3. CROWN original Keras/HDF5 models inside `models_crown.tar`.
     The archive contains model weights only, so the `crown`
     subcommand also needs an input center and target classes to build
     a real PANDA verification query.

Output schema (consumed by `src/file_formats.rs` / the `panda_prove` binary):

```json
{
    "name": "...",
    "activation": "relu" | "sigmoid" | "tanh",
    "input_dim": N,
    "output_dim": M,
    "weights": [[[...]]],     // per-layer (out_dim, in_dim)
    "biases":  [[...]],        // per-layer (out_dim,)
    "x_lower": [...],
    "x_upper": [...],
    "spec_c":  [[...]],
    "spec_d":  [...],
    "side":    "lower" | "upper" | "both",
    "precision_bits": <int>
}
```

`precision_bits` has NO default anywhere in this module: every converter
entry point requires it explicitly (per-model values live in
`evaluation/quant_params/`), and the SNARK's range-table width the codes
must fit into is a runtime table parameter, not a compile-time constant.

Soundness note: the converter treats VNNLib output assertions as
counterexample/unsafe-region constraints. This matches VNN-COMP-style
queries, where the `.vnnlib` file typically encodes the counterexample
region to rule out.

Usage:
    uv run python -m evaluation.preprocess.preprocessing onnx <model.onnx> <vnnlib_or_-> <output.json> precision_bits=<n>
    uv run python -m evaluation.preprocess.preprocessing fairproof <fairproof_dir> <output.json> precision_bits=<n>
    uv run python -m evaluation.preprocess.preprocessing crown <models_crown.tar> <archive_member> <output.json> center=<input.json> true_class=<i> target_class=<j> precision_bits=<n>
"""

import json
import re
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

NUM_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _broadcast_prefix_const(const: np.ndarray, in_dim: int) -> np.ndarray:
    """Broadcast an input-normalization constant to the flattened input.

    ERAN-style graphs normalize per channel: the constant has shape
    (1, C, 1, 1) while the flattened NCHW input has C·H·W coordinates,
    so each channel value repeats H·W times (channel-major order).
    Full-size and scalar constants pass through unchanged.
    """
    flat = const.reshape(-1)
    if flat.size == in_dim:
        return flat
    if flat.size == 1:
        return np.full(in_dim, float(flat[0]))
    if in_dim % flat.size == 0:
        return np.repeat(flat, in_dim // flat.size)
    raise ValueError(
        f"normalization constant of size {flat.size} does not broadcast "
        f"to input dim {in_dim}"
    )


def _onnx_layers_to_mlp(model, allow_trailing_activation=False):
    """Walk an ONNX Model proto and extract a flat MLP.

    Recognizes:
      - a constant input-normalization prefix before the first linear
        layer — any composition of `Sub` (x - mean) and `Div` (x / std),
        with constants supplied as initializers or `Constant` nodes
        (per-channel constants broadcast over the flattened input) —
        folded exactly into the first layer's weights and bias
      - `Flatten` / `Reshape` (no-ops for our flat-MLP view)
      - alternating `MatMul` + `Add` + `ReLU` blocks and `Gemm` blocks
      - trailing linear layer (no activation after)

    ``allow_trailing_activation``: ERAN's fully-connected nets apply the
    activation after EVERY linear layer, including the last. Our layer
    schema only carries activations between adjacent linear layers, so
    when enabled a trailing activation is preserved exactly by appending
    an identity output layer (W=I, b=0) after it.

    Raises if the graph has any other op (e.g., Conv — implementable
    but requires a specific benchmark to test against).
    """
    import onnx

    inits = {
        init.name: onnx.numpy_helper.to_array(init).astype(np.float64)
        for init in model.graph.initializer
    }
    # `Constant` nodes are just initializers spelled as ops (ERAN's
    # exports use them for the normalization mean/std).
    for node in model.graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    inits[node.output[0]] = onnx.numpy_helper.to_array(a.t).astype(
                        np.float64
                    )

    # Walk the graph. The input-normalization prefix is tracked as the
    # affine map x' = pre_scale ⊙ x + pre_shift (composed in graph
    # order) and folded into the first linear layer:
    # W·x' + b = (W ⊙ pre_scale)·x + (b + W·pre_shift).
    pre_scale: np.ndarray | float = 1.0
    pre_shift: np.ndarray | float = 0.0
    pre_seen = False
    layers_w = []  # list of (W, b) tuples
    activations = []  # list of activation kinds between adjacent linear layers
    pending_w = None  # accumulating MatMul output before Add

    def fold_prefix(W: np.ndarray, b: np.ndarray):
        nonlocal pre_scale, pre_shift, pre_seen
        if not pre_seen:
            return W, b
        in_dim = W.shape[1]
        scale = _broadcast_prefix_const(np.asarray(pre_scale, dtype=np.float64), in_dim)
        shift = _broadcast_prefix_const(np.asarray(pre_shift, dtype=np.float64), in_dim)
        W_folded = W * scale[None, :]
        b_folded = b + W @ shift
        pre_seen = False
        pre_scale, pre_shift = 1.0, 0.0
        return W_folded, b_folded

    for node in model.graph.node:
        op = node.op_type
        attr = {a.name: a for a in node.attribute}
        if op == "Constant":
            pass  # collected into `inits` above
        elif op == "Sub":
            # %x = Sub(%input, %const): x' = x - m
            if layers_w or pending_w is not None:
                raise ValueError("Sub after the first linear layer not supported")
            const_name = [n for n in node.input if n in inits]
            if not const_name:
                raise ValueError("Sub with no constant operand not supported")
            m = inits[const_name[0]].reshape(-1)
            pre_shift = np.asarray(pre_shift) - m
            pre_seen = True
        elif op == "Div":
            # %x = Div(%input, %const): x' = x / s
            if layers_w or pending_w is not None:
                raise ValueError("Div after the first linear layer not supported")
            const_name = [n for n in node.input if n in inits]
            if not const_name:
                raise ValueError("Div with no constant operand not supported")
            s = inits[const_name[0]].reshape(-1)
            pre_scale = np.asarray(pre_scale) / s
            pre_shift = np.asarray(pre_shift) / s
            pre_seen = True
        elif op in ("Flatten", "Reshape"):
            pass  # no-op for our flat-MLP view
        elif op == "MatMul":
            const_name = [n for n in node.input if n in inits]
            if len(const_name) != 1:
                raise ValueError("MatMul with non-constant weight not supported")
            W = inits[const_name[0]]
            if W.ndim != 2:
                raise ValueError(f"MatMul weight ndim != 2: shape {W.shape}")
            # ONNX MatMul: y = x @ W where W is (in, out). Our
            # convention is (out, in), so transpose.
            pending_w = W.T
        elif op == "Add":
            const_name = [n for n in node.input if n in inits]
            if len(const_name) != 1:
                raise ValueError("Add with non-constant operand not supported")
            b = inits[const_name[0]].reshape(-1)
            if pending_w is None:
                raise ValueError("Add without a preceding MatMul")
            W = pending_w
            if not layers_w:
                W, b = fold_prefix(W, b)
            layers_w.append((W, b))
            pending_w = None
        elif op == "Gemm":
            # Gemm fuses MatMul + Add: y = α (A·B) + β·C, with
            # transA, transB attributes. The VNN-COMP convention is
            # `Gemm[alpha=1, beta=1, transB=1](X, W, bias)` ⇒
            # y = X @ W^T + bias. We require alpha=beta=1, transA=0,
            # transB=1 (matches every benchmark we've inspected).
            alpha = attr.get("alpha")
            beta = attr.get("beta")
            transA = attr.get("transA")
            transB = attr.get("transB")
            if alpha is not None and abs(alpha.f - 1.0) > 1e-9:
                raise ValueError(f"Gemm with alpha != 1 unsupported (got {alpha.f})")
            if beta is not None and abs(beta.f - 1.0) > 1e-9:
                raise ValueError(f"Gemm with beta != 1 unsupported (got {beta.f})")
            if transA is not None and transA.i != 0:
                raise ValueError("Gemm with transA != 0 unsupported")
            tb_int = transB.i if transB is not None else 0
            const_inputs = [n for n in node.input[1:] if n in inits]
            if len(const_inputs) < 2:
                raise ValueError("Gemm without constant W and bias unsupported")
            W_raw = inits[const_inputs[0]]  # input[1]
            b = inits[const_inputs[1]].reshape(-1)  # input[2]
            if W_raw.ndim != 2:
                raise ValueError(f"Gemm weight ndim != 2: shape {W_raw.shape}")
            # We want our layer's (out, in) convention. With
            # transB=1, ONNX has W shape (out, in) already; with
            # transB=0, W is (in, out), needs transpose.
            W = W_raw if tb_int == 1 else W_raw.T
            if not layers_w:
                W, b = fold_prefix(W, b)
            layers_w.append((W, b))
        elif op == "Relu":
            activations.append("relu")
        elif op == "Sigmoid":
            activations.append("sigmoid")
        elif op == "Tanh":
            activations.append("tanh")
        else:
            raise ValueError(f"unsupported ONNX op: {op}")

    if pending_w is not None:
        raise ValueError("trailing MatMul with no Add")

    # The normalization prefix is only sound if it was consumed by the
    # first linear layer's fold. A Sub/Div that never reached a linear
    # layer would silently drop those operations — reject instead.
    if pre_seen:
        raise ValueError(
            "ONNX Sub/Div normalization prefix found that could not be "
            "folded into a subsequent linear layer. Only a constant "
            "prefix before the first MatMul/Add/Gemm is supported."
        )

    # A trailing activation (ERAN-style nets) is preserved exactly by an
    # identity output layer so the schema's "activations between linear
    # layers" invariant holds.
    if allow_trailing_activation and len(activations) == len(layers_w) and layers_w:
        out_dim = layers_w[-1][0].shape[0]
        layers_w.append((np.eye(out_dim), np.zeros(out_dim)))

    # Sanity: between every adjacent pair of layers there should be
    # exactly one activation; trailing layer must have no activation
    # after it.
    if len(activations) != len(layers_w) - 1:
        raise ValueError(
            f"expected {len(layers_w) - 1} activations between {len(layers_w)} "
            f"linear layers, got {len(activations)}"
        )

    return layers_w, activations


def _normalize_vnnlib(vnnlib_text):
    """Strip ; comments, then split into top-level (assert ...) forms.
    Each `(assert ...)` returned as a single whitespace-collapsed string
    so downstream regexes don't have to handle line breaks.
    """
    cleaned = []
    for line in vnnlib_text.splitlines():
        idx = line.find(";")
        if idx >= 0:
            line = line[:idx]
        cleaned.append(line)
    text = " ".join(cleaned)
    out = []
    depth = 0
    buf = []
    for ch in text:
        if ch == "(":
            if depth == 0:
                buf = []
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
            if depth == 0:
                form = "".join(buf).strip()
                if form.startswith("(assert"):
                    out.append(re.sub(r"\s+", " ", form))
        else:
            if depth > 0:
                buf.append(ch)
    return out


def _split_top_level(body):
    """Split `(a)(b)(c)` into ['(a)', '(b)', '(c)'] by paren-depth."""
    out = []
    depth = 0
    buf = []
    for ch in body:
        if ch == "(":
            depth += 1
        if depth > 0:
            buf.append(ch)
        if ch == ")":
            depth -= 1
            if depth == 0:
                out.append("".join(buf))
                buf = []
    return out


def _parse_vnnlib_input_box(vnnlib_text, input_dim, allow_synthetic_fallback=False):
    """Extract `[lo[i], hi[i]]` from VNNLib `assert (<= X_i const)` and
    `assert (>= X_i const)` forms. Supports multi-line asserts and
    inputs bound only inside disjunctive `(assert (or (and ...) ...))`
    blocks: in that case we take the **convex hull** across disjuncts so
    the resulting box covers every unsafe region described — sound for
    "prove no unsafe input is reachable".

    Fails closed by default: any input coordinate without an explicit
    upper or lower bound raises `ValueError`. Pass
    `allow_synthetic_fallback=True` only for exploratory runs.
    """
    lo = [-float("inf")] * input_dim
    hi = [float("inf")] * input_dim
    re_le_x = re.compile(rf"\(<=\s*X_(\d+)\s+({NUM_RE})\s*\)")
    re_ge_x = re.compile(rf"\(>=\s*X_(\d+)\s+({NUM_RE})\s*\)")

    asserts = _normalize_vnnlib(vnnlib_text)
    flat = []
    or_bodies = []
    for a in asserts:
        inner = a[len("(assert ") : -1].strip()
        if inner.startswith("(or "):
            or_bodies.append(inner)
        else:
            flat.append(inner)

    for body in flat:
        for m in re_le_x.finditer(body):
            i = int(m.group(1))
            hi[i] = min(hi[i], float(m.group(2)))
        for m in re_ge_x.finditer(body):
            i = int(m.group(1))
            lo[i] = max(lo[i], float(m.group(2)))

    for body in or_bodies:
        clauses = _split_top_level(body[len("(or ") : -1])
        u_lo = [float("inf")] * input_dim
        u_hi = [-float("inf")] * input_dim
        any_x_bound = False
        for c in clauses:
            c_lo = [-float("inf")] * input_dim
            c_hi = [float("inf")] * input_dim
            seen = False
            for m in re_le_x.finditer(c):
                seen = True
                i = int(m.group(1))
                c_hi[i] = min(c_hi[i], float(m.group(2)))
            for m in re_ge_x.finditer(c):
                seen = True
                i = int(m.group(1))
                c_lo[i] = max(c_lo[i], float(m.group(2)))
            if seen:
                any_x_bound = True
                for i in range(input_dim):
                    if c_lo[i] != -float("inf"):
                        u_lo[i] = min(u_lo[i], c_lo[i])
                    if c_hi[i] != float("inf"):
                        u_hi[i] = max(u_hi[i], c_hi[i])
        if any_x_bound:
            for i in range(input_dim):
                if u_lo[i] != float("inf"):
                    lo[i] = max(lo[i], u_lo[i])
                if u_hi[i] != -float("inf"):
                    hi[i] = min(hi[i], u_hi[i])

    missing_lo = [i for i in range(input_dim) if lo[i] == -float("inf")]
    missing_hi = [i for i in range(input_dim) if hi[i] == float("inf")]
    if missing_lo or missing_hi:
        if not allow_synthetic_fallback:
            raise ValueError(
                f"VNNLib spec is missing input bounds: lower for {missing_lo}, "
                f"upper for {missing_hi}. Pass --allow-synthetic-input-box to "
                f"override with a [-1, 1] default (exploratory only — produces "
                f"a fixture for a different verification problem)."
            )
        for i in range(input_dim):
            if lo[i] == -float("inf"):
                lo[i] = -1.0
            if hi[i] == float("inf"):
                hi[i] = 1.0
    return lo, hi


def _parse_vnnlib_output_props(
    vnnlib_text,
    output_dim,
    allow_identity_fallback=False,
):
    """Convert VNNLib counterexample assertions to property rows.

    Supported forms (multi-line OK; we pre-normalize whitespace):

        (assert (<= Y_i Y_j))            row: Y_i - Y_j
        (assert (>= Y_i Y_j))            row: Y_j - Y_i
        (assert (<= Y_i const))          row: Y_i - const
        (assert (>= Y_i const))          row: const - Y_i
        (assert (or (and <eq>...) ...))  each disjunct's single Y/Y or
                                          Y/const inequality becomes one
                                          row; safety = no disjunct can
                                          hold = EVERY row > 0.

    Returns (C, d, side). Lower-side means PANDA proves
    `row · Y - d > 0` for every emitted row.

    VNN-COMP-style `.vnnlib` files typically encode counterexamples.
    The emitted rows therefore prove the negation of the output
    assertions.

    Fails closed if no Y-assertions match (caller can pass
    allow_identity_fallback=True to fall back to identity output).
    """
    re_yy_le = re.compile(r"\(<=\s*Y_(\d+)\s+Y_(\d+)\s*\)")
    re_yy_ge = re.compile(r"\(>=\s*Y_(\d+)\s+Y_(\d+)\s*\)")
    re_yc_le = re.compile(rf"\(<=\s*Y_(\d+)\s+({NUM_RE})\s*\)")
    re_yc_ge = re.compile(rf"\(>=\s*Y_(\d+)\s+({NUM_RE})\s*\)")
    re_cy_le = re.compile(rf"\(<=\s*({NUM_RE})\s+Y_(\d+)\s*\)")
    re_cy_ge = re.compile(rf"\(>=\s*({NUM_RE})\s+Y_(\d+)\s*\)")

    def parse_eq_form(s):
        rows = []
        for m in re_yy_le.finditer(s):
            i = int(m.group(1))
            j = int(m.group(2))
            r = [0.0] * output_dim
            r[i] = 1.0
            r[j] = -1.0
            rows.append((r, 0.0))
        for m in re_yy_ge.finditer(s):
            i = int(m.group(1))
            j = int(m.group(2))
            r = [0.0] * output_dim
            r[j] = 1.0
            r[i] = -1.0
            rows.append((r, 0.0))
        for m in re_yc_le.finditer(s):
            i = int(m.group(1))
            v = float(m.group(2))
            r = [0.0] * output_dim
            r[i] = 1.0
            rows.append((r, v))
        for m in re_yc_ge.finditer(s):
            i = int(m.group(1))
            v = float(m.group(2))
            r = [0.0] * output_dim
            r[i] = -1.0
            rows.append((r, -v))
        for m in re_cy_le.finditer(s):
            v = float(m.group(1))
            i = int(m.group(2))
            r = [0.0] * output_dim
            r[i] = -1.0
            rows.append((r, -v))
        for m in re_cy_ge.finditer(s):
            v = float(m.group(1))
            i = int(m.group(2))
            r = [0.0] * output_dim
            r[i] = 1.0
            rows.append((r, v))
        return rows

    asserts = _normalize_vnnlib(vnnlib_text)
    rows = []
    for a in asserts:
        inner = a[len("(assert ") : -1].strip()
        if inner.startswith("(or "):
            # Disjunctive output spec: `(or (and ineq) (and ineq) ...)`.
            # The unsafe condition is "any disjunct's conjunction holds";
            # the safety condition we want to prove is "for every
            # disjunct, at least one inequality fails".
            #
            # We currently only support disjuncts with a SINGLE output
            # inequality each — flattening them into independent rows
            # gives the correct safety condition (every row > 0 ⇒ every
            # disjunct's single inequality fails ⇒ no disjunct holds).
            #
            # If a disjunct contains MULTIPLE output inequalities, naive
            # flattening would produce a STRONGER condition than the
            # original (would prove every inequality individually fails,
            # whereas the spec only requires AT LEAST ONE per disjunct
            # to fail). That's still sound for verification (proves
            # MORE than required), but semantically misleading when the
            # converted fixture is reported against the original
            # VNNLib query. So we fail closed here — better to
            # reject than to silently emit a stronger property.
            for clause in _split_top_level(inner[len("(or ") : -1]):
                clause_rows = parse_eq_form(clause)
                # Count Y-only inequalities in this clause (X-input
                # clauses are box-only; ignore for output-row counting).
                y_only_count = len(clause_rows)
                if y_only_count > 1:
                    raise ValueError(
                        "VNNLib disjunct contains multiple output "
                        "inequalities; flattening would produce a "
                        "stronger safety condition than the spec. "
                        "Disjunctive output specs with multi-inequality "
                        "conjuncts are not yet supported by this "
                        "converter — emit a fixture for a different "
                        "(single-inequality-per-disjunct) spec, or "
                        "extend the converter to preserve disjunctive "
                        "structure."
                    )
                rows.extend(clause_rows)
        else:
            rows.extend(parse_eq_form(inner))

    if not rows:
        if not allow_identity_fallback:
            raise ValueError(
                "VNNLib spec has no parseable output assertion of the form "
                "`(assert (<= Y_i Y_j))`, `(assert (>= Y_i Y_j))`, "
                "`(assert (<= Y_i const))`, or `(assert (>= Y_i const))` "
                "(possibly nested in `or`/`and`). Pass "
                "--allow-identity-output-property to fall back to a "
                "generic identity output property (exploratory only — does "
                "NOT prove the original VNNLib safety condition)."
            )
        return np.eye(output_dim).tolist(), [0.0] * output_dim, "both"
    spec_c = [r for r, _ in rows]
    spec_d = [d for _, d in rows]
    return spec_c, spec_d, "lower"


def convert_onnx(
    onnx_path,
    vnnlib_path,
    out_path,
    *,
    precision_bits,
    name=None,
    allow_synthetic_input_box=False,
    allow_identity_output_property=False,
):
    import onnx

    model = onnx.load(str(onnx_path))
    layers_w, activations = _onnx_layers_to_mlp(model)
    in_dim = layers_w[0][0].shape[1]
    out_dim = layers_w[-1][0].shape[0]
    weights = [W.tolist() for W, _ in layers_w]
    biases = [b.tolist() for _, b in layers_w]

    if vnnlib_path and Path(vnnlib_path).exists() and str(vnnlib_path) != "-":
        vnntext = Path(vnnlib_path).read_text()
        lo, hi = _parse_vnnlib_input_box(
            vnntext,
            in_dim,
            allow_synthetic_fallback=allow_synthetic_input_box,
        )
        spec_c, spec_d, side = _parse_vnnlib_output_props(
            vnntext,
            out_dim,
            allow_identity_fallback=allow_identity_output_property,
        )
    else:
        lo = [-0.01] * in_dim
        hi = [0.01] * in_dim
        spec_c = np.eye(out_dim).tolist()
        spec_d = [0.0] * out_dim
        side = "both"

    if len(spec_d) != len(spec_c):
        spec_d = [0.0] * len(spec_c)
    fixture = {
        "name": name or Path(onnx_path).stem,
        "activations": activations,
        "input_dim": in_dim,
        "output_dim": out_dim,
        "weights": weights,
        "biases": biases,
        "x_lower": lo,
        "x_upper": hi,
        "spec_c": spec_c,
        "spec_d": spec_d,
        "side": side,
        "precision_bits": int(precision_bits),
    }
    Path(out_path).write_text(json.dumps(fixture))
    print(
        f"wrote {out_path} (in={in_dim}, out={out_dim}, "
        f"{len(weights)} linear layers, activations={activations}, "
        f"property rows={len(spec_c)}, side={side}, "
        f"precision_bits={precision_bits})"
    )


def convert_fairproof(
    fairproof_dir,
    out_path,
    name="fairproof",
    epsilon=10.0,
    spec="margin",
    side="lower",
    *,
    precision_bits,
):
    """Fairproof's three-file format:
       - weights.json: {"weights": [W0, W1, ...], "biases": [b0, b1, ...]}
         where Wi has shape (out, in)
       - inputpoint.json: a single input point of length matching W0's
         in-dim
       - layer_sizes.json (advisory only — we derive shapes from the
         weight matrices themselves since `layer_sizes` doesn't
         consistently match across released fairproof artifacts).

    Spec selection (review: keep this in sync with the
    generated `evaluation/benchmarks/FairProof/fairproof_adult_14_8_2_2.json`
    from evaluation.benchmarks.fairproof.generate):

    - `spec="margin"` (default): produce a single-row fairness margin
      property `y[0] − y[1]` with `side="lower"`. This matches the
      shape FairProof's paper certifies (binary-classification
      label margin) and is what the SNARK benchmark currently runs.
    - `spec="identity"`: produce an n_out × n_out identity property
      with `side="both"`. Use this for a generic "compute output
      bounds" comparison aligned with LIRPA-CERT's Fairproof entry.

    The bundled `inputpoint.json` is distributed pre-multiplied by
    10**3 (integer-scaled), while the weights are the unscaled float
    model; the input is divided back to native units here.

    Default `epsilon=10.0` is the L∞ perturbation around the unscaled
    input point. `precision_bits` is required (no default:
    the model's value lives in `evaluation/quant_params/`); the
    hidden-layer preact codes must fit inside the SNARK's range
    table, whose width is a runtime table parameter rather than a
    compile-time constant.
    """
    fp = Path(fairproof_dir)
    weights_path = fp / "weights.json"
    if not weights_path.exists():
        weights_path = fp / "original_weights_unrounded_unscaled.json"
    raw = json.loads(weights_path.read_text())
    inp = json.loads((fp / "inputpoint.json").read_text())

    weights_list = []
    biases_list = []
    if isinstance(raw, dict) and "weights" in raw and "biases" in raw:
        weights_list = raw["weights"]
        biases_list = raw["biases"]
    elif isinstance(raw, list):
        for i in range(0, len(raw), 2):
            weights_list.append(raw[i])
            biases_list.append(raw[i + 1])
    else:
        raise ValueError("unsupported fairproof weights.json layout")

    n_in = len(weights_list[0][0])
    n_out = len(weights_list[-1])

    if isinstance(inp, dict):
        # Different fairproof variants ship the input under
        # `input` or `input_point`.
        if "input" in inp:
            x0 = np.array(inp["input"], dtype=float).reshape(-1)
        elif "input_point" in inp:
            x0 = np.array(inp["input_point"], dtype=float).reshape(-1)
        else:
            raise ValueError(
                f"unsupported fairproof inputpoint.json keys: {list(inp.keys())}"
            )
    elif isinstance(inp, list):
        x0 = np.array(inp, dtype=float).reshape(-1)
    else:
        raise ValueError("unsupported fairproof inputpoint.json layout")
    if len(x0) != n_in:
        x0 = x0[:n_in]
        if len(x0) != n_in:
            raise ValueError(
                f"fairproof input point length {len(x0)} != input_dim {n_in}"
            )
    # FairProof distributes the input point scaled by 10**3
    # (integer-scaled); the weights are unscaled floats. Undo the scaling.
    x0 = x0 / 10.0**3
    lo = (x0 - epsilon).tolist()
    hi = (x0 + epsilon).tolist()

    if spec == "margin":
        if n_out < 2:
            raise ValueError(f"fairproof margin spec requires n_out >= 2 (got {n_out})")
        spec_c = [[0.0] * n_out]
        spec_c[0][0] = 1.0
        spec_c[0][1] = -1.0
        spec_d = [0.0]
    elif spec == "identity":
        spec_c = np.eye(n_out).tolist()
        spec_d = [0.0] * n_out
    else:
        raise ValueError(f"unsupported spec mode: {spec!r}")

    activations = ["relu"] * (len(weights_list) - 1)
    fixture = {
        "name": name,
        "activations": activations,
        "input_dim": n_in,
        "output_dim": n_out,
        "weights": weights_list,
        "biases": biases_list,
        "x_lower": lo,
        "x_upper": hi,
        "spec_c": spec_c,
        "spec_d": spec_d,
        "side": side,
        "precision_bits": precision_bits,
    }
    Path(out_path).write_text(json.dumps(fixture))
    print(
        f"wrote {out_path} (in={n_in}, out={n_out}, "
        f"{len(weights_list)} linear layers, activations={activations}, "
        f"property rows={len(spec_c)}, side={side}, "
        f"epsilon={epsilon}, precision_bits={precision_bits})"
    )


def _decode_h5_attr_names(raw):
    if raw is None:
        return []
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in raw]


def _h5_get_by_slash(group, name):
    if name in group:
        return group[name][()]
    cur = group
    for part in name.split("/"):
        cur = cur[part]
    return cur[()]


def _load_keras_dense_layers_h5(model_path):
    """Load Dense layers from a Keras HDF5 model/weights file.

    Keras Dense kernels are `(in, out)`. PANDA fixtures use `(out, in)`,
    so this transposes every kernel.
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "CROWN conversion requires optional package `h5py` because "
            "`models_crown.tar` stores Keras/HDF5 files. Install it with "
            "`python3 -m pip install h5py` in the evaluation environment."
        ) from exc

    layers = []
    with h5py.File(model_path, "r") as f:
        root = f["model_weights"] if "model_weights" in f else f
        layer_names = _decode_h5_attr_names(root.attrs.get("layer_names"))
        if not layer_names:
            layer_names = sorted(k for k, v in root.items() if hasattr(v, "attrs"))
        for layer_name in layer_names:
            if layer_name not in root:
                continue
            layer_group = root[layer_name]
            weight_names = _decode_h5_attr_names(layer_group.attrs.get("weight_names"))
            if not weight_names and layer_name in layer_group:
                layer_group = layer_group[layer_name]
                weight_names = _decode_h5_attr_names(
                    layer_group.attrs.get("weight_names")
                )
            if not weight_names:
                continue
            kernel_name = next((n for n in weight_names if "kernel" in n), None)
            bias_name = next((n for n in weight_names if "bias" in n), None)
            if kernel_name is None or bias_name is None:
                continue
            kernel = np.asarray(
                _h5_get_by_slash(layer_group, kernel_name), dtype=np.float64
            )
            bias = np.asarray(
                _h5_get_by_slash(layer_group, bias_name), dtype=np.float64
            ).reshape(-1)
            if kernel.ndim != 2 or bias.ndim != 1:
                continue
            if kernel.shape[1] != bias.shape[0]:
                raise ValueError(
                    f"Keras Dense shape mismatch in {layer_name}: "
                    f"kernel {kernel.shape}, bias {bias.shape}"
                )
            layers.append((kernel.T, bias))
    if not layers:
        raise ValueError(f"no Dense layers found in Keras/HDF5 model {model_path}")
    for i in range(len(layers) - 1):
        if layers[i][0].shape[0] != layers[i + 1][0].shape[1]:
            raise ValueError(
                "non-sequential dense layer shapes in CROWN model: "
                f"layer {i} out={layers[i][0].shape[0]}, "
                f"layer {i + 1} in={layers[i + 1][0].shape[1]}"
            )
    return layers


# CROWN dense MLPs are named `<dataset>_<n>layer_<activation>_<hidden>`. The
# only two datasets the archive ships fully-connected nets for are MNIST
# (28*28 grayscale).
_CROWN_INPUT_DIMS = {"mnist": 28 * 28}


def _parse_crown_member_metadata(member):
    name = Path(member).name
    m = re.match(
        r"^(mnist)_(\d+)layer_(relu|sigmoid|tanh|arctan)_(\d+)(?:_.*)?$",
        name,
    )
    if not m:
        raise ValueError(
            "only MNIST CROWN models are supported; "
            f"cannot infer metadata from archive member {member!r}"
        )
    dataset, n_layers, activation, hidden = m.groups()
    return {
        "name": name,
        "dataset": dataset,
        "n_layers": int(n_layers),
        "activation": activation,
        "hidden": int(hidden),
        "input_dim": _CROWN_INPUT_DIMS[dataset],
    }


def _load_center_vector(center_path, input_dim):
    raw = json.loads(Path(center_path).read_text())
    if isinstance(raw, dict):
        for key in ("input", "input_point", "x", "x0", "center", "image"):
            if key in raw:
                raw = raw[key]
                break
    center = np.asarray(raw, dtype=np.float64).reshape(-1)
    if center.shape[0] != input_dim:
        raise ValueError(
            f"center vector has length {center.shape[0]}, expected {input_dim}"
        )
    return center


def convert_crown(
    archive_path,
    archive_member,
    out_path,
    *,
    center=None,
    epsilon=0.005,
    true_class=0,
    target_class=1,
    precision_bits,
    allow_zero_input=False,
):
    meta = _parse_crown_member_metadata(archive_member)
    if meta["activation"] == "arctan":
        raise ValueError(
            f"PANDA does not currently support arctan relaxations; "
            f"cannot convert CROWN model {archive_member!r}"
        )

    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"missing CROWN archive: {archive}")
    with tempfile.TemporaryDirectory(prefix="panda-crown-") as td:
        td_path = Path(td)
        with tarfile.open(archive) as tf:
            member = tf.getmember(archive_member)
            tf.extract(member, td_path)
        layers_w = _load_keras_dense_layers_h5(td_path / archive_member)

    in_dim = layers_w[0][0].shape[1]
    out_dim = layers_w[-1][0].shape[0]
    if in_dim != meta["input_dim"]:
        raise ValueError(
            f"CROWN metadata expected input_dim={meta['input_dim']}, "
            f"but model weights have input_dim={in_dim}"
        )
    if not (0 <= true_class < out_dim) or not (0 <= target_class < out_dim):
        raise ValueError(
            f"true_class/target_class must be in [0, {out_dim}); got "
            f"{true_class}, {target_class}"
        )
    if true_class == target_class:
        raise ValueError("true_class and target_class must differ")

    if center is not None:
        x0 = _load_center_vector(center, in_dim)
        center_source = str(center)
    elif allow_zero_input:
        x0 = np.zeros(in_dim, dtype=np.float64)
        center_source = "synthetic_zero_input"
    else:
        raise ValueError(
            "CROWN `models_crown.tar` contains only models, not verification "
            "queries. Pass center=<input.json> plus true_class=<i> and "
            "target_class=<j>, or pass --allow-zero-input for a synthetic example "
            "that is not an original CROWN-paper query."
        )

    lo = np.clip(x0 - epsilon, -0.5, 0.5)
    hi = np.clip(x0 + epsilon, -0.5, 0.5)
    spec_c = [[0.0] * out_dim]
    spec_c[0][true_class] = 1.0
    spec_c[0][target_class] = -1.0
    fixture = {
        "name": meta["name"],
        "description": (
            "CROWN-original dense MLP converted from models_crown.tar. "
            "The property is a targeted margin over an L_inf input box."
        ),
        "source": str(archive),
        "source_archive_member": archive_member,
        "center_source": center_source,
        "dataset": meta["dataset"],
        "activations": [meta["activation"]] * (len(layers_w) - 1),
        "input_dim": in_dim,
        "output_dim": out_dim,
        "weights": [W.tolist() for W, _ in layers_w],
        "biases": [b.tolist() for _, b in layers_w],
        "x_lower": lo.tolist(),
        "x_upper": hi.tolist(),
        "spec_c": spec_c,
        "spec_d": [0.0],
        "side": "lower",
        "precision_bits": precision_bits,
        "property_description": (
            f"class {true_class} beats class {target_class} for "
            f"||x - x0||_inf <= {epsilon}"
        ),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture))
    print(
        f"wrote {out_path} (CROWN {meta['name']}, in={in_dim}, out={out_dim}, "
        f"{len(layers_w)} linear layers, activation={meta['activation']}, "
        f"epsilon={epsilon}, center={center_source})"
    )


def main(argv):
    """CLI entry point.

    Optional flags (review, #2 — fail-closed by default):
      --allow-synthetic-input-box        Use [-1, 1] for any input
                                         coord whose VNNLib bound is
                                         missing (exploratory only).
      --allow-identity-output-property   Use an identity-output
                                         property if the VNNLib spec
                                         has no parseable `Y_i ≤/≥ Y_j`
                                         assertion (exploratory only;
                                         does NOT prove the original
                                         VNNLib safety condition).
      --allow-zero-input                 For CROWN only: build a synthetic
                                         example around the all-zero input
                                         when no real input center is given.
    """
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)
    args = list(argv[1:])
    allow_synth_input = "--allow-synthetic-input-box" in args
    allow_id_output = "--allow-identity-output-property" in args
    allow_zero_input = "--allow-zero-input" in args
    for a in list(args):
        if a.startswith("--vnnlib-semantics="):
            raise SystemExit(
                "--vnnlib-semantics has been removed: VNNLib output "
                "assertions are treated as counterexample/unsafe-region "
                "constraints."
            )
    args = [a for a in args if not a.startswith("--allow-")]
    if not args:
        print(__doc__)
        sys.exit(1)
    cmd = args[0]
    if cmd == "onnx":
        pos = [a for a in args[1:] if "=" not in a]
        kw = {}
        for a in args[1:]:
            if "=" in a:
                k, v = a.split("=", 1)
                kw[k.lstrip("-")] = v
        if len(pos) != 3 or set(kw) != {"precision_bits"}:
            print(
                "usage: onnx <model.onnx> <vnnlib_or_-> <output.json> "
                "precision_bits=<n> "
                "[--allow-synthetic-input-box] [--allow-identity-output-property]\n"
                "precision_bits is required — there is no default; the "
                "model's value lives in evaluation/quant_params/"
            )
            sys.exit(2)
        convert_onnx(
            pos[0],
            pos[1],
            pos[2],
            precision_bits=int(kw["precision_bits"]),
            allow_synthetic_input_box=allow_synth_input,
            allow_identity_output_property=allow_id_output,
        )
    elif cmd == "fairproof":
        # Optional kwargs for property/spec selection. Defaults match
        # the generated `evaluation/benchmarks/FairProof/
        # fairproof_adult_14_8_2_2.json` (from
        # evaluation.benchmarks.fairproof.generate;
        # (review: regen path was diverging from the
        # committed fixture).
        pos = [a for a in args[1:] if "=" not in a]
        kw = {}
        for a in args[1:]:
            if "=" in a:
                k, v = a.split("=", 1)
                kw[k.lstrip("-")] = v
        if len(pos) != 2 or "precision_bits" not in kw:
            print(
                "usage: fairproof <dir> <output.json> precision_bits=<n> "
                "[spec=margin|identity] [side=lower|upper|both] "
                "[epsilon=...]\n"
                "precision_bits is required — there is no default; the "
                "model's value lives in evaluation/quant_params/"
            )
            sys.exit(2)
        cast_kw = {}
        for k, v in kw.items():
            if k in ("spec", "side"):
                cast_kw[k] = v
            elif k == "epsilon":
                cast_kw[k] = float(v)
            elif k == "precision_bits":
                cast_kw[k] = int(v)
            else:
                print(f"unknown fairproof option: {k}")
                sys.exit(2)
        convert_fairproof(pos[0], pos[1], **cast_kw)
    elif cmd == "crown":
        pos = [a for a in args[1:] if "=" not in a]
        kw = {}
        for a in args[1:]:
            if "=" in a:
                k, v = a.split("=", 1)
                kw[k.lstrip("-")] = v
        if len(pos) != 3 or "precision_bits" not in kw:
            print(
                "usage: crown <models_crown.tar> <archive_member> <output.json> "
                "center=<input.json> true_class=<i> target_class=<j> "
                "precision_bits=<n> [epsilon=...] [--allow-zero-input]\n"
                "precision_bits is required — there is no default; the "
                "model's value lives in evaluation/quant_params/"
            )
            sys.exit(2)
        cast_kw = {"allow_zero_input": allow_zero_input}
        for k, v in kw.items():
            if k == "center":
                cast_kw[k] = v
            elif k == "epsilon":
                cast_kw[k] = float(v)
            elif k in ("true_class", "target_class", "precision_bits"):
                cast_kw[k] = int(v)
            else:
                print(f"unknown crown option: {k}")
                sys.exit(2)
        convert_crown(pos[0], pos[1], pos[2], **cast_kw)
    else:
        print(f"unknown subcommand: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
