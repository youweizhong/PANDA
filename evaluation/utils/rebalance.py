"""Activation-aware EXACT power-of-two rebalancing.

PANDA's fixed-point windows constrain the magnitude of preactivation
codes: `|code| < 2^table_bits` requires `|preact| < 2^(table_bits -
precision_bits)` real units. Rebalancing rescales each layer by
a power of two so the center preactivations sit around ±`target_preact`
real units — but ONLY where the transform is exact, so the rebalanced
network computes EXACTLY the original function up to one positive
power-of-two factor on the logits (argmax, margin signs, and hence the
robustness property are unchanged):

* a layer whose output feeds a **ReLU** may be rescaled — ReLU is
  positively homogeneous (`relu(z/c) = relu(z)/c` for `c > 0`), so the
  scale cascades through and is compensated in the next layer's weights;
* the **final (identity) layer** may always be rescaled — the logits are
  divided by one positive constant, which preserves the argmax and every
  margin's sign;
* a layer whose output feeds a **sigmoid/tanh** is a SCALE BARRIER:
  `sigma(z/c) != sigma(z)/c`, so rescaling its input would change the
  network. Such layers keep `c_i = 1` and the cascade restarts after
  them.

With cumulative per-layer scales `c_i` (`c_0 = 1` at the input), the
transform is `W_i' = W_i * (c_{i-1} / c_i)`, `b_i' = b_i / c_i`, giving
`h_i' = h_i / c_i` exactly for every layer under the constraints above.
For an all-ReLU network this reproduces the original generator's
`rebalance_layers` exactly; for an all-sigmoid/tanh network only the
final logit layer is rescaled. Power-of-two factors keep the transform
exact in binary floating point.

`target_preact == 0` means "no rebalancing" and is handled by the
CALLER skipping this module entirely (per-model values live in
`evaluation/quant_params/`; there are no defaults here).
"""

from __future__ import annotations

import numpy as np

#: Activations through which a power-of-two output scale passes exactly.
_HOMOGENEOUS = frozenset({"relu"})


def rebalance_layers_exact(
    layers: list[tuple[np.ndarray, np.ndarray]],
    activations: list[str],
    centers: list[np.ndarray],
    target_preact: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Rescale layers by powers of two so center preacts fit the window.

    ``activations[i]`` is the activation applied after layer ``i`` for
    ``i < len(layers) - 1``; the final layer is the identity logit layer.
    Peaks are measured in ORIGINAL coordinates over ``centers`` with the
    TRUE activation semantics, and each rescalable layer gets the
    power of two that brings its max |center preactivation| down to
    about ``target_preact`` (never below 1: layers that already fit are
    left alone). Non-rescalable layers (those feeding a sigmoid/tanh)
    keep scale 1.

    Callers must not pass ``target_preact <= 0``; ``0`` means "skip
    rebalancing" and is handled by not calling this function.
    """
    if target_preact <= 0:
        raise ValueError(
            "rebalance_layers_exact needs a positive target_preact; "
            "target_preact == 0 means skip rebalancing (do not call)"
        )
    n = len(layers)
    if n == 0:
        return []
    for i in range(n - 1):
        act = activations[i]
        if act not in _HOMOGENEOUS and act not in ("sigmoid", "tanh"):
            raise ValueError(f"unknown activation {act!r} at layer {i}")

    # Per-layer peak |preact| in original coordinates, propagated with
    # the true activations (the original helper hard-coded ReLU).
    hs = [np.asarray(c, dtype=np.float64) for c in centers]
    peaks: list[float] = []
    for i, (W, b) in enumerate(layers):
        W = np.asarray(W, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        pre = [W @ h + b for h in hs]
        peaks.append(max(float(np.max(np.abs(p))) for p in pre) or 1.0)
        if i < n - 1:
            act = activations[i]
            if act == "relu":
                hs = [np.maximum(p, 0.0) for p in pre]
            elif act == "sigmoid":
                hs = [1.0 / (1.0 + np.exp(-p)) for p in pre]
            elif act == "tanh":
                hs = [np.tanh(p) for p in pre]

    # Cumulative power-of-two scale per layer. A layer may only carry a
    # scale if its output is consumed through a positively homogeneous
    # function (ReLU, or the identity logits of the final layer).
    cs: list[float] = []
    for i, pk in enumerate(peaks):
        rescalable = i == n - 1 or activations[i] in _HOMOGENEOUS
        if rescalable:
            cs.append(2.0 ** max(0, int(np.ceil(np.log2(pk / target_preact)))))
        else:
            cs.append(1.0)

    out = []
    c_prev = 1.0
    for (W, b), c in zip(layers, cs):
        out.append((
            np.asarray(W, dtype=np.float64) * (c_prev / c),
            np.asarray(b, dtype=np.float64) / c,
        ))
        c_prev = c
    return out
