#!/usr/bin/env python3
"""Shared, self-contained helpers for the evaluation report tables.

Pure formatting / parsing / classification helpers used by
`evaluation.reporting.final_report`. Kept free of any pipeline state
(no merged chunk files, no orchestration) so the report stays
login-node-safe and depends only on the core `evaluation` package.
"""

from __future__ import annotations

import re
from pathlib import Path

from evaluation.schemas import Fixture, QuantizedResult

# MNIST fixtures run to tens of MB (the weights dominate), so the
# report reads epsilon from an 8 KB tail rather than parsing the body,
# and only fully parses a fixture for its structure below this guard.
FIXTURE_TAIL_BYTES = 8192
MAX_FIXTURE_BYTES = 16 * 1024 * 1024
_EPSILON_RE = re.compile(rb'"epsilon"\s*:\s*([0-9eE+.\-]+)')
_IMAGE_ID_RE = re.compile(r"img_?(\d+)")


# --- verification / baseline classification --------------------------------

def is_verified(result: QuantizedResult) -> bool:
    """An accepted PANDA proof of the property. Everything else is
    "unknown" (an honest prove/verify-time rejection, or a runner
    failure)."""
    return result.proof_status == "verified" and result.robust is True


def prefixed_name(name: str, prefix: str | None) -> str:
    """The float-baseline record id for a quantized record name."""
    if not prefix:
        return name
    tagged = prefix + "__"
    return name if name.startswith(tagged) else tagged + name


def crown_record_verified(float_raw: dict, side: str | None) -> bool | None:
    """Whether the vanilla-CROWN baseline certifies the property.

    Lower-side specs certify when EVERY spec-row lower bound is > 0
    (multi-row specs, e.g. the untargeted all-runner-up margins, need all
    rows); upper-side specs when every upper bound is < 0. Returns None
    when the baseline record carries no usable bound.
    """
    if side == "upper":
        bounds = float_raw.get("float_upper_bound")
        if not bounds:
            return None
        return all(v < 0.0 for v in bounds)
    bounds = float_raw.get("float_lower_bound")
    if not bounds:
        return None
    return all(v > 0.0 for v in bounds)


# --- LaTeX cell formatting -------------------------------------------------

def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def activation_label(activations: list[str]) -> str:
    if not activations:
        return "(none)"
    unique = sorted(set(activations))
    pretty = {"relu": "ReLU", "sigmoid": "Sigmoid", "tanh": "Tanh"}
    return ", ".join(pretty.get(a, a) for a in unique)


def fmt_uncert(mu: float | None, sd: float | None, decimals: int) -> str:
    """siunitx separate-uncertainty cell ``mu(sigma_int)``; '' when mu is
    None (the caller substitutes a braced ``{--}``)."""
    if mu is None:
        return ""
    mu_str = f"{mu:.{decimals}f}"
    if sd is None:
        return mu_str
    sd_int = round(sd * (10**decimals))
    return f"{mu_str}({sd_int})"


def fmt_proof_mb(mu_kb: float | None) -> str:
    if mu_kb is None:
        return "{--}"
    return f"{round(mu_kb / 1000.0)}"


def fmt_epsilon(value: float | None, decimals: int | None = None) -> str:
    """Epsilon-style cell; '/' where the benchmark has no scalar epsilon."""
    if value is None:
        return "/"
    if decimals is not None:
        return f"{value:.{decimals}f}"
    return f"{value:g}"


# --- structure / suite / epsilon from names and fixtures -------------------

def _split_set_tag(family: str) -> tuple[str, str]:
    """Split a stem into (base, parameter-set tag); tag is "" for base."""
    base, _, set_tag = family.partition("__")
    return base, set_tag


def suite_for_stem(stem: str) -> str | None:
    """Suite id for a set stem (mirrors the chunk config's families)."""
    base = _split_set_tag(stem)[0]
    if base.startswith("mnist_"):
        return "crown_original"
    if base.startswith("safenlp_"):
        return "safeNLP"
    if base == "lunarlander":
        return "LunarLander"
    if base == "fairproof":
        return "FairProof"
    return None


def structure_from_stem(suite: str | None, stem: str) -> str | None:
    r"""Structure cell parsed from the model stem, for the suites whose
    fixtures are too large to parse just for a label. ``mnist_{N}layer_
    {act}_{w}*`` has N linear layers of hidden width w."""
    base = _split_set_tag(stem)[0]
    tokens = base.split("_")
    try:
        if suite == "crown_original":
            n_linear = int(tokens[1].removesuffix("layer"))
            width = int(tokens[3])
            return f"{n_linear} \\times [{width}]"
    except (IndexError, ValueError):
        return None
    return None


def structure_label(suite: str | None, fixture: Fixture) -> str:
    r"""Architecture cell from a parsed fixture: the ``N \times [w]``
    convention (N linear layers, [w] the hidden width);
    non-uniform hidden widths are listed in order."""
    dims = [len(bias) for bias in fixture.biases]
    if suite in ("crown_original",):
        hidden = [d for d in dims[:-1] if d != fixture.output_dim]
    else:
        hidden = dims[:-1]
    if hidden and all(d == hidden[0] for d in hidden):
        return f"{len(hidden) + 1} \\times [{hidden[0]}]"
    widths = ", ".join(str(d) for d in hidden)
    return f"[{widths}]"


def epsilon_from_fixture_tail(path: Path) -> float | None:
    """The fixture's scalar epsilon, read from the file tail only."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            fh.seek(max(0, size - FIXTURE_TAIL_BYTES))
            tail = fh.read()
    except OSError:
        return None
    match = _EPSILON_RE.search(tail)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def image_id_from_name(name: str) -> int | None:
    """The integer image id embedded in a record/property name, e.g.
    ``..._img0018_least`` -> 18."""
    m = _IMAGE_ID_RE.search(name)
    return int(m.group(1)) if m else None


# --- output-bound drift ----------------------------------------------------
#
# "Drift" is the relative gap between the two OUTPUT bounds at the fixture
# box: ``(CROWN_bound - PANDA_bound) / CROWN_bound * 100%``, where
# ``CROWN_bound`` is the float-CROWN margin and ``PANDA_bound`` the
# quantized-CROWN margin the prover discharges. A drift above 100% means
# the PANDA margin fell to the wrong side of zero.

def bound_drift_pct(
    crown_bound: float | None, panda_bound: float | None
) -> float | None:
    """``(CROWN - PANDA) / CROWN * 100``. None when either bound is
    missing or the CROWN bound is 0 (relative drift undefined)."""
    if crown_bound is None or panda_bound is None or crown_bound == 0:
        return None
    return (crown_bound - panda_bound) / crown_bound * 100.0


def aligned_binding_drift(crown_bounds, panda_bounds, side: str | None):
    """Drift on the robustness-binding row, aligned by row INDEX.

    A multi-row spec (an untargeted all-runner-up margin) must compare the
    SAME margin in both bounds. Pick the binding row from the float
    bound — the min row for a lower-side spec, the max for upper-side —
    and read PANDA at that SAME index. Returns
    ``(crown_scalar, panda_scalar, drift_pct)`` with any element None
    when unavailable. For 1-row specs this is exactly the single margin.
    """
    if not crown_bounds:
        return None, None, None
    cvals = [float(v) for v in crown_bounds]
    if side == "upper":
        i = max(range(len(cvals)), key=lambda k: cvals[k])
    else:
        i = min(range(len(cvals)), key=lambda k: cvals[k])
    crown = cvals[i]
    panda = (
        float(panda_bounds[i])
        if panda_bounds is not None and i < len(panda_bounds)
        else None
    )
    return crown, panda, bound_drift_pct(crown, panda)
