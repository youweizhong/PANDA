"""Single reader for the per-set quantization parameters.

``evaluation/quant_params/`` holds ONE parameter set per JSON file. The
file stem is the benchmark identity: ``<model>.json`` is the model's
base set, and ``<model>__<tag>.json`` is an extra set of the same model
that the evaluation treats as its own benchmark — one extra line in
every report, labelled with the ``__<tag>`` postfix.

Every file carries the same four required integer keys plus optional
keys (four integers and the ``target_policy`` string — no flags, no
nulls, no cluster configuration):

- ``precision_bits``            fixed-point fractional bits baked into the
                                generated fixtures and used by the prover.
- ``target_preact``             per-layer power-of-two rebalancing target
                                (max |center preactivation| is scaled
                                down to about this value at
                                fixture-generation time), applied only
                                where EXACT: ReLU layers cascade,
                                sigmoid/tanh layers are scale barriers,
                                the final logit layer always rescales
                                (``evaluation/utils/rebalance.py``). ``0``
                                means the generator applies no
                                rebalancing.
- ``table_bits``                signed range / ReLU lookup-table
                                half-width: the SNARK tables cover
                                ``[-2^table_bits, 2^table_bits)``. Runtime
                                public parameter of both prover and
                                verifier.
- ``out_bound_range_bits``      the OUTPUT-stage range budget: the
                                final-pass output-bound slack/property
                                LogUps (the ``c·y − d`` margin checks,
                                once per proof) use ``[0, 2^bits)``.
                                Fixed per set — the runner never
                                escalates it; a property whose bound
                                overflows the window records an honest
                                rejection ("unknown"). Run the same
                                model at a different budget as an extra
                                ``<model>__<tag>`` set. 21 covers the
                                very-robust models whose margins
                                overflow 19 bits.
- ``gadget_range_bits``         OPTIONAL: the per-neuron gadget range
                                budget — every activation-gadget range
                                check (ReLU/sigmoid/tanh split
                                arithmetic, chunk moduli, scale
                                preconditions) and the hidden-pass
                                preact-bound inequalities use
                                ``[0, 2^bits)``. When ABSENT it equals
                                ``out_bound_range_bits``, which is
                                byte-for-byte the historical
                                single-parameter behavior — so legacy
                                sets keep their recorded semantics.
                                New sets should state it explicitly
                                (19 is the recommended default; a
                                narrow gadget table is what keeps
                                prover/verifier time down while the
                                output stage stays wide at 21).
- ``sigma_x_scale_log2``        OPTIONAL: log2 of the sigmoid/tanh table
                                INPUT scale ``s_x``. This is the lever
                                that sets the sigmoid/tanh output-bound
                                drift floor (quant vs float CROWN):
                                bigger ``s_x`` tightens the drift but
                                grows the σ half-table
                                (``2^(7 + s_x)`` entries) and proof
                                size. ReLU-only nets ignore it. Absent
                                == ``min(precision_bits, 14)``. NOTE:
                                the sshape endpoint gadget range-checks
                                the σ-table index against
                                ``gadget_range_bits``, so a preact of
                                real magnitude ``|x|`` is provable only
                                when ``|x| < 2^(gadget_range_bits - s_x)``
                                — pair a larger ``s_x`` with
                                ``gadget_range_bits >= 7 + s_x`` for the
                                full ``|x| < 128`` sigmoid/tanh domain
                                (a narrower budget only shrinks provable
                                coverage; it never accepts a false one).
- ``sigma_v_scale_log2``        OPTIONAL: log2 of the sigmoid/tanh table
                                VALUE scale ``s_v`` (σ-code magnitude).
                                Barely affects tanh drift; keep modest.
                                Absent == ``min(precision_bits + 2, 18)``.
- ``input_scale_log2``          OPTIONAL: log2 of the INPUT-box
                                quantization scale ``s_in``. Absent ==
                                the prover's default
                                ``pick_scale_pow2(x_box, precision_bits)``
                                (there is NO precision-derived fill — it
                                stays unset). A finer scale (larger value)
                                shrinks the eps-ball outward-rounding
                                drift, the dominant input-box drift term
                                at low precision, letting p14 recover
                                images that otherwise drift-fail. Runtime
                                public parameter (prover + verifier agree
                                via ``derive_public_scales``). Valid
                                ``[1, table_bits - 1]``: a saturated input
                                at ``2^input_scale_log2`` must fit the
                                signed range table, so
                                ``input_scale_log2 == table_bits`` is
                                infeasible (why in19 fails at tb19).
- ``target_policy``             OPTIONAL (string): the attack-target
                                policy of the FIXTURES this set proves.
                                Absent == ``"canonical"`` — the suite's
                                canonical fixtures (MNIST: random
                                targets). ``"least"`` marks a set whose
                                fixtures use the least-likely target
                                policy and therefore live in their own
                                fixture root
                                (``benchmarks/crown_original_least``).
                                (The final evaluation never uses this
                                key: its target policy is a runtime
                                argument — ``evaluate.sh --targets`` —
                                routed to policy-specific fixture
                                roots, and every default set is
                                canonical; the submit scripts assert
                                default sets stay canonical.) Policy sets
                                still resolve normally via ``--get`` /
                                ``load_set``.

Apart from the ``gadget_range_bits`` / ``sigma_*_scale_log2`` /
``target_policy`` fallbacks above there are NO defaults anywhere in code
and NO generator for these files: a model without a JSON file is a hard
error, and every value lives only here.
This module is intentionally stdlib-only so an external orchestrator can
invoke it on a login node via ``python3 -m evaluation.quant_params``
without a virtualenv.

CLI:

    --get <stem> <key>  print one value of one set (target_policy prints
                      "canonical" when the key is absent)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PARAMS_DIR = Path(__file__).resolve().parent

REQUIRED_KEYS = (
    "precision_bits",
    "target_preact",
    "table_bits",
    "out_bound_range_bits",
)

# Optional integer keys with their fallback semantics. gadget_range_bits
# defaults to out_bound_range_bits — the historical single-parameter
# behavior — so every legacy 4-key set keeps its recorded semantics.
# sigma_x_scale_log2 / sigma_v_scale_log2 default to the σ-scale formulas
# (see `default_sigma_scales`), keyed off precision_bits.
OPTIONAL_INT_KEYS = (
    "gadget_range_bits",
    "sigma_x_scale_log2",
    "sigma_v_scale_log2",
    "input_scale_log2",
)

# target_policy (string) defaults to "canonical" — the suite's canonical
# fixtures. Non-canonical values mark sets whose fixtures live in their
# own policy-specific roots; the submit scripts assert every default set
# stays canonical so nothing is proven against wrong-policy fixtures.
TARGET_POLICIES = ("canonical", "least")

OPTIONAL_KEYS = OPTIONAL_INT_KEYS + ("target_policy",)

# The single source of truth (mirror of Rust `default_sigma_scales`) for
# the default sigmoid/tanh table scales. `s_x` is the drift lever, capped
# so the σ half-table (2^(7 + s_x) entries) stays bounded; `s_v` is the
# σ-code magnitude, kept modest. Keep in lockstep with
# `src/snark/preprocess.rs::default_sigma_scales`.
SIGMA_X_SCALE_LOG2_MAX = 14
SIGMA_V_SCALE_LOG2_MAX = 18


def default_sigma_scales(precision_bits: int) -> tuple[int, int]:
    """Default ``(sigma_x_scale_log2, sigma_v_scale_log2)`` for a
    precision. Used when a set omits the optional σ-scale keys."""
    return (
        min(precision_bits, SIGMA_X_SCALE_LOG2_MAX),
        min(precision_bits + 2, SIGMA_V_SCALE_LOG2_MAX),
    )


class QuantParamsError(ValueError):
    """Raised for a missing or malformed quantization-parameter file."""


@dataclass(frozen=True)
class QuantParams:
    """One quantization parameter set (one JSON file)."""

    model: str
    tag: str  # "" for the base set
    precision_bits: int
    target_preact: int
    table_bits: int
    out_bound_range_bits: int
    gadget_range_bits: int
    sigma_x_scale_log2: int
    sigma_v_scale_log2: int
    target_policy: str = "canonical"
    # OPTIONAL: forces the input-box scale to 2^input_scale_log2 (a runtime
    # public parameter). ``None`` (absent) keeps the prover's default
    # ``pick_scale_pow2(x_box, precision_bits)`` — unlike the sigma keys,
    # there is no precision-derived fill, so it stays ``None`` when absent.
    input_scale_log2: int | None = None

    @property
    def stem(self) -> str:
        """Benchmark identity: the JSON file stem (`model` or `model__tag`)."""
        return f"{self.model}__{self.tag}" if self.tag else self.model

    @property
    def display_name(self) -> str:
        """Human-facing benchmark label: `model [tag]` for extra sets."""
        return f"{self.model} [{self.tag}]" if self.tag else self.model


def split_stem(stem: str) -> tuple[str, str]:
    """Split a file stem into (model, tag); tag is "" for base sets."""
    model, sep, tag = stem.partition("__")
    if sep and not tag:
        raise QuantParamsError(f"malformed quant_params stem {stem!r}: empty tag")
    return model, tag


def load_set(stem: str, params_dir: Path = PARAMS_DIR) -> QuantParams:
    """Load and validate one parameter set by file stem. Hard error on a
    missing file — there are no built-in defaults."""
    path = params_dir / f"{stem}.json"
    if not path.exists():
        raise QuantParamsError(
            f"missing quantization parameters: {path} — every benchmark "
            "set needs an explicit JSON file (there are no defaults)"
        )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise QuantParamsError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise QuantParamsError(f"{path}: expected a JSON object")
    extra = set(data) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS)
    missing = set(REQUIRED_KEYS) - set(data)
    if extra or missing:
        raise QuantParamsError(
            f"{path}: sets carry exactly {list(REQUIRED_KEYS)} plus "
            f"optionally {list(OPTIONAL_KEYS)}; "
            f"missing={sorted(missing)} unexpected={sorted(extra)}"
        )
    values: dict[str, int] = {}
    for key in REQUIRED_KEYS + OPTIONAL_INT_KEYS:
        if key in OPTIONAL_INT_KEYS and key not in data:
            continue
        value = data[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise QuantParamsError(f"{path}: {key} must be an integer, got {value!r}")
        values[key] = value
    policy = data.get("target_policy", "canonical")
    if policy not in TARGET_POLICIES:
        raise QuantParamsError(
            f"{path}: target_policy must be one of {list(TARGET_POLICIES)}, "
            f"got {policy!r}"
        )
    # gadget_range_bits absent == the historical single-parameter
    # behavior: gadgets range-check at the out-bound budget.
    values.setdefault("gadget_range_bits", values["out_bound_range_bits"])
    # σ table scales absent == the precision-derived defaults.
    default_sx, default_sv = default_sigma_scales(values["precision_bits"])
    values.setdefault("sigma_x_scale_log2", default_sx)
    values.setdefault("sigma_v_scale_log2", default_sv)
    model, tag = split_stem(stem)
    params = QuantParams(
        model=model,
        tag=tag,
        precision_bits=values["precision_bits"],
        target_preact=values["target_preact"],
        table_bits=values["table_bits"],
        out_bound_range_bits=values["out_bound_range_bits"],
        gadget_range_bits=values["gadget_range_bits"],
        sigma_x_scale_log2=values["sigma_x_scale_log2"],
        sigma_v_scale_log2=values["sigma_v_scale_log2"],
        target_policy=policy,
        input_scale_log2=values.get("input_scale_log2"),
    )
    _validate(params, path)
    return params


def _validate(p: QuantParams, path: Path) -> None:
    if p.precision_bits <= 1:
        raise QuantParamsError(f"{path}: precision_bits must be > 1")
    if p.precision_bits >= p.table_bits:
        # Mirrors the Rust SnarkParams::setup invariant: honest codes
        # must keep at least one bit of headroom inside the signed table.
        raise QuantParamsError(
            f"{path}: precision_bits ({p.precision_bits}) must be < "
            f"table_bits ({p.table_bits})"
        )
    if p.target_preact < 0 or (
        p.target_preact and p.target_preact & (p.target_preact - 1)
    ):
        raise QuantParamsError(
            f"{path}: target_preact must be 0 (no rebalancing) or a "
            f"positive power of two, got {p.target_preact}"
        )
    if p.out_bound_range_bits < 2:
        raise QuantParamsError(
            f"{path}: out_bound_range_bits ({p.out_bound_range_bits}) must be >= 2"
        )
    if p.gadget_range_bits < 2:
        raise QuantParamsError(
            f"{path}: gadget_range_bits ({p.gadget_range_bits}) must be >= 2"
        )
    if not 1 <= p.sigma_x_scale_log2 <= SIGMA_X_SCALE_LOG2_MAX:
        raise QuantParamsError(
            f"{path}: sigma_x_scale_log2 ({p.sigma_x_scale_log2}) must be in "
            f"[1, {SIGMA_X_SCALE_LOG2_MAX}]"
        )
    if not 1 <= p.sigma_v_scale_log2 <= SIGMA_V_SCALE_LOG2_MAX:
        raise QuantParamsError(
            f"{path}: sigma_v_scale_log2 ({p.sigma_v_scale_log2}) must be in "
            f"[1, {SIGMA_V_SCALE_LOG2_MAX}]"
        )
    # A saturated input at scale 2^input_scale_log2 quantizes to code
    # 2^input_scale_log2, which must land strictly inside the signed range
    # table [-2^table_bits, 2^table_bits) — so it must be <= table_bits - 1.
    # (This is exactly why in19 is infeasible at table_bits 19.) Mirrors the
    # Rust `validate_input_scale` guard.
    if p.input_scale_log2 is not None and not (
        1 <= p.input_scale_log2 <= p.table_bits - 1
    ):
        raise QuantParamsError(
            f"{path}: input_scale_log2 ({p.input_scale_log2}) must be in "
            f"[1, {p.table_bits - 1}] (a saturated input at 2^table_bits "
            f"hits the exclusive signed-range edge)"
        )


def all_sets(params_dir: Path = PARAMS_DIR) -> list[QuantParams]:
    """Every parameter set in the store, sorted by stem."""
    return [
        load_set(p.stem, params_dir) for p in sorted(params_dir.glob("*.json"))
    ]


def params_for_fixture_id(
    fixture_id: str, params_dir: Path = PARAMS_DIR
) -> QuantParams:
    """Resolve the base parameter set for a fixture id by longest embedded
    model name (fixture ids embed the model name but do not always start
    with it, e.g. ``mnist_3layer_relu_20_best__property_004`` and
    ``vnncomp2022_lunarlander_lunarlander_case_safe_0_000``)."""
    models = sorted(
        {split_stem(p.stem)[0] for p in params_dir.glob("*.json")},
        key=len,
        reverse=True,
    )
    for model in models:
        if fixture_id == model or model in fixture_id:
            return load_set(model, params_dir)
    raise QuantParamsError(
        f"no quantization-parameter file matches fixture id {fixture_id!r}"
    )
