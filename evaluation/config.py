"""Configuration manifest for the PANDA benchmark suites.

This module is the source of truth for what the evaluation runs and where each
stage reads or writes data. The rest of the package should ask this module for
suite, chunk, fixture, result, and report policy instead of hard-coding paths.

Each `Chunk` describes one runnable unit:

```text
fixture_root + filters
  -> evaluation.run_panda
  -> evaluation/results/quantized/<chunk>.json
  -> evaluation.run_float_crown
  -> evaluation/results/float/<chunk>.json
```

Each `SuiteConfig` groups one or more chunks for user-facing commands. For
example, `uv run panda-eval panda mnist` expands to the three MNIST chunks,
while `uv run panda-eval panda mnist_3layer` runs just one chunk.

The manifest also records reporting policy:

- `report_family_policy="fixture_parent"` groups CROWN-origin MNIST rows by the
  model directory that contains each property.
- `report_family_policy="fixed"` groups task-style suites such as SafeNLP,
  LunarLander, and FairProof into one row per task family.

The evaluation runs two INDEPENDENT tracks over the centered-L_inf-ball
suite (CROWN-origin MNIST):

- **Fixed-epsilon track**: fixtures baked at a fixed radius (MNIST
  0.01); PANDA proves+verifies each property and records
  verified/unknown plus prove/verify/proof-size stats over the verified
  subset. All quantization parameters (precision, table sizes, the
  range budgets) come from `evaluation/quant_params/` — one JSON
  file per parameter set; every property runs exactly once at the set's
  budgets: `out_bound_range_bits` for the final-pass output-margin
  checks and the optional `gadget_range_bits` for the per-neuron
  gadget checks (absent = equal to the out-bound budget, the
  historical single-parameter behavior). No escalation — an
  output-bound range overflow records an honest "unknown"; a
  different budget is a different `<model>__<tag>` parameter set.
  The optional `sigma_x_scale_log2` / `sigma_v_scale_log2` keys set
  the sigmoid/tanh table scales; absent, they default from precision.
- **crown_bin_search track**: per-model grouped inputs with NO epsilon; vanilla
  (float64) CROWN bisects the largest certified radius per property.
  No quantized pass and no proving happen on this track, so the two
  tracks share nothing and can run concurrently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = ROOT / "evaluation"
BENCHMARK_ROOT = EVALUATION_ROOT / "benchmarks"
# Per-model grouped inputs for the certified-radius sweep (weights stored once,
# raw x0 per image, no fixed epsilon). Written by
# evaluation.crown_bin_search.generate_inputs, read by evaluation.crown_bin_search.runner.
CROWN_BIN_SEARCH_BENCHMARK_ROOT = BENCHMARK_ROOT / "crown_bin_search"
RESULTS_ROOT = EVALUATION_ROOT / "results"
QUANTIZED_RESULTS_ROOT = RESULTS_ROOT / "quantized"
FLOAT_RESULTS_ROOT = RESULTS_ROOT / "float"
# Per-property certified-radius sweep (evaluation.crown_bin_search.runner): for
# each fixture we bisect the L_inf epsilon under vanilla (float64) CROWN
# instead of using the fixed one baked into x_lower/x_upper, then aggregate
# mean/std per model. Float-only: no quantized pass and no proving. Local-only,
# ignored by git like the rest of `results/`.
CROWN_BIN_SEARCH_RESULTS_ROOT = RESULTS_ROOT / "crown_bin_search"

FamilyPolicy = Literal["fixture_parent", "fixed"]


@dataclass(frozen=True)
class Chunk:
    """One runnable, reportable slice of the evaluation suite.

    A benchmark group (chunk) defines fixture discovery, one PANDA result JSON,
    one float-CROWN result JSON, and the report grouping policy for those records.
    Benchmark groups are designed to be independently rerunnable, which keeps
    expensive SNARK sweeps practical to resume after a fix.
    """

    id: str
    suite_id: str
    suite_label: str
    display_label: str
    fixture_root: Path
    quantized_result_path: Path
    float_result_path: Path
    fixture_filters: tuple[str, ...] = ()
    fixture_exclude_filters: tuple[str, ...] = ()
    selection_count: int | None = None
    name_prefix: str | None = None
    report_family_policy: FamilyPolicy = "fixed"
    report_family: str | None = None

    def runner_filters(self) -> list[str] | None:
        return list(self.fixture_filters) if self.fixture_filters else None

    def runner_exclude_filters(self) -> list[str] | None:
        if not self.fixture_exclude_filters:
            return None
        return list(self.fixture_exclude_filters)


@dataclass(frozen=True)
class SuiteConfig:
    """A top-level benchmark suite made from one or more chunks.

    Suite ids are accepted by the CLI as convenient aliases for a tuple of chunk
    ids. The suite metadata also documents the fixture-selection policy visible
    in pdoc and the README.
    """

    id: str
    label: str
    chunks: tuple[str, ...]
    fixture_root: Path
    selection_policy: str


CHUNKS: tuple[Chunk, ...] = (
    Chunk(
        id="mnist_2layer",
        suite_id="crown_original",
        suite_label="CROWN-origin MNIST",
        display_label="MNIST 2-layer",
        fixture_root=BENCHMARK_ROOT / "crown_original_random",
        fixture_filters=("2layer",),
        selection_count=100,
        name_prefix="crown_original",
        report_family_policy="fixture_parent",
        quantized_result_path=QUANTIZED_RESULTS_ROOT / "mnist_2layer.json",
        float_result_path=FLOAT_RESULTS_ROOT / "mnist_2layer.json",
    ),
    Chunk(
        id="mnist_3layer",
        suite_id="crown_original",
        suite_label="CROWN-origin MNIST",
        display_label="MNIST 3-layer",
        fixture_root=BENCHMARK_ROOT / "crown_original_random",
        fixture_filters=("3layer",),
        selection_count=100,
        name_prefix="crown_original",
        report_family_policy="fixture_parent",
        quantized_result_path=QUANTIZED_RESULTS_ROOT / "mnist_3layer.json",
        float_result_path=FLOAT_RESULTS_ROOT / "mnist_3layer.json",
    ),
    Chunk(
        id="mnist_4layer",
        suite_id="crown_original",
        suite_label="CROWN-origin MNIST",
        display_label="MNIST 4-layer",
        fixture_root=BENCHMARK_ROOT / "crown_original_random",
        fixture_filters=("4layer",),
        selection_count=100,
        name_prefix="crown_original",
        report_family_policy="fixture_parent",
        quantized_result_path=QUANTIZED_RESULTS_ROOT / "mnist_4layer.json",
        float_result_path=FLOAT_RESULTS_ROOT / "mnist_4layer.json",
    ),
    Chunk(
        id="safenlp_medical",
        suite_id="safeNLP",
        suite_label="SafeNLP",
        display_label="SafeNLP medical",
        fixture_root=BENCHMARK_ROOT / "safeNLP",
        fixture_filters=("safenlp_medical",),
        selection_count=100,
        name_prefix="safeNLP",
        report_family_policy="fixed",
        report_family="medical",
        quantized_result_path=QUANTIZED_RESULTS_ROOT / "safenlp_medical.json",
        float_result_path=FLOAT_RESULTS_ROOT / "safenlp_medical.json",
    ),
    Chunk(
        id="safenlp_ruarobot",
        suite_id="safeNLP",
        suite_label="SafeNLP",
        display_label="SafeNLP ruarobot",
        fixture_root=BENCHMARK_ROOT / "safeNLP",
        fixture_filters=("safenlp_ruarobot",),
        selection_count=100,
        name_prefix="safeNLP",
        report_family_policy="fixed",
        report_family="ruarobot",
        quantized_result_path=QUANTIZED_RESULTS_ROOT / "safenlp_ruarobot.json",
        float_result_path=FLOAT_RESULTS_ROOT / "safenlp_ruarobot.json",
    ),
    Chunk(
        id="lunarlander",
        suite_id="LunarLander",
        suite_label="LunarLander",
        display_label="LunarLander",
        fixture_root=BENCHMARK_ROOT / "LunarLander",
        selection_count=100,
        name_prefix="LunarLander",
        report_family_policy="fixed",
        report_family="lunarlander",
        quantized_result_path=QUANTIZED_RESULTS_ROOT / "lunarlander.json",
        float_result_path=FLOAT_RESULTS_ROOT / "lunarlander.json",
    ),
    Chunk(
        id="fairproof",
        suite_id="FairProof",
        suite_label="FairProof",
        display_label="FairProof Adult",
        fixture_root=BENCHMARK_ROOT / "FairProof",
        selection_count=1,
        name_prefix="FairProof",
        report_family_policy="fixed",
        report_family="fairproof_adult_14_8_2_2",
        quantized_result_path=QUANTIZED_RESULTS_ROOT / "fairproof.json",
        float_result_path=FLOAT_RESULTS_ROOT / "fairproof.json",
    ),
)

SUITES: tuple[SuiteConfig, ...] = (
    SuiteConfig(
        id="crown_original",
        label="CROWN-origin MNIST",
        chunks=("mnist_2layer", "mnist_3layer", "mnist_4layer"),
        fixture_root=BENCHMARK_ROOT / "crown_original_random",
        selection_policy=(
            "all 17 PANDA-supported MNIST ReLU/Sigmoid/Tanh models; shared "
            "random panel of 100 test images (seed 0), wrongly classified "
            "panel images dropped per model (no topping-up), RANDOM attack "
            "target drawn uniformly from the 9 classes != true_class, "
            "keyed on (target_seed, image_id)"
        ),
    ),
    SuiteConfig(
        id="safeNLP",
        label="SafeNLP",
        chunks=("safenlp_medical", "safenlp_ruarobot"),
        fixture_root=BENCHMARK_ROOT / "safeNLP",
        selection_policy=(
            "100 boxes per task sampled from ALL candidates (seed 0, no "
            "vanilla-CROWN pre-filter), unsafe-region VNNLib semantics; "
            "properties PANDA cannot certify record as unknown"
        ),
    ),
    SuiteConfig(
        id="LunarLander",
        label="LunarLander",
        chunks=("lunarlander",),
        fixture_root=BENCHMARK_ROOT / "LunarLander",
        selection_policy=(
            "all 100 local VNN-COMP LunarLander specs (no vanilla-CROWN "
            "pre-filter); properties PANDA cannot certify record as unknown"
        ),
    ),
    SuiteConfig(
        id="FairProof",
        label="FairProof",
        chunks=("fairproof",),
        fixture_root=BENCHMARK_ROOT / "FairProof",
        selection_policy=(
            "the single Adult margin fixture, generated from the "
            "downloaded FairProof example"
        ),
    ),
)


def all_chunks() -> tuple[Chunk, ...]:
    return CHUNKS


def chunk_ids() -> tuple[str, ...]:
    return tuple(chunk.id for chunk in CHUNKS)


def suite_ids() -> tuple[str, ...]:
    return tuple(suite.id for suite in SUITES)


def get_chunk(chunk_id: str) -> Chunk:
    for chunk in CHUNKS:
        if chunk.id == chunk_id:
            return chunk
    raise KeyError(f"unknown evaluation chunk: {chunk_id}")


def get_suite(suite_id: str) -> SuiteConfig:
    for suite in SUITES:
        if suite.id == suite_id:
            return suite
    raise KeyError(f"unknown evaluation suite: {suite_id}")


@dataclass(frozen=True)
class CrownBinSearchParams:
    """Search knobs for the certified-radius (epsilon search) sweep.

    The crown_bin_search track is FLOAT-ONLY: it bisects the L_inf radius under
    vanilla (float64) CROWN and never runs the quantized pass or produces
    proofs (the grouped inputs are written with ``bisect_iters = 0``, the
    binary's float-only mode). ``eps_hi`` is the upper bracket for the
    radius search and ``float_iters`` bounds the search steps, so a
    property CROWN already certifies at a very large radius still costs a
    bounded number of evaluations.
    """

    eps_hi: float
    float_iters: int


DEFAULT_CROWN_BIN_SEARCH_PARAMS = CrownBinSearchParams(eps_hi=0.5, float_iters=35)
SUITE_CROWN_BIN_SEARCH_PARAMS: dict[str, CrownBinSearchParams] = {
    "crown_original": DEFAULT_CROWN_BIN_SEARCH_PARAMS,
}
# Only centered-L_inf-ball suites (a scalar epsilon around x0) support radius
# crown_bin_search. VNNLib-derived suites (SafeNLP, LunarLander) use general
# hyperrectangles, so a single scalar epsilon is not meaningful there.
CROWN_BIN_SEARCH_SUITE_IDS: tuple[str, ...] = tuple(SUITE_CROWN_BIN_SEARCH_PARAMS)


def crown_bin_search_params_for(chunk: Chunk) -> CrownBinSearchParams:
    """Return the radius-search knobs for one chunk (per-suite, else default)."""
    return SUITE_CROWN_BIN_SEARCH_PARAMS.get(chunk.suite_id, DEFAULT_CROWN_BIN_SEARCH_PARAMS)


def crown_bin_search_result_path(chunk_id: str) -> Path:
    """Canonical per-model radius summary path for a chunk."""
    return CROWN_BIN_SEARCH_RESULTS_ROOT / f"{chunk_id}.json"


def crown_bin_search_chunks() -> tuple[Chunk, ...]:
    """Chunks eligible for the certified-radius search sweep."""
    return tuple(c for c in CHUNKS if c.suite_id in SUITE_CROWN_BIN_SEARCH_PARAMS)


def chunks_for_target(target: str) -> tuple[Chunk, ...]:
    """Resolve a CLI target to one or more config chunks."""
    if target == "all":
        return CHUNKS
    if target in chunk_ids():
        return (get_chunk(target),)
    if target in suite_ids():
        return tuple(get_chunk(chunk_id) for chunk_id in get_suite(target).chunks)
    # Friendly aliases retained as suite selectors.
    aliases = {
        "mnist": "crown_original",
        "crown": "crown_original",
        "safenlp": "safeNLP",
        "lunar": "LunarLander",
    }
    if target in aliases:
        return chunks_for_target(aliases[target])
    raise KeyError(f"unknown evaluation target: {target}")
