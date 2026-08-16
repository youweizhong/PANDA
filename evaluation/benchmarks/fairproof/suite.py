"""FairProof Adult suite materialization helpers."""

from __future__ import annotations

from evaluation.benchmarks.fairproof.generate import OUT_PATH, SOURCE_DIR, build_fixture

COUNT = 1
EPSILON = 10.0
# Quantization values (precision, table sizes, out-bound budgets) live
# in evaluation/quant_params/<model>.json — read them via
# evaluation.quant_params.load_set.

__all__ = [
    "COUNT",
    "EPSILON",
    "OUT_PATH",
    "SOURCE_DIR",
    "build_fixture",
]
