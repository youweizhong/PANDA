"""LunarLander final-panel selection helpers."""

from __future__ import annotations

from pathlib import Path

from evaluation.config import get_suite
from evaluation.benchmarks.lunarlander.sample import fixture_name

SUITE = get_suite("LunarLander")
# All 100 local VNN-COMP LunarLander specs (no vanilla-CROWN pre-filter).
COUNT = 100
SEED = 0
# Quantization values (precision, table sizes, out-bound budgets) live
# in evaluation/quant_params/<model>.json — read them via
# evaluation.quant_params.load_set.


def fixture_paths(root: Path = SUITE.fixture_root) -> list[Path]:
    return sorted(root.glob("*.json"))


__all__ = [
    "COUNT",
    "SEED",
    "fixture_name",
    "fixture_paths",
]
