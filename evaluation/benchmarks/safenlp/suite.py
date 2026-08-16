"""SafeNLP final-panel selection helpers."""

from __future__ import annotations

from pathlib import Path

from evaluation.config import get_chunk, get_suite
from evaluation.benchmarks.safenlp.sample import (
    scan_dataset,
    write_fixture,
)

SUITE = get_suite("safeNLP")
TASKS = ("medical", "ruarobot")
COUNT_PER_TASK = 100
SEED = 0
# Quantization values (precision, table sizes, out-bound budgets) live
# in evaluation/quant_params/<model>.json — read them via
# evaluation.quant_params.load_set.


def fixture_paths(
    task: str | None = None, root: Path = SUITE.fixture_root
) -> list[Path]:
    if task is None:
        return sorted(root.glob("*.json"))
    chunk = get_chunk(f"safenlp_{task}")
    token = chunk.fixture_filters[0]
    return sorted(root.glob(f"*{token}*.json"))


__all__ = [
    "COUNT_PER_TASK",
    "SEED",
    "TASKS",
    "fixture_paths",
    "scan_dataset",
    "write_fixture",
]
