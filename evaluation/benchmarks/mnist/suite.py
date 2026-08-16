"""CROWN-origin MNIST suite selection helpers."""

from __future__ import annotations

from pathlib import Path

from evaluation.config import get_suite
from evaluation.benchmarks.mnist.generate_least_likely import (
    SUPPORTED_ACTIVATIONS,
    fixture_for,
    select_models,
)

SUITE = get_suite("crown_original")
# Shared random test-image panel per run; each model keeps only the panel
# images it classifies correctly, so the per-model property count is AT
# MOST this size.
PANEL_SIZE = 100
PANEL_SEED = 0
EPSILON = 0.01
# Quantization values (precision, table sizes, out-bound budgets) live
# in evaluation/quant_params/<model>.json — read them via
# evaluation.quant_params.load_set.


def selected_model_names() -> list[str]:
    """Return the PANDA-supported CROWN MNIST model names in panel order."""
    return [row["meta"]["name"] for row in select_models("all")]


def fixture_paths(root: Path = SUITE.fixture_root) -> list[Path]:
    # Exclude the generation manifest, whose name embeds the panel size
    # (e.g. crown_allx100_manifest.json), so this stays correct
    # regardless of --panel-size.
    return sorted(
        path
        for path in root.rglob("*.json")
        if not path.name.endswith("_manifest.json")
    )


__all__ = [
    "PANEL_SIZE",
    "PANEL_SEED",
    "EPSILON",
    "SUPPORTED_ACTIVATIONS",
    "fixture_for",
    "fixture_paths",
    "selected_model_names",
]
