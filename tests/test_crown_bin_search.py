"""Unit tests for the float-only certified-radius search sweep glue.

The numeric core (float CROWN crown_bin_search) lives in the Rust binary; these
cover the Python model-file discovery and per-model mean/std aggregation
that turn the binary's per-property radii into the reported table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.crown_bin_search import runner as run_crown_bin_search


def test_chunk_of_derives_family_from_model_stem():
    assert run_crown_bin_search.chunk_of("mnist_2layer_relu_20_best") == "mnist_2layer"


def test_model_matches_target():
    m = run_crown_bin_search.model_matches
    assert m("mnist_2layer_relu_20_best", "all")
    assert m("mnist_2layer_relu_20_best", "mnist")
    assert m("mnist_2layer_relu_20_best", "crown")
    # chunk id selects the family
    assert m("mnist_2layer_relu_20_best", "mnist_2layer")
    assert not m("mnist_3layer_relu_20_best", "mnist_2layer")
    # a full model name selects exactly one
    assert m("mnist_2layer_relu_20_best", "mnist_2layer_relu_20_best")
    assert not m("mnist_2layer_relu_1024_best", "mnist_2layer_relu_20_best")


def test_std_is_zero_for_singletons():
    assert run_crown_bin_search._std([0.5]) == 0.0
    assert run_crown_bin_search._std([]) == 0.0
    assert run_crown_bin_search._std([0.0, 2.0]) == pytest.approx(2.0**0.5)


def _result(radii):
    return {
        "status": "ok",
        "radii": radii,
        "precision_bits": 14,
        "eps_hi": 0.5,
        "bisect_iters": 0,
        "runtime_secs": 4.2,
    }


def test_summarize_aggregates_float_radius_mean_std_and_counts():
    # Float-only records: r_star / r_prov carry the binary's -1 sentinel.
    radii = [
        {"image_id": 0, "r_star": -1.0, "r_prov": -1.0, "r_float": 0.12,
         "float_saturated": False},
        {"image_id": 1, "r_star": -1.0, "r_prov": -1.0, "r_float": 0.00,
         "float_saturated": False},
        {"image_id": 2, "r_star": -1.0, "r_prov": -1.0, "r_float": 0.22,
         "float_saturated": True},
    ]
    s = run_crown_bin_search.summarize(
        Path("/b/mnist_2layer_relu_20_best.json"), _result(radii), 5.0
    )

    assert s["model"] == "mnist_2layer_relu_20_best"
    assert s["chunk"] == "mnist_2layer"
    assert s["n_properties"] == 3
    assert s["n_ok"] == 3
    assert s["n_certified"] == 2  # r_float > 0 for two of three
    assert s["n_saturated"] == 1
    assert s["precision_bits"] == 14
    assert s["r_float_mean"] == pytest.approx((0.12 + 0.00 + 0.22) / 3)
    assert s["r_float_min"] == 0.0
    assert s["r_float_max"] == 0.22
    assert s["properties"] is radii


def test_summarize_skips_records_without_float_radius():
    radii = [
        {"image_id": 0, "r_float": None, "float_saturated": False},
        {"image_id": 1, "r_float": 0.22, "float_saturated": False},
    ]
    s = run_crown_bin_search.summarize(
        Path("/b/mnist_2layer_relu_20_best.json"), _result(radii), 1.0
    )
    assert s["n_properties"] == 2
    assert s["n_ok"] == 1
    assert s["r_float_mean"] == pytest.approx(0.22)


def test_summarize_handles_failed_run():
    s = run_crown_bin_search.summarize(
        Path("/b/mnist_2layer_relu_20_best.json"),
        {"status": "failed", "stderr_tail": "boom"},
        3.0,
    )
    assert s["status"] == "failed"
    assert s["n_ok"] == 0
    assert s["chunk"] == "mnist_2layer"
