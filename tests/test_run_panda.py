#!/usr/bin/env python3
"""Regression tests for PANDA runner fixture selection and range-bit policy."""

import json

from evaluation.quant_params import QuantParams
from evaluation import run_panda
from evaluation.run_panda import discover_fixtures, fixture_id


def write_fixture(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "activations": ["relu"],
                "weights": [[[1.0]]],
                "biases": [[0.0]],
                "x_lower": [0.0],
                "x_upper": [1.0],
                "spec_c": [[1.0]],
                "spec_d": [0.0],
                "side": "lower",
            }
        )
    )


def test_discover_fixtures_applies_exclude_filter_after_include(tmp_path):
    root = tmp_path / "crown_original"
    keep = root / "mnist_3layer_relu_20" / "property_000_img0000_least.json"
    skip = root / "mnist_3layer_relu_1024_best" / "property_001_img0001_least.json"
    other_depth = (
        root / "mnist_4layer_relu_1024_best" / "property_001_img0001_least.json"
    )
    for path in (keep, skip, other_depth):
        write_fixture(path)

    selected = discover_fixtures(
        root,
        filters=["3layer"],
        exclude_filters=["mnist_3layer_relu_1024_best__property_001_img0001_least"],
    )

    assert [fixture_id(path, root) for path in selected] == [
        "mnist_3layer_relu_20__property_000_img0000_least"
    ]


def test_discover_fixtures_skips_generation_manifests(tmp_path):
    # Per-model generation writes crown_<model>x<count>_manifest.json beside
    # the fixtures; its name embeds the model name, so it would match the
    # per-model --filter unless excluded by filename.
    root = tmp_path / "crown_original"
    fixture = root / "mnist_3layer_relu_20" / "property_000_img0000_least.json"
    write_fixture(fixture)
    manifest = root / "crown_mnist_3layer_relu_20x100_manifest.json"
    manifest.write_text(json.dumps({"rows": []}))

    selected = discover_fixtures(root, filters=["mnist_3layer_relu_20"])

    assert [fixture_id(path, root) for path in selected] == [
        "mnist_3layer_relu_20__property_000_img0000_least"
    ]


def _params(**overrides):
    kwargs = dict(
        model="mnist_2layer_relu_20_best",
        tag="",
        precision_bits=12,
        target_preact=0,
        table_bits=16,
        out_bound_range_bits=19,
        gadget_range_bits=19,
        # Precision-12 defaults: min(12,14)=12, min(14,18)=14.
        sigma_x_scale_log2=12,
        sigma_v_scale_log2=14,
    )
    kwargs.update(overrides)
    return QuantParams(**kwargs)


def _write_meta_fixture(tmp_path):
    # run_fixture re-reads the fixture JSON at the end to copy metadata
    # into the result record; a minimal file is enough.
    path = tmp_path / "property_000.json"
    path.write_text(json.dumps({"side": "lower", "precision_bits": 12}))
    return path


def test_run_fixture_runs_once_at_the_sets_fixed_budget(tmp_path, monkeypatch):
    path = _write_meta_fixture(tmp_path)
    params = _params(
        tag="ob21_g19",
        out_bound_range_bits=21,
        gadget_range_bits=19,
        sigma_x_scale_log2=14,
    )
    calls = []

    def fake_run_once(
        path,
        name,
        artifact_root,
        table_bits,
        out_bound_range_bits,
        gadget_range_bits,
        sigma_x_scale_log2,
        sigma_v_scale_log2,
        input_scale_log2=None,
        require_components=False,
    ):
        calls.append(
            (
                table_bits,
                out_bound_range_bits,
                gadget_range_bits,
                sigma_x_scale_log2,
                sigma_v_scale_log2,
            )
        )
        return {
            "name": name,
            "status": "ok",
            "proof_status": "verified",
            "robust": True,
        }

    monkeypatch.setattr(run_panda, "_run_fixture_once", fake_run_once)

    rec = run_panda.run_fixture(
        path, "bench", tmp_path, None, params, set_tag="ob21_g19"
    )

    # Exactly one run, at the set's budgets and σ scales.
    assert calls == [(16, 21, 19, 14, 14)]
    assert rec["proof_status"] == "verified"
    assert rec["out_bound_range_bits"] == 21
    assert rec["gadget_range_bits"] == 19
    assert rec["sigma_x_scale_log2"] == 14
    assert rec["sigma_v_scale_log2"] == 14
    assert rec["table_bits"] == 16
    assert rec["set_tag"] == "ob21_g19"
    # Fixture metadata copied from the JSON file.
    assert rec["side"] == "lower"
    assert rec["precision_bits"] == 12


def test_run_fixture_records_an_overflow_rejection_without_retrying(
    tmp_path, monkeypatch
):
    path = _write_meta_fixture(tmp_path)
    params = _params()
    calls = []

    def fake_run_once(
        path,
        name,
        artifact_root,
        table_bits,
        out_bound_range_bits,
        gadget_range_bits,
        sigma_x_scale_log2,
        sigma_v_scale_log2,
        input_scale_log2=None,
        require_components=False,
    ):
        calls.append(out_bound_range_bits)
        # An output-bound range overflow used to trigger the escalation
        # ladder; it must now stay an honest "unknown" at this budget.
        return {
            "name": name,
            "status": "ok",
            "proof_status": "verify_rejected",
            "rejection_reason": "OutputBoundRangeFailed",
            "robust": None,
        }

    monkeypatch.setattr(run_panda, "_run_fixture_once", fake_run_once)

    rec = run_panda.run_fixture(path, "bench", tmp_path, None, params)

    assert calls == [19]
    assert rec["proof_status"] == "verify_rejected"
    assert rec["out_bound_range_bits"] == 19
    assert rec["set_tag"] == ""


def test_run_fixture_fails_when_a_stale_harness_ignores_the_gadget_budget(
    tmp_path, monkeypatch
):
    # A harness built before the budget split never echoes
    # gadget_range_bits; running a split-budget set through it would
    # silently range-check every gadget at the wide budget while the
    # record claims the narrow one. _run_fixture_once must fail the
    # fixture instead of mislabeling it.
    path = _write_meta_fixture(tmp_path)
    # _run_fixture_once records the fixture path relative to the repo
    # root; point the module's root at tmp_path for this test.
    monkeypatch.setattr(run_panda, "PANDA_ROOT", tmp_path)

    def fake_run_one(
        path, name, artifact_root, table_bits, ob_bits, gadget_bits, sigma_x, sigma_v,
        input_scale_log2=None,
    ):
        # Old-harness output: echoes only the pre-split parameters (plus
        # the σ scales, so THIS test isolates the gadget-echo guard).
        return (
            "table parameters: range_table_bits=16 out_bound_range_bits=21\n"
            f"sigma scales: sigma_x_scale_log2={sigma_x} "
            f"sigma_v_scale_log2={sigma_v}\n"
            "online prove:    1.0s\nonline verify:   0.1s\n"
            "proof size:      1024 bytes\n",
            1.2,
            True,
        )

    monkeypatch.setattr(run_panda, "run_one", fake_run_one)

    rec = run_panda._run_fixture_once(path, "bench", None, 16, 21, 19, 12, 14)
    assert rec["status"] == "failed"
    assert "gadget_range_bits" in rec["error"]

    # Equal budgets are indistinguishable from the historical behavior,
    # so a missing gadget echo must NOT fail the fixture.
    rec = run_panda._run_fixture_once(path, "bench", None, 16, 21, 21, 12, 14)
    assert rec["status"] == "ok"


def test_run_fixture_fails_when_a_stale_harness_ignores_the_sigma_scales(
    tmp_path, monkeypatch
):
    # A harness built before the σ table scales became runtime parameters
    # never echoes them and silently proves at its old hard-coded scales
    # (s_x=2^11, s_v=2^16) while the record would claim the set's values.
    # Unlike the gadget guard there is NO benign "values coincide" case
    # (the defaults are precision-derived), so the guard is unconditional.
    path = _write_meta_fixture(tmp_path)
    monkeypatch.setattr(run_panda, "PANDA_ROOT", tmp_path)

    def fake_run_one(
        path, name, artifact_root, table_bits, ob_bits, gadget_bits, sigma_x, sigma_v,
        input_scale_log2=None,
    ):
        # Old-harness output: gadget echo present, no sigma echo.
        return (
            "table parameters: range_table_bits=16 out_bound_range_bits=19 "
            "gadget_range_bits=19\n"
            "online prove:    1.0s\nonline verify:   0.1s\n"
            "proof size:      1024 bytes\n",
            1.2,
            True,
        )

    monkeypatch.setattr(run_panda, "run_one", fake_run_one)

    rec = run_panda._run_fixture_once(path, "bench", None, 16, 19, 19, 14, 16)
    assert rec["status"] == "failed"
    assert "sigma" in rec["error"]

    # An echo of the WRONG value (harness resolved different scales than
    # the set requested) must also fail, not just a missing echo.
    def fake_run_one_wrong_value(
        path, name, artifact_root, table_bits, ob_bits, gadget_bits, sigma_x, sigma_v,
        input_scale_log2=None,
    ):
        return (
            "table parameters: range_table_bits=16 out_bound_range_bits=19 "
            "gadget_range_bits=19\n"
            "sigma scales: sigma_x_scale_log2=11 sigma_v_scale_log2=16\n"
            "online prove:    1.0s\nonline verify:   0.1s\n"
            "proof size:      1024 bytes\n",
            1.2,
            True,
        )

    monkeypatch.setattr(run_panda, "run_one", fake_run_one_wrong_value)
    rec = run_panda._run_fixture_once(path, "bench", None, 16, 19, 19, 14, 16)
    assert rec["status"] == "failed"
    assert "sigma" in rec["error"]
