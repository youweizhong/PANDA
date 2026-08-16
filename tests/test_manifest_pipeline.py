import importlib.util
import json

import pytest

from evaluation.benchmarks.mnist import generate_least_likely as gcp
from evaluation.config import (
    ROOT,
    SUITE_CROWN_BIN_SEARCH_PARAMS,
    all_chunks,
    chunks_for_target,
)
from evaluation.utils.onnx_vnnlib import _parse_vnnlib_output_props
from evaluation.reporting import table_common
from evaluation.result_store import load_json_records
from evaluation.run_panda import discover_fixtures
from evaluation.schemas import Fixture


def _skip_without_crown_archive() -> None:
    if not gcp.ARCHIVE.exists():
        pytest.skip("CROWN model archive is local-only (evaluation/third_party)")
    if importlib.util.find_spec("h5py") is None:
        pytest.skip("h5py is required to load CROWN Keras/HDF5 models")


def _all_result_files_exist() -> bool:
    return all(
        chunk.quantized_result_path.exists() and chunk.float_result_path.exists()
        for chunk in all_chunks()
    )


def _skip_without_result_files() -> None:
    if not _all_result_files_exist():
        pytest.skip(
            "generated evaluation/results artifacts are intentionally not checked in"
        )
    # Results may be synced from another machine without the (huge) fixture
    # panel; the result-vs-fixture consistency tests need both.
    for chunk in all_chunks():
        records = load_json_records(chunk.quantized_result_path)
        if records and not (ROOT / records[0]["fixture"]).exists():
            pytest.skip(
                "result files reference fixtures that are not present locally"
            )


def test_config_contains_expected_final_chunks_and_paths():
    chunks = {chunk.id: chunk for chunk in all_chunks()}

    assert list(chunks) == [
        "mnist_2layer",
        "mnist_3layer",
        "mnist_4layer",
        "safenlp_medical",
        "safenlp_ruarobot",
        "lunarlander",
        "fairproof",
    ]
    assert chunks["mnist_2layer"].selection_count == 100
    assert chunks["safenlp_medical"].selection_count == 100
    assert chunks["lunarlander"].selection_count == 100
    assert (
        chunks["mnist_3layer"]
        .quantized_result_path.as_posix()
        .endswith("evaluation/results/quantized/mnist_3layer.json")
    )
    assert (
        chunks["safenlp_ruarobot"]
        .float_result_path.as_posix()
        .endswith("evaluation/results/float/safenlp_ruarobot.json")
    )



def test_cifar_targets_do_not_resolve():
    # CIFAR-10 was dropped from the paper's evaluation; no CIFAR-flavored
    # selector may silently resolve to anything.
    for target in ("cifar10", "cifar", "eran", "cifar10_4_100"):
        with pytest.raises(KeyError):
            chunks_for_target(target)


def test_target_resolution_supports_chunks_suites_and_aliases():
    assert [chunk.id for chunk in chunks_for_target("mnist_3layer")] == ["mnist_3layer"]
    assert [chunk.id for chunk in chunks_for_target("safeNLP")] == [
        "safenlp_medical",
        "safenlp_ruarobot",
    ]
    assert [chunk.id for chunk in chunks_for_target("safenlp")] == [
        "safenlp_medical",
        "safenlp_ruarobot",
    ]


def test_crown_bin_search_params_cover_the_centered_ball_suite():
    # The float-only crown_bin_search track runs exactly over the
    # centered-ball suite: CROWN-origin MNIST.
    assert set(SUITE_CROWN_BIN_SEARCH_PARAMS) == {"crown_original"}
    for params in SUITE_CROWN_BIN_SEARCH_PARAMS.values():
        assert params.eps_hi > 0
        assert params.float_iters > 0


def test_fixed_epsilon_defaults():
    # Fixed-epsilon track: MNIST 0.01.
    assert gcp.DATASET_DEFAULT_EPSILON == {"mnist": 0.01}


def test_result_files_are_well_formed_and_complete():
    _skip_without_result_files()
    for chunk in all_chunks():
        quantized = load_json_records(chunk.quantized_result_path)
        floats = load_json_records(chunk.float_result_path)
        expected_count = len(
            discover_fixtures(
                chunk.fixture_root,
                chunk.runner_filters(),
                chunk.runner_exclude_filters(),
            )
        )

        assert len(quantized) == expected_count
        assert len(floats) == expected_count
        # Every record is either an accepted proof or an honest "unknown"
        # rejection; runner-level failures are not acceptable.
        assert all(record["status"] == "ok" for record in quantized)
        assert all(
            record["proof_status"]
            in ("verified", "prove_rejected", "verify_rejected")
            for record in quantized
        )
        assert all(record["status"] == "ok" for record in floats)


def test_safenlp_split_by_task_behavior():
    _skip_without_result_files()
    chunks = {chunk.id: chunk for chunk in all_chunks()}
    medical = load_json_records(chunks["safenlp_medical"].quantized_result_path)
    ruarobot = load_json_records(chunks["safenlp_ruarobot"].quantized_result_path)

    assert len(medical) == 100
    assert len(ruarobot) == 100
    assert all("safenlp_medical" in row["fixture"] for row in medical)
    assert all("safenlp_ruarobot" in row["fixture"] for row in ruarobot)


def test_converter_vnnlib_multi_inequality_disjunct_fails_closed():
    spec = """
    (declare-const Y_0 Real)
    (assert (or (and (>= Y_0 0.0) (<= Y_0 1.0))))
    """

    try:
        _parse_vnnlib_output_props(spec, output_dim=1)
    except ValueError as exc:
        assert "multiple output inequalities" in str(exc)
    else:
        raise AssertionError("expected fail-closed parser rejection")


def test_crown_record_verified_requires_all_spec_rows():
    # Multi-row specs (untargeted all-runner-up margins) certify only
    # when EVERY row's bound is on the right side.
    crv = table_common.crown_record_verified
    assert crv({"float_lower_bound": [0.4, 0.1]}, "lower") is True
    assert crv({"float_lower_bound": [0.4, -0.1]}, "lower") is False
    assert crv({"float_upper_bound": [-0.2, -0.3]}, "upper") is True
    assert crv({"float_upper_bound": [-0.2, 0.3]}, "upper") is False
    # Missing bound data is unknown, not false.
    assert crv({}, "lower") is None
    # side defaults to lower when unset.
    assert crv({"float_lower_bound": [0.5]}, None) is True


def test_fixture_schema_loads_referenced_quantized_record():
    _skip_without_result_files()
    chunk = chunks_for_target("fairproof")[0]
    record = load_json_records(chunk.quantized_result_path)[0]
    fixture_path = ROOT / record["fixture"]
    fixture = Fixture.from_record(fixture_path, json.loads(fixture_path.read_text()))

    assert fixture.n_params() == 144
    assert fixture.precision_bits == 8
    assert json.loads(fixture_path.read_text())["name"] == "fairproof_adult_14_8_2_2"


def test_mnist_model_selection_supports_exact_name():
    _skip_without_crown_archive()
    models = gcp.resolve_models("mnist_2layer_relu_20_best")
    assert len(models) == 1
    assert models[0]["meta"]["name"] == "mnist_2layer_relu_20_best"
