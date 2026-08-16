"""Unit tests for the single quantization-parameter reader.

`evaluation/quant_params/` holds ONE parameter set per JSON file with
four required integer keys, an optional `gadget_range_bits` (absent ==
the out-bound budget, the historical single-parameter behavior), and NO
other defaults in code. Negative cases use
tmp_path stores so they never depend on the checked-in files; one test
asserts every checked-in file loads cleanly.
"""

from __future__ import annotations

import json

import pytest

from evaluation.quant_params import (
    QuantParams,
    QuantParamsError,
    all_sets,
    load_set,
    params_for_fixture_id,
    split_stem,
)

VALID = {
    "precision_bits": 14,
    "target_preact": 0,
    "table_bits": 16,
    "out_bound_range_bits": 19,
}


def write_set(params_dir, stem, **overrides):
    data = dict(VALID)
    data.update(overrides)
    path = params_dir / f"{stem}.json"
    path.write_text(json.dumps(data))
    return path


# --- load_set ---------------------------------------------------------------


def test_load_set_happy_path(tmp_path):
    write_set(tmp_path, "mnist_2layer_relu_20_best")

    p = load_set("mnist_2layer_relu_20_best", tmp_path)

    assert p == QuantParams(
        model="mnist_2layer_relu_20_best",
        tag="",
        precision_bits=14,
        target_preact=0,
        table_bits=16,
        out_bound_range_bits=19,
        gadget_range_bits=19,
        # Absent σ keys default from precision 14: min(14,14)=14, min(16,18)=16.
        sigma_x_scale_log2=14,
        sigma_v_scale_log2=16,
    )
    assert p.stem == "mnist_2layer_relu_20_best"
    assert p.display_name == "mnist_2layer_relu_20_best"


def test_load_set_tagged_stem_and_display_name(tmp_path):
    write_set(tmp_path, "toynet_relu_9_200__tp8_tb19", target_preact=8)

    p = load_set("toynet_relu_9_200__tp8_tb19", tmp_path)

    assert p.model == "toynet_relu_9_200"
    assert p.tag == "tp8_tb19"
    assert p.stem == "toynet_relu_9_200__tp8_tb19"
    assert p.display_name == "toynet_relu_9_200 [tp8_tb19]"


def test_load_set_missing_file_is_a_hard_error(tmp_path):
    with pytest.raises(QuantParamsError, match="no defaults"):
        load_set("no_such_model", tmp_path)


def test_load_set_rejects_invalid_json_and_non_object(tmp_path):
    (tmp_path / "broken.json").write_text("{not json")
    with pytest.raises(QuantParamsError, match="invalid JSON"):
        load_set("broken", tmp_path)

    (tmp_path / "listy.json").write_text("[1, 2]")
    with pytest.raises(QuantParamsError, match="JSON object"):
        load_set("listy", tmp_path)


def test_load_set_rejects_extra_keys(tmp_path):
    write_set(tmp_path, "m", enabled=1)
    with pytest.raises(QuantParamsError, match="unexpected=\\['enabled'\\]"):
        load_set("m", tmp_path)


def test_load_set_rejects_missing_keys(tmp_path):
    data = dict(VALID)
    del data["table_bits"]
    (tmp_path / "m.json").write_text(json.dumps(data))
    with pytest.raises(QuantParamsError, match="missing=\\['table_bits'\\]"):
        load_set("m", tmp_path)


@pytest.mark.parametrize(
    "bad_value",
    [True, False, None, "19", 19.0],
    ids=["bool-true", "bool-false", "null", "string", "float"],
)
def test_load_set_rejects_non_integer_values(tmp_path, bad_value):
    write_set(tmp_path, "m", out_bound_range_bits=bad_value)
    with pytest.raises(QuantParamsError, match="must be an integer"):
        load_set("m", tmp_path)


def test_load_set_rejects_precision_at_or_above_table_bits(tmp_path):
    write_set(tmp_path, "m", precision_bits=16, table_bits=16)
    with pytest.raises(QuantParamsError, match="must be < "):
        load_set("m", tmp_path)

    write_set(tmp_path, "m", precision_bits=18, table_bits=16)
    with pytest.raises(QuantParamsError, match="must be < "):
        load_set("m", tmp_path)


def test_load_set_rejects_tiny_precision(tmp_path):
    write_set(tmp_path, "m", precision_bits=1)
    with pytest.raises(QuantParamsError, match="precision_bits must be > 1"):
        load_set("m", tmp_path)


@pytest.mark.parametrize("bad_target", [6, 3, -4, -1])
def test_load_set_rejects_non_power_of_two_target_preact(tmp_path, bad_target):
    write_set(tmp_path, "m", target_preact=bad_target)
    with pytest.raises(QuantParamsError, match="power of two"):
        load_set("m", tmp_path)


@pytest.mark.parametrize("ok_target", [0, 1, 2, 8, 1024])
def test_load_set_accepts_zero_and_power_of_two_target_preact(tmp_path, ok_target):
    write_set(tmp_path, "m", target_preact=ok_target)
    assert load_set("m", tmp_path).target_preact == ok_target


def test_load_set_rejects_tiny_out_bound_budget(tmp_path):
    # below the floor of 2
    write_set(tmp_path, "m", out_bound_range_bits=1)
    with pytest.raises(QuantParamsError, match="out_bound_range_bits"):
        load_set("m", tmp_path)


def test_gadget_range_bits_defaults_to_the_out_bound_budget(tmp_path):
    # Absent gadget_range_bits == the historical single-parameter
    # behavior: gadgets range-check at the out-bound budget, whatever
    # that is — 4-key sets keep their recorded semantics.
    write_set(tmp_path, "m", out_bound_range_bits=21)
    p = load_set("m", tmp_path)
    assert p.gadget_range_bits == 21


def test_gadget_range_bits_explicit_value_wins(tmp_path):
    write_set(tmp_path, "m", out_bound_range_bits=21, gadget_range_bits=19)
    p = load_set("m", tmp_path)
    assert p.out_bound_range_bits == 21
    assert p.gadget_range_bits == 19


def test_load_set_rejects_tiny_gadget_budget(tmp_path):
    write_set(tmp_path, "m", gadget_range_bits=1)
    with pytest.raises(QuantParamsError, match="gadget_range_bits"):
        load_set("m", tmp_path)


def test_load_set_rejects_non_integer_gadget_budget(tmp_path):
    write_set(tmp_path, "m", gadget_range_bits="19")
    with pytest.raises(QuantParamsError, match="must be an integer"):
        load_set("m", tmp_path)


# --- sigma_x/v_scale_log2 ---------------------------------------------------


def test_sigma_scales_default_to_the_precision_formulas(tmp_path):
    # Absent σ keys == the precision-derived defaults:
    # s_x = min(precision, 14), s_v = min(precision + 2, 18).
    write_set(tmp_path, "m", precision_bits=10)
    p = load_set("m", tmp_path)
    assert (p.sigma_x_scale_log2, p.sigma_v_scale_log2) == (10, 12)

    write_set(tmp_path, "m", precision_bits=14)
    p = load_set("m", tmp_path)
    assert (p.sigma_x_scale_log2, p.sigma_v_scale_log2) == (14, 16)

    # Above the caps both saturate.
    write_set(tmp_path, "m", precision_bits=20, table_bits=22)
    p = load_set("m", tmp_path)
    assert (p.sigma_x_scale_log2, p.sigma_v_scale_log2) == (14, 18)


def test_sigma_scales_explicit_values_win(tmp_path):
    write_set(tmp_path, "m", sigma_x_scale_log2=14, sigma_v_scale_log2=13)
    p = load_set("m", tmp_path)
    assert p.sigma_x_scale_log2 == 14
    assert p.sigma_v_scale_log2 == 13


def test_load_set_rejects_out_of_range_sigma_x(tmp_path):
    write_set(tmp_path, "m", sigma_x_scale_log2=15)  # above the cost cap of 14
    with pytest.raises(QuantParamsError, match="sigma_x_scale_log2"):
        load_set("m", tmp_path)


def test_load_set_rejects_out_of_range_sigma_v(tmp_path):
    write_set(tmp_path, "m", sigma_v_scale_log2=19)  # above the cap of 18
    with pytest.raises(QuantParamsError, match="sigma_v_scale_log2"):
        load_set("m", tmp_path)


def test_load_set_rejects_non_integer_sigma_scale(tmp_path):
    write_set(tmp_path, "m", sigma_x_scale_log2="14")
    with pytest.raises(QuantParamsError, match="must be an integer"):
        load_set("m", tmp_path)


def test_load_set_rejects_the_removed_ladder_ceiling_key(tmp_path):
    # The escalation ladder is gone; a stale out_bound_range_bits_max key
    # must be a hard error, not silently ignored.
    write_set(tmp_path, "m", out_bound_range_bits_max=21)
    with pytest.raises(
        QuantParamsError, match="unexpected=\\['out_bound_range_bits_max'\\]"
    ):
        load_set("m", tmp_path)


# --- split_stem -------------------------------------------------------------


def test_split_stem_edge_cases():
    assert split_stem("mnist_2layer_relu_20_best") == ("mnist_2layer_relu_20_best", "")
    assert split_stem("toynet_relu_9_200__tp8_tb19") == ("toynet_relu_9_200", "tp8_tb19")
    # Splits at the FIRST double underscore; the rest is the tag.
    assert split_stem("a__b__c") == ("a", "b__c")
    with pytest.raises(QuantParamsError, match="empty tag"):
        split_stem("model__")


# --- target_policy ----------------------------------------------------------


def test_target_policy_defaults_to_canonical(tmp_path):
    write_set(tmp_path, "mnist_2layer_relu_20_best")
    assert load_set("mnist_2layer_relu_20_best", tmp_path).target_policy == "canonical"


def test_target_policy_least_loads(tmp_path):
    write_set(tmp_path, "mnist_2layer_relu_20_best__least", target_policy="least")
    p = load_set("mnist_2layer_relu_20_best__least", tmp_path)
    assert p.target_policy == "least"
    assert p.tag == "least"


def test_target_policy_rejects_unknown_values(tmp_path):
    write_set(tmp_path, "model", target_policy="runner_up")
    with pytest.raises(QuantParamsError, match="target_policy"):
        load_set("model", tmp_path)


# --- params_for_fixture_id --------------------------------------------------


def test_params_for_fixture_id_prefers_the_longest_model_prefix(tmp_path):
    # Both stems are prefixes of the fixture id; the longer one must win.
    write_set(tmp_path, "mnist_3layer_relu_1024", precision_bits=10)
    write_set(tmp_path, "mnist_3layer_relu_1024_best", precision_bits=12)

    p = params_for_fixture_id("mnist_3layer_relu_1024_best__property_001", tmp_path)

    assert p.model == "mnist_3layer_relu_1024_best"
    assert p.precision_bits == 12


def test_params_for_fixture_id_resolves_the_base_set_never_a_tagged_one(tmp_path):
    write_set(tmp_path, "toynet_relu_9_200", precision_bits=14)
    write_set(tmp_path, "toynet_relu_9_200__tp8_tb19", precision_bits=12)

    p = params_for_fixture_id("toynet_relu_9_200__eran_0007", tmp_path)

    assert p.tag == ""
    assert p.precision_bits == 14


def test_params_for_fixture_id_unknown_model_is_a_hard_error(tmp_path):
    write_set(tmp_path, "some_model")
    with pytest.raises(QuantParamsError, match="no quantization-parameter file"):
        params_for_fixture_id("other_model__property_000", tmp_path)


# --- the checked-in store ---------------------------------------------------


def test_every_checked_in_parameter_file_loads_cleanly():
    sets = all_sets()

    assert sets, "the checked-in quant_params store must not be empty"
    stems = [p.stem for p in sets]
    assert stems == sorted(stems)
    assert len(stems) == len(set(stems))
    # Every tagged set rides on an existing base set.
    models = {p.model for p in sets if not p.tag}
    for p in sets:
        if p.tag:
            assert p.model in models, f"{p.stem} has no base set {p.model}"


def test_params_for_fixture_id_matches_embedded_model_name():
    # LunarLander fixture ids embed but do not start with the model stem
    # (regression: this used to raise QuantParamsError and kill the sweep).
    params = params_for_fixture_id(
        "vnncomp2022_lunarlander_lunarlander_case_safe_0_000"
    )
    assert params.model == "lunarlander"


def test_checked_in_sets_are_exactly_the_final_defaults():
    # The final evaluation proves one DEFAULT parameter set per roster
    # model — no experiment twins ship, and no set carries a target
    # policy (policies are runtime fixture routing, not parameter sets).
    sets = {p.stem: p for p in all_sets()}
    assert all(not p.tag for p in sets.values()), (
        "tagged experiment sets must not ship with the final evaluation"
    )
    assert all(p.target_policy == "canonical" for p in sets.values())

    mnist = [p for p in sets.values() if p.model.startswith("mnist_")]
    assert len(mnist) == 17
    # The output-bound budget is 21 for every MNIST model: at ob19 the
    # certifiable output-margin window caps at 2^(19-e) real units
    # (64 for ReLU at e=p-1=13, 32 for sigmoid/tanh at e=s_x=14), which
    # the deeper/wider nets' least- and random-margin bounds overflow
    # (up to ~128 real units), dropping P/C below 100%. ob21 lifts the
    # cap to 256 / 128 and covers every observed margin. The per-neuron
    # gadget budget stays at 19 (no gadget-range failures were observed).
    # A few sigmoid/tanh nets have a hidden pre-activation endpoint that
    # exceeds the sigma-table domain 2^(gadget - s_x) = 2^(19-14) = 32
    # real units at s_x=14, so the S-shape gadget rejects it
    # ("sshape_endpoint abs_l out of range"). Those models drop s_x by
    # just enough to widen the domain past their endpoint (s_x=13 ->
    # domain 64), keeping the rest at the low-drift s_x=14 default.
    sigma_x_13 = {"mnist_3layer_sigmoid_20"}
    # A few ReLU nets whose largest least-target output margin fits the
    # ob19 window (2^(19-13)=64 real units) stay at ob19 to save the
    # larger final-pass range table; the rest need ob21 (margins up to
    # ~128). See the P/C analysis.
    ob_19 = {
        "mnist_2layer_relu_20_best",
        "mnist_3layer_relu_1024_adv_retrain",
        "mnist_4layer_relu_1024_adv_retrain",
    }
    for p in mnist:
        assert (p.precision_bits, p.target_preact, p.table_bits) == (14, 0, 19)
        assert p.out_bound_range_bits == (19 if p.model in ob_19 else 21)
        assert p.gadget_range_bits == 19
        assert p.sigma_x_scale_log2 == (13 if p.model in sigma_x_13 else 14)

    others = {p.model for p in sets.values()} - {p.model for p in mnist}
    assert others == {"fairproof", "lunarlander", "safenlp_medical", "safenlp_ruarobot"}
