#!/usr/bin/env python3
"""Regression tests for `evaluation.preprocess.preprocessing`'s VNNLib parser.

Focus: the disjunctive output-spec handling, which had a soundness
risk where multi-inequality conjuncts inside an `(or (and ...) ...)`
clause would silently flatten into independent rows — producing a
fixture that proves a STRONGER condition than the original spec.
The parser now fails closed on multi-inequality clauses; these
tests pin that behaviour.

Run: `uv run pytest tests/test_preprocessing.py`
"""

import sys
import pytest

from evaluation.preprocess import preprocessing as e2f


def test_single_inequality_per_disjunct_flattens_correctly():
    """`(or (and (>= Y_0 Y_4)) (and (>= Y_1 Y_4)))` — typical mnist_fc.
    Each disjunct has ONE inequality; flattening to two independent
    rows IS the correct unsafe-region ruling-out condition (every
    negated row > 0 ⇔ every disjunct's single inequality fails ⇔ no
    disjunct holds)."""
    spec = """
    (declare-const Y_0 Real) (declare-const Y_1 Real)
    (declare-const Y_4 Real)
    (assert (or
        (and (>= Y_0 Y_4))
        (and (>= Y_1 Y_4))
    ))
    """
    spec_c, spec_d, side = e2f._parse_vnnlib_output_props(spec, output_dim=10)
    assert side == "lower", f"expected lower side, got {side}"
    assert len(spec_c) == 2, f"expected 2 rows, got {len(spec_c)}"
    # Unsafe row (Y_0 >= Y_4) is ruled out by proving Y_4 - Y_0 > 0.
    assert spec_c[0][4] == 1.0 and spec_c[0][0] == -1.0
    # Unsafe row (Y_1 >= Y_4) is ruled out by proving Y_4 - Y_1 > 0.
    assert spec_c[1][4] == 1.0 and spec_c[1][1] == -1.0
    assert spec_d == [0.0, 0.0]


def test_multi_inequality_disjunct_fails_closed():
    """A disjunct with two output inequalities must reject. Naive
    flattening would prove BOTH individually fail across all
    disjuncts, which is STRONGER than the original spec's "for each
    disjunct, at least one inequality fails" — semantically
    misleading. Parser now fails closed."""
    spec = """
    (declare-const Y_0 Real) (declare-const Y_1 Real)
    (assert (or
        (and (>= Y_0 0.0) (<= Y_0 1.0))
    ))
    """
    try:
        e2f._parse_vnnlib_output_props(spec, output_dim=2)
    except ValueError as ex:
        assert "multiple output inequalities" in str(ex), (
            f"expected 'multiple output inequalities' in error, got: {ex}"
        )
        return
    raise AssertionError("expected ValueError on multi-inequality disjunct")


def test_flat_y_const_inequality():
    """`(assert (<= Y_0 -2.5))` — a single output bound on Y_0."""
    spec = """
    (declare-const Y_0 Real) (declare-const Y_1 Real)
    (assert (<= Y_0 -2.5))
    """
    spec_c, spec_d, side = e2f._parse_vnnlib_output_props(spec, output_dim=2)
    assert side == "lower"
    assert len(spec_c) == 1
    # Y_0 ≤ -2.5 ⇒ unsafe; safe row: Y_0 - (-2.5) > 0 ⇒ row[0]=1, d=-2.5
    assert spec_c[0][0] == 1.0
    assert spec_c[0][1] == 0.0
    assert spec_d[0] == -2.5


def test_flat_y_y_inequality():
    """`(assert (<= Y_0 Y_1))` — pairwise output inequality (cartpole-style)."""
    spec = """
    (declare-const Y_0 Real) (declare-const Y_1 Real)
    (assert (<= Y_0 Y_1))
    """
    spec_c, spec_d, side = e2f._parse_vnnlib_output_props(spec, output_dim=2)
    assert side == "lower"
    assert len(spec_c) == 1
    # Y_0 ≤ Y_1 ⇒ unsafe; rule it out with Y_0 - Y_1 > 0.
    assert spec_c[0][0] == 1.0
    assert spec_c[0][1] == -1.0
    assert spec_d[0] == 0.0


def test_no_y_assertions_fails_closed():
    """No Y inequalities ⇒ failure (not silent identity fallback)."""
    spec = """
    (declare-const X_0 Real) (declare-const X_1 Real)
    (assert (>= X_0 0.0))
    """
    try:
        e2f._parse_vnnlib_output_props(spec, output_dim=2)
    except ValueError as ex:
        assert "no parseable output assertion" in str(ex)
        return
    raise AssertionError("expected ValueError on missing Y assertions")


def test_vnnlib_semantics_flag_is_removed():
    """The converter always treats VNNLib outputs as counterexample regions."""
    try:
        e2f.main(
            [
                "preprocessing.py",
                "onnx",
                "model.onnx",
                "spec.vnnlib",
                "out.json",
                "--vnnlib-semantics=property",
            ]
        )
    except SystemExit as ex:
        assert "has been removed" in str(ex)
        return
    raise AssertionError("expected SystemExit for removed --vnnlib-semantics flag")


def test_input_box_flat():
    spec = """
    (declare-const X_0 Real) (declare-const X_1 Real)
    (assert (<= X_0 1.0))
    (assert (>= X_0 -1.0))
    (assert (<= X_1 0.5))
    (assert (>= X_1 -0.5))
    """
    lo, hi = e2f._parse_vnnlib_input_box(spec, input_dim=2)
    assert lo == [-1.0, -0.5], f"got lo={lo}"
    assert hi == [1.0, 0.5], f"got hi={hi}"


def test_input_box_disjunctive_convex_hull():
    """Disjunct-A: X_0 ∈ [0, 1]; Disjunct-B: X_0 ∈ [2, 3].
    Convex hull: X_0 ∈ [0, 3]. Sound for "prove no unsafe input" since
    the proven box is a SUPERSET of every disjunct's box."""
    spec = """
    (declare-const X_0 Real) (declare-const Y_0 Real)
    (assert (or
        (and (>= X_0 0.0) (<= X_0 1.0) (>= Y_0 0.0))
        (and (>= X_0 2.0) (<= X_0 3.0) (>= Y_0 0.0))
    ))
    """
    # Note the second disjunct also touches X bounds, which the
    # flattening tests above would reject in the OUTPUT path.
    # For the INPUT path, the box convex-hull behaviour is still the
    # right contract — we verify that part here.
    lo, hi = e2f._parse_vnnlib_input_box(spec, input_dim=1)
    assert lo == [0.0], f"got lo={lo}"
    assert hi == [3.0], f"got hi={hi}"


def test_crown_member_metadata():
    meta = e2f._parse_crown_member_metadata("models/mnist_3layer_sigmoid_20")
    assert meta["name"] == "mnist_3layer_sigmoid_20"
    assert meta["dataset"] == "mnist"
    assert meta["n_layers"] == 3
    assert meta["activation"] == "sigmoid"
    assert meta["hidden"] == 20
    assert meta["input_dim"] == 784

    try:
        e2f._parse_crown_member_metadata("models/other_7layer_tanh_1024")
    except ValueError as exc:
        assert "only MNIST CROWN models are supported" in str(exc)
    else:
        raise AssertionError("non-MNIST CROWN models should not be supported")


def test_crown_member_metadata_rejects_cifar():
    # CIFAR-10 was dropped from the evaluation; CROWN-archive cifar
    # members must be rejected, not silently converted.
    with pytest.raises(ValueError):
        e2f._parse_crown_member_metadata("models/cifar_5layer_relu_2048_best")


def test_convert_crown_rejects_cifar_arctan():
    """arctan is unsupported by PANDA; conversion fails before any archive
    I/O, so the rejection holds even without the model archive present."""
    try:
        e2f.convert_crown(
            "definitely-missing.tar",
            "models/cifar_5layer_arctan_2048",
            "out.json",
            # precision_bits is a required keyword (no defaults live in
            # code); the arctan rejection fires before it is ever used.
            precision_bits=14,
        )
    except ValueError as exc:
        assert "arctan" in str(exc)
    else:
        raise AssertionError("expected arctan CIFAR conversion to be rejected")


if __name__ == "__main__":
    tests = [
        test_single_inequality_per_disjunct_flattens_correctly,
        test_multi_inequality_disjunct_fails_closed,
        test_flat_y_const_inequality,
        test_flat_y_y_inequality,
        test_no_y_assertions_fails_closed,
        test_input_box_flat,
        test_input_box_disjunctive_convex_hull,
        test_crown_member_metadata,
        test_crown_member_metadata_rejects_cifar,
        test_convert_crown_rejects_cifar_arctan,
        test_vnnlib_semantics_flag_is_removed,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
