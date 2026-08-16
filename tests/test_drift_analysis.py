"""Unit tests for the PANDA-vs-float-CROWN drift analysis helpers in
``evaluation.reporting.table_common`` (used by the final report).

The drift is the relative OUTPUT-bound gap
``(CROWN_bound - PANDA_bound) / CROWN_bound * 100%``. Covers the pure
helpers (image-id extraction, drift percentage, index-aligned binding
row) so they never depend on the checked-in results."""

from __future__ import annotations

from evaluation.reporting import table_common as tc


# --- image_id_from_name -----------------------------------------------------


def test_image_id_from_name():
    assert (
        tc.image_id_from_name("mnist_2layer_relu_20_best__property_000_img0018_least")
        == 18
    )
    assert tc.image_id_from_name("mnist_2layer_relu_20_best__img_0003") == 3
    assert tc.image_id_from_name("no_image_here") is None


# --- bound_drift_pct --------------------------------------------------------


def test_bound_drift_pct_basic():
    # Quantization loosened a 40.0 lower bound to 29.75: ~25.6% drift.
    assert abs(tc.bound_drift_pct(40.0, 29.7489013671875) - 25.6277) < 1e-3


def test_bound_drift_pct_exceeds_100_when_panda_crosses_zero():
    # PANDA margin fell below 0 -> drift > 100% (why it no longer certifies).
    assert tc.bound_drift_pct(40.0, -5.0) == (40.0 - (-5.0)) / 40.0 * 100.0


def test_bound_drift_pct_negative_when_panda_tighter():
    # A (rare) PANDA bound above CROWN yields a negative drift, not clamped.
    assert tc.bound_drift_pct(40.0, 44.0) < 0


def test_bound_drift_pct_undefined_cases():
    assert tc.bound_drift_pct(None, 1.0) is None
    assert tc.bound_drift_pct(1.0, None) is None
    assert tc.bound_drift_pct(0.0, 1.0) is None  # relative to zero is undefined


# --- aligned_binding_drift (multi-row row alignment) ------------------------


def test_aligned_binding_drift_single_row_is_the_margin():
    c, p, d = tc.aligned_binding_drift([0.40], [0.30], "lower")
    assert c == 0.40 and p == 0.30 and abs(d - 25.0) < 1e-9


def test_aligned_binding_drift_multirow_uses_same_class_not_min_vs_min():
    # float binding (min) is row 2 = 0.10; PANDA's own min is row 4 = 0.05,
    # but row 2 in PANDA is 0.11 (favorable rounding). The metric must
    # compare row 2 in BOTH (0.10 vs 0.11 -> -10%), not min-vs-min
    # (0.10 vs 0.05 -> +50%, comparing two different classes).
    crown = [0.30, 0.25, 0.10, 0.40, 0.22, 0.31, 0.28, 0.19, 0.33]
    panda = [0.29, 0.26, 0.11, 0.39, 0.05, 0.30, 0.27, 0.18, 0.32]
    c, p, d = tc.aligned_binding_drift(crown, panda, "lower")
    assert c == 0.10 and p == 0.11
    assert abs(d - (-10.0)) < 1e-9  # NOT +50%


def test_aligned_binding_drift_upper_side_uses_max_row():
    # Upper-side certifies when every row < 0, so the binding row is the max.
    crown = [-0.30, -0.05, -0.40]
    panda = [-0.28, -0.06, -0.44]
    c, p, d = tc.aligned_binding_drift(crown, panda, "upper")
    assert c == -0.05 and p == -0.06  # row 1 in both


def test_aligned_binding_drift_missing_panda():
    c, p, d = tc.aligned_binding_drift([0.4, 0.1], None, "lower")
    assert c == 0.1 and p is None and d is None
