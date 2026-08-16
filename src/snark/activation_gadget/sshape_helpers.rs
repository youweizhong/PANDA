//! Pure-math helpers for the S-shape (sigmoid/tanh) activation
//! gadgets in [`super::sshape_endpoint`] and
//! [`super::sshape_critical_point`].
//!
//! Three groups:
//!
//! 1. **σ envelope lookup** ([`lookup_sigma_envelope`]) — read
//!    `(σ_lower, σ_upper)` from the saturation-extended half-table,
//!    using `σ(-x) = 1 − σ(x)` (sigmoid) or `tanh(-x) = −tanh(x)`
//!    for negative `x`.
//! 2. **Critical-point search** ([`bisect_critical_point`],
//!    [`find_strict_z_code`], [`fd_slope_match_slacks`],
//!    [`fd_epsilon_crit`]) — given relaxation slope `d`, locate an
//!    integer code `z` near the critical point where
//!    `σ'(z) ≈ d/s_d` and produce the FD slope-match slacks.
//! 3. **Split arithmetic** ([`split_line_witnesses_upper`],
//!    [`split_line_witnesses_lower`], [`floor_div`], [`ceil_div`])
//!    — decompose the per-endpoint slack identity into a chain of
//!    smaller divisions so each remainder fits the 19-bit LogUp
//!    range table.

// ----- §1: σ envelope lookup -----

use crate::crown::network::ActivationKind;
use crate::snark::SigmaTables;

/// Look up `(σ_lower, σ_upper)` codes at integer input `x_int`.
///
/// For non-negative `x_int` the helper indexes the half-table over
/// `[0, bound_real · s_x)` directly. For negative `x_int` it
/// recovers the envelope via the σ symmetries
/// (`σ(-x) = 1 − σ(x)` for sigmoid, `tanh(-x) = −tanh(x)` for tanh).
///
/// Returns `None` when `|x_int| ≥ bound_real · s_x`. In-domain
/// saturation regions (`[32·s_x, 128·s_x)` for sigmoid,
/// `[16·s_x, 128·s_x)` for tanh) resolve via the table itself —
/// `SigmaTables::build` embeds saturation constants in those
/// entries. Out-of-domain inputs fail closed rather than silently
/// using a saturation guess the SNARK can't bind.
pub(crate) fn lookup_sigma_envelope(
    pre: &SigmaTables,
    kind: ActivationKind,
    x_int: i128,
    s_x_log2: i32,
    bound_real: i64,
) -> Option<(i128, i128)> {
    use crate::snark_primitives::finite_field::fr_to_signed_i128;
    let bound_int = (bound_real as i128) * (1i128 << s_x_log2);
    if x_int >= bound_int || x_int <= -bound_int {
        return None;
    }
    let (lower_table, upper_table) = match kind {
        ActivationKind::Sigmoid => (&pre.sigmoid_lower_fr, &pre.sigmoid_upper_fr),
        ActivationKind::Tanh => (&pre.tanh_lower_fr, &pre.tanh_upper_fr),
        ActivationKind::ReLU => return None,
    };
    if x_int >= 0 {
        let idx = x_int as usize;
        if idx >= lower_table.len() {
            return None;
        }
        let lo = fr_to_signed_i128(lower_table[idx])?;
        let up = fr_to_signed_i128(upper_table[idx])?;
        Some((lo, up))
    } else {
        let abs_idx = (-x_int) as usize;
        if abs_idx >= lower_table.len() {
            return None;
        }
        let lo_pos = fr_to_signed_i128(lower_table[abs_idx])?;
        let up_pos = fr_to_signed_i128(upper_table[abs_idx])?;
        match kind {
            ActivationKind::Sigmoid => {
                let s_v = 1i128 << pre.s_v_log2;
                Some((s_v - up_pos, s_v - lo_pos))
            }
            ActivationKind::Tanh => Some((-up_pos, -lo_pos)),
            ActivationKind::ReLU => None,
        }
    }
}

#[cfg(test)]
mod envelope_tests {
    use super::*;
    use crate::snark::preprocess::{
        TEST_SIGMA_V_SCALE_LOG2 as TS_V, TEST_SIGMA_X_SCALE_LOG2 as TS_X,
    };

    /// At the inner natural region, the looked-up envelope for sigmoid
    /// at non-negative x_int matches the table directly.
    #[test]
    fn lookup_sigma_envelope_sigmoid_positive_natural() {
        let pre = SigmaTables::shared(TS_X, TS_V);
        let s_x_log2 = TS_X;
        let bound_real = crate::snark::preprocess::SIGMOID_TABLE_X_BOUND_REAL;
        let s_x = 1i128 << s_x_log2;
        // σ(0.5) ≈ 0.6225
        let x_int = s_x / 2;
        let (lo, up) =
            lookup_sigma_envelope(&pre, ActivationKind::Sigmoid, x_int, s_x_log2, bound_real)
                .unwrap();
        assert!(lo <= up);
        assert!(lo >= 0);
        // dequantize
        let s_v = (1u64 << TS_V) as f64;
        let xr = (x_int as f64) / (s_x as f64);
        let true_sigma = 1.0 / (1.0 + (-xr).exp());
        assert!((lo as f64) <= true_sigma * s_v + 1e-9);
        assert!((up as f64) >= true_sigma * s_v - 1e-9);
    }

    /// Sigmoid at negative x_int recovers via symmetry σ(-x) = 1 - σ(x).
    #[test]
    fn lookup_sigma_envelope_sigmoid_negative_via_symmetry() {
        let pre = SigmaTables::shared(TS_X, TS_V);
        let s_x_log2 = TS_X;
        let bound_real = crate::snark::preprocess::SIGMOID_TABLE_X_BOUND_REAL;
        let s_x = 1i128 << s_x_log2;
        let x_int = -s_x / 2; // σ(-0.5) ≈ 0.378
        let (lo, up) =
            lookup_sigma_envelope(&pre, ActivationKind::Sigmoid, x_int, s_x_log2, bound_real)
                .unwrap();
        let s_v = (1u64 << TS_V) as f64;
        let xr = (x_int as f64) / (s_x as f64);
        let true_sigma = 1.0 / (1.0 + (-xr).exp());
        assert!(
            (lo as f64) <= true_sigma * s_v + 1e-9,
            "lo={lo}, σ·s_v={}",
            true_sigma * s_v
        );
        assert!(
            (up as f64) >= true_sigma * s_v - 1e-9,
            "up={up}, σ·s_v={}",
            true_sigma * s_v
        );
    }

    /// Tanh negative via odd symmetry: tanh(-x) = -tanh(x).
    #[test]
    fn lookup_sigma_envelope_tanh_negative_via_symmetry() {
        let pre = SigmaTables::shared(TS_X, TS_V);
        let s_x_log2 = TS_X;
        let bound_real = crate::snark::preprocess::TANH_TABLE_X_BOUND_REAL;
        let s_x = 1i128 << s_x_log2;
        let x_int = -s_x; // tanh(-1) ≈ -0.7616
        let (lo, up) =
            lookup_sigma_envelope(&pre, ActivationKind::Tanh, x_int, s_x_log2, bound_real).unwrap();
        let s_v = (1u64 << TS_V) as f64;
        let xr = (x_int as f64) / (s_x as f64);
        let true_t = xr.tanh();
        assert!(
            (lo as f64) <= true_t * s_v + 1e-9,
            "lo={lo}, tanh·s_v={}",
            true_t * s_v
        );
        assert!(
            (up as f64) >= true_t * s_v - 1e-9,
            "up={up}, tanh·s_v={}",
            true_t * s_v
        );
    }

    /// Boundary values: `x_int == ±bound_int` and beyond must now
    /// return None (review: out-of-domain inputs are
    /// rejected; saturation constants for the in-domain saturation
    /// regions are embedded in the public table itself).
    #[test]
    fn lookup_sigma_envelope_at_exact_boundaries_returns_none() {
        let pre = SigmaTables::shared(TS_X, TS_V);
        let s_x_log2 = TS_X;

        let sig_bound_real = crate::snark::preprocess::SIGMOID_TABLE_X_BOUND_REAL;
        let sig_bound_int = (sig_bound_real as i128) * (1i128 << s_x_log2);
        for x in [
            sig_bound_int,
            -sig_bound_int,
            sig_bound_int + 1,
            -sig_bound_int - 1,
            sig_bound_int + 12345,
            -(sig_bound_int + 999),
        ] {
            assert_eq!(
                lookup_sigma_envelope(&pre, ActivationKind::Sigmoid, x, s_x_log2, sig_bound_real),
                None,
                "sigmoid out-of-domain x={x} must return None"
            );
        }
        let tanh_bound_real = crate::snark::preprocess::TANH_TABLE_X_BOUND_REAL;
        let tanh_bound_int = (tanh_bound_real as i128) * (1i128 << s_x_log2);
        for x in [
            tanh_bound_int,
            -tanh_bound_int,
            tanh_bound_int + 1,
            -tanh_bound_int - 1,
        ] {
            assert_eq!(
                lookup_sigma_envelope(&pre, ActivationKind::Tanh, x, s_x_log2, tanh_bound_real),
                None,
                "tanh out-of-domain x={x} must return None"
            );
        }
    }

    /// In-domain saturation regions (`[32·s_x, 128·s_x)` for sigmoid,
    /// `[16·s_x, 128·s_x)` for tanh) still return saturation envelope
    /// values from the PUBLIC TABLE (which has saturation constants
    /// embedded in those entries by `SigmaTables::build`).
    #[test]
    fn lookup_sigma_envelope_in_domain_saturation_via_table() {
        use crate::snark::preprocess::{
            sigmoid_sat_left_lower, sigmoid_sat_left_upper, sigmoid_sat_right_lower,
            sigmoid_sat_right_upper,
        };
        let pre = SigmaTables::shared(TS_X, TS_V);
        let s_x_log2 = TS_X;
        let bound_real = crate::snark::preprocess::SIGMOID_TABLE_X_BOUND_REAL;
        let bound_int = (bound_real as i128) * (1i128 << s_x_log2);
        // 64·s_x is at real x = 64 — well into sigmoid right-saturation.
        let (lo, up) = lookup_sigma_envelope(
            &pre,
            ActivationKind::Sigmoid,
            bound_int / 2,
            s_x_log2,
            bound_real,
        )
        .unwrap();
        assert_eq!(lo, sigmoid_sat_right_lower(TS_V));
        assert_eq!(up, sigmoid_sat_right_upper(TS_V));
        // -64·s_x is at real x = -64 — sigmoid left saturation via
        // symmetry: σ(-64) ≈ 0, so via σ(-x) = 1 - σ(x), the upper-table
        // entry at idx=64·s_x is the right-sat constant, and the gadget
        // recovers σ_lower(-64)/σ_upper(-64) = (s_v - σ_upper(64),
        // s_v - σ_lower(64)) = (s_v - s_v, s_v - (s_v - 1)) = (0, 1).
        let (lo, up) = lookup_sigma_envelope(
            &pre,
            ActivationKind::Sigmoid,
            -bound_int / 2,
            s_x_log2,
            bound_real,
        )
        .unwrap();
        assert_eq!(lo, sigmoid_sat_left_lower(TS_V));
        assert_eq!(up, sigmoid_sat_left_upper(TS_V));
    }
}

// ----- §2: critical-point search -----
//
// The verifier checks the FD slope-match identity with no
// tolerance: slacks must be `≥ 0`. The prover finds a passing
// integer `z_code` by float-bisecting `σ'(z) = m`, rounding, then
// scanning nearby table indices. The prover-side search is a
// witness-generation convenience; soundness rests on the strict
// integer FD check, not on which `z` was tried.

/// Default radius for the nearby-z search. Sized to absorb the
/// envelope ±1 LSB plus `K·δ²/2` localisation error (both within
/// ~2 table steps).
pub const NEARBY_Z_SEARCH_RADIUS: i128 = 16;

/// Float bisection for `σ'(z) = m` with `z ≥ 0`, using `σ'(z) =
/// σ(1 − σ)` (sigmoid) or `σ'(z) = 1 − tanh²` (tanh).
///
/// Returns `None` when `m` is outside `[0, M]` with `M = 1/4`
/// (sigmoid) or `M = 1` (tanh). The result is float — the prover
/// quantizes it and then runs [`find_strict_z_code`] to land on a
/// `z_code` passing the strict FD check.
pub fn bisect_critical_point(
    kind: crate::crown::network::ActivationKind,
    m: f64,
    z_max: f64,
) -> Option<f64> {
    use crate::crown::network::ActivationKind;
    if !m.is_finite() || m < 0.0 {
        return None;
    }
    let m_max = match kind {
        ActivationKind::Sigmoid => 0.25,
        ActivationKind::Tanh => 1.0,
        ActivationKind::ReLU => return None,
    };
    if m > m_max {
        return None;
    }
    let deriv = |z: f64| -> f64 {
        match kind {
            ActivationKind::Sigmoid => {
                let s = 1.0 / (1.0 + (-z).exp());
                s * (1.0 - s)
            }
            ActivationKind::Tanh => {
                let t = z.tanh();
                1.0 - t * t
            }
            ActivationKind::ReLU => unreachable!(),
        }
    };
    // σ' is decreasing on `[0, ∞)`; bisect for the smallest `z`
    // with `σ'(z) ≤ m`.
    let mut lo = 0.0_f64;
    let mut hi = z_max.max(1e-9);
    if deriv(hi) > m {
        return None;
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if deriv(mid) > m {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

/// Search for a quantized `z_code` near `z_initial_code` passing the
/// strict FD inequalities against the supplied σ envelope table.
/// Returns the closest passing candidate, or `None` if none within
/// `radius` passes — in which case the cert must fail closed.
///
/// Domain assumption: `1 ≤ z_code < table_len − 1` so `z_code ± δ`
/// are both valid table indices.
pub fn find_strict_z_code(
    sigma_lower: &[i128],
    sigma_upper: &[i128],
    d_int: i128,
    s_d: i128,
    s_x: i128,
    s_v: i128,
    z_initial_code: i128,
    radius: i128,
) -> Option<i128> {
    let table_len = sigma_lower.len() as i128;
    if table_len < 3 {
        return None;
    }
    let lo_bound = 1i128;
    let hi_bound = table_len - 2;
    for r in 0..=radius {
        for sign in [-1i128, 1] {
            if sign == -1 && r == 0 {
                continue;
            }
            let cand = z_initial_code + sign * r;
            if cand < lo_bound || cand > hi_bound {
                continue;
            }
            let zi = cand as usize;
            if fd_slope_match_slacks(
                sigma_lower[zi],
                sigma_upper[zi - 1],
                sigma_upper[zi + 1],
                d_int,
                s_d,
                s_x,
                s_v,
            )
            .is_some()
            {
                return Some(cand);
            }
        }
    }
    None
}

// FD slope-match identities (post-inflection, `z ≥ 0`).
//
// All quantities are integer codes:
//   - `z`, `z ± δ` index the σ envelope half-table at scale `s_x`;
//     `δ = 1` table-step.
//   - σ-envelope codes live at scale `s_v`.
//   - `d` is the relaxation slope at scale `s_d`.
//
// The two checks, lifted to common scale `s_d · s_x · s_v`:
//
//   FD1:  (σ_lo(z) − σ_up(z − δ)) · s_d · s_x  ≥  d · s_v
//   FD2:  d · s_v                              ≥  (σ_up(z + δ) − σ_lo(z)) · s_d · s_x
//
// The committed slacks are `LHS − RHS` and `RHS − LHS` (in that
// order), each shifted by `+ ε_crit` to absorb integer envelope and
// localisation error.

/// FD slope-match slack pair. Both `≥ 0` witnesses an approximate
/// `σ'(x) = m` at some `x ∈ (z − δ, z + δ)`.
#[derive(Clone, Debug)]
pub struct FdSlopeMatchSlacks {
    pub slack_fd1: i128,
    pub slack_fd2: i128,
}

/// Public `ε_crit` budget added to both FD slack identities to
/// absorb integer envelope error (one LSB per σ lookup, scaled to
/// `s_d · s_x · s_v`) and the `K·δ²/2` FD localisation error. Both
/// prover and verifier derive this identically from the public
/// scales; it is a public coefficient, not a witness.
pub fn fd_epsilon_crit(s_d: i128, s_x: i128) -> Option<i128> {
    s_d.checked_mul(s_x)?.checked_mul(4)
}

/// Compute the FD slope-match slack pair at scale `s_d · s_x · s_v`
/// for witness table index `z ≥ 0` and committed slope `d`. Inputs
/// are σ-envelope codes at `z`, `z − δ`, `z + δ`; the public
/// `ε_crit` margin is added to both slacks.
///
/// Returns `None` if either slack would still be negative — `z` is
/// not even an `ε_crit`-loose critical point for `m = d / s_d` — or
/// on arithmetic overflow.
pub fn fd_slope_match_slacks(
    sigma_lo_z: i128,
    sigma_up_z_minus_delta: i128,
    sigma_up_z_plus_delta: i128,
    d_int: i128,
    s_d: i128,
    s_x: i128,
    s_v: i128,
) -> Option<FdSlopeMatchSlacks> {
    let s_d_s_x = s_d.checked_mul(s_x)?;
    let epsilon_crit = fd_epsilon_crit(s_d, s_x)?;
    let lhs_fd1 = sigma_lo_z
        .checked_sub(sigma_up_z_minus_delta)?
        .checked_mul(s_d_s_x)?;
    let lhs_fd2 = sigma_up_z_plus_delta
        .checked_sub(sigma_lo_z)?
        .checked_mul(s_d_s_x)?;
    let rhs_fd = d_int.checked_mul(s_v)?;
    let slack_fd1 = lhs_fd1.checked_sub(rhs_fd)?.checked_add(epsilon_crit)?;
    let slack_fd2 = rhs_fd.checked_sub(lhs_fd2)?.checked_add(epsilon_crit)?;
    if slack_fd1 < 0 || slack_fd2 < 0 {
        return None;
    }
    Some(FdSlopeMatchSlacks {
        slack_fd1,
        slack_fd2,
    })
}

#[cfg(test)]
mod critical_point_tests {
    use super::*;

    /// FD slope-match: a `z` near the true critical point produces
    /// non-negative slacks; a `z` far from it fails.
    #[test]
    fn fd_slope_match_at_inflection() {
        // Sigmoid at z = 0 has σ' = 1/4 (max). For z just above 0 with
        // m = 1/4 - tiny, both FD slacks should be ≥ 0.
        // Use synthetic σ values: σ(z) ≈ 0.5 + ε (just past inflection).
        // At z + δ: σ ≈ 0.5 + 2ε (rising). At z - δ: σ ≈ 0.5.
        // Difference per δ ≈ ε, so we expect m·δ ≈ ε.
        let s_v = 1i128 << 16;
        let s_x = 1i128 << 11;
        let s_d = 1i128 << 11;
        // Slope m = 1/4 → d_int = s_d/4 = 512.
        let d_int = s_d / 4;
        // Real diff per δ at z = 0: σ'(0) · δ = 0.25 · 2⁻¹¹ = 2⁻¹³.
        // At scale s_v: 2⁻¹³ · 2¹⁶ = 8 codes.
        // So σ_lo(z) - σ_up(z-δ) ≈ 8 codes (with ±2 envelope wobble).
        // Similarly σ_up(z+δ) - σ_lo(z) ≈ 8 codes.
        let sigma_lo_z = s_v / 2;
        let sigma_up_z_minus_delta = s_v / 2 - 8;
        let sigma_up_z_plus_delta = s_v / 2 + 8;
        let r = fd_slope_match_slacks(
            sigma_lo_z,
            sigma_up_z_minus_delta,
            sigma_up_z_plus_delta,
            d_int,
            s_d,
            s_x,
            s_v,
        )
        .expect("honest witness yields non-negative slacks");
        // Check magnitudes: slack_fd1 = LHS - RHS = 8·s_d·s_x - d·s_v
        //   = 8·2¹¹·2¹¹ - 512·2¹⁶ = 8·2²² - 2²⁵ = 2²⁵ - 2²⁵ = 0.
        // With the ε_crit margin (4·s_d·s_x), each slack equals the
        // canonical (LHS − RHS = 0) plus ε_crit.
        let epsilon = fd_epsilon_crit(s_d, s_x).unwrap();
        assert_eq!(r.slack_fd1, epsilon);
        assert_eq!(r.slack_fd2, epsilon);
    }

    /// FD slope-match: a wildly mismatched `z` (slope-z disagree)
    /// produces a negative slack ⇒ returns `None`.
    #[test]
    fn fd_slope_match_rejects_inconsistent_witness() {
        let s_v = 1i128 << 16;
        let s_x = 1i128 << 11;
        let s_d = 1i128 << 11;
        // Claim m = 0.5 (d_int = 1024) but provide σ values consistent
        // with σ' = 0.25 at z. FD1 should fail (or FD2).
        let d_int = 1024;
        let sigma_lo_z = s_v / 2;
        let sigma_up_z_minus_delta = s_v / 2 - 8;
        let sigma_up_z_plus_delta = s_v / 2 + 8;
        let r = fd_slope_match_slacks(
            sigma_lo_z,
            sigma_up_z_minus_delta,
            sigma_up_z_plus_delta,
            d_int,
            s_d,
            s_x,
            s_v,
        );
        assert!(r.is_none(), "claimed m=0.5 with σ'(z)≈0.25 must reject");
    }

    /// Bisection finds the sigmoid critical point at `m = 1/4 − ε`.
    /// At m = 1/4, σ'(z) = 1/4 ⇔ z = 0 (the max). For m slightly
    /// less, z > 0 small.
    #[test]
    fn bisect_critical_point_sigmoid_near_max() {
        let z = bisect_critical_point(crate::crown::network::ActivationKind::Sigmoid, 0.24, 32.0)
            .expect("m=0.24 < 0.25 should yield a finite z");
        assert!(z > 0.0 && z < 1.0, "z = {z}");
    }

    /// Bisection finds sigmoid z for tiny m: σ'(z) = 0.001 ⇒ z large.
    #[test]
    fn bisect_critical_point_sigmoid_tiny() {
        let z = bisect_critical_point(crate::crown::network::ActivationKind::Sigmoid, 0.001, 32.0)
            .expect("m=0.001 should yield z near 5.5");
        assert!(z > 4.0 && z < 8.0, "z = {z}");
    }

    /// Bisection rejects m > max derivative.
    #[test]
    fn bisect_critical_point_rejects_out_of_range() {
        // Sigmoid max derivative is 0.25.
        assert!(
            bisect_critical_point(crate::crown::network::ActivationKind::Sigmoid, 0.5, 32.0)
                .is_none()
        );
        // Tanh max derivative is 1.0.
        assert!(
            bisect_critical_point(crate::crown::network::ActivationKind::Tanh, 1.5, 32.0).is_none()
        );
    }

    /// Bisection for tanh: σ'(z) = 1 - tanh²(z). At z = 1, σ' ≈ 0.42.
    #[test]
    fn bisect_critical_point_tanh_medium() {
        let z = bisect_critical_point(crate::crown::network::ActivationKind::Tanh, 0.42, 32.0)
            .expect("m=0.42 should yield z near 1");
        assert!(z > 0.5 && z < 1.5, "z = {z}");
    }

    /// `find_strict_z_code` returns the seed when it already passes.
    #[test]
    fn find_strict_z_code_returns_seed_when_valid() {
        let s_v = 1i128 << 16;
        let s_x = 1i128 << 11;
        let s_d = 1i128 << 11;
        let table_len = 16usize;
        // Constant σ ⇒ σ' = 0. FD slacks at any z are 0 when d = 0.
        let sigma_lower = vec![32768i128; table_len];
        let sigma_upper = vec![32768i128; table_len]; // zero-width envelope
        let d_int = 0;
        let z = find_strict_z_code(
            &sigma_lower,
            &sigma_upper,
            d_int,
            s_d,
            s_x,
            s_v,
            /*seed*/ 4,
            /*radius*/ 8,
        );
        assert!(z.is_some(), "FD slacks at d=0, flat σ should be ≥ 0");
    }

    /// `find_strict_z_code` returns None when no nearby candidate
    /// passes (e.g., `m` outside σ's derivative range for any z).
    #[test]
    fn find_strict_z_code_returns_none_when_no_match() {
        // Synthetic: very flat table ⇒ σ'(z) ≈ 0 everywhere; large
        // d_int gives no z with non-negative FD1 slack.
        let s_v = 1i128 << 16;
        let s_x = 1i128 << 11;
        let s_d = 1i128 << 11;
        let table_len = 8usize;
        let sigma_lower = vec![32768i128; table_len];
        let sigma_upper = vec![32768i128 + 1; table_len];
        let d_int = 1024; // demands m ≈ 0.5 — impossible for flat σ
        let z = find_strict_z_code(&sigma_lower, &sigma_upper, d_int, s_d, s_x, s_v, 4, 8);
        assert!(z.is_none());
    }
}

// ----- §3: split arithmetic -----

use crate::quantization::quantized_scalar::Code;

/// Signed Euclidean floor division: returns `q = ⌊num / denom⌋`
/// (toward `−∞`) so the remainder is non-negative. Distinct from
/// Rust's `/` for negative numerators with positive denominators
/// (`-7 / 4 == -1` vs. `floor_div(-7, 4) == -2`).
///
/// Returns `None` for `denominator ≤ 0` or arithmetic overflow.
pub fn floor_div(numerator: Code, denominator: Code) -> Option<Code> {
    if denominator == 0 {
        return None;
    }
    if denominator < 0 {
        return None;
    }
    let q_trunc = numerator.checked_div(denominator)?;
    let r_trunc = numerator.checked_sub(q_trunc.checked_mul(denominator)?)?;
    if r_trunc < 0 {
        Some(q_trunc.checked_sub(1)?)
    } else {
        Some(q_trunc)
    }
}

/// Signed Euclidean ceil division: returns `q = ⌈num / denom⌉`
/// (toward `+∞`). Returns `None` for `denominator ≤ 0` or overflow.
pub fn ceil_div(numerator: Code, denominator: Code) -> Option<Code> {
    if denominator <= 0 {
        return None;
    }
    let q_trunc = numerator.checked_div(denominator)?;
    let r_trunc = numerator.checked_sub(q_trunc.checked_mul(denominator)?)?;
    if r_trunc > 0 {
        Some(q_trunc.checked_add(1)?)
    } else {
        Some(q_trunc)
    }
}

/// Per-endpoint witnesses for the upper-line split-arithmetic
/// identity. Every step uses floor division, so `line_sigma_code`
/// is a conservative LOWER estimate of `line(x)·s_v`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpperLineSplitWitness {
    /// `dx_step_1 = ⌊(d · x) / s_d⌋`.
    pub dx_step_1: Code,
    /// `d·x − dx_step_1·s_d ∈ [0, s_d)`.
    pub dx_step_1_rem: Code,
    /// `dx_sigma_code = ⌊(dx_step_1 · s_v) / s_w⌋`.
    pub dx_sigma_code: Code,
    /// `dx_step_1·s_v − dx_sigma_code·s_w ∈ [0, s_w)`.
    pub dx_sigma_rem: Code,
    /// `b_sigma_code = ⌊(b · s_v) / s_b⌋`.
    pub b_sigma_code: Code,
    /// `b·s_v − b_sigma_code·s_b ∈ [0, s_b)`.
    pub b_sigma_rem: Code,
    /// `dx_sigma_code + b_sigma_code`.
    pub line_sigma_code: Code,
    /// `line_sigma_code − sigma_upper_code`; must be `≥ 0` for the
    /// upper-line endpoint check to pass.
    pub diff: Code,
}

/// Per-endpoint witnesses for the lower-line split-arithmetic
/// identity. Every step uses ceil division, so `line_sigma_code` is
/// a conservative UPPER estimate of `line(x)·s_v`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LowerLineSplitWitness {
    pub dx_step_1: Code,
    pub dx_step_1_rem: Code,
    pub dx_sigma_code: Code,
    pub dx_sigma_rem: Code,
    pub b_sigma_code: Code,
    pub b_sigma_rem: Code,
    pub line_sigma_code: Code,
    /// `sigma_lower_code − line_sigma_code`; must be `≥ 0`.
    pub diff: Code,
}

/// Compute the upper-line split witnesses for one endpoint cell.
/// Every intermediate division rounds down, so the resulting
/// `diff ≥ 0` is strictly stronger than the real-valued
/// `line(x) ≥ σ_upper(x)/s_v`. Returns `None` on arithmetic
/// overflow or invalid scales.
pub fn split_line_witnesses_upper(
    d_code: Code,
    x_code: Code,
    b_code: Code,
    sigma_upper_code: Code,
    s_d: Code,
    s_b: Code,
    s_w: Code,
    s_v: Code,
) -> Option<UpperLineSplitWitness> {
    let dx = d_code.checked_mul(x_code)?;
    let dx_step_1 = floor_div(dx, s_d)?;
    let dx_step_1_rem = dx.checked_sub(dx_step_1.checked_mul(s_d)?)?;
    debug_assert!((0..s_d).contains(&dx_step_1_rem));

    let dx_step_1_sv = dx_step_1.checked_mul(s_v)?;
    let dx_sigma_code = floor_div(dx_step_1_sv, s_w)?;
    let dx_sigma_rem = dx_step_1_sv.checked_sub(dx_sigma_code.checked_mul(s_w)?)?;
    debug_assert!((0..s_w).contains(&dx_sigma_rem));

    let b_sv = b_code.checked_mul(s_v)?;
    let b_sigma_code = floor_div(b_sv, s_b)?;
    let b_sigma_rem = b_sv.checked_sub(b_sigma_code.checked_mul(s_b)?)?;
    debug_assert!((0..s_b).contains(&b_sigma_rem));

    let line_sigma_code = dx_sigma_code.checked_add(b_sigma_code)?;
    let diff = line_sigma_code.checked_sub(sigma_upper_code)?;
    Some(UpperLineSplitWitness {
        dx_step_1,
        dx_step_1_rem,
        dx_sigma_code,
        dx_sigma_rem,
        b_sigma_code,
        b_sigma_rem,
        line_sigma_code,
        diff,
    })
}

/// Compute the lower-line split witnesses for one endpoint cell.
/// Every intermediate division rounds up, so the resulting
/// `diff ≥ 0` is strictly stronger than the real-valued
/// `line(x) ≤ σ_lower(x)/s_v`.
pub fn split_line_witnesses_lower(
    d_code: Code,
    x_code: Code,
    b_code: Code,
    sigma_lower_code: Code,
    s_d: Code,
    s_b: Code,
    s_w: Code,
    s_v: Code,
) -> Option<LowerLineSplitWitness> {
    let dx = d_code.checked_mul(x_code)?;
    let dx_step_1 = ceil_div(dx, s_d)?;
    // Remainder convention: `dx_step_1·s_d − d·x ∈ [0, s_d)`.
    let dx_step_1_rem = dx_step_1.checked_mul(s_d)?.checked_sub(dx)?;
    debug_assert!((0..s_d).contains(&dx_step_1_rem));

    let dx_step_1_sv = dx_step_1.checked_mul(s_v)?;
    let dx_sigma_code = ceil_div(dx_step_1_sv, s_w)?;
    let dx_sigma_rem = dx_sigma_code.checked_mul(s_w)?.checked_sub(dx_step_1_sv)?;
    debug_assert!((0..s_w).contains(&dx_sigma_rem));

    let b_sv = b_code.checked_mul(s_v)?;
    let b_sigma_code = ceil_div(b_sv, s_b)?;
    let b_sigma_rem = b_sigma_code.checked_mul(s_b)?.checked_sub(b_sv)?;
    debug_assert!((0..s_b).contains(&b_sigma_rem));

    let line_sigma_code = dx_sigma_code.checked_add(b_sigma_code)?;
    let diff = sigma_lower_code.checked_sub(line_sigma_code)?;
    Some(LowerLineSplitWitness {
        dx_step_1,
        dx_step_1_rem,
        dx_sigma_code,
        dx_sigma_rem,
        b_sigma_code,
        b_sigma_rem,
        line_sigma_code,
        diff,
    })
}

#[cfg(test)]
mod split_arith_tests {
    use super::*;

    #[test]
    fn floor_div_positive_numerator() {
        assert_eq!(floor_div(7, 4), Some(1));
        assert_eq!(floor_div(8, 4), Some(2));
        assert_eq!(floor_div(0, 4), Some(0));
    }

    #[test]
    fn floor_div_negative_numerator() {
        assert_eq!(floor_div(-7, 4), Some(-2));
        assert_eq!(floor_div(-8, 4), Some(-2));
        assert_eq!(floor_div(-1, 4), Some(-1));
        assert_eq!(floor_div(-4, 4), Some(-1));
    }

    #[test]
    fn floor_div_remainder_invariant() {
        for n in -50i128..=50 {
            for d in 1i128..=8 {
                let q = floor_div(n, d).unwrap();
                let r = n - q * d;
                assert!(r >= 0 && r < d, "floor_div({n}, {d}) = {q}, r = {r}");
            }
        }
    }

    #[test]
    fn ceil_div_positive_numerator() {
        assert_eq!(ceil_div(7, 4), Some(2));
        assert_eq!(ceil_div(8, 4), Some(2));
        assert_eq!(ceil_div(0, 4), Some(0));
    }

    #[test]
    fn ceil_div_negative_numerator() {
        assert_eq!(ceil_div(-7, 4), Some(-1));
        assert_eq!(ceil_div(-8, 4), Some(-2));
        assert_eq!(ceil_div(-1, 4), Some(0));
    }

    #[test]
    fn ceil_div_remainder_invariant() {
        for n in -50i128..=50 {
            for d in 1i128..=8 {
                let q = ceil_div(n, d).unwrap();
                let r = q * d - n;
                assert!(r >= 0 && r < d, "ceil_div({n}, {d}) = {q}, r = {r}");
            }
        }
    }

    #[test]
    fn floor_div_rejects_zero_denominator() {
        assert_eq!(floor_div(7, 0), None);
    }

    #[test]
    fn ceil_div_rejects_nonpositive_denominator() {
        assert_eq!(ceil_div(7, 0), None);
        assert_eq!(ceil_div(7, -2), None);
    }

    /// Upper-line split: with d=4 (real 0.5), x=4 (real 0.5),
    /// b=0, scale s_d=s_w=8, s_b=8, s_v=2^8=256.
    /// line(x) = 0.5 · 0.5 + 0 = 0.25. line·s_v = 64.
    /// dx = 16. dx_step_1 = 16/8 = 2. dx_step_1·s_v = 512.
    /// dx_sigma_code = 512/8 = 64. b_sigma_code = 0.
    /// line_sigma_code = 64. ✓
    #[test]
    fn upper_line_split_simple() {
        let w = split_line_witnesses_upper(
            /*d*/ 4, /*x*/ 4, /*b*/ 0,
            /*sigma_upper*/ 50, // any value ≤ 64 → diff ≥ 0
            /*s_d*/ 8, /*s_b*/ 8, /*s_w*/ 8, /*s_v*/ 256,
        )
        .unwrap();
        assert_eq!(w.dx_step_1, 2);
        assert_eq!(w.dx_step_1_rem, 0);
        assert_eq!(w.dx_sigma_code, 64);
        assert_eq!(w.dx_sigma_rem, 0);
        assert_eq!(w.b_sigma_code, 0);
        assert_eq!(w.b_sigma_rem, 0);
        assert_eq!(w.line_sigma_code, 64);
        assert_eq!(w.diff, 14);
    }

    /// Upper-line split with negative x ⇒ negative dx ⇒ floor toward
    /// −∞ matters. d=4, x=−4, b=0. dx = −16. floor(-16/8) = -2.
    /// dx_step_1 = -2. dx_step_1·s_v = -512. floor(-512/8) = -64.
    /// dx_sigma_code = -64. line_sigma_code = -64. With sigma_upper
    /// at 0 (small), diff = -64 (negative ⇒ honest checks would
    /// reject this line as not above σ at x=-4, which is correct
    /// because line(-4) = -0.25 < σ_upper(-4) ≈ 0.5 for sigmoid).
    #[test]
    fn upper_line_split_negative_x_uses_floor_div() {
        let w = split_line_witnesses_upper(
            /*d*/ 4, /*x*/ -4, /*b*/ 0, /*sigma_upper*/ 0, /*s_d*/ 8,
            /*s_b*/ 8, /*s_w*/ 8, /*s_v*/ 256,
        )
        .unwrap();
        assert_eq!(w.dx_step_1, -2);
        assert_eq!(w.dx_step_1_rem, 0); // -16 - (-2)·8 = 0
        assert_eq!(w.dx_sigma_code, -64);
        assert_eq!(w.dx_sigma_rem, 0);
        assert_eq!(w.line_sigma_code, -64);
        assert_eq!(w.diff, -64);
    }

    /// Upper-line split with non-multiple d·x: d=5, x=3 ⇒ dx=15.
    /// floor(15/8) = 1. dx_step_1 = 1, rem = 7. dx_step_1·s_v = 256.
    /// floor(256/8) = 32. dx_sigma_code = 32, rem = 0.
    #[test]
    fn upper_line_split_nonzero_remainder() {
        let w = split_line_witnesses_upper(5, 3, 0, 0, 8, 8, 8, 256).unwrap();
        assert_eq!(w.dx_step_1, 1);
        assert_eq!(w.dx_step_1_rem, 7);
        assert!(w.dx_step_1_rem < 8); // ∈ [0, s_d)
        assert_eq!(w.dx_sigma_code, 32);
    }

    /// Lower-line split with the same numbers as upper: dx=16, but
    /// ceil(16/8) = 2 (same as floor when divisible). Then ceil(512/8)
    /// = 64. line_sigma_code = 64. With sigma_lower = 100, diff = 36.
    #[test]
    fn lower_line_split_simple() {
        let w = split_line_witnesses_lower(4, 4, 0, /*sigma_lower*/ 100, 8, 8, 8, 256).unwrap();
        assert_eq!(w.line_sigma_code, 64);
        assert_eq!(w.diff, 36);
    }

    /// Lower-line split with non-divisible: d=5, x=3, dx=15.
    /// ceil(15/8) = 2. dx_step_1 = 2, rem = 2·8 − 15 = 1.
    /// dx_step_1·s_v = 512. ceil(512/8) = 64. rem = 64·8 − 512 = 0.
    #[test]
    fn lower_line_split_uses_ceil_div() {
        let w = split_line_witnesses_lower(5, 3, 0, 0, 8, 8, 8, 256).unwrap();
        assert_eq!(w.dx_step_1, 2);
        assert_eq!(w.dx_step_1_rem, 1);
    }

    /// Lower-line split with negative x: d=5, x=-3, dx=-15.
    /// ceil(-15/8) = -1. rem = -1·8 − (-15) = 7. ∈ [0, 8) ✓
    #[test]
    fn lower_line_split_negative_x_uses_ceil_div() {
        let w = split_line_witnesses_lower(5, -3, 0, 0, 8, 8, 8, 256).unwrap();
        assert_eq!(w.dx_step_1, -1);
        assert_eq!(w.dx_step_1_rem, 7);
    }
}
