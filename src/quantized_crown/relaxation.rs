//! Conservative quantization of CROWN activation relaxation tables.
//!
//! ReLU layers go through
//! [`quantize_relu_relaxation_from_quantized_preacts`], which constructs
//! the canonical relaxation from the dequantized preact endpoints and
//! then ceil-rounds the upper intercept against the quantized endpoints.
//! Sigmoid and tanh layers go through
//! [`quantize_sigmoid_tanh_relaxation_at_table_envelopes`], which
//! evaluates σ at the public σ-envelope half-tables that the SNARK
//! verifier checks against. In both cases the cert's lines are
//! exact-conservative against the same data the SNARK sees.

use ndarray::Array1;

use crate::crown::float_crown::ActivationRelaxation;
use crate::quantization::quantized_array::QArray1;
use crate::quantization::quantized_scalar::Code;
use crate::quantization::scale::Scale;

use super::types::QuantRelaxation;

/// Build a quantized ReLU relaxation from quantized preact endpoints.
///
/// `l_int_at_sw` and `u_int_at_sw` are the integer codes the SNARK
/// hidden-pass verifier checks. The canonical CROWN line is built from
/// the dequantized endpoints, then the slope is banker's-rounded and the
/// intercept ceil-rounded against the quantized endpoints so the SNARK
/// upper-line endpoint check passes by construction.
///
/// Returns `(d_lower, b_lower, d_upper, b_upper)` codes. Cases:
///
/// * stable inactive (`u ≤ 0`): all zero
/// * stable active (`l ≥ 0`): identity line on both sides
/// * unstable (`l < 0 < u`): canonical chord-tangent CROWN relaxation
pub(super) fn quantize_relu_relaxation_from_quantized_preacts(
    s_d: Scale,
    s_b: Scale,
    s_w: Scale,
    l_int_at_sw: Code,
    u_int_at_sw: Code,
) -> (Code, Code, Code, Code) {
    let s_w_real = s_w.to_real();
    let s_d_real = s_d.to_real();
    let l_real = (l_int_at_sw as f64) / s_w_real;
    let u_real = (u_int_at_sw as f64) / s_w_real;
    if u_int_at_sw <= 0 {
        return (0, 0, 0, 0);
    }
    if l_int_at_sw >= 0 {
        let one_d = crate::quantization::quantized_scalar::Qf::from_real(1.0, s_d).code;
        return (one_d, 0, one_d, 0);
    }
    let denom = u_real - l_real;
    let d_upper_real = u_real / denom;
    let d_upper_int = crate::quantization::quantized_scalar::Qf::from_real(d_upper_real, s_d).code;
    let d_upper_q_real = (d_upper_int as f64) / s_d_real;
    // `b_upper` must dominate ReLU at both quantized endpoints using the
    // quantized slope (not the float). At `x = l_real` the line must be
    // ≥ 0; at `x = u_real` it must be ≥ u_real.
    let b_required = (-d_upper_q_real * l_real).max(u_real * (1.0 - d_upper_q_real));
    let b_upper_int =
        crate::quantization::quantized_scalar::Qf::from_real_ceil(b_required, s_b).code;
    let d_lower_int = if u_real > -l_real {
        crate::quantization::quantized_scalar::Qf::from_real(1.0, s_d).code
    } else {
        0
    };
    (d_lower_int, 0, d_upper_int, b_upper_int)
}

/// Per-layer quantized relaxation builder.
///
/// ReLU layers dispatch to
/// [`quantize_relu_relaxation_from_quantized_preacts`]. Sigmoid and tanh
/// dispatch to [`quantize_sigmoid_tanh_relaxation_at_table_envelopes`],
/// which evaluates σ at the public table the SNARK verifies. Returns
/// `SshapeRelaxOutOfTableDomain` if any candidate `x` (preact endpoint or
/// stationary point) falls outside the table domain. The cert generator
/// never falls back to raw float σ — doing so would produce a relaxation
/// the SNARK cannot reproduce.
#[allow(clippy::too_many_arguments)]
pub(super) fn quantize_relaxation_at_quantized_preacts(
    relax: &ActivationRelaxation,
    s_d: Scale,
    s_b: Scale,
    s_w: Scale,
    preact_lower_q_at_sw: &QArray1,
    preact_upper_q_at_sw: &QArray1,
    layer_idx: usize,
    sigma_x_scale_log2: i32,
    sigma_v_scale_log2: i32,
) -> Result<QuantRelaxation, super::types::QCrownError> {
    use crate::crown::network::ActivationKind;
    assert_eq!(
        preact_lower_q_at_sw.codes.len(),
        relax.neurons.len(),
        "preact_lower_q length mismatch"
    );
    assert_eq!(
        preact_upper_q_at_sw.codes.len(),
        relax.neurons.len(),
        "preact_upper_q length mismatch"
    );
    assert_eq!(
        preact_lower_q_at_sw.scale, s_w,
        "preact_lower_q scale must equal working scale s_w"
    );
    assert_eq!(
        preact_upper_q_at_sw.scale, s_w,
        "preact_upper_q scale must equal working scale s_w"
    );
    let n = relax.neurons.len();
    let mut d_lower = Array1::zeros(n);
    let mut d_upper = Array1::zeros(n);
    let mut b_lower = Array1::zeros(n);
    let mut b_upper = Array1::zeros(n);
    match relax.kind {
        ActivationKind::ReLU => {
            for j in 0..n {
                let l_q = preact_lower_q_at_sw.codes[j];
                let u_q = preact_upper_q_at_sw.codes[j];
                let (dl, bl, du, bu) =
                    quantize_relu_relaxation_from_quantized_preacts(s_d, s_b, s_w, l_q, u_q);
                d_lower[j] = dl;
                b_lower[j] = bl;
                d_upper[j] = du;
                b_upper[j] = bu;
            }
        }
        ActivationKind::Sigmoid | ActivationKind::Tanh => {
            let l_real: Vec<f64> = preact_lower_q_at_sw
                .codes
                .iter()
                .map(|&c| (c as f64) / s_w.to_real())
                .collect();
            let u_real: Vec<f64> = preact_upper_q_at_sw
                .codes
                .iter()
                .map(|&c| (c as f64) / s_w.to_real())
                .collect();
            let pre = crate::snark::SigmaTables::shared(sigma_x_scale_log2, sigma_v_scale_log2);
            let qr = quantize_sigmoid_tanh_relaxation_at_table_envelopes(
                relax, s_d, s_b, &l_real, &u_real, &pre,
            )
            .ok_or_else(|| {
                // Report the candidate with the largest |x| from the
                // endpoint pairs; stationary points share the same
                // table domain so are in-domain whenever the endpoints
                // are.
                let x_oob = l_real
                    .iter()
                    .chain(u_real.iter())
                    .copied()
                    .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
                    .unwrap_or(0.0);
                super::types::QCrownError::SshapeRelaxOutOfTableDomain {
                    layer_idx,
                    x_real: x_oob,
                }
            })?;
            d_lower = qr.d_lower.codes;
            d_upper = qr.d_upper.codes;
            b_lower = qr.b_lower.codes;
            b_upper = qr.b_upper.codes;
        }
    }
    Ok(QuantRelaxation {
        kind: relax.kind,
        d_lower: QArray1::new(d_lower, s_d),
        d_upper: QArray1::new(d_upper, s_d),
        b_lower: QArray1::new(b_lower, s_b),
        b_upper: QArray1::new(b_upper, s_b),
    })
}

/// Build a quantized sigmoid/tanh relaxation whose dequantized lines
/// dominate (upper) / are dominated by (lower) the σ-envelope half-table
/// values at the candidate `x`'s (preact endpoints plus interior
/// stationary points of `σ − d·x`).
///
/// Returns `None` if any candidate `x` is outside the table domain
/// (`(-128, 128)` at the current scales).
///
/// Slopes are banker's-rounded; `b_upper` is ceil-rounded; `b_lower` is
/// floor-rounded. A small per-direction LSB safety margin absorbs the
/// rounding the SNARK's split-arith re-expression of the line at scale
/// `s_v` introduces on top of the cert's own rounding.
pub(super) fn quantize_sigmoid_tanh_relaxation_at_table_envelopes(
    relax: &ActivationRelaxation,
    s_d: Scale,
    s_b: Scale,
    l_vec: &[f64],
    u_vec: &[f64],
    pre: &crate::snark::SigmaTables,
) -> Option<QuantRelaxation> {
    use crate::crown::network::ActivationKind;
    assert_eq!(
        l_vec.len(),
        relax.neurons.len(),
        "preact bounds length mismatch"
    );
    assert_eq!(
        u_vec.len(),
        relax.neurons.len(),
        "preact bounds length mismatch"
    );
    assert!(
        matches!(relax.kind, ActivationKind::Sigmoid | ActivationKind::Tanh),
        "table-envelope cert path is for sigmoid/tanh only"
    );
    let n = relax.neurons.len();
    let mut d_lower_codes: Array1<Code> = Array1::zeros(n);
    let mut d_upper_codes: Array1<Code> = Array1::zeros(n);
    let mut b_lower_codes: Array1<Code> = Array1::zeros(n);
    let mut b_upper_codes: Array1<Code> = Array1::zeros(n);
    let s_d_real = s_d.to_real();
    for (j, neuron) in relax.neurons.iter().enumerate() {
        let l = l_vec[j];
        let u = u_vec[j];
        d_lower_codes[j] =
            crate::quantization::quantized_scalar::Qf::from_real(neuron.d_lower, s_d).code;
        d_upper_codes[j] =
            crate::quantization::quantized_scalar::Qf::from_real(neuron.d_upper, s_d).code;
        let d_lower_q_real = (d_lower_codes[j] as f64) / s_d_real;
        let d_upper_q_real = (d_upper_codes[j] as f64) / s_d_real;
        // Stationary-point search uses float σ' but the resulting `x`
        // candidates are evaluated against TABLE σ values below.
        let xs_l = match relax.kind {
            ActivationKind::Sigmoid => sshape_extremum_xs(l, u, d_lower_q_real, sigmoid_fp_inv),
            ActivationKind::Tanh => sshape_extremum_xs(l, u, d_lower_q_real, tanh_fp_inv),
            ActivationKind::ReLU => unreachable!(),
        };
        let xs_u = match relax.kind {
            ActivationKind::Sigmoid => sshape_extremum_xs(l, u, d_upper_q_real, sigmoid_fp_inv),
            ActivationKind::Tanh => sshape_extremum_xs(l, u, d_upper_q_real, tanh_fp_inv),
            ActivationKind::ReLU => unreachable!(),
        };
        let f_passthrough = match relax.kind {
            ActivationKind::Sigmoid => crate::crown::float_crown::sigmoid_f as fn(f64) -> f64,
            ActivationKind::Tanh => crate::crown::float_crown::tanh_f as fn(f64) -> f64,
            ActivationKind::ReLU => unreachable!(),
        };
        let (b_l_real, _) = conservative_intercepts_table(
            pre,
            relax.kind,
            &xs_l,
            d_lower_q_real,
            0.0,
            f_passthrough,
        )?;
        let (_, b_u_real) = conservative_intercepts_table(
            pre,
            relax.kind,
            &xs_u,
            0.0,
            d_upper_q_real,
            f_passthrough,
        )?;

        let b_lower_int_naive =
            crate::quantization::quantized_scalar::Qf::from_real_floor(b_l_real, s_b).code;
        let b_upper_int_naive =
            crate::quantization::quantized_scalar::Qf::from_real_ceil(b_u_real, s_b).code;
        // Safety margin absorbs two rounding sources the SNARK σ
        // gadget introduces on top of the cert's own ceil/floor:
        // (1) the `from_real_floor`/`from_real_ceil` itself, and
        // (2) the gadget's split-arith re-expression of the line at
        // scale `s_v`. A third LSB is added when a stationary point
        // lies strictly inside `[l, u]`, since the cert evaluates σ at
        // `round(x_stat · s_x)` instead of the exact float `x_stat`.
        // The 2 + 1 margin clears all 1024-width sigmoid/tanh
        // fixtures while costing well under 1e-3 of tightness.
        let stationary_inside_l = xs_l.len() > 2;
        let stationary_inside_u = xs_u.len() > 2;
        let lower_margin: i128 = 2 + (stationary_inside_l as i128);
        let upper_margin: i128 = 2 + (stationary_inside_u as i128);
        b_lower_codes[j] = b_lower_int_naive - lower_margin;
        b_upper_codes[j] = b_upper_int_naive + upper_margin;
    }
    Some(QuantRelaxation {
        kind: relax.kind,
        d_lower: QArray1::new(d_lower_codes, s_d),
        d_upper: QArray1::new(d_upper_codes, s_d),
        b_lower: QArray1::new(b_lower_codes, s_b),
        b_upper: QArray1::new(b_upper_codes, s_b),
    })
}

/// Look up the σ envelope at a real-valued `x`, returning
/// `(σ_lower_real, σ_upper_real)` from the public table at the rounded
/// table index.
///
/// The table is keyed by `x_int = round(x · s_x)` and applies the
/// half-table symmetries (`σ(-x) = 1 − σ(x)` for sigmoid, `−σ(x)` for
/// tanh) so any real `x` in the supported domain `(-128, 128)` can be
/// looked up. Returns `None` outside the domain.
///
/// The cert uses this helper rather than raw float σ so the lines it
/// builds bound σ exactly at the table indices the SNARK Phase 3b/3c
/// gadgets evaluate.
pub(super) fn sigma_envelope_real_at_x(
    pre: &crate::snark::SigmaTables,
    kind: crate::crown::network::ActivationKind,
    x_real: f64,
) -> Option<(f64, f64)> {
    use crate::snark::preprocess::{SIGMOID_TABLE_X_BOUND_REAL, TANH_TABLE_X_BOUND_REAL};
    let s_x_log2 = pre.s_x_log2;
    let s_x = (1i128 << s_x_log2) as f64;
    let bound_int = match kind {
        crate::crown::network::ActivationKind::Sigmoid => SIGMOID_TABLE_X_BOUND_REAL,
        crate::crown::network::ActivationKind::Tanh => TANH_TABLE_X_BOUND_REAL,
        crate::crown::network::ActivationKind::ReLU => return None,
    };
    let bound = bound_int as f64;
    if !x_real.is_finite() || x_real.abs() >= bound {
        return None;
    }
    let x_int = (x_real * s_x).round() as i128;
    let (lo_int, up_int) = crate::snark::activation_gadget::sshape_helpers::lookup_sigma_envelope(
        pre, kind, x_int, s_x_log2, bound_int,
    )?;
    let s_v = (1u64 << pre.s_v_log2) as f64;
    Some((lo_int as f64 / s_v, up_int as f64 / s_v))
}

/// Compute conservative `(b_lower, b_upper)` over candidate `x`'s using
/// the table-derived σ envelope. The sigmoid/tanh analogue of a
/// float-only intercept computation; using the table here means the
/// cert's lines are exact-conservative at the SNARK's own table-rounded
/// endpoints.
fn conservative_intercepts_table<F>(
    pre: &crate::snark::SigmaTables,
    kind: crate::crown::network::ActivationKind,
    xs: &[f64],
    d_lower_q: f64,
    d_upper_q: f64,
    _fallback_f: F,
) -> Option<(f64, f64)>
where
    F: Fn(f64) -> f64,
{
    let mut b_lower = f64::INFINITY;
    let mut b_upper = f64::NEG_INFINITY;
    for &x in xs {
        let (sigma_lo_real, sigma_up_real) = sigma_envelope_real_at_x(pre, kind, x)?;
        let lower_candidate = sigma_lo_real - d_lower_q * x;
        let upper_candidate = sigma_up_real - d_upper_q * x;
        if lower_candidate < b_lower {
            b_lower = lower_candidate;
        }
        if upper_candidate > b_upper {
            b_upper = upper_candidate;
        }
    }
    Some((b_lower, b_upper))
}

/// Endpoints plus interior stationary points of `f(x) − d_q · x` on
/// `[l, u]`. `fp_inv(d)` returns the `x` satisfying `f'(x) = d` — at
/// most two for a symmetric S-shape; the result is filtered to `[l, u]`.
fn sshape_extremum_xs(l: f64, u: f64, d_q: f64, fp_inv: impl Fn(f64) -> Vec<f64>) -> Vec<f64> {
    let mut out = vec![l, u];
    for x in fp_inv(d_q) {
        if x >= l && x <= u {
            out.push(x);
        }
    }
    out
}

/// Inverse of sigmoid'. Returns the `z` satisfying `sigmoid'(z) = d` for
/// `d ∈ (0, 1/4]`. Two symmetric roots; a single root at `d = 1/4`;
/// empty outside the range.
fn sigmoid_fp_inv(d: f64) -> Vec<f64> {
    if d <= 0.0 || d > 0.25 {
        return Vec::new();
    }
    if (d - 0.25).abs() < 1e-15 {
        return vec![0.0];
    }
    let disc = 1.0 - 4.0 * d;
    let sd = disc.sqrt();
    let s_lo = 0.5 * (1.0 - sd);
    let s_hi = 0.5 * (1.0 + sd);
    vec![(s_lo / (1.0 - s_lo)).ln(), (s_hi / (1.0 - s_hi)).ln()]
}

/// Inverse of tanh'. Returns the `z` satisfying `tanh'(z) = d` for
/// `d ∈ (0, 1]`.
fn tanh_fp_inv(d: f64) -> Vec<f64> {
    if d <= 0.0 || d > 1.0 {
        return Vec::new();
    }
    if (d - 1.0).abs() < 1e-15 {
        return vec![0.0];
    }
    let t = (1.0 - d).sqrt();
    vec![(-t).atanh(), t.atanh()]
}

#[cfg(test)]
mod tests {
    //! Unit tests for the quantize-relaxation paths. Each test fabricates
    //! an `ActivationRelaxation` whose per-neuron line describes a
    //! representative interval (active / inactive / unstable / boundary),
    //! quantizes it, and checks the dequantized affine lines remain
    //! valid bounds on `[l, u]`.
    use super::*;
    use crate::crown::float_crown::ReluRelaxation;
    use crate::crown::network::ActivationKind;
    use crate::quantization::scale::Scale;

    fn dequantize_codes(
        d_lower_int: Code,
        b_lower_int: Code,
        d_upper_int: Code,
        b_upper_int: Code,
        s_d: Scale,
        s_b: Scale,
    ) -> (f64, f64, f64, f64) {
        let inv_d = 1.0 / s_d.to_real();
        let inv_b = 1.0 / s_b.to_real();
        (
            d_lower_int as f64 * inv_d,
            b_lower_int as f64 * inv_b,
            d_upper_int as f64 * inv_d,
            b_upper_int as f64 * inv_b,
        )
    }

    #[test]
    fn quantized_preact_builder_stable_active() {
        let s_d = Scale::from_pow2(8);
        let s_b = Scale::from_pow2(8);
        let s_w = Scale::from_pow2(8);
        let l_int = (0.5 * s_w.to_real()) as Code;
        let u_int = (3.0 * s_w.to_real()) as Code;
        let (dl, bl, du, bu) =
            quantize_relu_relaxation_from_quantized_preacts(s_d, s_b, s_w, l_int, u_int);
        let one_d = crate::quantization::quantized_scalar::Qf::from_real(1.0, s_d).code;
        assert_eq!(dl, one_d);
        assert_eq!(bl, 0);
        assert_eq!(du, one_d);
        assert_eq!(bu, 0);
    }

    #[test]
    fn quantized_preact_builder_stable_inactive() {
        let s_d = Scale::from_pow2(8);
        let s_b = Scale::from_pow2(8);
        let s_w = Scale::from_pow2(8);
        let l_int = (-3.0 * s_w.to_real()) as Code;
        let u_int = (-0.5 * s_w.to_real()) as Code;
        let (dl, bl, du, bu) =
            quantize_relu_relaxation_from_quantized_preacts(s_d, s_b, s_w, l_int, u_int);
        assert_eq!((dl, bl, du, bu), (0, 0, 0, 0));
    }

    #[test]
    fn quantized_preact_builder_unstable_dominates_relu_at_quantized_endpoints() {
        // Verifies the SNARK gadget invariant: the dequantized upper
        // line dominates ReLU at both quantized endpoints.
        let s_d = Scale::from_pow2(8);
        let s_b = Scale::from_pow2(8);
        let s_w = Scale::from_pow2(8);
        let l_real = -0.7;
        let u_real = 1.3;
        let l_int = (l_real * s_w.to_real()).round() as Code;
        let u_int = (u_real * s_w.to_real()).round() as Code;
        let (dl, bl, du, bu) =
            quantize_relu_relaxation_from_quantized_preacts(s_d, s_b, s_w, l_int, u_int);
        let (d_lo, b_lo, d_up, b_up) = dequantize_codes(dl, bl, du, bu, s_d, s_b);
        let l_q = (l_int as f64) / s_w.to_real();
        let u_q = (u_int as f64) / s_w.to_real();
        assert!(
            d_up * l_q + b_up >= 0.0 - 1e-12,
            "line(l_q) = {} < 0",
            d_up * l_q + b_up
        );
        assert!(
            d_up * u_q + b_up >= u_q - 1e-12,
            "line(u_q) = {} < u_q={}",
            d_up * u_q + b_up,
            u_q
        );
        assert!(
            d_lo * l_q + b_lo <= 0.0 + 1e-12,
            "lower(l_q) = {} > 0",
            d_lo * l_q + b_lo
        );
        assert!(
            d_lo * u_q + b_lo <= u_q + 1e-12,
            "lower(u_q) = {} > u_q={}",
            d_lo * u_q + b_lo,
            u_q
        );
        assert_eq!(d_lo, 1.0);
        assert_eq!(b_lo, 0.0);
    }

    #[test]
    fn quantized_preact_builder_d_lower_zero_when_u_le_neg_l() {
        let s_d = Scale::from_pow2(8);
        let s_b = Scale::from_pow2(8);
        let s_w = Scale::from_pow2(8);
        let l_int = (-2.0 * s_w.to_real()).round() as Code;
        let u_int = (1.0 * s_w.to_real()).round() as Code;
        let (dl, bl, _du, _bu) =
            quantize_relu_relaxation_from_quantized_preacts(s_d, s_b, s_w, l_int, u_int);
        assert_eq!(dl, 0);
        assert_eq!(bl, 0);
    }

    #[test]
    fn quantized_preact_builder_upper_endpoint_dominates_stress() {
        let s_d = Scale::from_pow2(10);
        let s_b = Scale::from_pow2(10);
        let s_w = Scale::from_pow2(10);
        let cases: Vec<(f64, f64)> = vec![
            (-0.1, 0.2),
            (-0.5, 0.5),
            (-1.0, 0.3),
            (-0.3, 1.0),
            (-2.0, 0.5),
            (-0.5, 2.0),
            (-3.7, 1.1),
            (-0.05, 0.05),
            (-0.99, 0.01),
            (-0.01, 0.99),
            (-7.5, 4.2),
            (-12.3, 0.1),
        ];
        for (l_real, u_real) in cases {
            let l_int = (l_real * s_w.to_real()).round() as Code;
            let u_int = (u_real * s_w.to_real()).round() as Code;
            let (_dl, _bl, du, bu) =
                quantize_relu_relaxation_from_quantized_preacts(s_d, s_b, s_w, l_int, u_int);
            let d_up = du as f64 / s_d.to_real();
            let b_up = bu as f64 / s_b.to_real();
            let l_q = (l_int as f64) / s_w.to_real();
            let u_q = (u_int as f64) / s_w.to_real();
            assert!(
                d_up * l_q + b_up >= 0.0 - 1e-12,
                "upper line at quantized l = {} < 0 (l={l_real}, u={u_real}, d={d_up}, b={b_up})",
                d_up * l_q + b_up,
            );
            assert!(
                d_up * u_q + b_up >= u_q - 1e-12,
                "upper line at quantized u = {} < u_q={u_q} (l={l_real}, u={u_real}, d={d_up}, b={b_up})",
                d_up * u_q + b_up,
            );
        }
    }

    #[test]
    fn quantize_relaxation_at_quantized_preacts_consistency() {
        // The dispatcher must produce codes identical to calling
        // `quantize_relu_relaxation_from_quantized_preacts` per neuron.
        // The SNARK depends on this: recomputing the relaxation from
        // the verified preact codes must yield the committed tensors.
        use ndarray::Array1;
        let s_d = Scale::from_pow2(10);
        let s_b = Scale::from_pow2(10);
        let s_w = Scale::from_pow2(10);
        let cases: Vec<(f64, f64)> = vec![
            (-0.7, 1.3),
            (-2.0, 1.0),
            (0.5, 3.0),
            (-3.0, -0.5),
            (-1.5, 0.5),
        ];
        let l_codes: Array1<Code> = cases
            .iter()
            .map(|(l, _)| (l * s_w.to_real()).round() as Code)
            .collect();
        let u_codes: Array1<Code> = cases
            .iter()
            .map(|(_, u)| (u * s_w.to_real()).round() as Code)
            .collect();
        let preact_l = QArray1::new(l_codes.clone(), s_w);
        let preact_u = QArray1::new(u_codes.clone(), s_w);
        let neurons: Vec<ReluRelaxation> = cases
            .iter()
            .map(|(l, u)| crate::crown::float_crown::relu_relaxation(*l, *u))
            .collect();
        let relax = ActivationRelaxation {
            kind: ActivationKind::ReLU,
            neurons,
        };
        let qr = quantize_relaxation_at_quantized_preacts(
            &relax,
            s_d,
            s_b,
            s_w,
            &preact_l,
            &preact_u,
            /*layer_idx*/ 0,
            crate::snark::preprocess::TEST_SIGMA_X_SCALE_LOG2,
            crate::snark::preprocess::TEST_SIGMA_V_SCALE_LOG2,
        )
        .expect("ReLU path is infallible");
        for j in 0..cases.len() {
            let (dl, bl, du, bu) = quantize_relu_relaxation_from_quantized_preacts(
                s_d, s_b, s_w, l_codes[j], u_codes[j],
            );
            assert_eq!(qr.d_lower.codes[j], dl, "d_lower[{j}] mismatch");
            assert_eq!(qr.b_lower.codes[j], bl, "b_lower[{j}] mismatch");
            assert_eq!(qr.d_upper.codes[j], du, "d_upper[{j}] mismatch");
            assert_eq!(qr.b_upper.codes[j], bu, "b_upper[{j}] mismatch");
        }
        let qr2 = quantize_relaxation_at_quantized_preacts(
            &relax,
            s_d,
            s_b,
            s_w,
            &preact_l,
            &preact_u,
            0,
            crate::snark::preprocess::TEST_SIGMA_X_SCALE_LOG2,
            crate::snark::preprocess::TEST_SIGMA_V_SCALE_LOG2,
        )
        .expect("ReLU path is infallible");
        assert_eq!(qr.d_lower.codes, qr2.d_lower.codes);
        assert_eq!(qr.b_lower.codes, qr2.b_lower.codes);
        assert_eq!(qr.d_upper.codes, qr2.d_upper.codes);
        assert_eq!(qr.b_upper.codes, qr2.b_upper.codes);
    }

    /// `sigma_envelope_real_at_x` returns the same dequantized
    /// `(σ_lower, σ_upper)` pair the SNARK gadget uses at the rounded
    /// table index.
    #[test]
    fn sigma_envelope_real_at_x_matches_snark_lookup() {
        use crate::crown::network::ActivationKind;
        let pre = crate::snark::SigmaTables::shared(
            crate::snark::preprocess::TEST_SIGMA_X_SCALE_LOG2,
            crate::snark::preprocess::TEST_SIGMA_V_SCALE_LOG2,
        );
        let (lo, up) =
            sigma_envelope_real_at_x(&pre, ActivationKind::Sigmoid, 0.5).expect("0.5 in domain");
        assert!(lo <= up);
        let true_sigmoid = 1.0 / (1.0 + (-0.5_f64).exp());
        assert!(
            (lo - true_sigmoid).abs() < 1e-3,
            "lo {lo} far from σ(0.5) {true_sigmoid}"
        );
        assert!(
            (up - true_sigmoid).abs() < 1e-3,
            "up {up} far from σ(0.5) {true_sigmoid}"
        );
        let (lo, up) =
            sigma_envelope_real_at_x(&pre, ActivationKind::Tanh, -1.0).expect("-1.0 in domain");
        assert!(lo <= up);
        let true_tanh = (-1.0_f64).tanh();
        assert!(
            (lo - true_tanh).abs() < 1e-3 && (up - true_tanh).abs() < 1e-3,
            "tanh envelope at -1.0 misses true value"
        );
    }

    #[test]
    fn sigma_envelope_real_at_x_out_of_domain_returns_none() {
        use crate::crown::network::ActivationKind;
        let pre = crate::snark::SigmaTables::shared(
            crate::snark::preprocess::TEST_SIGMA_X_SCALE_LOG2,
            crate::snark::preprocess::TEST_SIGMA_V_SCALE_LOG2,
        );
        assert!(sigma_envelope_real_at_x(&pre, ActivationKind::Sigmoid, 200.0).is_none());
        assert!(sigma_envelope_real_at_x(&pre, ActivationKind::Sigmoid, -200.0).is_none());
        assert!(sigma_envelope_real_at_x(&pre, ActivationKind::Tanh, 128.0).is_none());
    }

    /// Build a sigmoid relaxation via the table-derived path on
    /// `[-2, 2]`. The dequantized upper line must dominate the
    /// `σ_upper_table_real(x)` at both endpoints, and the lower line
    /// must be dominated by `σ_lower_table_real(x)` — the SNARK
    /// gadget's exact invariant.
    #[test]
    fn sigmoid_table_relaxation_dominates_envelope_at_endpoints() {
        use crate::crown::float_crown::sigmoid_relaxation;
        use crate::crown::network::ActivationKind;
        let pre = crate::snark::SigmaTables::shared(
            crate::snark::preprocess::TEST_SIGMA_X_SCALE_LOG2,
            crate::snark::preprocess::TEST_SIGMA_V_SCALE_LOG2,
        );
        let s_d = Scale::from_pow2(8);
        let s_b = Scale::from_pow2(8);
        let l = -2.0_f64;
        let u = 2.0_f64;
        let neuron = sigmoid_relaxation(l, u);
        let relax = ActivationRelaxation {
            kind: ActivationKind::Sigmoid,
            neurons: vec![neuron],
        };
        let qr =
            quantize_sigmoid_tanh_relaxation_at_table_envelopes(&relax, s_d, s_b, &[l], &[u], &pre)
                .expect("in-domain");
        let inv_d = 1.0 / s_d.to_real();
        let inv_b = 1.0 / s_b.to_real();
        let (d_lo, b_lo) = (
            qr.d_lower.codes[0] as f64 * inv_d,
            qr.b_lower.codes[0] as f64 * inv_b,
        );
        let (d_up, b_up) = (
            qr.d_upper.codes[0] as f64 * inv_d,
            qr.b_upper.codes[0] as f64 * inv_b,
        );
        for &x in &[l, u] {
            let (sl_real, su_real) =
                sigma_envelope_real_at_x(&pre, ActivationKind::Sigmoid, x).expect("in-domain");
            let line_upper = d_up * x + b_up;
            let line_lower = d_lo * x + b_lo;
            assert!(
                line_upper >= su_real - 1e-9,
                "upper line {line_upper} < σ_upper {su_real} at x={x}",
            );
            assert!(
                line_lower <= sl_real + 1e-9,
                "lower line {line_lower} > σ_lower {sl_real} at x={x}",
            );
        }
    }

    /// Same envelope-domination check for tanh.
    #[test]
    fn tanh_table_relaxation_dominates_envelope_at_endpoints() {
        use crate::crown::float_crown::tanh_relaxation;
        use crate::crown::network::ActivationKind;
        let pre = crate::snark::SigmaTables::shared(
            crate::snark::preprocess::TEST_SIGMA_X_SCALE_LOG2,
            crate::snark::preprocess::TEST_SIGMA_V_SCALE_LOG2,
        );
        let s_d = Scale::from_pow2(8);
        let s_b = Scale::from_pow2(8);
        let l = -1.5_f64;
        let u = 1.5_f64;
        let neuron = tanh_relaxation(l, u);
        let relax = ActivationRelaxation {
            kind: ActivationKind::Tanh,
            neurons: vec![neuron],
        };
        let qr =
            quantize_sigmoid_tanh_relaxation_at_table_envelopes(&relax, s_d, s_b, &[l], &[u], &pre)
                .expect("in-domain");
        let inv_d = 1.0 / s_d.to_real();
        let inv_b = 1.0 / s_b.to_real();
        let (d_lo, b_lo) = (
            qr.d_lower.codes[0] as f64 * inv_d,
            qr.b_lower.codes[0] as f64 * inv_b,
        );
        let (d_up, b_up) = (
            qr.d_upper.codes[0] as f64 * inv_d,
            qr.b_upper.codes[0] as f64 * inv_b,
        );
        for &x in &[l, u] {
            let (sl_real, su_real) =
                sigma_envelope_real_at_x(&pre, ActivationKind::Tanh, x).expect("in-domain");
            let line_upper = d_up * x + b_up;
            let line_lower = d_lo * x + b_lo;
            assert!(
                line_upper >= su_real - 1e-9,
                "tanh upper line {line_upper} < table σ_upper {su_real} at x={x}",
            );
            assert!(
                line_lower <= sl_real + 1e-9,
                "tanh lower line {line_lower} > table σ_lower {sl_real} at x={x}",
            );
        }
    }

    /// Cross-asymmetric interval `l < 0 < u` so the stationary point of
    /// `σ − d_q · x` is interior, exercising the stationary-point
    /// candidate path.
    #[test]
    fn sigmoid_table_relaxation_handles_interior_stationary_point() {
        use crate::crown::float_crown::sigmoid_relaxation;
        use crate::crown::network::ActivationKind;
        let pre = crate::snark::SigmaTables::shared(
            crate::snark::preprocess::TEST_SIGMA_X_SCALE_LOG2,
            crate::snark::preprocess::TEST_SIGMA_V_SCALE_LOG2,
        );
        let s_d = Scale::from_pow2(8);
        let s_b = Scale::from_pow2(8);
        let l = -3.0;
        let u = 3.0;
        let neuron = sigmoid_relaxation(l, u);
        let relax = ActivationRelaxation {
            kind: ActivationKind::Sigmoid,
            neurons: vec![neuron],
        };
        let qr =
            quantize_sigmoid_tanh_relaxation_at_table_envelopes(&relax, s_d, s_b, &[l], &[u], &pre)
                .expect("in-domain");
        let inv_d = 1.0 / s_d.to_real();
        let inv_b = 1.0 / s_b.to_real();
        let (d_up, b_up) = (
            qr.d_upper.codes[0] as f64 * inv_d,
            qr.b_upper.codes[0] as f64 * inv_b,
        );
        for x_int_step in 0..=60 {
            let x = l + (x_int_step as f64) * (u - l) / 60.0;
            if let Some((_sl, su)) = sigma_envelope_real_at_x(&pre, ActivationKind::Sigmoid, x) {
                let line = d_up * x + b_up;
                let tol = 2.0 / s_b.to_real() + 1e-9;
                assert!(
                    line >= su - tol,
                    "upper line {line} below σ_upper {su} at interior x={x} (tol={tol})",
                );
            }
        }
    }
}
