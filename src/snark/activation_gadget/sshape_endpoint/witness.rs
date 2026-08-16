//! Witness construction for the endpoint gadget: scale precondition,
//! preact-to-table-index conversion, kind-aware σ_used reconstruction
//! from the half-table, and per-batch split-arithmetic witnesses.

use ark_bn254::Fr;

use crate::crown::network::ActivationKind;
use crate::quantization::scale::Scale;
use crate::snark_primitives::finite_field::fr_to_signed_i128;

use crate::snark::errors::SnarkError;

/// Each split-arith remainder is bounded by one of `{s_d, s_b, s_w}`,
/// so each scale must individually fit in `[0, 2^range_bits)` at this
/// proof’s gadget budget (19 default).
pub fn scale_precondition_holds(s_d: Scale, s_b: Scale, s_w: Scale, range_bits: usize) -> bool {
    if !s_d.is_pow2() || !s_b.is_pow2() || !s_w.is_pow2() {
        return false;
    }
    let s_d_e = match s_d.pow2_exponent() {
        Ok(e) => e,
        Err(_) => return false,
    };
    let s_b_e = match s_b.pow2_exponent() {
        Ok(e) => e,
        Err(_) => return false,
    };
    let s_w_e = match s_w.pow2_exponent() {
        Ok(e) => e,
        Err(_) => return false,
    };
    if s_d_e < 0 || s_b_e < 0 || s_w_e < 0 {
        return false;
    }
    let bits = range_bits as i32;
    s_d_e <= bits && s_b_e <= bits && s_w_e <= bits
}

/// Convert preact code at `s_w` to absolute table index at
/// `s_x = 2^s_x_log2`. Returns `None` if the conversion isn't
/// representable in i128 or if `s_w` isn't a power of two.
pub fn preact_to_abs_table_index(preact_code: i128, s_w: Scale, s_x_log2: i32) -> Option<i128> {
    let s_w_e = s_w.pow2_exponent().ok()?;
    let abs_code = preact_code.checked_abs()?;
    if s_x_log2 >= s_w_e {
        Some(abs_code.checked_shl((s_x_log2 - s_w_e) as u32)?)
    } else {
        Some(abs_code >> (s_w_e - s_x_log2))
    }
}

/// Reconstruct `σ_used` at signed `x` from the positive half-table
/// codes `(σ_upper_at_abs, σ_lower_at_abs)` and the sign bit. Uses the
/// closed form `σ_used = same + sign · (neg_correction − same)` where
/// `same` is the same-direction envelope at `|x|` and `neg_correction`
/// is `s_v − opp` (sigmoid) or `−opp` (tanh).
pub(crate) fn compute_sigma_used(
    kind: ActivationKind,
    line: super::types::SshapeLineKind,
    sigma_upper_at_abs: i128,
    sigma_lower_at_abs: i128,
    sign: i128,
    s_v: i128,
) -> i128 {
    use super::types::SshapeLineKind;
    let (same, opp) = match line {
        SshapeLineKind::Upper => (sigma_upper_at_abs, sigma_lower_at_abs),
        SshapeLineKind::Lower => (sigma_lower_at_abs, sigma_upper_at_abs),
    };
    let neg_correction = match kind {
        ActivationKind::Sigmoid => s_v - opp,
        ActivationKind::Tanh => -opp,
        ActivationKind::ReLU => unreachable!(),
    };
    same + sign * (neg_correction - same)
}

/// `compute_sigma_used` lifted to `Fr` for the sumcheck inner loop.
pub(crate) fn compute_sigma_used_fr(
    kind: ActivationKind,
    line: super::types::SshapeLineKind,
    sigma_upper_at_abs: Fr,
    sigma_lower_at_abs: Fr,
    sign: Fr,
    s_v_fr: Fr,
) -> Fr {
    use super::types::SshapeLineKind;
    let (same, opp) = match line {
        SshapeLineKind::Upper => (sigma_upper_at_abs, sigma_lower_at_abs),
        SshapeLineKind::Lower => (sigma_lower_at_abs, sigma_upper_at_abs),
    };
    let neg_correction = match kind {
        ActivationKind::Sigmoid => s_v_fr - opp,
        ActivationKind::Tanh => -opp,
        ActivationKind::ReLU => unreachable!(),
    };
    same + sign * (neg_correction - same)
}

/// Per-cell split-arithmetic witness bundle, holding every integer
/// code that feeds the combined sumcheck.
#[derive(Clone, Debug)]
pub(crate) struct SplitArithCellWitness {
    pub abs_l: i128,
    pub sign: i128,
    pub sigma_upper_at_abs: i128,
    pub sigma_lower_at_abs: i128,
    pub dx_step_1: i128,
    pub dx_step_1_rem: i128,
    pub dx_sigma_code: i128,
    pub dx_sigma_rem: i128,
    pub b_sigma_code: i128,
    pub b_sigma_rem: i128,
    pub diff: i128,
}

/// Build per-cell split-arithmetic witnesses for one
/// `(kind, line, endpoint)` instance. Upper-line cells use floor
/// division (the line is a conservative lower estimate); lower-line
/// cells use ceil division. Padding cells get all-zero values.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_witnesses(
    pre: &crate::snark::SigmaTables,
    kind: ActivationKind,
    line: super::types::SshapeLineKind,
    preact_codes: &[i128],
    d_padded: &[i128],
    b_padded: &[i128],
    s_d_code: i128,
    s_b_code: i128,
    s_w_code: i128,
    s_v_code: i128,
    n_padded: usize,
) -> Result<Vec<SplitArithCellWitness>, SnarkError> {
    use super::super::sshape_helpers::{split_line_witnesses_lower, split_line_witnesses_upper};
    use super::types::SshapeLineKind;
    let n = preact_codes.len();
    let s_w_log2 = (s_w_code as u128).trailing_zeros() as i32;
    if (1i128 << s_w_log2) != s_w_code {
        return Err(SnarkError::ShapeMismatch {
            what: "sshape_endpoint: s_w not power of two",
        });
    }
    let _ = s_v_code;
    let table_upper = match kind {
        ActivationKind::Sigmoid => &pre.sigmoid_upper_fr,
        ActivationKind::Tanh => &pre.tanh_upper_fr,
        ActivationKind::ReLU => unreachable!(),
    };
    let table_lower = match kind {
        ActivationKind::Sigmoid => &pre.sigmoid_lower_fr,
        ActivationKind::Tanh => &pre.tanh_lower_fr,
        ActivationKind::ReLU => unreachable!(),
    };
    let table_len = table_upper.len();
    let table_len_i128 = table_len as i128;

    // Padding rows lookup table[0] so the LogUp witness ⊆ table.
    let s_up_at_0 = fr_to_signed_i128(table_upper[0]).ok_or(SnarkError::FieldDecodeOutOfRange {
        which: "sshape: σ_upper[0] decode",
    })?;
    let s_lo_at_0 = fr_to_signed_i128(table_lower[0]).ok_or(SnarkError::FieldDecodeOutOfRange {
        which: "sshape: σ_lower[0] decode",
    })?;

    let mut out: Vec<SplitArithCellWitness> = Vec::with_capacity(n_padded);
    for i in 0..n_padded {
        if i >= n {
            // Padding row: zero witnesses masked by the is_real MLE
            // in the combined sumcheck.
            out.push(SplitArithCellWitness {
                abs_l: 0,
                sign: 0,
                sigma_upper_at_abs: s_up_at_0,
                sigma_lower_at_abs: s_lo_at_0,
                dx_step_1: 0,
                dx_step_1_rem: 0,
                dx_sigma_code: 0,
                dx_sigma_rem: 0,
                b_sigma_code: 0,
                b_sigma_rem: 0,
                diff: 0,
            });
            continue;
        }
        let l_i = preact_codes[i];
        let abs_table_idx = preact_to_abs_table_index(l_i, Scale::from_pow2(s_w_log2), pre.s_x_log2)
            .ok_or(SnarkError::ShapeMismatch {
                what: "sshape_endpoint: abs table index conversion",
            })?;
        if abs_table_idx < 0 || abs_table_idx >= table_len_i128 {
            return Err(SnarkError::RelaxationSoundnessSshapeInvalid {
                layer_idx: 0,
                activation: match kind {
                    ActivationKind::Sigmoid => "sigmoid",
                    ActivationKind::Tanh => "tanh",
                    _ => "?",
                },
                side: "endpoint",
                detail: "sshape_endpoint: preact |l| out of half-table domain \
                         (|l|·s_x/s_w >= table_len). Domain is |l| < 128·s_w; \
                         saturation outside that requires a separate extension.",
            });
        }
        let s_up = fr_to_signed_i128(table_upper[abs_table_idx as usize]).ok_or(
            SnarkError::FieldDecodeOutOfRange {
                which: "sshape: σ_upper decode",
            },
        )?;
        let s_lo = fr_to_signed_i128(table_lower[abs_table_idx as usize]).ok_or(
            SnarkError::FieldDecodeOutOfRange {
                which: "sshape: σ_lower decode",
            },
        )?;
        let sign_bit: i128 = if l_i < 0 { 1 } else { 0 };
        let s_used = compute_sigma_used(kind, line, s_up, s_lo, sign_bit, s_v_code);

        let (
            dx_step_1_v,
            dx_step_1_rem_v,
            dx_sigma_code_v,
            dx_sigma_rem_v,
            b_sigma_code_v,
            b_sigma_rem_v,
            diff_v,
        ) = match line {
            SshapeLineKind::Upper => {
                let w = split_line_witnesses_upper(
                    d_padded[i],
                    l_i,
                    b_padded[i],
                    s_used,
                    s_d_code,
                    s_b_code,
                    s_w_code,
                    s_v_code,
                )
                .ok_or(SnarkError::ShapeMismatch {
                    what: "sshape_endpoint: split-arith witness compute (upper)",
                })?;
                (
                    w.dx_step_1,
                    w.dx_step_1_rem,
                    w.dx_sigma_code,
                    w.dx_sigma_rem,
                    w.b_sigma_code,
                    w.b_sigma_rem,
                    w.diff,
                )
            }
            SshapeLineKind::Lower => {
                let w = split_line_witnesses_lower(
                    d_padded[i],
                    l_i,
                    b_padded[i],
                    s_used,
                    s_d_code,
                    s_b_code,
                    s_w_code,
                    s_v_code,
                )
                .ok_or(SnarkError::ShapeMismatch {
                    what: "sshape_endpoint: split-arith witness compute (lower)",
                })?;
                (
                    w.dx_step_1,
                    w.dx_step_1_rem,
                    w.dx_sigma_code,
                    w.dx_sigma_rem,
                    w.b_sigma_code,
                    w.b_sigma_rem,
                    w.diff,
                )
            }
        };
        if diff_v < 0 {
            return Err(SnarkError::RelaxationSoundnessSshapeInvalid {
                layer_idx: 0,
                activation: match kind {
                    ActivationKind::Sigmoid => "sigmoid",
                    ActivationKind::Tanh => "tanh",
                    _ => "?",
                },
                side: "endpoint",
                detail: "sshape_endpoint: line does NOT bound σ_used at this \
                         endpoint (split-arith diff < 0). Cert generator should ceil/\
                         floor against the table envelope, not raw float σ.",
            });
        }
        out.push(SplitArithCellWitness {
            abs_l: abs_table_idx,
            sign: sign_bit,
            sigma_upper_at_abs: s_up,
            sigma_lower_at_abs: s_lo,
            dx_step_1: dx_step_1_v,
            dx_step_1_rem: dx_step_1_rem_v,
            dx_sigma_code: dx_sigma_code_v,
            dx_sigma_rem: dx_sigma_rem_v,
            b_sigma_code: b_sigma_code_v,
            b_sigma_rem: b_sigma_rem_v,
            diff: diff_v,
        });
    }
    Ok(out)
}
