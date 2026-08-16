//! Per-neuron witness construction for the critical-point gadget:
//! pick a critical-point index `z`, look up σ envelopes at
//! `z, z − δ, z + δ`, and build the FD slacks, line/σ split-arith
//! intermediates, and gated-gap bookkeeping.

use ark_bn254::Fr;

use crate::crown::network::ActivationKind;
use crate::quantization::scale::Scale;
use crate::snark_primitives::finite_field::fr_to_signed_i128;

use super::super::sshape_endpoint::SshapeLineKind;
use super::super::sshape_helpers::fd_slope_match_slacks;
use crate::snark::errors::SnarkError;

/// Scale precondition for the FD slope-match identity. After the
/// slacks were decomposed into two `2^GADGET_RANGE_BITS` halves
/// the only remaining requirement is that `s_d` fits in one chunk.
pub fn fd_scale_precondition_holds(s_d: Scale, s_w: Scale, range_bits: usize) -> bool {
    let s_d_e = match s_d.pow2_exponent() {
        Ok(e) => e,
        Err(_) => return false,
    };
    let _ = s_w;
    if s_d_e < 0 {
        return false;
    }
    let bits = range_bits as i32;
    s_d_e <= bits
}

/// Per-neuron witness bundle (integer codes, pre-Fr-lift).
pub(crate) struct NeuronWitness {
    pub z: i128,
    pub sigma_lo_z: i128,
    pub sigma_up_z: i128,
    pub sigma_lo_zmd: i128,
    pub sigma_up_zmd: i128,
    pub sigma_lo_zpd: i128,
    pub sigma_up_zpd: i128,
    pub slack_fd1: i128,
    pub slack_fd2: i128,
    /// `slack_fdN = slack_fdN_high · 2¹⁹ + slack_fdN_low` with both
    /// halves in `[0, 2¹⁹)`.
    pub slack_fd1_high: i128,
    pub slack_fd1_low: i128,
    pub slack_fd2_high: i128,
    pub slack_fd2_low: i128,
    /// `delta = u − z` (Upper) / `−z − l` (Lower).
    pub factor_a: i128,
    /// Line-σ gap at scale `s_v`. Upper: `U(z)·s_v − σ_up(z)`.
    /// Lower: `σ_lo(−z) − L(−z)·s_v`.
    pub factor_b: i128,
    /// Split-arith intermediates feeding `factor_b`. `dz` always uses
    /// floor (`d, z ≥ 0`); `b_sigma` flips floor/ceil per direction so
    /// the line value is a conservative lower (upper) estimate for the
    /// upper (lower) direction.
    pub dz_step_1: i128,
    pub dz_step_1_rem: i128,
    pub dz_sigma_code: i128,
    pub dz_sigma_rem: i128,
    pub b_sigma_code: i128,
    pub b_sigma_rem: i128,
    /// `is_active ∈ {0, 1}`, 1 iff `d != 0`. Gates the FD identities so
    /// they are not enforced for degenerate `d = 0` cells (no finite
    /// critical point exists when the relaxation slope is zero).
    pub is_active: i128,
    /// `inside_bit ∈ {0, 1}`, 1 iff `delta ≥ 0` (i.e., the chosen
    /// critical point `z` is inside `[l, u]`).
    pub inside_bit: i128,
    /// `slack_pos = inside·delta + (1−inside)·(−delta − 1)`. Combined
    /// with booleanity and a chunked range check, this binds
    /// `inside_bit` to the sign of `delta`.
    pub slack_pos: i128,
    pub slack_pos_high: i128,
    pub slack_pos_low: i128,
    /// `gated_gap = inside_bit · factor_b`. Range-checked `≥ 0`: the
    /// relaxation soundness condition `factor_b ≥ 0` is enforced only
    /// when the critical point is inside `[l, u]`.
    pub gated_gap: i128,
    pub gated_gap_high: i128,
    pub gated_gap_low: i128,
}

/// Find a critical-point index `z` for slope `m = d / s_d`: float
/// bisection on `σ'(z) = m`, quantize, then search the rounded
/// neighbourhood for a candidate that passes the strict integer FD
/// check. Returns `None` if no nearby candidate passes (the caller
/// fails closed).
fn find_z_witness(
    kind: ActivationKind,
    sigma_lower: &[i128],
    sigma_upper: &[i128],
    d_int: i128,
    s_d: i128,
    s_x: i128,
    s_v: i128,
) -> Option<i128> {
    use super::super::sshape_helpers::{
        bisect_critical_point, find_strict_z_code, NEARBY_Z_SEARCH_RADIUS,
    };
    let table_len = sigma_lower.len();
    if table_len < 3 {
        return None;
    }
    if d_int < 0 {
        return None;
    }
    // Sigmoid and tanh both saturate well before the table's
    // half-domain bound of 128 real units; 32 is comfortably past
    // saturation for both.
    let z_max_float: f64 = 32.0;
    let m: f64 = (d_int as f64) / (s_d as f64);
    let z_float = bisect_critical_point(kind, m, z_max_float)?;
    let s_x_real: f64 = s_x as f64;
    let z_initial_code: i128 = (z_float * s_x_real).round() as i128;
    find_strict_z_code(
        sigma_lower,
        sigma_upper,
        d_int,
        s_d,
        s_x,
        s_v,
        z_initial_code,
        NEARBY_Z_SEARCH_RADIUS,
    )
}

/// Compute all per-neuron witnesses for one line direction.
///
/// Upper: `factor_a = u − z`, `factor_b = U(z) − σ_up(z)`.
/// Lower: `factor_a = (−z) − l`, `factor_b = σ_lo(−z) − L(−z)`. The
/// lower direction recovers `σ_lo(−z)` from `σ_up(z)` via the σ
/// symmetries so only the upper-half σ commit is fresh.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_witnesses(
    pre: &crate::snark::SigmaTables,
    kind: ActivationKind,
    line: SshapeLineKind,
    layer_idx: usize,
    preact_lower_codes: &[i128],
    preact_upper_codes: &[i128],
    d_padded: &[i128],
    b_padded: &[i128],
    s_d_code: i128,
    s_b_code: i128,
    s_w_code: i128,
    s_x_code: i128,
    s_v_code: i128,
    n: usize,
    n_padded: usize,
    range_bits: usize,
) -> Result<Vec<NeuronWitness>, SnarkError> {
    let table_upper_fr = match kind {
        ActivationKind::Sigmoid => &pre.sigmoid_upper_fr,
        ActivationKind::Tanh => &pre.tanh_upper_fr,
        ActivationKind::ReLU => unreachable!(),
    };
    let table_lower_fr = match kind {
        ActivationKind::Sigmoid => &pre.sigmoid_lower_fr,
        ActivationKind::Tanh => &pre.tanh_lower_fr,
        ActivationKind::ReLU => unreachable!(),
    };
    let table_len = table_upper_fr.len();
    let mut sigma_lower_int: Vec<i128> = Vec::with_capacity(table_len);
    let mut sigma_upper_int: Vec<i128> = Vec::with_capacity(table_len);
    for i in 0..table_len {
        sigma_lower_int.push(fr_to_signed_i128(table_lower_fr[i]).ok_or(
            SnarkError::FieldDecodeOutOfRange {
                which: "sshape3c: σ_lower decode",
            },
        )?);
        sigma_upper_int.push(fr_to_signed_i128(table_upper_fr[i]).ok_or(
            SnarkError::FieldDecodeOutOfRange {
                which: "sshape3c: σ_upper decode",
            },
        )?);
    }

    let bound = 1i128 << range_bits;
    let mut out: Vec<NeuronWitness> = Vec::with_capacity(n_padded);
    for j in 0..n {
        let d_j = d_padded[j];
        let b_j = b_padded[j];
        let l_j = preact_lower_codes[j];
        let u_j = preact_upper_codes[j];
        // d = 0 is degenerate (e.g., sigmoid neurons with u == l): no
        // finite critical point exists for m = 0. Mark the cell
        // inactive so the FD identities are gated out; Phase 3b's
        // endpoint check still bounds the constant line.
        let is_active: i128 = if d_j != 0 { 1 } else { 0 };
        let z = if is_active == 1 {
            match find_z_witness(
                kind,
                &sigma_lower_int,
                &sigma_upper_int,
                d_j,
                s_d_code,
                s_x_code,
                s_v_code,
            ) {
                Some(z) => z,
                None => {
                    return Err(SnarkError::RelaxationSoundnessSshapeInvalid {
                        layer_idx,
                        activation: match kind {
                            ActivationKind::Sigmoid => "sigmoid",
                            ActivationKind::Tanh => "tanh",
                            _ => "?",
                        },
                        side: "critical-point",
                        detail: "sshape3c: no critical-point witness z found in table for slope m \
                                 — slope outside activation range or scale mismatch",
                    });
                }
            }
        } else {
            // d = 0 path: any valid index. σ-LogUp / factor_b still
            // evaluate at this index but their identities are
            // satisfied trivially.
            1
        };
        let zi = z as usize;
        let sigma_lo_z = sigma_lower_int[zi];
        let sigma_up_z = sigma_upper_int[zi];
        let sigma_lo_zmd = sigma_lower_int[zi - 1];
        let sigma_up_zmd = sigma_upper_int[zi - 1];
        let sigma_lo_zpd = sigma_lower_int[zi + 1];
        let sigma_up_zpd = sigma_upper_int[zi + 1];
        // Inactive cells use dummy zero slacks; the chunked range
        // check accepts them since `0 = 0·M + 0`.
        let (slack_fd1, slack_fd2) = if is_active == 1 {
            let r = fd_slope_match_slacks(
                sigma_lo_z,
                sigma_up_zmd,
                sigma_up_zpd,
                d_j,
                s_d_code,
                s_x_code,
                s_v_code,
            )
            .ok_or(SnarkError::RelaxationSoundnessSshapeInvalid {
                layer_idx,
                activation: match kind {
                    ActivationKind::Sigmoid => "sigmoid",
                    ActivationKind::Tanh => "tanh",
                    _ => "?",
                },
                side: "critical-point",
                detail: "sshape3c: FD slacks negative at chosen z (cert / scale issue)",
            })?;
            (r.slack_fd1, r.slack_fd2)
        } else {
            (0i128, 0i128)
        };
        let chunk_modulus: i128 = bound;
        let chunk_max: i128 =
            chunk_modulus
                .checked_mul(chunk_modulus)
                .ok_or(SnarkError::ShapeMismatch {
                    what: "sshape3c: chunk_max overflow",
                })?;
        if slack_fd1 < 0 || slack_fd1 >= chunk_max || slack_fd2 < 0 || slack_fd2 >= chunk_max {
            return Err(SnarkError::RelaxationSoundnessSshapeInvalid {
                layer_idx,
                activation: "sshape3c",
                side: "critical-point",
                detail: "sshape3c: FD slack out of two-chunk range (≥ 2^(2·GADGET_RANGE_BITS))",
            });
        }
        let slack_fd1_high = slack_fd1 / chunk_modulus;
        let slack_fd1_low = slack_fd1 - slack_fd1_high * chunk_modulus;
        let slack_fd2_high = slack_fd2 / chunk_modulus;
        let slack_fd2_low = slack_fd2 - slack_fd2_high * chunk_modulus;
        // Convert z (table-index, scale s_x) to preact scale s_w via a
        // right shift. Requires s_x ≥ s_w.
        let s_x_log2 = s_x_code.trailing_zeros() as i32;
        let s_w_log2 = s_w_code.trailing_zeros() as i32;
        if s_x_log2 < s_w_log2 {
            return Err(SnarkError::Reserved {
                what: "sshape3c: s_w > s_x not yet supported",
            });
        }
        let z_at_sw = z >> (s_x_log2 - s_w_log2);
        // Split-arith chain: line value at scale s_v via three
        // floor/ceil divisions, then take the signed difference with
        // σ_used.
        use super::super::sshape_helpers::{ceil_div, floor_div};
        let dz = d_j.checked_mul(z_at_sw).ok_or(SnarkError::ShapeMismatch {
            what: "sshape3c: d·z overflow",
        })?;
        let dz_step_1 = floor_div(dz, s_d_code).ok_or(SnarkError::ShapeMismatch {
            what: "sshape3c: floor_div d·z / s_d",
        })?;
        let dz_step_1_rem = dz
            .checked_sub(
                dz_step_1
                    .checked_mul(s_d_code)
                    .ok_or(SnarkError::ShapeMismatch {
                        what: "sshape3c: dz_step_1·s_d overflow",
                    })?,
            )
            .ok_or(SnarkError::ShapeMismatch {
                what: "sshape3c: dz_step_1_rem subtraction",
            })?;
        let dz_step_1_sv = dz_step_1
            .checked_mul(s_v_code)
            .ok_or(SnarkError::ShapeMismatch {
                what: "sshape3c: dz_step_1·s_v overflow",
            })?;
        let dz_sigma_code = floor_div(dz_step_1_sv, s_w_code).ok_or(SnarkError::ShapeMismatch {
            what: "sshape3c: floor_div dz·s_v / s_w",
        })?;
        let dz_sigma_rem =
            dz_step_1_sv
                .checked_sub(dz_sigma_code.checked_mul(s_w_code).ok_or(
                    SnarkError::ShapeMismatch {
                        what: "sshape3c: dz_sigma·s_w overflow",
                    },
                )?)
                .ok_or(SnarkError::ShapeMismatch {
                    what: "sshape3c: dz_sigma_rem subtraction",
                })?;
        let b_sv = b_j.checked_mul(s_v_code).ok_or(SnarkError::ShapeMismatch {
            what: "sshape3c: b·s_v overflow",
        })?;
        let (b_sigma_code, b_sigma_rem) = match line {
            SshapeLineKind::Upper => {
                let q = floor_div(b_sv, s_b_code).ok_or(SnarkError::ShapeMismatch {
                    what: "sshape3c: floor_div b·s_v / s_b",
                })?;
                let rem = b_sv
                    .checked_sub(q.checked_mul(s_b_code).ok_or(SnarkError::ShapeMismatch {
                        what: "sshape3c: b_sigma·s_b overflow",
                    })?)
                    .ok_or(SnarkError::ShapeMismatch {
                        what: "sshape3c: b_sigma_rem subtraction (upper)",
                    })?;
                (q, rem)
            }
            SshapeLineKind::Lower => {
                let q = ceil_div(b_sv, s_b_code).ok_or(SnarkError::ShapeMismatch {
                    what: "sshape3c: ceil_div b·s_v / s_b",
                })?;
                let rem = q
                    .checked_mul(s_b_code)
                    .and_then(|v| v.checked_sub(b_sv))
                    .ok_or(SnarkError::ShapeMismatch {
                        what: "sshape3c: b_sigma_rem subtraction (lower)",
                    })?;
                (q, rem)
            }
        };
        let (factor_a, factor_b) = match line {
            SshapeLineKind::Upper => {
                let fa = u_j.checked_sub(z_at_sw).ok_or(SnarkError::ShapeMismatch {
                    what: "sshape3c: factor_a overflow",
                })?;
                let line_at_sv =
                    dz_sigma_code
                        .checked_add(b_sigma_code)
                        .ok_or(SnarkError::ShapeMismatch {
                            what: "sshape3c: line_at_sv overflow (upper)",
                        })?;
                let fb = line_at_sv
                    .checked_sub(sigma_up_z)
                    .ok_or(SnarkError::ShapeMismatch {
                        what: "sshape3c: factor_b overflow (upper)",
                    })?;
                (fa, fb)
            }
            SshapeLineKind::Lower => {
                let neg_z = z_at_sw.checked_neg().ok_or(SnarkError::ShapeMismatch {
                    what: "sshape3c: −z overflow",
                })?;
                let fa = neg_z.checked_sub(l_j).ok_or(SnarkError::ShapeMismatch {
                    what: "sshape3c: factor_a overflow",
                })?;
                // σ_lo(−z) via the σ symmetry.
                let sigma_lo_neg_z = match kind {
                    ActivationKind::Sigmoid => {
                        s_v_code
                            .checked_sub(sigma_up_z)
                            .ok_or(SnarkError::ShapeMismatch {
                                what: "sshape3c: σ_lo(−z) overflow",
                            })?
                    }
                    ActivationKind::Tanh => {
                        sigma_up_z.checked_neg().ok_or(SnarkError::ShapeMismatch {
                            what: "sshape3c: σ_lo(−z) overflow",
                        })?
                    }
                    ActivationKind::ReLU => unreachable!(),
                };
                // L(−z)·s_v upper estimate = −dz_sigma_code + b_sigma_code, so
                // factor_b = σ_lo(−z) + dz_sigma_code − b_sigma_code.
                let fb = sigma_lo_neg_z
                    .checked_add(dz_sigma_code)
                    .and_then(|v| v.checked_sub(b_sigma_code))
                    .ok_or(SnarkError::ShapeMismatch {
                        what: "sshape3c: factor_b overflow (lower)",
                    })?;
                (fa, fb)
            }
        };
        let inside_bit: i128 = if factor_a >= 0 { 1 } else { 0 };
        let slack_pos = if inside_bit == 1 {
            factor_a
        } else {
            // factor_a < 0 ⇒ -factor_a - 1 ≥ 0
            (-factor_a)
                .checked_sub(1)
                .ok_or(SnarkError::ShapeMismatch {
                    what: "sshape3c: -factor_a - 1 overflow",
                })?
        };
        if slack_pos < 0 {
            return Err(SnarkError::RelaxationSoundnessSshapeInvalid {
                layer_idx,
                activation: "sshape3c",
                side: "critical-point",
                detail: "sshape3c: slack_pos < 0 — internal witness construction bug",
            });
        }
        let gated_gap = factor_b
            .checked_mul(inside_bit)
            .ok_or(SnarkError::ShapeMismatch {
                what: "sshape3c: gated_gap (inside·factor_b) overflow",
            })?;
        if gated_gap < 0 {
            // The line falls below σ at an interior critical point —
            // the relaxation is invalid.
            return Err(SnarkError::RelaxationSoundnessSshapeInvalid {
                layer_idx,
                activation: "sshape3c",
                side: "critical-point",
                detail: "sshape3c: gated_gap < 0 — line below σ at interior critical point (invalid relaxation)",
            });
        }
        let chunk_modulus = bound;
        let slack_pos_high = slack_pos / chunk_modulus;
        let slack_pos_low = slack_pos - slack_pos_high * chunk_modulus;
        if slack_pos_high < 0 || slack_pos_high >= bound {
            return Err(SnarkError::Reserved {
                what: "sshape3c: slack_pos_high out of range — slack_pos exceeds 2^(2·GADGET_RANGE_BITS)",
            });
        }
        debug_assert!(slack_pos_low >= 0 && slack_pos_low < chunk_modulus);
        let gated_gap_high = gated_gap / chunk_modulus;
        let gated_gap_low = gated_gap - gated_gap_high * chunk_modulus;
        if gated_gap_high < 0 || gated_gap_high >= bound {
            return Err(SnarkError::Reserved {
                what: "sshape3c: gated_gap_high out of range — gated_gap exceeds 2^(2·GADGET_RANGE_BITS)",
            });
        }
        debug_assert!(gated_gap_low >= 0 && gated_gap_low < chunk_modulus);
        out.push(NeuronWitness {
            z,
            sigma_lo_z,
            sigma_up_z,
            sigma_lo_zmd,
            sigma_up_zmd,
            sigma_lo_zpd,
            sigma_up_zpd,
            slack_fd1,
            slack_fd2,
            slack_fd1_high,
            slack_fd1_low,
            slack_fd2_high,
            slack_fd2_low,
            factor_a,
            factor_b,
            dz_step_1,
            dz_step_1_rem,
            dz_sigma_code,
            dz_sigma_rem,
            b_sigma_code,
            b_sigma_rem,
            is_active,
            inside_bit,
            slack_pos,
            slack_pos_high,
            slack_pos_low,
            gated_gap,
            gated_gap_high,
            gated_gap_low,
        });
    }
    // Pad with neutral witnesses.
    let pad_z = 1i128;
    let pad_sigma_lo = sigma_lower_int[1];
    let pad_sigma_up = sigma_upper_int[1];
    let pad_sigma_lo_zmd = sigma_lower_int[0];
    let pad_sigma_up_zmd = sigma_upper_int[0];
    let pad_sigma_lo_zpd = sigma_lower_int[2];
    let pad_sigma_up_zpd = sigma_upper_int[2];
    while out.len() < n_padded {
        out.push(NeuronWitness {
            z: pad_z,
            sigma_lo_z: pad_sigma_lo,
            sigma_up_z: pad_sigma_up,
            sigma_lo_zmd: pad_sigma_lo_zmd,
            sigma_up_zmd: pad_sigma_up_zmd,
            sigma_lo_zpd: pad_sigma_lo_zpd,
            sigma_up_zpd: pad_sigma_up_zpd,
            slack_fd1: 0,
            slack_fd2: 0,
            slack_fd1_high: 0,
            slack_fd1_low: 0,
            slack_fd2_high: 0,
            slack_fd2_low: 0,
            factor_a: 0,
            factor_b: 0,
            dz_step_1: 0,
            dz_step_1_rem: 0,
            dz_sigma_code: 0,
            dz_sigma_rem: 0,
            b_sigma_code: 0,
            b_sigma_rem: 0,
            // Padding rows are masked by is_real in the sumcheck;
            // the values here satisfy every identity trivially.
            is_active: 0,
            inside_bit: 1,
            slack_pos: 0,
            slack_pos_high: 0,
            slack_pos_low: 0,
            gated_gap: 0,
            gated_gap_high: 0,
            gated_gap_low: 0,
        });
    }
    Ok(out)
}

/// Lift a vector of `i128` codes to Fr.
pub(crate) fn lift(codes: &[i128]) -> Vec<Fr> {
    codes
        .iter()
        .map(|&v| crate::snark_primitives::finite_field::signed_lift_to_fr(v))
        .collect()
}
