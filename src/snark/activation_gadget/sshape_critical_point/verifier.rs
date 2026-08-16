//! Verifier for the critical-point gadget. Mirrors the prover's
//! transcript order and replays every check at FS-derived points.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;

use crate::crown::network::ActivationKind;
use crate::quantization::scale::Scale;
use crate::snark_primitives::finite_field::signed_lift_to_fr;
use crate::snark_primitives::logup_gkr::verify_circuit_with_top as logup_verify_circuit_with_top;
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use super::super::relu_upper_endpoint::{squeeze_round_challenge_4, verify_pos_range};
use super::super::sshape_endpoint::SshapeLineKind;
use crate::snark::commitment::multilinear_extensions::{build_eq_table, eval_multilinear_full};
use crate::snark::commitment::pcs_helpers::{
    hyrax_verify_at, hyrax_verify_batched_at, BatchVerifySpec,
};
use crate::snark::errors::SnarkError;
use crate::snark::output_bound::absorb_commitment;
use crate::snark::params::SnarkParams;

use super::types::SshapeCriticalPointProof;

/// Verify a `SshapeCriticalPointProof`. The verifier reads the
/// preact codes only through Hyrax opens at this gadget's `r_final`;
/// the opened evals feed the `factor_a` identity.
#[allow(clippy::too_many_arguments)]
pub fn verify_sshape_critical_point(
    proof: &SshapeCriticalPointProof,
    expected_layer_idx: usize,
    kind: ActivationKind,
    line: SshapeLineKind,
    n_real_neurons: usize,
    preact_lower_commit: &<HyraxBn254 as MlPcs>::Commitment,
    preact_upper_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_line_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_line_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s_d: Scale,
    s_b: Scale,
    s_w: Scale,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    let kind_tag = match kind {
        ActivationKind::Sigmoid => 0u8,
        ActivationKind::Tanh => 1u8,
        ActivationKind::ReLU => {
            return Err(SnarkError::ShapeMismatch {
                what: "sshape3c verify called for ReLU",
            });
        }
    };
    if proof.kind_tag != kind_tag {
        return Err(SnarkError::ArchitectureMismatch {
            what: "sshape3c: kind_tag mismatch",
        });
    }
    if proof.line_tag != line.tag() {
        return Err(SnarkError::ArchitectureMismatch {
            what: "sshape3c: line_tag mismatch",
        });
    }
    if proof.layer_idx != expected_layer_idx {
        return Err(SnarkError::ArchitectureMismatch {
            what: "sshape3c: layer_idx mismatch",
        });
    }

    let s_d_e = s_d.pow2_exponent().map_err(|_| SnarkError::ShapeMismatch {
        what: "sshape3c: s_d not pow2",
    })?;
    let s_b_e = s_b.pow2_exponent().map_err(|_| SnarkError::ShapeMismatch {
        what: "sshape3c: s_b not pow2",
    })?;
    let s_w_e = s_w.pow2_exponent().map_err(|_| SnarkError::ShapeMismatch {
        what: "sshape3c: s_w not pow2",
    })?;
    if s_w_e != params.sigma_x_scale_log2 {
        return Err(SnarkError::Reserved {
            what: "sshape3c verify: requires s_w = s_x = 2^sigma_x_scale_log2",
        });
    }
    let bits = params.gadget_range_bits as i32;
    if s_d_e > bits || s_b_e > bits || s_w_e > bits {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape3c verify: per-scale precondition failed",
        });
    }
    let s_d_code = 1i128 << s_d_e;
    let s_b_code = 1i128 << s_b_e;
    let s_w_code = 1i128 << s_w_e;
    let s_x_code = 1i128 << params.sigma_x_scale_log2;
    let s_v_code = 1i128 << params.sigma_v_scale_log2;

    let n = n_real_neurons;
    if n == 0 {
        return Err(SnarkError::Reserved {
            what: "sshape3c verify: requires n ≥ 1",
        });
    }
    if proof.n_real != n {
        return Err(SnarkError::ArchitectureMismatch {
            what: "sshape3c: n_real mismatch (verifier — proof's n_real differs from \
                   the public n_real_neurons argument)",
        });
    }
    let n_vars = crate::snark::commitment::commit::native_vector_n_vars(n);
    if n_vars != proof.n_vars {
        return Err(SnarkError::ArchitectureMismatch {
            what: "sshape3c: n_vars mismatch",
        });
    }
    let n_padded = 1usize << n_vars;
    if n > n_padded {
        return Err(SnarkError::ShapeMismatch {
            what: "sshape3c verify: n > n_padded",
        });
    }

    sponge.absorb(&(expected_layer_idx as u64));
    sponge.absorb(&kind_tag);
    sponge.absorb(&proof.line_tag);
    sponge.absorb(&(n_vars as u64));
    sponge.absorb(&(n as u64));
    absorb_commitment(sponge, d_line_commit);
    absorb_commitment(sponge, b_line_commit);
    absorb_commitment(sponge, preact_lower_commit);
    absorb_commitment(sponge, preact_upper_commit);

    absorb_commitment(sponge, &proof.z_commit);
    absorb_commitment(sponge, &proof.sigma_lo_z_commit);
    absorb_commitment(sponge, &proof.sigma_up_z_commit);
    absorb_commitment(sponge, &proof.sigma_lo_zmd_commit);
    absorb_commitment(sponge, &proof.sigma_up_zmd_commit);
    absorb_commitment(sponge, &proof.sigma_lo_zpd_commit);
    absorb_commitment(sponge, &proof.sigma_up_zpd_commit);
    absorb_commitment(sponge, &proof.slack_fd1_commit);
    absorb_commitment(sponge, &proof.slack_fd2_commit);
    absorb_commitment(sponge, &proof.slack_fd1_high_commit);
    absorb_commitment(sponge, &proof.slack_fd1_low_commit);
    absorb_commitment(sponge, &proof.slack_fd2_high_commit);
    absorb_commitment(sponge, &proof.slack_fd2_low_commit);
    absorb_commitment(sponge, &proof.factor_a_commit);
    absorb_commitment(sponge, &proof.factor_b_commit);
    absorb_commitment(sponge, &proof.dz_step_1_commit);
    absorb_commitment(sponge, &proof.dz_step_1_rem_commit);
    absorb_commitment(sponge, &proof.dz_sigma_code_commit);
    absorb_commitment(sponge, &proof.dz_sigma_rem_commit);
    absorb_commitment(sponge, &proof.b_sigma_code_commit);
    absorb_commitment(sponge, &proof.b_sigma_rem_commit);
    absorb_commitment(sponge, &proof.is_active_commit);
    absorb_commitment(sponge, &proof.inside_commit);
    absorb_commitment(sponge, &proof.slack_pos_commit);
    absorb_commitment(sponge, &proof.slack_pos_high_commit);
    absorb_commitment(sponge, &proof.slack_pos_low_commit);
    absorb_commitment(sponge, &proof.gated_gap_commit);
    absorb_commitment(sponge, &proof.gated_gap_high_commit);
    absorb_commitment(sponge, &proof.gated_gap_low_commit);

    verify_pos_range(&proof.z_range, n_vars, &proof.z_commit, params, sponge)?;
    verify_pos_range(
        &proof.slack_fd1_high_range,
        n_vars,
        &proof.slack_fd1_high_commit,
        params,
        sponge,
    )?;
    verify_pos_range(
        &proof.slack_fd1_low_range,
        n_vars,
        &proof.slack_fd1_low_commit,
        params,
        sponge,
    )?;
    verify_pos_range(
        &proof.slack_fd2_high_range,
        n_vars,
        &proof.slack_fd2_high_commit,
        params,
        sponge,
    )?;
    verify_pos_range(
        &proof.slack_fd2_low_range,
        n_vars,
        &proof.slack_fd2_low_commit,
        params,
        sponge,
    )?;
    verify_pos_range(
        &proof.slack_pos_high_range,
        n_vars,
        &proof.slack_pos_high_commit,
        params,
        sponge,
    )?;
    verify_pos_range(
        &proof.slack_pos_low_range,
        n_vars,
        &proof.slack_pos_low_commit,
        params,
        sponge,
    )?;
    verify_pos_range(
        &proof.gated_gap_high_range,
        n_vars,
        &proof.gated_gap_high_commit,
        params,
        sponge,
    )?;
    verify_pos_range(
        &proof.gated_gap_low_range,
        n_vars,
        &proof.gated_gap_low_commit,
        params,
        sponge,
    )?;
    verify_pos_range(
        &proof.dz_step_1_rem_range,
        n_vars,
        &proof.dz_step_1_rem_commit,
        params,
        sponge,
    )?;
    verify_pos_range(
        &proof.dz_sigma_rem_range,
        n_vars,
        &proof.dz_sigma_rem_commit,
        params,
        sponge,
    )?;
    verify_pos_range(
        &proof.b_sigma_rem_range,
        n_vars,
        &proof.b_sigma_rem_commit,
        params,
        sponge,
    )?;

    sponge.absorb(&((3 * n_padded) as u64));
    let envelope_combine_alpha_1 = sponge.squeeze_field_elements::<Fr>(1)[0];
    let envelope_combine_alpha_2 = sponge.squeeze_field_elements::<Fr>(1)[0];
    if envelope_combine_alpha_1 != proof.envelope_combine_alpha_1
        || envelope_combine_alpha_2 != proof.envelope_combine_alpha_2
    {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape3c: envelope alpha mismatch",
        });
    }
    let table_upper = match kind {
        ActivationKind::Sigmoid => &params.preprocessed.sigma.sigmoid_upper_fr,
        ActivationKind::Tanh => &params.preprocessed.sigma.tanh_upper_fr,
        ActivationKind::ReLU => unreachable!(),
    };
    let table_lower = match kind {
        ActivationKind::Sigmoid => &params.preprocessed.sigma.sigmoid_lower_fr,
        ActivationKind::Tanh => &params.preprocessed.sigma.tanh_lower_fr,
        ActivationKind::ReLU => unreachable!(),
    };
    let table_len = table_upper.len();
    if proof.envelope_table_len != table_len {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape3c: envelope_table_len mismatch",
        });
    }
    let envelope_witness_expected_len = (3 * n_padded).next_power_of_two().max(2);
    if proof.envelope_witness_len != envelope_witness_expected_len {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape3c: envelope_witness_len mismatch",
        });
    }
    absorb_commitment(sponge, &proof.envelope_mult_commit);

    sponge.absorb(&(proof.envelope_witness_len as u64));
    sponge.absorb(&(table_len as u64));
    let envelope_logup_beta = sponge.squeeze_field_elements::<Fr>(1)[0];
    if envelope_logup_beta != proof.envelope_logup_beta {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape3c: envelope beta mismatch",
        });
    }
    sponge.absorb(&envelope_combine_alpha_1);
    sponge.absorb(&envelope_combine_alpha_2);
    sponge.absorb(&envelope_logup_beta);

    let lookup_n_vars = proof.envelope_lookup_proof.layers.len();
    let table_n_vars = proof.envelope_table_proof.layers.len();
    let lookup_top_num_eval = proof.envelope_lookup_top[0] * proof.envelope_lookup_top[3]
        + proof.envelope_lookup_top[1] * proof.envelope_lookup_top[2];
    let table_top_num_eval = proof.envelope_table_top[0] * proof.envelope_table_top[3]
        + proof.envelope_table_top[1] * proof.envelope_table_top[2];
    let lookup_top_denom_eval = proof.envelope_lookup_top[2] * proof.envelope_lookup_top[3];
    let table_top_denom_eval = proof.envelope_table_top[2] * proof.envelope_table_top[3];
    logup_verify_circuit_with_top(
        &proof.envelope_lookup_proof,
        lookup_n_vars,
        proof.envelope_lookup_top,
        lookup_top_num_eval,
        sponge,
    )
    .map_err(SnarkError::LogUp)?;
    logup_verify_circuit_with_top(
        &proof.envelope_table_proof,
        table_n_vars,
        proof.envelope_table_top,
        table_top_num_eval,
        sponge,
    )
    .map_err(SnarkError::LogUp)?;
    use ark_ff::Zero;
    let cancel =
        lookup_top_num_eval * table_top_denom_eval + lookup_top_denom_eval * table_top_num_eval;
    if !<Fr as Zero>::is_zero(&cancel) {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape3c: top-fraction cancellation",
        });
    }

    let table_bottom_pt = proof.envelope_table_proof.bottom_point.clone();
    let combined_table_eval = envelope_combine_alpha_1
        * crate::snark::commitment::table_mle::identity_mle_eval(&table_bottom_pt)
        + envelope_combine_alpha_2 * eval_multilinear_full(table_upper, &table_bottom_pt)
        + eval_multilinear_full(table_lower, &table_bottom_pt);
    let expected_table_bottom_denom = combined_table_eval - proof.envelope_logup_beta;
    if proof.envelope_table_proof.bottom_denom != expected_table_bottom_denom {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape3c: envelope table bottom_denom mismatch",
        });
    }

    hyrax_verify_at(
        &params.verifier_key,
        &proof.envelope_mult_commit,
        &table_bottom_pt,
        proof.envelope_table_proof.bottom_num,
        &proof.envelope_mult_open,
        proof.envelope_mult_n_vars,
        sponge,
    )?;

    // LogUp witness-side single-point bind: open the 7 σ commits at
    // the cell index half of bottom_point and reconstruct W(bp) from
    // the 4-row eq selector.
    let bp_full = proof.envelope_lookup_proof.bottom_point.clone();
    if bp_full.len() < n_vars + 2 {
        return Err(SnarkError::ShapeMismatch {
            what: "sshape3c verify: LogUp bottom_point too short for row decomposition",
        });
    }
    let bp_high = &bp_full[..n_vars];
    let bp_row_high = bp_full[n_vars];
    let bp_row_low = bp_full[n_vars + 1];
    let envelope_bp_specs = [
        BatchVerifySpec {
            commitment: &proof.z_commit,
            commit_n_vars: n_vars,
            value: proof.envelope_z_eval,
        },
        BatchVerifySpec {
            commitment: &proof.sigma_lo_z_commit,
            commit_n_vars: n_vars,
            value: proof.envelope_sigma_lo_z_eval,
        },
        BatchVerifySpec {
            commitment: &proof.sigma_up_z_commit,
            commit_n_vars: n_vars,
            value: proof.envelope_sigma_up_z_eval,
        },
        BatchVerifySpec {
            commitment: &proof.sigma_lo_zmd_commit,
            commit_n_vars: n_vars,
            value: proof.envelope_sigma_lo_zmd_eval,
        },
        BatchVerifySpec {
            commitment: &proof.sigma_up_zmd_commit,
            commit_n_vars: n_vars,
            value: proof.envelope_sigma_up_zmd_eval,
        },
        BatchVerifySpec {
            commitment: &proof.sigma_lo_zpd_commit,
            commit_n_vars: n_vars,
            value: proof.envelope_sigma_lo_zpd_eval,
        },
        BatchVerifySpec {
            commitment: &proof.sigma_up_zpd_commit,
            commit_n_vars: n_vars,
            value: proof.envelope_sigma_up_zpd_eval,
        },
    ];
    let envelope_bp_ok = hyrax_verify_batched_at(
        &params.verifier_key,
        &envelope_bp_specs,
        bp_high,
        &proof.envelope_witness_batched_open,
        sponge,
    )?;
    if !envelope_bp_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "sshape3c::envelope_witness_batched_open",
        });
    }
    let one = Fr::from(1u64);
    let e_row0 = (one - bp_row_high) * (one - bp_row_low);
    let e_row1 = (one - bp_row_high) * bp_row_low;
    let e_row2 = bp_row_high * (one - bp_row_low);
    let e_row3 = bp_row_high * bp_row_low;
    let s_up_pad =
        crate::snark_primitives::finite_field::fr_to_signed_i128(table_upper[0]).unwrap_or(0);
    let s_lo_pad =
        crate::snark_primitives::finite_field::fr_to_signed_i128(table_lower[0]).unwrap_or(0);
    let s_up_pad_fr = signed_lift_to_fr(s_up_pad);
    let s_lo_pad_fr = signed_lift_to_fr(s_lo_pad);
    let w_row0 = envelope_combine_alpha_1 * proof.envelope_z_eval
        + envelope_combine_alpha_2 * proof.envelope_sigma_up_z_eval
        + proof.envelope_sigma_lo_z_eval;
    let w_row1 = envelope_combine_alpha_1 * (proof.envelope_z_eval - one)
        + envelope_combine_alpha_2 * proof.envelope_sigma_up_zmd_eval
        + proof.envelope_sigma_lo_zmd_eval;
    let w_row2 = envelope_combine_alpha_1 * (proof.envelope_z_eval + one)
        + envelope_combine_alpha_2 * proof.envelope_sigma_up_zpd_eval
        + proof.envelope_sigma_lo_zpd_eval;
    let w_row3 = envelope_combine_alpha_2 * s_up_pad_fr + s_lo_pad_fr;
    let w_at_bp = e_row0 * w_row0 + e_row1 * w_row1 + e_row2 * w_row2 + e_row3 * w_row3;
    let expected_bottom_denom_plus_beta = w_at_bp;
    if proof.envelope_lookup_proof.bottom_denom + envelope_logup_beta
        != expected_bottom_denom_plus_beta
    {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape3c: LogUp witness-side single-point bind mismatch",
        });
    }

    let combined_rho_a = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_b = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_c = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_d = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_e = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_f = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_g = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_h = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_i = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_j = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_k = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_l = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_m = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_n = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_o = sponge.squeeze_field_elements::<Fr>(1)[0];
    if combined_rho_a != proof.combined_rho_a
        || combined_rho_b != proof.combined_rho_b
        || combined_rho_c != proof.combined_rho_c
        || combined_rho_d != proof.combined_rho_d
        || combined_rho_e != proof.combined_rho_e
        || combined_rho_f != proof.combined_rho_f
        || combined_rho_g != proof.combined_rho_g
        || combined_rho_h != proof.combined_rho_h
        || combined_rho_i != proof.combined_rho_i
        || combined_rho_j != proof.combined_rho_j
        || combined_rho_k != proof.combined_rho_k
        || combined_rho_l != proof.combined_rho_l
        || combined_rho_m != proof.combined_rho_m
        || combined_rho_n != proof.combined_rho_n
        || combined_rho_o != proof.combined_rho_o
    {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape3c: rho mismatch",
        });
    }

    let r_test: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(n_vars);
    if r_test != proof.r_test {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape3c: r_test mismatch",
        });
    }

    if proof.rounds.len() != n_vars || proof.r_final.len() != n_vars {
        return Err(SnarkError::ArchitectureMismatch {
            what: "sshape3c: round count mismatch",
        });
    }
    let mut current_sum = Fr::from(0u64);
    let mut r_final_check: Vec<Fr> = Vec::with_capacity(n_vars);
    for poly in proof.rounds.iter() {
        if poly.at_zero + poly.at_one != current_sum {
            return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
                which: "sshape3c: round invariant",
            });
        }
        let r_i = squeeze_round_challenge_4(sponge, poly);
        current_sum = poly.evaluate(r_i);
        r_final_check.push(r_i);
    }
    if r_final_check != proof.r_final {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape3c: r_final mismatch",
        });
    }

    let r_specs = [
        BatchVerifySpec {
            commitment: d_line_commit,
            commit_n_vars: n_vars,
            value: proof.d_eval,
        },
        BatchVerifySpec {
            commitment: b_line_commit,
            commit_n_vars: n_vars,
            value: proof.b_eval,
        },
        BatchVerifySpec {
            commitment: &proof.z_commit,
            commit_n_vars: n_vars,
            value: proof.z_eval,
        },
        BatchVerifySpec {
            commitment: &proof.sigma_lo_z_commit,
            commit_n_vars: n_vars,
            value: proof.sigma_lo_z_eval,
        },
        BatchVerifySpec {
            commitment: &proof.sigma_up_z_commit,
            commit_n_vars: n_vars,
            value: proof.sigma_up_z_eval,
        },
        BatchVerifySpec {
            commitment: &proof.sigma_lo_zmd_commit,
            commit_n_vars: n_vars,
            value: proof.sigma_lo_zmd_eval,
        },
        BatchVerifySpec {
            commitment: &proof.sigma_up_zmd_commit,
            commit_n_vars: n_vars,
            value: proof.sigma_up_zmd_eval,
        },
        BatchVerifySpec {
            commitment: &proof.sigma_lo_zpd_commit,
            commit_n_vars: n_vars,
            value: proof.sigma_lo_zpd_eval,
        },
        BatchVerifySpec {
            commitment: &proof.sigma_up_zpd_commit,
            commit_n_vars: n_vars,
            value: proof.sigma_up_zpd_eval,
        },
        BatchVerifySpec {
            commitment: &proof.slack_fd1_commit,
            commit_n_vars: n_vars,
            value: proof.slack_fd1_eval,
        },
        BatchVerifySpec {
            commitment: &proof.slack_fd2_commit,
            commit_n_vars: n_vars,
            value: proof.slack_fd2_eval,
        },
        BatchVerifySpec {
            commitment: &proof.factor_a_commit,
            commit_n_vars: n_vars,
            value: proof.factor_a_eval,
        },
        BatchVerifySpec {
            commitment: &proof.factor_b_commit,
            commit_n_vars: n_vars,
            value: proof.factor_b_eval,
        },
        BatchVerifySpec {
            commitment: &proof.dz_step_1_commit,
            commit_n_vars: n_vars,
            value: proof.dz_step_1_eval,
        },
        BatchVerifySpec {
            commitment: &proof.dz_step_1_rem_commit,
            commit_n_vars: n_vars,
            value: proof.dz_step_1_rem_eval,
        },
        BatchVerifySpec {
            commitment: &proof.dz_sigma_code_commit,
            commit_n_vars: n_vars,
            value: proof.dz_sigma_code_eval,
        },
        BatchVerifySpec {
            commitment: &proof.dz_sigma_rem_commit,
            commit_n_vars: n_vars,
            value: proof.dz_sigma_rem_eval,
        },
        BatchVerifySpec {
            commitment: &proof.b_sigma_code_commit,
            commit_n_vars: n_vars,
            value: proof.b_sigma_code_eval,
        },
        BatchVerifySpec {
            commitment: &proof.b_sigma_rem_commit,
            commit_n_vars: n_vars,
            value: proof.b_sigma_rem_eval,
        },
        BatchVerifySpec {
            commitment: &proof.slack_fd1_high_commit,
            commit_n_vars: n_vars,
            value: proof.slack_fd1_high_eval,
        },
        BatchVerifySpec {
            commitment: &proof.slack_fd1_low_commit,
            commit_n_vars: n_vars,
            value: proof.slack_fd1_low_eval,
        },
        BatchVerifySpec {
            commitment: &proof.slack_fd2_high_commit,
            commit_n_vars: n_vars,
            value: proof.slack_fd2_high_eval,
        },
        BatchVerifySpec {
            commitment: &proof.slack_fd2_low_commit,
            commit_n_vars: n_vars,
            value: proof.slack_fd2_low_eval,
        },
        BatchVerifySpec {
            commitment: &proof.inside_commit,
            commit_n_vars: n_vars,
            value: proof.inside_eval,
        },
        BatchVerifySpec {
            commitment: &proof.slack_pos_commit,
            commit_n_vars: n_vars,
            value: proof.slack_pos_eval,
        },
        BatchVerifySpec {
            commitment: &proof.slack_pos_high_commit,
            commit_n_vars: n_vars,
            value: proof.slack_pos_high_eval,
        },
        BatchVerifySpec {
            commitment: &proof.slack_pos_low_commit,
            commit_n_vars: n_vars,
            value: proof.slack_pos_low_eval,
        },
        BatchVerifySpec {
            commitment: &proof.gated_gap_commit,
            commit_n_vars: n_vars,
            value: proof.gated_gap_eval,
        },
        BatchVerifySpec {
            commitment: &proof.gated_gap_high_commit,
            commit_n_vars: n_vars,
            value: proof.gated_gap_high_eval,
        },
        BatchVerifySpec {
            commitment: &proof.gated_gap_low_commit,
            commit_n_vars: n_vars,
            value: proof.gated_gap_low_eval,
        },
        BatchVerifySpec {
            commitment: &proof.is_active_commit,
            commit_n_vars: n_vars,
            value: proof.is_active_eval,
        },
    ];
    let batched_ok = hyrax_verify_batched_at(
        &params.verifier_key,
        &r_specs,
        &proof.r_final,
        &proof.batched_open_at_r,
        sponge,
    )?;
    if !batched_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "sshape3c::batched_open_at_r",
        });
    }

    // Open the hidden-pass preact commits at r_final; the opened
    // evals replace the previous public-Vec MLE evaluation.
    let preact_l_open_ok = hyrax_verify_at(
        &params.verifier_key,
        preact_lower_commit,
        &proof.r_final,
        proof.preact_l_eval,
        &proof.preact_l_open_at_r_final,
        n_vars,
        sponge,
    )?;
    if !preact_l_open_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "sshape3c: preact_lower_commit open at r_final",
        });
    }
    let preact_u_open_ok = hyrax_verify_at(
        &params.verifier_key,
        preact_upper_commit,
        &proof.r_final,
        proof.preact_u_eval,
        &proof.preact_u_open_at_r_final,
        n_vars,
        sponge,
    )?;
    if !preact_u_open_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "sshape3c: preact_upper_commit open at r_final",
        });
    }

    let c_fd_sigma_fr = signed_lift_to_fr(s_d_code * s_x_code);
    let c_fd_d_fr = signed_lift_to_fr(s_v_code);
    let epsilon_crit_fr = signed_lift_to_fr(
        super::super::sshape_helpers::fd_epsilon_crit(s_d_code, s_x_code).ok_or(
            SnarkError::ShapeMismatch {
                what: "sshape3c verify: ε_crit overflow",
            },
        )?,
    );
    let s_d_fr = signed_lift_to_fr(s_d_code);
    let s_b_fr = signed_lift_to_fr(s_b_code);
    let s_w_fr = signed_lift_to_fr(s_w_code);
    let s_v_fr = signed_lift_to_fr(s_v_code);
    let line_sign_fr: Fr = match line {
        SshapeLineKind::Upper => Fr::from(1u64),
        SshapeLineKind::Lower => -Fr::from(1u64),
    };

    let eq_eval = eval_multilinear_full(&build_eq_table(&proof.r_test), &proof.r_final);
    let fd1_id = proof.slack_fd1_eval
        - (proof.sigma_lo_z_eval - proof.sigma_up_zmd_eval) * c_fd_sigma_fr
        + proof.d_eval * c_fd_d_fr
        - epsilon_crit_fr;
    let fd2_id = proof.slack_fd2_eval - proof.d_eval * c_fd_d_fr
        + (proof.sigma_up_zpd_eval - proof.sigma_lo_z_eval) * c_fd_sigma_fr
        - epsilon_crit_fr;
    let factor_a_id = match line {
        SshapeLineKind::Upper => proof.factor_a_eval - (proof.preact_u_eval - proof.z_eval),
        SshapeLineKind::Lower => proof.factor_a_eval - (-proof.z_eval - proof.preact_l_eval),
    };
    let factor_b_id = match line {
        SshapeLineKind::Upper => {
            proof.factor_b_eval - proof.dz_sigma_code_eval - proof.b_sigma_code_eval
                + proof.sigma_up_z_eval
        }
        SshapeLineKind::Lower => {
            let sigma_lo_neg_z = if matches!(kind, ActivationKind::Sigmoid) {
                s_v_fr - proof.sigma_up_z_eval
            } else {
                -proof.sigma_up_z_eval
            };
            proof.factor_b_eval - sigma_lo_neg_z - proof.dz_sigma_code_eval
                + proof.b_sigma_code_eval
        }
    };
    let id_dz =
        proof.d_eval * proof.z_eval - proof.dz_step_1_eval * s_d_fr - proof.dz_step_1_rem_eval;
    let id_dzs =
        proof.dz_step_1_eval * s_v_fr - proof.dz_sigma_code_eval * s_w_fr - proof.dz_sigma_rem_eval;
    let id_bs = line_sign_fr * (proof.b_eval * s_v_fr - proof.b_sigma_code_eval * s_b_fr)
        - proof.b_sigma_rem_eval;
    let chunk_modulus_fr = signed_lift_to_fr(1i128 << params.gadget_range_bits);
    let id_slack1_chunk = proof.slack_fd1_eval
        - proof.slack_fd1_high_eval * chunk_modulus_fr
        - proof.slack_fd1_low_eval;
    let id_slack2_chunk = proof.slack_fd2_eval
        - proof.slack_fd2_high_eval * chunk_modulus_fr
        - proof.slack_fd2_low_eval;
    let one_fr = Fr::from(1u64);
    let id_boolean = proof.inside_eval * (one_fr - proof.inside_eval);
    use ark_ff::AdditiveGroup;
    let id_slack_pos = proof.slack_pos_eval - proof.inside_eval.double() * proof.factor_a_eval
        + proof.factor_a_eval
        + one_fr
        - proof.inside_eval;
    let id_slack_pos_chunk = proof.slack_pos_eval
        - proof.slack_pos_high_eval * chunk_modulus_fr
        - proof.slack_pos_low_eval;
    let id_gated_gap = proof.gated_gap_eval - proof.inside_eval * proof.factor_b_eval;
    let id_gated_gap_chunk = proof.gated_gap_eval
        - proof.gated_gap_high_eval * chunk_modulus_fr
        - proof.gated_gap_low_eval;
    let fd1_id_gated = proof.is_active_eval * fd1_id;
    let fd2_id_gated = proof.is_active_eval * fd2_id;
    let id_active_boolean = proof.is_active_eval * (one_fr - proof.is_active_eval);
    let id_active_zero = (one_fr - proof.is_active_eval) * proof.d_eval;
    let is_real_table: Vec<Fr> = (0..n_padded)
        .map(|j| {
            if j < n {
                Fr::from(1u64)
            } else {
                Fr::from(0u64)
            }
        })
        .collect();
    let is_real_eval = eval_multilinear_full(&is_real_table, &proof.r_final);
    let lhs = eq_eval
        * is_real_eval
        * (fd1_id_gated
            + combined_rho_a * fd2_id_gated
            + combined_rho_b * factor_a_id
            + combined_rho_c * factor_b_id
            + combined_rho_d * id_dz
            + combined_rho_e * id_dzs
            + combined_rho_f * id_bs
            + combined_rho_g * id_slack1_chunk
            + combined_rho_h * id_slack2_chunk
            + combined_rho_i * id_boolean
            + combined_rho_j * id_slack_pos
            + combined_rho_k * id_slack_pos_chunk
            + combined_rho_l * id_gated_gap
            + combined_rho_m * id_gated_gap_chunk
            + combined_rho_n * id_active_boolean
            + combined_rho_o * id_active_zero);
    if lhs != current_sum {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape3c: final identity at r_final",
        });
    }

    Ok(())
}
