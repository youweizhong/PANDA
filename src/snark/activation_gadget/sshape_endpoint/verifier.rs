//! Verifier for the endpoint gadget. Mirrors the prover's transcript
//! order and replays every check at FS-derived points.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::AdditiveGroup;

use crate::crown::network::ActivationKind;
use crate::quantization::scale::Scale;
use crate::snark_primitives::finite_field::signed_lift_to_fr;
use crate::snark_primitives::logup_gkr::verify_circuit_with_top;
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use super::super::relu_upper_endpoint::{squeeze_round_challenge_4, verify_pos_range};
use crate::snark::commitment::multilinear_extensions::{build_eq_table, eval_multilinear_full};
use crate::snark::commitment::pcs_helpers::{
    hyrax_verify_at, hyrax_verify_batched_at, BatchVerifySpec,
};
use crate::snark::errors::SnarkError;
use crate::snark::output_bound::absorb_commitment;
use crate::snark::params::SnarkParams;

use super::envelope_logup::sigma_envelope_3col_table_mle_eval;
use super::types::{SshapeEndpointKind, SshapeEndpointProof, SshapeLineKind};
use super::witness::{compute_sigma_used_fr, scale_precondition_holds};

/// Verify a `SshapeEndpointProof`. `n_real_neurons` is the public
/// cell count `n`; `preact_commit` is the hidden-pass commit through
/// which the verifier accesses the private preact codes.
#[allow(clippy::too_many_arguments)]
pub fn verify_sshape_at_endpoint(
    proof: &SshapeEndpointProof,
    expected_layer_idx: usize,
    kind: ActivationKind,
    line: SshapeLineKind,
    endpoint: SshapeEndpointKind,
    n_real_neurons: usize,
    preact_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_line_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_line_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s_d: Scale,
    s_b: Scale,
    s_w: Scale,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    if proof.layer_idx != expected_layer_idx {
        return Err(SnarkError::ArchitectureMismatch {
            what: "sshape_endpoint: layer_idx mismatch",
        });
    }
    let kind_tag_expected: u8 = match kind {
        ActivationKind::Sigmoid => 0,
        ActivationKind::Tanh => 1,
        ActivationKind::ReLU => {
            return Err(SnarkError::ShapeMismatch {
                what: "sshape_endpoint: ReLU verifier called",
            });
        }
    };
    if proof.kind_tag != kind_tag_expected {
        return Err(SnarkError::ArchitectureMismatch {
            what: "sshape_endpoint: kind tag mismatch",
        });
    }
    let endpoint_tag_expected = endpoint.tag();
    if proof.endpoint_tag != endpoint_tag_expected {
        return Err(SnarkError::ArchitectureMismatch {
            what: "sshape_endpoint: endpoint tag mismatch (U(l) vs U(u) replay)",
        });
    }
    let line_tag_expected = line.tag();
    if proof.line_tag != line_tag_expected {
        return Err(SnarkError::ArchitectureMismatch {
            what: "sshape_endpoint: line tag mismatch (upper-line vs lower-line replay)",
        });
    }
    if !scale_precondition_holds(s_d, s_b, s_w, params.gadget_range_bits) {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape_endpoint: scale precondition (verify)",
        });
    }
    let n = n_real_neurons;
    if n == 0 {
        return Err(SnarkError::Reserved {
            what: "sshape_endpoint: requires n ≥ 1 (verifier)",
        });
    }
    if proof.n_real != n {
        return Err(SnarkError::ArchitectureMismatch {
            what: "sshape_endpoint: n_real mismatch (verifier — proof's n_real differs \
                   from n_real_neurons input)",
        });
    }
    let n_vars = crate::snark::commitment::commit::native_vector_n_vars(n);
    if proof.n_vars != n_vars {
        return Err(SnarkError::ArchitectureMismatch {
            what: "sshape_endpoint: n_vars mismatch",
        });
    }
    let n_padded = 1usize << n_vars;
    if n > n_padded {
        return Err(SnarkError::ShapeMismatch {
            what: "sshape_endpoint: n > n_padded (verifier)",
        });
    }
    let s_d_code = 1i128 << s_d.pow2_exponent().unwrap();
    let s_b_code = 1i128 << s_b.pow2_exponent().unwrap();
    let s_w_code = 1i128 << s_w.pow2_exponent().unwrap();
    let s_v_code = 1i128 << params.sigma_v_scale_log2;
    let s_x_log2 = params.sigma_x_scale_log2;
    let s_w_log2_i = s_w.pow2_exponent().unwrap();
    if s_x_log2 < s_w_log2_i {
        return Err(SnarkError::Reserved {
            what: "sshape_endpoint: s_w > s_x not yet supported (verify)",
        });
    }
    let s_x_over_s_w_code: i128 = 1i128 << (s_x_log2 - s_w_log2_i);
    let s_x_over_s_w_fr = signed_lift_to_fr(s_x_over_s_w_code);

    sponge.absorb(&(proof.layer_idx as u64));
    sponge.absorb(&proof.kind_tag);
    sponge.absorb(&proof.endpoint_tag);
    sponge.absorb(&proof.line_tag);
    sponge.absorb(&(n_vars as u64));
    sponge.absorb(&(n as u64));
    absorb_commitment(sponge, d_line_commit);
    absorb_commitment(sponge, b_line_commit);
    absorb_commitment(sponge, preact_commit);

    absorb_commitment(sponge, &proof.abs_l_commit);
    absorb_commitment(sponge, &proof.sign_commit);
    absorb_commitment(sponge, &proof.sigma_upper_at_abs_commit);
    absorb_commitment(sponge, &proof.sigma_lower_at_abs_commit);
    absorb_commitment(sponge, &proof.dx_step_1_commit);
    absorb_commitment(sponge, &proof.dx_step_1_rem_commit);
    absorb_commitment(sponge, &proof.dx_sigma_code_commit);
    absorb_commitment(sponge, &proof.dx_sigma_rem_commit);
    absorb_commitment(sponge, &proof.b_sigma_code_commit);
    absorb_commitment(sponge, &proof.b_sigma_rem_commit);
    absorb_commitment(sponge, &proof.diff_commit);

    verify_pos_range(
        &proof.abs_l_range,
        n_vars,
        &proof.abs_l_commit,
        params,
        sponge,
    )?;
    verify_pos_range(
        &proof.dx_step_1_rem_range,
        n_vars,
        &proof.dx_step_1_rem_commit,
        params,
        sponge,
    )?;
    verify_pos_range(
        &proof.dx_sigma_rem_range,
        n_vars,
        &proof.dx_sigma_rem_commit,
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
    verify_pos_range(
        &proof.diff_range,
        n_vars,
        &proof.diff_commit,
        params,
        sponge,
    )?;

    sponge.absorb(&(n_padded as u64));
    let envelope_combine_alpha_1 = sponge.squeeze_field_elements::<Fr>(1)[0];
    if envelope_combine_alpha_1 != proof.envelope_combine_alpha_1 {
        return Err(SnarkError::TranscriptMismatch);
    }
    let envelope_combine_alpha_2 = sponge.squeeze_field_elements::<Fr>(1)[0];
    if envelope_combine_alpha_2 != proof.envelope_combine_alpha_2 {
        return Err(SnarkError::TranscriptMismatch);
    }
    // Derive expected LogUp dimensions from public inputs rather
    // than trusting the proof-supplied lengths.
    let table_upper_for_dim = match kind {
        ActivationKind::Sigmoid => &params.preprocessed.sigma.sigmoid_upper_fr,
        ActivationKind::Tanh => &params.preprocessed.sigma.tanh_upper_fr,
        ActivationKind::ReLU => unreachable!(),
    };
    let expected_witness_len = n_padded;
    let expected_table_len = table_upper_for_dim.len();
    if proof.envelope_witness_len != expected_witness_len
        || proof.envelope_table_len != expected_table_len
    {
        return Err(SnarkError::ArchitectureMismatch {
            what: "sshape_endpoint: envelope LogUp dimension mismatch with public inputs",
        });
    }
    if !expected_witness_len.is_power_of_two() || !expected_table_len.is_power_of_two() {
        return Err(SnarkError::ShapeMismatch {
            what: "sshape_endpoint: envelope dimensions not power-of-two",
        });
    }
    absorb_commitment(sponge, &proof.envelope_mult_commit);
    sponge.absorb(&(expected_witness_len as u64));
    sponge.absorb(&(expected_table_len as u64));
    let envelope_logup_beta = sponge.squeeze_field_elements::<Fr>(1)[0];
    if envelope_logup_beta != proof.envelope_logup_beta {
        return Err(SnarkError::TranscriptMismatch);
    }
    sponge.absorb(&envelope_combine_alpha_1);
    sponge.absorb(&envelope_combine_alpha_2);
    sponge.absorb(&envelope_logup_beta);

    let lookup_n = (expected_witness_len.trailing_zeros() as usize).saturating_sub(1);
    let table_n = (expected_table_len.trailing_zeros() as usize).saturating_sub(1);
    if proof.envelope_lookup_proof.bottom_point.len() != lookup_n + 1
        || proof.envelope_table_proof.bottom_point.len() != table_n + 1
    {
        return Err(SnarkError::ArchitectureMismatch {
            what: "sshape_endpoint: LogUp bottom_point length mismatch",
        });
    }
    let lookup_top_num = proof.envelope_lookup_top[0] * proof.envelope_lookup_top[3]
        + proof.envelope_lookup_top[1] * proof.envelope_lookup_top[2];
    let table_top_num = proof.envelope_table_top[0] * proof.envelope_table_top[3]
        + proof.envelope_table_top[1] * proof.envelope_table_top[2];
    verify_circuit_with_top(
        &proof.envelope_lookup_proof,
        lookup_n,
        proof.envelope_lookup_top,
        lookup_top_num,
        sponge,
    )
    .map_err(SnarkError::LogUp)?;
    verify_circuit_with_top(
        &proof.envelope_table_proof,
        table_n,
        proof.envelope_table_top,
        table_top_num,
        sponge,
    )
    .map_err(SnarkError::LogUp)?;

    // Enforce LogUp top-fraction cancellation
    // (lookup_num · table_den + lookup_den · table_num == 0). The
    // per-side GKR verifications above don't bind the two sums to
    // each other on their own.
    let lookup_top_den = proof.envelope_lookup_top[2] * proof.envelope_lookup_top[3];
    let table_top_den = proof.envelope_table_top[2] * proof.envelope_table_top[3];
    let combined = lookup_top_num * table_top_den + lookup_top_den * table_top_num;
    if combined != Fr::ZERO {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape_endpoint: σ-envelope LogUp top-fraction cancellation",
        });
    }

    let logup_point = proof.envelope_lookup_proof.bottom_point.clone();
    let logup_items = [
        BatchVerifySpec {
            commitment: &proof.abs_l_commit,
            commit_n_vars: n_vars,
            value: proof.envelope_abs_l_eval,
        },
        BatchVerifySpec {
            commitment: &proof.sigma_upper_at_abs_commit,
            commit_n_vars: n_vars,
            value: proof.envelope_sigma_upper_at_abs_eval,
        },
        BatchVerifySpec {
            commitment: &proof.sigma_lower_at_abs_commit,
            commit_n_vars: n_vars,
            value: proof.envelope_sigma_lower_at_abs_eval,
        },
    ];
    let logup_opens_ok = hyrax_verify_batched_at(
        &params.verifier_key,
        &logup_items,
        &logup_point,
        &proof.envelope_witness_batched_open,
        sponge,
    )?;
    if !logup_opens_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "sshape_endpoint::envelope_witness_batched_open",
        });
    }

    if proof.envelope_lookup_proof.bottom_denom
        != envelope_combine_alpha_1 * proof.envelope_abs_l_eval
            + envelope_combine_alpha_2 * proof.envelope_sigma_upper_at_abs_eval
            + proof.envelope_sigma_lower_at_abs_eval
            - envelope_logup_beta
    {
        return Err(SnarkError::PerTensorRangeWitnessNotBound);
    }

    let mult_bottom_pt = proof.envelope_table_proof.bottom_point.clone();
    let mult_open_ok = hyrax_verify_at(
        &params.verifier_key,
        &proof.envelope_mult_commit,
        &mult_bottom_pt,
        proof.envelope_table_proof.bottom_num,
        &proof.envelope_mult_open,
        proof.envelope_mult_n_vars,
        sponge,
    )?;
    if !mult_open_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "sshape_endpoint::envelope_mult_open",
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
    let canonical_table_eval = sigma_envelope_3col_table_mle_eval(
        &mult_bottom_pt,
        envelope_combine_alpha_1,
        envelope_combine_alpha_2,
        table_upper,
        table_lower,
    );
    if proof.envelope_table_proof.bottom_denom != canonical_table_eval - envelope_logup_beta {
        return Err(SnarkError::LogUpTableNotCanonical {
            which: "sshape_endpoint::envelope_table",
        });
    }

    let combined_rho_a = sponge.squeeze_field_elements::<Fr>(1)[0];
    if combined_rho_a != proof.combined_rho_a {
        return Err(SnarkError::TranscriptMismatch);
    }
    let combined_rho_b = sponge.squeeze_field_elements::<Fr>(1)[0];
    if combined_rho_b != proof.combined_rho_b {
        return Err(SnarkError::TranscriptMismatch);
    }
    let combined_rho_c = sponge.squeeze_field_elements::<Fr>(1)[0];
    if combined_rho_c != proof.combined_rho_c {
        return Err(SnarkError::TranscriptMismatch);
    }
    let combined_rho_d = sponge.squeeze_field_elements::<Fr>(1)[0];
    if combined_rho_d != proof.combined_rho_d {
        return Err(SnarkError::TranscriptMismatch);
    }
    let combined_rho_e = sponge.squeeze_field_elements::<Fr>(1)[0];
    if combined_rho_e != proof.combined_rho_e {
        return Err(SnarkError::TranscriptMismatch);
    }
    let s_d_fr = signed_lift_to_fr(s_d_code);
    let s_b_fr = signed_lift_to_fr(s_b_code);
    let s_w_fr = signed_lift_to_fr(s_w_code);
    let s_v_fr = signed_lift_to_fr(s_v_code);

    let r_test: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(n_vars);
    if r_test != proof.r_test {
        return Err(SnarkError::TranscriptMismatch);
    }
    // Derive r_final from the verifier's own sumcheck challenges and
    // bind it against `proof.r_final`.
    let mut current_sum = Fr::ZERO;
    let mut derived_r_final: Vec<Fr> = Vec::with_capacity(n_vars);
    for round in proof.rounds.iter() {
        if round.at_zero + round.at_one != current_sum {
            return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
                which: "sshape_endpoint: sumcheck round invariant (verify)",
            });
        }
        let r_i = squeeze_round_challenge_4(sponge, round);
        derived_r_final.push(r_i);
        current_sum = round.evaluate(r_i);
    }
    if proof.r_final.len() != n_vars {
        return Err(SnarkError::ShapeMismatch {
            what: "sshape_endpoint: r_final length",
        });
    }
    if derived_r_final != proof.r_final {
        return Err(SnarkError::TranscriptMismatch);
    }

    let r_final = derived_r_final;
    let r_items = [
        BatchVerifySpec {
            commitment: d_line_commit,
            commit_n_vars: n_vars,
            value: proof.d_line_eval,
        },
        BatchVerifySpec {
            commitment: b_line_commit,
            commit_n_vars: n_vars,
            value: proof.b_line_eval,
        },
        BatchVerifySpec {
            commitment: &proof.abs_l_commit,
            commit_n_vars: n_vars,
            value: proof.abs_l_eval,
        },
        BatchVerifySpec {
            commitment: &proof.sign_commit,
            commit_n_vars: n_vars,
            value: proof.sign_eval,
        },
        BatchVerifySpec {
            commitment: &proof.sigma_upper_at_abs_commit,
            commit_n_vars: n_vars,
            value: proof.sigma_upper_at_abs_eval,
        },
        BatchVerifySpec {
            commitment: &proof.sigma_lower_at_abs_commit,
            commit_n_vars: n_vars,
            value: proof.sigma_lower_at_abs_eval,
        },
        BatchVerifySpec {
            commitment: &proof.dx_step_1_commit,
            commit_n_vars: n_vars,
            value: proof.dx_step_1_eval,
        },
        BatchVerifySpec {
            commitment: &proof.dx_step_1_rem_commit,
            commit_n_vars: n_vars,
            value: proof.dx_step_1_rem_eval,
        },
        BatchVerifySpec {
            commitment: &proof.dx_sigma_code_commit,
            commit_n_vars: n_vars,
            value: proof.dx_sigma_code_eval,
        },
        BatchVerifySpec {
            commitment: &proof.dx_sigma_rem_commit,
            commit_n_vars: n_vars,
            value: proof.dx_sigma_rem_eval,
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
            commitment: &proof.diff_commit,
            commit_n_vars: n_vars,
            value: proof.diff_eval,
        },
    ];
    let opens_ok = hyrax_verify_batched_at(
        &params.verifier_key,
        &r_items,
        &r_final,
        &proof.batched_open_at_r,
        sponge,
    )?;
    if !opens_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "sshape_endpoint::batched_open_at_r",
        });
    }

    // Verify the open of `preact_commit` at r_final; the opened eval
    // becomes `l_eval` in the final identity below.
    let preact_open_ok = crate::snark::commitment::pcs_helpers::hyrax_verify_at(
        &params.verifier_key,
        preact_commit,
        &r_final,
        proof.preact_eval_at_r_final,
        &proof.preact_open_at_r_final,
        n_vars,
        sponge,
    )?;
    if !preact_open_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "sshape_endpoint: preact_open_at_r_final",
        });
    }
    let l_eval = proof.preact_eval_at_r_final;
    let eq_eval = eval_multilinear_full(&build_eq_table(&proof.r_test), &r_final);
    let s_used_eval = compute_sigma_used_fr(
        kind,
        line,
        proof.sigma_upper_at_abs_eval,
        proof.sigma_lower_at_abs_eval,
        proof.sign_eval,
        s_v_fr,
    );
    let one = Fr::from(1u64);
    let two = Fr::from(2u64);
    let line_sign_fr: Fr = match line {
        SshapeLineKind::Upper => Fr::from(1u64),
        SshapeLineKind::Lower => -Fr::from(1u64),
    };
    let id_1 = proof.d_line_eval * l_eval
        - proof.dx_step_1_eval * s_d_fr
        - line_sign_fr * proof.dx_step_1_rem_eval;
    let id_2 = proof.dx_step_1_eval * s_v_fr
        - proof.dx_sigma_code_eval * s_w_fr
        - line_sign_fr * proof.dx_sigma_rem_eval;
    let id_3 = proof.b_line_eval * s_v_fr
        - proof.b_sigma_code_eval * s_b_fr
        - line_sign_fr * proof.b_sigma_rem_eval;
    let id_4 = line_sign_fr * (proof.dx_sigma_code_eval + proof.b_sigma_code_eval - s_used_eval)
        - proof.diff_eval;
    let id_5 = proof.sign_eval * (one - proof.sign_eval);
    let id_6 =
        l_eval * s_x_over_s_w_fr - proof.abs_l_eval + two * proof.sign_eval * proof.abs_l_eval;
    let is_real_table: Vec<Fr> = (0..n_padded)
        .map(|j| {
            if j < n {
                Fr::from(1u64)
            } else {
                Fr::from(0u64)
            }
        })
        .collect();
    let is_real_eval = eval_multilinear_full(&is_real_table, &r_final);
    let expected = eq_eval
        * is_real_eval
        * (id_1
            + combined_rho_a * id_2
            + combined_rho_b * id_3
            + combined_rho_c * id_4
            + combined_rho_d * id_5
            + combined_rho_e * id_6);
    if expected != current_sum {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape_endpoint: final identity (verify, signed, line-aware)",
        });
    }

    Ok(())
}
