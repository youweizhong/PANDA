//! Activation-layer backward arithmetic check and chain driver.
//!
//! Proves the per-layer `b_acc` update is consistent with the
//! activation relaxation: at each activation step the prover commits
//! `delta_b_doubled[i] = Σ_j A_pos[i,j] · b_pos[j] + A_neg[i,j] · b_neg[j]`
//! (with `A_neg = A_old − A_pos`) and proves the claim through an
//! eq-weighted two-product sumcheck. The j-axis vectors swap between
//! `(b_lower, b_upper)` and `(b_upper, b_lower)` depending on which
//! bound direction is being driven.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_std::rand::RngCore;

use crate::quantized_crown::{ActivationStepTrace, BackwardTrace, BoundDir, QuantRelaxation};
use crate::snark_primitives::finite_field::signed_lift_to_fr;

use crate::snark::backward_pass::signed_components::relu_lookup;
use crate::snark::commitment::commit::{
    PassCommitments, PassProverStates, ProverPolyStates, TensorCommitments,
};
use crate::snark::commitment::multilinear_extensions::{
    build_eq_table_tiled, eval_eq, eval_multilinear_full, mle_table_from_matrix,
    mle_table_from_vector, tile_j_along_i,
};
use crate::snark::commitment::pcs_helpers::{
    hyrax_open_at, hyrax_open_batched_at, hyrax_verify_at, hyrax_verify_batched_at, BatchOpenSpec,
    BatchVerifySpec,
};
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;
use crate::snark::proof::ActivationLayerStepProof;

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_activation_backward_proofs(
    trace: &BackwardTrace,
    direction: BoundDir,
    cert_relaxations: &[Option<QuantRelaxation>],
    pass_com: &PassCommitments,
    pass_st: &PassProverStates,
    commitments: &TensorCommitments,
    prover_states: &ProverPolyStates,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<Vec<ActivationLayerStepProof>, SnarkError> {
    let _timing = crate::timing::scope("act_backward");
    let mut out: Vec<ActivationLayerStepProof> = Vec::with_capacity(trace.activation_steps.len());
    for (step_idx, step) in trace.activation_steps.iter().enumerate() {
        let proof = build_one_activation_step(
            step,
            step_idx,
            direction,
            cert_relaxations,
            pass_com,
            pass_st,
            commitments,
            prover_states,
            params,
            sponge,
            rng,
        )?;
        out.push(proof);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn build_one_activation_step(
    step: &ActivationStepTrace,
    step_idx: usize,
    direction: BoundDir,
    cert_relaxations: &[Option<QuantRelaxation>],
    pass_com: &PassCommitments,
    pass_st: &PassProverStates,
    commitments: &TensorCommitments,
    prover_states: &ProverPolyStates,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<ActivationLayerStepProof, SnarkError> {
    let relax = cert_relaxations[step.layer_idx]
        .as_ref()
        .expect("activation has relaxation");
    sponge.absorb(&(step.layer_idx as u64));
    sponge.absorb(&(direction as u64));

    let (a_old_evals, a_old_log_dims) = mle_table_from_matrix(&step.a_old);
    let (a_pos_evals, a_pos_log_dims) = mle_table_from_matrix(&step.a_pos);
    debug_assert_eq!(a_old_log_dims, a_pos_log_dims);
    let b_lower_evals = mle_table_from_vector(&relax.b_lower);
    let b_upper_evals = mle_table_from_vector(&relax.b_upper);

    let (lns, lni) = a_old_log_dims;
    sponge.absorb(&(lns as u64));
    sponge.absorb(&(lni as u64));
    let r_spec: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(lns);

    let eq_table = build_eq_table_tiled(&r_spec, lns, lni);
    let n = a_old_evals.len();
    let a_neg_evals: Vec<Fr> = (0..n).map(|i| a_old_evals[i] - a_pos_evals[i]).collect();
    let (pos_line_full, neg_line_full) = match direction {
        BoundDir::Lower => (
            tile_j_along_i(&b_lower_evals, lns, lni),
            tile_j_along_i(&b_upper_evals, lns, lni),
        ),
        BoundDir::Upper => (
            tile_j_along_i(&b_upper_evals, lns, lni),
            tile_j_along_i(&b_lower_evals, lns, lni),
        ),
    };

    let n_spec = step.a_old.nrows();
    let n_neur = step.a_old.ncols();
    let mut delta_b_doubled = vec![Fr::from(0u64); n_spec];
    for (i, slot) in delta_b_doubled.iter_mut().enumerate().take(n_spec) {
        let mut acc = Fr::from(0u64);
        for j in 0..n_neur {
            let a_ij = signed_lift_to_fr(step.a_old.codes[[i, j]]);
            let a_pos_ij = signed_lift_to_fr(step.a_pos.codes[[i, j]]);
            let a_neg_ij = a_ij - a_pos_ij;
            let pos_l = match direction {
                BoundDir::Lower => signed_lift_to_fr(relax.b_lower.codes[j]),
                BoundDir::Upper => signed_lift_to_fr(relax.b_upper.codes[j]),
            };
            let neg_l = match direction {
                BoundDir::Lower => signed_lift_to_fr(relax.b_upper.codes[j]),
                BoundDir::Upper => signed_lift_to_fr(relax.b_lower.codes[j]),
            };
            acc += a_pos_ij * pos_l + a_neg_ij * neg_l;
        }
        *slot = acc;
    }
    let delta_b_padded = {
        let mut t = vec![Fr::from(0u64); 1usize << lns];
        for (slot, v) in t.iter_mut().zip(delta_b_doubled.iter()).take(n_spec) {
            *slot = *v;
        }
        t
    };
    let delta_b_claim = eval_multilinear_full(&delta_b_padded, &r_spec);
    sponge.absorb(&delta_b_claim);

    let sumcheck = relu_lookup::prove_eq_two_product_sumcheck(
        &eq_table,
        &a_pos_evals,
        &pos_line_full,
        &a_neg_evals,
        &neg_line_full,
        delta_b_claim,
        sponge,
    )?;

    let r_full = sumcheck.r_full.clone();
    let r_j = r_full[lns..].to_vec();
    let a_old_aux = &pass_st.chain_a[step.layer_idx + 1];
    let a_old_com = &pass_com.chain_a[step.layer_idx + 1];
    let a_pos_aux = &pass_st.activation_a_pos[step_idx];
    let a_pos_com = &pass_com.activation_a_pos[step_idx];
    let relax_com = commitments.relaxation[step.layer_idx]
        .as_ref()
        .expect("activation has relaxation commit");
    let relax_st = prover_states.relaxation[step.layer_idx]
        .as_ref()
        .expect("activation has relaxation states");
    let (b_pos_com, b_pos_aux, b_neg_com, b_neg_aux) = match direction {
        BoundDir::Lower => (
            &relax_com.b_lower,
            &relax_st.b_lower,
            &relax_com.b_upper,
            &relax_st.b_upper,
        ),
        BoundDir::Upper => (
            &relax_com.b_upper,
            &relax_st.b_upper,
            &relax_com.b_lower,
            &relax_st.b_lower,
        ),
    };
    let matrix_n_vars = crate::snark::commitment::commit::n_vars_from_logs(&[lns, lni]);
    let bvec_n_vars = crate::snark::commitment::commit::n_vars_from_logs(&[lni]);
    let r_full_items = [
        BatchOpenSpec {
            aux: a_old_aux,
            commitment: a_old_com,
            commit_n_vars: matrix_n_vars,
        },
        BatchOpenSpec {
            aux: a_pos_aux,
            commitment: a_pos_com,
            commit_n_vars: matrix_n_vars,
        },
    ];
    let (r_full_vals, a_full_batched_open) =
        hyrax_open_batched_at(&params.committer_key, &r_full_items, &r_full, sponge, rng)?;
    let a_old_val = r_full_vals[0];
    let a_pos_val = r_full_vals[1];

    let r_j_items = [
        BatchOpenSpec {
            aux: b_pos_aux,
            commitment: b_pos_com,
            commit_n_vars: bvec_n_vars,
        },
        BatchOpenSpec {
            aux: b_neg_aux,
            commitment: b_neg_com,
            commit_n_vars: bvec_n_vars,
        },
    ];
    let (r_j_vals, b_j_batched_open) =
        hyrax_open_batched_at(&params.committer_key, &r_j_items, &r_j, sponge, rng)?;
    let b_pos_val = r_j_vals[0];
    let b_neg_val = r_j_vals[1];
    // Bind delta_b_claim to the committed pre-rescale bias-doubled
    // tensor by opening it at r_spec.
    let bias_doubled_aux = &pass_st.activation_bias_doubled[step_idx];
    let bias_doubled_com = &pass_com.activation_bias_doubled[step_idx];
    let (bias_doubled_val, bias_doubled_open) = hyrax_open_at(
        &params.committer_key,
        bias_doubled_aux,
        bias_doubled_com,
        &r_spec,
        sponge,
        rng,
    )?;
    debug_assert_eq!(a_pos_val, sumcheck.p1_eval);
    debug_assert_eq!(b_pos_val, sumcheck.q1_eval);
    debug_assert_eq!(b_neg_val, sumcheck.q2_eval);
    debug_assert_eq!(a_old_val - a_pos_val, sumcheck.p2_eval);
    debug_assert_eq!(
        bias_doubled_val, delta_b_claim,
        "activation_backward: committed bias_doubled(r_spec) must equal delta_b_claim"
    );
    let _ = cert_relaxations;

    Ok(ActivationLayerStepProof {
        layer_idx: step.layer_idx,
        a_old_log_dims,
        r_spec,
        delta_b_claim,
        sumcheck,
        a_full_batched_open,
        b_j_batched_open,
        bias_doubled_open,
        a_old_eval: a_old_val,
    })
}

pub(crate) fn verify_activation_backward_chain(
    proofs: &[ActivationLayerStepProof],
    direction: BoundDir,
    pass_com: &PassCommitments,
    commitments: &TensorCommitments,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    for (step_idx, step_proof) in proofs.iter().enumerate() {
        sponge.absorb(&(step_proof.layer_idx as u64));
        sponge.absorb(&(direction as u64));
        let (lns, lni) = step_proof.a_old_log_dims;
        sponge.absorb(&(lns as u64));
        sponge.absorb(&(lni as u64));
        let r_spec_check: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(lns);
        if r_spec_check != step_proof.r_spec {
            return Err(SnarkError::TranscriptMismatch);
        }
        sponge.absorb(&step_proof.delta_b_claim);
        relu_lookup::verify_eq_two_product_sumcheck(
            &step_proof.sumcheck,
            lns + lni,
            step_proof.delta_b_claim,
            sponge,
        )
        .map_err(|_| SnarkError::ActivationLayerRejected {
            layer: step_proof.layer_idx,
        })?;

        let r_full = step_proof.sumcheck.r_full.clone();
        let r_j = r_full[lns..].to_vec();

        // The eq table is `eq(i, r_spec)` tiled across j, so its MLE
        // at r_full reduces to `eq(r_spec, r_full_i)` — the j tail
        // collapses to 1.
        let r_full_i = &r_full[..lns];
        let expected_eq_eval = eval_eq(&step_proof.r_spec, r_full_i);
        if step_proof.sumcheck.eq_eval != expected_eq_eval {
            return Err(SnarkError::ActivationLayerRejected {
                layer: step_proof.layer_idx,
            });
        }
        let a_old_com = &pass_com.chain_a[step_proof.layer_idx + 1];
        let a_pos_com = &pass_com.activation_a_pos[step_idx];
        let relax_com = commitments.relaxation[step_proof.layer_idx]
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing relaxation commit",
            })?;
        let (b_pos_com, b_neg_com) = match direction {
            BoundDir::Lower => (&relax_com.b_lower, &relax_com.b_upper),
            BoundDir::Upper => (&relax_com.b_upper, &relax_com.b_lower),
        };
        let bias_doubled_com = &pass_com.activation_bias_doubled[step_idx];
        let a_matrix_nv = crate::snark::commitment::commit::n_vars_from_logs(&[lns, lni]);
        let b_vec_nv = crate::snark::commitment::commit::n_vars_from_logs(&[lni]);
        let bacc_nv = crate::snark::commitment::commit::n_vars_from_logs(&[lns]);

        let r_full_items = [
            BatchVerifySpec {
                commitment: a_old_com,
                value: step_proof.a_old_eval,
                commit_n_vars: a_matrix_nv,
            },
            BatchVerifySpec {
                commitment: a_pos_com,
                value: step_proof.sumcheck.p1_eval,
                commit_n_vars: a_matrix_nv,
            },
        ];
        let r_full_ok = hyrax_verify_batched_at(
            &params.verifier_key,
            &r_full_items,
            &r_full,
            &step_proof.a_full_batched_open,
            sponge,
        )?;
        if !r_full_ok {
            return Err(SnarkError::PcsOpenRejected {
                which: "act_A_full_batch",
            });
        }

        let r_j_items = [
            BatchVerifySpec {
                commitment: b_pos_com,
                value: step_proof.sumcheck.q1_eval,
                commit_n_vars: b_vec_nv,
            },
            BatchVerifySpec {
                commitment: b_neg_com,
                value: step_proof.sumcheck.q2_eval,
                commit_n_vars: b_vec_nv,
            },
        ];
        let r_j_ok = hyrax_verify_batched_at(
            &params.verifier_key,
            &r_j_items,
            &r_j,
            &step_proof.b_j_batched_open,
            sponge,
        )?;
        if !r_j_ok {
            return Err(SnarkError::PcsOpenRejected {
                which: "act_b_j_batch",
            });
        }

        let bias_ok = hyrax_verify_at(
            &params.verifier_key,
            bias_doubled_com,
            &step_proof.r_spec,
            step_proof.delta_b_claim,
            &step_proof.bias_doubled_open,
            bacc_nv,
            sponge,
        )?;
        if !bias_ok {
            return Err(SnarkError::PcsOpenRejected {
                which: "act_bias_doubled",
            });
        }

        if step_proof.sumcheck.p2_eval != step_proof.a_old_eval - step_proof.sumcheck.p1_eval {
            return Err(SnarkError::ActivationLayerRejected {
                layer: step_proof.layer_idx,
            });
        }
    }
    Ok(())
}
