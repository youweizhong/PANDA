//! Concretize-step proof. Runs the eq-two-product sumcheck over the
//! tiled `(A_+, A_-, x_l, x_u)` layout, batched-opens the involved
//! commits at the sumcheck challenge points, and binds the result to
//! the committed `target_doubled` tensor at the spec point.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_std::rand::RngCore;

use crate::quantized_crown::{BoundDir, ConcretizeTrace, QuantCert};
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
use crate::snark::proof::ConcretizeStepProof;

/// Build the concretize-step proof for one bound direction.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_concretize_proof(
    cert: &QuantCert,
    concretize: &ConcretizeTrace,
    direction: BoundDir,
    pass_com: &PassCommitments,
    pass_st: &PassProverStates,
    commitments: &TensorCommitments,
    prover_states: &ProverPolyStates,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<ConcretizeStepProof, SnarkError> {
    let _timing = crate::timing::scope("concretize");
    sponge.absorb(&(direction as u64));

    let (a_final_evals, a_log_dims) = mle_table_from_matrix(&concretize.a_final);
    let (a_pos_evals, a_pos_log_dims) = mle_table_from_matrix(&concretize.a_pos);
    debug_assert_eq!(a_log_dims, a_pos_log_dims);
    let x_lower_evals = mle_table_from_vector(&cert.x_lower);
    let x_upper_evals = mle_table_from_vector(&cert.x_upper);

    let (lns, lni) = a_log_dims;
    sponge.absorb(&(lns as u64));
    sponge.absorb(&(lni as u64));
    let r_spec: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(lns);

    let eq_table = build_eq_table_tiled(&r_spec, lns, lni);
    let n = a_final_evals.len();
    let a_neg_evals: Vec<Fr> = (0..n).map(|i| a_final_evals[i] - a_pos_evals[i]).collect();
    let (x_pos_full, x_neg_full) = match direction {
        BoundDir::Lower => (
            tile_j_along_i(&x_lower_evals, lns, lni),
            tile_j_along_i(&x_upper_evals, lns, lni),
        ),
        BoundDir::Upper => (
            tile_j_along_i(&x_upper_evals, lns, lni),
            tile_j_along_i(&x_lower_evals, lns, lni),
        ),
    };

    let target_doubled_padded: Vec<Fr> = {
        let mut t = vec![Fr::from(0u64); 1usize << lns];
        for (slot, code) in t.iter_mut().zip(concretize.target_doubled.codes.iter()) {
            *slot = signed_lift_to_fr(*code);
        }
        t
    };
    let target_doubled_claim = eval_multilinear_full(&target_doubled_padded, &r_spec);
    sponge.absorb(&target_doubled_claim);

    let sumcheck = relu_lookup::prove_eq_two_product_sumcheck(
        &eq_table,
        &a_pos_evals,
        &x_pos_full,
        &a_neg_evals,
        &x_neg_full,
        target_doubled_claim,
        sponge,
    )?;

    let r_full = sumcheck.r_full.clone();
    let r_j = r_full[lns..].to_vec();
    let a_final_aux = &pass_st.chain_a[0];
    let a_final_com = &pass_com.chain_a[0];
    let a_pos_aux = pass_st
        .concretize_a_pos
        .as_ref()
        .ok_or(SnarkError::ShapeMismatch {
            what: "missing concretize_a_pos state",
        })?;
    let a_pos_com = pass_com
        .concretize_a_pos
        .as_ref()
        .ok_or(SnarkError::ShapeMismatch {
            what: "missing concretize_a_pos commit",
        })?;
    let (x_pos_com, x_pos_aux, x_neg_com, x_neg_aux) = match direction {
        BoundDir::Lower => (
            &commitments.x_lower,
            &prover_states.x_lower,
            &commitments.x_upper,
            &prover_states.x_upper,
        ),
        BoundDir::Upper => (
            &commitments.x_upper,
            &prover_states.x_upper,
            &commitments.x_lower,
            &prover_states.x_lower,
        ),
    };
    // (A_final, A_pos) batch-open at r_full; (x_pos, x_neg) at r_j.
    let matrix_n_vars = crate::snark::commitment::commit::n_vars_from_logs(&[lns, lni]);
    let xvec_n_vars = crate::snark::commitment::commit::n_vars_from_logs(&[lni]);
    let r_full_items = [
        BatchOpenSpec {
            aux: a_final_aux,
            commitment: a_final_com,
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
    let a_final_val = r_full_vals[0];
    let a_pos_val = r_full_vals[1];

    let r_j_items = [
        BatchOpenSpec {
            aux: x_pos_aux,
            commitment: x_pos_com,
            commit_n_vars: xvec_n_vars,
        },
        BatchOpenSpec {
            aux: x_neg_aux,
            commitment: x_neg_com,
            commit_n_vars: xvec_n_vars,
        },
    ];
    let (r_j_vals, x_j_batched_open) =
        hyrax_open_batched_at(&params.committer_key, &r_j_items, &r_j, sponge, rng)?;
    let x_pos_val = r_j_vals[0];
    let x_neg_val = r_j_vals[1];
    // Bind target_doubled_claim to its committed pre-rescale tensor.
    let target_doubled_aux =
        pass_st
            .concretize_target_doubled
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing concretize_target_doubled state",
            })?;
    let target_doubled_com =
        pass_com
            .concretize_target_doubled
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing concretize_target_doubled commit",
            })?;
    let (target_doubled_val, target_doubled_open) = hyrax_open_at(
        &params.committer_key,
        target_doubled_aux,
        target_doubled_com,
        &r_spec,
        sponge,
        rng,
    )?;
    debug_assert_eq!(a_pos_val, sumcheck.p1_eval);
    debug_assert_eq!(x_pos_val, sumcheck.q1_eval);
    debug_assert_eq!(x_neg_val, sumcheck.q2_eval);
    debug_assert_eq!(a_final_val - a_pos_val, sumcheck.p2_eval);
    debug_assert_eq!(
        target_doubled_val, target_doubled_claim,
        "concretize: committed target_doubled(r_spec) must equal target_doubled_claim"
    );
    Ok(ConcretizeStepProof {
        a_final_log_dims: a_log_dims,
        r_spec,
        target_doubled_claim,
        sumcheck,
        a_full_batched_open,
        x_j_batched_open,
        target_doubled_open,
        a_final_eval: a_final_val,
    })
}

/// Verify a concretize-step proof.
pub(crate) fn verify_concretize_proof(
    proof: &ConcretizeStepProof,
    direction: BoundDir,
    pass_com: &PassCommitments,
    commitments: &TensorCommitments,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    sponge.absorb(&(direction as u64));
    let (lns, lni) = proof.a_final_log_dims;
    sponge.absorb(&(lns as u64));
    sponge.absorb(&(lni as u64));
    let r_spec_check: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(lns);
    if r_spec_check != proof.r_spec {
        return Err(SnarkError::TranscriptMismatch);
    }
    sponge.absorb(&proof.target_doubled_claim);
    relu_lookup::verify_eq_two_product_sumcheck(
        &proof.sumcheck,
        lns + lni,
        proof.target_doubled_claim,
        sponge,
    )
    .map_err(|_| SnarkError::ConcretizeRejected)?;

    let r_full = proof.sumcheck.r_full.clone();
    let r_j = r_full[lns..].to_vec();

    // Bind eq_eval to its canonical value at the tiled-eq layout.
    let r_full_i = &r_full[..lns];
    let expected_eq_eval = eval_eq(&proof.r_spec, r_full_i);
    if proof.sumcheck.eq_eval != expected_eq_eval {
        return Err(SnarkError::ConcretizeRejected);
    }
    let a_final_com = &pass_com.chain_a[0];
    let a_pos_com = pass_com
        .concretize_a_pos
        .as_ref()
        .ok_or(SnarkError::ShapeMismatch {
            what: "missing concretize_a_pos commit",
        })?;
    let (x_pos_com, x_neg_com) = match direction {
        BoundDir::Lower => (&commitments.x_lower, &commitments.x_upper),
        BoundDir::Upper => (&commitments.x_upper, &commitments.x_lower),
    };
    let target_doubled_com =
        pass_com
            .concretize_target_doubled
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing concretize_target_doubled commit (verify)",
            })?;
    let a_final_nv = crate::snark::commitment::commit::n_vars_from_logs(&[lns, lni]);
    let x_vec_nv = crate::snark::commitment::commit::n_vars_from_logs(&[lni]);
    let target_nv = crate::snark::commitment::commit::n_vars_from_logs(&[lns]);

    let r_full_items = [
        BatchVerifySpec {
            commitment: a_final_com,
            value: proof.a_final_eval,
            commit_n_vars: a_final_nv,
        },
        BatchVerifySpec {
            commitment: a_pos_com,
            value: proof.sumcheck.p1_eval,
            commit_n_vars: a_final_nv,
        },
    ];
    let r_full_ok = hyrax_verify_batched_at(
        &params.verifier_key,
        &r_full_items,
        &r_full,
        &proof.a_full_batched_open,
        sponge,
    )?;
    if !r_full_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "concretize_A_full_batch",
        });
    }

    let r_j_items = [
        BatchVerifySpec {
            commitment: x_pos_com,
            value: proof.sumcheck.q1_eval,
            commit_n_vars: x_vec_nv,
        },
        BatchVerifySpec {
            commitment: x_neg_com,
            value: proof.sumcheck.q2_eval,
            commit_n_vars: x_vec_nv,
        },
    ];
    let r_j_ok = hyrax_verify_batched_at(
        &params.verifier_key,
        &r_j_items,
        &r_j,
        &proof.x_j_batched_open,
        sponge,
    )?;
    if !r_j_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "concretize_x_j_batch",
        });
    }

    let target_ok = hyrax_verify_at(
        &params.verifier_key,
        target_doubled_com,
        &proof.r_spec,
        proof.target_doubled_claim,
        &proof.target_doubled_open,
        target_nv,
        sponge,
    )?;
    if !target_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "concretize_target_doubled",
        });
    }

    if proof.sumcheck.p2_eval != proof.a_final_eval - proof.sumcheck.p1_eval {
        return Err(SnarkError::ConcretizeRejected);
    }
    Ok(())
}
