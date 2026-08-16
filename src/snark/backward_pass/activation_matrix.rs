//! Activation-step matrix-path arithmetic check.
//!
//! At each activation step the prover claims a per-cell update
//! `a_d_doubled[i, j] = A_pos[i, j] · d_pos[j] + A_neg[i, j] · d_neg[j]`
//! with `A_neg = A_old − A_pos`. This module proves that identity
//! with an eq-weighted two-product sumcheck over the joint `(i, j)`
//! axis (the same degree-3 sumcheck used by `activation_step` and
//! `concretize`), then binds every operand back to its committed MLE
//! via Hyrax opens at the sumcheck final point.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;

use crate::quantized_crown::{ActivationStepTrace, BackwardTrace, BoundDir, QuantRelaxation};
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use crate::snark::backward_pass::signed_components::relu_lookup;
use crate::snark::commitment::commit::{
    PassCommitments, PassProverStates, ProverPolyStates, TensorCommitments,
};
use crate::snark::commitment::multilinear_extensions::{
    build_eq_table, eval_eq, mle_table_from_matrix, mle_table_from_vector, tile_j_along_i,
};
use crate::snark::commitment::pcs_helpers::{
    hyrax_open_at, hyrax_open_batched_at, hyrax_verify_at, hyrax_verify_batched_at, BatchOpenSpec,
    BatchVerifySpec,
};
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;

/// One activation step's matrix-path proof.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ActivationMatrixStepProof {
    pub layer_idx: usize,
    /// `(log n_spec, log n_neurons)` of the per-step matrices.
    pub log_dims: (usize, usize),
    /// FS random over the full `(i, j)` axis. Length `lns + lni`.
    pub r_init: Vec<Fr>,
    /// `a_d_doubled(r_init)` — the verifier's claim.
    pub a_d_claim: Fr,
    /// Eq-weighted two-product sumcheck binding the claim to
    /// `A_pos · D_pos + A_neg · D_neg`.
    pub sumcheck: relu_lookup::EqTwoProductProof,
    /// PCS open of `a_d_doubled` at `r_init` (singleton).
    pub a_d_open: <HyraxBn254 as MlPcs>::Proof,
    /// Batched PCS open of `(A_old, A_pos)` at `r_full`.
    pub a_full_batched_open: <HyraxBn254 as MlPcs>::Proof,
    /// Batched PCS open of `(d_pos, d_neg)` at the j-axis tail of
    /// `r_full`.
    pub d_j_batched_open: <HyraxBn254 as MlPcs>::Proof,
    /// `A_old(r_full)`, used to recover `A_neg = A_old − A_pos`.
    pub a_old_eval: Fr,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_activation_matrix_proofs(
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
) -> Result<Vec<ActivationMatrixStepProof>, SnarkError> {
    let _timing = crate::timing::scope("act_matrix");
    let mut out = Vec::with_capacity(trace.activation_steps.len());
    for (step_idx, step) in trace.activation_steps.iter().enumerate() {
        let proof = build_one_activation_matrix_step(
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
fn build_one_activation_matrix_step(
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
) -> Result<ActivationMatrixStepProof, SnarkError> {
    let relax = cert_relaxations[step.layer_idx]
        .as_ref()
        .expect("activation has relaxation");

    sponge.absorb(&(step.layer_idx as u64));
    sponge.absorb(&(direction as u64));

    let (a_old_evals, a_old_log_dims) = mle_table_from_matrix(&step.a_old);
    let (a_pos_evals, _) = mle_table_from_matrix(&step.a_pos);
    let (a_d_evals, a_d_log_dims) = mle_table_from_matrix(&step.a_d_doubled);
    debug_assert_eq!(a_old_log_dims, a_d_log_dims);
    let d_lower_evals = mle_table_from_vector(&relax.d_lower);
    let d_upper_evals = mle_table_from_vector(&relax.d_upper);

    let (lns, lni) = a_old_log_dims;
    sponge.absorb(&(lns as u64));
    sponge.absorb(&(lni as u64));
    let r_init: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(lns + lni);

    let eq_table = build_eq_table(&r_init);

    // d_pos / d_neg are j-axis vectors; tile along i so they match
    // the joint (i, j) layout of A_pos / A_neg in the sumcheck.
    let (d_pos_tiled, d_neg_tiled) = match direction {
        BoundDir::Lower => (
            tile_j_along_i(&d_lower_evals, lns, lni),
            tile_j_along_i(&d_upper_evals, lns, lni),
        ),
        BoundDir::Upper => (
            tile_j_along_i(&d_upper_evals, lns, lni),
            tile_j_along_i(&d_lower_evals, lns, lni),
        ),
    };

    let a_d_claim = crate::snark::commitment::multilinear_extensions::eval_multilinear_full(
        &a_d_evals, &r_init,
    );
    sponge.absorb(&a_d_claim);

    let n = a_old_evals.len();
    let a_neg_evals: Vec<Fr> = (0..n).map(|i| a_old_evals[i] - a_pos_evals[i]).collect();

    let sumcheck = relu_lookup::prove_eq_two_product_sumcheck(
        &eq_table,
        &a_pos_evals,
        &d_pos_tiled,
        &a_neg_evals,
        &d_neg_tiled,
        a_d_claim,
        sponge,
    )?;

    debug_assert_eq!(
        sumcheck.eq_eval,
        eval_eq(&sumcheck.r_full, &r_init),
        "activation_matrix: sumcheck.eq_eval must equal eq(r_full, r_init)"
    );

    let r_full = sumcheck.r_full.clone();
    let r_j = r_full[lns..].to_vec();

    let a_d_aux = &pass_st.activation_a_d_doubled[step_idx];
    let a_d_com = &pass_com.activation_a_d_doubled[step_idx];
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
    let (d_pos_com, d_pos_aux, d_neg_com, d_neg_aux) = match direction {
        BoundDir::Lower => (
            &relax_com.d_lower,
            &relax_st.d_lower,
            &relax_com.d_upper,
            &relax_st.d_upper,
        ),
        BoundDir::Upper => (
            &relax_com.d_upper,
            &relax_st.d_upper,
            &relax_com.d_lower,
            &relax_st.d_lower,
        ),
    };

    let (a_d_val, a_d_open) = hyrax_open_at(
        &params.committer_key,
        a_d_aux,
        a_d_com,
        &r_init,
        sponge,
        rng,
    )?;
    debug_assert_eq!(a_d_val, a_d_claim);

    // (A_old, A_pos) share the matrix shape, so batch them into a
    // single Hyrax open at r_full.
    let (lns, lni) = a_old_log_dims;
    let matrix_n_vars = crate::snark::commitment::commit::n_vars_from_logs(&[lns, lni]);
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

    // (d_pos, d_neg) share the vector shape; batch them at r_j.
    let dvec_n_vars = crate::snark::commitment::commit::n_vars_from_logs(&[lni]);
    let r_j_items = [
        BatchOpenSpec {
            aux: d_pos_aux,
            commitment: d_pos_com,
            commit_n_vars: dvec_n_vars,
        },
        BatchOpenSpec {
            aux: d_neg_aux,
            commitment: d_neg_com,
            commit_n_vars: dvec_n_vars,
        },
    ];
    let (r_j_vals, d_j_batched_open) =
        hyrax_open_batched_at(&params.committer_key, &r_j_items, &r_j, sponge, rng)?;
    let d_pos_val = r_j_vals[0];
    let d_neg_val = r_j_vals[1];

    debug_assert_eq!(a_pos_val, sumcheck.p1_eval);
    debug_assert_eq!(d_pos_val, sumcheck.q1_eval);
    debug_assert_eq!(d_neg_val, sumcheck.q2_eval);
    debug_assert_eq!(a_old_val - a_pos_val, sumcheck.p2_eval);

    Ok(ActivationMatrixStepProof {
        layer_idx: step.layer_idx,
        log_dims: a_old_log_dims,
        r_init,
        a_d_claim,
        sumcheck,
        a_d_open,
        a_full_batched_open,
        d_j_batched_open,
        a_old_eval: a_old_val,
    })
}

pub(crate) fn verify_activation_matrix_chain(
    proofs: &[ActivationMatrixStepProof],
    direction: BoundDir,
    pass_com: &PassCommitments,
    commitments: &TensorCommitments,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    for (step_idx, proof) in proofs.iter().enumerate() {
        sponge.absorb(&(proof.layer_idx as u64));
        sponge.absorb(&(direction as u64));
        let (lns, lni) = proof.log_dims;
        sponge.absorb(&(lns as u64));
        sponge.absorb(&(lni as u64));
        let r_init_check: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(lns + lni);
        if r_init_check != proof.r_init {
            return Err(SnarkError::TranscriptMismatch);
        }
        sponge.absorb(&proof.a_d_claim);
        relu_lookup::verify_eq_two_product_sumcheck(
            &proof.sumcheck,
            lns + lni,
            proof.a_d_claim,
            sponge,
        )
        .map_err(|_| SnarkError::ActivationMatrixRejected {
            layer: proof.layer_idx,
        })?;

        // Pin eq_eval to its canonical value at r_full / r_init so a
        // malicious prover cannot smuggle a different eq factor.
        let expected_eq_eval = eval_eq(&proof.sumcheck.r_full, &proof.r_init);
        if proof.sumcheck.eq_eval != expected_eq_eval {
            return Err(SnarkError::ActivationMatrixRejected {
                layer: proof.layer_idx,
            });
        }

        let r_full = proof.sumcheck.r_full.clone();
        let r_j = r_full[lns..].to_vec();
        let a_d_com = &pass_com.activation_a_d_doubled[step_idx];
        let a_old_com = &pass_com.chain_a[proof.layer_idx + 1];
        let a_pos_com = &pass_com.activation_a_pos[step_idx];
        let relax_com =
            commitments.relaxation[proof.layer_idx]
                .as_ref()
                .ok_or(SnarkError::ShapeMismatch {
                    what: "missing relaxation commit (act_matrix)",
                })?;
        let (d_pos_com, d_neg_com) = match direction {
            BoundDir::Lower => (&relax_com.d_lower, &relax_com.d_upper),
            BoundDir::Upper => (&relax_com.d_upper, &relax_com.d_lower),
        };
        let matrix_nv = crate::snark::commitment::commit::n_vars_from_logs(&[lns, lni]);
        let dvec_nv = crate::snark::commitment::commit::n_vars_from_logs(&[lni]);

        let a_d_ok = hyrax_verify_at(
            &params.verifier_key,
            a_d_com,
            &proof.r_init,
            proof.a_d_claim,
            &proof.a_d_open,
            matrix_nv,
            sponge,
        )?;
        if !a_d_ok {
            return Err(SnarkError::PcsOpenRejected {
                which: "act_matrix_a_d",
            });
        }

        let r_full_items = [
            BatchVerifySpec {
                commitment: a_old_com,
                value: proof.a_old_eval,
                commit_n_vars: matrix_nv,
            },
            BatchVerifySpec {
                commitment: a_pos_com,
                value: proof.sumcheck.p1_eval,
                commit_n_vars: matrix_nv,
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
                which: "act_matrix_a_full_batch",
            });
        }

        let r_j_items = [
            BatchVerifySpec {
                commitment: d_pos_com,
                value: proof.sumcheck.q1_eval,
                commit_n_vars: dvec_nv,
            },
            BatchVerifySpec {
                commitment: d_neg_com,
                value: proof.sumcheck.q2_eval,
                commit_n_vars: dvec_nv,
            },
        ];
        let r_j_ok = hyrax_verify_batched_at(
            &params.verifier_key,
            &r_j_items,
            &r_j,
            &proof.d_j_batched_open,
            sponge,
        )?;
        if !r_j_ok {
            return Err(SnarkError::PcsOpenRejected {
                which: "act_matrix_d_j_batch",
            });
        }

        // A_neg = A_old − A_pos at r_full.
        if proof.sumcheck.p2_eval != proof.a_old_eval - proof.sumcheck.p1_eval {
            return Err(SnarkError::ActivationMatrixRejected {
                layer: proof.layer_idx,
            });
        }
    }
    Ok(())
}
