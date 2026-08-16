//! Linear-layer backward arithmetic check and chain driver.
//!
//! At each linear step the prover must show `A_W = A_old · W` and
//! `A_b = A_old · b`. We batch the two claims with an FS-derived
//! `α`, run a single inner-product sumcheck over the shared inner
//! axis `j`, and bind every operand back to its commit via Hyrax
//! opens at the sumcheck's final point.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;

use crate::quantized_crown::{BackwardTrace, QuantCert};
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};
use crate::snark_primitives::sumcheck::{
    prove_inner_product_with_sponge, verify_inner_product_with_sponge, InnerProductProof,
};

use crate::snark::commitment::commit::{
    n_vars_from_logs, CommittedAux, PassCommitments, PassProverStates, ProverPolyStates,
    TensorCommitments,
};
use crate::snark::commitment::multilinear_extensions::{
    build_eq_table, concat, eval_multilinear_full, mle_table_from_matrix, mle_table_from_vector,
    partial_eval_lsb, partial_eval_msb,
};
use crate::snark::commitment::pcs_helpers::{hyrax_open_at, hyrax_verify_at};
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;
use crate::snark::proof::LinearLayerStepProof;

/// One linear backward step's batched matmul + matvec proof.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct LinearBackwardProof {
    pub r_spec: Vec<Fr>,
    pub r_out: Vec<Fr>,
    pub batching_alpha: Fr,
    pub matmul_claim: Fr,
    pub matvec_claim: Fr,
    pub sumcheck: InnerProductProof<Fr>,
    pub r_j: Vec<Fr>,
    pub a_old_eval: Fr,
    pub w_eval: Fr,
    pub b_eval: Fr,
    pub w_open: <HyraxBn254 as MlPcs>::Proof,
    pub b_open: <HyraxBn254 as MlPcs>::Proof,
    pub a_old_open: <HyraxBn254 as MlPcs>::Proof,
    pub a_w_open: <HyraxBn254 as MlPcs>::Proof,
    pub a_b_open: <HyraxBn254 as MlPcs>::Proof,
}

/// Prover-side commit handles for the five tensors opened by a
/// linear backward step.
pub struct LinearBackwardCommitContext<'a> {
    pub ck: &'a <HyraxBn254 as MlPcs>::CommitterKey,
    pub(crate) w_aux: &'a CommittedAux,
    pub w_commitment: &'a <HyraxBn254 as MlPcs>::Commitment,
    pub(crate) b_aux: &'a CommittedAux,
    pub b_commitment: &'a <HyraxBn254 as MlPcs>::Commitment,
    pub(crate) a_old_aux: &'a CommittedAux,
    pub a_old_commitment: &'a <HyraxBn254 as MlPcs>::Commitment,
    pub(crate) a_w_aux: &'a CommittedAux,
    pub a_w_commitment: &'a <HyraxBn254 as MlPcs>::Commitment,
    pub(crate) a_b_aux: &'a CommittedAux,
    pub a_b_commitment: &'a <HyraxBn254 as MlPcs>::Commitment,
    pub max_num_vars: usize,
}

/// Verifier-side commit handles mirroring [`LinearBackwardCommitContext`].
pub struct LinearBackwardVerifyContext<'a> {
    pub vk: &'a <HyraxBn254 as MlPcs>::VerifierKey,
    pub w_commitment: &'a <HyraxBn254 as MlPcs>::Commitment,
    pub b_commitment: &'a <HyraxBn254 as MlPcs>::Commitment,
    pub a_old_commitment: &'a <HyraxBn254 as MlPcs>::Commitment,
    pub a_w_commitment: &'a <HyraxBn254 as MlPcs>::Commitment,
    pub a_b_commitment: &'a <HyraxBn254 as MlPcs>::Commitment,
    pub max_num_vars: usize,
}

/// Five `(point, value)` pairs returned by [`verify_linear_backward`]
/// for downstream binding against commits.
#[derive(Clone, Debug)]
pub struct LinearBackwardOpenings {
    pub a_old: (Vec<Fr>, Fr),
    pub w: (Vec<Fr>, Fr),
    pub b: (Vec<Fr>, Fr),
    pub a_w: (Vec<Fr>, Fr),
    pub a_b: (Vec<Fr>, Fr),
}

#[allow(clippy::too_many_arguments)]
pub fn prove_linear_backward(
    a_old_evals: &[Fr],
    a_old_log_dims: (usize, usize),
    w_evals: &[Fr],
    w_log_dims: (usize, usize),
    b_evals: &[Fr],
    a_w_evals: &[Fr],
    a_b_evals: &[Fr],
    commit_ctx: &LinearBackwardCommitContext<'_>,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<LinearBackwardProof, SnarkError> {
    let (lns, lni) = a_old_log_dims;
    let (lni2, lno) = w_log_dims;
    if lni != lni2 {
        return Err(SnarkError::ShapeMismatch {
            what: "linear backward: A_old's inner dim != W's inner dim",
        });
    }
    if a_old_evals.len() != 1 << (lns + lni) {
        return Err(SnarkError::ShapeMismatch {
            what: "A_old eval table size",
        });
    }
    if w_evals.len() != 1 << (lni + lno) {
        return Err(SnarkError::ShapeMismatch {
            what: "W eval table size",
        });
    }
    if b_evals.len() != 1 << lni {
        return Err(SnarkError::ShapeMismatch {
            what: "b eval table size",
        });
    }
    if a_w_evals.len() != 1 << (lns + lno) {
        return Err(SnarkError::ShapeMismatch {
            what: "A_W eval table size",
        });
    }
    if a_b_evals.len() != 1 << lns {
        return Err(SnarkError::ShapeMismatch {
            what: "A_b eval table size",
        });
    }

    sponge.absorb(&(lns as u64));
    sponge.absorb(&(lni as u64));
    sponge.absorb(&(lno as u64));
    let r_spec: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(lns);
    let r_out: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(lno);
    let batching_alpha: Fr = sponge.squeeze_field_elements::<Fr>(1)[0];

    let left = partial_eval_msb(a_old_evals, &r_spec);
    let w_at_out = partial_eval_lsb(w_evals, &r_out, lno);
    debug_assert_eq!(left.len(), 1 << lni);
    debug_assert_eq!(w_at_out.len(), 1 << lni);
    debug_assert_eq!(b_evals.len(), 1 << lni);
    let right: Vec<Fr> = (0..(1 << lni))
        .map(|j| w_at_out[j] + batching_alpha * b_evals[j])
        .collect();

    let matmul_claim = eval_multilinear_full(a_w_evals, &concat(&r_spec, &r_out));
    let matvec_claim = eval_multilinear_full(a_b_evals, &r_spec);
    let combined_claim = matmul_claim + batching_alpha * matvec_claim;

    sponge.absorb(&matmul_claim);
    sponge.absorb(&matvec_claim);
    sponge.absorb(&combined_claim);

    let (sumcheck, r_j) = prove_inner_product_with_sponge::<Fr, _>(&left, &right, sponge)
        .map_err(SnarkError::Sumcheck)?;

    let a_old_eval = left
        .iter()
        .zip(build_eq_table(&r_j).iter())
        .map(|(a, eq)| *a * *eq)
        .sum::<Fr>();
    let mut full_point = Vec::with_capacity(lns + lni);
    full_point.extend_from_slice(&r_spec);
    full_point.extend_from_slice(&r_j);
    let a_old_eval_check = eval_multilinear_full(a_old_evals, &full_point);
    debug_assert_eq!(a_old_eval, a_old_eval_check);

    let w_eval = {
        let mut p = Vec::with_capacity(lni + lno);
        p.extend_from_slice(&r_j);
        p.extend_from_slice(&r_out);
        eval_multilinear_full(w_evals, &p)
    };
    let b_eval = eval_multilinear_full(b_evals, &r_j);

    let (w_val, w_open) = hyrax_open_at(
        commit_ctx.ck,
        commit_ctx.w_aux,
        commit_ctx.w_commitment,
        &concat(&r_j, &r_out),
        sponge,
        rng,
    )?;
    let (b_val, b_open) = hyrax_open_at(
        commit_ctx.ck,
        commit_ctx.b_aux,
        commit_ctx.b_commitment,
        &r_j,
        sponge,
        rng,
    )?;
    let (a_old_val, a_old_open) = hyrax_open_at(
        commit_ctx.ck,
        commit_ctx.a_old_aux,
        commit_ctx.a_old_commitment,
        &concat(&r_spec, &r_j),
        sponge,
        rng,
    )?;
    let (a_w_val, a_w_open) = hyrax_open_at(
        commit_ctx.ck,
        commit_ctx.a_w_aux,
        commit_ctx.a_w_commitment,
        &concat(&r_spec, &r_out),
        sponge,
        rng,
    )?;
    let (a_b_val, a_b_open) = hyrax_open_at(
        commit_ctx.ck,
        commit_ctx.a_b_aux,
        commit_ctx.a_b_commitment,
        &r_spec,
        sponge,
        rng,
    )?;
    debug_assert_eq!(w_val, w_eval);
    debug_assert_eq!(b_val, b_eval);
    debug_assert_eq!(a_old_val, a_old_eval);
    debug_assert_eq!(a_w_val, matmul_claim);
    debug_assert_eq!(a_b_val, matvec_claim);

    Ok(LinearBackwardProof {
        r_spec,
        r_out,
        batching_alpha,
        matmul_claim,
        matvec_claim,
        sumcheck,
        r_j,
        a_old_eval,
        w_eval,
        b_eval,
        w_open,
        b_open,
        a_old_open,
        a_w_open,
        a_b_open,
    })
}

/// Verify a [`LinearBackwardProof`].
pub fn verify_linear_backward(
    proof: &LinearBackwardProof,
    a_old_log_dims: (usize, usize),
    w_log_dims: (usize, usize),
    verify_ctx: &LinearBackwardVerifyContext<'_>,
    sponge: &mut impl CryptographicSponge,
) -> Result<LinearBackwardOpenings, SnarkError> {
    let (lns, lni) = a_old_log_dims;
    let (lni2, lno) = w_log_dims;
    if lni != lni2 {
        return Err(SnarkError::ShapeMismatch {
            what: "verifier linear backward: A_old / W inner-dim mismatch",
        });
    }
    if proof.r_spec.len() != lns || proof.r_out.len() != lno || proof.r_j.len() != lni {
        return Err(SnarkError::ShapeMismatch {
            what: "challenge length mismatch",
        });
    }
    sponge.absorb(&(lns as u64));
    sponge.absorb(&(lni as u64));
    sponge.absorb(&(lno as u64));
    let r_spec_check: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(lns);
    let r_out_check: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(lno);
    let alpha_check: Fr = sponge.squeeze_field_elements::<Fr>(1)[0];
    if r_spec_check != proof.r_spec
        || r_out_check != proof.r_out
        || alpha_check != proof.batching_alpha
    {
        return Err(SnarkError::TranscriptMismatch);
    }

    sponge.absorb(&proof.matmul_claim);
    sponge.absorb(&proof.matvec_claim);
    let combined = proof.matmul_claim + proof.batching_alpha * proof.matvec_claim;
    sponge.absorb(&combined);

    let final_value = proof.a_old_eval * (proof.w_eval + proof.batching_alpha * proof.b_eval);
    let r_j = verify_inner_product_with_sponge::<Fr, _>(
        combined,
        lni,
        &proof.sumcheck,
        final_value,
        sponge,
    )
    .map_err(SnarkError::Sumcheck)?;
    if r_j != proof.r_j {
        return Err(SnarkError::TranscriptMismatch);
    }

    let w_nv = n_vars_from_logs(&[lni, lno]);
    let b_nv = n_vars_from_logs(&[lni]);
    let a_old_nv = n_vars_from_logs(&[lns, lni]);
    let a_w_nv = n_vars_from_logs(&[lns, lno]);
    let a_b_nv = n_vars_from_logs(&[lns]);
    let checks = [
        (
            "W",
            verify_ctx.w_commitment,
            concat(&proof.r_j, &proof.r_out),
            proof.w_eval,
            &proof.w_open,
            w_nv,
        ),
        (
            "b",
            verify_ctx.b_commitment,
            proof.r_j.clone(),
            proof.b_eval,
            &proof.b_open,
            b_nv,
        ),
        (
            "A_old",
            verify_ctx.a_old_commitment,
            concat(&proof.r_spec, &proof.r_j),
            proof.a_old_eval,
            &proof.a_old_open,
            a_old_nv,
        ),
        (
            "A_W",
            verify_ctx.a_w_commitment,
            concat(&proof.r_spec, &proof.r_out),
            proof.matmul_claim,
            &proof.a_w_open,
            a_w_nv,
        ),
        (
            "A_b",
            verify_ctx.a_b_commitment,
            proof.r_spec.clone(),
            proof.matvec_claim,
            &proof.a_b_open,
            a_b_nv,
        ),
    ];
    for (which, com, point, value, open_proof, nv) in checks {
        let ok = hyrax_verify_at(verify_ctx.vk, com, &point, value, open_proof, nv, sponge)?;
        if !ok {
            return Err(SnarkError::PcsOpenRejected { which });
        }
    }

    Ok(LinearBackwardOpenings {
        a_old: (concat(&proof.r_spec, &proof.r_j), proof.a_old_eval),
        w: (concat(&proof.r_j, &proof.r_out), proof.w_eval),
        b: (proof.r_j.clone(), proof.b_eval),
        a_w: (concat(&proof.r_spec, &proof.r_out), proof.matmul_claim),
        a_b: (proof.r_spec.clone(), proof.matvec_claim),
    })
}

/// Produce one [`LinearLayerStepProof`] per linear step in `trace`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_linear_backward_proofs(
    cert: &QuantCert,
    trace: &BackwardTrace,
    pass_com: &PassCommitments,
    pass_st: &PassProverStates,
    commitments: &TensorCommitments,
    prover_states: &ProverPolyStates,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<Vec<LinearLayerStepProof>, SnarkError> {
    let _timing = crate::timing::scope("linear");
    let mut out: Vec<LinearLayerStepProof> = Vec::with_capacity(trace.linear_steps.len());
    for (step_idx, step) in trace.linear_steps.iter().enumerate() {
        let w = cert.weights[step.layer_idx]
            .as_ref()
            .expect("linear layer has weights");
        let b = cert.biases[step.layer_idx]
            .as_ref()
            .expect("linear layer has biases");
        let (a_old_evals, a_old_log_dims) = mle_table_from_matrix(&step.a_old);
        let (w_evals, w_log_dims) = mle_table_from_matrix(w);
        let b_evals_padded = mle_table_from_vector(b);
        let (a_w_evals, _) = mle_table_from_matrix(&step.a_w);
        let a_b_evals_padded = mle_table_from_vector(&step.a_b);
        sponge.absorb(&(step.layer_idx as u64));

        let w_aux = prover_states.weight[step.layer_idx]
            .as_ref()
            .expect("prover state for layer's weight commit");
        let b_aux = prover_states.bias[step.layer_idx]
            .as_ref()
            .expect("prover state for layer's bias commit");
        let w_commitment = commitments.weight[step.layer_idx]
            .as_ref()
            .expect("commitment for layer's weight");
        let b_commitment = commitments.bias[step.layer_idx]
            .as_ref()
            .expect("commitment for layer's bias");
        let a_old_aux = &pass_st.chain_a[step.layer_idx + 1];
        let a_old_commitment = &pass_com.chain_a[step.layer_idx + 1];
        let a_w_aux = &pass_st.linear_a_w[step_idx];
        let a_w_commitment = &pass_com.linear_a_w[step_idx];
        let a_b_aux = &pass_st.linear_a_b[step_idx];
        let a_b_commitment = &pass_com.linear_a_b[step_idx];
        let commit_ctx = LinearBackwardCommitContext {
            ck: &params.committer_key,
            w_aux,
            w_commitment,
            b_aux,
            b_commitment,
            a_old_aux,
            a_old_commitment,
            a_w_aux,
            a_w_commitment,
            a_b_aux,
            a_b_commitment,
            max_num_vars: params.max_num_vars,
        };

        let proof = prove_linear_backward(
            &a_old_evals,
            a_old_log_dims,
            &w_evals,
            w_log_dims,
            &b_evals_padded,
            &a_w_evals,
            &a_b_evals_padded,
            &commit_ctx,
            sponge,
            rng,
        )?;
        out.push(LinearLayerStepProof {
            layer_idx: step.layer_idx,
            a_old_log_dims,
            w_log_dims,
            proof,
        });
    }
    Ok(out)
}

/// Replay the per-layer transcript and verify each step proof.
pub(crate) fn verify_linear_backward_chain(
    proofs: &[LinearLayerStepProof],
    commitments: &TensorCommitments,
    pass_com: &PassCommitments,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    for (step_idx, step_proof) in proofs.iter().enumerate() {
        sponge.absorb(&(step_proof.layer_idx as u64));
        let w_commitment = commitments.weight[step_proof.layer_idx]
            .as_ref()
            .expect("commitment for layer's weight");
        let b_commitment = commitments.bias[step_proof.layer_idx]
            .as_ref()
            .expect("commitment for layer's bias");
        let a_old_commitment = &pass_com.chain_a[step_proof.layer_idx + 1];
        let a_w_commitment = &pass_com.linear_a_w[step_idx];
        let a_b_commitment = &pass_com.linear_a_b[step_idx];
        let verify_ctx = LinearBackwardVerifyContext {
            vk: &params.verifier_key,
            w_commitment,
            b_commitment,
            a_old_commitment,
            a_w_commitment,
            a_b_commitment,
            max_num_vars: params.max_num_vars,
        };
        verify_linear_backward(
            &step_proof.proof,
            step_proof.a_old_log_dims,
            step_proof.w_log_dims,
            &verify_ctx,
            sponge,
        )
        .map_err(|e| match e {
            SnarkError::PcsOpenRejected { which } => SnarkError::PcsOpenRejected { which },
            _ => SnarkError::LinearLayerRejected {
                layer: step_proof.layer_idx,
            },
        })?;
    }
    Ok(())
}
