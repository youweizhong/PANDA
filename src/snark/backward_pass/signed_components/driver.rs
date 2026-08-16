//! Driver wiring for the ReLU-decomposition lookup gadget (the
//! sign-correctness check).

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_std::rand::RngCore;

use crate::quantized_crown::{BackwardTrace, ConcretizeTrace};

use super::relu_lookup;
use crate::snark::commitment::commit::{PassCommitments, PassProverStates};
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;
use crate::snark::proof::{ActivationLayerStepProof, ConcretizeStepProof};

// ---------------------------------------------------------------------------
// ReLU-decomposition lookup driver (active).
// ---------------------------------------------------------------------------

/// Emit one ReLU lookup proof per activation step, binding
/// `A_pos = ReLU(A_old)` cell-wise.
pub(crate) fn build_relu_proofs_activation(
    trace: &BackwardTrace,
    pass_com: &PassCommitments,
    pass_st: &PassProverStates,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<Vec<relu_lookup::ReluStepProof>, SnarkError> {
    let _timing = crate::timing::scope("relu_chain");
    let mut out = Vec::with_capacity(trace.activation_steps.len());
    for (step_idx, step) in trace.activation_steps.iter().enumerate() {
        let a_aux = &pass_st.chain_a[step.layer_idx + 1];
        let a_com = &pass_com.chain_a[step.layer_idx + 1];
        let a_pos_aux = &pass_st.activation_a_pos[step_idx];
        let a_pos_com = &pass_com.activation_a_pos[step_idx];
        let proof = relu_lookup::prove_relu_step(
            &step.a_old.codes,
            &step.a_pos.codes,
            a_aux,
            a_com,
            a_pos_aux,
            a_pos_com,
            params,
            sponge,
            rng,
        )?;
        out.push(proof);
    }
    Ok(out)
}

/// ReLU lookup proof for the concretize step (bottom of the chain).
pub(crate) fn build_relu_proof_concretize(
    concretize: &ConcretizeTrace,
    pass_com: &PassCommitments,
    pass_st: &PassProverStates,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<relu_lookup::ReluStepProof, SnarkError> {
    let _timing = crate::timing::scope("relu_chain");
    let a_aux = &pass_st.chain_a[0];
    let a_com = &pass_com.chain_a[0];
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
    relu_lookup::prove_relu_step(
        &concretize.a_final.codes,
        &concretize.a_pos.codes,
        a_aux,
        a_com,
        a_pos_aux,
        a_pos_com,
        params,
        sponge,
        rng,
    )
}

/// Verify the per-activation-step ReLU lookup proofs against the
/// committed `(A_old, A_pos)` tensors.
pub(crate) fn verify_relu_proofs_activation(
    proofs: &[relu_lookup::ReluStepProof],
    activation_steps: &[ActivationLayerStepProof],
    pass_com: &PassCommitments,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    if proofs.len() != activation_steps.len() {
        return Err(SnarkError::ShapeMismatch {
            what: "relu proofs vs activation step count",
        });
    }
    for (step_idx, (relu_proof, step)) in proofs.iter().zip(activation_steps.iter()).enumerate() {
        let a_com = &pass_com.chain_a[step.layer_idx + 1];
        let a_pos_com = &pass_com.activation_a_pos[step_idx];
        let (lns, lni) = step.a_old_log_dims;
        let n_padded = 1usize << (lns + lni);
        let commit_n_vars = crate::snark::commitment::commit::n_vars_from_logs(&[lns, lni]);
        relu_lookup::verify_relu_step(
            relu_proof,
            a_com,
            a_pos_com,
            params,
            n_padded,
            commit_n_vars,
            sponge,
        )?;
    }
    Ok(())
}

/// Verify the concretize-step ReLU lookup proof.
pub(crate) fn verify_relu_proof_concretize(
    relu_proof: &relu_lookup::ReluStepProof,
    concretize_proof: &ConcretizeStepProof,
    pass_com: &PassCommitments,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    let a_com = &pass_com.chain_a[0];
    let a_pos_com = pass_com
        .concretize_a_pos
        .as_ref()
        .ok_or(SnarkError::ShapeMismatch {
            what: "missing concretize_a_pos commit",
        })?;
    let (lns, lni) = concretize_proof.a_final_log_dims;
    let n_padded = 1usize << (lns + lni);
    let commit_n_vars = crate::snark::commitment::commit::n_vars_from_logs(&[lns, lni]);
    relu_lookup::verify_relu_step(
        relu_proof,
        a_com,
        a_pos_com,
        params,
        n_padded,
        commit_n_vars,
        sponge,
    )
}
