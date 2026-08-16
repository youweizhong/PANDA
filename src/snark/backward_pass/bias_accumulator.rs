//! Per-step `b_acc` chain-update binding.
//!
//! Ties the running bias accumulator together across the backward
//! pass. At every linear and activation step the gadget proves
//! `chain_b_acc[layer] = chain_b_acc[layer + 1] + delta`, where
//! `delta` is the post-rescale per-step contribution
//! (`linear_prod_w` for linear steps, `activation_bias_delta` for
//! activation steps). Each step squeezes one FS point and emits a
//! single batched Hyrax open over the three operands followed by a
//! linear identity check on the opened evals.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;

use crate::crown::network::{LayerShape, NetworkArchitecture};
use crate::crown::output_property::Property;
use crate::quantized_crown::BackwardTrace;
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use crate::snark::commitment::commit::{native_vector_n_vars, PassCommitments, PassProverStates};
use crate::snark::commitment::pcs_helpers::{
    hyrax_open_batched_at, hyrax_verify_batched_at, BatchOpenSpec, BatchVerifySpec,
};
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;

/// Step-kind tag for the canonical event ordering.
#[derive(Clone, Copy, Debug)]
pub enum BAccStepKind {
    Linear,
    Activation,
}

/// One step's `b_acc`-binding proof.
///
/// The three operands (`b_new`, `b_old`, `delta`) share the same FS
/// point `r` and commit `n_vars`, so they ride a single Hyrax batched
/// open. The claimed evals stay public so the verifier can re-check
/// `b_new(r) = b_old(r) + delta(r)`.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BAccStepProof {
    pub kind: u64, // 0 = Linear, 1 = Activation
    pub layer_idx: usize,
    /// Step index within its kind (linear-step or activation-step).
    pub step_idx: usize,
    pub n_vars: usize,
    pub r: Vec<Fr>,
    pub batched_open: <HyraxBn254 as MlPcs>::Proof,
    pub b_new_eval: Fr,
    pub b_old_eval: Fr,
    pub delta_eval: Fr,
}

fn n_vars_for_b_acc(n_spec: usize) -> usize {
    // Match the native commit `n_vars` of `commit_vector` so the FS
    // points align with the committed tensor sizes.
    native_vector_n_vars(n_spec)
}

/// Walk the trace in network-backward order (matching the rescale
/// gadget's canonical ordering) and emit one `b_acc`-binding proof
/// per step.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_b_acc_step_proofs(
    trace: &BackwardTrace,
    pass_com: &PassCommitments,
    pass_st: &PassProverStates,
    n_vars: usize,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<Vec<BAccStepProof>, SnarkError> {
    let _timing = crate::timing::scope("b_acc");
    let mut events: Vec<BAccStepProof> = Vec::new();

    enum StepRef<'a> {
        Linear(usize, &'a crate::quantized_crown::LinearStepTrace),
        Activation(usize, &'a crate::quantized_crown::ActivationStepTrace),
    }
    let mut steps: Vec<StepRef<'_>> =
        Vec::with_capacity(trace.linear_steps.len() + trace.activation_steps.len());
    for (i, s) in trace.linear_steps.iter().enumerate() {
        steps.push(StepRef::Linear(i, s));
    }
    for (i, s) in trace.activation_steps.iter().enumerate() {
        steps.push(StepRef::Activation(i, s));
    }
    steps.sort_by_key(|s| {
        std::cmp::Reverse(match s {
            StepRef::Linear(_, st) => st.layer_idx,
            StepRef::Activation(_, st) => st.layer_idx,
        })
    });

    for step_ref in steps.into_iter() {
        let (kind, layer_idx, step_idx, delta_aux, delta_com) = match step_ref {
            StepRef::Linear(idx, st) => (
                BAccStepKind::Linear,
                st.layer_idx,
                idx,
                &pass_st.linear_prod_w[idx],
                &pass_com.linear_prod_w[idx],
            ),
            StepRef::Activation(idx, st) => (
                BAccStepKind::Activation,
                st.layer_idx,
                idx,
                &pass_st.activation_bias_delta[idx],
                &pass_com.activation_bias_delta[idx],
            ),
        };
        let b_new_aux = &pass_st.chain_b_acc[layer_idx];
        let b_new_com = &pass_com.chain_b_acc[layer_idx];
        let b_old_aux = &pass_st.chain_b_acc[layer_idx + 1];
        let b_old_com = &pass_com.chain_b_acc[layer_idx + 1];

        sponge.absorb(&(kind as u64));
        sponge.absorb(&(layer_idx as u64));
        sponge.absorb(&(step_idx as u64));
        sponge.absorb(&(n_vars as u64));
        let r: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(n_vars);

        // All three commits share the same n_vars, so batch them.
        let items = [
            BatchOpenSpec {
                aux: b_new_aux,
                commitment: b_new_com,
                commit_n_vars: n_vars,
            },
            BatchOpenSpec {
                aux: b_old_aux,
                commitment: b_old_com,
                commit_n_vars: n_vars,
            },
            BatchOpenSpec {
                aux: delta_aux,
                commitment: delta_com,
                commit_n_vars: n_vars,
            },
        ];
        let (vals, batched_open) =
            hyrax_open_batched_at(&params.committer_key, &items, &r, sponge, rng)?;
        let b_new_eval = vals[0];
        let b_old_eval = vals[1];
        let delta_eval = vals[2];
        debug_assert_eq!(
            b_new_eval,
            b_old_eval + delta_eval,
            "b_acc_step: b_new(r) must equal b_old(r) + delta(r)"
        );

        events.push(BAccStepProof {
            kind: kind as u64,
            layer_idx,
            step_idx,
            n_vars,
            r,
            batched_open,
            b_new_eval,
            b_old_eval,
            delta_eval,
        });
    }
    Ok(events)
}

/// Replay the canonical event ordering from the public architecture
/// and verify each step proof.
pub(crate) fn verify_b_acc_step_proofs(
    proofs: &[BAccStepProof],
    pass_com: &PassCommitments,
    arch: &NetworkArchitecture,
    n_vars: usize,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    let layers = arch.layers();
    let mut linear_step_idx = 0usize;
    let mut act_step_idx = 0usize;
    let mut expected: Vec<(BAccStepKind, usize, usize)> = Vec::new();
    for i in (0..layers.len()).rev() {
        match &layers[i] {
            LayerShape::Linear { .. } => {
                expected.push((BAccStepKind::Linear, i, linear_step_idx));
                linear_step_idx += 1;
            }
            LayerShape::Activation { .. } => {
                expected.push((BAccStepKind::Activation, i, act_step_idx));
                act_step_idx += 1;
            }
        }
    }
    if expected.len() != proofs.len() {
        return Err(SnarkError::ShapeMismatch {
            what: "b_acc_step: count mismatch",
        });
    }

    for ((expected_kind, expected_layer, expected_step), proof) in
        expected.iter().zip(proofs.iter())
    {
        let kind_u = *expected_kind as u64;
        if proof.kind != kind_u
            || proof.layer_idx != *expected_layer
            || proof.step_idx != *expected_step
            || proof.n_vars != n_vars
        {
            return Err(SnarkError::ShapeMismatch {
                what: "b_acc_step: kind/layer/step/n_vars mismatch",
            });
        }
        let delta_com: &<HyraxBn254 as MlPcs>::Commitment = match expected_kind {
            BAccStepKind::Linear => &pass_com.linear_prod_w[*expected_step],
            BAccStepKind::Activation => &pass_com.activation_bias_delta[*expected_step],
        };
        let b_new_com = &pass_com.chain_b_acc[*expected_layer];
        let b_old_com = &pass_com.chain_b_acc[*expected_layer + 1];

        sponge.absorb(&kind_u);
        sponge.absorb(&(*expected_layer as u64));
        sponge.absorb(&(*expected_step as u64));
        sponge.absorb(&(n_vars as u64));
        let r: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(n_vars);
        if r != proof.r {
            return Err(SnarkError::TranscriptMismatch);
        }

        let cnv = n_vars;
        let items = [
            BatchVerifySpec {
                commitment: b_new_com,
                value: proof.b_new_eval,
                commit_n_vars: cnv,
            },
            BatchVerifySpec {
                commitment: b_old_com,
                value: proof.b_old_eval,
                commit_n_vars: cnv,
            },
            BatchVerifySpec {
                commitment: delta_com,
                value: proof.delta_eval,
                commit_n_vars: cnv,
            },
        ];
        let ok = hyrax_verify_batched_at(
            &params.verifier_key,
            &items,
            &proof.r,
            &proof.batched_open,
            sponge,
        )?;
        if !ok {
            return Err(SnarkError::PcsOpenRejected {
                which: "b_acc_step_batched",
            });
        }

        if proof.b_new_eval != proof.b_old_eval + proof.delta_eval {
            return Err(SnarkError::BAccStepBindingFailed {
                layer: *expected_layer,
            });
        }
    }
    Ok(())
}

/// Standard `n_vars` for the `b_acc` layout (`log2(n_spec)`, ≥ 1).
pub(crate) fn b_acc_n_vars(property: &Property) -> usize {
    n_vars_for_b_acc(property.c_matrix.nrows())
}
