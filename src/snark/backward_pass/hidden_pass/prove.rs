//! Hidden-layer pass prover.
//!
//! Walks the hidden Linear layers in forward order and emits one
//! [`HiddenLayerPassProof`] per layer. Shared public-witness tensors
//! (weights, biases, relaxations, input box) are reused from the
//! final pass's `TensorCommitments`; only chain tensors and per-step
//! intermediates are committed fresh.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_serialize::CanonicalSerialize;
use ark_std::rand::RngCore;

use crate::quantized_crown::{HiddenLayerPass, QuantCert};
use crate::snark_primitives::finite_field::signed_lift_to_fr;
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

/// Absorb a Hyrax commitment into the FS sponge by canonical
/// serialization.
fn absorb_chain_commit(
    sponge: &mut impl CryptographicSponge,
    commitment: &<HyraxBn254 as MlPcs>::Commitment,
) {
    let mut buf = Vec::new();
    commitment
        .serialize_compressed(&mut buf)
        .expect("serialize commitment");
    sponge.absorb(&buf);
}

use super::{absorb_hidden_pass_tag, identity_mle_eval};
use crate::quantized_crown::BoundDir;
use crate::snark::backward_pass::activation_matrix::build_activation_matrix_proofs;
use crate::snark::backward_pass::activation_step::build_activation_backward_proofs;
use crate::snark::backward_pass::bias_accumulator::build_b_acc_step_proofs;
use crate::snark::backward_pass::linear_step::build_linear_backward_proofs;
use crate::snark::backward_pass::signed_components::driver::{
    build_relu_proof_concretize, build_relu_proofs_activation,
};
use crate::snark::commitment::commit::{
    commit_pass, native_vector_n_vars, ProverPolyStates, TensorCommitments,
};
use crate::snark::commitment::pcs_helpers::hyrax_open_at;
use crate::snark::concretization::concretize::build_concretize_proof;
use crate::snark::errors::SnarkError;
use crate::snark::output_bound::{prove_output_bound_inequality, OutputBoundIneqProof};
use crate::snark::params::SnarkParams;
use crate::snark::proof::{ChainInitFromIdentityProof, HiddenLayerPassProof};
use crate::snark::rescaling::driver::build_rescale_proofs;

/// Prover-side aux exported from a single hidden pass.
///
/// Holds the Hyrax aux and commit for the per-pass preact bounds so
/// downstream relaxation-soundness gadgets can open the same commits
/// at their own FS-derived points without re-receiving raw codes.
pub(crate) struct HiddenPassProverAux {
    pub target_layer_idx: usize,
    pub preact_lower_aux: crate::snark::commitment::commit::CommittedAux,
    pub preact_upper_aux: crate::snark::commitment::commit::CommittedAux,
    pub preact_lower_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub preact_upper_commit: <HyraxBn254 as MlPcs>::Commitment,
}

/// Build one [`HiddenLayerPassProof`] per hidden Linear layer.
///
/// Returns the proofs alongside per-pass prover aux (preact aux +
/// commits) so downstream gadgets can open the preact commits at
/// their own FS points.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_hidden_passes(
    cert: &QuantCert,
    hidden_passes_trace: &[HiddenLayerPass],
    commitments: &TensorCommitments,
    prover_states: &ProverPolyStates,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut impl RngCore,
) -> Result<(Vec<HiddenLayerPassProof>, Vec<HiddenPassProverAux>), SnarkError> {
    let mut out: Vec<HiddenLayerPassProof> = Vec::with_capacity(hidden_passes_trace.len());
    let mut auxes: Vec<HiddenPassProverAux> = Vec::with_capacity(hidden_passes_trace.len());
    for hp in hidden_passes_trace {
        let (proof, aux) =
            prove_one_hidden_pass(cert, hp, commitments, prover_states, params, sponge, rng)?;
        out.push(proof);
        auxes.push(aux);
    }
    Ok((out, auxes))
}

#[allow(clippy::too_many_arguments)]
fn prove_one_hidden_pass(
    cert: &QuantCert,
    hp: &HiddenLayerPass,
    commitments: &TensorCommitments,
    prover_states: &ProverPolyStates,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut impl RngCore,
) -> Result<(HiddenLayerPassProof, HiddenPassProverAux), SnarkError> {
    absorb_hidden_pass_tag(sponge, hp.target_layer_idx, hp.n_spec);

    // Commit per-pass preactivation bounds and absorb the commits
    // into the FS sponge. The preact values are private witnesses;
    // downstream relaxation-soundness gadgets open these commits at
    // their own FS-derived points to bind their algebraic identities.
    let commit_timing = crate::timing::scope("trace_commit");
    let preact_lower_codes_v: Vec<Fr> = hp
        .preact_lower
        .codes
        .iter()
        .map(|&c| signed_lift_to_fr(c))
        .collect();
    let preact_upper_codes_v: Vec<Fr> = hp
        .preact_upper
        .codes
        .iter()
        .map(|&c| signed_lift_to_fr(c))
        .collect();
    let n_native_pa = preact_lower_codes_v.len().max(2);
    let mut preact_n_vars_pa = n_native_pa.next_power_of_two().trailing_zeros() as usize;
    if preact_n_vars_pa % 2 == 1 {
        preact_n_vars_pa += 1;
    }
    if preact_n_vars_pa < 2 {
        preact_n_vars_pa = 2;
    }
    let padded_len_pa = 1usize << preact_n_vars_pa;
    let mut preact_lo_padded_pa = preact_lower_codes_v;
    preact_lo_padded_pa.resize(padded_len_pa, ark_bn254::Fr::from(0u64));
    let mut preact_hi_padded_pa = preact_upper_codes_v;
    preact_hi_padded_pa.resize(padded_len_pa, ark_bn254::Fr::from(0u64));
    let (preact_lower_commit_pa, preact_lower_state_pa) =
        crate::snark_primitives::polynomial_commitment::HyraxBn254::commit(
            &params.committer_key,
            &preact_lo_padded_pa,
            Some(rng),
        )
        .map_err(crate::snark::errors::SnarkError::Pcs)?;
    let (preact_upper_commit_pa, preact_upper_state_pa) =
        crate::snark_primitives::polynomial_commitment::HyraxBn254::commit(
            &params.committer_key,
            &preact_hi_padded_pa,
            Some(rng),
        )
        .map_err(crate::snark::errors::SnarkError::Pcs)?;
    crate::snark::rescaling::absorb_commitment(sponge, &preact_lower_commit_pa);
    crate::snark::rescaling::absorb_commitment(sponge, &preact_upper_commit_pa);
    let preact_lower_aux_pa: crate::snark::commitment::commit::CommittedAux =
        (preact_lo_padded_pa, preact_lower_state_pa);
    let preact_upper_aux_pa: crate::snark::commitment::commit::CommittedAux =
        (preact_hi_padded_pa, preact_upper_state_pa);

    // `commit_pass` sizes chain_a / chain_b_acc from the trace's
    // step counts (target+2 entries: one per walked layer plus the
    // initial identity slot at target+1).
    let (pass_lower_com, pass_lower_st) = commit_pass(&hp.lower_trace, &params.committer_key, rng)?;
    let (pass_upper_com, pass_upper_st) = commit_pass(&hp.upper_trace, &params.committer_key, rng)?;
    drop(commit_timing);

    // Pin chain_a[target+1] = identity and chain_b_acc[target+1] = 0
    // for both directions; the verifier re-derives the canonical MLE
    // evaluations and rejects on mismatch.
    let chain_init_timing = crate::timing::scope("chain_init");
    let chain_init_lower = prove_chain_init_from_identity(
        &pass_lower_com,
        &pass_lower_st,
        hp.target_layer_idx,
        hp.n_spec,
        cert.scales.spec,
        params,
        sponge,
        rng,
    )?;
    let chain_init_upper = prove_chain_init_from_identity(
        &pass_upper_com,
        &pass_upper_st,
        hp.target_layer_idx,
        hp.n_spec,
        cert.scales.spec,
        params,
        sponge,
        rng,
    )?;
    drop(chain_init_timing);

    // Per-step proofs in both directions. The shared gadgets index
    // `pass_com` / `pass_st` for chain commits and `commitments` for
    // weights / biases / relaxations, so they pick up the shorter
    // chain automatically.
    let linear_backward_lower = build_linear_backward_proofs(
        cert,
        &hp.lower_trace,
        &pass_lower_com,
        &pass_lower_st,
        commitments,
        prover_states,
        params,
        sponge,
        rng,
    )?;
    let linear_backward_upper = build_linear_backward_proofs(
        cert,
        &hp.upper_trace,
        &pass_upper_com,
        &pass_upper_st,
        commitments,
        prover_states,
        params,
        sponge,
        rng,
    )?;
    let activation_backward_lower = build_activation_backward_proofs(
        &hp.lower_trace,
        BoundDir::Lower,
        &cert.relaxations,
        &pass_lower_com,
        &pass_lower_st,
        commitments,
        prover_states,
        params,
        sponge,
        rng,
    )?;
    let activation_backward_upper = build_activation_backward_proofs(
        &hp.upper_trace,
        BoundDir::Upper,
        &cert.relaxations,
        &pass_upper_com,
        &pass_upper_st,
        commitments,
        prover_states,
        params,
        sponge,
        rng,
    )?;

    let concretize_lower = build_concretize_proof(
        cert,
        hp.lower_trace
            .concretize
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "hidden pass: missing lower concretize trace",
            })?,
        BoundDir::Lower,
        &pass_lower_com,
        &pass_lower_st,
        commitments,
        prover_states,
        params,
        sponge,
        rng,
    )?;
    let concretize_upper = build_concretize_proof(
        cert,
        hp.upper_trace
            .concretize
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "hidden pass: missing upper concretize trace",
            })?,
        BoundDir::Upper,
        &pass_upper_com,
        &pass_upper_st,
        commitments,
        prover_states,
        params,
        sponge,
        rng,
    )?;

    let relu_lower_activation = build_relu_proofs_activation(
        &hp.lower_trace,
        &pass_lower_com,
        &pass_lower_st,
        params,
        sponge,
        rng,
    )?;
    let relu_upper_activation = build_relu_proofs_activation(
        &hp.upper_trace,
        &pass_upper_com,
        &pass_upper_st,
        params,
        sponge,
        rng,
    )?;
    let relu_lower_concretize = build_relu_proof_concretize(
        hp.lower_trace.concretize.as_ref().expect("checked above"),
        &pass_lower_com,
        &pass_lower_st,
        params,
        sponge,
        rng,
    )?;
    let relu_upper_concretize = build_relu_proof_concretize(
        hp.upper_trace.concretize.as_ref().expect("checked above"),
        &pass_upper_com,
        &pass_upper_st,
        params,
        sponge,
        rng,
    )?;

    let rescale_lower = build_rescale_proofs(
        &hp.lower_trace,
        &pass_lower_com,
        &pass_lower_st,
        crate::quantized_crown::BoundDir::Lower,
        params,
        sponge,
        rng,
    )?;
    let rescale_upper = build_rescale_proofs(
        &hp.upper_trace,
        &pass_upper_com,
        &pass_upper_st,
        crate::quantized_crown::BoundDir::Upper,
        params,
        sponge,
        rng,
    )?;

    // Hidden-pass n_spec is the target layer's output dim; the b_acc
    // helpers normally take a `Property`, so we synthesize one with
    // n_spec rows just to size the layout.
    let bacc_n_vars = native_vector_n_vars(hp.n_spec);

    let b_acc_step_lower = build_b_acc_step_proofs(
        &hp.lower_trace,
        &pass_lower_com,
        &pass_lower_st,
        bacc_n_vars,
        params,
        sponge,
        rng,
    )?;
    let b_acc_step_upper = build_b_acc_step_proofs(
        &hp.upper_trace,
        &pass_upper_com,
        &pass_upper_st,
        bacc_n_vars,
        params,
        sponge,
        rng,
    )?;

    let activation_matrix_lower = build_activation_matrix_proofs(
        &hp.lower_trace,
        BoundDir::Lower,
        &cert.relaxations,
        &pass_lower_com,
        &pass_lower_st,
        commitments,
        prover_states,
        params,
        sponge,
        rng,
    )?;
    let activation_matrix_upper = build_activation_matrix_proofs(
        &hp.upper_trace,
        BoundDir::Upper,
        &cert.relaxations,
        &pass_upper_com,
        &pass_upper_st,
        commitments,
        prover_states,
        params,
        sponge,
        rng,
    )?;

    // Output-bound inequality: `preact_<dir>` against
    // `b_acc_final + acc_w` via a slack range LogUp. The claimed
    // witness reuses the per-pass `preact_*_commit` so the same MLE
    // is bound here and in every downstream relaxation gadget.
    let bound_n_vars = native_vector_n_vars(hp.n_spec);
    let bound_n_padded = 1usize << bound_n_vars;
    debug_assert_eq!(bound_n_vars, preact_n_vars_pa);
    let output_bound_lower = build_hidden_output_bound(
        &hp.preact_lower,
        BoundDir::Lower,
        bound_n_vars,
        bound_n_padded,
        &pass_lower_com,
        &pass_lower_st,
        &preact_lower_aux_pa,
        &preact_lower_commit_pa,
        params,
        sponge,
        rng,
    )?;
    let output_bound_upper = build_hidden_output_bound(
        &hp.preact_upper,
        BoundDir::Upper,
        bound_n_vars,
        bound_n_padded,
        &pass_upper_com,
        &pass_upper_st,
        &preact_upper_aux_pa,
        &preact_upper_commit_pa,
        params,
        sponge,
        rng,
    )?;

    let aux = HiddenPassProverAux {
        target_layer_idx: hp.target_layer_idx,
        preact_lower_aux: preact_lower_aux_pa,
        preact_upper_aux: preact_upper_aux_pa,
        preact_lower_commit: preact_lower_commit_pa.clone(),
        preact_upper_commit: preact_upper_commit_pa.clone(),
    };
    Ok((
        HiddenLayerPassProof {
            target_layer_idx: hp.target_layer_idx,
            n_spec: hp.n_spec,
            pass_lower: pass_lower_com,
            pass_upper: pass_upper_com,
            preact_lower_commit: preact_lower_commit_pa,
            preact_upper_commit: preact_upper_commit_pa,
            preact_n_vars: preact_n_vars_pa as u32,
            chain_init_lower,
            chain_init_upper,
            linear_backward_lower,
            linear_backward_upper,
            activation_backward_lower,
            activation_backward_upper,
            concretize_lower,
            concretize_upper,
            relu_lower_activation,
            relu_upper_activation,
            relu_lower_concretize,
            relu_upper_concretize,
            rescale_lower,
            rescale_upper,
            b_acc_step_lower,
            b_acc_step_upper,
            activation_matrix_lower,
            activation_matrix_upper,
            output_bound_lower,
            output_bound_upper,
        },
        aux,
    ))
}

/// Pin `chain_a[target + 1]` to the canonical quantized identity
/// matrix and `chain_b_acc[target + 1]` to zero, via Hyrax opens at
/// FS-derived points.
#[allow(clippy::too_many_arguments)]
fn prove_chain_init_from_identity(
    pass_com: &crate::snark::commitment::commit::PassCommitments,
    pass_st: &crate::snark::commitment::commit::PassProverStates,
    target_layer_idx: usize,
    n_spec: usize,
    spec_scale: crate::quantization::scale::Scale,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<ChainInitFromIdentityProof, SnarkError> {
    let chain_top = target_layer_idx + 1;
    let chain_a_com = pass_com
        .chain_a
        .get(chain_top)
        .ok_or(SnarkError::ShapeMismatch {
            what: "hidden pass: chain_a out-of-range at target+1",
        })?;
    let chain_a_aux = pass_st
        .chain_a
        .get(chain_top)
        .ok_or(SnarkError::ShapeMismatch {
            what: "hidden pass: chain_a state out-of-range at target+1",
        })?;
    let chain_b_com = pass_com
        .chain_b_acc
        .get(chain_top)
        .ok_or(SnarkError::ShapeMismatch {
            what: "hidden pass: chain_b_acc out-of-range at target+1",
        })?;
    let chain_b_aux = pass_st
        .chain_b_acc
        .get(chain_top)
        .ok_or(SnarkError::ShapeMismatch {
            what: "hidden pass: chain_b_acc state out-of-range at target+1",
        })?;

    let a_n_vars = crate::snark::commitment::commit::native_matrix_n_vars(n_spec, n_spec);
    let b_n_vars = native_vector_n_vars(n_spec);

    // Bind chain commits before squeezing the FS points; otherwise a
    // prover could craft chain_a / chain_b_acc to match identity /
    // zero only at the pre-known FS points.
    absorb_chain_commit(sponge, chain_a_com);
    absorb_chain_commit(sponge, chain_b_com);
    sponge.absorb(&(a_n_vars as u64));
    sponge.absorb(&(b_n_vars as u64));
    let r_a: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(a_n_vars);
    let r_b: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(b_n_vars);

    let (chain_a_eval, chain_a_open) = hyrax_open_at(
        &params.committer_key,
        chain_a_aux,
        chain_a_com,
        &r_a,
        sponge,
        rng,
    )?;
    let (chain_b_eval, chain_b_open) = hyrax_open_at(
        &params.committer_key,
        chain_b_aux,
        chain_b_com,
        &r_b,
        sponge,
        rng,
    )?;

    debug_assert_eq!(
        chain_a_eval,
        identity_mle_eval(&r_a, n_spec, spec_scale),
        "hidden pass chain_init: chain_a[L] MLE eval must equal canonical identity at r_a"
    );
    debug_assert_eq!(
        chain_b_eval,
        Fr::from(0u64),
        "hidden pass chain_init: chain_b_acc[L] MLE eval must equal 0 at r_b"
    );

    Ok(ChainInitFromIdentityProof {
        a_n_vars,
        b_n_vars,
        r_a,
        r_b,
        chain_a_eval,
        chain_b_eval,
        chain_a_open,
        chain_b_open,
    })
}

/// Run `prove_output_bound_inequality` for one hidden-pass
/// direction. Threads the per-pass preact aux/commit as the claimed
/// witness so the inequality is proven against the exact MLE that
/// downstream relaxation gadgets consume.
#[allow(clippy::too_many_arguments)]
fn build_hidden_output_bound(
    preact: &crate::quantization::quantized_array::QArray1,
    direction: BoundDir,
    n_vars: usize,
    n_padded: usize,
    pass_com: &crate::snark::commitment::commit::PassCommitments,
    pass_st: &crate::snark::commitment::commit::PassProverStates,
    preact_aux: &crate::snark::commitment::commit::CommittedAux,
    preact_commit: &<crate::snark_primitives::polynomial_commitment::HyraxBn254 as crate::snark_primitives::polynomial_commitment::MlPcs>::Commitment,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut impl RngCore,
) -> Result<OutputBoundIneqProof, SnarkError> {
    let _timing = crate::timing::scope("ob_hidden");
    let acc_w_aux = pass_st
        .concretize_acc_w
        .as_ref()
        .ok_or(SnarkError::ShapeMismatch {
            what: "hidden pass: missing concretize_acc_w state",
        })?;
    let acc_w_com = pass_com
        .concretize_acc_w
        .as_ref()
        .ok_or(SnarkError::ShapeMismatch {
            what: "hidden pass: missing concretize_acc_w commit",
        })?;
    let b_acc_final_aux = pass_st
        .chain_b_acc
        .first()
        .expect("hidden pass: chain_b_acc non-empty");
    let b_acc_final_com = pass_com
        .chain_b_acc
        .first()
        .expect("hidden pass: chain_b_acc non-empty");

    let mut claimed_codes = vec![0i128; n_padded];
    for (slot, &c) in claimed_codes.iter_mut().zip(preact.codes.iter()) {
        *slot = c;
    }
    let b_acc_final_codes: Vec<i128> = b_acc_final_aux
        .0
        .iter()
        .take(n_padded)
        .map(|f| crate::snark_primitives::finite_field::fr_to_signed_i128(*f).unwrap_or(0))
        .collect();
    let acc_w_codes: Vec<i128> = acc_w_aux
        .0
        .iter()
        .take(n_padded)
        .map(|f| crate::snark_primitives::finite_field::fr_to_signed_i128(*f).unwrap_or(0))
        .collect();

    // Hidden-pass preact bounds are private witnesses, not a public
    // threshold; pass `None` so the gadget skips the property check.
    // Preact-bound slacks are per-neuron-scale quantities, so this
    // runs at the narrow gadget budget — the wide out-bound window is
    // reserved for the final pass.
    prove_output_bound_inequality(
        direction,
        params.gadget_range_bits,
        n_vars,
        &claimed_codes,
        &b_acc_final_codes,
        &acc_w_codes,
        None,
        b_acc_final_aux,
        b_acc_final_com,
        acc_w_aux,
        acc_w_com,
        Some((preact_aux, preact_commit)),
        params,
        sponge,
        rng,
    )
}
