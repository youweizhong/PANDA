//! Top-level [`prove_final_pass`] driver.
//!
//! Drives the entire final-pass proof: commits every tensor, runs the
//! per-tensor range LogUps, the per-step backward/concretize/rescale
//! gadgets, the output-bound inequalities, the hidden-layer passes,
//! and the per-activation-layer relaxation-soundness gadgets. Helper
//! drivers live in [`helpers`].

pub(super) mod helpers;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_std::rand::RngCore;

use crate::quantized_crown::{quantized_backward_bound_with_trace_scaled, BoundDir};

use crate::crown::network::{ActivationKind, Layer};
use crate::snark::activation_gadget::sshape_endpoint::SshapeLineKind;
use crate::snark::activation_gadget::{
    prove_relu_d_boolean, prove_relu_lower_offset, prove_relu_upper_endpoint,
    prove_sshape_critical_point, prove_sshape_lower_at_lower, prove_sshape_lower_at_upper,
    prove_sshape_upper_at_lower, prove_sshape_upper_at_upper, ReluDBooleanProof,
    ReluLowerOffsetProof, ReluUpperEndpointProof, SshapeCriticalPointProof, SshapeEndpointProof,
};
use crate::snark::backward_pass::activation_matrix::build_activation_matrix_proofs;
use crate::snark::backward_pass::activation_step::build_activation_backward_proofs;
use crate::snark::backward_pass::bias_accumulator::{b_acc_n_vars, build_b_acc_step_proofs};
use crate::snark::backward_pass::chain_init::{chain_init_n_vars, prove_chain_init};
use crate::snark::backward_pass::linear_step::build_linear_backward_proofs;
use crate::snark::backward_pass::signed_components::driver::{
    build_relu_proof_concretize, build_relu_proofs_activation,
};
use crate::snark::commitment::commit::{absorb_commitments, commit_all_tensors};
use crate::snark::commitment::public_binding::prove_public_binding;
use crate::snark::concretization::concretize::build_concretize_proof;
use crate::snark::errors::SnarkError;
use crate::snark::params::{SnarkParams, SnarkStatement};
use crate::snark::proof::SnarkProof;
use crate::snark::rescaling::driver::build_rescale_proofs;
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use self::helpers::{
    build_layer_scales_commit, build_output_bound_inequality, build_tensor_range_proofs,
};

// Verifier reaches `absorb_layer_scales` via this re-export and must
// invoke it at the same FS-transcript position as the prover.
pub(super) use self::helpers::absorb_layer_scales;

/// Generate a final-pass SNARK proof for the statement. The caller
/// supplies the FS sponge (typically a fresh Merlin sponge seeded
/// with a session label) and a CSPRNG.
pub fn prove_final_pass(
    statement: &SnarkStatement,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut impl RngCore,
) -> Result<SnarkProof, SnarkError> {
    // Build the cert (float CROWN + quantization) and capture per-pass
    // linear-step traces.
    let nc_timing = crate::timing::scope("nc_total");
    let (cert, lower_trace, upper_trace, hidden_passes_trace) =
        quantized_backward_bound_with_trace_scaled(
            &statement.network,
            &statement.property,
            &statement.x_lower,
            &statement.x_upper,
            params.precision_bits,
            params.input_scale_log2,
            params.sigma_x_scale_log2,
            params.sigma_v_scale_log2,
        )
        .map_err(SnarkError::QCrown)?;
    drop(nc_timing);
    let _zk_timing = crate::timing::scope("zk_total");

    // Commit every tensor (incl. per-pass chain + step intermediates).
    let commit_timing = crate::timing::scope("trace_commit");
    let (commitments, prover_states) = commit_all_tensors(
        &cert,
        &statement.network,
        &params.committer_key,
        lower_trace.as_ref(),
        upper_trace.as_ref(),
        rng,
    )?;
    absorb_commitments(sponge, &commitments);

    // Per-layer scales: a single Hyrax commit over the packed
    // `[weight_c, weight_e, bias_c, bias_e, relax_d_c, relax_d_e,
    // relax_b_c, relax_b_e]` column, absorbed before any rescale
    // challenge fires. Each rescale event opens at the unit-vector
    // `(class, layer_idx)` index to bind its `(c, e)` source.
    let layer_scales = build_layer_scales_commit(&statement.network, &cert);
    let (layer_scales_packed_fr, layer_scales_n_vars) =
        self::helpers::pack_layer_scales_to_fr(&layer_scales);
    let (layer_scales_commit, layer_scales_state) =
        HyraxBn254::commit(&params.committer_key, &layer_scales_packed_fr, Some(rng))
            .map_err(SnarkError::Pcs)?;
    let layer_scales_pub = crate::snark::proof::LayerScalesHyraxCommit {
        n_layers: layer_scales.weight_c.len() as u32,
        n_vars: layer_scales_n_vars as u32,
        commit: layer_scales_commit.clone(),
    };
    absorb_layer_scales(sponge, &layer_scales_pub);
    drop(commit_timing);
    // Per-layer Hyrax opens of the bundled scales column. The verifier
    // replays the same loop and reconstructs a synthetic
    // `LayerScalesCommit` for downstream gadgets.
    let layer_scale_opens = self::helpers::build_layer_scale_opens(
        &statement.network,
        &layer_scales_packed_fr,
        &layer_scales_state,
        &layer_scales_commit,
        layer_scales_n_vars,
        layer_scales.weight_c.len(),
        params,
        sponge,
        rng,
    )?;

    // Per-tensor range LogUps: one per public-witness tensor (input
    // box, weights, biases, relaxation coefficients), each bound to
    // its committed tensor and multiplicities.
    let tensor_range_proofs = build_tensor_range_proofs(
        &statement.network,
        &commitments,
        &prover_states,
        params,
        sponge,
        rng,
    )?;

    let target_scale = cert.scales.working;

    // Per-linear-layer backward proofs.
    let linear_backward_lower = match lower_trace.as_ref() {
        Some(t) => Some(build_linear_backward_proofs(
            &cert,
            t,
            commitments.pass_lower.as_ref().expect("pass_lower commits"),
            prover_states
                .pass_lower
                .as_ref()
                .expect("pass_lower states"),
            &commitments,
            &prover_states,
            params,
            sponge,
            rng,
        )?),
        None => None,
    };
    let linear_backward_upper = match upper_trace.as_ref() {
        Some(t) => Some(build_linear_backward_proofs(
            &cert,
            t,
            commitments.pass_upper.as_ref().expect("pass_upper commits"),
            prover_states
                .pass_upper
                .as_ref()
                .expect("pass_upper states"),
            &commitments,
            &prover_states,
            params,
            sponge,
            rng,
        )?),
        None => None,
    };
    let activation_backward_lower = match lower_trace.as_ref() {
        Some(t) => Some(build_activation_backward_proofs(
            t,
            BoundDir::Lower,
            &cert.relaxations,
            commitments.pass_lower.as_ref().expect("pass_lower commits"),
            prover_states
                .pass_lower
                .as_ref()
                .expect("pass_lower states"),
            &commitments,
            &prover_states,
            params,
            sponge,
            rng,
        )?),
        None => None,
    };
    let activation_backward_upper = match upper_trace.as_ref() {
        Some(t) => Some(build_activation_backward_proofs(
            t,
            BoundDir::Upper,
            &cert.relaxations,
            commitments.pass_upper.as_ref().expect("pass_upper commits"),
            prover_states
                .pass_upper
                .as_ref()
                .expect("pass_upper states"),
            &commitments,
            &prover_states,
            params,
            sponge,
            rng,
        )?),
        None => None,
    };
    let concretize_lower = match lower_trace.as_ref().and_then(|t| t.concretize.as_ref()) {
        Some(c) => Some(build_concretize_proof(
            &cert,
            c,
            BoundDir::Lower,
            commitments.pass_lower.as_ref().expect("pass_lower commits"),
            prover_states
                .pass_lower
                .as_ref()
                .expect("pass_lower states"),
            &commitments,
            &prover_states,
            params,
            sponge,
            rng,
        )?),
        None => None,
    };
    let concretize_upper = match upper_trace.as_ref().and_then(|t| t.concretize.as_ref()) {
        Some(c) => Some(build_concretize_proof(
            &cert,
            c,
            BoundDir::Upper,
            commitments.pass_upper.as_ref().expect("pass_upper commits"),
            prover_states
                .pass_upper
                .as_ref()
                .expect("pass_upper states"),
            &commitments,
            &prover_states,
            params,
            sponge,
            rng,
        )?),
        None => None,
    };

    // ReLU-lookup proofs.
    let relu_lower_activation = match lower_trace.as_ref() {
        Some(t) => Some(build_relu_proofs_activation(
            t,
            commitments.pass_lower.as_ref().expect("pass_lower"),
            prover_states
                .pass_lower
                .as_ref()
                .expect("pass_lower states"),
            params,
            sponge,
            rng,
        )?),
        None => None,
    };
    let relu_upper_activation = match upper_trace.as_ref() {
        Some(t) => Some(build_relu_proofs_activation(
            t,
            commitments.pass_upper.as_ref().expect("pass_upper"),
            prover_states
                .pass_upper
                .as_ref()
                .expect("pass_upper states"),
            params,
            sponge,
            rng,
        )?),
        None => None,
    };
    let relu_lower_concretize = match lower_trace.as_ref().and_then(|t| t.concretize.as_ref()) {
        Some(c) => Some(build_relu_proof_concretize(
            c,
            commitments.pass_lower.as_ref().expect("pass_lower"),
            prover_states
                .pass_lower
                .as_ref()
                .expect("pass_lower states"),
            params,
            sponge,
            rng,
        )?),
        None => None,
    };
    let relu_upper_concretize = match upper_trace.as_ref().and_then(|t| t.concretize.as_ref()) {
        Some(c) => Some(build_relu_proof_concretize(
            c,
            commitments.pass_upper.as_ref().expect("pass_upper"),
            prover_states
                .pass_upper
                .as_ref()
                .expect("pass_upper states"),
            params,
            sponge,
            rng,
        )?),
        None => None,
    };

    // Per-pass rescale-gadget proofs.
    let rescale_lower = match lower_trace.as_ref() {
        Some(t) => Some(build_rescale_proofs(
            t,
            commitments.pass_lower.as_ref().expect("pass_lower"),
            prover_states
                .pass_lower
                .as_ref()
                .expect("pass_lower states"),
            BoundDir::Lower,
            params,
            sponge,
            rng,
        )?),
        None => None,
    };
    let rescale_upper = match upper_trace.as_ref() {
        Some(t) => Some(build_rescale_proofs(
            t,
            commitments.pass_upper.as_ref().expect("pass_upper"),
            prover_states
                .pass_upper
                .as_ref()
                .expect("pass_upper states"),
            BoundDir::Upper,
            params,
            sponge,
            rng,
        )?),
        None => None,
    };

    // Chain init / b_acc step / activation matrix path.
    let (a_n_vars, b_n_vars) =
        chain_init_n_vars(&statement.network.architecture(), &statement.property);
    let bacc_n_vars = b_acc_n_vars(&statement.property);

    let chain_init_lower = match lower_trace.as_ref() {
        Some(_) => Some(prove_chain_init(
            commitments.pass_lower.as_ref().expect("pass_lower"),
            prover_states
                .pass_lower
                .as_ref()
                .expect("pass_lower states"),
            &commitments,
            &prover_states.spec_c,
            &prover_states.spec_d,
            a_n_vars,
            b_n_vars,
            params,
            sponge,
            rng,
        )?),
        None => None,
    };
    let chain_init_upper = match upper_trace.as_ref() {
        Some(_) => Some(prove_chain_init(
            commitments.pass_upper.as_ref().expect("pass_upper"),
            prover_states
                .pass_upper
                .as_ref()
                .expect("pass_upper states"),
            &commitments,
            &prover_states.spec_c,
            &prover_states.spec_d,
            a_n_vars,
            b_n_vars,
            params,
            sponge,
            rng,
        )?),
        None => None,
    };

    let b_acc_step_lower = match lower_trace.as_ref() {
        Some(t) => Some(build_b_acc_step_proofs(
            t,
            commitments.pass_lower.as_ref().expect("pass_lower"),
            prover_states
                .pass_lower
                .as_ref()
                .expect("pass_lower states"),
            bacc_n_vars,
            params,
            sponge,
            rng,
        )?),
        None => None,
    };
    let b_acc_step_upper = match upper_trace.as_ref() {
        Some(t) => Some(build_b_acc_step_proofs(
            t,
            commitments.pass_upper.as_ref().expect("pass_upper"),
            prover_states
                .pass_upper
                .as_ref()
                .expect("pass_upper states"),
            bacc_n_vars,
            params,
            sponge,
            rng,
        )?),
        None => None,
    };

    let activation_matrix_lower = match lower_trace.as_ref() {
        Some(t) => Some(build_activation_matrix_proofs(
            t,
            BoundDir::Lower,
            &cert.relaxations,
            commitments.pass_lower.as_ref().expect("pass_lower"),
            prover_states
                .pass_lower
                .as_ref()
                .expect("pass_lower states"),
            &commitments,
            &prover_states,
            params,
            sponge,
            rng,
        )?),
        None => None,
    };
    let activation_matrix_upper = match upper_trace.as_ref() {
        Some(t) => Some(build_activation_matrix_proofs(
            t,
            BoundDir::Upper,
            &cert.relaxations,
            commitments.pass_upper.as_ref().expect("pass_upper"),
            prover_states
                .pass_upper
                .as_ref()
                .expect("pass_upper states"),
            &commitments,
            &prover_states,
            params,
            sponge,
            rng,
        )?),
        None => None,
    };

    // Public-statement binding for (spec_c, spec_d, x_lower, x_upper).
    let public_binding = Some(prove_public_binding(
        &statement.network,
        &statement.property,
        &statement.x_lower,
        &statement.x_upper,
        params.precision_bits,
        &commitments,
        &prover_states.spec_c,
        &prover_states.spec_d,
        &prover_states.x_lower,
        &prover_states.x_upper,
        params,
        sponge,
        rng,
    )?);

    // Per-pass output-bound inequality. Binds the private claimed
    // bound to `b_acc_final + acc_w` via slack + identity, and to the
    // public threshold via the in-SNARK property check.
    let n_spec = statement.property.n_spec();
    let n_vars_bound = crate::snark::commitment::commit::native_vector_n_vars(n_spec);
    let n_padded_bound = 1usize << n_vars_bound;
    let lower_t_real = statement.property.lower_threshold_or_zero();
    let upper_t_real = statement.property.upper_threshold_or_zero();
    let mut lower_threshold_codes = vec![0i128; n_padded_bound];
    for (slot, &v) in lower_threshold_codes.iter_mut().zip(lower_t_real.iter()) {
        *slot = crate::quantization::quantized_scalar::Qf::from_real(v, target_scale).code;
    }
    let mut upper_threshold_codes = vec![0i128; n_padded_bound];
    for (slot, &v) in upper_threshold_codes.iter_mut().zip(upper_t_real.iter()) {
        *slot = crate::quantization::quantized_scalar::Qf::from_real(v, target_scale).code;
    }
    let output_bound_lower = build_output_bound_inequality(
        &cert,
        &commitments,
        &prover_states,
        BoundDir::Lower,
        &lower_threshold_codes,
        params,
        sponge,
        rng,
    )?;
    let output_bound_upper = build_output_bound_inequality(
        &cert,
        &commitments,
        &prover_states,
        BoundDir::Upper,
        &upper_threshold_codes,
        params,
        sponge,
        rng,
    )?;

    // Hidden-layer preactivation bound proofs (one per hidden Linear
    // layer). Reuses the shared commitments; only chain and per-step
    // tensors are committed afresh per hidden pass.
    let (hidden_passes, hidden_pass_auxes) =
        crate::snark::backward_pass::hidden_pass::prove_hidden_passes(
            &cert,
            &hidden_passes_trace,
            &commitments,
            &prover_states,
            params,
            sponge,
            rng,
        )?;

    // Per-ReLU-layer relaxation-soundness: lower-line offset
    // `b_lower=0`, slope-discreteness `d_lower ∈ {0, s_d}`, and the
    // upper-line endpoint validity. Sigmoid/tanh skips these.
    let mut relu_lower_offset_proofs: Vec<ReluLowerOffsetProof> = Vec::new();
    let mut relu_d_boolean_proofs: Vec<ReluDBooleanProof> = Vec::new();
    let mut relu_upper_endpoint_proofs: Vec<ReluUpperEndpointProof> = Vec::new();
    for (layer_idx, layer) in statement.network.layers().iter().enumerate() {
        if let Layer::Activation {
            kind: ActivationKind::ReLU,
        } = layer
        {
            let relax_state = prover_states
                .relaxation
                .get(layer_idx)
                .and_then(|s| s.as_ref())
                .expect("ReLU activation layer has relaxation prover state");
            let relax_commit = commitments
                .relaxation
                .get(layer_idx)
                .and_then(|c| c.as_ref())
                .expect("ReLU activation layer has relaxation commit");

            // b_lower MLE ≡ 0
            let b_lower_proof = prove_relu_lower_offset(
                layer_idx,
                &relax_state.b_lower,
                &relax_commit.b_lower,
                params,
                sponge,
                rng,
            )?;
            relu_lower_offset_proofs.push(b_lower_proof);

            // d_lower[j] ∈ {0, s_d}; s_d is the code for real 1.0 in
            // the d-tensor scale.
            let d_scale = crate::quantization::scale::Scale {
                c: layer_scales.relax_d_c[layer_idx],
                e: layer_scales.relax_d_e[layer_idx],
            };
            let s_d_code = crate::quantization::quantized_scalar::Qf::from_real(1.0, d_scale).code;
            let s_d_fr = crate::snark_primitives::finite_field::signed_lift_to_fr(s_d_code);
            let d_lower_fr: Vec<ark_bn254::Fr> = relax_state.d_lower.0.clone();
            let d_proof = prove_relu_d_boolean(
                layer_idx,
                &d_lower_fr,
                &relax_state.d_lower,
                &relax_commit.d_lower,
                s_d_fr,
                params,
                sponge,
                rng,
            )?;
            relu_d_boolean_proofs.push(d_proof);

            // ReLU upper-line endpoint validity. Convexity then lifts
            // endpoint validity to the whole interval.
            let preceding_linear_idx = layer_idx
                .checked_sub(1)
                .expect("activation layer must be preceded by a linear layer");
            let hp_aux = hidden_pass_auxes
                .iter()
                .find(|a| a.target_layer_idx == preceding_linear_idx)
                .ok_or(SnarkError::ShapeMismatch {
                    what:
                        "relu_upper_endpoint: missing hidden-pass prover aux for preceding linear",
                })?;
            let n_vars_native = (hp_aux.preact_lower_aux.0.len().trailing_zeros()) as usize;

            let b_scale = crate::quantization::scale::Scale {
                c: layer_scales.relax_b_c[layer_idx],
                e: layer_scales.relax_b_e[layer_idx],
            };
            let upper_proof = prove_relu_upper_endpoint(
                layer_idx,
                n_vars_native,
                &hp_aux.preact_lower_aux,
                &hp_aux.preact_lower_commit,
                &hp_aux.preact_upper_aux,
                &hp_aux.preact_upper_commit,
                &relax_state.d_upper,
                &relax_commit.d_upper,
                &relax_state.b_upper,
                &relax_commit.b_upper,
                d_scale,
                target_scale,
                b_scale,
                params,
                sponge,
                rng,
            )?;
            relu_upper_endpoint_proofs.push(upper_proof);
        }
    }

    // Per-sigmoid/tanh-layer endpoint validity (four inequalities per
    // neuron). ReLU layers go through the convexity gadgets above.
    let mut sshape_upper_at_lower_proofs: Vec<SshapeEndpointProof> = Vec::new();
    let mut sshape_upper_at_upper_proofs: Vec<SshapeEndpointProof> = Vec::new();
    let mut sshape_lower_at_lower_proofs: Vec<SshapeEndpointProof> = Vec::new();
    let mut sshape_lower_at_upper_proofs: Vec<SshapeEndpointProof> = Vec::new();
    for (layer_idx, layer) in statement.network.layers().iter().enumerate() {
        if let Layer::Activation { kind } = layer {
            let kind = *kind;
            if !matches!(kind, ActivationKind::Sigmoid | ActivationKind::Tanh) {
                continue;
            }
            let relax_state = prover_states
                .relaxation
                .get(layer_idx)
                .and_then(|s| s.as_ref())
                .ok_or(SnarkError::ShapeMismatch {
                    what: "sshape Phase 3b: missing relaxation prover state for sigmoid/tanh layer",
                })?;
            let relax_commit = commitments
                .relaxation
                .get(layer_idx)
                .and_then(|c| c.as_ref())
                .ok_or(SnarkError::ShapeMismatch {
                    what: "sshape Phase 3b: missing relaxation commit for sigmoid/tanh layer",
                })?;
            let d_scale = crate::quantization::scale::Scale {
                c: layer_scales.relax_d_c[layer_idx],
                e: layer_scales.relax_d_e[layer_idx],
            };
            let b_scale = crate::quantization::scale::Scale {
                c: layer_scales.relax_b_c[layer_idx],
                e: layer_scales.relax_b_e[layer_idx],
            };
            // Preceding Linear layer's hidden pass holds the canonical
            // preact_lower / preact_upper at the working scale.
            let preceding_linear_idx =
                layer_idx
                    .checked_sub(1)
                    .ok_or(SnarkError::ArchitectureMismatch {
                        what: "sigmoid/tanh activation has no preceding Linear layer",
                    })?;
            let hp = hidden_passes_trace
                .iter()
                .find(|hp| hp.target_layer_idx == preceding_linear_idx)
                .ok_or(SnarkError::ShapeMismatch {
                    what: "sshape Phase 3b: no hidden pass for the Linear preceding sigmoid/tanh",
                })?;
            let hp_aux = hidden_pass_auxes
                .iter()
                .find(|a| a.target_layer_idx == preceding_linear_idx)
                .ok_or(SnarkError::ShapeMismatch {
                    what: "sshape Phase 3b: no hidden-pass prover aux for the preceding Linear",
                })?;
            let preact_l: Vec<i128> = hp.preact_lower.codes.iter().copied().collect();
            let preact_u: Vec<i128> = hp.preact_upper.codes.iter().copied().collect();
            let preact_l = preact_l.as_slice();
            let preact_u = preact_u.as_slice();
            let proof_ul = prove_sshape_upper_at_lower(
                layer_idx,
                kind,
                preact_l,
                &hp_aux.preact_lower_aux,
                &hp_aux.preact_lower_commit,
                &relax_state.d_upper,
                &relax_commit.d_upper,
                &relax_state.b_upper,
                &relax_commit.b_upper,
                d_scale,
                b_scale,
                target_scale,
                params,
                sponge,
                rng,
            )?;
            sshape_upper_at_lower_proofs.push(proof_ul);
            let proof_uu = prove_sshape_upper_at_upper(
                layer_idx,
                kind,
                preact_u,
                &hp_aux.preact_upper_aux,
                &hp_aux.preact_upper_commit,
                &relax_state.d_upper,
                &relax_commit.d_upper,
                &relax_state.b_upper,
                &relax_commit.b_upper,
                d_scale,
                b_scale,
                target_scale,
                params,
                sponge,
                rng,
            )?;
            sshape_upper_at_upper_proofs.push(proof_uu);
            let proof_ll = prove_sshape_lower_at_lower(
                layer_idx,
                kind,
                preact_l,
                &hp_aux.preact_lower_aux,
                &hp_aux.preact_lower_commit,
                &relax_state.d_lower,
                &relax_commit.d_lower,
                &relax_state.b_lower,
                &relax_commit.b_lower,
                d_scale,
                b_scale,
                target_scale,
                params,
                sponge,
                rng,
            )?;
            sshape_lower_at_lower_proofs.push(proof_ll);
            let proof_lu = prove_sshape_lower_at_upper(
                layer_idx,
                kind,
                preact_u,
                &hp_aux.preact_upper_aux,
                &hp_aux.preact_upper_commit,
                &relax_state.d_lower,
                &relax_commit.d_lower,
                &relax_state.b_lower,
                &relax_commit.b_lower,
                d_scale,
                b_scale,
                target_scale,
                params,
                sponge,
                rng,
            )?;
            sshape_lower_at_upper_proofs.push(proof_lu);
        }
    }

    // Per-sigmoid/tanh-layer critical-point proofs (FD slope-match +
    // conditional product); two per layer (upper line, lower line).
    let mut sshape_critical_point_upper_proofs: Vec<SshapeCriticalPointProof> = Vec::new();
    let mut sshape_critical_point_lower_proofs: Vec<SshapeCriticalPointProof> = Vec::new();
    for (layer_idx, layer) in statement.network.layers().iter().enumerate() {
        if let Layer::Activation { kind } = layer {
            let kind = *kind;
            if !matches!(kind, ActivationKind::Sigmoid | ActivationKind::Tanh) {
                continue;
            }
            let relax_state = prover_states
                .relaxation
                .get(layer_idx)
                .and_then(|s| s.as_ref())
                .ok_or(SnarkError::ShapeMismatch {
                    what: "sshape3c: missing relaxation prover state",
                })?;
            let relax_commit = commitments
                .relaxation
                .get(layer_idx)
                .and_then(|c| c.as_ref())
                .ok_or(SnarkError::ShapeMismatch {
                    what: "sshape3c: missing relaxation commit",
                })?;
            let d_scale = crate::quantization::scale::Scale {
                c: layer_scales.relax_d_c[layer_idx],
                e: layer_scales.relax_d_e[layer_idx],
            };
            let b_scale = crate::quantization::scale::Scale {
                c: layer_scales.relax_b_c[layer_idx],
                e: layer_scales.relax_b_e[layer_idx],
            };
            let preceding_linear_idx =
                layer_idx
                    .checked_sub(1)
                    .ok_or(SnarkError::ArchitectureMismatch {
                        what: "sigmoid/tanh activation has no preceding Linear",
                    })?;
            let hp = hidden_passes_trace
                .iter()
                .find(|hp| hp.target_layer_idx == preceding_linear_idx)
                .ok_or(SnarkError::ShapeMismatch {
                    what: "sshape3c: no hidden pass for preceding linear",
                })?;
            let hp_aux = hidden_pass_auxes
                .iter()
                .find(|a| a.target_layer_idx == preceding_linear_idx)
                .ok_or(SnarkError::ShapeMismatch {
                    what: "sshape3c: no hidden-pass prover aux for preceding linear",
                })?;
            let preact_l: Vec<i128> = hp.preact_lower.codes.iter().copied().collect();
            let preact_u: Vec<i128> = hp.preact_upper.codes.iter().copied().collect();
            let proof_u = prove_sshape_critical_point(
                layer_idx,
                kind,
                SshapeLineKind::Upper,
                &preact_l,
                &preact_u,
                &hp_aux.preact_lower_aux,
                &hp_aux.preact_lower_commit,
                &hp_aux.preact_upper_aux,
                &hp_aux.preact_upper_commit,
                &relax_state.d_upper,
                &relax_commit.d_upper,
                &relax_state.b_upper,
                &relax_commit.b_upper,
                d_scale,
                b_scale,
                target_scale,
                params,
                sponge,
                rng,
            )?;
            sshape_critical_point_upper_proofs.push(proof_u);
            let proof_l = prove_sshape_critical_point(
                layer_idx,
                kind,
                SshapeLineKind::Lower,
                &preact_l,
                &preact_u,
                &hp_aux.preact_lower_aux,
                &hp_aux.preact_lower_commit,
                &hp_aux.preact_upper_aux,
                &hp_aux.preact_upper_commit,
                &relax_state.d_lower,
                &relax_commit.d_lower,
                &relax_state.b_lower,
                &relax_commit.b_lower,
                d_scale,
                b_scale,
                target_scale,
                params,
                sponge,
                rng,
            )?;
            sshape_critical_point_lower_proofs.push(proof_l);
        }
    }

    Ok(SnarkProof {
        commitments,
        tensor_range_proofs,
        layer_scales: layer_scales_pub,
        layer_scale_opens,
        target_scale_c: target_scale.c,
        target_scale_e: target_scale.e,
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
        output_bound_lower,
        output_bound_upper,
        chain_init_lower,
        chain_init_upper,
        b_acc_step_lower,
        b_acc_step_upper,
        activation_matrix_lower,
        activation_matrix_upper,
        public_binding,
        hidden_passes,
        relu_lower_offset_proofs,
        relu_d_boolean_proofs,
        relu_upper_endpoint_proofs,
        sshape_upper_at_lower_proofs,
        sshape_upper_at_upper_proofs,
        sshape_lower_at_lower_proofs,
        sshape_lower_at_upper_proofs,
        sshape_critical_point_upper_proofs,
        sshape_critical_point_lower_proofs,
    })
}
