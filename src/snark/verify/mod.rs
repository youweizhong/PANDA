//! Top-level [`verify_final_pass`] driver.
//!
//! Single public entry point for the final-pass verifier. Replays the
//! prover's FS transcript and walks per-step backward / concretize /
//! rescale gadgets plus the per-activation-layer relaxation-soundness
//! gadgets. Pre-flight shape and architecture checks live in
//! [`checks`].

// `checks` is `pub(super)` so the hidden-pass verifier (sibling
// under `crate::snark`) can call `check_target_codes_in_range` on
// its still-public `preact_<dir>_codes`.
pub(super) mod checks;

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;

use crate::quantized_crown::BoundDir;

use crate::crown::network::{ActivationKind, LayerShape};
use crate::snark::activation_gadget::sshape_endpoint::SshapeLineKind;
use crate::snark::activation_gadget::{
    verify_relu_d_boolean, verify_relu_lower_offset, verify_relu_upper_endpoint,
    verify_sshape_critical_point, verify_sshape_lower_at_lower, verify_sshape_lower_at_upper,
    verify_sshape_upper_at_lower, verify_sshape_upper_at_upper,
};
use crate::snark::backward_pass::activation_matrix::verify_activation_matrix_chain;
use crate::snark::backward_pass::activation_step::verify_activation_backward_chain;
use crate::snark::backward_pass::bias_accumulator::{b_acc_n_vars, verify_b_acc_step_proofs};
use crate::snark::backward_pass::chain_init::{chain_init_n_vars, verify_chain_init};
use crate::snark::backward_pass::linear_step::verify_linear_backward_chain;
use crate::snark::backward_pass::signed_components::driver::{
    verify_relu_proof_concretize, verify_relu_proofs_activation,
};
use crate::snark::commitment::architecture::{
    check_activation_chain_shape, check_linear_chain_shape, check_pass_commit_lengths,
};
use crate::snark::commitment::commit::{absorb_commitments, native_vector_n_vars};
use crate::snark::commitment::public_binding::verify_public_binding;
use crate::snark::concretization::concretize::verify_concretize_proof;
use crate::snark::errors::SnarkError;
use crate::snark::output_bound::verify_output_bound_inequality;
use crate::snark::params::{SnarkParams, SnarkVerifierStatement, VerifiedBound};
use crate::snark::proof::SnarkProof;
use crate::snark::rescaling::driver::verify_rescale_proofs;

use self::checks::{
    check_layer_scales_shape, require_mandatory_components, verify_tensor_range_proofs,
};

/// Verify a final-pass proof against the public statement. Returns
/// `VerifiedBound` with both directions `None` (the bound is private).
pub fn verify_final_pass(
    statement: &SnarkVerifierStatement,
    proof: &SnarkProof,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<VerifiedBound, SnarkError> {
    let arch = &statement.architecture;

    // Mandatory-presence check: the public statement determines which
    // proof components are required.
    require_mandatory_components(statement, proof)?;

    // Architecture validation: bind per-pass commit counts and
    // per-step (layer_idx, log_dims) to the public network.
    if let Some(pass_com) = &proof.commitments.pass_lower {
        check_pass_commit_lengths(pass_com, arch)?;
    }
    if let Some(pass_com) = &proof.commitments.pass_upper {
        check_pass_commit_lengths(pass_com, arch)?;
    }
    let lin_dims_lower: Option<Vec<_>> = proof.linear_backward_lower.as_ref().map(|v| {
        v.iter()
            .map(|p| (p.layer_idx, p.a_old_log_dims, p.w_log_dims))
            .collect()
    });
    if let Some(d) = &lin_dims_lower {
        check_linear_chain_shape(d, arch, &statement.property)?;
    }
    let lin_dims_upper: Option<Vec<_>> = proof.linear_backward_upper.as_ref().map(|v| {
        v.iter()
            .map(|p| (p.layer_idx, p.a_old_log_dims, p.w_log_dims))
            .collect()
    });
    if let Some(d) = &lin_dims_upper {
        check_linear_chain_shape(d, arch, &statement.property)?;
    }
    let act_dims_lower: Option<Vec<_>> = proof
        .activation_backward_lower
        .as_ref()
        .map(|v| v.iter().map(|p| (p.layer_idx, p.a_old_log_dims)).collect());
    if let Some(d) = &act_dims_lower {
        check_activation_chain_shape(d, arch, &statement.property)?;
    }
    let act_dims_upper: Option<Vec<_>> = proof
        .activation_backward_upper
        .as_ref()
        .map(|v| v.iter().map(|p| (p.layer_idx, p.a_old_log_dims)).collect());
    if let Some(d) = &act_dims_upper {
        check_activation_chain_shape(d, arch, &statement.property)?;
    }
    let amat_dims_lower: Option<Vec<_>> = proof
        .activation_matrix_lower
        .as_ref()
        .map(|v| v.iter().map(|p| (p.layer_idx, p.log_dims)).collect());
    if let Some(d) = &amat_dims_lower {
        check_activation_chain_shape(d, arch, &statement.property)?;
    }
    let amat_dims_upper: Option<Vec<_>> = proof
        .activation_matrix_upper
        .as_ref()
        .map(|v| v.iter().map(|p| (p.layer_idx, p.log_dims)).collect());
    if let Some(d) = &amat_dims_upper {
        check_activation_chain_shape(d, arch, &statement.property)?;
    }

    absorb_commitments(sponge, &proof.commitments);

    // Absorb the layer-scales Hyrax commit before rescale challenges
    // fire, then verify per-layer opens to reconstruct the synthetic
    // `LayerScalesCommit` downstream gadgets consume.
    super::prove::absorb_layer_scales(sponge, &proof.layer_scales);
    let synthetic_layer_scales = super::prove::helpers::verify_layer_scale_opens(
        arch,
        &proof.layer_scales,
        &proof.layer_scale_opens,
        params,
        sponge,
    )?;
    check_layer_scales_shape(&synthetic_layer_scales, arch)?;
    // Typed accessor wrapping the verified scales with an extra
    // exponent-range check; gadgets consume `Scale` via this.
    let scale_acc = crate::snark::commitment::layer_scale_api::LayerScaleAccessor::new(
        &synthetic_layer_scales,
    )?;

    // Per-tensor range LogUps in the same canonical order the prover
    // emits them.
    verify_tensor_range_proofs(
        arch,
        &proof.commitments,
        &proof.tensor_range_proofs,
        params,
        sponge,
    )?;

    // Per-linear-layer backward arithmetic checks.
    if let Some(proofs) = &proof.linear_backward_lower {
        let pass_com = proof
            .commitments
            .pass_lower
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_lower commits",
            })?;
        verify_linear_backward_chain(proofs, &proof.commitments, pass_com, params, sponge)?;
    }
    if let Some(proofs) = &proof.linear_backward_upper {
        let pass_com = proof
            .commitments
            .pass_upper
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_upper commits",
            })?;
        verify_linear_backward_chain(proofs, &proof.commitments, pass_com, params, sponge)?;
    }
    if let Some(proofs) = &proof.activation_backward_lower {
        let pass_com = proof
            .commitments
            .pass_lower
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_lower commits",
            })?;
        verify_activation_backward_chain(
            proofs,
            BoundDir::Lower,
            pass_com,
            &proof.commitments,
            params,
            sponge,
        )?;
    }
    if let Some(proofs) = &proof.activation_backward_upper {
        let pass_com = proof
            .commitments
            .pass_upper
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_upper commits",
            })?;
        verify_activation_backward_chain(
            proofs,
            BoundDir::Upper,
            pass_com,
            &proof.commitments,
            params,
            sponge,
        )?;
    }
    if let Some(c) = &proof.concretize_lower {
        let pass_com = proof
            .commitments
            .pass_lower
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_lower commits",
            })?;
        verify_concretize_proof(
            c,
            BoundDir::Lower,
            pass_com,
            &proof.commitments,
            params,
            sponge,
        )?;
    }
    if let Some(c) = &proof.concretize_upper {
        let pass_com = proof
            .commitments
            .pass_upper
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_upper commits",
            })?;
        verify_concretize_proof(
            c,
            BoundDir::Upper,
            pass_com,
            &proof.commitments,
            params,
            sponge,
        )?;
    }

    // ReLU-lookup checks.
    if let (Some(relu_proofs), Some(activation_proofs)) = (
        &proof.relu_lower_activation,
        &proof.activation_backward_lower,
    ) {
        let pass_com = proof
            .commitments
            .pass_lower
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_lower commits",
            })?;
        verify_relu_proofs_activation(relu_proofs, activation_proofs, pass_com, params, sponge)?;
    }
    if let (Some(relu_proofs), Some(activation_proofs)) = (
        &proof.relu_upper_activation,
        &proof.activation_backward_upper,
    ) {
        let pass_com = proof
            .commitments
            .pass_upper
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_upper commits",
            })?;
        verify_relu_proofs_activation(relu_proofs, activation_proofs, pass_com, params, sponge)?;
    }
    if let (Some(relu_proof), Some(con_proof)) =
        (&proof.relu_lower_concretize, &proof.concretize_lower)
    {
        let pass_com = proof
            .commitments
            .pass_lower
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_lower commits",
            })?;
        verify_relu_proof_concretize(relu_proof, con_proof, pass_com, params, sponge)?;
    }
    if let (Some(relu_proof), Some(con_proof)) =
        (&proof.relu_upper_concretize, &proof.concretize_upper)
    {
        let pass_com = proof
            .commitments
            .pass_upper
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_upper commits",
            })?;
        verify_relu_proof_concretize(relu_proof, con_proof, pass_com, params, sponge)?;
    }

    // Per-pass rescale-gadget checks. (c1, c2) is derived from the
    // committed layer_scales plus working/input scales.
    let working = crate::quantization::scale::Scale {
        c: proof.target_scale_c,
        e: proof.target_scale_e,
    };
    let pb = proof
        .public_binding
        .as_ref()
        .expect("public_binding is mandatory; require_mandatory_components rejected if missing");
    let input_scale = crate::quantization::scale::Scale {
        c: pb.input_c,
        e: pb.input_e,
    };
    // Validate working / input scales before the rescale gadget
    // consumes them; extreme `e` would otherwise overflow the shift
    // width in `Scale::ratio_as_c1_c2`.
    working
        .validate_for_pin()
        .map_err(|_| SnarkError::ArchitectureMismatch {
            what: "target_scale (working) fails validation",
        })?;
    input_scale
        .validate_for_pin()
        .map_err(|_| SnarkError::ArchitectureMismatch {
            what: "public_binding.input scale fails validation",
        })?;
    if let Some(rescale_proofs) = &proof.rescale_lower {
        let pass_com = proof
            .commitments
            .pass_lower
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_lower commits (rescale)",
            })?;
        let has_con = proof.concretize_lower.is_some();
        verify_rescale_proofs(
            rescale_proofs,
            pass_com,
            arch,
            &statement.property,
            has_con,
            &synthetic_layer_scales,
            working,
            input_scale,
            BoundDir::Lower,
            params,
            sponge,
        )?;
    }
    if let Some(rescale_proofs) = &proof.rescale_upper {
        let pass_com = proof
            .commitments
            .pass_upper
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_upper commits (rescale)",
            })?;
        let has_con = proof.concretize_upper.is_some();
        verify_rescale_proofs(
            rescale_proofs,
            pass_com,
            arch,
            &statement.property,
            has_con,
            &synthetic_layer_scales,
            working,
            input_scale,
            BoundDir::Upper,
            params,
            sponge,
        )?;
    }

    // Chain init / b_acc step / activation matrix.
    let (a_n_vars, b_n_vars) = chain_init_n_vars(arch, &statement.property);
    let bacc_n_vars = b_acc_n_vars(&statement.property);

    if let Some(ci) = &proof.chain_init_lower {
        let pass_com = proof
            .commitments
            .pass_lower
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_lower (chain_init)",
            })?;
        verify_chain_init(
            ci,
            pass_com,
            &proof.commitments,
            a_n_vars,
            b_n_vars,
            params,
            sponge,
        )?;
    }
    if let Some(ci) = &proof.chain_init_upper {
        let pass_com = proof
            .commitments
            .pass_upper
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_upper (chain_init)",
            })?;
        verify_chain_init(
            ci,
            pass_com,
            &proof.commitments,
            a_n_vars,
            b_n_vars,
            params,
            sponge,
        )?;
    }

    if let Some(steps) = &proof.b_acc_step_lower {
        let pass_com = proof
            .commitments
            .pass_lower
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_lower (b_acc_step)",
            })?;
        verify_b_acc_step_proofs(steps, pass_com, arch, bacc_n_vars, params, sponge)?;
    }
    if let Some(steps) = &proof.b_acc_step_upper {
        let pass_com = proof
            .commitments
            .pass_upper
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_upper (b_acc_step)",
            })?;
        verify_b_acc_step_proofs(steps, pass_com, arch, bacc_n_vars, params, sponge)?;
    }

    if let Some(steps) = &proof.activation_matrix_lower {
        let pass_com = proof
            .commitments
            .pass_lower
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_lower (activation_matrix)",
            })?;
        verify_activation_matrix_chain(
            steps,
            BoundDir::Lower,
            pass_com,
            &proof.commitments,
            params,
            sponge,
        )?;
    }
    if let Some(steps) = &proof.activation_matrix_upper {
        let pass_com = proof
            .commitments
            .pass_upper
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_upper (activation_matrix)",
            })?;
        verify_activation_matrix_chain(
            steps,
            BoundDir::Upper,
            pass_com,
            &proof.commitments,
            params,
            sponge,
        )?;
    }

    // Public-statement binding.
    if let Some(pb) = &proof.public_binding {
        verify_public_binding(
            pb,
            &statement.architecture,
            &statement.property,
            &statement.x_lower,
            &statement.x_upper,
            params.precision_bits,
            &proof.commitments,
            proof.target_scale_c,
            proof.target_scale_e,
            params,
            sponge,
        )?;
    }

    // Per-pass output-bound inequality. Bound tensor has shape
    // `(n_spec,)`; commit n_vars is the architecture-derived native
    // vector size.
    let n_spec = statement.property.c_matrix.nrows();
    let bound_n_vars = native_vector_n_vars(n_spec);
    let bound_n_padded = 1usize << bound_n_vars;
    let target_scale = crate::quantization::scale::Scale {
        c: proof.target_scale_c,
        e: proof.target_scale_e,
    };
    let lower_t_real = statement.property.lower_threshold_or_zero();
    let upper_t_real = statement.property.upper_threshold_or_zero();
    let mut lower_threshold_padded_fr = vec![Fr::from(0u64); bound_n_padded];
    for (slot, &v) in lower_threshold_padded_fr
        .iter_mut()
        .zip(lower_t_real.iter())
    {
        let code = crate::quantization::quantized_scalar::Qf::from_real(v, target_scale).code;
        *slot = crate::snark_primitives::finite_field::signed_lift_to_fr(code);
    }
    let mut upper_threshold_padded_fr = vec![Fr::from(0u64); bound_n_padded];
    for (slot, &v) in upper_threshold_padded_fr
        .iter_mut()
        .zip(upper_t_real.iter())
    {
        let code = crate::quantization::quantized_scalar::Qf::from_real(v, target_scale).code;
        *slot = crate::snark_primitives::finite_field::signed_lift_to_fr(code);
    }
    if let Some(ob_proof) = &proof.output_bound_lower {
        let pass_com = proof
            .commitments
            .pass_lower
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_lower commits (output_bound)",
            })?;
        let acc_w_com = pass_com
            .concretize_acc_w
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing concretize_acc_w (output_bound)",
            })?;
        let b_acc_final_com = pass_com
            .chain_b_acc
            .first()
            .ok_or(SnarkError::ShapeMismatch {
                what: "empty chain_b_acc (output_bound)",
            })?;
        verify_output_bound_inequality(
            ob_proof,
            BoundDir::Lower,
            params.out_bound_range_bits,
            bound_n_vars,
            Some(&lower_threshold_padded_fr),
            b_acc_final_com,
            acc_w_com,
            params,
            sponge,
        )?;
    }
    if let Some(ob_proof) = &proof.output_bound_upper {
        let pass_com = proof
            .commitments
            .pass_upper
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing pass_upper commits (output_bound)",
            })?;
        let acc_w_com = pass_com
            .concretize_acc_w
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "missing concretize_acc_w (output_bound)",
            })?;
        let b_acc_final_com = pass_com
            .chain_b_acc
            .first()
            .ok_or(SnarkError::ShapeMismatch {
                what: "empty chain_b_acc (output_bound)",
            })?;
        verify_output_bound_inequality(
            ob_proof,
            BoundDir::Upper,
            params.out_bound_range_bits,
            bound_n_vars,
            Some(&upper_threshold_padded_fr),
            b_acc_final_com,
            acc_w_com,
            params,
            sponge,
        )?;
    }

    // Hidden-layer preactivation bound proofs.
    crate::snark::backward_pass::hidden_pass::verify_hidden_passes(
        arch,
        &proof.hidden_passes,
        &proof.commitments,
        &synthetic_layer_scales,
        working,
        input_scale,
        params,
        sponge,
    )?;

    // Per-ReLU-layer relaxation-soundness: `b_lower=0`,
    // `d_lower ∈ {0, s_d}`, and upper-line endpoint validity.
    let mut b_iter = proof.relu_lower_offset_proofs.iter();
    let mut d_iter = proof.relu_d_boolean_proofs.iter();
    let mut up_iter = proof.relu_upper_endpoint_proofs.iter();
    let mut relu_layer_count = 0usize;
    for (layer_idx, layer) in statement.architecture.layers().iter().enumerate() {
        if let LayerShape::Activation {
            kind: ActivationKind::ReLU,
        } = layer
        {
            relu_layer_count += 1;
            let bprof = b_iter.next().ok_or(SnarkError::MissingRequiredComponent {
                what: "relu_lower_offset_proofs has fewer entries than the number of ReLU layers",
            })?;
            let dprof = d_iter.next().ok_or(SnarkError::MissingRequiredComponent {
                what: "relu_d_boolean_proofs has fewer entries than the number of ReLU layers",
            })?;
            let upprof = up_iter.next().ok_or(SnarkError::MissingRequiredComponent {
                what: "relu_upper_endpoint_proofs has fewer entries than the number of ReLU layers",
            })?;
            let relax_commit = proof
                .commitments
                .relaxation
                .get(layer_idx)
                .and_then(|c| c.as_ref())
                .ok_or(SnarkError::ShapeMismatch {
                    what: "relaxation commit missing for ReLU activation layer",
                })?;
            let expected_n_vars =
                crate::snark::commitment::architecture::relaxation_n_vars(arch, layer_idx);
            // b_lower MLE ≡ 0.
            verify_relu_lower_offset(
                bprof,
                layer_idx,
                &relax_commit.b_lower,
                expected_n_vars,
                params,
                sponge,
            )?;
            // d_lower[j] ∈ {0, s_d}; scale via the typed accessor.
            let d_scale = scale_acc.relax_d_scale(layer_idx)?;
            let s_d_code = crate::quantization::quantized_scalar::Qf::from_real(1.0, d_scale).code;
            let s_d_fr = crate::snark_primitives::finite_field::signed_lift_to_fr(s_d_code);
            verify_relu_d_boolean(
                dprof,
                layer_idx,
                expected_n_vars,
                s_d_fr,
                &relax_commit.d_lower,
                params,
                sponge,
            )?;
            // ReLU upper-line endpoint validity: the verifier passes
            // the preact commits, and the gadget binds them through a
            // `(preact, relu) ⊆ T_ReLU` LogUp.
            let preceding_linear_idx =
                layer_idx
                    .checked_sub(1)
                    .ok_or(SnarkError::ArchitectureMismatch {
                        what: "ReLU activation has no preceding Linear",
                    })?;
            let hp = proof
                .hidden_passes
                .iter()
                .find(|hp| hp.target_layer_idx == preceding_linear_idx)
                .ok_or(SnarkError::MissingRequiredComponent {
                    what: "no hidden pass for the Linear preceding a ReLU layer",
                })?;
            let n_vars_native = hp.preact_n_vars as usize;
            let b_scale = scale_acc.relax_b_scale(layer_idx)?;
            let target_scale = crate::quantization::scale::Scale {
                c: proof.target_scale_c,
                e: proof.target_scale_e,
            };
            verify_relu_upper_endpoint(
                upprof,
                layer_idx,
                n_vars_native,
                &hp.preact_lower_commit,
                &hp.preact_upper_commit,
                &relax_commit.d_upper,
                &relax_commit.b_upper,
                d_scale,
                target_scale,
                b_scale,
                params,
                sponge,
            )?;
        }
        // Sigmoid/tanh layers are handled by the loops below.
    }
    if b_iter.next().is_some() || d_iter.next().is_some() || up_iter.next().is_some() {
        return Err(SnarkError::ArchitectureMismatch {
            what: "relu relaxation-soundness proofs have more entries than ReLU layers",
        });
    }
    if relu_layer_count != proof.relu_lower_offset_proofs.len()
        || relu_layer_count != proof.relu_d_boolean_proofs.len()
        || relu_layer_count != proof.relu_upper_endpoint_proofs.len()
    {
        return Err(SnarkError::ArchitectureMismatch {
            what: "relu relaxation-soundness proof count != ReLU layer count",
        });
    }

    // Per-sigmoid/tanh-layer endpoint-validity checks (four per
    // neuron). ReLU layers go through the gadgets above.
    let mut sshape_layer_count = 0usize;
    let mut ul_iter = proof.sshape_upper_at_lower_proofs.iter();
    let mut uu_iter = proof.sshape_upper_at_upper_proofs.iter();
    let mut ll_iter = proof.sshape_lower_at_lower_proofs.iter();
    let mut lu_iter = proof.sshape_lower_at_upper_proofs.iter();
    for (layer_idx, layer) in statement.architecture.layers().iter().enumerate() {
        if let LayerShape::Activation { kind } = layer {
            let kind = *kind;
            if !matches!(kind, ActivationKind::Sigmoid | ActivationKind::Tanh) {
                continue;
            }
            sshape_layer_count += 1;
            let proof_ul = ul_iter.next().ok_or(SnarkError::MissingRequiredComponent {
                what: "sshape_upper_at_lower_proofs has fewer entries than the number of sigmoid/tanh layers",
            })?;
            let proof_uu = uu_iter.next().ok_or(SnarkError::MissingRequiredComponent {
                what: "sshape_upper_at_upper_proofs has fewer entries than the number of sigmoid/tanh layers",
            })?;
            let proof_ll = ll_iter.next().ok_or(SnarkError::MissingRequiredComponent {
                what: "sshape_lower_at_lower_proofs has fewer entries than the number of sigmoid/tanh layers",
            })?;
            let proof_lu = lu_iter.next().ok_or(SnarkError::MissingRequiredComponent {
                what: "sshape_lower_at_upper_proofs has fewer entries than the number of sigmoid/tanh layers",
            })?;
            let relax_commit = proof
                .commitments
                .relaxation
                .get(layer_idx)
                .and_then(|c| c.as_ref())
                .ok_or(SnarkError::ShapeMismatch {
                    what: "sshape Phase 3b: missing relaxation commit for sigmoid/tanh layer",
                })?;
            let d_scale = scale_acc.relax_d_scale(layer_idx)?;
            let b_scale = scale_acc.relax_b_scale(layer_idx)?;
            let target_scale_v = crate::quantization::scale::Scale {
                c: proof.target_scale_c,
                e: proof.target_scale_e,
            };
            let preceding_linear_idx =
                layer_idx
                    .checked_sub(1)
                    .ok_or(SnarkError::ArchitectureMismatch {
                        what: "sigmoid/tanh activation has no preceding Linear layer",
                    })?;
            let hp = proof
                .hidden_passes
                .iter()
                .find(|hp| hp.target_layer_idx == preceding_linear_idx)
                .ok_or(SnarkError::MissingRequiredComponent {
                    what: "sshape Phase 3b: no hidden pass for the Linear preceding sigmoid/tanh",
                })?;
            // Real neuron count comes from the public hidden-pass
            // `n_spec`; the verifier never reads raw preact codes here.
            let n_real = hp.n_spec;
            verify_sshape_upper_at_lower(
                proof_ul,
                layer_idx,
                kind,
                n_real,
                &hp.preact_lower_commit,
                &relax_commit.d_upper,
                &relax_commit.b_upper,
                d_scale,
                b_scale,
                target_scale_v,
                params,
                sponge,
            )?;
            verify_sshape_upper_at_upper(
                proof_uu,
                layer_idx,
                kind,
                n_real,
                &hp.preact_upper_commit,
                &relax_commit.d_upper,
                &relax_commit.b_upper,
                d_scale,
                b_scale,
                target_scale_v,
                params,
                sponge,
            )?;
            verify_sshape_lower_at_lower(
                proof_ll,
                layer_idx,
                kind,
                n_real,
                &hp.preact_lower_commit,
                &relax_commit.d_lower,
                &relax_commit.b_lower,
                d_scale,
                b_scale,
                target_scale_v,
                params,
                sponge,
            )?;
            verify_sshape_lower_at_upper(
                proof_lu,
                layer_idx,
                kind,
                n_real,
                &hp.preact_upper_commit,
                &relax_commit.d_lower,
                &relax_commit.b_lower,
                d_scale,
                b_scale,
                target_scale_v,
                params,
                sponge,
            )?;
        }
    }
    if ul_iter.next().is_some()
        || uu_iter.next().is_some()
        || ll_iter.next().is_some()
        || lu_iter.next().is_some()
    {
        return Err(SnarkError::ArchitectureMismatch {
            what: "sshape Phase 3b endpoint proofs have more entries than sigmoid/tanh layers",
        });
    }

    // Per-sigmoid/tanh-layer critical-point validity. Runs in its own
    // loop because the prover emits all endpoint proofs first, then
    // all critical-point proofs; interleaving would break the FS
    // transcript for networks with multiple S-shape layers.
    let mut cp_u_iter = proof.sshape_critical_point_upper_proofs.iter();
    let mut cp_l_iter = proof.sshape_critical_point_lower_proofs.iter();
    for (layer_idx, layer) in statement.architecture.layers().iter().enumerate() {
        if let LayerShape::Activation { kind } = layer {
            let kind = *kind;
            if !matches!(kind, ActivationKind::Sigmoid | ActivationKind::Tanh) {
                continue;
            }
            let proof_cp_u = cp_u_iter.next().ok_or(SnarkError::MissingRequiredComponent {
                what: "sshape_critical_point_upper_proofs has fewer entries than the number of sigmoid/tanh layers",
            })?;
            let proof_cp_l = cp_l_iter.next().ok_or(SnarkError::MissingRequiredComponent {
                what: "sshape_critical_point_lower_proofs has fewer entries than the number of sigmoid/tanh layers",
            })?;
            let relax_commit = proof
                .commitments
                .relaxation
                .get(layer_idx)
                .and_then(|c| c.as_ref())
                .ok_or(SnarkError::ShapeMismatch {
                    what: "sshape Phase 3c: missing relaxation commit for sigmoid/tanh layer",
                })?;
            let d_scale = scale_acc.relax_d_scale(layer_idx)?;
            let b_scale = scale_acc.relax_b_scale(layer_idx)?;
            let target_scale_v = crate::quantization::scale::Scale {
                c: proof.target_scale_c,
                e: proof.target_scale_e,
            };
            let preceding_linear_idx =
                layer_idx
                    .checked_sub(1)
                    .ok_or(SnarkError::ArchitectureMismatch {
                        what: "sigmoid/tanh activation has no preceding Linear layer",
                    })?;
            let hp = proof
                .hidden_passes
                .iter()
                .find(|hp| hp.target_layer_idx == preceding_linear_idx)
                .ok_or(SnarkError::MissingRequiredComponent {
                    what: "sshape Phase 3c: no hidden pass for the Linear preceding sigmoid/tanh",
                })?;
            let n_real = hp.n_spec;
            verify_sshape_critical_point(
                proof_cp_u,
                layer_idx,
                kind,
                SshapeLineKind::Upper,
                n_real,
                &hp.preact_lower_commit,
                &hp.preact_upper_commit,
                &relax_commit.d_upper,
                &relax_commit.b_upper,
                d_scale,
                b_scale,
                target_scale_v,
                params,
                sponge,
            )?;
            verify_sshape_critical_point(
                proof_cp_l,
                layer_idx,
                kind,
                SshapeLineKind::Lower,
                n_real,
                &hp.preact_lower_commit,
                &hp.preact_upper_commit,
                &relax_commit.d_lower,
                &relax_commit.b_lower,
                d_scale,
                b_scale,
                target_scale_v,
                params,
                sponge,
            )?;
        }
    }
    if cp_u_iter.next().is_some() || cp_l_iter.next().is_some() {
        return Err(SnarkError::ArchitectureMismatch {
            what: "sshape Phase 3c proofs have more entries than sigmoid/tanh layers",
        });
    }
    if sshape_layer_count != proof.sshape_upper_at_lower_proofs.len()
        || sshape_layer_count != proof.sshape_upper_at_upper_proofs.len()
        || sshape_layer_count != proof.sshape_lower_at_lower_proofs.len()
        || sshape_layer_count != proof.sshape_lower_at_upper_proofs.len()
        || sshape_layer_count != proof.sshape_critical_point_upper_proofs.len()
        || sshape_layer_count != proof.sshape_critical_point_lower_proofs.len()
    {
        return Err(SnarkError::ArchitectureMismatch {
            what: "sshape Phase 3b proof count != sigmoid/tanh layer count",
        });
    }

    // The claimed bound is private. The verifier's output is purely
    // accept/reject; the `VerifiedBound` shape is preserved for API
    // compatibility but both directions are always `None`.
    let _ = proof.target_scale_c;
    let _ = proof.target_scale_e;
    Ok(VerifiedBound {
        lower: None,
        upper: None,
    })
}
