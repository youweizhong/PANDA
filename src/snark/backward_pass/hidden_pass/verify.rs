//! Hidden-layer pass verifier; mirrors [`super::prove`].
//!
//! Derives the list of hidden Linear layers from the public
//! architecture and, for each pass: absorbs the FS tag and per-pass
//! preact commits, verifies the chain-init-from-identity proof,
//! delegates per-step proofs to the shared gadgets, then asserts
//! that `output_bound_*.claimed_commit` byte-matches the per-pass
//! `preact_*_commit` and runs `verify_output_bound_inequality` to
//! bind the committed preact bound to `b_acc_final + acc_w`.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_serialize::CanonicalSerialize;

use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

/// Absorb a Hyrax commitment into the FS sponge by canonical
/// serialization. Mirrors the prover-side helper.
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

use crate::crown::network::{LayerShape, NetworkArchitecture};

use super::{absorb_hidden_pass_tag, identity_mle_eval};
use crate::quantized_crown::BoundDir;
use crate::snark::backward_pass::activation_matrix::verify_activation_matrix_chain;
use crate::snark::backward_pass::activation_step::verify_activation_backward_chain;
use crate::snark::backward_pass::bias_accumulator::verify_b_acc_step_proofs;
use crate::snark::backward_pass::linear_step::verify_linear_backward_chain;
use crate::snark::backward_pass::signed_components::driver::{
    verify_relu_proof_concretize, verify_relu_proofs_activation,
};
use crate::snark::commitment::architecture::{
    check_activation_chain_shape, check_linear_chain_shape, check_pass_commit_lengths,
};
use crate::snark::commitment::commit::{
    native_matrix_n_vars, native_vector_n_vars, TensorCommitments,
};
use crate::snark::commitment::pcs_helpers::hyrax_verify_at;
use crate::snark::concretization::concretize::verify_concretize_proof;
use crate::snark::errors::SnarkError;
use crate::snark::output_bound::verify_output_bound_inequality;
use crate::snark::params::SnarkParams;
use crate::snark::proof::{ChainInitFromIdentityProof, HiddenLayerPassProof};
use crate::snark::rescaling::driver::verify_rescale_proofs;

/// Verify every hidden-layer pass proof against the public
/// architecture and the final pass's shared commitments.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_hidden_passes(
    arch: &NetworkArchitecture,
    proofs: &[HiddenLayerPassProof],
    commitments: &TensorCommitments,
    layer_scales: &crate::snark::proof::LayerScalesCommit,
    working: crate::quantization::scale::Scale,
    input_scale: crate::quantization::scale::Scale,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    let expected_targets = hidden_linear_indices(arch);
    if proofs.len() != expected_targets.len() {
        return Err(SnarkError::ArchitectureMismatch {
            what: "hidden_passes: count != number of hidden Linear layers in network",
        });
    }
    for (proof, &expected_idx) in proofs.iter().zip(expected_targets.iter()) {
        if proof.target_layer_idx != expected_idx {
            return Err(SnarkError::ArchitectureMismatch {
                what: "hidden_passes: target_layer_idx mismatch with architecture",
            });
        }
        verify_one_hidden_pass(
            arch,
            proof,
            commitments,
            layer_scales,
            working,
            input_scale,
            params,
            sponge,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_one_hidden_pass(
    arch: &NetworkArchitecture,
    proof: &HiddenLayerPassProof,
    commitments: &TensorCommitments,
    layer_scales: &crate::snark::proof::LayerScalesCommit,
    working: crate::quantization::scale::Scale,
    input_scale: crate::quantization::scale::Scale,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    // Architecture sanity: target layer must be Linear with output
    // dim matching `proof.n_spec`, and the chain must hold target+2
    // entries.
    let target = proof.target_layer_idx;
    let layers = arch.layers();
    let expected_n_spec = match layers.get(target) {
        Some(LayerShape::Linear { out_dim, .. }) => *out_dim,
        _ => {
            return Err(SnarkError::ArchitectureMismatch {
                what: "hidden_pass: target_layer_idx not Linear",
            });
        }
    };
    if expected_n_spec != proof.n_spec {
        return Err(SnarkError::ArchitectureMismatch {
            what: "hidden_pass: n_spec != target Linear output dim",
        });
    }
    let expected_chain_len = target + 2;
    if proof.pass_lower.chain_a.len() != expected_chain_len
        || proof.pass_upper.chain_a.len() != expected_chain_len
    {
        return Err(SnarkError::ArchitectureMismatch {
            what: "hidden_pass: chain_a length != target+2",
        });
    }
    // Truncate the architecture to the target layer so the per-step
    // shape checks (which key off `arch.layers().len()`) accept the
    // shorter chain.
    let truncated_arch = arch.truncate_to(target);

    check_pass_commit_lengths(&proof.pass_lower, &truncated_arch)?;
    check_pass_commit_lengths(&proof.pass_upper, &truncated_arch)?;

    let lin_dims_lower: Vec<_> = proof
        .linear_backward_lower
        .iter()
        .map(|p| (p.layer_idx, p.a_old_log_dims, p.w_log_dims))
        .collect();
    check_linear_chain_shape_for_truncated(&lin_dims_lower, &truncated_arch, proof.n_spec)?;
    let lin_dims_upper: Vec<_> = proof
        .linear_backward_upper
        .iter()
        .map(|p| (p.layer_idx, p.a_old_log_dims, p.w_log_dims))
        .collect();
    check_linear_chain_shape_for_truncated(&lin_dims_upper, &truncated_arch, proof.n_spec)?;
    let act_dims_lower: Vec<_> = proof
        .activation_backward_lower
        .iter()
        .map(|p| (p.layer_idx, p.a_old_log_dims))
        .collect();
    check_activation_chain_shape_for_truncated(&act_dims_lower, &truncated_arch, proof.n_spec)?;
    let act_dims_upper: Vec<_> = proof
        .activation_backward_upper
        .iter()
        .map(|p| (p.layer_idx, p.a_old_log_dims))
        .collect();
    check_activation_chain_shape_for_truncated(&act_dims_upper, &truncated_arch, proof.n_spec)?;

    let bound_n_vars_check = native_vector_n_vars(proof.n_spec);
    if proof.preact_n_vars as usize != bound_n_vars_check {
        return Err(SnarkError::ArchitectureMismatch {
            what: "hidden_pass: preact_n_vars != native_vector_n_vars(n_spec)",
        });
    }

    absorb_hidden_pass_tag(sponge, target, proof.n_spec);

    // Bind the per-pass preact commits into the FS sponge. The preact
    // values themselves are private witnesses; downstream gadgets
    // open these same commits at their own FS-derived points.
    crate::snark::rescaling::absorb_commitment(sponge, &proof.preact_lower_commit);
    crate::snark::rescaling::absorb_commitment(sponge, &proof.preact_upper_commit);

    // Pin chain_a[target+1] = identity, chain_b_acc[target+1] = 0
    // before any subsequent FS challenges.
    verify_chain_init_from_identity(
        &proof.chain_init_lower,
        &proof.pass_lower,
        target,
        proof.n_spec,
        working, // hidden pass uses spec_scale = working (cert::scales.spec)
        params,
        sponge,
    )?;
    verify_chain_init_from_identity(
        &proof.chain_init_upper,
        &proof.pass_upper,
        target,
        proof.n_spec,
        working,
        params,
        sponge,
    )?;

    verify_linear_backward_chain(
        &proof.linear_backward_lower,
        commitments,
        &proof.pass_lower,
        params,
        sponge,
    )?;
    verify_linear_backward_chain(
        &proof.linear_backward_upper,
        commitments,
        &proof.pass_upper,
        params,
        sponge,
    )?;
    verify_activation_backward_chain(
        &proof.activation_backward_lower,
        BoundDir::Lower,
        &proof.pass_lower,
        commitments,
        params,
        sponge,
    )?;
    verify_activation_backward_chain(
        &proof.activation_backward_upper,
        BoundDir::Upper,
        &proof.pass_upper,
        commitments,
        params,
        sponge,
    )?;
    verify_concretize_proof(
        &proof.concretize_lower,
        BoundDir::Lower,
        &proof.pass_lower,
        commitments,
        params,
        sponge,
    )?;
    verify_concretize_proof(
        &proof.concretize_upper,
        BoundDir::Upper,
        &proof.pass_upper,
        commitments,
        params,
        sponge,
    )?;
    verify_relu_proofs_activation(
        &proof.relu_lower_activation,
        &proof.activation_backward_lower,
        &proof.pass_lower,
        params,
        sponge,
    )?;
    verify_relu_proofs_activation(
        &proof.relu_upper_activation,
        &proof.activation_backward_upper,
        &proof.pass_upper,
        params,
        sponge,
    )?;
    verify_relu_proof_concretize(
        &proof.relu_lower_concretize,
        &proof.concretize_lower,
        &proof.pass_lower,
        params,
        sponge,
    )?;
    verify_relu_proof_concretize(
        &proof.relu_upper_concretize,
        &proof.concretize_upper,
        &proof.pass_upper,
        params,
        sponge,
    )?;

    // Rescale driver. Hidden passes share the final pass's scales;
    // a synthetic property carries the n_spec rows that the driver
    // reads from `property.c_matrix`.
    verify_rescale_proofs(
        &proof.rescale_lower,
        &proof.pass_lower,
        &truncated_arch,
        &synthetic_property(proof.n_spec, truncated_arch.output_dim()),
        true, // has_concretize
        layer_scales,
        working,
        input_scale,
        crate::quantized_crown::BoundDir::Lower,
        params,
        sponge,
    )?;
    verify_rescale_proofs(
        &proof.rescale_upper,
        &proof.pass_upper,
        &truncated_arch,
        &synthetic_property(proof.n_spec, truncated_arch.output_dim()),
        true,
        layer_scales,
        working,
        input_scale,
        crate::quantized_crown::BoundDir::Upper,
        params,
        sponge,
    )?;

    let bacc_n_vars = native_vector_n_vars(proof.n_spec);
    verify_b_acc_step_proofs(
        &proof.b_acc_step_lower,
        &proof.pass_lower,
        &truncated_arch,
        bacc_n_vars,
        params,
        sponge,
    )?;
    verify_b_acc_step_proofs(
        &proof.b_acc_step_upper,
        &proof.pass_upper,
        &truncated_arch,
        bacc_n_vars,
        params,
        sponge,
    )?;
    verify_activation_matrix_chain(
        &proof.activation_matrix_lower,
        BoundDir::Lower,
        &proof.pass_lower,
        commitments,
        params,
        sponge,
    )?;
    verify_activation_matrix_chain(
        &proof.activation_matrix_upper,
        BoundDir::Upper,
        &proof.pass_upper,
        commitments,
        params,
        sponge,
    )?;

    // Output-bound inequality: the inner gadget commits the preact
    // codes itself and binds them to `b_acc_final + acc_w` via the
    // slack identity.
    let bound_n_vars = native_vector_n_vars(proof.n_spec);
    let _bound_n_padded = 1usize << bound_n_vars;
    let acc_w_com_l =
        proof
            .pass_lower
            .concretize_acc_w
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "hidden_pass: missing concretize_acc_w (lower output_bound)",
            })?;
    let acc_w_com_u =
        proof
            .pass_upper
            .concretize_acc_w
            .as_ref()
            .ok_or(SnarkError::ShapeMismatch {
                what: "hidden_pass: missing concretize_acc_w (upper output_bound)",
            })?;
    let b_acc_final_com_l =
        proof
            .pass_lower
            .chain_b_acc
            .first()
            .ok_or(SnarkError::ShapeMismatch {
                what: "hidden_pass: empty chain_b_acc (lower output_bound)",
            })?;
    let b_acc_final_com_u =
        proof
            .pass_upper
            .chain_b_acc
            .first()
            .ok_or(SnarkError::ShapeMismatch {
                what: "hidden_pass: empty chain_b_acc (upper output_bound)",
            })?;
    // The output_bound's claimed_commit must be byte-identical to
    // the per-pass preact commit so the inequality binds the same
    // MLE that downstream relaxation gadgets consume.
    let mut buf_a = Vec::new();
    let mut buf_b = Vec::new();
    use ark_serialize::CanonicalSerialize;
    proof
        .output_bound_lower
        .claimed_commit
        .serialize_compressed(&mut buf_a)
        .map_err(|_| SnarkError::ShapeMismatch {
            what: "hidden_pass: serialize output_bound_lower claimed_commit",
        })?;
    proof
        .preact_lower_commit
        .serialize_compressed(&mut buf_b)
        .map_err(|_| SnarkError::ShapeMismatch {
            what: "hidden_pass: serialize preact_lower_commit",
        })?;
    if buf_a != buf_b {
        return Err(SnarkError::ArchitectureMismatch {
            what: "hidden_pass: output_bound_lower.claimed_commit != preact_lower_commit",
        });
    }
    let mut buf_a = Vec::new();
    let mut buf_b = Vec::new();
    proof
        .output_bound_upper
        .claimed_commit
        .serialize_compressed(&mut buf_a)
        .map_err(|_| SnarkError::ShapeMismatch {
            what: "hidden_pass: serialize output_bound_upper claimed_commit",
        })?;
    proof
        .preact_upper_commit
        .serialize_compressed(&mut buf_b)
        .map_err(|_| SnarkError::ShapeMismatch {
            what: "hidden_pass: serialize preact_upper_commit",
        })?;
    if buf_a != buf_b {
        return Err(SnarkError::ArchitectureMismatch {
            what: "hidden_pass: output_bound_upper.claimed_commit != preact_upper_commit",
        });
    }

    // Hidden-pass preact bounds are private witnesses; pass `None`
    // to skip the in-SNARK property threshold check. Hidden passes
    // run at the narrow gadget budget, mirroring the prover.
    verify_output_bound_inequality(
        &proof.output_bound_lower,
        BoundDir::Lower,
        params.gadget_range_bits,
        bound_n_vars,
        None,
        b_acc_final_com_l,
        acc_w_com_l,
        params,
        sponge,
    )?;
    verify_output_bound_inequality(
        &proof.output_bound_upper,
        BoundDir::Upper,
        params.gadget_range_bits,
        bound_n_vars,
        None,
        b_acc_final_com_u,
        acc_w_com_u,
        params,
        sponge,
    )?;

    Ok(())
}

fn verify_chain_init_from_identity(
    proof: &ChainInitFromIdentityProof,
    pass_com: &crate::snark::commitment::commit::PassCommitments,
    target_layer_idx: usize,
    n_spec: usize,
    spec_scale: crate::quantization::scale::Scale,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    let chain_top = target_layer_idx + 1;
    let chain_a_com = pass_com
        .chain_a
        .get(chain_top)
        .ok_or(SnarkError::ShapeMismatch {
            what: "hidden_pass chain_init: chain_a out-of-range",
        })?;
    let chain_b_com = pass_com
        .chain_b_acc
        .get(chain_top)
        .ok_or(SnarkError::ShapeMismatch {
            what: "hidden_pass chain_init: chain_b_acc out-of-range",
        })?;
    let expected_a_n_vars = native_matrix_n_vars(n_spec, n_spec);
    let expected_b_n_vars = native_vector_n_vars(n_spec);
    if proof.a_n_vars != expected_a_n_vars || proof.b_n_vars != expected_b_n_vars {
        return Err(SnarkError::ShapeMismatch {
            what: "hidden_pass chain_init: n_vars mismatch",
        });
    }

    // Mirror the prover: bind chain commits before squeezing.
    absorb_chain_commit(sponge, chain_a_com);
    absorb_chain_commit(sponge, chain_b_com);
    sponge.absorb(&(expected_a_n_vars as u64));
    sponge.absorb(&(expected_b_n_vars as u64));
    let r_a: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(expected_a_n_vars);
    let r_b: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(expected_b_n_vars);
    if r_a != proof.r_a || r_b != proof.r_b {
        return Err(SnarkError::TranscriptMismatch);
    }

    let a_ok = hyrax_verify_at(
        &params.verifier_key,
        chain_a_com,
        &proof.r_a,
        proof.chain_a_eval,
        &proof.chain_a_open,
        expected_a_n_vars,
        sponge,
    )?;
    if !a_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "hidden_pass_chain_init_a",
        });
    }
    let b_ok = hyrax_verify_at(
        &params.verifier_key,
        chain_b_com,
        &proof.r_b,
        proof.chain_b_eval,
        &proof.chain_b_open,
        expected_b_n_vars,
        sponge,
    )?;
    if !b_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "hidden_pass_chain_init_b",
        });
    }

    let expected_a = identity_mle_eval(&r_a, n_spec, spec_scale);
    if proof.chain_a_eval != expected_a {
        return Err(SnarkError::ChainInitMismatch);
    }
    if proof.chain_b_eval != Fr::from(0u64) {
        return Err(SnarkError::ChainInitMismatch);
    }
    Ok(())
}

/// Indices of every Linear layer immediately followed by an
/// Activation, in forward order — the target layers for hidden
/// passes.
fn hidden_linear_indices(arch: &NetworkArchitecture) -> Vec<usize> {
    let layers = arch.layers();
    let n = layers.len();
    let mut out = Vec::new();
    for idx in 0..n {
        if matches!(layers[idx], LayerShape::Linear { .. })
            && idx + 1 < n
            && matches!(layers[idx + 1], LayerShape::Activation { .. })
        {
            out.push(idx);
        }
    }
    out
}

/// Synthetic `Property` carrying the hidden-pass `n_spec` and the
/// truncated network's output dim. Only consumed for the
/// `c_matrix.nrows()` / `.ncols()` shape lookups the per-step gadgets
/// perform.
fn synthetic_property(n_spec: usize, out_dim: usize) -> crate::crown::output_property::Property {
    use ndarray::{Array1, Array2};
    crate::crown::output_property::Property::new(
        Array2::<f64>::eye(n_spec.max(out_dim))
            .slice(ndarray::s![0..n_spec, 0..out_dim])
            .to_owned(),
        Array1::<f64>::zeros(n_spec),
        crate::crown::output_property::Side::Both,
    )
    .expect("synthetic property: dims valid by construction")
}

/// Wrapper around `check_linear_chain_shape` that takes the
/// hidden-pass `n_spec` instead of reading it off a property.
fn check_linear_chain_shape_for_truncated(
    proofs_layer_idx_dims: &[(usize, (usize, usize), (usize, usize))],
    arch: &NetworkArchitecture,
    n_spec: usize,
) -> Result<(), SnarkError> {
    use ndarray::{Array1, Array2};
    let synth = crate::crown::output_property::Property::new(
        Array2::<f64>::zeros((n_spec, arch.output_dim())),
        Array1::<f64>::zeros(n_spec),
        crate::crown::output_property::Side::Both,
    )
    .expect("synth property valid");
    check_linear_chain_shape(proofs_layer_idx_dims, arch, &synth)
}

fn check_activation_chain_shape_for_truncated(
    proofs_layer_idx_dims: &[(usize, (usize, usize))],
    arch: &NetworkArchitecture,
    n_spec: usize,
) -> Result<(), SnarkError> {
    use ndarray::{Array1, Array2};
    let synth = crate::crown::output_property::Property::new(
        Array2::<f64>::zeros((n_spec, arch.output_dim())),
        Array1::<f64>::zeros(n_spec),
        crate::crown::output_property::Side::Both,
    )
    .expect("synth property valid");
    check_activation_chain_shape(proofs_layer_idx_dims, arch, &synth)
}
