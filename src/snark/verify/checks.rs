//! Pre-flight shape and architecture checks for the verifier.
//!
//! Validates that the prover-supplied components are shape-consistent
//! with the public statement, and replays the per-tensor LogUp range
//! check against each committed tensor. All cheap checks here run
//! before the heavy per-step gadgets so malformed proofs reject fast.

use ark_crypto_primitives::sponge::CryptographicSponge;

use crate::crown::network::{ActivationKind, LayerShape, NetworkArchitecture};

use crate::snark::commitment::commit::{
    native_matrix_n_vars, native_vector_n_vars, TensorCommitments,
};
use crate::snark::commitment::range_per_tensor::{verify_tensor_range_logup, TensorRangeProof};
use crate::snark::errors::SnarkError;
use crate::snark::params::{SnarkParams, SnarkVerifierStatement};
use crate::snark::proof::SnarkProof;

/// Reject if any proof component required by the public statement is
/// missing. `Property::Side` selects which per-pass components are
/// required; `public_binding` is required unconditionally.
pub(super) fn require_mandatory_components(
    statement: &SnarkVerifierStatement,
    proof: &SnarkProof,
) -> Result<(), SnarkError> {
    macro_rules! require {
        ($cond:expr, $name:literal) => {
            if !$cond {
                return Err(SnarkError::MissingRequiredComponent { what: $name });
            }
        };
    }
    require!(proof.public_binding.is_some(), "public_binding");

    let side = statement.property.side;
    if side.needs_lower() {
        require!(proof.commitments.pass_lower.is_some(), "pass_lower commits");
        require!(
            proof.linear_backward_lower.is_some(),
            "linear_backward_lower"
        );
        require!(
            proof.activation_backward_lower.is_some(),
            "activation_backward_lower"
        );
        require!(proof.concretize_lower.is_some(), "concretize_lower");
        require!(
            proof.relu_lower_activation.is_some(),
            "relu_lower_activation"
        );
        require!(
            proof.relu_lower_concretize.is_some(),
            "relu_lower_concretize"
        );
        require!(proof.rescale_lower.is_some(), "rescale_lower");
        require!(proof.chain_init_lower.is_some(), "chain_init_lower");
        require!(proof.b_acc_step_lower.is_some(), "b_acc_step_lower");
        require!(
            proof.activation_matrix_lower.is_some(),
            "activation_matrix_lower"
        );
        require!(proof.output_bound_lower.is_some(), "output_bound_lower");
    }
    if side.needs_upper() {
        require!(proof.commitments.pass_upper.is_some(), "pass_upper commits");
        require!(
            proof.linear_backward_upper.is_some(),
            "linear_backward_upper"
        );
        require!(
            proof.activation_backward_upper.is_some(),
            "activation_backward_upper"
        );
        require!(proof.concretize_upper.is_some(), "concretize_upper");
        require!(
            proof.relu_upper_activation.is_some(),
            "relu_upper_activation"
        );
        require!(
            proof.relu_upper_concretize.is_some(),
            "relu_upper_concretize"
        );
        require!(proof.rescale_upper.is_some(), "rescale_upper");
        require!(proof.chain_init_upper.is_some(), "chain_init_upper");
        require!(proof.b_acc_step_upper.is_some(), "b_acc_step_upper");
        require!(
            proof.activation_matrix_upper.is_some(),
            "activation_matrix_upper"
        );
        require!(proof.output_bound_upper.is_some(), "output_bound_upper");
    }

    // Per sigmoid/tanh activation layer the proof must carry four
    // endpoint-validity vectors of length `sshape_layer_count`. The
    // per-layer checks run inside `verify_final_pass`.
    let arch = &statement.architecture;
    let sshape_layer_count = arch
        .layers()
        .iter()
        .filter(|l| {
            matches!(
                l,
                LayerShape::Activation {
                    kind: ActivationKind::Sigmoid
                } | LayerShape::Activation {
                    kind: ActivationKind::Tanh
                }
            )
        })
        .count();
    if proof.sshape_upper_at_lower_proofs.len() != sshape_layer_count {
        return Err(SnarkError::MissingRequiredComponent {
            what: "sshape_upper_at_lower_proofs count != sigmoid/tanh layer count",
        });
    }
    if proof.sshape_upper_at_upper_proofs.len() != sshape_layer_count {
        return Err(SnarkError::MissingRequiredComponent {
            what: "sshape_upper_at_upper_proofs count != sigmoid/tanh layer count",
        });
    }
    if proof.sshape_lower_at_lower_proofs.len() != sshape_layer_count {
        return Err(SnarkError::MissingRequiredComponent {
            what: "sshape_lower_at_lower_proofs count != sigmoid/tanh layer count",
        });
    }
    if proof.sshape_lower_at_upper_proofs.len() != sshape_layer_count {
        return Err(SnarkError::MissingRequiredComponent {
            what: "sshape_lower_at_upper_proofs count != sigmoid/tanh layer count",
        });
    }
    if proof.sshape_critical_point_upper_proofs.len() != sshape_layer_count {
        return Err(SnarkError::MissingRequiredComponent {
            what: "sshape_critical_point_upper_proofs count != sigmoid/tanh layer count",
        });
    }
    if proof.sshape_critical_point_lower_proofs.len() != sshape_layer_count {
        return Err(SnarkError::MissingRequiredComponent {
            what: "sshape_critical_point_lower_proofs count != sigmoid/tanh layer count",
        });
    }
    Ok(())
}

/// Walk every range-checked public-witness tensor in the canonical
/// prover order and verify the per-tensor LogUp against its commitment.
pub(super) fn verify_tensor_range_proofs(
    arch: &NetworkArchitecture,
    commitments: &TensorCommitments,
    proofs: &[TensorRangeProof],
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    let mut iter = proofs.iter();
    let mut next = || -> Result<&TensorRangeProof, SnarkError> {
        iter.next().ok_or(SnarkError::ArchitectureMismatch {
            what: "tensor_range_proofs: fewer per-tensor proofs than expected",
        })
    };

    // Input box.
    {
        let p = next()?;
        let n_vars = native_vector_n_vars(arch.input_dim());
        verify_tensor_range_logup(p, &commitments.x_lower, n_vars, params, sponge)?;
    }
    {
        let p = next()?;
        let n_vars = native_vector_n_vars(arch.input_dim());
        verify_tensor_range_logup(p, &commitments.x_upper, n_vars, params, sponge)?;
    }

    for (i, layer) in arch.layers().iter().enumerate() {
        match layer {
            LayerShape::Linear { in_dim, out_dim } => {
                let p = next()?;
                let w_n_vars = native_matrix_n_vars(*out_dim, *in_dim);
                let w_com =
                    commitments.weight[i]
                        .as_ref()
                        .ok_or(SnarkError::ArchitectureMismatch {
                            what: "tensor_range_proofs: missing weight commit at linear layer",
                        })?;
                verify_tensor_range_logup(p, w_com, w_n_vars, params, sponge)?;

                let p = next()?;
                let b_n_vars = native_vector_n_vars(*out_dim);
                let b_com =
                    commitments.bias[i]
                        .as_ref()
                        .ok_or(SnarkError::ArchitectureMismatch {
                            what: "tensor_range_proofs: missing bias commit at linear layer",
                        })?;
                verify_tensor_range_logup(p, b_com, b_n_vars, params, sponge)?;
            }
            LayerShape::Activation { .. } => {
                let chain_cols = crate::snark::commitment::architecture::chain_a_cols(arch);
                let neuron_count = chain_cols[i + 1];
                let n_vars = native_vector_n_vars(neuron_count);
                let rc =
                    commitments.relaxation[i]
                        .as_ref()
                        .ok_or(SnarkError::ArchitectureMismatch {
                        what: "tensor_range_proofs: missing relaxation commits at activation layer",
                    })?;
                let p = next()?;
                verify_tensor_range_logup(p, &rc.d_lower, n_vars, params, sponge)?;
                let p = next()?;
                verify_tensor_range_logup(p, &rc.d_upper, n_vars, params, sponge)?;
                let p = next()?;
                verify_tensor_range_logup(p, &rc.b_lower, n_vars, params, sponge)?;
                let p = next()?;
                verify_tensor_range_logup(p, &rc.b_upper, n_vars, params, sponge)?;
            }
        }
    }

    if iter.next().is_some() {
        return Err(SnarkError::ArchitectureMismatch {
            what: "tensor_range_proofs: more per-tensor proofs than expected",
        });
    }
    Ok(())
}

/// Validate a `LayerScalesCommit` against the public architecture:
/// per-array length must equal the layer count, and every meaningful
/// `(c, e)` must satisfy `Scale::validate_for_pin`. Sentinel `(0, 0)`
/// is allowed only at the layer kind that doesn't carry that scale.
pub(super) fn check_layer_scales_shape(
    scales: &crate::snark::proof::LayerScalesCommit,
    arch: &NetworkArchitecture,
) -> Result<(), SnarkError> {
    let n = arch.layers().len();
    let len_ok = scales.weight_c.len() == n
        && scales.weight_e.len() == n
        && scales.bias_c.len() == n
        && scales.bias_e.len() == n
        && scales.relax_d_c.len() == n
        && scales.relax_d_e.len() == n
        && scales.relax_b_c.len() == n
        && scales.relax_b_e.len() == n;
    if !len_ok {
        return Err(SnarkError::ArchitectureMismatch {
            what: "layer_scales: per-array length != n_layers",
        });
    }
    for (i, layer) in arch.layers().iter().enumerate() {
        match layer {
            LayerShape::Linear { .. } => {
                let w = crate::quantization::scale::Scale {
                    c: scales.weight_c[i],
                    e: scales.weight_e[i],
                };
                let b = crate::quantization::scale::Scale {
                    c: scales.bias_c[i],
                    e: scales.bias_e[i],
                };
                w.validate_for_pin().map_err(|_| SnarkError::ArchitectureMismatch {
                    what: "layer_scales: linear layer weight scale fails validation (c > 0 and |e| <= MAX_SCALE_E_ABS)",
                })?;
                b.validate_for_pin()
                    .map_err(|_| SnarkError::ArchitectureMismatch {
                        what: "layer_scales: linear layer bias scale fails validation",
                    })?;
            }
            LayerShape::Activation { .. } => {
                let d = crate::quantization::scale::Scale {
                    c: scales.relax_d_c[i],
                    e: scales.relax_d_e[i],
                };
                let b = crate::quantization::scale::Scale {
                    c: scales.relax_b_c[i],
                    e: scales.relax_b_e[i],
                };
                d.validate_for_pin()
                    .map_err(|_| SnarkError::ArchitectureMismatch {
                        what: "layer_scales: activation relax_d scale fails validation",
                    })?;
                b.validate_for_pin()
                    .map_err(|_| SnarkError::ArchitectureMismatch {
                        what: "layer_scales: activation relax_b scale fails validation",
                    })?;
            }
        }
    }
    Ok(())
}

// `check_target_codes_in_range` lives on only for the still-public
// hidden-pass `preact_<dir>_codes`. The final-pass claimed bound is
// fully hidden and goes through `OutputBoundIneqProof` instead.
