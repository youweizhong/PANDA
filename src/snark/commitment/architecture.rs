//! Architecture-binding helpers used by the verifier to re-derive
//! canonical `(layer_idx, dims)` tuples for every per-step proof
//! component and to check the prover's claimed tuples against them.
//!
//! Helpers consume `&NetworkArchitecture` (a public-shape view)
//! rather than `&Network`, so any code path that tries to read a
//! weight or bias here will not compile.

use crate::crown::network::{LayerShape, NetworkArchitecture};
use crate::crown::output_property::Property;

use crate::snark::commitment::commit::native_vector_n_vars;
use crate::snark::commitment::multilinear_extensions::next_pow2_log;
use crate::snark::errors::SnarkError;

/// Backward-walk the architecture and return `A_cols` for every
/// `chain_a[idx]`. Starts at `output_dim()`; Linear collapses to
/// its input dim; Activation preserves dim.
pub(crate) fn chain_a_cols(arch: &NetworkArchitecture) -> Vec<usize> {
    let layers = arch.layers();
    let n_layers = layers.len();
    let mut out = vec![0usize; n_layers + 1];
    out[n_layers] = arch.output_dim();
    let mut a_cols = arch.output_dim();
    for i in (0..n_layers).rev() {
        match &layers[i] {
            LayerShape::Linear { in_dim, .. } => a_cols = *in_dim,
            LayerShape::Activation { .. } => {}
        }
        out[i] = a_cols;
    }
    out
}

/// Neuron count of the activation at `layer_idx`, equal to the
/// `A_cols` flowing into it.
pub(crate) fn activation_neuron_count(arch: &NetworkArchitecture, layer_idx: usize) -> usize {
    chain_a_cols(arch)[layer_idx + 1]
}

/// Native commit `n_vars` for any of the per-activation relaxation
/// vectors (`d_lower`, `d_upper`, `b_lower`, `b_upper`).
pub(crate) fn relaxation_n_vars(arch: &NetworkArchitecture, layer_idx: usize) -> usize {
    native_vector_n_vars(activation_neuron_count(arch, layer_idx))
}

/// Canonical `(layer_idx, a_old_log_dims, w_log_dims)` per linear
/// step in the prover's backward traversal order.
pub(crate) fn linear_step_metadata(
    arch: &NetworkArchitecture,
    property: &Property,
) -> Vec<(usize, (usize, usize), (usize, usize))> {
    let log_spec = next_pow2_log(property.c_matrix.nrows());
    let mut out = Vec::new();
    let mut a_cols = arch.output_dim();
    let layers = arch.layers();
    for i in (0..layers.len()).rev() {
        match &layers[i] {
            LayerShape::Linear { in_dim, out_dim } => {
                debug_assert_eq!(*out_dim, a_cols);
                let a_old_log_dims = (log_spec, next_pow2_log(*out_dim));
                let w_log_dims = (next_pow2_log(*out_dim), next_pow2_log(*in_dim));
                out.push((i, a_old_log_dims, w_log_dims));
                a_cols = *in_dim;
            }
            LayerShape::Activation { .. } => {}
        }
    }
    out
}

/// Canonical `(layer_idx, a_old_log_dims)` per activation step in
/// backward traversal order.
pub(crate) fn activation_step_metadata(
    arch: &NetworkArchitecture,
    property: &Property,
) -> Vec<(usize, (usize, usize))> {
    let log_spec = next_pow2_log(property.c_matrix.nrows());
    let mut out = Vec::new();
    let mut a_cols = arch.output_dim();
    let layers = arch.layers();
    for i in (0..layers.len()).rev() {
        match &layers[i] {
            LayerShape::Linear { in_dim, .. } => {
                a_cols = *in_dim;
            }
            LayerShape::Activation { .. } => {
                let log_neur = next_pow2_log(a_cols);
                out.push((i, (log_spec, log_neur)));
            }
        }
    }
    out
}

/// Returns `(n_linear_steps, n_activation_steps)` for the architecture.
pub(crate) fn step_counts(arch: &NetworkArchitecture) -> (usize, usize) {
    let mut linear = 0;
    let mut activation = 0;
    for layer in arch.layers() {
        match layer {
            LayerShape::Linear { .. } => linear += 1,
            LayerShape::Activation { .. } => activation += 1,
        }
    }
    (linear, activation)
}

/// Chain length `n_layers + 1` that the verifier expects.
pub(crate) fn chain_len(arch: &NetworkArchitecture) -> usize {
    arch.layers().len() + 1
}

/// Assert the prover's per-linear-step `(layer_idx, dims)` tuples
/// match the canonical ones derived from the public architecture.
pub(crate) fn check_linear_chain_shape(
    proofs_layer_idx_dims: &[(usize, (usize, usize), (usize, usize))],
    arch: &NetworkArchitecture,
    property: &Property,
) -> Result<(), SnarkError> {
    let expected = linear_step_metadata(arch, property);
    if proofs_layer_idx_dims.len() != expected.len() {
        return Err(SnarkError::ArchitectureMismatch {
            what: "linear_backward step count != n_linear in network",
        });
    }
    for ((p_layer, p_a, p_w), (e_layer, e_a, e_w)) in
        proofs_layer_idx_dims.iter().zip(expected.iter())
    {
        if p_layer != e_layer || p_a != e_a || p_w != e_w {
            return Err(SnarkError::ArchitectureMismatch {
                what: "linear_backward step (layer_idx, dims) mismatch",
            });
        }
    }
    Ok(())
}

/// Assert the prover's per-activation-step `(layer_idx, dims)`
/// tuples match the canonical ones.
pub(crate) fn check_activation_chain_shape(
    proofs_layer_idx_dims: &[(usize, (usize, usize))],
    arch: &NetworkArchitecture,
    property: &Property,
) -> Result<(), SnarkError> {
    let expected = activation_step_metadata(arch, property);
    if proofs_layer_idx_dims.len() != expected.len() {
        return Err(SnarkError::ArchitectureMismatch {
            what: "activation step count != n_activation in network",
        });
    }
    for ((p_layer, p_dims), (e_layer, e_dims)) in proofs_layer_idx_dims.iter().zip(expected.iter())
    {
        if p_layer != e_layer || p_dims != e_dims {
            return Err(SnarkError::ArchitectureMismatch {
                what: "activation step (layer_idx, dims) mismatch",
            });
        }
    }
    Ok(())
}

/// Verify all per-pass commit vectors have the lengths implied
/// by the public architecture.
pub(crate) fn check_pass_commit_lengths(
    pass_com: &crate::snark::commitment::commit::PassCommitments,
    arch: &NetworkArchitecture,
) -> Result<(), SnarkError> {
    let (n_lin, n_act) = step_counts(arch);
    let chain = chain_len(arch);
    if pass_com.chain_a.len() != chain || pass_com.chain_b_acc.len() != chain {
        return Err(SnarkError::ArchitectureMismatch {
            what: "chain length",
        });
    }
    if pass_com.linear_a_w.len() != n_lin
        || pass_com.linear_a_b.len() != n_lin
        || pass_com.linear_prod_w.len() != n_lin
    {
        return Err(SnarkError::ArchitectureMismatch {
            what: "linear commit count",
        });
    }
    if pass_com.activation_a_pos.len() != n_act
        || pass_com.activation_a_d_doubled.len() != n_act
        || pass_com.activation_bias_doubled.len() != n_act
        || pass_com.activation_bias_delta.len() != n_act
    {
        return Err(SnarkError::ArchitectureMismatch {
            what: "activation commit count",
        });
    }
    Ok(())
}
