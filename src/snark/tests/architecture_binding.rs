//! Tampers caught by architecture binding: the verifier derives every
//! per-step `(layer_idx, log_dims)` and every commit `n_vars` from the
//! public `network` / `property` and rejects mismatches as
//! `SnarkError::ArchitectureMismatch`. These tampers should fail
//! up-front, before any sumcheck/PCS work.

use super::fixtures::{expect_reject_after_tamper, prove_small_relu};
use crate::snark::errors::SnarkError;

#[test]
fn full_pass_chain_rejects_tampered_linear_step_log_dims() {
    // Tamper linear-step `a_old_log_dims.0` to a non-matching value.
    let mut p = prove_small_relu();
    if let Some(chain) = p.proof.linear_backward_lower.as_mut() {
        assert!(!chain.is_empty());
        chain[0].a_old_log_dims.0 += 1;
    }
    let err = expect_reject_after_tamper(&p, "tampered linear log_dims");
    assert!(
        matches!(err, SnarkError::ArchitectureMismatch { .. }),
        "tampered linear log_dims must be rejected by architecture binding, got {err:?}"
    );
}

#[test]
fn full_pass_chain_rejects_tampered_activation_step_log_dims() {
    // Tamper activation-step `a_old_log_dims.1` to a non-matching value.
    let mut p = prove_small_relu();
    if let Some(chain) = p.proof.activation_backward_lower.as_mut() {
        assert!(!chain.is_empty());
        chain[0].a_old_log_dims.1 += 1;
    }
    let err = expect_reject_after_tamper(&p, "tampered activation log_dims");
    assert!(
        matches!(err, SnarkError::ArchitectureMismatch { .. }),
        "tampered activation log_dims must be rejected by architecture binding, got {err:?}"
    );
}

#[test]
fn full_pass_chain_rejects_tampered_activation_matrix_log_dims() {
    // Tamper activation_matrix `log_dims.0` to a non-matching value.
    let mut p = prove_small_relu();
    if let Some(steps) = p.proof.activation_matrix_lower.as_mut() {
        assert!(!steps.is_empty());
        steps[0].log_dims.0 += 1;
    }
    let err = expect_reject_after_tamper(&p, "tampered activation_matrix log_dims");
    assert!(
        matches!(err, SnarkError::ArchitectureMismatch { .. }),
        "tampered activation_matrix log_dims must be rejected, got {err:?}"
    );
}

#[test]
fn full_pass_chain_rejects_tampered_output_bound_n_vars() {
    // Tamper output_bound `n_vars` to a non-matching even value. The
    // verifier derives the expected size from
    // `native_vector_n_vars(n_spec)`.
    let mut p = prove_small_relu();
    if let Some(ob) = p.proof.output_bound_lower.as_mut() {
        ob.n_vars += 2;
    }
    let err = expect_reject_after_tamper(&p, "tampered output_bound n_vars");
    assert!(
        matches!(err, SnarkError::ArchitectureMismatch { .. }),
        "tampered output_bound n_vars must be rejected, got {err:?}"
    );
}
