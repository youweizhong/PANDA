//! Verifier for the per-ReLU-layer `b_lower = 0` gadget.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::Zero;
use ark_serialize::CanonicalSerialize;

use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use crate::snark::activation_gadget::proof::ReluLowerOffsetProof;
use crate::snark::commitment::pcs_helpers::hyrax_verify_at;
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;

/// Verify a [`ReluLowerOffsetProof`] against a public `b_lower_commit`
/// and the architecture-derived `expected_b_lower_n_vars`.
///
/// The caller drives the same sponge state the prover used and
/// indexes `b_lower_commit` from
/// `commitments.relaxation[layer_idx].b_lower`. Restrict invocation
/// to `ActivationKind::ReLU` layers (sigmoid/tanh canonical
/// `b_lower ≠ 0` would correctly reject here).
pub(crate) fn verify_relu_lower_offset(
    proof: &ReluLowerOffsetProof,
    expected_layer_idx: usize,
    b_lower_commit: &<HyraxBn254 as MlPcs>::Commitment,
    expected_b_lower_n_vars: usize,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    if proof.layer_idx != expected_layer_idx {
        return Err(SnarkError::ArchitectureMismatch {
            what: "relaxation_soundness: layer_idx mismatch",
        });
    }
    if proof.b_lower_n_vars != expected_b_lower_n_vars {
        return Err(SnarkError::ArchitectureMismatch {
            what: "relaxation_soundness: b_lower_n_vars mismatch with public architecture",
        });
    }
    if proof.r.len() != proof.b_lower_n_vars {
        return Err(SnarkError::ShapeMismatch {
            what: "relaxation_soundness: r length must equal b_lower_n_vars",
        });
    }

    sponge.absorb(&(proof.layer_idx as u64));
    sponge.absorb(&(proof.b_lower_n_vars as u64));
    let mut buf = Vec::new();
    b_lower_commit
        .serialize_compressed(&mut buf)
        .expect("serialize commitment");
    sponge.absorb(&buf);
    let expected_r = sponge.squeeze_field_elements::<Fr>(proof.b_lower_n_vars);
    if expected_r != proof.r {
        return Err(SnarkError::TranscriptMismatch);
    }

    let ok = hyrax_verify_at(
        &params.verifier_key,
        b_lower_commit,
        &proof.r,
        proof.b_lower_eval,
        &proof.b_lower_open,
        proof.b_lower_n_vars,
        sponge,
    )?;
    if !ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "crate::snark::activation_gadget::b_lower",
        });
    }

    if !proof.b_lower_eval.is_zero() {
        return Err(SnarkError::RelaxationSoundnessReluLowerOffsetNonZero {
            layer_idx: proof.layer_idx,
        });
    }

    Ok(())
}
