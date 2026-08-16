//! Prover for the per-ReLU-layer `b_lower = 0` gadget.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::Zero;
use ark_serialize::CanonicalSerialize;
use ark_std::rand::RngCore;

use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use crate::snark::activation_gadget::proof::ReluLowerOffsetProof;
use crate::snark::commitment::commit::CommittedAux;
use crate::snark::commitment::pcs_helpers::hyrax_open_at;
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;

/// Prove `b_lower[j] = 0` for every neuron `j` of a single ReLU layer.
///
/// The caller is responsible for invoking this only on ReLU layers
/// (sigmoid/tanh have non-zero canonical `b_lower`) and for driving
/// the same FS sponge state the verifier will replay. The challenge
/// point `r` is squeezed after absorbing `(layer_idx, n_vars,
/// b_lower_commit)`.
pub(crate) fn prove_relu_lower_offset(
    layer_idx: usize,
    b_lower_aux: &CommittedAux,
    b_lower_commit: &<HyraxBn254 as MlPcs>::Commitment,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<ReluLowerOffsetProof, SnarkError> {
    let _timing = crate::timing::scope("relu_gadget");
    let n_vars = b_lower_aux.0.len().trailing_zeros() as usize;
    if !b_lower_aux.0.len().is_power_of_two() {
        return Err(SnarkError::ShapeMismatch {
            what: "crate::snark::activation_gadget::prove_relu_lower_offset: \
                   b_lower MLE length must be a power of two",
        });
    }

    // Defence in depth — the commit is already absorbed earlier by
    // the top-level driver, but binding the gadget call site here
    // makes per-layer FS state explicit.
    sponge.absorb(&(layer_idx as u64));
    sponge.absorb(&(n_vars as u64));
    let mut buf = Vec::new();
    b_lower_commit
        .serialize_compressed(&mut buf)
        .expect("serialize commitment");
    sponge.absorb(&buf);

    let r: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(n_vars);

    let (b_lower_eval, b_lower_open) = hyrax_open_at(
        &params.committer_key,
        b_lower_aux,
        b_lower_commit,
        &r,
        sponge,
        rng,
    )?;

    debug_assert!(
        b_lower_eval.is_zero(),
        "crate::snark::activation_gadget::prove_relu_lower_offset: prover's b_lower MLE \
         evaluated to non-zero — this means the cert generator violated the \
         ReLU canonical invariant `b_lower = 0`. Either the cert is wrong, \
         or this gadget was invoked on a non-ReLU layer."
    );

    Ok(ReluLowerOffsetProof {
        layer_idx,
        b_lower_n_vars: n_vars,
        r,
        b_lower_eval,
        b_lower_open,
    })
}
