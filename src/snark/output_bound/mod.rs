//! Output-bound binding gadget. Ties a claimed per-pass output bound
//! to the CROWN-computed `b_acc_final + acc_w`: the [`inequality`]
//! submodule commits a per-cell non-negative `slack` and proves the
//! slack identity at a Fiat-Shamir random `r`. Shared helpers
//! ([`absorb_commitment`], [`build_pos_multiplicities`]) live here.
//!
//! The slack range budget is a per-proof RUNTIME parameter selected by
//! the CALLER of the inequality gadget: the final pass runs at
//! [`crate::snark::params::SnarkParams::out_bound_range_bits`] (the
//! wide window — the only place in the whole proof that uses it) and
//! the hidden-pass preact bounds run at
//! [`crate::snark::params::SnarkParams::gadget_range_bits`]. Both ride
//! in from [`crate::snark::preprocess::Preprocessed::build`]. There is
//! no default, no environment variable, and no fixed supported list —
//! the evaluation reads the budgets from the per-model
//! quantization-parameter JSONs.

mod inequality;

#[cfg(test)]
mod tests;

pub(crate) use inequality::{
    prove_output_bound_inequality, verify_output_bound_inequality, OutputBoundIneqProof,
};

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_serialize::CanonicalSerialize;

use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

/// Absorb a Hyrax commitment into the FS sponge by canonical-serialising it.
pub(crate) fn absorb_commitment(
    sponge: &mut impl CryptographicSponge,
    commitment: &<HyraxBn254 as MlPcs>::Commitment,
) {
    let mut buf = Vec::new();
    commitment
        .serialize_compressed(&mut buf)
        .expect("serialize commitment");
    sponge.absorb(&buf);
}

/// Build the positive-range multiplicity vector: bucket each slack
/// value `v ∈ [0, 2^bits)` into `mults[v]`. Out-of-range cells are
/// dropped; the LogUp identity then fails on them.
pub(crate) fn build_pos_multiplicities(slack: &[i128], bits: usize) -> Vec<u64> {
    let len = 1usize << bits;
    let bound = 1u128 << bits;
    let mut mults = vec![0u64; len];
    for &v in slack {
        if v >= 0 && (v as u128) < bound {
            mults[v as usize] += 1;
        }
    }
    mults
}
