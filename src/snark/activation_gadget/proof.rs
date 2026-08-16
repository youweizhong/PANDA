//! Proof struct for the per-ReLU-layer `b_lower = 0` gadget.

use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

/// Per-ReLU-layer proof that `b_lower[j] = 0` for every neuron `j`.
///
/// Carries one Hyrax open of the committed `b_lower` at a Fiat-Shamir
/// point `r`, plus the claimed eval (asserted to be `0` by the
/// verifier). The `b_lower` commitment lives in
/// `commitments.relaxation[layer_idx]` — this proof carries no
/// commitment of its own.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ReluLowerOffsetProof {
    /// Network index of the activation layer this proof binds.
    pub layer_idx: usize,
    /// Commit-side `n_vars` of `b_lower` for this layer. The verifier
    /// re-derives this from the public architecture and rejects on
    /// mismatch.
    pub b_lower_n_vars: usize,
    /// FS-derived BE point at which `b_lower` is opened.
    pub r: Vec<Fr>,
    /// Claimed `b_lower~(r)`. Verifier asserts this equals zero.
    pub b_lower_eval: Fr,
    /// Hyrax open of the committed `b_lower` at `r`.
    pub b_lower_open: <HyraxBn254 as MlPcs>::Proof,
}
