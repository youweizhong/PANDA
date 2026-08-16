//! Generic cryptographic primitives the PANDA SNARK is built on.
//! Nothing here is PANDA-specific; each submodule would slot into a
//! different SNARK project unchanged.
//!
//! * [`finite_field`] — BN254 `Fr` helpers (signed lift, serializable
//!   wrapper).
//! * [`polynomial_commitment`] — [`MlPcs`](polynomial_commitment::MlPcs)
//!   trait plus the default [`HyraxBn254`](polynomial_commitment::HyraxBn254)
//!   instantiation.
//! * [`hyrax_pcs`] — native Hyrax implementation that exposes per-row
//!   Pedersen randomness for batched dot-product opens.
//! * [`sumcheck`] — degree-2/3/4 round polynomials and an
//!   inner-product sumcheck.
//! * [`logup_gkr`] — LogUp-GKR fraction-tree prover and verifier.

pub mod finite_field;
pub mod hyrax_pcs;
pub mod logup_gkr;
pub mod polynomial_commitment;
pub mod sumcheck;
