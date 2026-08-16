//! LogUp-GKR lookup argument over BN254's scalar field.
//!
//! Given a witness column `w[0..N]`, a public table column `t[0..N]`,
//! and prover-supplied multiplicities `m[0..N]`, LogUp proves Häbock's
//! fractional sum identity
//!
//! ```text
//!   Σ_i (-1) / (w[i] - α)  +  Σ_j m[j] / (t[j] - α)  =  0
//! ```
//!
//! at a verifier-chosen `α`. We discharge the two sums with a layered
//! GKR fraction-tree: bottom-layer leaves are the per-position
//! fractions; each upper layer pair-adds via `(a/b + c/d)`; each
//! transition is verified by a degree-3 sumcheck on
//! `eq(x, r) · [num_lo · denom_hi + num_hi · denom_lo + λ · denom_lo · denom_hi]`,
//! with `λ` batching the per-layer (num, denom) claim. After the
//! sumcheck the prover sends the four MLE evaluations
//! `{num_lo, num_hi, denom_lo, denom_hi}` at the challenge point; a
//! fresh `β` folds them into the next layer's claim.
//!
//! All sumcheck plumbing reuses [`crate::snark_primitives::sumcheck`].
//! Fiat-Shamir runs through the same `CryptographicSponge` as every
//! other gadget in this codebase.
//!
//! # Module split
//!
//! - `types` — data structures (`Fraction`, `LogUpLayer`,
//!   `LogUpCircuit`, `LayerProof`, `LogUpProof`, `LogUpError`) plus the
//!   shared `absorb_round_poly` transcript helper.
//! - `prove` — `prove_circuit` and the `prove_lookup` convenience
//!   wrapper.
//! - `verify` — `verify_circuit` / `verify_circuit_with_top` plus the
//!   private `eq` evaluation helper.

mod prove;
mod types;
mod verify;

#[cfg(test)]
mod tests;

pub use prove::{prove_circuit, prove_lookup};
pub use types::{Fraction, LayerProof, LogUpCircuit, LogUpError, LogUpLayer, LogUpProof};
pub use verify::{verify_circuit, verify_circuit_with_top};
