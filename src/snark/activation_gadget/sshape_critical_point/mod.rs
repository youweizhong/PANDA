//! Sigmoid/tanh critical-point validity (Phase 3c of the PANDA SNARK).
//!
//! Per (sigmoid/tanh layer, line direction), the gadget proves for
//! every neuron that the committed relaxation slope is realised by
//! `σ'` somewhere inside the preact interval. The proof combines a
//! finite-difference slope-match (no `σ'` lookup table), a per-cell
//! split-arith chain for the line/σ gap, an inside-bit gated-gap
//! check, and a d = 0 gate.
//!
//! Together with the Phase 3b endpoint gadget, this binds the
//! committed `(d_line, b_line)` pair to a valid sigmoid/tanh
//! relaxation over `[l, u]`.

mod prover;
#[cfg(test)]
mod tests;
mod types;
mod verifier;
mod witness;

pub use prover::prove_sshape_critical_point;
pub use types::SshapeCriticalPointProof;
pub use verifier::verify_sshape_critical_point;
