//! SNARK proofs for the CROWN backward pass.
//!
//! Each module proves one step of the chain that walks the network
//! back from the output property `(C, d)` to the input box. Linear
//! steps use a sumcheck-based matmul+matvec; activation steps use a
//! per-cell signed pick between the lower and upper relaxation
//! envelopes, with the per-cell sign hidden by the ReLU
//! decomposition `A = A_+ + A_-`.
//!
//! Chain-level helpers live alongside the step gadgets:
//! `chain_init` seeds `(A, d)` from the property at the output
//! layer, `bias_accumulator` ties the running `d` across steps, and
//! `hidden_pass/` runs the same machinery on a shorter chain that
//! ends at a hidden layer (used to bind committed preactivation
//! bounds).

pub mod activation_matrix;
pub mod activation_step;
pub mod bias_accumulator;
pub mod chain_init;
pub mod hidden_pass;
pub mod linear_step;
pub mod signed_components;
