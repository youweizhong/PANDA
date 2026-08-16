//! Concretization step. Per layer the prover proves
//!     L = A_+ · x_l + A_- · x_u + d
//!     U = A_+ · x_u + A_- · x_l + d
//! via a sumcheck-based matrix-multiplication proof, with the signed
//! components `A_+ = max(A, 0)` and `A_- = min(A, 0)` bound through
//! the shared ReLU-decomposition lookup.

pub mod concretize;
