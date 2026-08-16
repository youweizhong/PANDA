//! Per-cell sign machinery for the activation step.
//!
//! The activation step needs to pick the lower or upper relaxation
//! line per cell according to the sign of the parent coefficient
//! `A[i, j]`. Instead of committing a boolean selector, we use a
//! ReLU decomposition `A_+ = max(A, 0)`, `A_- = A - A_+`; the lookup
//! gadget below proves `A_+ = ReLU(A)` cell-wise so the sign stays
//! hidden.
//!
//! * `relu_lookup` — `(A, A_+) ⊆ T_ReLU` LogUp gadget plus shared
//!   eq-weighted two-product sumcheck used by activation_step and
//!   activation_matrix.
//! * `driver` — runs one `relu_lookup` per activation step and one
//!   per concretize call.

pub mod driver;
pub mod relu_lookup;
