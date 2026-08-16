//! SNARK-internal commitment plumbing: Hyrax PCS wrappers, MLE
//! helpers, and the per-statement bindings every gadget consumes.
//!
//! * `commit` — typed `CommittedAux` plus `TensorCommitments` /
//!   `PassCommitments` and the driver that walks a `QuantCert`.
//! * `multilinear_extensions` — small MLE helpers (eq tables,
//!   eval at a point, partial eval).
//! * `pcs_helpers` — open / verify wrappers, including a batched
//!   single-point open across multiple commits.
//! * `public_binding` — bind the public statement (architecture,
//!   property, input box, precision) into committed evals.
//! * `architecture` — public-shape view of the network for the
//!   verifier (no weights or biases).
//! * `range_per_tensor` — per-committed-witness range LogUp.
//! * `table_mle` — closed-form MLE evaluation for public lookup
//!   tables (range, ReLU, σ envelope).
//! * `layer_scale_api` — typed accessor for per-layer `(c, e)`
//!   scales that the verifier consumes.

pub mod architecture;
pub mod commit;
pub mod layer_scale_api;
pub mod multilinear_extensions;
pub mod pcs_helpers;
pub mod public_binding;
pub mod range_per_tensor;
pub mod table_mle;
