//! Per-activation-layer relaxation-soundness gadgets.
//!
//! The cert generator commits per-layer relaxation tensors
//! `(d_lower, b_lower, d_upper, b_upper)` and the backward-pass
//! gadgets consume them as inputs. This module is the SNARK's
//! relaxation-line check: it certifies that those committed tensors
//! actually bound the per-neuron activation on `[preact_lower,
//! preact_upper]`. ReLU and the S-shape (sigmoid/tanh) cases get
//! separate sub-gadgets.
//!
//! Per ReLU layer the gadgets are:
//!
//! * `prove_relu_lower_offset` / `verify_relu_lower_offset` —
//!   `b_lower[j] = 0` via one Hyrax open at a random point.
//! * `relu_d_boolean::{prove,verify}_relu_d_boolean` —
//!   `d_lower[j] ∈ {0, s_d}` via a degree-3 sumcheck.
//! * `relu_upper_endpoint::{prove,verify}_relu_upper_endpoint` —
//!   upper-line dominates ReLU at the two preactivation endpoints
//!   (convexity then lifts validity to the whole interval).
//!
//! Per sigmoid/tanh layer the gadgets are:
//!
//! * `sshape_endpoint::{prove,verify}_sshape_*` — four endpoint
//!   checks `(line_upper(l) ≥ σ(l))`, `(line_upper(u) ≥ σ(u))`,
//!   `(line_lower(l) ≤ σ(l))`, `(line_lower(u) ≤ σ(u))`.
//! * `sshape_critical_point::{prove,verify}_sshape_critical_point` —
//!   tangency at the FD-resolved critical point inside `[l, u]`.
//!
//! Pure-math helpers (envelope lookup, critical-point search, split
//! arithmetic) live in [`sshape_helpers`].

pub(crate) mod proof;
pub(crate) mod prove;
pub(crate) mod verify;

pub(crate) mod relu_d_boolean;
pub(crate) mod relu_upper_endpoint;

pub(crate) mod sshape_critical_point;
pub(crate) mod sshape_endpoint;
pub(crate) mod sshape_helpers;

pub use proof::ReluLowerOffsetProof;
pub use relu_d_boolean::ReluDBooleanProof;
pub use relu_upper_endpoint::ReluUpperEndpointProof;
pub use sshape_critical_point::SshapeCriticalPointProof;
pub use sshape_endpoint::SshapeEndpointProof;

pub(crate) use prove::prove_relu_lower_offset;
pub(crate) use relu_d_boolean::{prove_relu_d_boolean, verify_relu_d_boolean};
pub(crate) use relu_upper_endpoint::{prove_relu_upper_endpoint, verify_relu_upper_endpoint};
pub(crate) use sshape_critical_point::{prove_sshape_critical_point, verify_sshape_critical_point};
pub(crate) use sshape_endpoint::{
    prove_sshape_lower_at_lower, prove_sshape_lower_at_upper, prove_sshape_upper_at_lower,
    prove_sshape_upper_at_upper, quant_cert_sshape_endpoints_ok, verify_sshape_lower_at_lower,
    verify_sshape_lower_at_upper, verify_sshape_upper_at_lower, verify_sshape_upper_at_upper,
};
pub(crate) use verify::verify_relu_lower_offset;
