//! PANDA: scalable zero-knowledge proofs of robustness and fairness
//! for neural networks, without revealing private model parameters.
//!
//! Core modules:
//!
//!   * [`file_formats`] — The JSON schemas used to configure the prover and verifier (`fixture.json` and `quant_params.json`).
//!   * [`crown`] — CROWN bound computation (no SNARK): the
//!     [`Network`] and
//!     [`Property`] types, plus the
//!     float-precision reference implementation in [`crown::float_crown`].
//!   * [`quantized_crown`] — integer / fixed-point CROWN certificate
//!     generator. Takes a Network + Property and produces a
//!     [`quantized_crown::QuantCert`].
//!   * [`quantization`] — generic integer-arithmetic foundation shared
//!     by the CROWN crates.
//!   * [`snark_primitives`] — finite-field helpers, polynomial
//!     commitments (trait + Hyrax), sumcheck, and LogUp-GKR.
//!   * [`snark`] — the PANDA SNARK proper: gadgets, prover, verifier.

// Structural style lints deliberately not "fixed": the prover/verifier
// signatures mirror the paper's protocol objects (commitments, opening
// states, transcripts), so collapsing arguments into structs or aliasing
// the PCS-generic types would obscure the protocol correspondence, and
// the batch loops index several same-length vectors in lockstep.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]

pub mod file_formats;

pub mod crown;
pub mod quantization;
pub mod quantized_crown;
pub mod snark;
pub mod snark_primitives;
pub mod timing;

// Ergonomic re-exports so callers can write panda::Network instead of
// panda::crown::network::Network. Hidden from rustdoc to keep the main page clean.
#[doc(hidden)]
pub use crown::float_crown::{
    backward_bound, concretize, recompute_target_bounds, relax_layer, ActivationRelaxation,
    BackwardBound, PlainCert,
};
#[doc(hidden)]
pub use crown::network::{ActivationKind, Layer, Network, NetworkError};
#[doc(hidden)]
pub use crown::output_property::{Property, Side};

#[doc(hidden)]
pub use quantized_crown::{
    quantized_backward_bound, quantized_backward_bound_scaled, LayerKind, LayerScales, QCrownError,
    QuantCert, QuantRelaxation, QuantScales,
};
#[doc(hidden)]
pub use snark::{
    default_sigma_scales, quantized_cert_snark_provable, validate_sigma_scales, MAX_TABLE_BITS,
    SIGMA_V_SCALE_LOG2_MAX, SIGMA_X_SCALE_LOG2_MAX,
};
