//! Quantized backward CROWN for flat MLPs — the integer mirror of
//! [`crate::crown::float_crown`].
//!
//! Given a plaintext network, output property, and input box, the
//! pipeline:
//!
//!   1. runs the float backward CROWN engine to get pre-activation
//!      bounds and canonical activation relaxations,
//!   2. picks per-tensor power-of-two scales (via
//!      [`crate::quantization::quantized_array::pick_scale_pow2`]) for
//!      every weight, bias, relaxation slope, and the input box,
//!   3. quantizes all tensors,
//!   4. re-runs the backward sweep in integer codes with
//!      accumulate-first matmul and one rescale per output element,
//!      logging every rescale witness for the SNARK driver,
//!   5. concretizes on the quantized input box and reports the lower /
//!      upper bound at the working scale.
//!
//! The output is a [`QuantCert`] containing the integer-coded final
//! bound, every per-tensor scale, and the full
//! [`crate::quantization::quantized_scalar::RescaleEntry`] witness
//! stream.
//!
//! This module produces an honest quantized certificate. Proving it to a
//! verifier, lifting codes into `Fr`, and running LogUp-GKR range
//! lookups all live in `crate::snark::*`.

mod backward;
mod relaxation;
mod scales;
mod types;

pub use backward::{
    quantized_backward_bound, quantized_backward_bound_scaled, quantized_backward_bound_with_trace,
    quantized_backward_bound_with_trace_scaled,
};
pub use types::{
    ActivationStepTrace, BackwardTrace, BoundDir, ConcretizeTrace, HiddenLayerPass, LayerKind,
    LayerScales, LinearStepTrace, QCrownError, QuantCert, QuantRelaxation, QuantScales,
};
