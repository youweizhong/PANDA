//! Integer-arithmetic foundation shared by [`crate::quantized_crown`] and
//! [`crate::snark`]. Every value the prover commits to is an integer code
//! with an attached power-of-two [`scale::Scale`] (`c · 2^e`); this module
//! collects the data types and helpers that make that arithmetic correct.
//!
//! * [`scale`] — the per-tensor `(c, e)` power-of-two scale plus rescale-
//!   ratio arithmetic.
//! * [`quantized_scalar`] — a single scalar code with an attached `Scale`.
//! * [`quantized_array`] — `ndarray` of integer codes sharing one `Scale`.
//! * [`range_obligations`] — the range-check obligations collected from a
//!   quantized cert and fed into the SNARK's LogUp range gadgets.

pub mod quantized_array;
pub mod quantized_scalar;
pub mod range_obligations;
pub mod scale;
