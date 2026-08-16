//! CROWN bound computation, without any SNARK machinery.
//!
//! Defines the [`Network`](network::Network) type, the
//! [`Property`](output_property::Property) describing what the prover
//! wants to certify, and the float-precision reference implementation of
//! backward CROWN in [`float_crown`].

pub mod float_crown;
pub mod network;
pub mod output_property;
