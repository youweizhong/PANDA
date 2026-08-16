//! Module facade for the SNARK test suite. Tests live under
//! `snark::tests::*`:
//!
//! * [`fixtures`] — networks, sponges, proved-fixture helpers.
//! * [`e2e_roundtrips`] — full prove + verify round-trips.
//! * [`missing_components`] — verifier rejects when a mandatory
//!   `Option<...>` subproof is dropped.
//! * [`full_pass_tampers`] — single-field tampers across the proof.
//! * [`architecture_binding`] — `log_dims` / `n_vars` tampers caught
//!   by architecture binding.
//! * [`primitives`] — sub-protocol primitives tested in isolation.


mod fixtures;

mod architecture_binding;
mod e2e_roundtrips;
mod full_pass_tampers;
mod missing_components;
mod primitives;
