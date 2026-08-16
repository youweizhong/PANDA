//! PANDA SNARK: gadgets, prover, and verifier.
//!
//! Module layout (one submodule per paper section):
//!
//! * `backward_pass` — §4 backward pass: activation step, matrix
//!   update, ReLU signed components, linear-step sumcheck, and the
//!   hidden-layer chain driver / bias accumulator.
//! * `concretization` — §4 concretization of `L = A_+ · x_l + A_- · x_u + d`.
//! * `output_bound` — final inequality / equality on the claimed
//!   output bound vs the public threshold.
//! * `activation_gadget` — §5 four-point activation gadget (ReLU upper-line
//!   endpoint validity, `d ∈ {0, s_d}` booleanity, S-shape endpoints,
//!   FD critical-point check).
//! * `rescaling` — per-event rescale proofs.
//! * `commitment` — SNARK-internal plumbing (commit state, MLE helpers,
//!   PCS wrappers, public-statement binding, architecture view,
//!   per-tensor range LogUp, canonical-MLE evaluation, layer-scale opens).
//! * `prove` / `verify` — top-level drivers.

pub(crate) mod activation_gadget;
pub(crate) mod backward_pass;
pub(crate) mod commitment;
pub(crate) mod concretization;
pub(crate) mod output_bound;
pub(crate) mod rescaling;

pub(crate) mod errors;
pub(crate) mod params;
pub(crate) mod preprocess;
pub(crate) mod proof;
pub(crate) mod prove;
pub(crate) mod verify;

#[cfg(test)]
mod tests;

// Re-exports forming the crate's public SNARK API.

pub use backward_pass::linear_step::{
    prove_linear_backward, verify_linear_backward, LinearBackwardCommitContext,
    LinearBackwardOpenings, LinearBackwardProof, LinearBackwardVerifyContext,
};
pub use commitment::commit::{
    PassCommitments, PassProverStates, ProverPolyStates, RelaxationCommitments, RelaxationStates,
    TensorCommitments,
};
pub use errors::SnarkError;
pub use params::{SnarkParams, SnarkStatement, VerifiedBound};
pub use preprocess::{
    default_sigma_scales, validate_input_scale, validate_sigma_scales, Preprocessed, SigmaTables,
    MAX_TABLE_BITS,
    SIGMA_V_SCALE_LOG2_MAX, SIGMA_X_SCALE_LOG2_MAX,
};
pub use proof::{ActivationLayerStepProof, ConcretizeStepProof, LinearLayerStepProof, SnarkProof};
pub use prove::prove_final_pass;
pub use verify::verify_final_pass;

/// Would `prove_final_pass` accept this quantized cert at the given
/// range budgets? A crypto-free check (microseconds) that reproduces
/// the prover's range checks exactly, using only cert data. Lets
/// the certified-radius bisection find the largest *SNARK-provable* epsilon —
/// and the smallest sufficient range budgets — without running a single proof.
///
/// Two prover rejects not implied by `target_lower > 0` are modeled:
///
/// * the sigmoid/tanh endpoint gadgets' split-arith range checks, which
///   run at the per-neuron `gadget_range_bits` budget, and
/// * the output-bound property check's `prop_slack` LogUp, which runs
///   at the `out_bound_range_bits` budget: for a zero threshold (every
///   PANDA robustness spec) the slack IS the target bound code, so
///   every `target_lower` code must fit `[0, 2^bits)` (dually
///   `-target_upper` when an upper claim is present). Very robust images
///   have large margins whose codes overflow the 19-bit window — that is
///   exactly what forces the tanh_20 nets to the 21-bit out-bound
///   budget (while the per-neuron gadgets stay at 19).
///
/// Both budgets must match what the proof stage will use (runtime
/// public parameters; the evaluation reads them from the per-model
/// quantization-parameter JSONs). `sigma_x_scale_log2` /
/// `sigma_v_scale_log2` must be the same σ scales the cert was built at
/// (the cert generator forced the working scale to `2^sigma_x_scale_log2`),
/// so the endpoint replicas index the identical σ tables.
pub fn quantized_cert_snark_provable(
    network: &crate::crown::network::Network,
    cert: &crate::quantized_crown::QuantCert,
    out_bound_range_bits: usize,
    gadget_range_bits: usize,
    sigma_x_scale_log2: i32,
    sigma_v_scale_log2: i32,
) -> bool {
    quant_cert_out_bound_codes_ok(cert, out_bound_range_bits)
        && activation_gadget::quant_cert_sshape_endpoints_ok(
            network,
            cert,
            gadget_range_bits,
            sigma_x_scale_log2,
            sigma_v_scale_log2,
        )
        .is_ok()
}

/// Crypto-free replica of the output-bound property check's range
/// constraint (see `output_bound::inequality`): with a zero public
/// threshold, `prop_slack = target_code - 0` (lower) / `0 - target_code`
/// (upper) must lie in `[0, 2^bits)`. Mirrors the prover's
/// `build_pos_multiplicities` semantics — out-of-range cells fail the
/// LogUp identity and reject the proof.
fn quant_cert_out_bound_codes_ok(
    cert: &crate::quantized_crown::QuantCert,
    out_bound_range_bits: usize,
) -> bool {
    let bound = 1i128 << out_bound_range_bits;
    let lower_ok = cert
        .target_lower
        .as_ref()
        .is_none_or(|t| t.codes.iter().all(|&c| c >= 0 && c < bound));
    let upper_ok = cert
        .target_upper
        .as_ref()
        .is_none_or(|t| t.codes.iter().all(|&c| -c >= 0 && -c < bound));
    lower_ok && upper_ok
}
