//! All errors the SNARK driver can raise.

use thiserror::Error;

use crate::quantized_crown::QCrownError;
use crate::snark_primitives::logup_gkr::LogUpError;
use crate::snark_primitives::polynomial_commitment::HyraxError;
use crate::snark_primitives::sumcheck::SumcheckError;

#[derive(Debug, Error)]
pub enum SnarkError {
    #[error("PCS error: {0:?}")]
    Pcs(HyraxError),
    #[error("LogUp error: {0}")]
    LogUp(LogUpError),
    #[error("quantized CROWN error: {0}")]
    QCrown(#[source] QCrownError),
    #[error("evaluation table size {got} exceeds the SnarkParams budget (max 2^{max_vars})")]
    BudgetExceeded { got: usize, max_vars: usize },
    #[error("verifier rejected the LogUp range proof")]
    RangeRejected,
    #[error("verifier rejected the LogUp identity (sums don't cancel)")]
    LogUpIdentityFailed,
    /// Fail-closed marker for engineering preconditions that the proof
    /// system rejects rather than scaffolds. Used by low-level gadgets
    /// for scale / shape guards (e.g. a direct Phase 3c call with a
    /// non-matching preact scale). A returned `Reserved` is a real
    /// reject; the verifier will not accept.
    #[error("unsupported configuration (rejected, not scaffolded): {what}")]
    Reserved { what: &'static str },
    #[error("shape mismatch in linear-backward primitive: {what}")]
    ShapeMismatch { what: &'static str },
    #[error("invalid runtime SNARK parameter: {what}")]
    InvalidParameter { what: &'static str },
    #[error("sumcheck error: {0}")]
    Sumcheck(#[source] SumcheckError),
    #[error("verifier transcript diverged from prover (challenge mismatch)")]
    TranscriptMismatch,
    #[error("per-layer linear-backward proof rejected at layer {layer}")]
    LinearLayerRejected { layer: usize },
    #[error("PCS opening proof rejected: {which}")]
    PcsOpenRejected { which: &'static str },
    #[error("per-layer activation-backward proof rejected at layer {layer}")]
    ActivationLayerRejected { layer: usize },
    #[error("concretize-step proof rejected")]
    ConcretizeRejected,
    #[error("sign-correctness lookup identity failed (witness ⊆ table check)")]
    SignLookupIdentityFailed,
    #[error("sign-correctness binding failed (bottom denom ≠ α·A + sel − β)")]
    SignLookupBindingFailed,
    #[error("ReLU-lookup identity failed (witness ⊆ T_ReLU check)")]
    ReluLookupIdentityFailed,
    #[error("ReLU-lookup binding failed (bottom denom ≠ α·A + A_pos − β)")]
    ReluLookupBindingFailed,
    #[error("rescale gadget: identity slack_lo = 2c1·qx − c2(2qz − 1) failed at random r")]
    RescaleIdentityFailed,
    #[error("rescale gadget: range LogUp identity failed (slack_lo not in [0, 2c2))")]
    RescaleRangeIdentityFailed,
    #[error("rescale gadget: range LogUp binding failed (bottom denom ≠ slack_lo(r) − β)")]
    RescaleRangeBindingFailed,
    #[error("rescale gadget: (c1, c2) declared in proof differ from event description")]
    RescaleScaleMismatch,
    #[error("output bound: identity slack(r) = ±(claimed − b_acc_final − acc_w)(r) failed")]
    OutputBoundIdentityFailed,
    #[error("output bound: range LogUp identity / binding failed (slack outside [0, 2^k))")]
    OutputBoundRangeFailed,
    #[error("output bound: equality claim bound(r) = (b_acc_final + acc_w)(r) failed")]
    OutputBoundEqualityFailed,
    #[error("chain init: chain_a[L] != spec_c or chain_b_acc[L] != spec_d")]
    ChainInitMismatch,
    #[error(
        "b_acc step binding: chain_b_acc[layer] != chain_b_acc[layer+1] + delta at layer {layer}"
    )]
    BAccStepBindingFailed { layer: usize },
    #[error("activation matrix-path arithmetic rejected at layer {layer}")]
    ActivationMatrixRejected { layer: usize },
    #[error("public-statement binding: committed spec_c/spec_d/x_box mismatch the public statement quantized at the canonical scales")]
    PublicBindingFailed,
    #[error("architecture binding: {what}")]
    ArchitectureMismatch { what: &'static str },
    #[error("required final-pass proof component is missing: {what}")]
    MissingRequiredComponent { what: &'static str },
    #[error(
        "LogUp table-side binding failed: proof.table_proof.bottom_denom \
         differs from the canonical table MLE evaluation at bottom_point ({which})"
    )]
    LogUpTableNotCanonical { which: &'static str },
    #[error(
        "per-tensor range LogUp: lookup-side bottom_denom does not match \
         (tensor_eval − α) — committed tensor differs from the LogUp witness"
    )]
    PerTensorRangeWitnessNotBound,
    #[error(
        "ReLU relaxation soundness: b_lower MLE evaluated to a non-zero \
         value at the FS-derived point at layer {layer_idx}. The canonical \
         ReLU CROWN construction has b_lower = 0 for every neuron; a non-zero \
         eval means the prover committed an invalid relaxation tensor."
    )]
    RelaxationSoundnessReluLowerOffsetNonZero { layer_idx: usize },
    #[error("relaxation soundness sumcheck round-check failed at round {round}")]
    SumcheckRoundCheckFailed { round: usize },
    #[error("relaxation soundness final identity check failed: {which}")]
    RelaxationSoundnessFinalCheckFailed { which: &'static str },
    #[error(
        "ReLU upper-line endpoint validity failed at layer {layer_idx}: \
         the dequantized upper line `d_upper · x + b_upper` does not \
         dominate `ReLU(x)` at preact endpoint {endpoint:?}. The committed \
         relaxation is invalid for this neuron."
    )]
    RelaxationSoundnessReluUpperEndpointInvalid {
        layer_idx: usize,
        endpoint: &'static str,
    },
    #[error(
        "S-shape relaxation soundness failed at layer {layer_idx} (\
         activation {activation}, side {side}): {detail}. The committed \
         relaxation is invalid for sigmoid/tanh."
    )]
    RelaxationSoundnessSshapeInvalid {
        layer_idx: usize,
        activation: &'static str,
        side: &'static str,
        detail: &'static str,
    },
    #[error(
        "field decode out of range while {which}: a committed Fr value's \
         canonical signed lift does not fit in i128 (or otherwise lies \
         outside the per-tensor range). Earlier checks should have rejected \
         this; reaching here indicates a verifier ordering/validation bug."
    )]
    FieldDecodeOutOfRange { which: &'static str },
}

impl From<HyraxError> for SnarkError {
    fn from(e: HyraxError) -> Self {
        SnarkError::Pcs(e)
    }
}

impl From<LogUpError> for SnarkError {
    fn from(e: LogUpError) -> Self {
        SnarkError::LogUp(e)
    }
}
