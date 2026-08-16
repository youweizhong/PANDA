//! Data structures for the quantized backward-CROWN certificate and the
//! per-step traces the SNARK driver consumes. Includes [`BoundDir`] and
//! [`QCrownError`].

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::crown::network::ActivationKind;
use crate::quantization::quantized_array::{
    QArray1, QArray2, QArrayError, PRECISION_BITS_ARITH_CEILING,
};
use crate::quantization::quantized_scalar::RescaleEntry;
use crate::quantization::scale::Scale;

/// Per-layer scale registry, indexed by the same layer index as `Network`.
/// Linear layers populate `weight` / `bias`; activation layers populate
/// `relax_d` / `relax_b`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerScales {
    pub kind: LayerKind,
    pub weight: Option<Scale>,
    pub bias: Option<Scale>,
    pub relax_d: Option<Scale>,
    pub relax_b: Option<Scale>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerKind {
    Linear,
    Activation(ActivationKind),
}

/// All scales used by a single quantized backward run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantScales {
    /// Working scale carried by the backward CROWN `A`-matrix and bias
    /// accumulator. Every accumulator is rescaled back to this scale.
    pub working: Scale,
    /// Scale of the input box `(x_lower, x_upper)`.
    pub input: Scale,
    /// Scale used to quantize the property's `C` matrix and offset `d`.
    /// Pinned to `working` so the very first `b_acc` is already at the
    /// working scale.
    pub spec: Scale,
    pub layers: Vec<LayerScales>,
}

/// A quantized backward-CROWN certificate: every tensor, every recorded
/// rescale witness.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantCert {
    pub scales: QuantScales,
    pub weights: Vec<Option<QArray2>>,
    pub biases: Vec<Option<QArray1>>,
    pub relaxations: Vec<Option<QuantRelaxation>>,
    pub x_lower: QArray1,
    pub x_upper: QArray1,
    pub spec_c: QArray2,
    pub spec_d: QArray1,
    /// Final target bound at the working scale, one or both sides.
    pub target_lower: Option<QArray1>,
    pub target_upper: Option<QArray1>,
    /// Per-hidden-Linear-layer preactivation bounds at the working scale,
    /// indexed by network layer position. `None` for layers that aren't a
    /// hidden Linear (activation layers and the final output Linear).
    pub preact_lower: Vec<Option<QArray1>>,
    pub preact_upper: Vec<Option<QArray1>>,
    /// Every rescale witness emitted during the backward sweep, in order.
    pub witnesses: Vec<RescaleEntry>,
}

impl QuantCert {
    /// Dequantize the final bound at the working scale.
    pub fn final_bound_real(&self) -> (Option<Array1<f64>>, Option<Array1<f64>>) {
        (
            self.target_lower.as_ref().map(|v| v.to_real()),
            self.target_upper.as_ref().map(|v| v.to_real()),
        )
    }
}

/// Per-layer quantized activation relaxation. Mirrors
/// [`crate::crown::float_crown::ActivationRelaxation`] with codes at the
/// per-layer relaxation scales.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantRelaxation {
    pub kind: ActivationKind,
    pub d_lower: QArray1,
    pub d_upper: QArray1,
    pub b_lower: QArray1,
    pub b_upper: QArray1,
}

/// Captured intermediate state for a single linear-layer step in the
/// backward CROWN chain.
///
/// Both pre-rescale (`a_w`, `a_b`) and post-rescale (`a_new`, `b_acc_new`)
/// are recorded: the SNARK per-layer arithmetic sumcheck binds
/// pre-rescale matrices to the upstream `a_old`, and the rescale gadget
/// binds pre-rescale to post-rescale.
#[derive(Clone, Debug)]
pub struct LinearStepTrace {
    pub layer_idx: usize,
    pub a_old: QArray2,
    pub b_acc_old: QArray1,
    /// Pre-rescale matmul, scale `a_old.scale ⊗ w.scale`.
    pub a_w: QArray2,
    /// Pre-rescale matvec, scale `a_old.scale ⊗ b.scale`.
    pub a_b: QArray1,
    /// Post-rescale, at the working scale.
    pub a_new: QArray2,
    /// Post-rescale plus add, at the working scale.
    pub b_acc_new: QArray1,
}

/// Captured intermediate state for a single activation-layer step.
///
/// The ReLU-decomposition sign-pick witness stores
/// `a_pos = ReLU(a_old) = max(a_old, 0)`; the complement
/// `a_neg = a_old − a_pos` is recoverable linearly and is not committed.
/// The `selectors` field is retained DELIBERATELY: it is still
/// committed by the SNARK (commitment/commit.rs), so removing it would
/// change proof sizes and prover timings versus the recorded
/// evaluation. It carries no live check — sign correctness is proven
/// via the `a_pos` ReLU decomposition.
#[derive(Clone, Debug)]
pub struct ActivationStepTrace {
    pub layer_idx: usize,
    pub a_old: QArray2,
    pub b_acc_old: QArray1,
    /// Boolean sign selectors; still committed (see the struct docs)
    /// but no longer backed by a sign-correctness check.
    pub selectors: QArray2,
    /// `a_pos = ReLU(a_old)`; same shape and scale as `a_old`.
    pub a_pos: QArray2,
    /// Pre-rescale `a_old · d_pick`, at scale `a_old.scale ⊗ relax_d.scale`.
    pub a_d_doubled: QArray2,
    /// Pre-rescale `Σ_j a_old · b_pick`, at scale `a_old.scale ⊗ relax_b.scale`.
    pub bias_delta_doubled: QArray1,
    pub a_new: QArray2,
    pub b_acc_new: QArray1,
}

/// Captured concretize-step state. Same ReLU decomposition convention as
/// [`ActivationStepTrace`]: `a_pos = ReLU(a_final)`.
#[derive(Clone, Debug)]
pub struct ConcretizeTrace {
    pub a_final: QArray2,
    pub b_acc_final: QArray1,
    /// Boolean sign selectors; still committed (see
    /// [`ActivationStepTrace`] docs) but carry no live check.
    pub selectors: QArray2,
    pub a_pos: QArray2,
    /// Pre-rescale concretize accumulator.
    pub target_doubled: QArray1,
    pub final_target: QArray1,
}

/// Trace of one full backward pass (lower or upper).
#[derive(Clone, Debug)]
pub struct BackwardTrace {
    pub linear_steps: Vec<LinearStepTrace>,
    pub activation_steps: Vec<ActivationStepTrace>,
    pub final_target: QArray1,
    pub concretize: Option<ConcretizeTrace>,
}

/// Per-hidden-Linear-layer pass artifacts.
///
/// A hidden Linear layer is a Linear layer immediately followed by an
/// Activation. For each one we run two backward passes — one per
/// direction — starting from an identity spec at the layer's output and
/// walking down to layer 0, then concretizing on the input box. The
/// resulting `preact_<dir>` is the per-neuron preactivation bound at the
/// working scale.
#[derive(Clone, Debug)]
pub struct HiddenLayerPass {
    pub target_layer_idx: usize,
    pub n_spec: usize,
    pub lower_trace: BackwardTrace,
    pub upper_trace: BackwardTrace,
    pub preact_lower: QArray1,
    pub preact_upper: QArray1,
}

/// Which direction of the bound the engine is computing.
#[derive(Copy, Clone, Debug)]
pub enum BoundDir {
    Lower,
    Upper,
}

#[derive(Debug, Error)]
pub enum QCrownError {
    #[error(
        "precision_bits must be in (1, {PRECISION_BITS_ARITH_CEILING}) \
         (Code-width arithmetic guard; the SNARK additionally requires \
         precision_bits < range_table_half_bits at setup), got {bits}"
    )]
    HeadroomOutOfRange { bits: i32 },
    #[error("float CROWN failed: {0}")]
    FloatPlaintext(#[source] crate::crown::float_crown::PlaintextError),
    #[error("quantized arithmetic failed: {0}")]
    QArray(#[source] QArrayError),
    /// A sigmoid/tanh preact endpoint or stationary point fell outside
    /// the σ half-table domain `(-128, 128)`. The cert generator fails
    /// closed instead of falling back to raw float σ — a float
    /// fallback would produce a relaxation the SNARK verifier cannot
    /// reproduce.
    #[error(
        "sigmoid/tanh relaxation at layer {layer_idx}: candidate x = {x_real} is outside the \
         Phase 3a table domain (-128, 128); cert generator fails closed"
    )]
    SshapeRelaxOutOfTableDomain { layer_idx: usize, x_real: f64 },
}
