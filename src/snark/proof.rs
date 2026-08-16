//! `SnarkProof` and the per-step proof structs.

use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use crate::snark::activation_gadget::{
    ReluDBooleanProof, ReluLowerOffsetProof, ReluUpperEndpointProof, SshapeCriticalPointProof,
    SshapeEndpointProof,
};
use crate::snark::backward_pass::activation_matrix::ActivationMatrixStepProof;
use crate::snark::backward_pass::bias_accumulator::BAccStepProof;
use crate::snark::backward_pass::chain_init::ChainInitProof;
use crate::snark::backward_pass::linear_step::LinearBackwardProof;
use crate::snark::backward_pass::signed_components::relu_lookup;
use crate::snark::backward_pass::signed_components::relu_lookup::ReluStepProof;
use crate::snark::commitment::commit::{PassCommitments, TensorCommitments};
use crate::snark::commitment::public_binding::PublicBindingProof;
use crate::snark::output_bound::OutputBoundIneqProof;

/// SNARK proof: tensor commitments + per-tensor range proofs + every
/// per-step / per-gadget subproof, indexed by side (lower/upper) and
/// layer position.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SnarkProof {
    pub commitments: TensorCommitments,
    /// One range LogUp per public-witness tensor (input box, weights,
    /// biases, ReLU relaxation coefficients), in the canonical iteration
    /// order defined by `prove::build_tensor_range_proofs` /
    /// `verify::verify_tensor_range_proofs`. Each proof binds the LogUp
    /// witness to its committed tensor.
    pub tensor_range_proofs: Vec<crate::snark::commitment::range_per_tensor::TensorRangeProof>,
    /// Per-layer quantization scales, Hyrax-committed and bound into the
    /// FS sponge before any rescale challenge so the prover cannot adapt
    /// scales after seeing the rescale challenges. Conceptually part of
    /// the public statement: the verifier accepts for the network
    /// described by `(commitments, layer_scales, working_scale,
    /// input_scale)`.
    pub layer_scales: LayerScalesHyraxCommit,
    /// Per-layer Hyrax opens of the bundled `LayerScalesHyraxCommit`
    /// column. The verifier reconstructs a synthetic
    /// `LayerScalesCommit` (sentinel-zero for `None` entries) from these
    /// opens and feeds it through the rescale driver and sshape gadgets.
    pub layer_scale_opens: LayerScaleOpens,
    /// Working scale for the (private) target bound. Both prover and
    /// verifier use it to quantize the public threshold
    /// (`Property::lower_threshold` / `upper_threshold`) before the
    /// in-SNARK property check. Also absorbed into the FS sponge. The
    /// bound codes themselves live inside `output_bound_*.claimed_commit`
    /// and are hidden from the public proof surface.
    pub target_scale_c: i64,
    pub target_scale_e: i32,
    /// Per-linear-layer lower-bound backward arithmetic proofs.
    pub linear_backward_lower: Option<Vec<LinearLayerStepProof>>,
    /// Per-linear-layer upper-bound backward arithmetic proofs.
    pub linear_backward_upper: Option<Vec<LinearLayerStepProof>>,
    /// Per-activation-layer arithmetic proofs (b_acc matvec sumcheck).
    pub activation_backward_lower: Option<Vec<ActivationLayerStepProof>>,
    pub activation_backward_upper: Option<Vec<ActivationLayerStepProof>>,
    /// Final concretize-step proofs (one per side requested).
    pub concretize_lower: Option<ConcretizeStepProof>,
    pub concretize_upper: Option<ConcretizeStepProof>,
    /// ReLU lookups: one per activation step (binds `A_pos = ReLU(A_old)`)
    /// and one for concretize.
    pub relu_lower_activation: Option<Vec<relu_lookup::ReluStepProof>>,
    pub relu_upper_activation: Option<Vec<relu_lookup::ReluStepProof>>,
    pub relu_lower_concretize: Option<relu_lookup::ReluStepProof>,
    pub relu_upper_concretize: Option<relu_lookup::ReluStepProof>,
    /// Per-pass rescale-gadget proofs in canonical event order.
    pub rescale_lower: Option<Vec<crate::snark::rescaling::RescaleEventProof>>,
    pub rescale_upper: Option<Vec<crate::snark::rescaling::RescaleEventProof>>,
    /// Final-pass output-bound inequality proofs (per direction). Binds
    /// the public claimed bound to `b_acc_final + acc_w` via
    /// `claimed_upper ≥ computed` / `claimed_lower ≤ computed`.
    pub output_bound_lower: Option<OutputBoundIneqProof>,
    pub output_bound_upper: Option<OutputBoundIneqProof>,
    /// Per-pass chain-init binding (`chain_a[L] = spec_c` and
    /// `chain_b_acc[L] = spec_d`).
    pub chain_init_lower: Option<ChainInitProof>,
    pub chain_init_upper: Option<ChainInitProof>,
    /// Per-step b_acc-update binding
    /// (`chain_b_acc[layer] = chain_b_acc[layer+1] + delta`).
    pub b_acc_step_lower: Option<Vec<BAccStepProof>>,
    pub b_acc_step_upper: Option<Vec<BAccStepProof>>,
    /// Per-activation-step matrix-path arithmetic (proves
    /// `a_d_doubled[i,j] = A_pos·d_pos + A_neg·d_neg` cell-wise).
    pub activation_matrix_lower: Option<Vec<ActivationMatrixStepProof>>,
    pub activation_matrix_upper: Option<Vec<ActivationMatrixStepProof>>,
    /// Public-statement binding: ties committed `spec_c`, `spec_d`,
    /// `x_lower`, `x_upper` to canonical MLE evals of the public
    /// `(property, x_box)` quantized at deterministic scales.
    pub public_binding: Option<PublicBindingProof>,

    /// Per-hidden-Linear-layer preactivation bound proofs, in network
    /// forward order. Each entry pairs a lower- and upper-direction
    /// backward CROWN proof from an identity spec at that layer's
    /// output, walking down to the input box. Reuses the final-pass
    /// weight/bias/relaxation commits in `commitments`; only chain
    /// tensors and per-step intermediates are newly committed per pass.
    pub hidden_passes: Vec<HiddenLayerPassProof>,

    /// Per-ReLU-layer lower-line offset validity proof: asserts
    /// `b_lower[j] = 0` for every neuron via a single Hyrax open +
    /// `eval == 0` check at a Fiat-Shamir-derived point. Closes the gap
    /// where a malicious prover could commit a non-zero `b_lower` to
    /// bias the lower line and shift the proven preact bound.
    pub relu_lower_offset_proofs: Vec<ReluLowerOffsetProof>,
    /// Per-ReLU-layer proof that `d_lower[j] ∈ {0, s_d}` for every
    /// neuron (canonical CROWN ReLU slope is real-valued 0 or 1, i.e.
    /// integer code 0 or `s_d`). Degree-3 sumcheck on
    /// `Σ_j eq(j, r)·d[j]·(s_d − d[j]) = 0`.
    pub relu_d_boolean_proofs: Vec<ReluDBooleanProof>,
    /// Per-ReLU-layer upper-line endpoint validity proof: enforces
    /// `d_upper · preact + b_upper ≥ ReLU(preact)` at both endpoints of
    /// every neuron. By convexity of ReLU + line affineness, endpoint
    /// validity implies validity over the whole `[l, u]` interval.
    /// Implemented as a working-scale slack + LogUp range check +
    /// degree-3 sumcheck binding.
    pub relu_upper_endpoint_proofs: Vec<ReluUpperEndpointProof>,

    /// Per sigmoid/tanh layer, four endpoint validity proofs (one per
    /// `(line, side)` combination, in network forward order; ReLU
    /// layers are excluded). Each vec has the same length:
    ///
    /// - `sshape_upper_at_lower_proofs` — `U(l[j]) ≥ σ_upper(l[j])`
    /// - `sshape_upper_at_upper_proofs` — `U(u[j]) ≥ σ_upper(u[j])`
    /// - `sshape_lower_at_lower_proofs` — `σ_lower(l[j]) ≥ L(l[j])`
    /// - `sshape_lower_at_upper_proofs` — `σ_lower(u[j]) ≥ L(u[j])`
    ///
    /// Together with `sshape_critical_point_*_proofs` below, these
    /// imply validity over the whole `[l, u]` interval by S-shape
    /// geometry + line affineness.
    pub sshape_upper_at_lower_proofs: Vec<SshapeEndpointProof>,
    pub sshape_upper_at_upper_proofs: Vec<SshapeEndpointProof>,
    pub sshape_lower_at_lower_proofs: Vec<SshapeEndpointProof>,
    pub sshape_lower_at_upper_proofs: Vec<SshapeEndpointProof>,

    /// Per sigmoid/tanh layer × line direction, the FD-based critical-
    /// point validity proof (one entry per layer × {upper, lower} in
    /// network forward order). See `sshape_critical_point` for the FD
    /// reduction.
    pub sshape_critical_point_upper_proofs: Vec<SshapeCriticalPointProof>,
    pub sshape_critical_point_lower_proofs: Vec<SshapeCriticalPointProof>,
}

/// Per-layer scales, indexed by layer position. For each layer only the
/// relevant scale entries are meaningful; the others are sentinel
/// `(c=0, e=0)`:
///
/// - At a `Layer::Linear` index `i`: `weight[i]` and `bias[i]` are
///   meaningful; `relax_d[i]`, `relax_b[i]` are sentinel.
/// - At a `Layer::Activation` index `i`: `relax_d[i]` and `relax_b[i]`
///   are meaningful; `weight[i]`, `bias[i]` are sentinel.
///
/// The per-class `Vec<i64>/<i32>` values no longer live in the public
/// `SnarkProof`; they are bundled into one Fr column
/// (`pack_layer_scales_to_fr`) and Hyrax-committed once. This struct
/// survives only as a prover-side helper for organizing the unpacked
/// values; it is not serialized into the proof.
/// `CanonicalSerialize`/`CanonicalDeserialize` are kept for internal
/// helpers and tests.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct LayerScalesCommit {
    pub weight_c: Vec<i64>,
    pub weight_e: Vec<i32>,
    pub bias_c: Vec<i64>,
    pub bias_e: Vec<i32>,
    pub relax_d_c: Vec<i64>,
    pub relax_d_e: Vec<i32>,
    pub relax_b_c: Vec<i64>,
    pub relax_b_e: Vec<i32>,
}

/// Public-facing Hyrax commitment to the bundled per-layer scales. The
/// verifier reconstructs the `(c, e)` values it needs from
/// `LayerScaleOpens` (opens at deterministic `(class, layer_idx)`
/// indices into this column).
///
/// Packing layout: 8 contiguous blocks of `n_layers` Fr entries
/// (`signed_lift_to_fr`-encoded), in canonical class order
/// `[weight_c, weight_e, bias_c, bias_e, relax_d_c, relax_d_e,
/// relax_b_c, relax_b_e]`, then zero-padded to a power-of-two column
/// length (even-vars compliant for Hyrax).
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct LayerScalesHyraxCommit {
    pub n_layers: u32,
    pub n_vars: u32,
    pub commit: <HyraxBn254 as MlPcs>::Commitment,
}

/// Per-layer scale open: two Hyrax opens (c and e) at the deterministic
/// unit-vector indices for this layer's scale class in the bundled
/// `LayerScalesHyraxCommit` column.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct LayerScaleOpenCE {
    pub c_eval: Fr,
    pub e_eval: Fr,
    pub c_open: <HyraxBn254 as MlPcs>::Proof,
    pub e_open: <HyraxBn254 as MlPcs>::Proof,
}

/// Per-layer × per-class scale opens. Indexed by layer position;
/// `Some(...)` for layers where the scale class is meaningful per the
/// public architecture, `None` for sentinel layers.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct LayerScaleOpens {
    pub weight: Vec<Option<LayerScaleOpenCE>>,
    pub bias: Vec<Option<LayerScaleOpenCE>>,
    pub relax_d: Vec<Option<LayerScaleOpenCE>>,
    pub relax_b: Vec<Option<LayerScaleOpenCE>>,
}

/// Encoding identifiers for the 8 packed scale classes. Order MUST
/// match the canonical pack order in `pack_layer_scales_to_fr`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleClass {
    WeightC = 0,
    WeightE = 1,
    BiasC = 2,
    BiasE = 3,
    RelaxDC = 4,
    RelaxDE = 5,
    RelaxBC = 6,
    RelaxBE = 7,
}

/// One linear layer's backward step proof in the chain.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct LinearLayerStepProof {
    pub layer_idx: usize,
    pub a_old_log_dims: (usize, usize),
    pub w_log_dims: (usize, usize),
    pub proof: LinearBackwardProof,
}

/// Per-activation-layer backward step proof.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ActivationLayerStepProof {
    pub layer_idx: usize,
    /// `(log n_spec, log n_neurons)` of the per-step matrices.
    pub a_old_log_dims: (usize, usize),
    /// `r_spec` chosen by Fiat-Shamir for the b_acc claim.
    pub r_spec: Vec<Fr>,
    /// `delta_b_doubled~(r_spec)` — the pre-rescale matvec sum at
    /// `r_spec`. The rescale gadget ties this to
    /// `(b_acc_new − b_acc_old)~(r_spec)`.
    pub delta_b_claim: Fr,
    /// Eq-two-product sumcheck
    /// `Σ_x eq · (A_pos · pos_line + A_neg · neg_line)`.
    pub sumcheck: relu_lookup::EqTwoProductProof,
    /// Batched Hyrax open of `(A_old, A_pos)` at `r_full`.
    pub a_full_batched_open: <HyraxBn254 as MlPcs>::Proof,
    /// Batched Hyrax open of `(b_pos_line, b_neg_line)` at `r_j`.
    pub b_j_batched_open: <HyraxBn254 as MlPcs>::Proof,
    /// Hyrax open of `activation_bias_doubled[step]` at `r_spec`. Binds
    /// `delta_b_claim` to the committed pre-rescale tensor so the
    /// rescale gadget operates on the same data as the activation step.
    pub bias_doubled_open: <HyraxBn254 as MlPcs>::Proof,
    pub a_old_eval: Fr,
}

/// Concretize-step backward proof.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ConcretizeStepProof {
    /// `(log n_spec, log n_input)` of the per-pass concretize matrix.
    pub a_final_log_dims: (usize, usize),
    pub r_spec: Vec<Fr>,
    /// `target_doubled~(r_spec)` — the pre-rescale concretize sum.
    pub target_doubled_claim: Fr,
    pub sumcheck: relu_lookup::EqTwoProductProof,
    /// Batched Hyrax open of `(A_final, A_pos)` at `r_full`.
    pub a_full_batched_open: <HyraxBn254 as MlPcs>::Proof,
    /// Batched Hyrax open of `(x_pos, x_neg)` at `r_j`.
    pub x_j_batched_open: <HyraxBn254 as MlPcs>::Proof,
    /// Hyrax open of `concretize_target_doubled` at `r_spec`. Binds the
    /// claim to the committed pre-rescale tensor.
    pub target_doubled_open: <HyraxBn254 as MlPcs>::Proof,
    pub a_final_eval: Fr,
}

/// Chain-init-from-identity proof. Used by hidden-layer passes where
/// the initial chain_a is the canonical identity matrix (no committed
/// `spec_c` to bind against). The verifier evaluates the canonical
/// identity MLE at a Fiat-Shamir-derived point and asserts the prover's
/// chain_a open at the same point matches.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ChainInitFromIdentityProof {
    /// Native commit `n_vars` for `chain_a[target_layer_idx + 1]`.
    pub a_n_vars: usize,
    /// Native commit `n_vars` for `chain_b_acc[target_layer_idx + 1]`.
    pub b_n_vars: usize,
    pub r_a: Vec<Fr>,
    pub r_b: Vec<Fr>,
    pub chain_a_eval: Fr,
    pub chain_b_eval: Fr,
    pub chain_a_open: <HyraxBn254 as MlPcs>::Proof,
    pub chain_b_open: <HyraxBn254 as MlPcs>::Proof,
}

/// One hidden-Linear-layer preactivation bound proof. Internally runs
/// the same per-step gadgets as the final pass (linear / activation
/// backward, ReLU LogUp, rescale, b_acc step, activation matrix,
/// concretize, output_bound inequality) on a shorter chain — from
/// layer 0 up to the target Linear layer — with an identity spec_c
/// instead of the property's C matrix.
///
/// Reuses the final pass's weight/bias/relaxation/x_box commits.
/// Newly committed per pass: `pass_lower`/`pass_upper` chain tensors
/// plus per-step intermediates.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HiddenLayerPassProof {
    /// Network index of the target Linear layer this pass bounds.
    pub target_layer_idx: usize,
    /// `n_spec` for this hidden pass = output dimension of the target
    /// Linear layer (and the row count of the identity spec_c).
    pub n_spec: usize,
    /// Per-direction pass commitments. Public-witness commits are not
    /// re-committed here — the verifier indexes them out of the final
    /// pass's `commitments`.
    pub pass_lower: PassCommitments,
    pub pass_upper: PassCommitments,
    /// Hyrax commits to the per-pass preactivation bounds. The preact
    /// values are private witnesses; downstream gadgets bind these
    /// commits via Hyrax opens at gadget-internal FS-derived points and
    /// a `(preact, relu_fr) ⊆ T_ReLU` LogUp inside the ReLU gadget.
    pub preact_lower_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub preact_upper_commit: <HyraxBn254 as MlPcs>::Commitment,
    /// Number of MLE variables for the per-pass preact commits
    /// (even-padded). Required to lift short opening points into the
    /// commit's native shape.
    pub preact_n_vars: u32,
    /// Pins the identity spec_c via canonical-MLE check.
    pub chain_init_lower: ChainInitFromIdentityProof,
    pub chain_init_upper: ChainInitFromIdentityProof,
    pub linear_backward_lower: Vec<LinearLayerStepProof>,
    pub linear_backward_upper: Vec<LinearLayerStepProof>,
    pub activation_backward_lower: Vec<ActivationLayerStepProof>,
    pub activation_backward_upper: Vec<ActivationLayerStepProof>,
    pub concretize_lower: ConcretizeStepProof,
    pub concretize_upper: ConcretizeStepProof,
    pub relu_lower_activation: Vec<ReluStepProof>,
    pub relu_upper_activation: Vec<ReluStepProof>,
    pub relu_lower_concretize: ReluStepProof,
    pub relu_upper_concretize: ReluStepProof,
    pub rescale_lower: Vec<crate::snark::rescaling::RescaleEventProof>,
    pub rescale_upper: Vec<crate::snark::rescaling::RescaleEventProof>,
    pub b_acc_step_lower: Vec<BAccStepProof>,
    pub b_acc_step_upper: Vec<BAccStepProof>,
    pub activation_matrix_lower: Vec<ActivationMatrixStepProof>,
    pub activation_matrix_upper: Vec<ActivationMatrixStepProof>,
    pub output_bound_lower: OutputBoundIneqProof,
    pub output_bound_upper: OutputBoundIneqProof,
}
