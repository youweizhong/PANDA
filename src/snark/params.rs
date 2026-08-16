//! Public parameters, statement, and verifier output.

use std::sync::Arc;

use ark_std::rand::RngCore;
use ndarray::Array1;

use crate::crown::network::{ActivationKind, Layer, Network};
use crate::crown::output_property::Property;
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use crate::snark::errors::SnarkError;
use crate::snark::preprocess::Preprocessed;

/// SNARK setup parameters: Hyrax committer/verifier keys, the working
/// bit-budget, and a handle to the preprocessed lookup tables built for
/// this proof's runtime table parameters.
///
/// The table sizes are RUNTIME public parameters carried by the
/// [`Preprocessed`] instance: `range_table_half_bits` (signed range /
/// ReLU tables cover `[-2^k, 2^k)`), `out_bound_range_bits`
/// (the final-pass output-bound slack/property LogUps use
/// `[0, 2^bits)`, once per proof), and `gadget_range_bits` (every
/// per-neuron gadget range LogUp, chunk modulus, scale precondition,
/// and the hidden-pass preact-bound inequalities). Prover and verifier
/// must construct their params from the same three values; the
/// verifier recomputes every table from them and pins each proof's
/// claimed table widths against them.
#[derive(Clone)]
pub struct SnarkParams {
    pub committer_key: <HyraxBn254 as MlPcs>::CommitterKey,
    pub verifier_key: <HyraxBn254 as MlPcs>::VerifierKey,
    pub max_num_vars: usize,
    pub precision_bits: i32,
    /// Output-bound range budget for THIS proof's final pass (copied
    /// from `preprocessed.out_bound_range_bits` at setup).
    pub out_bound_range_bits: usize,
    /// Per-neuron gadget range budget for THIS proof (copied from
    /// `preprocessed.gadget_range_bits` at setup).
    pub gadget_range_bits: usize,
    /// Log2 of the sigmoid/tanh input scale for THIS proof (copied from
    /// `preprocessed.sigma_x_scale_log2` at setup). The cert generator
    /// forces the working scale to `2^sigma_x_scale_log2` and the
    /// sshape gadgets index the σ tables at it.
    pub sigma_x_scale_log2: i32,
    /// Log2 of the sigmoid/tanh value scale for THIS proof (copied from
    /// `preprocessed.sigma_v_scale_log2` at setup).
    pub sigma_v_scale_log2: i32,
    /// Log2 of the input-box quantization scale for THIS proof (copied
    /// from `preprocessed.input_scale_log2` at setup). `None` = the
    /// default `pick_scale_pow2(x_box, precision_bits)`; `Some(e)` forces
    /// `input = 2^e` on both the cert generator and the public-binding
    /// re-derivation. Public parameter; prover and verifier must agree.
    pub input_scale_log2: Option<i32>,
    /// Public lookup tables for this proof's runtime table parameters.
    /// Shared between prover and verifier.
    pub preprocessed: Arc<Preprocessed>,
}

impl SnarkParams {
    /// Half-width of the signed range / ReLU tables for this
    /// proof (runtime public parameter, from the preprocessed tables).
    pub fn range_table_half_bits(&self) -> i32 {
        self.preprocessed.range_table_half_bits
    }

    /// Pick `max_num_vars` as an upper bound over every Hyrax-committed
    /// tensor's native size. Each commit pads only to its own pow-2
    /// shape, so the key only needs to be wide enough for the single
    /// biggest tensor in the proof.
    ///
    /// Every table-size parameter rides in `preprocessed` (see
    /// [`Preprocessed::build`]); there are no environment or
    /// compile-time defaults. `precision_bits` must leave at least one
    /// bit of headroom inside the signed range table
    /// (`precision_bits < range_table_half_bits`), the invariant that
    /// keeps every honestly-quantized code strictly inside the table.
    pub fn setup(
        network: &Network,
        property: &Property,
        precision_bits: i32,
        preprocessed: Arc<Preprocessed>,
        rng: &mut impl RngCore,
    ) -> Result<Self, SnarkError> {
        let out_bound_range_bits = preprocessed.out_bound_range_bits;
        let gadget_range_bits = preprocessed.gadget_range_bits;
        let sigma_x_scale_log2 = preprocessed.sigma_x_scale_log2;
        let sigma_v_scale_log2 = preprocessed.sigma_v_scale_log2;
        let input_scale_log2 = preprocessed.input_scale_log2;
        if precision_bits <= 1 || precision_bits >= preprocessed.range_table_half_bits {
            return Err(SnarkError::InvalidParameter {
                what: "precision_bits must be in (1, range_table_half_bits)",
            });
        }
        // Native n_vars helpers (mirror commit::native_*_n_vars).
        let bump_even = |n: usize| {
            let n = if n % 2 == 1 { n + 1 } else { n };
            n.max(2)
        };
        let log = |n: usize| -> usize {
            if n <= 1 {
                0
            } else {
                n.next_power_of_two().trailing_zeros() as usize
            }
        };
        let nv_vec = |len: usize| bump_even(log(len));
        let nv_mat = |rows: usize, cols: usize| bump_even(log(rows) + log(cols));

        let n_spec = property.c_matrix.nrows();
        let n_out = network.output_dim();
        let mut max_nv = 0usize;
        // Public-statement commits.
        max_nv = max_nv.max(nv_mat(n_spec, n_out));
        max_nv = max_nv.max(nv_vec(property.d_vector.len()));
        // Per-layer weight + bias + relaxation + chain commits.
        let mut a_cols = n_out;
        for layer in network.layers().iter().rev() {
            match layer {
                Layer::Linear { weight, bias } => {
                    let w_rows = weight.nrows();
                    let w_cols = weight.ncols();
                    max_nv = max_nv.max(nv_mat(w_rows, w_cols));
                    max_nv = max_nv.max(nv_vec(bias.len()));
                    max_nv = max_nv.max(nv_mat(n_spec, a_cols));
                    max_nv = max_nv.max(nv_vec(n_spec));
                    a_cols = w_cols;
                    max_nv = max_nv.max(nv_mat(n_spec, w_cols));
                }
                Layer::Activation { kind } => {
                    max_nv = max_nv.max(nv_vec(a_cols));
                    max_nv = max_nv.max(nv_mat(n_spec, a_cols));
                    max_nv = max_nv.max(nv_vec(n_spec));
                    if matches!(kind, ActivationKind::Sigmoid | ActivationKind::Tanh) {
                        // Envelope multiplicity commits are sized by the
                        // fixed SigmaTables domain, independent of the
                        // runtime table bits — the key must cover them
                        // even at small range_table_half_bits.
                        let sigma_len = preprocessed
                            .sigma
                            .sigmoid_upper_fr
                            .len()
                            .max(preprocessed.sigma.tanh_upper_fr.len());
                        max_nv = max_nv.max(nv_vec(sigma_len));
                    }
                }
            }
        }
        let input_dim = network.input_dim();
        max_nv = max_nv.max(nv_vec(input_dim));

        // LogUp multiplicity vectors, sized from the runtime table
        // parameters carried by `preprocessed`.
        let logup_mult_n_vars = {
            let nv = (preprocessed.range_table_half_bits as usize) + 1;
            if nv % 2 == 1 {
                nv + 1
            } else {
                nv
            }
        };
        let out_bound_mult_n_vars = {
            let nv = out_bound_range_bits;
            if nv % 2 == 1 { nv + 1 } else { nv }.max(2)
        };
        let gadget_mult_n_vars = {
            let nv = gadget_range_bits;
            if nv % 2 == 1 { nv + 1 } else { nv }.max(2)
        };
        max_nv = max_nv
            .max(logup_mult_n_vars)
            .max(out_bound_mult_n_vars)
            .max(gadget_mult_n_vars);

        // Per-event rescale gadget mult tensors are sized at `2·c2`,
        // which can exceed `2^range_table_half_bits` for high-precision
        // benchmarks. A 4-var margin (16x slack) covers typical c2
        // growth without doing a full pre-cert sizing pass.
        let rescale_margin: usize = 4;
        max_nv = max_nv.max(logup_mult_n_vars + rescale_margin);

        let num_vars = max_nv.max(2);
        debug_assert!(num_vars.is_multiple_of(2));
        let (ck, vk) = HyraxBn254::setup(num_vars, rng).map_err(SnarkError::Pcs)?;
        Ok(Self {
            committer_key: ck,
            verifier_key: vk,
            max_num_vars: num_vars,
            precision_bits,
            out_bound_range_bits,
            gadget_range_bits,
            sigma_x_scale_log2,
            sigma_v_scale_log2,
            input_scale_log2,
            preprocessed,
        })
    }
}

/// Public statement: what the verifier knows before any proof.
///
/// **Contract on `network`.** Only the *architecture* (layer count,
/// kinds, dimensions) is public; weight and bias values are private and
/// reach the verifier solely through the prover's commitments inside
/// `SnarkProof`. The `Network` here is a convenience container — the
/// verifier reads only shape information from it and must never inspect
/// weight/bias values. The audit invariant: substituting a same-shape
/// but different-valued `Network` must not change the verifier's
/// accept/reject decision for an honestly-produced proof. The
/// architecture-only verifier view lives in `SnarkVerifierStatement`
/// below; the prover still receives the full network here.
#[derive(Clone, Debug)]
pub struct SnarkStatement {
    pub network: Network,
    pub property: Property,
    pub x_lower: Array1<f64>,
    pub x_upper: Array1<f64>,
}

impl SnarkStatement {
    /// Project to the public-only verifier view: drop weights and
    /// biases, keep only architecture, property, and input box.
    pub fn to_verifier(&self) -> SnarkVerifierStatement {
        SnarkVerifierStatement {
            architecture: self.network.architecture(),
            property: self.property.clone(),
            x_lower: self.x_lower.clone(),
            x_upper: self.x_upper.clone(),
        }
    }
}

/// Verifier-facing public statement: carries the architecture-only view
/// of the network (layer shapes and activation kinds) and never weights
/// or biases. Constructed via [`SnarkStatement::to_verifier`];
/// `verify_final_pass` consumes this type, so the verifier API is
/// type-level prevented from reading private weights.
#[derive(Clone, Debug)]
pub struct SnarkVerifierStatement {
    pub architecture: crate::crown::network::NetworkArchitecture,
    pub property: Property,
    pub x_lower: Array1<f64>,
    pub x_upper: Array1<f64>,
}

/// Verifier output. The final claimed bound is not exposed at the
/// verifier boundary: the in-SNARK property check inside
/// `OutputBoundIneqProof` binds the (private) claimed bound to the
/// public threshold (`Property::lower_threshold` /
/// `upper_threshold`, defaulting to zero). A successful
/// `verify_final_pass` means "`C·y + d ≥ lower_threshold` and/or
/// `C·y + d ≤ upper_threshold` holds for every `x` in the input box."
/// The struct shape is preserved for API compatibility; both `lower`
/// and `upper` are always `None`.
#[derive(Clone, Debug)]
pub struct VerifiedBound {
    pub lower: Option<Array1<f64>>,
    pub upper: Option<Array1<f64>>,
}
