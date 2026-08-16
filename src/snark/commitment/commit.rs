//! Commitment data structures and the prover-side committer driver.
//!
//! Defines `TensorCommitments` (the public commit surface) plus
//! `PassCommitments` / `PassProverStates` for the per-pass chain,
//! and the helpers (`commit_matrix`, `commit_vector`,
//! `commit_all_tensors`, `commit_pass`) that walk a [`QuantCert`]
//! and a [`BackwardTrace`] to emit every commit the SNARK needs.
//!
//! Also exposes `absorb_commitments` for transcript binding and the
//! MLE-padding helpers (`pad_*_native`, `pad_*_evals_*`) shared
//! across submodules.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;

use crate::crown::network::Network;
use crate::quantized_crown::{BackwardTrace, QuantCert};
use crate::snark_primitives::finite_field::signed_lift_to_fr;
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use crate::snark::commitment::multilinear_extensions::next_pow2_log;
use crate::snark::errors::SnarkError;

/// Padded MLE evaluations plus the Hyrax `CommitmentState` — what
/// the prover keeps after commit so it can later open at any point.
pub(crate) type CommittedAux = (Vec<Fr>, <HyraxBn254 as MlPcs>::CommitmentState);

/// Commitments to every private tensor in a [`QuantCert`].
/// Each tensor is committed once and reused across all proof steps.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct TensorCommitments {
    /// One per layer; `None` for activation layers.
    pub weight: Vec<Option<<HyraxBn254 as MlPcs>::Commitment>>,
    pub bias: Vec<Option<<HyraxBn254 as MlPcs>::Commitment>>,
    /// One per layer; `None` for linear layers. Carries the four
    /// relaxation commitments `(d_lower, d_upper, b_lower, b_upper)`.
    pub relaxation: Vec<Option<RelaxationCommitments>>,
    pub x_lower: <HyraxBn254 as MlPcs>::Commitment,
    pub x_upper: <HyraxBn254 as MlPcs>::Commitment,
    pub spec_c: <HyraxBn254 as MlPcs>::Commitment,
    pub spec_d: <HyraxBn254 as MlPcs>::Commitment,
    pub target_lower: Option<<HyraxBn254 as MlPcs>::Commitment>,
    pub target_upper: Option<<HyraxBn254 as MlPcs>::Commitment>,
    /// Per-pass chain commits for the lower-bound and upper-bound passes.
    pub pass_lower: Option<PassCommitments>,
    pub pass_upper: Option<PassCommitments>,
}

/// Per-pass chain commits for one bound direction.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PassCommitments {
    /// Chain of `A` commits, length `n_layers + 1`. Index `n_layers`
    /// is the initial `A = spec_c`; index `0` is the final `A` going
    /// into concretize.
    pub chain_a: Vec<<HyraxBn254 as MlPcs>::Commitment>,
    pub chain_b_acc: Vec<<HyraxBn254 as MlPcs>::Commitment>,
    /// Per linear step, in backward-traversal order.
    pub linear_a_w: Vec<<HyraxBn254 as MlPcs>::Commitment>,
    pub linear_a_b: Vec<<HyraxBn254 as MlPcs>::Commitment>,
    /// Per-activation-step sign-selector commits. Retained
    /// deliberately for proof-size stability — no live check consumes
    /// them; see the `ActivationStepTrace` docs in
    /// `quantized_crown::types`.
    pub activation_sel: Vec<<HyraxBn254 as MlPcs>::Commitment>,
    pub concretize_sel: Option<<HyraxBn254 as MlPcs>::Commitment>,
    /// Per activation step: `A_pos = ReLU(A_old)`. The ReLU lookup
    /// proof binds these to `chain_a[layer_idx + 1]`.
    pub activation_a_pos: Vec<<HyraxBn254 as MlPcs>::Commitment>,
    pub concretize_a_pos: Option<<HyraxBn254 as MlPcs>::Commitment>,
    pub concretize_target_doubled: Option<<HyraxBn254 as MlPcs>::Commitment>,
    /// Per linear step: post-rescale `prod_w = b_acc_new − b_acc_old`.
    /// Bound to `linear_a_b[step]` by the rescale gadget.
    pub linear_prod_w: Vec<<HyraxBn254 as MlPcs>::Commitment>,
    /// Per activation step: pre-rescale matrix `a_old · d_pick`.
    pub activation_a_d_doubled: Vec<<HyraxBn254 as MlPcs>::Commitment>,
    /// Per activation step: pre-rescale bias delta `Σ_j a_old · b_pick`.
    pub activation_bias_doubled: Vec<<HyraxBn254 as MlPcs>::Commitment>,
    /// Per activation step: `bias_delta = b_acc_new − b_acc_old`.
    pub activation_bias_delta: Vec<<HyraxBn254 as MlPcs>::Commitment>,
    /// Concretize-step `acc_w = final_target − b_acc_final` (post-rescale).
    pub concretize_acc_w: Option<<HyraxBn254 as MlPcs>::Commitment>,
}

/// Four relaxation-tensor commits for one activation layer.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct RelaxationCommitments {
    pub d_lower: <HyraxBn254 as MlPcs>::Commitment,
    pub d_upper: <HyraxBn254 as MlPcs>::Commitment,
    pub b_lower: <HyraxBn254 as MlPcs>::Commitment,
    pub b_upper: <HyraxBn254 as MlPcs>::Commitment,
}

/// Prover-side state retained across `commit` and `open`: the
/// per-tensor Hyrax `CommitmentState` plus the padded evaluation
/// tables needed at open time.
pub struct ProverPolyStates {
    pub weight: Vec<Option<CommittedAux>>,
    pub bias: Vec<Option<CommittedAux>>,
    /// Per-activation-layer relaxation states; `None` for linear layers.
    pub relaxation: Vec<Option<RelaxationStates>>,
    pub x_lower: CommittedAux,
    pub x_upper: CommittedAux,
    pub spec_c: CommittedAux,
    pub spec_d: CommittedAux,
    /// Per-pass chain states; same index space as the matching commit chain.
    pub pass_lower: Option<PassProverStates>,
    pub pass_upper: Option<PassProverStates>,
}

/// Prover state for one activation layer's four relaxation commits.
pub struct RelaxationStates {
    pub d_lower: CommittedAux,
    pub d_upper: CommittedAux,
    pub b_lower: CommittedAux,
    pub b_upper: CommittedAux,
}

/// Prover state mirror of [`PassCommitments`].
pub struct PassProverStates {
    /// `chain_a[i]` = state for the `A` after backward processing
    /// layers `[i, ..., L-1]`; length `n_layers + 1`.
    pub chain_a: Vec<CommittedAux>,
    pub chain_b_acc: Vec<CommittedAux>,
    /// Per linear step in trace order.
    pub linear_a_w: Vec<CommittedAux>,
    pub linear_a_b: Vec<CommittedAux>,
    /// Per activation step in trace order.
    pub activation_sel: Vec<CommittedAux>,
    pub concretize_sel: Option<CommittedAux>,
    pub activation_a_pos: Vec<CommittedAux>,
    pub concretize_a_pos: Option<CommittedAux>,
    pub concretize_target_doubled: Option<CommittedAux>,
    pub linear_prod_w: Vec<CommittedAux>,
    pub activation_a_d_doubled: Vec<CommittedAux>,
    pub activation_bias_doubled: Vec<CommittedAux>,
    pub activation_bias_delta: Vec<CommittedAux>,
    pub concretize_acc_w: Option<CommittedAux>,
}

/// Commit a single padded MLE evaluation table to Hyrax and bundle
/// the prover state. `padded_evals.len()` must be `2^n_vars` with
/// `n_vars` even (Hyrax requires an even number of variables).
pub(crate) fn commit_padded(
    ck: &<HyraxBn254 as MlPcs>::CommitterKey,
    padded_evals: Vec<Fr>,
    rng: &mut impl RngCore,
) -> Result<(<HyraxBn254 as MlPcs>::Commitment, CommittedAux), SnarkError> {
    debug_assert!(padded_evals.len().is_power_of_two());
    debug_assert!(
        padded_evals.len().trailing_zeros().is_multiple_of(2),
        "Hyrax requires an even number of variables"
    );
    let (com, st) = HyraxBn254::commit(ck, &padded_evals, Some(rng))?;
    Ok((com, (padded_evals, st)))
}

/// Commit a `QArray2` at its native Hyrax-friendly MLE size, i.e.
/// `2^native_matrix_n_vars(rows, cols)` rather than padding up to
/// any global maximum.
pub(crate) fn commit_matrix(
    ck: &<HyraxBn254 as MlPcs>::CommitterKey,
    m: &crate::quantization::quantized_array::QArray2,
    rng: &mut impl RngCore,
) -> Result<(<HyraxBn254 as MlPcs>::Commitment, CommittedAux), SnarkError> {
    let (evals, _n_vars, _log_dims) = pad_matrix_native(&m.codes);
    commit_padded(ck, evals, rng)
}

/// Commit a `QArray1` at its native Hyrax-friendly MLE size.
pub(crate) fn commit_vector(
    ck: &<HyraxBn254 as MlPcs>::CommitterKey,
    v: &crate::quantization::quantized_array::QArray1,
    rng: &mut impl RngCore,
) -> Result<(<HyraxBn254 as MlPcs>::Commitment, CommittedAux), SnarkError> {
    let codes: Vec<i128> = v.codes.iter().copied().collect();
    let (evals, _n_vars) = pad_vector_native(&codes);
    commit_padded(ck, evals, rng)
}

/// Native commit `n_vars` from a list of per-axis log-dims:
/// the sum, rounded up to the next even integer ≥ 2.
pub(crate) fn n_vars_from_logs(logs: &[usize]) -> usize {
    let n: usize = logs.iter().sum();
    if n % 2 == 1 {
        n + 1
    } else {
        n.max(2)
    }
}

/// Native commit-time `n_vars` for a 1-D vector of `len` cells:
/// `ceil_log2(len)` bumped up to the next even integer ≥ 2.
pub(crate) fn native_vector_n_vars(len: usize) -> usize {
    let mut n = next_pow2_log(len);
    if n % 2 == 1 {
        n += 1;
    }
    n.max(2)
}

/// Native commit-time `n_vars` for a `(rows × cols)` matrix at the
/// row-major MLE layout: `log_rows + log_cols`, bumped up to the
/// next even integer ≥ 2.
pub(crate) fn native_matrix_n_vars(rows: usize, cols: usize) -> usize {
    let mut n = next_pow2_log(rows) + next_pow2_log(cols);
    if n % 2 == 1 {
        n += 1;
    }
    n.max(2)
}

/// Pad a 1-D code vector to its native Hyrax-friendly MLE size.
/// Returns `(padded_evals, n_vars)`.
pub(crate) fn pad_vector_native(codes: &[i128]) -> (Vec<Fr>, usize) {
    let n_vars = native_vector_n_vars(codes.len());
    let target = 1usize << n_vars;
    let mut out = vec![Fr::from(0u64); target];
    for (slot, code) in out.iter_mut().zip(codes.iter()).take(codes.len()) {
        *slot = signed_lift_to_fr(*code);
    }
    (out, n_vars)
}

/// Pad a 2-D code matrix to its native Hyrax-friendly MLE size at
/// the row-major layout `cell[i, j] = index i * pow_cols + j`,
/// zero-padding past the natural `pow_rows * pow_cols` block.
/// Returns `(evals, n_vars, (log_rows, log_cols))`.
pub(crate) fn pad_matrix_native(codes: &ndarray::Array2<i128>) -> (Vec<Fr>, usize, (usize, usize)) {
    let rows = codes.nrows();
    let cols = codes.ncols();
    let log_rows = next_pow2_log(rows);
    let log_cols = next_pow2_log(cols);
    let pow_cols = 1usize << log_cols;
    let n_vars = native_matrix_n_vars(rows, cols);
    let target = 1usize << n_vars;
    let mut out = vec![Fr::from(0u64); target];
    for i in 0..rows {
        for j in 0..cols {
            out[i * pow_cols + j] = signed_lift_to_fr(codes[[i, j]]);
        }
    }
    (out, n_vars, (log_rows, log_cols))
}

/// Pad a 2-D matrix to next-pow2 per axis (no extra max-padding).
/// Returns `(evals, log_rows + log_cols)`.
pub(crate) fn pad_matrix_evals_2d(
    m: &crate::quantization::quantized_array::QArray2,
) -> (Vec<Fr>, usize) {
    let rows = m.nrows();
    let cols = m.ncols();
    let log_rows = next_pow2_log(rows);
    let log_cols = next_pow2_log(cols);
    let pow_rows = 1usize << log_rows;
    let pow_cols = 1usize << log_cols;
    let mut out = vec![Fr::from(0u64); pow_rows * pow_cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * pow_cols + j] = signed_lift_to_fr(m.codes[[i, j]]);
        }
    }
    (out, log_rows + log_cols)
}

/// Pad a 1-D code vector to next-pow2. Returns `(evals, log_n)`.
pub(crate) fn pad_vector_evals_1d(
    v: &crate::quantization::quantized_array::QArray1,
) -> (Vec<Fr>, usize) {
    let log_n = next_pow2_log(v.codes.len());
    let pow_n = 1usize << log_n;
    let mut out = vec![Fr::from(0u64); pow_n];
    for (slot, &c) in out.iter_mut().zip(v.codes.iter()) {
        *slot = signed_lift_to_fr(c);
    }
    (out, log_n)
}

/// Walk a [`QuantCert`] and its per-pass traces, emitting every
/// Hyrax commit and prover state the SNARK driver needs.
pub(crate) fn commit_all_tensors(
    cert: &QuantCert,
    network: &Network,
    ck: &<HyraxBn254 as MlPcs>::CommitterKey,
    lower_trace: Option<&BackwardTrace>,
    upper_trace: Option<&BackwardTrace>,
    rng: &mut impl RngCore,
) -> Result<(TensorCommitments, ProverPolyStates), SnarkError> {
    let n = network.layers().len();
    let mut weight: Vec<Option<<HyraxBn254 as MlPcs>::Commitment>> = Vec::with_capacity(n);
    let mut bias: Vec<Option<<HyraxBn254 as MlPcs>::Commitment>> = Vec::with_capacity(n);
    let mut relaxation: Vec<Option<RelaxationCommitments>> = Vec::with_capacity(n);
    let mut weight_state: Vec<Option<CommittedAux>> = Vec::with_capacity(n);
    let mut bias_state: Vec<Option<CommittedAux>> = Vec::with_capacity(n);
    let mut relaxation_state: Vec<Option<RelaxationStates>> = Vec::with_capacity(n);

    for i in 0..n {
        match (&cert.weights[i], &cert.biases[i]) {
            (Some(w), Some(b)) => {
                let (w_com, w_aux) = commit_matrix(ck, w, rng)?;
                let (b_com, b_aux) = commit_vector(ck, b, rng)?;
                weight.push(Some(w_com));
                bias.push(Some(b_com));
                weight_state.push(Some(w_aux));
                bias_state.push(Some(b_aux));
                relaxation.push(None);
                relaxation_state.push(None);
            }
            _ => {
                weight.push(None);
                bias.push(None);
                weight_state.push(None);
                bias_state.push(None);
                let rel = cert.relaxations[i]
                    .as_ref()
                    .expect("activation has relaxation");
                let (dl_com, dl_aux) = commit_vector(ck, &rel.d_lower, rng)?;
                let (du_com, du_aux) = commit_vector(ck, &rel.d_upper, rng)?;
                let (bl_com, bl_aux) = commit_vector(ck, &rel.b_lower, rng)?;
                let (bu_com, bu_aux) = commit_vector(ck, &rel.b_upper, rng)?;
                relaxation.push(Some(RelaxationCommitments {
                    d_lower: dl_com,
                    d_upper: du_com,
                    b_lower: bl_com,
                    b_upper: bu_com,
                }));
                relaxation_state.push(Some(RelaxationStates {
                    d_lower: dl_aux,
                    d_upper: du_aux,
                    b_lower: bl_aux,
                    b_upper: bu_aux,
                }));
            }
        }
    }

    let (x_lower, x_lower_aux) = commit_vector(ck, &cert.x_lower, rng)?;
    let (x_upper, x_upper_aux) = commit_vector(ck, &cert.x_upper, rng)?;
    let (spec_c, spec_c_aux) = commit_matrix(ck, &cert.spec_c, rng)?;
    let (spec_d, spec_d_aux) = commit_vector(ck, &cert.spec_d, rng)?;

    let target_lower = cert
        .target_lower
        .as_ref()
        .map(|t| {
            let codes: Vec<i128> = t.codes.iter().copied().collect();
            let (evs, _n_vars) = pad_vector_native(&codes);
            HyraxBn254::commit(ck, &evs, Some(rng)).map(|(c, _)| c)
        })
        .transpose()?;
    let target_upper = cert
        .target_upper
        .as_ref()
        .map(|t| {
            let codes: Vec<i128> = t.codes.iter().copied().collect();
            let (evs, _n_vars) = pad_vector_native(&codes);
            HyraxBn254::commit(ck, &evs, Some(rng)).map(|(c, _)| c)
        })
        .transpose()?;

    let pass_lower_built = match lower_trace {
        Some(t) => Some(commit_pass(t, ck, rng)?),
        None => None,
    };
    let pass_upper_built = match upper_trace {
        Some(t) => Some(commit_pass(t, ck, rng)?),
        None => None,
    };
    let (pass_lower_com, pass_lower_st) = match pass_lower_built {
        Some((c, s)) => (Some(c), Some(s)),
        None => (None, None),
    };
    let (pass_upper_com, pass_upper_st) = match pass_upper_built {
        Some((c, s)) => (Some(c), Some(s)),
        None => (None, None),
    };

    Ok((
        TensorCommitments {
            weight,
            bias,
            relaxation,
            x_lower,
            x_upper,
            spec_c,
            spec_d,
            target_lower,
            target_upper,
            pass_lower: pass_lower_com,
            pass_upper: pass_upper_com,
        },
        ProverPolyStates {
            weight: weight_state,
            bias: bias_state,
            relaxation: relaxation_state,
            x_lower: x_lower_aux,
            x_upper: x_upper_aux,
            spec_c: spec_c_aux,
            spec_d: spec_d_aux,
            pass_lower: pass_lower_st,
            pass_upper: pass_upper_st,
        },
    ))
}

/// Commit every per-step tensor for one pass.
///
/// Chain indexing: for a step at layer `idx`, `a_old = chain_a[idx + 1]`
/// and `a_new = chain_a[idx]`. `chain_a[n_layers]` is the initial
/// `A = spec_c`; `chain_a[0]` is the final `A` going into concretize.
pub(crate) fn commit_pass(
    trace: &BackwardTrace,
    ck: &<HyraxBn254 as MlPcs>::CommitterKey,
    rng: &mut impl RngCore,
) -> Result<(PassCommitments, PassProverStates), SnarkError> {
    let n_layers = trace.linear_steps.len() + trace.activation_steps.len();
    let chain_len = n_layers + 1;
    let mut chain_a_holders: Vec<Option<crate::quantization::quantized_array::QArray2>> =
        vec![None; chain_len];
    let mut chain_b_holders: Vec<Option<crate::quantization::quantized_array::QArray1>> =
        vec![None; chain_len];
    for s in &trace.linear_steps {
        chain_a_holders[s.layer_idx + 1].get_or_insert_with(|| s.a_old.clone());
        chain_a_holders[s.layer_idx].get_or_insert_with(|| s.a_new.clone());
        chain_b_holders[s.layer_idx + 1].get_or_insert_with(|| s.b_acc_old.clone());
        chain_b_holders[s.layer_idx].get_or_insert_with(|| s.b_acc_new.clone());
    }
    for s in &trace.activation_steps {
        chain_a_holders[s.layer_idx + 1].get_or_insert_with(|| s.a_old.clone());
        chain_a_holders[s.layer_idx].get_or_insert_with(|| s.a_new.clone());
        chain_b_holders[s.layer_idx + 1].get_or_insert_with(|| s.b_acc_old.clone());
        chain_b_holders[s.layer_idx].get_or_insert_with(|| s.b_acc_new.clone());
    }
    let chain_a_codes: Vec<crate::quantization::quantized_array::QArray2> = chain_a_holders
        .into_iter()
        .map(|m| m.expect("chain hole"))
        .collect();
    let chain_b_codes: Vec<crate::quantization::quantized_array::QArray1> = chain_b_holders
        .into_iter()
        .map(|m| m.expect("chain hole"))
        .collect();
    let mut chain_a_com = Vec::with_capacity(chain_a_codes.len());
    let mut chain_a_st = Vec::with_capacity(chain_a_codes.len());
    for m in &chain_a_codes {
        let (c, s) = commit_matrix(ck, m, rng)?;
        chain_a_com.push(c);
        chain_a_st.push(s);
    }
    let mut chain_b_com = Vec::with_capacity(chain_b_codes.len());
    let mut chain_b_st = Vec::with_capacity(chain_b_codes.len());
    for v in &chain_b_codes {
        let (c, s) = commit_vector(ck, v, rng)?;
        chain_b_com.push(c);
        chain_b_st.push(s);
    }
    let mut linear_a_w_com = Vec::with_capacity(trace.linear_steps.len());
    let mut linear_a_w_st = Vec::with_capacity(trace.linear_steps.len());
    let mut linear_a_b_com = Vec::with_capacity(trace.linear_steps.len());
    let mut linear_a_b_st = Vec::with_capacity(trace.linear_steps.len());
    let mut linear_prod_w_com = Vec::with_capacity(trace.linear_steps.len());
    let mut linear_prod_w_st = Vec::with_capacity(trace.linear_steps.len());
    for step in &trace.linear_steps {
        let (c, s) = commit_matrix(ck, &step.a_w, rng)?;
        linear_a_w_com.push(c);
        linear_a_w_st.push(s);
        let (c, s) = commit_vector(ck, &step.a_b, rng)?;
        linear_a_b_com.push(c);
        linear_a_b_st.push(s);
        let prod_w = crate::quantization::quantized_array::QArray1::new(
            &step.b_acc_new.codes - &step.b_acc_old.codes,
            step.b_acc_new.scale,
        );
        let (c, s) = commit_vector(ck, &prod_w, rng)?;
        linear_prod_w_com.push(c);
        linear_prod_w_st.push(s);
    }
    let mut activation_sel_com = Vec::with_capacity(trace.activation_steps.len());
    let mut activation_sel_st = Vec::with_capacity(trace.activation_steps.len());
    let mut activation_a_pos_com = Vec::with_capacity(trace.activation_steps.len());
    let mut activation_a_pos_st = Vec::with_capacity(trace.activation_steps.len());
    let mut activation_a_d_com = Vec::with_capacity(trace.activation_steps.len());
    let mut activation_a_d_st = Vec::with_capacity(trace.activation_steps.len());
    let mut activation_bias_d_com = Vec::with_capacity(trace.activation_steps.len());
    let mut activation_bias_d_st = Vec::with_capacity(trace.activation_steps.len());
    let mut activation_bias_delta_com = Vec::with_capacity(trace.activation_steps.len());
    let mut activation_bias_delta_st = Vec::with_capacity(trace.activation_steps.len());
    for step in &trace.activation_steps {
        let (c, s) = commit_matrix(ck, &step.selectors, rng)?;
        activation_sel_com.push(c);
        activation_sel_st.push(s);
        let (cp, sp) = commit_matrix(ck, &step.a_pos, rng)?;
        activation_a_pos_com.push(cp);
        activation_a_pos_st.push(sp);
        let (c, s) = commit_matrix(ck, &step.a_d_doubled, rng)?;
        activation_a_d_com.push(c);
        activation_a_d_st.push(s);
        let (c, s) = commit_vector(ck, &step.bias_delta_doubled, rng)?;
        activation_bias_d_com.push(c);
        activation_bias_d_st.push(s);
        let bias_delta = crate::quantization::quantized_array::QArray1::new(
            &step.b_acc_new.codes - &step.b_acc_old.codes,
            step.b_acc_new.scale,
        );
        let (c, s) = commit_vector(ck, &bias_delta, rng)?;
        activation_bias_delta_com.push(c);
        activation_bias_delta_st.push(s);
    }
    let (
        concretize_sel_com,
        concretize_sel_st,
        concretize_td_com,
        concretize_td_st,
        concretize_a_pos_com,
        concretize_a_pos_st,
        concretize_acc_w_com,
        concretize_acc_w_st,
    ) = if let Some(c) = trace.concretize.as_ref() {
        let (sel_c, sel_s) = commit_matrix(ck, &c.selectors, rng)?;
        let (td_c, td_s) = commit_vector(ck, &c.target_doubled, rng)?;
        let (ap_c, ap_s) = commit_matrix(ck, &c.a_pos, rng)?;
        let acc_w = crate::quantization::quantized_array::QArray1::new(
            &c.final_target.codes - &c.b_acc_final.codes,
            c.final_target.scale,
        );
        let (aw_c, aw_s) = commit_vector(ck, &acc_w, rng)?;
        (
            Some(sel_c),
            Some(sel_s),
            Some(td_c),
            Some(td_s),
            Some(ap_c),
            Some(ap_s),
            Some(aw_c),
            Some(aw_s),
        )
    } else {
        (None, None, None, None, None, None, None, None)
    };

    Ok((
        PassCommitments {
            chain_a: chain_a_com,
            chain_b_acc: chain_b_com,
            linear_a_w: linear_a_w_com,
            linear_a_b: linear_a_b_com,
            activation_sel: activation_sel_com,
            concretize_sel: concretize_sel_com,
            concretize_target_doubled: concretize_td_com,
            activation_a_pos: activation_a_pos_com,
            concretize_a_pos: concretize_a_pos_com,
            linear_prod_w: linear_prod_w_com,
            activation_a_d_doubled: activation_a_d_com,
            activation_bias_doubled: activation_bias_d_com,
            activation_bias_delta: activation_bias_delta_com,
            concretize_acc_w: concretize_acc_w_com,
        },
        PassProverStates {
            chain_a: chain_a_st,
            chain_b_acc: chain_b_st,
            linear_a_w: linear_a_w_st,
            linear_a_b: linear_a_b_st,
            activation_sel: activation_sel_st,
            concretize_sel: concretize_sel_st,
            concretize_target_doubled: concretize_td_st,
            activation_a_pos: activation_a_pos_st,
            concretize_a_pos: concretize_a_pos_st,
            linear_prod_w: linear_prod_w_st,
            activation_a_d_doubled: activation_a_d_st,
            activation_bias_doubled: activation_bias_d_st,
            activation_bias_delta: activation_bias_delta_st,
            concretize_acc_w: concretize_acc_w_st,
        },
    ))
}

/// Absorb the public commitment surface into the transcript so
/// prover and verifier share the same sponge state.
pub(crate) fn absorb_commitments(
    sponge: &mut impl CryptographicSponge,
    commitments: &TensorCommitments,
) {
    fn absorb<S: CryptographicSponge, T: CanonicalSerialize>(sponge: &mut S, t: &T) {
        let mut buf = Vec::new();
        t.serialize_compressed(&mut buf)
            .expect("serialize commitment");
        sponge.absorb(&buf);
    }
    for w in commitments.weight.iter().flatten() {
        absorb(sponge, w);
    }
    for b in commitments.bias.iter().flatten() {
        absorb(sponge, b);
    }
    for r in commitments.relaxation.iter().flatten() {
        absorb(sponge, &r.d_lower);
        absorb(sponge, &r.d_upper);
        absorb(sponge, &r.b_lower);
        absorb(sponge, &r.b_upper);
    }
    absorb(sponge, &commitments.x_lower);
    absorb(sponge, &commitments.x_upper);
    absorb(sponge, &commitments.spec_c);
    absorb(sponge, &commitments.spec_d);
    if let Some(t) = &commitments.target_lower {
        absorb(sponge, t);
    }
    if let Some(t) = &commitments.target_upper {
        absorb(sponge, t);
    }
}
