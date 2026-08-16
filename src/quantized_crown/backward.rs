//! Quantized backward-CROWN driver. Two public entry points:
//!
//! - [`quantized_backward_bound`] — produces a [`QuantCert`].
//! - [`quantized_backward_bound_with_trace`] — same plus per-pass
//!   [`BackwardTrace`]s capturing the pre-/post-rescale intermediates the
//!   SNARK driver consumes.
//!
//! The integer engine ([`run_quant_pass`], [`apply_activation_quant`],
//! [`concretize_quant`]) is private to this submodule.

use ndarray::{Array1, Array2};

use super::types::{
    ActivationStepTrace, BackwardTrace, BoundDir, ConcretizeTrace, HiddenLayerPass,
    LinearStepTrace, QCrownError, QuantCert, QuantRelaxation, QuantScales,
};
use crate::crown::float_crown::backward_bound as float_backward_bound;
use crate::crown::network::{Layer, Network};
use crate::crown::output_property::Property;
use crate::quantization::quantized_array::{
    matmul, matvec, pick_scale_pow2, quantize_matrix, quantize_vector, quantize_vector_ceil,
    quantize_vector_floor, QArray1, QArray2, QArrayError, PRECISION_BITS_ARITH_CEILING,
};
use crate::quantization::quantized_scalar::{Code, RescaleEntry};

/// Run the full quantized backward-CROWN pipeline and return only the
/// certificate, at the DEFAULT sigmoid/tanh table scales for
/// `precision_bits` (see [`crate::snark::default_sigma_scales`]). Thin
/// wrapper over [`quantized_backward_bound_with_trace`] that discards
/// trace data, so the cert returned here is byte-for-byte what the
/// SNARK driver sees at the same scales.
pub fn quantized_backward_bound(
    network: &Network,
    property: &Property,
    x_lower: &Array1<f64>,
    x_upper: &Array1<f64>,
    precision_bits: i32,
) -> Result<QuantCert, QCrownError> {
    let (s_x, s_v) = crate::snark::default_sigma_scales(precision_bits);
    quantized_backward_bound_scaled(
        network, property, x_lower, x_upper, precision_bits, None, s_x, s_v,
    )
}

/// Like [`quantized_backward_bound`] but at explicit sigmoid/tanh table
/// scales `(sigma_x_scale_log2, sigma_v_scale_log2)`. `s_x` sets the
/// working scale for sigmoid/tanh nets (and hence the output-bound
/// drift floor); the SNARK prover must build its `Preprocessed` at the
/// same scales. Has no effect on ReLU-only nets, which never touch the
/// σ tables.
///
/// `input_scale_log2` optionally forces the input-box quantization scale
/// to `2^input_scale_log2` instead of the default
/// `pick_scale_pow2(x_box, precision_bits)`. A finer input scale shrinks
/// the outward-rounding drift of the eps-ball (the dominant input-box drift
/// term at low precision); `None` keeps the auto scale. The SNARK
/// verifier recomputes the same scale from the public parameter, so the
/// prover and verifier must carry the identical value.
#[allow(clippy::too_many_arguments)]
pub fn quantized_backward_bound_scaled(
    network: &Network,
    property: &Property,
    x_lower: &Array1<f64>,
    x_upper: &Array1<f64>,
    precision_bits: i32,
    input_scale_log2: Option<i32>,
    sigma_x_scale_log2: i32,
    sigma_v_scale_log2: i32,
) -> Result<QuantCert, QCrownError> {
    let (cert, _lower_trace, _upper_trace, _hidden_passes) =
        quantized_backward_bound_with_trace_scaled(
            network,
            property,
            x_lower,
            x_upper,
            precision_bits,
            input_scale_log2,
            sigma_x_scale_log2,
            sigma_v_scale_log2,
        )?;
    Ok(cert)
}

/// Run the full quantized backward-CROWN pipeline, returning the cert
/// alongside per-pass linear-step, activation-step, and concretize
/// traces, plus one [`HiddenLayerPass`] per hidden Linear layer, at the
/// DEFAULT σ scales for `precision_bits`.
///
/// The SNARK driver consumes these to build per-layer matmul / matvec /
/// eq-product sumchecks and rescale-gadget proofs.
pub fn quantized_backward_bound_with_trace(
    network: &Network,
    property: &Property,
    x_lower: &Array1<f64>,
    x_upper: &Array1<f64>,
    precision_bits: i32,
) -> Result<
    (
        QuantCert,
        Option<BackwardTrace>,
        Option<BackwardTrace>,
        Vec<HiddenLayerPass>,
    ),
    QCrownError,
> {
    let (s_x, s_v) = crate::snark::default_sigma_scales(precision_bits);
    quantized_backward_bound_with_trace_scaled(
        network,
        property,
        x_lower,
        x_upper,
        precision_bits,
        None,
        s_x,
        s_v,
    )
}

/// [`quantized_backward_bound_with_trace`] at explicit σ scales. The
/// sigmoid/tanh working scale is forced to `2^sigma_x_scale_log2` and
/// the relaxation envelopes are read from the σ tables built at
/// `(sigma_x_scale_log2, sigma_v_scale_log2)`, so the cert is provable
/// under a `Preprocessed`/`SnarkParams` carrying the same public scales.
#[allow(clippy::too_many_arguments)]
pub fn quantized_backward_bound_with_trace_scaled(
    network: &Network,
    property: &Property,
    x_lower: &Array1<f64>,
    x_upper: &Array1<f64>,
    precision_bits: i32,
    input_scale_log2: Option<i32>,
    sigma_x_scale_log2: i32,
    sigma_v_scale_log2: i32,
) -> Result<
    (
        QuantCert,
        Option<BackwardTrace>,
        Option<BackwardTrace>,
        Vec<HiddenLayerPass>,
    ),
    QCrownError,
> {
    if precision_bits <= 1 || precision_bits >= PRECISION_BITS_ARITH_CEILING {
        return Err(QCrownError::HeadroomOutOfRange {
            bits: precision_bits,
        });
    }
    let plain = float_backward_bound(network, property, x_lower, x_upper)
        .map_err(QCrownError::FloatPlaintext)?;
    // Pick the working scale first: the relax-d / relax-b cap below
    // requires it. Without the cap the SNARK's slack identity gains
    // fractional terms when q_b > q_w.
    let input_scale = match input_scale_log2 {
        Some(e) => crate::quantization::scale::Scale::from_pow2(e),
        None => pick_scale_pow2(
            &concat_for_scale(&[x_lower.as_slice().unwrap(), x_upper.as_slice().unwrap()]),
            precision_bits,
        ),
    };
    let spec_values: Vec<f64> = property
        .c_matrix
        .iter()
        .copied()
        .chain(property.d_vector.iter().copied())
        .collect();
    let mut working = pick_scale_pow2(&spec_values, precision_bits);
    // The SNARK σ-envelope half-tables are indexed at scale
    // `s_x = 2^sigma_x_scale_log2` (a runtime public parameter).
    // Sigmoid/tanh endpoint and critical-point gadgets need the cert's
    // working scale to equal `s_x`; otherwise the committed `(d, b)`
    // line — generated at the cert's scale — will not bound σ at the
    // rescaled endpoints used in-gadget. Raising `s_x` is the lever
    // that tightens the sigmoid/tanh output-bound drift. ReLU-only
    // networks keep the user-driven working scale.
    let has_sshape = network.layers().iter().any(|l| {
        matches!(
            l,
            crate::crown::network::Layer::Activation {
                kind: crate::crown::network::ActivationKind::Sigmoid
                    | crate::crown::network::ActivationKind::Tanh,
            }
        )
    });
    if has_sshape && working.e != sigma_x_scale_log2 {
        working = crate::quantization::scale::Scale::from_pow2(sigma_x_scale_log2);
    }
    let layer_scales = super::scales::pick_layer_scales_with_max_e(
        network,
        &plain.relaxations,
        precision_bits,
        Some(working.e),
    );
    let scales = QuantScales {
        working,
        input: input_scale,
        spec: working,
        layers: layer_scales,
    };
    // Relaxations are NOT built upfront. They are built per layer in
    // the chain loop below from QUANTIZED preact codes, which removes
    // the float-vs-quantized interval mismatch that previously made
    // the SNARK upper-line endpoint check fail on deeper benches.
    let (weights, biases) = quantize_layer_weights_and_biases(network, &scales);
    let x_lower_q = quantize_vector_floor(x_lower, scales.input);
    let x_upper_q = quantize_vector_ceil(x_upper, scales.input);
    let spec_c = quantize_matrix(&property.c_matrix, scales.spec);
    let spec_d = quantize_vector(&property.d_vector, scales.spec);

    let mut witnesses: Vec<RescaleEntry> = Vec::new();

    let mut hidden_passes: Vec<HiddenLayerPass> = Vec::new();
    let (relaxations, preact_lower, preact_upper) = build_chain_layer_by_layer(
        network,
        &weights,
        &biases,
        &plain.relaxations,
        &x_lower_q,
        &x_upper_q,
        &scales,
        &mut witnesses,
        Some(&mut hidden_passes),
        sigma_x_scale_log2,
        sigma_v_scale_log2,
    )?;

    let mut lower_trace = property.side.needs_lower().then(|| BackwardTrace {
        linear_steps: Vec::new(),
        activation_steps: Vec::new(),
        final_target: QArray1::new(
            Array1::<crate::quantization::quantized_scalar::Code>::zeros(0),
            scales.working,
        ),
        concretize: None,
    });
    let mut upper_trace = property.side.needs_upper().then(|| BackwardTrace {
        linear_steps: Vec::new(),
        activation_steps: Vec::new(),
        final_target: QArray1::new(
            Array1::<crate::quantization::quantized_scalar::Code>::zeros(0),
            scales.working,
        ),
        concretize: None,
    });
    let target_lower = if property.side.needs_lower() {
        Some(run_quant_pass(
            network,
            &weights,
            &biases,
            &relaxations,
            &spec_c,
            &spec_d,
            &x_lower_q,
            &x_upper_q,
            &scales,
            BoundDir::Lower,
            &mut witnesses,
            lower_trace.as_mut(),
        )?)
    } else {
        None
    };
    let target_upper = if property.side.needs_upper() {
        Some(run_quant_pass(
            network,
            &weights,
            &biases,
            &relaxations,
            &spec_c,
            &spec_d,
            &x_lower_q,
            &x_upper_q,
            &scales,
            BoundDir::Upper,
            &mut witnesses,
            upper_trace.as_mut(),
        )?)
    } else {
        None
    };

    let cert = QuantCert {
        scales,
        weights,
        biases,
        relaxations,
        x_lower: x_lower_q,
        x_upper: x_upper_q,
        spec_c,
        spec_d,
        target_lower,
        target_upper,
        preact_lower,
        preact_upper,
        witnesses,
    };
    Ok((cert, lower_trace, upper_trace, hidden_passes))
}

/// Build quantized relaxations layer-by-layer in forward order using
/// quantized preact bounds produced by the chain itself.
///
/// For each hidden Linear layer `idx`, run the chain backward to compute
/// `preact_lower[idx]` and `preact_upper[idx]` as integer codes at the
/// working scale. For the activation at `idx + 1`, build the relaxation
/// from those exact quantized endpoints. The SNARK then checks the line
/// at the same endpoints the cert generator built it for.
///
/// Returns `(relaxations, preact_lower, preact_upper)`, and (optionally)
/// per-pass traces for the SNARK driver.
#[allow(clippy::too_many_arguments)]
fn build_chain_layer_by_layer(
    network: &Network,
    weights: &[Option<QArray2>],
    biases: &[Option<QArray1>],
    plain_relaxations: &[Option<crate::crown::float_crown::ActivationRelaxation>],
    x_lower_q: &QArray1,
    x_upper_q: &QArray1,
    scales: &QuantScales,
    witnesses: &mut Vec<RescaleEntry>,
    mut hidden_passes_out: Option<&mut Vec<HiddenLayerPass>>,
    sigma_x_scale_log2: i32,
    sigma_v_scale_log2: i32,
) -> Result<
    (
        Vec<Option<QuantRelaxation>>,
        Vec<Option<QArray1>>,
        Vec<Option<QArray1>>,
    ),
    QCrownError,
> {
    let layers = network.layers();
    let n_layers = layers.len();
    let mut relaxations: Vec<Option<QuantRelaxation>> = vec![None; n_layers];
    let mut preact_lower_q: Vec<Option<QArray1>> = vec![None; n_layers];
    let mut preact_upper_q: Vec<Option<QArray1>> = vec![None; n_layers];

    for idx in 0..n_layers {
        match &layers[idx] {
            Layer::Linear { .. } => {
                let is_hidden =
                    idx + 1 < n_layers && matches!(layers[idx + 1], Layer::Activation { .. });
                if !is_hidden {
                    continue;
                }
                let n_spec = match &layers[idx] {
                    Layer::Linear { weight, .. } => weight.nrows(),
                    _ => unreachable!(),
                };
                let identity = identity_matrix_q(n_spec, scales.spec);
                let zero = QArray1::new(
                    Array1::<crate::quantization::quantized_scalar::Code>::zeros(n_spec),
                    scales.spec,
                );
                let mut lower_trace = BackwardTrace {
                    linear_steps: Vec::new(),
                    activation_steps: Vec::new(),
                    final_target: QArray1::new(
                        Array1::<crate::quantization::quantized_scalar::Code>::zeros(0),
                        scales.working,
                    ),
                    concretize: None,
                };
                let lower = run_quant_pass_at(
                    network,
                    weights,
                    biases,
                    &relaxations,
                    &identity,
                    &zero,
                    x_lower_q,
                    x_upper_q,
                    scales,
                    BoundDir::Lower,
                    witnesses,
                    Some(&mut lower_trace),
                    idx,
                )?;
                let mut upper_trace = BackwardTrace {
                    linear_steps: Vec::new(),
                    activation_steps: Vec::new(),
                    final_target: QArray1::new(
                        Array1::<crate::quantization::quantized_scalar::Code>::zeros(0),
                        scales.working,
                    ),
                    concretize: None,
                };
                let upper = run_quant_pass_at(
                    network,
                    weights,
                    biases,
                    &relaxations,
                    &identity,
                    &zero,
                    x_lower_q,
                    x_upper_q,
                    scales,
                    BoundDir::Upper,
                    witnesses,
                    Some(&mut upper_trace),
                    idx,
                )?;
                preact_lower_q[idx] = Some(lower.clone());
                preact_upper_q[idx] = Some(upper.clone());
                if let Some(out) = hidden_passes_out.as_deref_mut() {
                    out.push(HiddenLayerPass {
                        target_layer_idx: idx,
                        n_spec,
                        lower_trace,
                        upper_trace,
                        preact_lower: lower,
                        preact_upper: upper,
                    });
                }
            }
            Layer::Activation { .. } => {
                let prev = idx
                    .checked_sub(1)
                    .expect("activation layer must be preceded by a hidden Linear layer");
                let l_q = preact_lower_q[prev]
                    .as_ref()
                    .expect("preact_lower_q at preceding hidden Linear must be set");
                let u_q = preact_upper_q[prev]
                    .as_ref()
                    .expect("preact_upper_q at preceding hidden Linear must be set");
                let rel_float = plain_relaxations[idx]
                    .as_ref()
                    .expect("activation has float relaxation populated");
                let s_d = scales.layers[idx]
                    .relax_d
                    .expect("activation has relax_d scale");
                let s_b = scales.layers[idx]
                    .relax_b
                    .expect("activation has relax_b scale");
                relaxations[idx] =
                    Some(super::relaxation::quantize_relaxation_at_quantized_preacts(
                        rel_float,
                        s_d,
                        s_b,
                        scales.working,
                        l_q,
                        u_q,
                        idx,
                        sigma_x_scale_log2,
                        sigma_v_scale_log2,
                    )?);
            }
        }
    }

    Ok((relaxations, preact_lower_q, preact_upper_q))
}

fn identity_matrix_q(n: usize, scale: crate::quantization::scale::Scale) -> QArray2 {
    use ndarray::Array2;
    let one_code = crate::quantization::quantized_scalar::Qf::from_real(1.0, scale).code;
    let mut codes = Array2::<crate::quantization::quantized_scalar::Code>::zeros((n, n));
    for i in 0..n {
        codes[[i, i]] = one_code;
    }
    QArray2::new(codes, scale)
}

/// Quantize per-layer weights and biases at their pinned scales.
///
/// Relaxation tensors are deliberately not built here: they are
/// constructed per-layer in [`build_chain_layer_by_layer`] from the
/// quantized preact endpoints.
#[allow(clippy::type_complexity)]
fn quantize_layer_weights_and_biases(
    network: &Network,
    scales: &QuantScales,
) -> (Vec<Option<QArray2>>, Vec<Option<QArray1>>) {
    let weights: Vec<Option<QArray2>> = network
        .layers()
        .iter()
        .zip(scales.layers.iter())
        .map(|(l, s)| match l {
            Layer::Linear { weight, .. } => {
                Some(quantize_matrix(weight, s.weight.expect("linear scale")))
            }
            Layer::Activation { .. } => None,
        })
        .collect();
    let biases: Vec<Option<QArray1>> = network
        .layers()
        .iter()
        .zip(scales.layers.iter())
        .map(|(l, s)| match l {
            Layer::Linear { bias, .. } => {
                Some(quantize_vector(bias, s.bias.expect("linear scale")))
            }
            Layer::Activation { .. } => None,
        })
        .collect();
    (weights, biases)
}

#[allow(clippy::too_many_arguments)]
fn run_quant_pass(
    network: &Network,
    weights: &[Option<QArray2>],
    biases: &[Option<QArray1>],
    relaxations: &[Option<QuantRelaxation>],
    spec_c: &QArray2,
    spec_d: &QArray1,
    x_lower: &QArray1,
    x_upper: &QArray1,
    scales: &QuantScales,
    dir: BoundDir,
    witnesses: &mut Vec<RescaleEntry>,
    trace: Option<&mut BackwardTrace>,
) -> Result<QArray1, QCrownError> {
    let final_idx = network.layers().len() - 1;
    run_quant_pass_at(
        network,
        weights,
        biases,
        relaxations,
        spec_c,
        spec_d,
        x_lower,
        x_upper,
        scales,
        dir,
        witnesses,
        trace,
        final_idx,
    )
}

/// Generic backward CROWN driver.
///
/// `top_layer_idx` is the index of the last layer whose backward step is
/// applied: `final_idx` for the final pass, or `target_layer_idx` for a
/// hidden-layer preactivation pass. Initial `A = spec_c`, `b_acc = spec_d`
/// are interpreted at the `top_layer_idx + 1` boundary; the loop walks
/// `(0..=top_layer_idx).rev()` and then concretizes on the input box.
/// Matches the float `backward_pass` in `float_crown`.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_quant_pass_at(
    network: &Network,
    weights: &[Option<QArray2>],
    biases: &[Option<QArray1>],
    relaxations: &[Option<QuantRelaxation>],
    spec_c: &QArray2,
    spec_d: &QArray1,
    x_lower: &QArray1,
    x_upper: &QArray1,
    scales: &QuantScales,
    dir: BoundDir,
    witnesses: &mut Vec<RescaleEntry>,
    mut trace: Option<&mut BackwardTrace>,
    top_layer_idx: usize,
) -> Result<QArray1, QCrownError> {
    let layers = network.layers();

    let mut a = spec_c.clone();
    let mut b_acc = spec_d.clone();

    for i in (0..=top_layer_idx).rev() {
        match &layers[i] {
            Layer::Linear { .. } => {
                let w = weights[i].as_ref().expect("linear layer has weight");
                let b = biases[i].as_ref().expect("linear layer has bias");
                let a_before = a.clone();
                let b_acc_before = b_acc.clone();
                let a_b = matvec(&a, b).map_err(QCrownError::QArray)?;
                // Bound-direction-aware rounding keeps b_acc a sound
                // under-/over-approximation of the float value.
                let bias_dir = match dir {
                    BoundDir::Lower => crate::quantization::quantized_scalar::RoundDir::Floor,
                    BoundDir::Upper => crate::quantization::quantized_scalar::RoundDir::Ceil,
                };
                let (prod_w, mut ws) = crate::quantization::quantized_array::rescale_vector_dir(
                    &a_b,
                    scales.working,
                    bias_dir,
                )
                .map_err(QCrownError::QArray)?;
                witnesses.append(&mut ws);
                b_acc = b_acc.add(&prod_w).map_err(QCrownError::QArray)?;
                let a_w = matmul(&a, w).map_err(QCrownError::QArray)?;
                let (new_a, mut ws) =
                    crate::quantization::quantized_array::rescale_matrix(&a_w, scales.working)
                        .map_err(QCrownError::QArray)?;
                witnesses.append(&mut ws);
                a = new_a;
                if let Some(t) = trace.as_mut() {
                    t.linear_steps.push(LinearStepTrace {
                        layer_idx: i,
                        a_old: a_before,
                        b_acc_old: b_acc_before,
                        a_w,
                        a_b,
                        a_new: a.clone(),
                        b_acc_new: b_acc.clone(),
                    });
                }
            }
            Layer::Activation { .. } => {
                let relax = relaxations[i].as_ref().expect("activation has relaxation");
                let a_before = a.clone();
                let b_acc_before = b_acc.clone();
                let (selectors, a_pos, a_d_doubled, bias_delta_doubled) =
                    apply_activation_quant(&mut a, &mut b_acc, relax, scales, dir, witnesses)?;
                if let Some(t) = trace.as_mut() {
                    t.activation_steps.push(ActivationStepTrace {
                        layer_idx: i,
                        a_old: a_before,
                        b_acc_old: b_acc_before,
                        selectors,
                        a_pos,
                        a_d_doubled,
                        bias_delta_doubled,
                        a_new: a.clone(),
                        b_acc_new: b_acc.clone(),
                    });
                }
            }
        }
    }

    let x = match dir {
        BoundDir::Lower => x_lower,
        BoundDir::Upper => x_upper,
    };
    let x_other = match dir {
        BoundDir::Lower => x_upper,
        BoundDir::Upper => x_lower,
    };
    let target = concretize_quant(
        &a,
        &b_acc,
        x,
        x_other,
        scales,
        dir,
        witnesses,
        trace.as_deref_mut(),
    )?;
    if let Some(t) = trace.as_mut() {
        t.final_target = target.clone();
    }
    Ok(target)
}

fn apply_activation_quant(
    a: &mut QArray2,
    b_acc: &mut QArray1,
    relax: &QuantRelaxation,
    scales: &QuantScales,
    dir: BoundDir,
    witnesses: &mut Vec<RescaleEntry>,
) -> Result<(QArray2, QArray2, QArray2, QArray1), QCrownError> {
    let n_rows = a.nrows();
    let n_cols = a.ncols();
    let mut new_a_codes_doubled = Array2::<Code>::zeros((n_rows, n_cols));
    let mut bias_delta_doubled = Array1::<Code>::zeros(n_rows);
    // Selectors are 1 when the picked line is the "positive coeff" one for
    // this direction, 0 otherwise. Stored as `Code` for QArray2 uniformity.
    let mut sel_codes = Array2::<Code>::zeros((n_rows, n_cols));
    let mut a_pos_codes = Array2::<Code>::zeros((n_rows, n_cols));
    for i in 0..n_rows {
        for j in 0..n_cols {
            let coeff = a.codes[[i, j]];
            let pick_lower = match dir {
                BoundDir::Lower => coeff >= 0,
                BoundDir::Upper => coeff < 0,
            };
            sel_codes[[i, j]] = if coeff >= 0 { 1 } else { 0 };
            a_pos_codes[[i, j]] = coeff.max(0);
            let d_pick = if pick_lower {
                relax.d_lower.codes[j]
            } else {
                relax.d_upper.codes[j]
            };
            let b_pick = if pick_lower {
                relax.b_lower.codes[j]
            } else {
                relax.b_upper.codes[j]
            };
            new_a_codes_doubled[[i, j]] = coeff
                .checked_mul(d_pick)
                .ok_or(QCrownError::QArray(QArrayError::OverflowOnMul))?;
            let prod = coeff
                .checked_mul(b_pick)
                .ok_or(QCrownError::QArray(QArrayError::OverflowOnMul))?;
            bias_delta_doubled[i] = bias_delta_doubled[i]
                .checked_add(prod)
                .ok_or(QCrownError::QArray(QArrayError::OverflowOnAdd))?;
        }
    }
    let a_d_scale = a
        .scale
        .compose(relax.d_lower.scale)
        .map_err(|e| QCrownError::QArray(QArrayError::ScaleCompose(e)))?;
    let a_b_scale = a
        .scale
        .compose(relax.b_lower.scale)
        .map_err(|e| QCrownError::QArray(QArrayError::ScaleCompose(e)))?;
    let new_a_doubled = QArray2::new(new_a_codes_doubled, a_d_scale);
    let bias_doubled = QArray1::new(bias_delta_doubled, a_b_scale);
    // Per-direction rounding on the bias-delta accumulator: Floor for
    // Lower, Ceil for Upper. The chain matrix `A` keeps HalfAway
    // (banker's) — adding sound away-from-zero rounding to `A` would
    // require per-element sign tracking in the SNARK gadget, and the
    // residual drift is already bounded by 0.5 LSB per rescale.
    let (new_a, mut ws) =
        crate::quantization::quantized_array::rescale_matrix(&new_a_doubled, scales.working)
            .map_err(QCrownError::QArray)?;
    witnesses.append(&mut ws);
    let bias_dir = match dir {
        BoundDir::Lower => crate::quantization::quantized_scalar::RoundDir::Floor,
        BoundDir::Upper => crate::quantization::quantized_scalar::RoundDir::Ceil,
    };
    let (bias_delta, mut ws) = crate::quantization::quantized_array::rescale_vector_dir(
        &bias_doubled,
        scales.working,
        bias_dir,
    )
    .map_err(QCrownError::QArray)?;
    witnesses.append(&mut ws);
    let a_old_scale = scales.working;
    *a = new_a;
    *b_acc = b_acc.add(&bias_delta).map_err(QCrownError::QArray)?;
    Ok((
        QArray2::new(sel_codes, crate::quantization::scale::Scale::from_pow2(0)),
        QArray2::new(a_pos_codes, a_old_scale),
        new_a_doubled,
        bias_doubled,
    ))
}

#[allow(clippy::too_many_arguments)]
fn concretize_quant(
    a: &QArray2,
    b_acc: &QArray1,
    x_same: &QArray1,
    x_other: &QArray1,
    scales: &QuantScales,
    dir: BoundDir,
    witnesses: &mut Vec<RescaleEntry>,
    mut trace: Option<&mut BackwardTrace>,
) -> Result<QArray1, QCrownError> {
    let n_rows = a.nrows();
    let n_cols = a.ncols();
    debug_assert_eq!(x_same.len(), n_cols);
    debug_assert_eq!(x_other.len(), n_cols);
    let mut acc_codes = Array1::<Code>::zeros(n_rows);
    let mut sel_codes = Array2::<Code>::zeros((n_rows, n_cols));
    let mut a_pos_codes = Array2::<Code>::zeros((n_rows, n_cols));
    for i in 0..n_rows {
        let mut row_acc: Code = 0;
        for j in 0..n_cols {
            let coeff = a.codes[[i, j]];
            let x_pos = x_same.codes[j];
            let x_neg = x_other.codes[j];
            sel_codes[[i, j]] = if coeff >= 0 { 1 } else { 0 };
            a_pos_codes[[i, j]] = coeff.max(0);
            let chosen = if coeff >= 0 { x_pos } else { x_neg };
            let prod = coeff
                .checked_mul(chosen)
                .ok_or(QCrownError::QArray(QArrayError::OverflowOnMul))?;
            row_acc = row_acc
                .checked_add(prod)
                .ok_or(QCrownError::QArray(QArrayError::OverflowOnAdd))?;
        }
        acc_codes[i] = row_acc;
    }
    let acc_scale = a
        .scale
        .compose(x_same.scale)
        .map_err(|e| QCrownError::QArray(QArrayError::ScaleCompose(e)))?;
    let acc = QArray1::new(acc_codes, acc_scale);
    let target_doubled = acc.clone();
    let acc_dir = match dir {
        BoundDir::Lower => crate::quantization::quantized_scalar::RoundDir::Floor,
        BoundDir::Upper => crate::quantization::quantized_scalar::RoundDir::Ceil,
    };
    let (acc_w, mut ws) =
        crate::quantization::quantized_array::rescale_vector_dir(&acc, scales.working, acc_dir)
            .map_err(QCrownError::QArray)?;
    witnesses.append(&mut ws);
    let final_target = acc_w.add(b_acc).map_err(QCrownError::QArray)?;
    if let Some(t) = trace.as_mut() {
        t.concretize = Some(ConcretizeTrace {
            a_final: a.clone(),
            b_acc_final: b_acc.clone(),
            selectors: QArray2::new(sel_codes, crate::quantization::scale::Scale::from_pow2(0)),
            a_pos: QArray2::new(a_pos_codes, a.scale),
            target_doubled,
            final_target: final_target.clone(),
        });
    }
    Ok(final_target)
}

fn concat_for_scale(slices: &[&[f64]]) -> Vec<f64> {
    let mut v = Vec::with_capacity(slices.iter().map(|s| s.len()).sum());
    for s in slices {
        v.extend_from_slice(s);
    }
    v
}
