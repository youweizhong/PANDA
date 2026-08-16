//! Per-backward-pass driver for the rescale gadget.
//!
//! Defines the canonical event ordering (network-backward, layer by
//! layer: matrix then vector per step, plus concretize) and the
//! prove/verify walkers that apply
//! [`crate::snark::rescaling::prove_rescale_event`] /
//! [`crate::snark::rescaling::verify_rescale_event`] at each event.
//! The verifier rebuilds the expected event layout from the public
//! architecture + property so no trace is required.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_std::rand::RngCore;

use crate::crown::network::{LayerShape, NetworkArchitecture};
use crate::crown::output_property::Property;
use crate::quantized_crown::BackwardTrace;

use crate::snark::commitment::commit::{
    pad_matrix_evals_2d, pad_vector_evals_1d, PassCommitments, PassProverStates,
};
use crate::snark::commitment::multilinear_extensions::next_pow2_log;
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;

/// Minimum n_vars for the rescale gadget. LogUp needs ≥ 2 vars for a
/// non-trivial bottom-bind, and Hyrax requires even num_vars.
pub(crate) const RESCALE_MIN_N_VARS: usize = 2;

fn target_n_vars(n_vars: usize) -> usize {
    let mut t = n_vars.max(RESCALE_MIN_N_VARS);
    if t % 2 == 1 {
        t += 1;
    }
    t
}

/// Bump `slack_lo` and `n_vars` up to `target_n_vars` by appending
/// the direction-specific zero-cell slack value (so the identity at
/// `qx = qz = 0` holds for the padding cells).
fn bump_slack_to_min(
    mut slack: Vec<i128>,
    n_vars: usize,
    c2: i128,
    dir: crate::quantization::quantized_scalar::RoundDir,
) -> (Vec<i128>, usize) {
    let target = target_n_vars(n_vars);
    if target == n_vars {
        return (slack, n_vars);
    }
    let new_len = 1usize << target;
    let zero_slack = match dir {
        crate::quantization::quantized_scalar::RoundDir::HalfAway => c2,
        crate::quantization::quantized_scalar::RoundDir::Floor => 0,
        crate::quantization::quantized_scalar::RoundDir::Ceil => 2 * c2 - 2,
    };
    slack.resize(new_len, zero_slack);
    (slack, target)
}

/// Zero-pad a qx/qz MLE eval table up to `target_n_vars` slots.
fn bump_evals_to_min(mut evals: Vec<Fr>, n_vars: usize) -> (Vec<Fr>, usize) {
    let target = target_n_vars(n_vars);
    if target == n_vars {
        return (evals, n_vars);
    }
    let new_len = 1usize << target;
    evals.resize(new_len, Fr::from(0u64));
    (evals, target)
}

fn rescale_vector_slack_padded_dir(
    src: &crate::quantization::quantized_array::QArray1,
    target: crate::quantization::scale::Scale,
    dir: crate::quantization::quantized_scalar::RoundDir,
) -> Result<(Vec<i128>, i128, i128), SnarkError> {
    let (_out, ws) = crate::quantization::quantized_array::rescale_vector_dir(src, target, dir)
        .map_err(|e| SnarkError::QCrown(crate::quantized_crown::QCrownError::QArray(e)))?;
    let c1 = ws.first().map(|w| w.c1).unwrap_or(1);
    let c2 = ws.first().map(|w| w.c2).unwrap_or(1);
    let log_n = next_pow2_log(src.codes.len());
    let pow_n = 1usize << log_n;
    // Direction-dependent zero-cell slack value.
    let zero_slack = match dir {
        crate::quantization::quantized_scalar::RoundDir::HalfAway => c2,
        crate::quantization::quantized_scalar::RoundDir::Floor => 0,
        crate::quantization::quantized_scalar::RoundDir::Ceil => 2 * c2 - 2,
    };
    let mut padded = vec![zero_slack; pow_n];
    for (slot, w) in padded.iter_mut().zip(ws.iter()) {
        *slot = w.slack_lo;
    }
    Ok((padded, c1, c2))
}

/// Same as the vector form for a matrix; returns the `slack_lo` MLE
/// table at the 2D-padded layout (size `2^{log_rows + log_cols}`).
fn rescale_matrix_slack_padded(
    src: &crate::quantization::quantized_array::QArray2,
    target: crate::quantization::scale::Scale,
) -> Result<(Vec<i128>, i128, i128), SnarkError> {
    let (_out, ws) = crate::quantization::quantized_array::rescale_matrix(src, target)
        .map_err(|e| SnarkError::QCrown(crate::quantized_crown::QCrownError::QArray(e)))?;
    let c1 = ws.first().map(|w| w.c1).unwrap_or(1);
    let c2 = ws.first().map(|w| w.c2).unwrap_or(1);
    let rows = src.nrows();
    let cols = src.ncols();
    let log_rows = next_pow2_log(rows);
    let log_cols = next_pow2_log(cols);
    let pow_rows = 1usize << log_rows;
    let pow_cols = 1usize << log_cols;
    let mut padded = vec![c2; pow_rows * pow_cols];
    for ((i, j), w) in (0..rows)
        .flat_map(|i| (0..cols).map(move |j| (i, j)))
        .zip(ws.iter())
    {
        padded[i * pow_cols + j] = w.slack_lo;
    }
    Ok((padded, c1, c2))
}

/// One slot in the canonical event order of a backward pass.
#[derive(Clone, Copy, Debug)]
enum RescaleEventKind {
    LinearMatrix(usize),
    LinearVector(usize),
    ActMatrix(usize),
    ActVector(usize),
    ConcretizeVector,
}

#[derive(Clone, Copy, Debug)]
struct RescaleEventLayout {
    kind: RescaleEventKind,
    /// Layer index (`usize::MAX` for concretize).
    layer_idx: usize,
    n_vars: usize,
}

/// Sequence of expected rescale events derived from the public
/// architecture and property; verifier-friendly (no trace required).
fn expected_rescale_layout(
    arch: &NetworkArchitecture,
    property: &Property,
    has_concretize: bool,
) -> Vec<RescaleEventLayout> {
    let layers = arch.layers();
    let n_spec = property.c_matrix.nrows();
    let log_spec = next_pow2_log(n_spec);
    let mut a_cols = arch.output_dim();
    let mut linear_step_idx = 0usize;
    let mut act_step_idx = 0usize;
    let mut out: Vec<RescaleEventLayout> = Vec::new();
    let bump = target_n_vars;
    for i in (0..layers.len()).rev() {
        match &layers[i] {
            LayerShape::Linear { in_dim, .. } => {
                let n_vars_m = log_spec + next_pow2_log(*in_dim);
                out.push(RescaleEventLayout {
                    kind: RescaleEventKind::LinearMatrix(linear_step_idx),
                    layer_idx: i,
                    n_vars: bump(n_vars_m),
                });
                out.push(RescaleEventLayout {
                    kind: RescaleEventKind::LinearVector(linear_step_idx),
                    layer_idx: i,
                    n_vars: bump(log_spec),
                });
                linear_step_idx += 1;
                a_cols = *in_dim;
            }
            LayerShape::Activation { .. } => {
                let n_vars_m = log_spec + next_pow2_log(a_cols);
                out.push(RescaleEventLayout {
                    kind: RescaleEventKind::ActMatrix(act_step_idx),
                    layer_idx: i,
                    n_vars: bump(n_vars_m),
                });
                out.push(RescaleEventLayout {
                    kind: RescaleEventKind::ActVector(act_step_idx),
                    layer_idx: i,
                    n_vars: bump(log_spec),
                });
                act_step_idx += 1;
            }
        }
    }
    if has_concretize {
        out.push(RescaleEventLayout {
            kind: RescaleEventKind::ConcretizeVector,
            layer_idx: usize::MAX,
            n_vars: bump(log_spec),
        });
    }
    out
}

/// Walk the trace and emit per-event rescale proofs in canonical
/// order. `bound_dir` picks Floor / Ceil for vector events; chain-
/// matrix events keep `HalfAway`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_rescale_proofs(
    trace: &BackwardTrace,
    pass_com: &PassCommitments,
    pass_st: &PassProverStates,
    bound_dir: crate::quantized_crown::BoundDir,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<Vec<crate::snark::rescaling::RescaleEventProof>, SnarkError> {
    let _timing = crate::timing::scope("rescale");
    let bias_dir = match bound_dir {
        crate::quantized_crown::BoundDir::Lower => {
            crate::quantization::quantized_scalar::RoundDir::Floor
        }
        crate::quantized_crown::BoundDir::Upper => {
            crate::quantization::quantized_scalar::RoundDir::Ceil
        }
    };
    let target_scale = trace
        .linear_steps
        .first()
        .map(|s| s.a_new.scale)
        .or_else(|| trace.activation_steps.first().map(|s| s.a_new.scale))
        .ok_or(SnarkError::ShapeMismatch {
            what: "rescale: empty trace",
        })?;
    let mut events: Vec<crate::snark::rescaling::RescaleEventProof> = Vec::new();

    // Network-backward order matches `expected_rescale_layout`.
    enum StepRef<'a> {
        Linear(usize, &'a crate::quantized_crown::LinearStepTrace),
        Activation(usize, &'a crate::quantized_crown::ActivationStepTrace),
    }
    let mut steps: Vec<StepRef<'_>> =
        Vec::with_capacity(trace.linear_steps.len() + trace.activation_steps.len());
    for (i, s) in trace.linear_steps.iter().enumerate() {
        steps.push(StepRef::Linear(i, s));
    }
    for (i, s) in trace.activation_steps.iter().enumerate() {
        steps.push(StepRef::Activation(i, s));
    }
    steps.sort_by_key(|s| {
        std::cmp::Reverse(match s {
            StepRef::Linear(_, st) => st.layer_idx,
            StepRef::Activation(_, st) => st.layer_idx,
        })
    });

    for step_ref in steps.into_iter() {
        match step_ref {
            StepRef::Linear(step_idx, step) => {
                let (qx_evals, n_vars_m) = pad_matrix_evals_2d(&step.a_w);
                let (qz_evals, n_vars_post) = pad_matrix_evals_2d(&step.a_new);
                if n_vars_m != n_vars_post {
                    return Err(SnarkError::ShapeMismatch {
                        what: "rescale: linear matrix pre/post shape mismatch",
                    });
                }
                let (slack_lo, c1, c2) = rescale_matrix_slack_padded(&step.a_w, target_scale)?;
                let (slack_lo, n_vars) = bump_slack_to_min(
                    slack_lo,
                    n_vars_m,
                    c2,
                    crate::quantization::quantized_scalar::RoundDir::HalfAway,
                );
                let (qx_evals, _) = bump_evals_to_min(qx_evals, n_vars_m);
                let (qz_evals, _) = bump_evals_to_min(qz_evals, n_vars_m);
                let desc = crate::snark::rescaling::RescaleEventDesc {
                    c1,
                    c2,
                    n_vars,
                    dir: crate::quantization::quantized_scalar::RoundDir::HalfAway,
                };
                let proof = crate::snark::rescaling::prove_rescale_event(
                    &desc,
                    &slack_lo,
                    &qx_evals,
                    &qz_evals,
                    &pass_st.linear_a_w[step_idx],
                    &pass_com.linear_a_w[step_idx],
                    &pass_st.chain_a[step.layer_idx],
                    &pass_com.chain_a[step.layer_idx],
                    params,
                    sponge,
                    rng,
                )?;
                events.push(proof);

                // Vector rescale a_b → prod_w (direction-aware rounding).
                let (qx_evals, n_vars_v) = pad_vector_evals_1d(&step.a_b);
                let prod_w = crate::quantization::quantized_array::QArray1::new(
                    &step.b_acc_new.codes - &step.b_acc_old.codes,
                    step.b_acc_new.scale,
                );
                let (qz_evals, n_vars_v_post) = pad_vector_evals_1d(&prod_w);
                if n_vars_v != n_vars_v_post {
                    return Err(SnarkError::ShapeMismatch {
                        what: "rescale: linear vector pre/post shape mismatch",
                    });
                }
                let (slack_lo, c1, c2) =
                    rescale_vector_slack_padded_dir(&step.a_b, target_scale, bias_dir)?;
                let (slack_lo, n_vars) = bump_slack_to_min(slack_lo, n_vars_v, c2, bias_dir);
                let (qx_evals, _) = bump_evals_to_min(qx_evals, n_vars_v);
                let (qz_evals, _) = bump_evals_to_min(qz_evals, n_vars_v);
                let desc = crate::snark::rescaling::RescaleEventDesc {
                    c1,
                    c2,
                    n_vars,
                    dir: bias_dir,
                };
                let proof = crate::snark::rescaling::prove_rescale_event(
                    &desc,
                    &slack_lo,
                    &qx_evals,
                    &qz_evals,
                    &pass_st.linear_a_b[step_idx],
                    &pass_com.linear_a_b[step_idx],
                    &pass_st.linear_prod_w[step_idx],
                    &pass_com.linear_prod_w[step_idx],
                    params,
                    sponge,
                    rng,
                )?;
                events.push(proof);
            }
            StepRef::Activation(step_idx, step) => {
                let (qx_evals, n_vars_m) = pad_matrix_evals_2d(&step.a_d_doubled);
                let (qz_evals, n_vars_post) = pad_matrix_evals_2d(&step.a_new);
                if n_vars_m != n_vars_post {
                    return Err(SnarkError::ShapeMismatch {
                        what: "rescale: activation matrix pre/post shape mismatch",
                    });
                }
                let (slack_lo, c1, c2) =
                    rescale_matrix_slack_padded(&step.a_d_doubled, target_scale)?;
                let (slack_lo, n_vars) = bump_slack_to_min(
                    slack_lo,
                    n_vars_m,
                    c2,
                    crate::quantization::quantized_scalar::RoundDir::HalfAway,
                );
                let (qx_evals, _) = bump_evals_to_min(qx_evals, n_vars_m);
                let (qz_evals, _) = bump_evals_to_min(qz_evals, n_vars_m);
                let desc = crate::snark::rescaling::RescaleEventDesc {
                    c1,
                    c2,
                    n_vars,
                    dir: crate::quantization::quantized_scalar::RoundDir::HalfAway,
                };
                let proof = crate::snark::rescaling::prove_rescale_event(
                    &desc,
                    &slack_lo,
                    &qx_evals,
                    &qz_evals,
                    &pass_st.activation_a_d_doubled[step_idx],
                    &pass_com.activation_a_d_doubled[step_idx],
                    &pass_st.chain_a[step.layer_idx],
                    &pass_com.chain_a[step.layer_idx],
                    params,
                    sponge,
                    rng,
                )?;
                events.push(proof);

                // Vector rescale bias_delta_doubled → bias_delta.
                let (qx_evals, n_vars_v) = pad_vector_evals_1d(&step.bias_delta_doubled);
                let bias_delta = crate::quantization::quantized_array::QArray1::new(
                    &step.b_acc_new.codes - &step.b_acc_old.codes,
                    step.b_acc_new.scale,
                );
                let (qz_evals, n_vars_v_post) = pad_vector_evals_1d(&bias_delta);
                if n_vars_v != n_vars_v_post {
                    return Err(SnarkError::ShapeMismatch {
                        what: "rescale: activation vector pre/post shape mismatch",
                    });
                }
                let (slack_lo, c1, c2) = rescale_vector_slack_padded_dir(
                    &step.bias_delta_doubled,
                    target_scale,
                    bias_dir,
                )?;
                let (slack_lo, n_vars) = bump_slack_to_min(slack_lo, n_vars_v, c2, bias_dir);
                let (qx_evals, _) = bump_evals_to_min(qx_evals, n_vars_v);
                let (qz_evals, _) = bump_evals_to_min(qz_evals, n_vars_v);
                let desc = crate::snark::rescaling::RescaleEventDesc {
                    c1,
                    c2,
                    n_vars,
                    dir: bias_dir,
                };
                let proof = crate::snark::rescaling::prove_rescale_event(
                    &desc,
                    &slack_lo,
                    &qx_evals,
                    &qz_evals,
                    &pass_st.activation_bias_doubled[step_idx],
                    &pass_com.activation_bias_doubled[step_idx],
                    &pass_st.activation_bias_delta[step_idx],
                    &pass_com.activation_bias_delta[step_idx],
                    params,
                    sponge,
                    rng,
                )?;
                events.push(proof);
            }
        }
    }

    if let Some(c) = trace.concretize.as_ref() {
        let (qx_evals, n_vars_v) = pad_vector_evals_1d(&c.target_doubled);
        let acc_w = crate::quantization::quantized_array::QArray1::new(
            &c.final_target.codes - &c.b_acc_final.codes,
            c.final_target.scale,
        );
        let (qz_evals, n_vars_v_post) = pad_vector_evals_1d(&acc_w);
        if n_vars_v != n_vars_v_post {
            return Err(SnarkError::ShapeMismatch {
                what: "rescale: concretize vector pre/post shape mismatch",
            });
        }
        let (slack_lo, c1, c2) =
            rescale_vector_slack_padded_dir(&c.target_doubled, target_scale, bias_dir)?;
        let (slack_lo, n_vars) = bump_slack_to_min(slack_lo, n_vars_v, c2, bias_dir);
        let (qx_evals, _) = bump_evals_to_min(qx_evals, n_vars_v);
        let (qz_evals, _) = bump_evals_to_min(qz_evals, n_vars_v);
        let desc = crate::snark::rescaling::RescaleEventDesc {
            c1,
            c2,
            n_vars,
            dir: bias_dir,
        };
        let proof = crate::snark::rescaling::prove_rescale_event(
            &desc,
            &slack_lo,
            &qx_evals,
            &qz_evals,
            pass_st
                .concretize_target_doubled
                .as_ref()
                .ok_or(SnarkError::ShapeMismatch {
                    what: "rescale: missing concretize_target_doubled state",
                })?,
            pass_com
                .concretize_target_doubled
                .as_ref()
                .ok_or(SnarkError::ShapeMismatch {
                    what: "rescale: missing concretize_target_doubled commit",
                })?,
            pass_st
                .concretize_acc_w
                .as_ref()
                .ok_or(SnarkError::ShapeMismatch {
                    what: "rescale: missing concretize_acc_w state",
                })?,
            pass_com
                .concretize_acc_w
                .as_ref()
                .ok_or(SnarkError::ShapeMismatch {
                    what: "rescale: missing concretize_acc_w commit",
                })?,
            params,
            sponge,
            rng,
        )?;
        events.push(proof);
    }

    Ok(events)
}

/// Walk the architecture/property to derive the canonical event
/// layout, then verify each per-event proof. The expected `(c1, c2)`
/// is computed from `(layer_scales, working, input_scale)` and bound
/// against the prover's claim.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_rescale_proofs(
    proofs: &[crate::snark::rescaling::RescaleEventProof],
    pass_com: &PassCommitments,
    arch: &NetworkArchitecture,
    property: &Property,
    has_concretize: bool,
    layer_scales: &crate::snark::proof::LayerScalesCommit,
    working: crate::quantization::scale::Scale,
    input_scale: crate::quantization::scale::Scale,
    bound_dir: crate::quantized_crown::BoundDir,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    let bias_dir = match bound_dir {
        crate::quantized_crown::BoundDir::Lower => {
            crate::quantization::quantized_scalar::RoundDir::Floor
        }
        crate::quantized_crown::BoundDir::Upper => {
            crate::quantization::quantized_scalar::RoundDir::Ceil
        }
    };
    let layout = expected_rescale_layout(arch, property, has_concretize);
    if layout.len() != proofs.len() {
        return Err(SnarkError::ShapeMismatch {
            what: "rescale: event count mismatch",
        });
    }
    for (event, proof) in layout.iter().zip(proofs.iter()) {
        if event.n_vars != proof.n_vars {
            return Err(SnarkError::ShapeMismatch {
                what: "rescale: per-event n_vars mismatch",
            });
        }
        // Expected (c1, c2) from committed layer_scales + public
        // working/input scales. Reject if it disagrees with the proof.
        let expected_qx_scale: crate::quantization::scale::Scale = match event.kind {
            RescaleEventKind::LinearMatrix(_) => {
                let weight = crate::quantization::scale::Scale {
                    c: layer_scales.weight_c[event.layer_idx],
                    e: layer_scales.weight_e[event.layer_idx],
                };
                working
                    .compose(weight)
                    .map_err(|_| SnarkError::RescaleScaleMismatch)?
            }
            RescaleEventKind::LinearVector(_) => {
                let bias = crate::quantization::scale::Scale {
                    c: layer_scales.bias_c[event.layer_idx],
                    e: layer_scales.bias_e[event.layer_idx],
                };
                working
                    .compose(bias)
                    .map_err(|_| SnarkError::RescaleScaleMismatch)?
            }
            RescaleEventKind::ActMatrix(_) => {
                let d = crate::quantization::scale::Scale {
                    c: layer_scales.relax_d_c[event.layer_idx],
                    e: layer_scales.relax_d_e[event.layer_idx],
                };
                working
                    .compose(d)
                    .map_err(|_| SnarkError::RescaleScaleMismatch)?
            }
            RescaleEventKind::ActVector(_) => {
                let b = crate::quantization::scale::Scale {
                    c: layer_scales.relax_b_c[event.layer_idx],
                    e: layer_scales.relax_b_e[event.layer_idx],
                };
                working
                    .compose(b)
                    .map_err(|_| SnarkError::RescaleScaleMismatch)?
            }
            RescaleEventKind::ConcretizeVector => working
                .compose(input_scale)
                .map_err(|_| SnarkError::RescaleScaleMismatch)?,
        };
        // c1/c2 = working / qx_scale. ratio_as_c1_c2 is fallible:
        // reject a malicious composed scale before any shift-width
        // overflow rather than after.
        let expected_ratio = working
            .ratio_as_c1_c2(expected_qx_scale)
            .map_err(|_| SnarkError::RescaleScaleMismatch)?;
        let expected_c1_fr =
            crate::snark_primitives::finite_field::signed_lift_to_fr(expected_ratio.c1);
        let expected_c2_fr =
            crate::snark_primitives::finite_field::signed_lift_to_fr(expected_ratio.c2);
        if expected_c1_fr != proof.c1_fr || expected_c2_fr != proof.c2_fr {
            return Err(SnarkError::RescaleScaleMismatch);
        }
        // Matrix events round HalfAway; vector events follow bias_dir.
        let event_dir = match event.kind {
            RescaleEventKind::LinearMatrix(_) | RescaleEventKind::ActMatrix(_) => {
                crate::quantization::quantized_scalar::RoundDir::HalfAway
            }
            RescaleEventKind::LinearVector(_)
            | RescaleEventKind::ActVector(_)
            | RescaleEventKind::ConcretizeVector => bias_dir,
        };
        let desc = crate::snark::rescaling::RescaleEventDesc {
            c1: expected_ratio.c1,
            c2: expected_ratio.c2,
            n_vars: event.n_vars,
            dir: event_dir,
        };
        let (qx_com, qz_com) = match event.kind {
            RescaleEventKind::LinearMatrix(step_idx) => (
                &pass_com.linear_a_w[step_idx],
                &pass_com.chain_a[event.layer_idx],
            ),
            RescaleEventKind::LinearVector(step_idx) => (
                &pass_com.linear_a_b[step_idx],
                &pass_com.linear_prod_w[step_idx],
            ),
            RescaleEventKind::ActMatrix(step_idx) => (
                &pass_com.activation_a_d_doubled[step_idx],
                &pass_com.chain_a[event.layer_idx],
            ),
            RescaleEventKind::ActVector(step_idx) => (
                &pass_com.activation_bias_doubled[step_idx],
                &pass_com.activation_bias_delta[step_idx],
            ),
            RescaleEventKind::ConcretizeVector => (
                pass_com
                    .concretize_target_doubled
                    .as_ref()
                    .ok_or(SnarkError::ShapeMismatch {
                        what: "rescale: missing concretize_target_doubled (verify)",
                    })?,
                pass_com
                    .concretize_acc_w
                    .as_ref()
                    .ok_or(SnarkError::ShapeMismatch {
                        what: "rescale: missing concretize_acc_w (verify)",
                    })?,
            ),
        };
        crate::snark::rescaling::verify_rescale_event(
            proof, &desc, qx_com, qz_com, params, sponge,
        )?;
    }
    Ok(())
}
