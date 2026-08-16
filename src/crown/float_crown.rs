//! Plaintext (f64) backward-CROWN bound generation for flat MLPs.
//!
//! The algorithm runs in three phases:
//!   1. forward sweep — for every linear layer, run backward CROWN with
//!      `C = I` to obtain pre-activation lower/upper bounds;
//!   2. for every activation layer, build a per-neuron canonical CROWN
//!      relaxation from its predecessor's pre-activation bounds;
//!   3. one final backward CROWN sweep using the full property matrix `C`.
//!
//! This module is the reference implementation. The quantized engine in
//! [`crate::quantized_crown`] sits next to it and follows the same
//! structure with integer arithmetic.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::crown::network::{ActivationKind, Layer, Network};
use crate::crown::output_property::{Property, Side};

/// Per-neuron linear relaxation: two affine bounds
/// `d_lower · z + b_lower ≤ φ(z) ≤ d_upper · z + b_upper` valid on the
/// neuron's pre-activation interval. Used uniformly for ReLU, sigmoid,
/// and tanh.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReluRelaxation {
    pub d_lower: f64,
    pub b_lower: f64,
    pub d_upper: f64,
    pub b_upper: f64,
}

/// Canonical CROWN ReLU relaxation for a single neuron with pre-activation
/// bounds `[l, u]`.
pub fn relu_relaxation(l: f64, u: f64) -> ReluRelaxation {
    if l >= 0.0 {
        return ReluRelaxation {
            d_lower: 1.0,
            b_lower: 0.0,
            d_upper: 1.0,
            b_upper: 0.0,
        };
    }
    if u <= 0.0 {
        return ReluRelaxation {
            d_lower: 0.0,
            b_lower: 0.0,
            d_upper: 0.0,
            b_upper: 0.0,
        };
    }
    let d_upper = u / (u - l);
    let b_upper = -l * u / (u - l);
    let d_lower = if u > -l { 1.0 } else { 0.0 };
    ReluRelaxation {
        d_lower,
        b_lower: 0.0,
        d_upper,
        b_upper,
    }
}

/// Canonical CROWN relaxation for a symmetric S-shaped activation.
///
/// `f` is the activation and `fp` its derivative. Uses a chord on the
/// purely convex or purely concave half-line, and a tangent + chord pair
/// in the mixed `l < 0 < u` case (following auto_LiRPA).
pub fn sshape_relaxation(
    l: f64,
    u: f64,
    f: impl Fn(f64) -> f64,
    fp: impl Fn(f64) -> f64,
) -> ReluRelaxation {
    let fl = f(l);
    let fu = f(u);
    if u == l {
        return ReluRelaxation {
            d_lower: 0.0,
            b_lower: fl,
            d_upper: 0.0,
            b_upper: fu,
        };
    }
    if u <= 0.0 {
        // Convex region: upper line is the chord, lower line is the
        // midpoint tangent.
        let k = (fu - fl) / (u - l);
        let b_chord = fl - k * l;
        let m = 0.5 * (l + u);
        let s = fp(m);
        let fm = f(m);
        return ReluRelaxation {
            d_lower: s,
            b_lower: fm - s * m,
            d_upper: k,
            b_upper: b_chord,
        };
    }
    if l >= 0.0 {
        // Concave region: roles swapped from the convex branch.
        let k = (fu - fl) / (u - l);
        let b_chord = fl - k * l;
        let m = 0.5 * (l + u);
        let s = fp(m);
        let fm = f(m);
        return ReluRelaxation {
            d_lower: k,
            b_lower: b_chord,
            d_upper: s,
            b_upper: fm - s * m,
        };
    }
    let k_direct = (fu - fl) / (u - l);
    let sl = fp(l);
    let g_at_l = fl + sl * (u - l) - fu;
    let (d_l_s, d_l_b) = if g_at_l <= 0.0 {
        let d = bisect(|z| f(z) + fp(z) * (u - z) - fu, l, 0.0);
        let s = fp(d);
        (s, f(d) - s * d)
    } else {
        (k_direct, fl - k_direct * l)
    };
    let su = fp(u);
    let h_at_u = fu + su * (l - u) - fl;
    let (d_u_s, d_u_b) = if h_at_u >= 0.0 {
        let d = bisect(|z| f(z) + fp(z) * (l - z) - fl, 0.0, u);
        let s = fp(d);
        (s, f(d) - s * d)
    } else {
        (k_direct, fl - k_direct * l)
    };
    ReluRelaxation {
        d_lower: d_l_s,
        b_lower: d_l_b,
        d_upper: d_u_s,
        b_upper: d_u_b,
    }
}

/// Bisection finder for `g(z) = 0` with `g(lo) * g(hi) <= 0` and `g`
/// monotone on `[lo, hi]`. 60 iterations is well below f64 epsilon.
fn bisect(g: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    let _timing = crate::timing::tangent_scope();
    let mut a = lo;
    let mut b = hi;
    let ga = g(a);
    if ga == 0.0 {
        return a;
    }
    let gb = g(b);
    if gb == 0.0 {
        return b;
    }
    debug_assert!(ga.signum() != gb.signum() || ga == 0.0 || gb == 0.0);
    for _ in 0..60 {
        let m = 0.5 * (a + b);
        let gm = g(m);
        if gm == 0.0 {
            return m;
        }
        if gm.signum() == ga.signum() {
            a = m;
        } else {
            b = m;
        }
    }
    0.5 * (a + b)
}

/// CROWN sigmoid relaxation on the pre-activation interval `[l, u]`.
pub fn sigmoid_relaxation(l: f64, u: f64) -> ReluRelaxation {
    sshape_relaxation(l, u, sigmoid_f, sigmoid_fp)
}

/// CROWN tanh relaxation on the pre-activation interval `[l, u]`.
pub fn tanh_relaxation(l: f64, u: f64) -> ReluRelaxation {
    sshape_relaxation(l, u, tanh_f, tanh_fp)
}

/// Numerically stable logistic sigmoid.
pub fn sigmoid_f(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Derivative of [`sigmoid_f`].
pub fn sigmoid_fp(z: f64) -> f64 {
    let s = sigmoid_f(z);
    s * (1.0 - s)
}

/// Hyperbolic tangent.
pub fn tanh_f(z: f64) -> f64 {
    z.tanh()
}

/// Derivative of [`tanh_f`].
pub fn tanh_fp(z: f64) -> f64 {
    let t = z.tanh();
    1.0 - t * t
}

/// Per-layer relaxation table: one [`ReluRelaxation`] per neuron in the
/// activation layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivationRelaxation {
    pub kind: ActivationKind,
    pub neurons: Vec<ReluRelaxation>,
}

/// Build the per-neuron relaxation table for an activation layer from its
/// predecessor's pre-activation bounds `l_vec` and `u_vec`.
pub fn relax_layer(
    kind: ActivationKind,
    l_vec: &Array1<f64>,
    u_vec: &Array1<f64>,
) -> Result<ActivationRelaxation, PlaintextError> {
    if l_vec.len() != u_vec.len() {
        return Err(PlaintextError::PreactShapeMismatch {
            l: l_vec.len(),
            u: u_vec.len(),
        });
    }
    match kind {
        ActivationKind::ReLU => {
            let neurons = (0..l_vec.len())
                .map(|j| relu_relaxation(l_vec[j], u_vec[j]))
                .collect();
            Ok(ActivationRelaxation { kind, neurons })
        }
        ActivationKind::Sigmoid => {
            let neurons = (0..l_vec.len())
                .map(|j| sigmoid_relaxation(l_vec[j], u_vec[j]))
                .collect();
            Ok(ActivationRelaxation { kind, neurons })
        }
        ActivationKind::Tanh => {
            let neurons = (0..l_vec.len())
                .map(|j| tanh_relaxation(l_vec[j], u_vec[j]))
                .collect();
            Ok(ActivationRelaxation { kind, neurons })
        }
    }
}

/// Concretize the linear functional `A x + b` over the input box
/// `[x_lower, x_upper]` to the requested `mode`. `Side::Both` is rejected
/// — call once per side.
pub fn concretize(
    a_matrix: &Array2<f64>,
    b_vec: &Array1<f64>,
    x_lower: &Array1<f64>,
    x_upper: &Array1<f64>,
    mode: Side,
) -> Array1<f64> {
    debug_assert_eq!(a_matrix.ncols(), x_lower.len());
    debug_assert_eq!(x_lower.len(), x_upper.len());
    debug_assert_eq!(a_matrix.nrows(), b_vec.len());
    let pos = a_matrix.mapv(|v| v.max(0.0));
    let neg = a_matrix.mapv(|v| v.min(0.0));
    match mode {
        Side::Lower => pos.dot(x_lower) + neg.dot(x_upper) + b_vec,
        Side::Upper => pos.dot(x_upper) + neg.dot(x_lower) + b_vec,
        Side::Both => unreachable!("call concretize once per side"),
    }
}

#[derive(Copy, Clone, Debug)]
enum BoundDir {
    Lower,
    Upper,
}

/// Inputs shared by every backward-CROWN sweep within a single
/// [`backward_bound`] call. `relaxations[i]` is `Some` only for activation
/// layers up to the current sweep's `start_idx`.
struct BackwardCtx<'a> {
    layers: &'a [Layer],
    relaxations: &'a [Option<ActivationRelaxation>],
    x_lower: &'a Array1<f64>,
    x_upper: &'a Array1<f64>,
}

fn backward_pass(
    ctx: &BackwardCtx<'_>,
    start_idx: usize,
    c_matrix: &Array2<f64>,
    d_vec: &Array1<f64>,
    dir: BoundDir,
) -> Array1<f64> {
    let layers = ctx.layers;
    let relaxations = ctx.relaxations;
    let x_lower = ctx.x_lower;
    let x_upper = ctx.x_upper;
    let mut a = c_matrix.clone();
    let mut b_acc = d_vec.clone();
    for i in (0..=start_idx).rev() {
        match &layers[i] {
            Layer::Linear { weight, bias } => {
                b_acc += &a.dot(bias);
                a = a.dot(weight);
            }
            Layer::Activation { .. } => {
                let relax = relaxations[i]
                    .as_ref()
                    .expect("activation relaxation must be populated before backward sweep");
                let neurons = &relax.neurons;
                debug_assert_eq!(a.ncols(), neurons.len());
                for spec in 0..a.nrows() {
                    let mut delta_b = 0.0;
                    for j in 0..neurons.len() {
                        let coeff = a[[spec, j]];
                        let neuron = &neurons[j];
                        let pick_lower = match dir {
                            BoundDir::Lower => coeff >= 0.0,
                            BoundDir::Upper => coeff < 0.0,
                        };
                        let (d_pick, b_pick) = if pick_lower {
                            (neuron.d_lower, neuron.b_lower)
                        } else {
                            (neuron.d_upper, neuron.b_upper)
                        };
                        a[[spec, j]] = coeff * d_pick;
                        delta_b += coeff * b_pick;
                    }
                    b_acc[spec] += delta_b;
                }
            }
        }
    }
    let mode = match dir {
        BoundDir::Lower => Side::Lower,
        BoundDir::Upper => Side::Upper,
    };
    concretize(&a, &b_acc, x_lower, x_upper, mode)
}

/// Plaintext CROWN certificate.
///
/// Holds the pre-activation bounds at every linear layer, the canonical
/// activation relaxations, and the final target bounds for the requested
/// side(s). Entries in the per-layer vectors are `None` at activation
/// positions for `preact_*`, and `None` at linear positions for
/// `relaxations`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlainCert {
    pub preact_lower: Vec<Option<Array1<f64>>>,
    pub preact_upper: Vec<Option<Array1<f64>>>,
    pub relaxations: Vec<Option<ActivationRelaxation>>,
    pub target_lower: Option<Array1<f64>>,
    pub target_upper: Option<Array1<f64>>,
}

/// Final target bounds for the certified property — the headline numbers
/// without the rest of the [`PlainCert`].
#[derive(Clone, Debug)]
pub struct BackwardBound {
    pub lower: Option<Array1<f64>>,
    pub upper: Option<Array1<f64>>,
}

/// Absorb float round-off in a CROWN preact pair `(lo, up)`.
///
/// Mathematically `lo <= up` always holds, but the two backward passes
/// accumulate noise independently and can cross by a few ULP at neurons
/// whose true bound is essentially constant. Inversions within a
/// per-neuron tolerance are symmetrised around the midpoint and widened
/// by `tol` (sound, slightly looser). Larger inversions return
/// `PreactCrossed`.
fn clamp_or_reject_float_inversion(
    lo: &mut Array1<f64>,
    up: &mut Array1<f64>,
    layer_idx: usize,
) -> Result<(), PlaintextError> {
    debug_assert_eq!(lo.len(), up.len());
    // 1024·ULP covers the noise budget for the widest MNIST-scale layers
    // in scope by several orders of magnitude.
    const ULP_BUDGET: f64 = 1024.0 * f64::EPSILON;
    for j in 0..lo.len() {
        if lo[j] <= up[j] {
            continue;
        }
        let scale = lo[j].abs().max(up[j].abs()).max(1.0);
        let tol = scale * ULP_BUDGET;
        let crossing = lo[j] - up[j];
        if crossing > tol {
            return Err(PlaintextError::PreactCrossed { layer: layer_idx });
        }
        let mid = 0.5 * (lo[j] + up[j]);
        lo[j] = mid - tol;
        up[j] = mid + tol;
    }
    Ok(())
}

/// Generate the plaintext CROWN certificate: forward sweep over the
/// linear layers, build the activation relaxations, then one final
/// backward pass with the property matrix.
///
/// Inputs:
///
/// * `network` — the MLP being certified.
/// * `property` — the output linear property `C·y + d` and the side(s).
/// * `x_lower`, `x_upper` — the input box; must match `network.input_dim()`
///   and satisfy `x_lower ≤ x_upper` element-wise.
///
/// Returns a [`PlainCert`] on success, or a [`PlaintextError`] if the
/// shapes are inconsistent or if a pre-activation envelope crosses by
/// more than the tolerated float noise.
pub fn backward_bound(
    network: &Network,
    property: &Property,
    x_lower: &Array1<f64>,
    x_upper: &Array1<f64>,
) -> Result<PlainCert, PlaintextError> {
    if x_lower.len() != x_upper.len() || x_lower.len() != network.input_dim() {
        return Err(PlaintextError::InputBoxMismatch {
            x_lower: x_lower.len(),
            x_upper: x_upper.len(),
            input_dim: network.input_dim(),
        });
    }
    if x_lower.iter().zip(x_upper.iter()).any(|(l, u)| l > u) {
        return Err(PlaintextError::InputBoxInverted);
    }
    if property.output_dim() != network.output_dim() {
        return Err(PlaintextError::PropertyOutputDimMismatch {
            property: property.output_dim(),
            network: network.output_dim(),
        });
    }

    let layers = network.layers();
    let n_layers = layers.len();
    let mut preact_lower: Vec<Option<Array1<f64>>> = vec![None; n_layers];
    let mut preact_upper: Vec<Option<Array1<f64>>> = vec![None; n_layers];
    let mut relaxations: Vec<Option<ActivationRelaxation>> = vec![None; n_layers];

    for idx in 0..n_layers {
        match &layers[idx] {
            Layer::Linear { weight, .. } => {
                let n_out = weight.nrows();
                let identity = Array2::<f64>::eye(n_out);
                let zero = Array1::<f64>::zeros(n_out);
                let ctx = BackwardCtx {
                    layers,
                    relaxations: &relaxations,
                    x_lower,
                    x_upper,
                };
                let mut lo = backward_pass(&ctx, idx, &identity, &zero, BoundDir::Lower);
                let mut up = backward_pass(&ctx, idx, &identity, &zero, BoundDir::Upper);
                clamp_or_reject_float_inversion(&mut lo, &mut up, idx)?;
                preact_lower[idx] = Some(lo);
                preact_upper[idx] = Some(up);
            }
            Layer::Activation { kind } => {
                let prev = idx.checked_sub(1).ok_or(PlaintextError::ActivationFirst)?;
                let lo = preact_lower[prev]
                    .as_ref()
                    .ok_or(PlaintextError::MissingPreactBounds { layer: prev })?;
                let up = preact_upper[prev]
                    .as_ref()
                    .ok_or(PlaintextError::MissingPreactBounds { layer: prev })?;
                relaxations[idx] = Some(relax_layer(*kind, lo, up)?);
            }
        }
    }

    let final_idx = n_layers - 1;
    let ctx = BackwardCtx {
        layers,
        relaxations: &relaxations,
        x_lower,
        x_upper,
    };
    let target_lower = property.side.needs_lower().then(|| {
        backward_pass(
            &ctx,
            final_idx,
            &property.c_matrix,
            &property.d_vector,
            BoundDir::Lower,
        )
    });
    let target_upper = property.side.needs_upper().then(|| {
        backward_pass(
            &ctx,
            final_idx,
            &property.c_matrix,
            &property.d_vector,
            BoundDir::Upper,
        )
    });

    if let (Some(l), Some(u)) = (target_lower.as_ref(), target_upper.as_ref()) {
        if l.iter().zip(u.iter()).any(|(lv, uv)| lv > uv) {
            return Err(PlaintextError::TargetCrossed);
        }
    }

    Ok(PlainCert {
        preact_lower,
        preact_upper,
        relaxations,
        target_lower,
        target_upper,
    })
}

/// Recompute the final target bounds using a caller-supplied
/// `relaxations` table instead of the canonical one. Useful for tamper
/// tests that drive CROWN with a non-canonical relaxation to observe the
/// resulting soundness failure.
pub fn recompute_target_bounds(
    network: &Network,
    relaxations: &[Option<ActivationRelaxation>],
    property: &Property,
    x_lower: &Array1<f64>,
    x_upper: &Array1<f64>,
) -> BackwardBound {
    let layers = network.layers();
    let final_idx = layers.len() - 1;
    let ctx = BackwardCtx {
        layers,
        relaxations,
        x_lower,
        x_upper,
    };
    let lower = property.side.needs_lower().then(|| {
        backward_pass(
            &ctx,
            final_idx,
            &property.c_matrix,
            &property.d_vector,
            BoundDir::Lower,
        )
    });
    let upper = property.side.needs_upper().then(|| {
        backward_pass(
            &ctx,
            final_idx,
            &property.c_matrix,
            &property.d_vector,
            BoundDir::Upper,
        )
    });
    BackwardBound { lower, upper }
}

impl PlainCert {
    /// Final target bounds for the certified property.
    pub fn final_bound(&self) -> BackwardBound {
        BackwardBound {
            lower: self.target_lower.clone(),
            upper: self.target_upper.clone(),
        }
    }

    /// Coarse soundness check used by tests: every corner of the input
    /// box must produce an output that lies inside the certificate's
    /// reported bounds, up to a small numerical slack.
    pub fn corners_inside(
        &self,
        network: &Network,
        property: &Property,
        x_lower: &Array1<f64>,
        x_upper: &Array1<f64>,
    ) -> bool {
        let n = x_lower.len();
        let total = 1usize << n;
        for mask in 0..total {
            let mut x = Array1::<f64>::zeros(n);
            for i in 0..n {
                x[i] = if (mask >> i) & 1 == 1 {
                    x_upper[i]
                } else {
                    x_lower[i]
                };
            }
            let y = network.forward(&x);
            let v = property.c_matrix.dot(&y) + &property.d_vector;
            if let Some(lo) = &self.target_lower {
                if v.iter().zip(lo.iter()).any(|(vv, ll)| *vv < *ll - 1e-9) {
                    return false;
                }
            }
            if let Some(up) = &self.target_upper {
                if v.iter().zip(up.iter()).any(|(vv, uu)| *vv > *uu + 1e-9) {
                    return false;
                }
            }
        }
        true
    }
}

/// Errors raised by the plaintext CROWN engine.
#[derive(Debug, Error)]
pub enum PlaintextError {
    #[error("input box dims mismatch: x_lower={x_lower}, x_upper={x_upper}, network input_dim={input_dim}")]
    InputBoxMismatch {
        x_lower: usize,
        x_upper: usize,
        input_dim: usize,
    },
    #[error("input box has at least one coordinate where lower > upper")]
    InputBoxInverted,
    #[error("property output dim {property} != network output dim {network}")]
    PropertyOutputDimMismatch { property: usize, network: usize },
    #[error("preactivation lower vector has length {l} but upper has length {u}")]
    PreactShapeMismatch { l: usize, u: usize },
    #[error("preactivation bounds cross at layer {layer}: lower > upper")]
    PreactCrossed { layer: usize },
    #[error("target bounds cross: lower > upper")]
    TargetCrossed,
    #[error("activation appears as the first layer")]
    ActivationFirst,
    #[error("missing pre-activation bounds for layer {layer}")]
    MissingPreactBounds { layer: usize },
}
