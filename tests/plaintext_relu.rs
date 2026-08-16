//! End-to-end plaintext CROWN tests for flat ReLU MLPs.
//!
//! Each test builds a small hand-rolled MLP, generates the cert, and
//! checks one of:
//!   1. canonical relaxation values for known pre-activation bounds,
//!   2. corner-soundness — the cert's target bounds enclose the
//!      network output at every corner of the input box,
//!   3. tamper detection — mutating the relaxation or pre-activation
//!      bounds makes corner-soundness fail.

use ndarray::{array, Array1, Array2};
use panda::crown::float_crown::{relu_relaxation, ReluRelaxation};
use panda::{
    backward_bound, recompute_target_bounds, ActivationKind, ActivationRelaxation, BackwardBound,
    Layer, Network, PlainCert, Property, Side,
};

fn small_relu_net() -> Network {
    // 2 -> 3 -> 2: one hidden ReLU layer.
    let w1 = array![[1.0, 2.0], [-1.0, 1.0], [0.5, -0.5]];
    let b1 = array![0.0, 0.5, -0.25];
    let w2 = array![[1.0, -1.0, 2.0], [0.0, 1.0, 1.0]];
    let b2 = array![0.1, -0.2];
    Network::new(vec![
        Layer::linear(w1, b1).unwrap(),
        Layer::relu(),
        Layer::linear(w2, b2).unwrap(),
    ])
    .unwrap()
}

fn id_property(out: usize) -> Property {
    Property::new(Array2::eye(out), Array1::zeros(out), Side::Both).unwrap()
}

#[test]
fn relu_relaxation_table_active() {
    let r = relu_relaxation(0.5, 1.5);
    assert_eq!(r.d_lower, 1.0);
    assert_eq!(r.b_lower, 0.0);
    assert_eq!(r.d_upper, 1.0);
    assert_eq!(r.b_upper, 0.0);
}

#[test]
fn relu_relaxation_table_inactive() {
    let r = relu_relaxation(-2.0, -0.5);
    assert_eq!(r.d_lower, 0.0);
    assert_eq!(r.b_lower, 0.0);
    assert_eq!(r.d_upper, 0.0);
    assert_eq!(r.b_upper, 0.0);
}

#[test]
fn relu_relaxation_table_unstable_positive_dom() {
    // u > -l so the lower-line slope is 1.
    let r = relu_relaxation(-1.0, 3.0);
    assert_eq!(r.d_lower, 1.0);
    assert_eq!(r.b_lower, 0.0);
    let expected_d_upper = 3.0 / (3.0 - -1.0);
    let expected_b_upper = -(-1.0) * 3.0 / (3.0 - -1.0);
    assert_eq!(r.d_upper, expected_d_upper);
    assert_eq!(r.b_upper, expected_b_upper);
}

#[test]
fn relu_relaxation_table_unstable_negative_dom() {
    // u < -l so the lower-line slope is 0.
    let r = relu_relaxation(-3.0, 1.0);
    assert_eq!(r.d_lower, 0.0);
    assert_eq!(r.b_lower, 0.0);
}

#[test]
fn cert_bounds_enclose_corners_for_small_net() {
    let net = small_relu_net();
    let prop = id_property(net.output_dim());
    let x_l = array![-1.0, -0.5];
    let x_u = array![1.0, 0.75];
    let cert = backward_bound(&net, &prop, &x_l, &x_u).unwrap();
    assert!(cert.target_lower.is_some());
    assert!(cert.target_upper.is_some());
    assert!(cert.corners_inside(&net, &prop, &x_l, &x_u));
}

#[test]
fn linear_only_net_is_exact() {
    // A pure linear network has zero relaxation gap; the bound should equal
    // the IBP of the affine functional and therefore be tight at the
    // corresponding box corner.
    let w1 = array![[1.0, -2.0]];
    let b1 = array![0.5];
    let net = Network::new(vec![Layer::linear(w1, b1).unwrap()]).unwrap();
    let prop = id_property(net.output_dim());
    let x_l = array![-1.0, -1.0];
    let x_u = array![1.0, 1.0];
    let cert = backward_bound(&net, &prop, &x_l, &x_u).unwrap();
    let lo = cert.target_lower.as_ref().unwrap();
    let up = cert.target_upper.as_ref().unwrap();
    // x1 in [-1,1], x2 in [-1,1]. y = x1 - 2*x2 + 0.5. min at (-1, 1) -> -2.5;
    // max at (1, -1) -> 3.5.
    assert!((lo[0] - (-2.5)).abs() < 1e-12);
    assert!((up[0] - 3.5).abs() < 1e-12);
}

#[test]
fn upper_only_property_skips_lower_pass() {
    let net = small_relu_net();
    let prop = Property::new(
        Array2::eye(net.output_dim()),
        Array1::zeros(net.output_dim()),
        Side::Upper,
    )
    .unwrap();
    let x_l = array![-1.0, -0.5];
    let x_u = array![1.0, 0.75];
    let cert = backward_bound(&net, &prop, &x_l, &x_u).unwrap();
    assert!(cert.target_lower.is_none());
    assert!(cert.target_upper.is_some());
}

#[test]
fn property_with_nontrivial_c_uses_one_backward_run() {
    // C is a 2×2 mixing matrix on the network output; we feed it as one
    // matrix into one backward pass (per-row splitting is forbidden) and
    // check the bound encloses corner evaluations of `C @ y + d`.
    let net = small_relu_net();
    let c = array![[1.0, -1.0], [2.0, 1.0]];
    let d = array![0.25, -0.5];
    let prop = Property::new(c.clone(), d.clone(), Side::Both).unwrap();
    let x_l = array![-1.0, -0.5];
    let x_u = array![1.0, 0.75];
    let cert = backward_bound(&net, &prop, &x_l, &x_u).unwrap();

    // The cert's bound is on `C @ y + d`. Evaluate that on every corner.
    let lo = cert.target_lower.as_ref().unwrap();
    let up = cert.target_upper.as_ref().unwrap();
    for mask in 0..4 {
        let x = array![
            if mask & 1 == 1 { x_u[0] } else { x_l[0] },
            if mask & 2 == 2 { x_u[1] } else { x_l[1] }
        ];
        let y = net.forward(&x);
        let v = c.dot(&y) + &d;
        for k in 0..2 {
            assert!(
                v[k] >= lo[k] - 1e-9 && v[k] <= up[k] + 1e-9,
                "spec {} corner {} value {} outside [{}, {}]",
                k,
                mask,
                v[k],
                lo[k],
                up[k]
            );
        }
    }
}

#[test]
fn malicious_tamper_breaks_corner_soundness() {
    // If we shrink the cert's reported upper bound below the chord, at
    // least one corner of the box should no longer fit.
    let net = small_relu_net();
    let prop = id_property(net.output_dim());
    let x_l = array![-1.0, -0.5];
    let x_u = array![1.0, 0.75];
    let mut cert = backward_bound(&net, &prop, &x_l, &x_u).unwrap();
    let bad_upper = cert.target_upper.as_ref().unwrap() - 5.0;
    cert.target_upper = Some(bad_upper);
    assert!(!cert.corners_inside(&net, &prop, &x_l, &x_u));
}

#[test]
fn malicious_relaxation_tamper_breaks_soundness() {
    // Replace the canonical relaxation with one that pretends every neuron
    // is inactive (d_lower = d_upper = 0). The recomputed CROWN bound
    // collapses the network's contribution and the actual network output
    // at some box corner falls outside the (tampered) bound.
    let net = small_relu_net();
    let prop = id_property(net.output_dim());
    let x_l = array![-1.0, -0.5];
    let x_u = array![1.0, 0.75];
    let cert = backward_bound(&net, &prop, &x_l, &x_u).unwrap();
    // All-zero relaxation: pretends every neuron is in the inactive regime.
    let zeroed: Vec<Option<ActivationRelaxation>> = cert
        .relaxations
        .iter()
        .map(|opt| {
            opt.as_ref().map(|rel| ActivationRelaxation {
                kind: rel.kind,
                neurons: rel
                    .neurons
                    .iter()
                    .map(|_| ReluRelaxation {
                        d_lower: 0.0,
                        b_lower: 0.0,
                        d_upper: 0.0,
                        b_upper: 0.0,
                    })
                    .collect(),
            })
        })
        .collect();
    let tampered = recompute_target_bounds(&net, &zeroed, &prop, &x_l, &x_u);
    let lo = tampered.lower.expect("lower side requested");
    let up = tampered.upper.expect("upper side requested");

    // Find a corner whose actual output violates the tampered bound. The
    // relaxation says ReLU(z) = 0 everywhere, so the bound collapses to a
    // small affine functional in `x`; at least one corner lies outside.
    let mut violated = false;
    for mask in 0..4 {
        let x = array![
            if mask & 1 == 1 { x_u[0] } else { x_l[0] },
            if mask & 2 == 2 { x_u[1] } else { x_l[1] }
        ];
        let y = net.forward(&x);
        let v = prop.c_matrix.dot(&y) + &prop.d_vector;
        for k in 0..2 {
            if v[k] < lo[k] - 1e-9 || v[k] > up[k] + 1e-9 {
                violated = true;
            }
        }
    }
    assert!(violated, "tampered relaxation must produce an unsound cert");
}

#[test]
fn input_box_inverted_rejected() {
    let net = small_relu_net();
    let prop = id_property(net.output_dim());
    let x_l = array![1.0, 0.0];
    let x_u = array![-1.0, 0.5];
    let err = backward_bound(&net, &prop, &x_l, &x_u);
    assert!(err.is_err());
}

#[test]
fn property_dim_mismatch_rejected() {
    let net = small_relu_net();
    let bad = Property::new(
        Array2::eye(net.output_dim() + 1),
        Array1::zeros(net.output_dim() + 1),
        Side::Both,
    )
    .unwrap();
    let x_l = array![-1.0, -0.5];
    let x_u = array![1.0, 0.75];
    let err = backward_bound(&net, &bad, &x_l, &x_u);
    assert!(err.is_err());
}

// Compile-time sanity check that the public `BackwardBound` and
// `ActivationKind` exports are reachable.
#[allow(dead_code)]
fn _public_export_check() -> (BackwardBound, ActivationKind, PlainCert) {
    panic!("compile-time only");
}
