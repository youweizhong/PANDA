//! Quantized backward-CROWN cert tests: drift vs the float-CROWN
//! bound on the same network/property/box, plus consistency checks
//! on the recorded rescale witnesses.

use ndarray::{array, Array1, Array2};
use panda::{
    backward_bound, quantized_backward_bound, Layer, Network, Property, Side,
};

fn small_relu_net() -> Network {
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
fn quantized_bound_drift_within_tolerance() {
    let net = small_relu_net();
    let prop = id_property(net.output_dim());
    let x_l = array![-1.0, -0.5];
    let x_u = array![1.0, 0.75];
    let plain = backward_bound(&net, &prop, &x_l, &x_u).unwrap();
    let quant = quantized_backward_bound(&net, &prop, &x_l, &x_u, 14).unwrap();
    let (qlo, qup) = quant.final_bound_real();
    let plo = plain.target_lower.as_ref().unwrap();
    let pup = plain.target_upper.as_ref().unwrap();
    let qlo = qlo.unwrap();
    let qup = qup.unwrap();

    // Quantized bound should be slightly looser than float (lower side
    // <= float lower; upper side >= float upper) by no more than ~0.05
    // at 14-bit precision on this small net.
    for k in 0..plo.len() {
        assert!(
            qlo[k] <= plo[k] + 0.05 && qlo[k] >= plo[k] - 0.5,
            "lower drift at spec {k}: float={} quant={}",
            plo[k],
            qlo[k]
        );
        assert!(
            qup[k] >= pup[k] - 0.05 && qup[k] <= pup[k] + 0.5,
            "upper drift at spec {k}: float={} quant={}",
            pup[k],
            qup[k]
        );
    }
}

#[test]
fn quantized_bound_encloses_corners() {
    let net = small_relu_net();
    let prop = id_property(net.output_dim());
    let x_l = array![-1.0, -0.5];
    let x_u = array![1.0, 0.75];
    let quant = quantized_backward_bound(&net, &prop, &x_l, &x_u, 14).unwrap();
    let (qlo, qup) = quant.final_bound_real();
    let qlo = qlo.unwrap();
    let qup = qup.unwrap();

    // Corner near-soundness: every box corner's network output should
    // land inside the quantized bound up to rescale drift (empirically
    // <= 0.01 at 14-bit precision on this net).
    let slop = 0.01;
    for mask in 0..4 {
        let x = array![
            if mask & 1 == 1 { x_u[0] } else { x_l[0] },
            if mask & 2 == 2 { x_u[1] } else { x_l[1] }
        ];
        let y = net.forward(&x);
        for k in 0..2 {
            assert!(
                y[k] >= qlo[k] - slop && y[k] <= qup[k] + slop,
                "corner {mask} spec {k}: y={} outside quant [{}, {}] beyond slop {slop}",
                y[k],
                qlo[k],
                qup[k]
            );
        }
    }
}

#[test]
fn rescale_witnesses_are_consistent() {
    let net = small_relu_net();
    let prop = id_property(net.output_dim());
    let x_l = array![-1.0, -0.5];
    let x_u = array![1.0, 0.75];
    let quant = quantized_backward_bound(&net, &prop, &x_l, &x_u, 14).unwrap();
    assert!(!quant.witnesses.is_empty());
    for w in &quant.witnesses {
        // Range obligation: both slacks must be non-negative.
        assert!(w.slack_lo >= 0, "slack_lo {} < 0 in {:?}", w.slack_lo, w);
        assert!(w.slack_hi >= 0, "slack_hi {} < 0 in {:?}", w.slack_hi, w);
        // qy == 1 holds while the engine emits only pure rescales (no
        // true-div gadget yet).
        assert_eq!(w.qy, 1);
    }
}

#[test]
fn upper_only_property_skips_lower_pass_q() {
    let net = small_relu_net();
    let prop = Property::new(
        Array2::eye(net.output_dim()),
        Array1::zeros(net.output_dim()),
        Side::Upper,
    )
    .unwrap();
    let x_l = array![-1.0, -0.5];
    let x_u = array![1.0, 0.75];
    let quant = quantized_backward_bound(&net, &prop, &x_l, &x_u, 14).unwrap();
    assert!(quant.target_lower.is_none());
    assert!(quant.target_upper.is_some());
}

#[test]
fn quantized_runs_with_nontrivial_property() {
    let net = small_relu_net();
    let c = array![[1.0, -1.0], [2.0, 1.0]];
    let d = array![0.25, -0.5];
    let prop = Property::new(c.clone(), d.clone(), Side::Both).unwrap();
    let x_l = array![-1.0, -0.5];
    let x_u = array![1.0, 0.75];
    let quant = quantized_backward_bound(&net, &prop, &x_l, &x_u, 14).unwrap();
    let (qlo, qup) = quant.final_bound_real();
    let qlo = qlo.unwrap();
    let qup = qup.unwrap();
    for mask in 0..4 {
        let x = array![
            if mask & 1 == 1 { x_u[0] } else { x_l[0] },
            if mask & 2 == 2 { x_u[1] } else { x_l[1] }
        ];
        let y = net.forward(&x);
        let v = c.dot(&y) + &d;
        for k in 0..2 {
            assert!(
                v[k] >= qlo[k] - 0.01 && v[k] <= qup[k] + 0.01,
                "C·y+d corner {mask} spec {k}: {} outside [{}, {}]",
                v[k],
                qlo[k],
                qup[k]
            );
        }
    }
}

#[test]
fn quantized_rejects_bad_precision() {
    let net = small_relu_net();
    let prop = id_property(net.output_dim());
    let x_l = array![-1.0, -0.5];
    let x_u = array![1.0, 0.75];
    assert!(quantized_backward_bound(&net, &prop, &x_l, &x_u, 0).is_err());
    assert!(quantized_backward_bound(&net, &prop, &x_l, &x_u, 100).is_err());
}

#[test]
fn deterministic_codes_across_runs() {
    let net = small_relu_net();
    let prop = id_property(net.output_dim());
    let x_l = array![-1.0, -0.5];
    let x_u = array![1.0, 0.75];
    let q1 = quantized_backward_bound(&net, &prop, &x_l, &x_u, 14).unwrap();
    let q2 = quantized_backward_bound(&net, &prop, &x_l, &x_u, 14).unwrap();
    let lo1 = q1.target_lower.as_ref().unwrap();
    let lo2 = q2.target_lower.as_ref().unwrap();
    assert_eq!(lo1.codes, lo2.codes);
    assert_eq!(lo1.scale, lo2.scale);
    let up1 = q1.target_upper.as_ref().unwrap();
    let up2 = q2.target_upper.as_ref().unwrap();
    assert_eq!(up1.codes, up2.codes);
    assert_eq!(q1.witnesses.len(), q2.witnesses.len());
}

#[test]
fn linear_only_quant_runs_cleanly() {
    let w1: Array2<f64> = array![[1.0, -2.0]];
    let b1: Array1<f64> = array![0.5];
    let net = Network::new(vec![Layer::linear(w1, b1).unwrap()]).unwrap();
    let prop = id_property(net.output_dim());
    let x_l = array![-1.0, -1.0];
    let x_u = array![1.0, 1.0];
    let plain = backward_bound(&net, &prop, &x_l, &x_u).unwrap();
    let quant = quantized_backward_bound(&net, &prop, &x_l, &x_u, 14).unwrap();
    let (qlo, qup) = quant.final_bound_real();
    let qlo = qlo.unwrap();
    let qup = qup.unwrap();
    // Linear-only nets have zero relaxation gap; only rescale drift remains.
    let plo = &plain.target_lower.as_ref().unwrap()[0];
    let pup = &plain.target_upper.as_ref().unwrap()[0];
    assert!((plo - qlo[0]).abs() < 0.01);
    assert!((pup - qup[0]).abs() < 0.01);
}

#[test]
fn drift_shrinks_with_more_precision() {
    let net = small_relu_net();
    let prop = id_property(net.output_dim());
    let x_l = array![-1.0, -0.5];
    let x_u = array![1.0, 0.75];
    let plain = backward_bound(&net, &prop, &x_l, &x_u).unwrap();
    let plo = &plain.target_lower.as_ref().unwrap()[0];

    let q_low = quantized_backward_bound(&net, &prop, &x_l, &x_u, 8).unwrap();
    let q_high = quantized_backward_bound(&net, &prop, &x_l, &x_u, 14).unwrap();
    let (qlo_low, _) = q_low.final_bound_real();
    let (qlo_high, _) = q_high.final_bound_real();
    let drift_low = (qlo_low.unwrap()[0] - plo).abs();
    let drift_high = (qlo_high.unwrap()[0] - plo).abs();
    assert!(
        drift_high <= drift_low + 1e-9,
        "drift did not shrink: 8b={drift_low} 14b={drift_high}"
    );
}
