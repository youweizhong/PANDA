//! Dev tool: compute both the float-CROWN and quantized-CROWN bounds
//! on the same fixture, dump per-layer pre-activation codes, and
//! print a forward eval at the input center for sanity.
//!
//! Usage:
//!     cargo run --release --bin panda_drift_check -- <fixture.json>
//!
//! The fixture is the unified PANDA schema produced by
//! `evaluation/preprocess/preprocessing.py`.

use std::env;
use std::fs;
use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde::Deserialize;

use panda::{
    backward_bound, quantized_backward_bound, ActivationKind, Layer, Network, Property, Side,
};

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct External {
    activations: Vec<String>,
    input_dim: usize,
    output_dim: usize,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
    x_lower: Vec<f64>,
    x_upper: Vec<f64>,
    spec_c: Vec<Vec<f64>>,
    spec_d: Vec<f64>,
    side: String,
    precision_bits: i32,
}

fn parse_kind(s: &str) -> ActivationKind {
    match s {
        "relu" => ActivationKind::ReLU,
        "sigmoid" => ActivationKind::Sigmoid,
        "tanh" => ActivationKind::Tanh,
        other => panic!("unsupported: {other}"),
    }
}

fn build_network(fix: &External) -> Network {
    let n_linear = fix.weights.len();
    let mut layers = Vec::new();
    for i in 0..n_linear {
        let rows = fix.weights[i].len();
        let cols = fix.weights[i][0].len();
        let mut w = Array2::<f64>::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                w[[r, c]] = fix.weights[i][r][c];
            }
        }
        let b = Array1::<f64>::from(fix.biases[i].clone());
        layers.push(Layer::linear(w, b).unwrap());
        if i + 1 < n_linear {
            layers.push(Layer::Activation {
                kind: parse_kind(&fix.activations[i]),
            });
        }
    }
    Network::new(layers).unwrap()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = PathBuf::from(&args[1]);
    let fix: External = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    let net = build_network(&fix);
    let n_spec = fix.spec_c.len();
    let n_out = fix.output_dim;
    let mut c = Array2::<f64>::zeros((n_spec, n_out));
    for i in 0..n_spec {
        for j in 0..n_out {
            c[[i, j]] = fix.spec_c[i][j];
        }
    }
    let d = Array1::<f64>::from(fix.spec_d.clone());
    let side = match fix.side.as_str() {
        "lower" => Side::Lower,
        "upper" => Side::Upper,
        "both" => Side::Both,
        _ => panic!(),
    };
    let property = Property::new(c, d, side).unwrap();
    let xl: Array1<f64> = fix.x_lower.clone().into();
    let xu: Array1<f64> = fix.x_upper.clone().into();

    println!("=== {} ===", path.display());
    println!("precision_bits={}, side={}", fix.precision_bits, fix.side);

    let plain = backward_bound(&net, &property, &xl, &xu).unwrap();
    println!("\nFloat CROWN:");
    if let Some(lo) = plain.target_lower.as_ref() {
        println!("  lower: {:?}", lo.as_slice().unwrap());
    }
    if let Some(up) = plain.target_upper.as_ref() {
        println!("  upper: {:?}", up.as_slice().unwrap());
    }

    let quant = quantized_backward_bound(&net, &property, &xl, &xu, fix.precision_bits).unwrap();
    println!("  working_scale={:?}", quant.scales.working);
    println!("  input_scale={:?}", quant.scales.input);
    if let Some(tl) = quant.target_lower.as_ref() {
        println!(
            "  target_lower codes: {:?} scale={:?}",
            tl.codes.as_slice().unwrap(),
            tl.scale
        );
    }
    if let Some(tu) = quant.target_upper.as_ref() {
        println!(
            "  target_upper codes: {:?} scale={:?}",
            tu.codes.as_slice().unwrap(),
            tu.scale
        );
    }
    println!("  preact_lower (per hidden Linear):");
    for (i, pl) in quant.preact_lower.iter().enumerate() {
        if let Some(p) = pl.as_ref() {
            println!(
                "    layer {i}: codes={:?} max_abs={}",
                p.codes.as_slice().unwrap(),
                p.codes.iter().map(|c| c.abs()).max().unwrap_or(0)
            );
        }
    }
    println!("  preact_upper:");
    for (i, pu) in quant.preact_upper.iter().enumerate() {
        if let Some(p) = pu.as_ref() {
            println!(
                "    layer {i}: codes={:?} max_abs={}",
                p.codes.as_slice().unwrap(),
                p.codes.iter().map(|c| c.abs()).max().unwrap_or(0)
            );
        }
    }
    let (qlo, qup) = quant.final_bound_real();
    println!("\nQuantized CROWN:");
    if let Some(lo) = qlo.as_ref() {
        println!("  lower: {:?}", lo.as_slice().unwrap());
    }
    if let Some(up) = qup.as_ref() {
        println!("  upper: {:?}", up.as_slice().unwrap());
    }

    // Forward eval at the box center; should land inside both bounds.
    let mut h = (xl.clone() + xu.clone()) * 0.5;
    for layer in net.layers() {
        h = match layer {
            Layer::Linear { weight, bias } => weight.dot(&h) + bias,
            Layer::Activation { kind } => match kind {
                ActivationKind::ReLU => h.mapv(|v| v.max(0.0)),
                ActivationKind::Sigmoid => h.mapv(|v| {
                    if v >= 0.0 {
                        1.0 / (1.0 + (-v).exp())
                    } else {
                        let e = v.exp();
                        e / (1.0 + e)
                    }
                }),
                ActivationKind::Tanh => h.mapv(|v| v.tanh()),
            },
        };
    }
    println!("\nForward at input center:");
    println!("  output: {:?}", h.as_slice().unwrap());
}
