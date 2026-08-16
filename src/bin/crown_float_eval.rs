//! Run plain-text (float64) CROWN backward bound on a fixture file
//! and print a JSON record to stdout. This is the paper's "Vanilla
//! CROWN" reference column; the quantized SNARK proves a quantized
//! version of the same bound.
//!
//! Run from the workspace root:
//!
//! ```text
//!     cargo run --release --bin crown_float_eval -- \
//!         <fixture-path>
//! ```
//!
//! Two fixture schemas are auto-detected:
//!
//! 1. CROWN MNIST (`evaluation/benchmarks/crown_original*/<model>/<name>.json`): single
//!    `activation` string plus `input_dim`, `n_classes`. The property
//!    is built from a sibling `test_images.json` and the
//!    `(image_idx, other_class, epsilon)` CLI flags.
//!
//! 2. Unified suites (`evaluation/benchmarks/{FairProof,safeNLP,LunarLander}/<name>.json`): unified
//!    schema produced by `evaluation/preprocess/preprocessing.py` with
//!    per-layer `activations`, an input box, and a property
//!    `(spec_c, spec_d, side)` taken directly from the JSON.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use ndarray::{Array1, Array2};
use serde::Deserialize;

use panda::{backward_bound, ActivationKind, Layer, Network, Property, Side};

#[derive(Debug, Deserialize)]
struct CrownMnist {
    architecture: Option<String>,
    activation: String,
    input_dim: usize,
    n_classes: usize,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct CrownTestImages {
    images: Vec<Vec<f64>>,
    labels: Vec<usize>,
}

#[derive(Debug, Deserialize)]
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
}

fn build_network_from_weights(
    weights: &[Vec<Vec<f64>>],
    biases: &[Vec<f64>],
    activations: &[ActivationKind],
) -> Network {
    let n_linear = weights.len();
    assert_eq!(biases.len(), n_linear);
    assert_eq!(activations.len(), n_linear.saturating_sub(1));
    let mut layers = Vec::with_capacity(2 * n_linear - 1);
    for i in 0..n_linear {
        let rows = weights[i].len();
        let cols = weights[i][0].len();
        let mut w = Array2::<f64>::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                w[[r, c]] = weights[i][r][c];
            }
        }
        let b = Array1::<f64>::from(biases[i].clone());
        layers.push(Layer::linear(w, b).unwrap());
        if i + 1 < n_linear {
            layers.push(Layer::Activation {
                kind: activations[i],
            });
        }
    }
    Network::new(layers).unwrap()
}

fn parse_kind(s: &str) -> ActivationKind {
    match s {
        "relu" => ActivationKind::ReLU,
        "sigmoid" => ActivationKind::Sigmoid,
        "tanh" => ActivationKind::Tanh,
        other => panic!("unsupported activation kind: {other}"),
    }
}

fn run_crown_mnist(
    path: &PathBuf,
    image_idx: usize,
    other_class: usize,
    epsilon: f64,
) -> serde_json::Value {
    let mlp: CrownMnist = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    let images_path = path.parent().unwrap().join("test_images.json");
    let images: CrownTestImages =
        serde_json::from_str(&fs::read_to_string(&images_path).unwrap()).unwrap();

    let kind = parse_kind(&mlp.activation);
    let n_linear = mlp.weights.len();
    let activations = vec![kind; n_linear.saturating_sub(1)];
    let net = build_network_from_weights(&mlp.weights, &mlp.biases, &activations);

    let true_class = images.labels[image_idx];
    let img: Array1<f64> = images.images[image_idx].clone().into();
    let x_lower = img.mapv(|v| v - epsilon);
    let x_upper = img.mapv(|v| v + epsilon);

    // Robustness property: logit(true_class) - logit(other_class) >= 0.
    let mut c = Array2::<f64>::zeros((1, mlp.n_classes));
    c[[0, true_class]] = 1.0;
    c[[0, other_class]] = -1.0;
    let d = Array1::<f64>::zeros(1);
    let property = Property::new(c, d, Side::Lower).unwrap();

    let t0 = Instant::now();
    let cert = backward_bound(&net, &property, &x_lower, &x_upper).unwrap();
    let elapsed = t0.elapsed().as_secs_f64();
    let lower = cert
        .target_lower
        .as_ref()
        .map(|v| v.to_vec())
        .unwrap_or_default();
    let robust = lower.first().copied().unwrap_or(0.0) > 0.0;
    serde_json::json!({
        "name": path.file_stem().unwrap().to_string_lossy(),
        "schema": "crown_mnist",
        "activation": mlp.activation,
        "n_linear": n_linear,
        "input_dim": mlp.input_dim,
        "image_idx": image_idx,
        "true_class": true_class,
        "other_class": other_class,
        "epsilon": epsilon,
        "float_runtime_secs": elapsed,
        "float_lower_bound": lower,
        "robust_at_float": robust,
        "architecture": mlp.architecture,
    })
}

fn run_external(path: &PathBuf) -> serde_json::Value {
    let fix: External = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    let activations: Vec<ActivationKind> = fix.activations.iter().map(|s| parse_kind(s)).collect();
    let net = build_network_from_weights(&fix.weights, &fix.biases, &activations);
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
        other => panic!("bad side: {other}"),
    };
    let property = Property::new(c, d, side).unwrap();
    let x_lower: Array1<f64> = fix.x_lower.clone().into();
    let x_upper: Array1<f64> = fix.x_upper.clone().into();

    let t0 = Instant::now();
    let cert = backward_bound(&net, &property, &x_lower, &x_upper).unwrap();
    let elapsed = t0.elapsed().as_secs_f64();
    let lower = cert.target_lower.as_ref().map(|v| v.to_vec());
    let upper = cert.target_upper.as_ref().map(|v| v.to_vec());
    serde_json::json!({
        "name": path.file_stem().unwrap().to_string_lossy(),
        "schema": "external",
        "activations": fix.activations,
        "n_linear": fix.weights.len(),
        "input_dim": fix.input_dim,
        "output_dim": fix.output_dim,
        "side": fix.side,
        "float_runtime_secs": elapsed,
        "float_lower_bound": lower,
        "float_upper_bound": upper,
    })
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: cargo run --release --bin crown_float_eval -- \\
        <fixture-path> [image_idx other_class epsilon]"
        );
        std::process::exit(1);
    }
    let path = PathBuf::from(&args[1]);
    let body = fs::read_to_string(&path).expect("read fixture");
    // Schema detection: external fixtures carry `activations` (a
    // list); CROWN MNIST carries a single `activation` string.
    let v: serde_json::Value = serde_json::from_str(&body).expect("parse");
    let result = if v.get("activations").is_some() {
        run_external(&path)
    } else {
        let image_idx: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(0);
        let other_class: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1);
        let epsilon: f64 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(0.005);
        run_crown_mnist(&path, image_idx, other_class, epsilon)
    };
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}
