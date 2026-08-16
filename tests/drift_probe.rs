//! Drift probe: measure (float CROWN − quantized CROWN) / float × 100% on a
//! real trained network at varying fixed-point precision, to see (a) how the
//! drift grows with depth/activation and (b) whether raising precision shrinks
//! it. Reads a crown_bin_search GROUP file (weights + biases + activations + items),
//! which is the same trained net the fixed-epsilon evaluation proves.
//!
//! Usage:
//!   cargo test --release --test drift_probe -- <group.json> <eps> [n_items] [p1,p2,...]
//! e.g.
//!   cargo test --release --test drift_probe -- \
//!     evaluation/benchmarks/crown_bin_search/mnist_4layer_tanh_1024.json 0.01 20 14,16,18,20

//! An optional 5th CLI argument pins `sigma_x_scale_log2` (the σ table
//! input scale) for every precision, overriding the per-precision
//! default; `s_v` stays at its default. This makes a clean A/B of the
//! sigmoid/tanh drift at different `s_x` (e.g. the historical 11 vs a
//! larger 13/14) within one binary.

use ndarray::{Array1, Array2};
use serde::Deserialize;

use panda::{
    backward_bound, default_sigma_scales, quantized_backward_bound,
    quantized_backward_bound_scaled, ActivationKind, Layer, Network, Property, Side,
};

#[derive(Deserialize)]
struct Item {
    image_id: i64,
    x0: Vec<f64>,
    spec_c: Vec<Vec<f64>>,
    spec_d: Vec<f64>,
}

#[derive(Deserialize)]
struct Group {
    activations: Vec<String>,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
    clip_lo: f64,
    clip_hi: f64,
    items: Vec<Item>,
}

fn parse_kind(s: &str) -> ActivationKind {
    match s {
        "relu" => ActivationKind::ReLU,
        "sigmoid" => ActivationKind::Sigmoid,
        "tanh" => ActivationKind::Tanh,
        other => panic!("unknown activation {other}"),
    }
}

fn build_network(
    weights: &[Vec<Vec<f64>>],
    biases: &[Vec<f64>],
    activations: &[ActivationKind],
) -> Network {
    let n_linear = weights.len();
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

fn make_box(x0: &[f64], eps: f64, clo: f64, chi: f64) -> (Array1<f64>, Array1<f64>) {
    let lo: Array1<f64> = x0
        .iter()
        .map(|&v| (v - eps).max(clo))
        .collect::<Vec<_>>()
        .into();
    let hi: Array1<f64> = x0
        .iter()
        .map(|&v| (v + eps).min(chi))
        .collect::<Vec<_>>()
        .into();
    (lo, hi)
}

fn property_from_spec(spec_c: &[Vec<f64>], spec_d: &[f64], n_out: usize) -> Property {
    let n_spec = spec_c.len();
    let mut c = Array2::<f64>::zeros((n_spec, n_out));
    for i in 0..n_spec {
        for j in 0..n_out {
            c[[i, j]] = spec_c[i][j];
        }
    }
    let d = Array1::<f64>::from(spec_d.to_vec());
    Property::new(c, d, Side::Lower).unwrap()
}

/// Binding (min) row of a lower-side bound vector.
fn min_bound(v: &Array1<f64>) -> f64 {
    v.iter().cloned().fold(f64::INFINITY, f64::min)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // This target lives under tests/ with `harness = false`, so a bare
    // `cargo test` invokes it with no arguments. Treat that as a no-op
    // success (nothing to probe) rather than a test failure; only a
    // partially-specified invocation is an error.
    if args.len() == 1 {
        println!("drift_probe: no arguments — nothing to probe (skipping).");
        println!("usage: cargo test --release --test drift_probe -- <group.json> <eps> [n_items] [p1,p2,...]");
        return;
    }
    if args.len() < 3 {
        eprintln!("usage: drift_probe <group.json> <eps> [n_items] [p1,p2,...]");
        std::process::exit(2);
    }
    let path = &args[1];
    let eps: f64 = args[2].parse().expect("eps");
    let n_items: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(20);
    let precisions: Vec<i32> = args
        .get(4)
        .map(|s| s.split(',').map(|p| p.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![14, 16, 18, 20]);
    // Optional σ input-scale override. When absent, each precision uses
    // its default s_x (= min(precision, 14)); the ReLU control ignores
    // it entirely (ReLU never touches the σ tables).
    let sigma_x_override: Option<i32> = args.get(5).map(|s| s.parse().expect("s_x_log2"));

    let raw = std::fs::read_to_string(path).expect("read group");
    let g: Group = serde_json::from_str(&raw).expect("parse group");
    let acts: Vec<ActivationKind> = g.activations.iter().map(|s| parse_kind(s)).collect();
    let net = build_network(&g.weights, &g.biases, &acts);
    let n_out = net.output_dim();
    let arch: Vec<&str> = g.activations.iter().map(|s| s.as_str()).collect();
    let sx_desc = match sigma_x_override {
        Some(sx) => format!("s_x=2^{sx} (pinned)"),
        None => "s_x=default per precision (min(p,14))".to_string(),
    };
    println!(
        "net: {} linear layers, activations {:?}, eps={eps}, {} items (of {}), {sx_desc}",
        g.weights.len(),
        arch,
        n_items.min(g.items.len()),
        g.items.len()
    );
    println!(
        "{:>8} {:>12} {}",
        "image",
        "float",
        precisions
            .iter()
            .map(|p| format!("q@{p:<2}   drift%@{p}"))
            .collect::<Vec<_>>()
            .join("  ")
    );

    // Per-precision accumulators over images that float CROWN certifies (>0).
    let mut sum_drift = vec![0.0f64; precisions.len()];
    let mut n_drift = vec![0usize; precisions.len()];
    let mut n_qfail = vec![0usize; precisions.len()];
    let mut n_float_pos = 0usize;

    for it in g.items.iter().take(n_items) {
        let prop = property_from_spec(&it.spec_c, &it.spec_d, n_out);
        let (lo, hi) = make_box(&it.x0, eps, g.clip_lo, g.clip_hi);
        let float_b = match backward_bound(&net, &prop, &lo, &hi) {
            Ok(c) => c.target_lower.as_ref().map(min_bound),
            Err(_) => None,
        };
        let fb = match float_b {
            Some(v) => v,
            None => continue,
        };
        let float_pos = fb > 0.0;
        if float_pos {
            n_float_pos += 1;
        }
        let mut cells = String::new();
        for (k, &p) in precisions.iter().enumerate() {
            let qbound = match sigma_x_override {
                Some(sx) => {
                    // Pin s_x; keep s_v at its default for this precision.
                    let (_, sv) = default_sigma_scales(p);
                    quantized_backward_bound_scaled(&net, &prop, &lo, &hi, p, None, sx, sv)
                }
                None => quantized_backward_bound(&net, &prop, &lo, &hi, p),
            };
            let qb = match qbound {
                Ok(c) => c.final_bound_real().0.as_ref().map(min_bound),
                Err(_) => None,
            };
            match qb {
                Some(q) => {
                    let drift = if fb != 0.0 {
                        (fb - q) / fb * 100.0
                    } else {
                        f64::NAN
                    };
                    cells += &format!(" {q:>8.4} {drift:>9.2}");
                    if float_pos && fb != 0.0 {
                        sum_drift[k] += drift;
                        n_drift[k] += 1;
                    }
                }
                None => {
                    cells += &format!(" {:>8} {:>9}", "QFAIL", "--");
                    if float_pos {
                        n_qfail[k] += 1;
                    }
                }
            }
        }
        println!("{:>8} {:>12.4} {}", it.image_id, fb, cells);
    }

    println!("\n=== average drift over the {n_float_pos} float-certified images ===");
    for (k, &p) in precisions.iter().enumerate() {
        let avg = if n_drift[k] > 0 {
            sum_drift[k] / n_drift[k] as f64
        } else {
            f64::NAN
        };
        println!(
            "precision {p:>2}: avg drift {avg:>8.2}%   (over {} imgs; {} quant failures)",
            n_drift[k], n_qfail[k]
        );
    }
}
