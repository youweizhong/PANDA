//! Integration harness that runs prove + verify on one generated
//! unified-schema fixture under `evaluation/benchmarks/**/*.json`.
//! Fixtures are generated locally (see evaluation/README.md) — none
//! are committed.
//!
//! Each fixture carries the full benchmark spec inline (network
//! weights/biases, input box, property `(spec_c, spec_d, side)`,
//! `precision_bits`, plus human-readable metadata). For 1-row
//! Lower-side properties the robustness verdict is computed inline
//! from the verified bound.
//!
//! The heavy `benchmark_fixture_from_env` test is `#[ignore]`d by
//! default; select one fixture via the `PANDA_BENCHMARK_FIXTURE`
//! env var. The SNARK table sizes are RUNTIME parameters and must be
//! supplied explicitly — there are no defaults:
//!
//! * `PANDA_RANGE_TABLE_BITS` — signed range / ReLU table half-width
//! * `PANDA_OUT_BOUND_RANGE_BITS` — output-bound range budget
//!
//! The evaluation runner (`evaluation/run_panda.py`) reads both
//! from the per-model `evaluation/quant_params/*.json` file. Example:
//!
//! ```text
//! PANDA_BENCHMARK_FIXTURE=evaluation/benchmarks/.../foo.json \
//! PANDA_RANGE_TABLE_BITS=19 PANDA_OUT_BOUND_RANGE_BITS=19 \
//!     cargo test --release --test benchmarks \
//!     benchmark_fixture_from_env -- --ignored --exact --nocapture
//! ```

use ark_serialize::CanonicalSerialize;
use ark_std::test_rng;
use ndarray::{Array1, Array2};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use panda::crown::network::{Layer, Network};
use panda::crown::output_property::{Property, Side};
use panda::quantized_backward_bound_scaled;
use panda::snark::{
    prove_final_pass, verify_final_pass, Preprocessed, SnarkParams, SnarkStatement,
};



fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("evaluation")
        .join("benchmarks")
}

fn load_path(path: &Path) -> panda::file_formats::Fixture {
    let s = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("can't read {}: {e}", path.display()));
    serde_json::from_str(&s).expect("parse unified fixture")
}

fn discover_json_files(dir: &Path, out: &mut Vec<PathBuf>) {
    if !dir.exists() {
        return;
    }
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("can't read benchmark dir {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("read_dir entry").path();
        if path.is_dir() {
            discover_json_files(&path, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some("json") {
            out.push(path);
        }
    }
}

fn looks_like_fixture(path: &Path) -> bool {
    let Ok(s) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) else {
        return false;
    };
    [
        "activations",
        "weights",
        "biases",
        "x_lower",
        "x_upper",
        "spec_c",
        "spec_d",
        "side",
    ]
    .iter()
    .all(|k| v.get(*k).is_some())
}

fn discover_fixtures() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    discover_json_files(&fixture_dir(), &mut paths);
    paths.retain(|p| looks_like_fixture(p));
    paths.sort();
    paths
}

fn validate(fix: &panda::file_formats::Fixture) {
    let n_linear = fix.weights.len();
    assert!(
        n_linear >= 1,
        "fixture: weights must have at least one layer"
    );
    assert_eq!(
        fix.weights.len(),
        fix.biases.len(),
        "fixture: weights/biases length mismatch"
    );
    assert_eq!(
        fix.activations.len(),
        n_linear - 1,
        "fixture: activations.len() != weights.len() - 1"
    );
    for (i, w) in fix.weights.iter().enumerate() {
        assert!(!w.is_empty(), "fixture: weights[{i}] is empty");
        let cols = w[0].len();
        assert!(cols > 0, "fixture: weights[{i}][0] is empty");
        for (r, row) in w.iter().enumerate() {
            assert_eq!(
                row.len(),
                cols,
                "fixture: weights[{i}][{r}] not rectangular"
            );
        }
        assert_eq!(
            fix.biases[i].len(),
            w.len(),
            "fixture: biases[{i}].len() != weights[{i}].nrows()"
        );
    }
    assert_eq!(
        fix.weights[0][0].len(),
        fix.input_dim.unwrap_or(fix.x_lower.len()),
        "fixture: weights[0].ncols != input_dim"
    );
    let last = fix.weights.len() - 1;
    assert_eq!(
        fix.weights[last].len(),
        fix.output_dim.unwrap_or_else(|| fix.weights.last().map_or(0, |w| w.len())),
        "fixture: weights[last].nrows != output_dim"
    );
    let input_dim = fix.input_dim.unwrap_or(fix.x_lower.len());
    let output_dim = fix.output_dim.unwrap_or_else(|| fix.weights.last().map_or(0, |w| w.len()));

    assert_eq!(fix.x_lower.len(), input_dim);
    assert_eq!(fix.x_upper.len(), input_dim);
    assert!(!fix.spec_c.is_empty());
    let spec_cols = fix.spec_c[0].len();
    assert_eq!(
        spec_cols, output_dim,
        "fixture: spec_c[0].len != output_dim"
    );
    for (i, row) in fix.spec_c.iter().enumerate() {
        assert_eq!(row.len(), spec_cols, "fixture: spec_c[{i}] not rectangular");
    }
    assert_eq!(
        fix.spec_d.len(),
        fix.spec_c.len(),
        "fixture: spec_d.len != spec_c.nrows"
    );
}

fn build_network(fix: &panda::file_formats::Fixture) -> Network {
    let n_linear = fix.weights.len();
    let mut layers = Vec::with_capacity(2 * n_linear - 1);
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
            layers.push(match fix.activations[i].as_str() {
                "relu" => Layer::relu(),
                "sigmoid" => Layer::sigmoid(),
                "tanh" => Layer::tanh(),
                other => panic!("unsupported activation: {other}"),
            });
        }
    }
    Network::new(layers).unwrap()
}

fn build_property(fix: &panda::file_formats::Fixture) -> Property {
    let n_spec = fix.spec_c.len();
    let n_out = fix.spec_c[0].len();
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
        other => panic!("unsupported side: {other}"),
    };
    Property::new(c, d, side).expect("property valid")
}

fn fresh_sponge() -> ark_crypto_primitives::sponge::merlin::Transcript {
    use ark_crypto_primitives::sponge::CryptographicSponge;
    <ark_crypto_primitives::sponge::merlin::Transcript as CryptographicSponge>::new(
        &b"panda-bench".as_slice(),
    )
}

fn artifact_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn artifact_run_dir(name: &str) -> Option<PathBuf> {
    std::env::var_os("PANDA_BENCHMARK_ARTIFACT_DIR")
        .map(PathBuf::from)
        .map(|root| root.join(artifact_name(name)))
}

fn write_artifact_json(dir: Option<&Path>, file_name: &str, value: serde_json::Value) {
    let Some(dir) = dir else {
        return;
    };
    std::fs::create_dir_all(dir)
        .unwrap_or_else(|e| panic!("can't create artifact dir {}: {e}", dir.display()));
    let path = dir.join(file_name);
    std::fs::write(&path, serde_json::to_string_pretty(&value).unwrap())
        .unwrap_or_else(|e| panic!("can't write artifact {}: {e}", path.display()));
}

fn initialize_artifacts(dir: Option<&Path>, fixture_path: &Path) {
    let Some(dir) = dir else {
        return;
    };
    std::fs::create_dir_all(dir)
        .unwrap_or_else(|e| panic!("can't create artifact dir {}: {e}", dir.display()));
    std::fs::copy(fixture_path, dir.join("witness_fixture.json"))
        .unwrap_or_else(|e| panic!("can't copy fixture into {}: {e}", dir.display()));
    std::fs::write(
        dir.join("README.md"),
        "This directory was generated by `tests/benchmarks.rs`.\n\n\
         - `witness_fixture.json` is the PANDA-ready benchmark fixture used by the prover. \
           It contains the private model weights/biases plus the public input box and property.\n\
         - `proof.bin` is the compressed canonical SNARK proof, present only when proving succeeds.\n\
         - `result.json` records timing, proof size, and accept/reject status.\n\
         - `stdout.log` is written by `evaluation/run_panda.py` when that runner is used.\n\n\
         The current harness does not dump every internal prover intermediate tensor; those witnesses \
         are consumed inside `prove_final_pass` and can be regenerated from the fixture.\n",
    )
    .unwrap_or_else(|e| panic!("can't write artifact README in {}: {e}", dir.display()));
}

/// Read a required runtime parameter from the environment. No default:
/// proving at a silently-wrong table size would be worse than failing.
fn required_env_usize(key: &str) -> usize {
    let raw = std::env::var(key).unwrap_or_else(|_| {
        panic!(
            "{key} must be set (runtime SNARK table parameter; the \
             evaluation reads it from evaluation/quant_params/<model>.json)"
        )
    });
    raw.parse::<usize>()
        .unwrap_or_else(|_| panic!("{key}={raw:?} is not a positive integer"))
}

#[allow(clippy::too_many_arguments)] // mirrors the quant_params set: one arg per public parameter
fn run_path(
    path: &Path,
    name: &str,
    range_table_bits: i32,
    out_bound_range_bits: usize,
    gadget_range_bits: usize,
    sigma_x_scale_override: Option<i32>,
    sigma_v_scale_override: Option<i32>,
    input_scale_override: Option<i32>,
) {
    println!("=== {name} ===");
    println!("fixture: {}", path.display());
    println!(
        "table parameters: range_table_bits={range_table_bits} \
         out_bound_range_bits={out_bound_range_bits} \
         gadget_range_bits={gadget_range_bits}"
    );
    let artifact_dir = artifact_run_dir(name);
    initialize_artifacts(artifact_dir.as_deref(), path);
    let fix = load_path(path);
    validate(&fix);

    // Sigmoid/tanh table scales: runtime public parameters. Absent env
    // overrides fall back to the default derived from the fixture's
    // precision (a single source of truth in `default_sigma_scales`).
    let (default_sigma_x, default_sigma_v) = panda::default_sigma_scales(fix.precision_bits);
    let sigma_x_scale_log2 = sigma_x_scale_override.unwrap_or(default_sigma_x);
    let sigma_v_scale_log2 = sigma_v_scale_override.unwrap_or(default_sigma_v);
    println!(
        "sigma scales: sigma_x_scale_log2={sigma_x_scale_log2} \
         sigma_v_scale_log2={sigma_v_scale_log2}"
    );
    // Input-box quantization scale: an optional runtime public parameter.
    // Absent = the harness keeps the default `pick_scale_pow2` auto-scale.
    // The runner's stale-harness guard matches on this echoed line.
    match input_scale_override {
        Some(e) => println!("input scale: input_scale_log2={e}"),
        None => println!("input scale: input_scale_log2=auto (pick_scale_pow2)"),
    }
    if let Some(desc) = fix.description.as_deref() {
        println!("description: {desc}");
    }
    if let Some(arch) = fix.architecture.as_deref() {
        println!("architecture: {arch}");
    }
    if let Some(prop) = fix.property_description.as_deref() {
        println!("property: {prop}");
    }
    println!(
        "input_dim={}, output_dim={}, n_linear={}, activations={:?}, n_spec={}, side={}, h={}",
        fix.input_dim.unwrap_or(fix.x_lower.len()),
        fix.output_dim.unwrap_or_else(|| fix.weights.last().map_or(0, |w| w.len())),
        fix.weights.len(),
        fix.activations,
        fix.spec_c.len(),
        fix.side,
        fix.precision_bits,
    );

    let net = build_network(&fix);
    let property = build_property(&fix);
    let x_lower: Array1<f64> = fix.x_lower.clone().into();
    let x_upper: Array1<f64> = fix.x_upper.clone().into();
    let stmt = SnarkStatement {
        network: net,
        property,
        x_lower,
        x_upper,
    };

    // Quantized (PANDA) CROWN output bound at the fixture box — the same
    // bound the SNARK proves, recomputed crypto-free (~ms) so it is
    // recorded independently of whether the proof is later discharged.
    // The verifier keeps the bound private, so this is the only place
    // it enters the result record; the report diffs it against the
    // float-CROWN baseline's `float_lower_bound` / `float_upper_bound`.
    let (quant_lower_bound, quant_upper_bound) = match quantized_backward_bound_scaled(
        &stmt.network,
        &stmt.property,
        &stmt.x_lower,
        &stmt.x_upper,
        fix.precision_bits,
        input_scale_override,
        sigma_x_scale_log2,
        sigma_v_scale_log2,
    ) {
        Ok(cert) => cert.final_bound_real(),
        Err(e) => {
            println!("quantized bound: unavailable ({e:?})");
            (None, None)
        }
    };
    let bound_json = |v: &Option<Array1<f64>>| -> Option<String> {
        v.as_ref().map(|arr| {
            let items: Vec<String> = arr.iter().map(|x| format!("{x}")).collect();
            format!("[{}]", items.join(", "))
        })
    };
    if let Some(js) = bound_json(&quant_lower_bound) {
        println!("quantized_lower_bound_json: {js}");
    }
    if let Some(js) = bound_json(&quant_upper_bound) {
        println!("quantized_upper_bound_json: {js}");
    }

    let mut rng = test_rng();
    let preprocess_t0 = Instant::now();
    let preprocessed: Arc<Preprocessed> = Arc::new(
        Preprocessed::build(
            range_table_bits,
            out_bound_range_bits,
            gadget_range_bits,
            sigma_x_scale_log2,
            sigma_v_scale_log2,
            input_scale_override,
        )
        .expect("valid runtime table parameters"),
    );
    let preprocess_secs = preprocess_t0.elapsed().as_secs_f64();
    let setup_t0 = Instant::now();
    let params = SnarkParams::setup(
        &stmt.network,
        &stmt.property,
        fix.precision_bits,
        Arc::clone(&preprocessed),
        &mut rng,
    )
    .unwrap();
    let setup_secs = setup_t0.elapsed().as_secs_f64();
    println!(
        "preprocessing: tables={:.3}s + Hyrax_key(max_num_vars={})={:.3}s",
        preprocess_secs, params.max_num_vars, setup_secs
    );

    let mut prover_sponge = fresh_sponge();
    panda::timing::reset();
    let prove_t0 = Instant::now();
    let prove_res = prove_final_pass(&stmt, &params, &mut prover_sponge, &mut rng);
    let prove_secs = prove_t0.elapsed().as_secs_f64();
    println!("prover components v1: {}", panda::timing::snapshot_json());
    let proof = match prove_res {
        Ok(p) => p,
        Err(e) => {
            // Prove-time reject = "property does not hold" (or scale
            // precondition fails); recorded as a controlled outcome.
            println!(
                "online prove:    {:.3}s (rejected at prove time: {:?})",
                prove_secs, e
            );
            println!(
                "⇒ NOT robust: prover rejected — property does not hold (or scale precondition)."
            );
            write_artifact_json(
                artifact_dir.as_deref(),
                "result.json",
                serde_json::json!({
                    "name": name,
                    "fixture": path.display().to_string(),
                    "status": "prove_rejected",
                    "range_table_bits": range_table_bits,
                    "out_bound_range_bits": out_bound_range_bits,
                    "gadget_range_bits": gadget_range_bits,
                    "sigma_x_scale_log2": sigma_x_scale_log2,
                    "sigma_v_scale_log2": sigma_v_scale_log2,
                    "input_scale_log2": input_scale_override,
                    "prove_secs": prove_secs,
                    "error": format!("{:?}", e),
                    "property_verified": false,
                }),
            );
            return;
        }
    };
    println!("online prove:    {:.3}s", prove_secs);

    let mut proof_bytes: Vec<u8> = Vec::new();
    proof.serialize_compressed(&mut proof_bytes).unwrap();
    println!(
        "proof size:      {} bytes ({:.2} KB / {:.3} MB)",
        proof_bytes.len(),
        proof_bytes.len() as f64 / 1024.0,
        proof_bytes.len() as f64 / (1024.0 * 1024.0),
    );
    if let Some(dir) = artifact_dir.as_deref() {
        std::fs::create_dir_all(dir)
            .unwrap_or_else(|e| panic!("can't create artifact dir {}: {e}", dir.display()));
        std::fs::write(dir.join("proof.bin"), &proof_bytes)
            .unwrap_or_else(|e| panic!("can't write proof artifact in {}: {e}", dir.display()));
    }

    let mut verifier_sponge = fresh_sponge();
    let verify_t0 = Instant::now();
    let verify_res = verify_final_pass(&stmt.to_verifier(), &proof, &params, &mut verifier_sponge);
    let verify_secs = verify_t0.elapsed().as_secs_f64();
    let bound = match verify_res {
        Ok(b) => {
            println!("online verify:   {:.3}s", verify_secs);
            println!(
                "ratio (prove/verify): {:.1}×",
                prove_secs / verify_secs.max(1e-9)
            );
            b
        }
        Err(e) => {
            // Verifier reject = the property verdict is false.
            println!("online verify:   {:.3}s (rejected: {:?})", verify_secs, e);
            println!("⇒ NOT robust: verifier rejected — property does not hold.");
            write_artifact_json(
                artifact_dir.as_deref(),
                "result.json",
                serde_json::json!({
                    "name": name,
                    "fixture": path.display().to_string(),
                    "status": "verify_rejected",
                    "range_table_bits": range_table_bits,
                    "out_bound_range_bits": out_bound_range_bits,
                    "gadget_range_bits": gadget_range_bits,
                    "sigma_x_scale_log2": sigma_x_scale_log2,
                    "sigma_v_scale_log2": sigma_v_scale_log2,
                    "input_scale_log2": input_scale_override,
                    "prove_secs": prove_secs,
                    "verify_secs": verify_secs,
                    "proof_bytes": proof_bytes.len(),
                    "error": format!("{:?}", e),
                    "property_verified": false,
                }),
            );
            return;
        }
    };

    // A successful verify means the in-SNARK property check held;
    // the verifier no longer returns a dequantized bound, so empty
    // lower/upper just means "property discharged".
    if bound.lower.is_none() && bound.upper.is_none() {
        println!("  ⇒ ROBUST: property discharged (in-SNARK property check passed).");
    }
    if let Some(lo) = bound.lower.as_ref() {
        let preview: Vec<String> = lo.iter().take(8).map(|v| format!("{v:+.4}")).collect();
        let suffix = if lo.len() > 8 { ", ..." } else { "" };
        println!("verified lower bound: [{}{suffix}]", preview.join(", "));
        // 1-row Lower-side benchmarks: robustness iff lower bound > 0.
        if fix.spec_c.len() == 1 && fix.side == "lower" {
            if lo[0] > 0.0 {
                println!("  ⇒ ROBUST: property discharged.");
            } else {
                println!("  ⇒ NOT robust: lower bound is non-positive.");
            }
        }
    }
    if let Some(up) = bound.upper.as_ref() {
        let preview: Vec<String> = up.iter().take(8).map(|v| format!("{v:+.4}")).collect();
        let suffix = if up.len() > 8 { ", ..." } else { "" };
        println!("verified upper bound: [{}{suffix}]", preview.join(", "));
    }
    write_artifact_json(
        artifact_dir.as_deref(),
        "result.json",
        serde_json::json!({
            "name": name,
            "fixture": path.display().to_string(),
            "status": "verified",
            "range_table_bits": range_table_bits,
            "out_bound_range_bits": out_bound_range_bits,
            "gadget_range_bits": gadget_range_bits,
            "sigma_x_scale_log2": sigma_x_scale_log2,
            "sigma_v_scale_log2": sigma_v_scale_log2,
            "input_scale_log2": input_scale_override,
            "prove_secs": prove_secs,
            "verify_secs": verify_secs,
            "proof_bytes": proof_bytes.len(),
            "proof_kb": proof_bytes.len() as f64 / 1024.0,
            "proof_mb": proof_bytes.len() as f64 / (1024.0 * 1024.0),
            "property_verified": true,
            "verified_bound_public": bound.lower.is_some() || bound.upper.is_some(),
        }),
    );
}

#[test]
fn generated_benchmark_fixtures_are_well_formed() {
    let paths = discover_fixtures();
    if paths.is_empty() {
        eprintln!(
            "no generated PANDA fixtures under {}; skipping schema check",
            fixture_dir().display()
        );
        return;
    }
    for path in paths {
        let fix = load_path(&path);
        validate(&fix);
    }
}

#[test]
#[ignore = "heavy: prove+verify one fixture selected by PANDA_BENCHMARK_FIXTURE"]
fn benchmark_fixture_from_env() {
    let fixture = std::env::var_os("PANDA_BENCHMARK_FIXTURE")
        .expect("set PANDA_BENCHMARK_FIXTURE to an evaluation/benchmarks/**/*.json file");
    let path = PathBuf::from(fixture);
    let name = std::env::var("PANDA_BENCHMARK_NAME")
        .ok()
        .or_else(|| path.file_stem().and_then(|s| s.to_str()).map(str::to_owned))
        .unwrap_or_else(|| path.display().to_string());
    let range_table_bits = required_env_usize("PANDA_RANGE_TABLE_BITS") as i32;
    let out_bound_range_bits = required_env_usize("PANDA_OUT_BOUND_RANGE_BITS");
    // Per-neuron gadget budget; unset reproduces the historical
    // single-parameter behavior (gadget == out-bound). The runner
    // cross-checks the echoed value, so an old harness paired with a
    // split-budget parameter set fails loudly instead of mislabeling.
    let gadget_range_bits = match std::env::var("PANDA_GADGET_RANGE_BITS") {
        Ok(raw) => raw.parse::<usize>().unwrap_or_else(|_| {
            panic!("PANDA_GADGET_RANGE_BITS={raw:?} is not a positive integer")
        }),
        Err(_) => out_bound_range_bits,
    };
    // Sigmoid/tanh table scales; unset falls back to the default derived
    // from the fixture's precision inside `run_path`. The runner
    // (`run_panda.py`) always sets both from the quant_params set.
    let sigma_x_scale_override = optional_env_i32("PANDA_SIGMA_X_SCALE_LOG2");
    let sigma_v_scale_override = optional_env_i32("PANDA_SIGMA_V_SCALE_LOG2");
    let input_scale_override = optional_env_i32("PANDA_INPUT_SCALE_LOG2");
    run_path(
        &path,
        &name,
        range_table_bits,
        out_bound_range_bits,
        gadget_range_bits,
        sigma_x_scale_override,
        sigma_v_scale_override,
        input_scale_override,
    );
}

/// Parse an optional signed-integer env var; `None` when unset.
fn optional_env_i32(key: &str) -> Option<i32> {
    match std::env::var(key) {
        Ok(raw) => Some(
            raw.parse::<i32>()
                .unwrap_or_else(|_| panic!("{key}={raw:?} is not an integer")),
        ),
        Err(_) => None,
    }
}
