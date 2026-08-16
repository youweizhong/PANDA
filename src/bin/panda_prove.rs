//! Stand-alone PANDA prover: read a fixture JSON, generate a SNARK
//! proof, and write the compressed proof bytes to disk.
//!
//! Usage:
//!     cargo run --release --bin panda_prove -- \
//!         <fixture.json> <out.bin> <quant_params.json>
//!
//! The fixture JSON is the unified PANDA format produced by
//! `evaluation/preprocess/preprocessing.py`. The trailing argument is the path to
//! the quantization parameter JSON file, which contains the table sizes
//! (`table_bits`, `out_bound_range_bits`, etc.). The verifier must be
//! invoked with the same values — it recomputes the lookup tables from
//! them. The evaluation reads them from
//! `evaluation/quant_params/<model>.json`. The output
//! is a binary blob holding the canonical-serialised SNARK proof. Pair
//! with `panda_verify.rs`.

use std::path::Path;
use std::sync::Arc;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_serialize::CanonicalSerialize;
use ark_std::test_rng;
use ndarray::{Array1, Array2};

use panda::crown::network::{Layer, Network};
use panda::crown::output_property::{Property, Side};
use panda::snark::{prove_final_pass, Preprocessed, SnarkParams, SnarkStatement};


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
        let b = Array1::from(fix.biases[i].clone());
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
    let d = Array1::from(fix.spec_d.clone());
    let side = match fix.side.as_str() {
        "lower" => Side::Lower,
        "upper" => Side::Upper,
        "both" => Side::Both,
        other => panic!("unsupported side: {other}"),
    };
    Property::new(c, d, side).expect("property valid")
}

fn fresh_sponge() -> ark_crypto_primitives::sponge::merlin::Transcript {
    <ark_crypto_primitives::sponge::merlin::Transcript as CryptographicSponge>::new(
        &b"panda-bench".as_slice(),
    )
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() != 4 {
        eprintln!("usage: panda_prove <fixture.json> <out.bin> <quant_params.json>");
        std::process::exit(2);
    }
    let fixture_path = Path::new(&argv[1]);
    let out_path = Path::new(&argv[2]);
    let quant_params_path = Path::new(&argv[3]);

    let raw_params = std::fs::read_to_string(quant_params_path)
        .unwrap_or_else(|e| panic!("can't read {}: {e}", quant_params_path.display()));
    let quant_params: panda::file_formats::QuantParams = serde_json::from_str(&raw_params)
        .expect("parse quant_params JSON");

    let range_table_bits = quant_params.table_bits;
    let out_bound_range_bits = quant_params.out_bound_range_bits;
    let gadget_range_bits = quant_params.gadget_range_bits
        .unwrap_or(out_bound_range_bits);

    let raw = std::fs::read_to_string(fixture_path)
        .unwrap_or_else(|e| panic!("can't read {}: {e}", fixture_path.display()));
    let fix: panda::file_formats::Fixture = serde_json::from_str(&raw).expect("parse fixture JSON");

    let stmt = SnarkStatement {
        network: build_network(&fix),
        property: build_property(&fix),
        x_lower: Array1::from(fix.x_lower.clone()),
        x_upper: Array1::from(fix.x_upper.clone()),
    };

    let mut rng = test_rng();
    let (default_sigma_x, default_sigma_v) = panda::default_sigma_scales(fix.precision_bits);
    let sigma_x = quant_params.sigma_x_scale_log2.unwrap_or(default_sigma_x);
    let sigma_v = quant_params.sigma_v_scale_log2.unwrap_or(default_sigma_v);

    let input_scale_log2 = quant_params.input_scale_log2
        .or_else(|| std::env::var("PANDA_INPUT_SCALE_LOG2").ok().and_then(|s| s.trim().parse().ok()));
    let preprocessed: Arc<Preprocessed> = Arc::new(
        Preprocessed::build(
            range_table_bits,
            out_bound_range_bits,
            gadget_range_bits,
            sigma_x,
            sigma_v,
            input_scale_log2,
        )
        .expect("valid runtime table parameters"),
    );
    let params = SnarkParams::setup(
        &stmt.network,
        &stmt.property,
        fix.precision_bits,
        Arc::clone(&preprocessed),
        &mut rng,
    )
    .expect("setup params");

    let mut sponge = fresh_sponge();
    let t0 = std::time::Instant::now();
    let proof = prove_final_pass(&stmt, &params, &mut sponge, &mut rng)
        .expect("prove (the property may not actually hold for this fixture)");
    let elapsed = t0.elapsed().as_secs_f64();

    let mut bytes: Vec<u8> = Vec::new();
    proof
        .serialize_compressed(&mut bytes)
        .expect("serialize proof");
    std::fs::write(out_path, &bytes)
        .unwrap_or_else(|e| panic!("can't write {}: {e}", out_path.display()));

    println!(
        "proved in {:.3}s; wrote {} ({} bytes, {:.2} MB)",
        elapsed,
        out_path.display(),
        bytes.len(),
        bytes.len() as f64 / (1024.0 * 1024.0)
    );
}
