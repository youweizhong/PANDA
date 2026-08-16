//! Stand-alone PANDA verifier: read a fixture JSON and a previously
//! emitted proof, run the SNARK verifier, and print accept/reject plus
//! the certified bound.
//!
//! Usage:
//!     cargo run --release --bin panda_verify -- \
//!         <fixture.json> <proof.bin> <quant_params.json>
//!
//! The trailing argument is the path to the quantization parameter JSON file,
//! which contains the table sizes (`table_bits`, `out_bound_range_bits`, etc.)
//! and must equal the values the prover was invoked with — the verifier
//! recomputes every lookup table from them at runtime and rejects proofs
//! whose claimed table widths disagree.
//!
//! The verifier only consults the public statement (architecture,
//! property, input box, precision and table parameters) plus the proof
//! bytes;
//! the private weights and biases inside the fixture are not read by
//! `verify_final_pass`. The full fixture path is passed only so this
//! example can rebuild the public statement without a separate
//! verifier-only fixture format.

use std::path::Path;
use std::sync::Arc;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_serialize::CanonicalDeserialize;
use ark_std::test_rng;
use ndarray::{Array1, Array2};

use panda::crown::network::{Layer, Network};
use panda::crown::output_property::{Property, Side};
use panda::snark::{verify_final_pass, Preprocessed, SnarkParams, SnarkProof, SnarkStatement};


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
        eprintln!("usage: panda_verify <fixture.json> <proof.bin> <quant_params.json>");
        std::process::exit(2);
    }
    let fixture_path = Path::new(&argv[1]);
    let proof_path = Path::new(&argv[2]);
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
    // Must match the prover's σ scales (default from the fixture's
    // precision); a mismatch would reject at verification.
    let (default_sigma_x, default_sigma_v) = panda::default_sigma_scales(fix.precision_bits);
    let sigma_x = quant_params.sigma_x_scale_log2.unwrap_or(default_sigma_x);
    let sigma_v = quant_params.sigma_v_scale_log2.unwrap_or(default_sigma_v);

    // Must match the prover's input scale too;
    // a mismatch rejects with PublicBindingFailed.
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

    let proof_bytes = std::fs::read(proof_path)
        .unwrap_or_else(|e| panic!("can't read {}: {e}", proof_path.display()));
    let proof = SnarkProof::deserialize_compressed(&proof_bytes[..]).expect("deserialize proof");

    let mut sponge = fresh_sponge();
    let t0 = std::time::Instant::now();
    let result = verify_final_pass(&stmt.to_verifier(), &proof, &params, &mut sponge);
    let elapsed = t0.elapsed().as_secs_f64();

    match result {
        Ok(bound) => {
            println!("verified in {:.3}s — property HOLDS.", elapsed);
            if let Some(lo) = bound.lower {
                println!("  certified lower bound: {:?}", lo.to_vec());
            }
            if let Some(up) = bound.upper {
                println!("  certified upper bound: {:?}", up.to_vec());
            }
            std::process::exit(0);
        }
        Err(e) => {
            println!("verifier rejected after {:.3}s: {e:?}", elapsed);
            std::process::exit(1);
        }
    }
}
