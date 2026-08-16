//! Shared fixtures and helpers for the SNARK test suite. Sibling test
//! modules use only what is re-exported here; they should not reach
//! into private snark internals directly except for the documented
//! hooks below.
//!
//! Helpers fall into three buckets:
//! * Network / property fixtures — small reproducible CROWN problems
//!   fed to `SnarkParams::setup` + `prove_final_pass`.
//! * `fresh_sponge` — a fixed-label Merlin sponge; constructed afresh
//!   per call so prover and verifier never share state.
//! * `Proved` bundle helpers — `prove_small_relu()` and friends return
//!   `Proved { stmt, params, proof }`. `verify_with_fresh_sponge` /
//!   `expect_reject_after_tamper` run the verifier against it.

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_std::test_rng;
use ndarray::{array, Array1, Array2};

use super::super::{
    prove_final_pass, verify_final_pass, SnarkError, SnarkParams, SnarkProof, SnarkStatement,
    VerifiedBound,
};
use crate::crown::network::{Layer, Network};
use crate::crown::output_property::{Property, Side};

/// Merlin sponge with a fixed test label. Constructed afresh on every
/// call so prover and verifier never share state by accident.
pub(super) fn fresh_sponge() -> ark_crypto_primitives::sponge::merlin::Transcript {
    <ark_crypto_primitives::sponge::merlin::Transcript as CryptographicSponge>::new(
        &b"panda-snark-test".as_slice(),
    )
}

// ---- Network / property fixtures. -----------------------------------

/// Two-input × three-hidden × two-output ReLU MLP. The workhorse
/// fixture for every full-pass test — small enough to prove in a
/// fraction of a second.
pub(super) fn small_relu_2x3x2() -> (Network, Property, Array1<f64>, Array1<f64>) {
    let w1: Array2<f64> = array![[1.0, 2.0], [-1.0, 1.0], [0.5, -0.5]];
    let b1: Array1<f64> = array![0.0, 0.5, -0.25];
    let w2: Array2<f64> = array![[1.0, -1.0, 2.0], [0.0, 1.0, 1.0]];
    let b2: Array1<f64> = array![0.1, -0.2];
    let net = Network::new(vec![
        Layer::linear(w1, b1).unwrap(),
        Layer::relu(),
        Layer::linear(w2, b2).unwrap(),
    ])
    .unwrap();
    // Threshold range chosen to fit the in-SNARK property check's
    // `OUT_BOUND_RANGE_BITS = 19` slack budget. Network outputs stay
    // well within ±10.
    let prop = Property::new_with_thresholds(
        Array2::eye(net.output_dim()),
        Array1::zeros(net.output_dim()),
        Side::Both,
        Some(Array1::from_elem(net.output_dim(), -10.0)),
        Some(Array1::from_elem(net.output_dim(), 10.0)),
    )
    .unwrap();
    (net, prop, array![-1.0, -0.5], array![1.0, 0.75])
}

/// Three-input × 4 × 4 × 2 ReLU MLP — exercises a multi-step chain
/// with two ReLU layers.
pub(super) fn deeper_relu_3x4x4x2() -> (Network, Property, Array1<f64>, Array1<f64>) {
    let w1: Array2<f64> = array![
        [0.5, -0.25, 0.75],
        [-0.5, 0.5, -0.5],
        [0.25, 0.25, 0.0],
        [-0.125, -0.5, 0.5]
    ];
    let b1: Array1<f64> = array![0.1, -0.2, 0.05, 0.0];
    let w2: Array2<f64> = array![
        [0.5, 0.25, -0.5, 0.75],
        [-0.5, 0.5, 0.5, 0.0],
        [0.25, -0.25, 0.5, -0.5],
        [0.0, 0.5, 0.5, 0.5]
    ];
    let b2: Array1<f64> = array![0.0, 0.1, -0.1, 0.05];
    let w3: Array2<f64> = array![[1.0, -1.0, 0.5, 0.5], [0.0, 1.0, -1.0, 1.0]];
    let b3: Array1<f64> = array![0.0, 0.0];
    let net = Network::new(vec![
        Layer::linear(w1, b1).unwrap(),
        Layer::relu(),
        Layer::linear(w2, b2).unwrap(),
        Layer::relu(),
        Layer::linear(w3, b3).unwrap(),
    ])
    .unwrap();
    let prop = Property::new_with_thresholds(
        Array2::eye(net.output_dim()),
        Array1::zeros(net.output_dim()),
        Side::Both,
        Some(Array1::from_elem(net.output_dim(), -10.0)),
        Some(Array1::from_elem(net.output_dim(), 10.0)),
    )
    .unwrap();
    (net, prop, array![-0.5, -0.5, -0.5], array![0.5, 0.5, 0.5])
}

/// Five-layer ReLU MLP. Unused at runtime (its roundtrip test is
/// `#[ignore]`d because the CROWN bound diverges past the `2^19` LogUp
/// range table). Kept so the test can be re-enabled when the slack
/// table grows.
pub(super) fn five_layer_relu() -> (Network, Property, Array1<f64>, Array1<f64>) {
    fn mk(rows: usize, cols: usize, layer: u64) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(i, j)| {
            let n = (i as u64).wrapping_mul(13)
                + (j as u64).wrapping_mul(7)
                + layer.wrapping_mul(31)
                + 1;
            let h = n.wrapping_mul(0x9E3779B1) >> 24;
            (h as f64 / 256.0) * 0.1 - 0.05
        })
    }
    let mk_b = |dim: usize| Array1::from_vec(vec![0.01; dim]);
    let w1 = mk(6, 4, 1);
    let w2 = mk(6, 6, 2);
    let w3 = mk(4, 6, 3);
    let w4 = mk(2, 4, 4);
    let net = Network::new(vec![
        Layer::linear(w1, mk_b(6)).unwrap(),
        Layer::relu(),
        Layer::linear(w2, mk_b(6)).unwrap(),
        Layer::relu(),
        Layer::linear(w3, mk_b(4)).unwrap(),
        Layer::relu(),
        Layer::linear(w4, mk_b(2)).unwrap(),
    ])
    .unwrap();
    let prop = Property::new_with_thresholds(
        Array2::eye(net.output_dim()),
        Array1::zeros(net.output_dim()),
        Side::Both,
        Some(Array1::from_elem(net.output_dim(), -10.0)),
        Some(Array1::from_elem(net.output_dim(), 10.0)),
    )
    .unwrap();
    (
        net,
        prop,
        Array1::from_vec(vec![-0.25; 4]),
        Array1::from_vec(vec![0.25; 4]),
    )
}

/// Two-input × three-hidden × two-output sigmoid MLP. Used by Phase 3c
/// tamper coverage at `precision_bits = 12` so that `pick_scale_pow2`
/// lands on `s_w = 2^11 = s_x`.
pub(super) fn small_sigmoid_2x3x2() -> (Network, Property, Array1<f64>, Array1<f64>) {
    let w1: Array2<f64> = array![[0.5, -0.5], [-0.5, 0.25], [0.25, 0.5]];
    let b1: Array1<f64> = array![0.0, 0.1, -0.05];
    let w2: Array2<f64> = array![[1.0, -1.0, 0.5], [-0.5, 0.5, 1.0]];
    let b2: Array1<f64> = array![0.0, 0.05];
    let net = Network::new(vec![
        Layer::linear(w1, b1).unwrap(),
        Layer::sigmoid(),
        Layer::linear(w2, b2).unwrap(),
    ])
    .unwrap();
    let prop = Property::new_with_thresholds(
        Array2::eye(net.output_dim()),
        Array1::zeros(net.output_dim()),
        Side::Both,
        Some(Array1::from_elem(net.output_dim(), -10.0)),
        Some(Array1::from_elem(net.output_dim(), 10.0)),
    )
    .unwrap();
    (net, prop, array![-0.5, -0.5], array![0.5, 0.5])
}

/// Two-input × three-hidden × two-output tanh MLP.
pub(super) fn small_tanh_2x3x2() -> (Network, Property, Array1<f64>, Array1<f64>) {
    let w1: Array2<f64> = array![[0.3, -0.2], [-0.1, 0.4], [0.2, 0.3]];
    let b1: Array1<f64> = array![0.0, 0.05, -0.02];
    let w2: Array2<f64> = array![[1.0, -1.0, 0.5], [-0.5, 0.5, 1.0]];
    let b2: Array1<f64> = array![0.0, 0.05];
    let net = Network::new(vec![
        Layer::linear(w1, b1).unwrap(),
        Layer::tanh(),
        Layer::linear(w2, b2).unwrap(),
    ])
    .unwrap();
    let prop = Property::new_with_thresholds(
        Array2::eye(net.output_dim()),
        Array1::zeros(net.output_dim()),
        Side::Both,
        Some(Array1::from_elem(net.output_dim(), -10.0)),
        Some(Array1::from_elem(net.output_dim(), 10.0)),
    )
    .unwrap();
    (net, prop, array![-0.5, -0.5], array![0.5, 0.5])
}

/// Property with `n_spec = 5` (ceil_log2 = 3, odd) — exercises the
/// even-bumped `native_vector_n_vars` codepath in output_bound. Same
/// tiny network as `small_relu_2x3x2`, with a wider public spec matrix.
pub(super) fn multi_spec_relu_odd_ceil_log2() -> (Network, Property, Array1<f64>, Array1<f64>) {
    let (net, _, x_l, x_u) = small_relu_2x3x2();
    let n_spec = 5usize;
    let out_dim = net.output_dim();
    assert_eq!(out_dim, 2);
    let c_data: Vec<f64> = vec![1.0, -1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
    let c_matrix = Array2::from_shape_vec((n_spec, out_dim), c_data).unwrap();
    let d_vector = Array1::zeros(n_spec);
    let prop = Property::new_with_thresholds(
        c_matrix,
        d_vector,
        Side::Both,
        Some(Array1::from_elem(n_spec, -10.0)),
        Some(Array1::from_elem(n_spec, 10.0)),
    )
    .unwrap();
    (net, prop, x_l, x_u)
}

// ---- Proved-fixture bundle + helpers. -------------------------------

/// Bundle returned by `prove_*` helpers. Tamper tests mutate `proof` in
/// place; missing-component tests set `Option<...>` fields to `None`.
/// Both then call `expect_reject_after_tamper(&p, ...)`.
pub(super) struct Proved {
    pub stmt: SnarkStatement,
    pub params: SnarkParams,
    pub proof: SnarkProof,
}

/// Run the full prover on `(network, property, x_lower, x_upper)` with
/// a fresh sponge and `precision_bits = 14`.
fn prove_with(net: Network, prop: Property, x_lower: Array1<f64>, x_upper: Array1<f64>) -> Proved {
    let mut rng = test_rng();
    let stmt = SnarkStatement {
        network: net,
        property: prop,
        x_lower,
        x_upper,
    };
    let params = SnarkParams::setup(
        &stmt.network,
        &stmt.property,
        14,
        crate::snark::preprocess::test_shared(19, 19, 19),
        &mut rng,
    )
    .unwrap();
    let mut prover_sponge = fresh_sponge();
    let proof = prove_final_pass(&stmt, &params, &mut prover_sponge, &mut rng).unwrap();
    Proved {
        stmt,
        params,
        proof,
    }
}

/// `prove_with` on `small_relu_2x3x2`. The workhorse for every
/// full-pass tamper / missing-component test.
pub(super) fn prove_small_relu() -> Proved {
    let (net, prop, x_l, x_u) = small_relu_2x3x2();
    prove_with(net, prop, x_l, x_u)
}

/// `prove_with` on `deeper_relu_3x4x4x2`.
pub(super) fn prove_deeper_relu() -> Proved {
    let (net, prop, x_l, x_u) = deeper_relu_3x4x4x2();
    prove_with(net, prop, x_l, x_u)
}

/// `prove_with` on `multi_spec_relu_odd_ceil_log2`.
pub(super) fn prove_multi_spec_odd_ceil_log2() -> Proved {
    let (net, prop, x_l, x_u) = multi_spec_relu_odd_ceil_log2();
    prove_with(net, prop, x_l, x_u)
}

/// `prove_with` on `(net, prop, x_l, x_u)` at a configurable precision.
/// Sigmoid/tanh fixtures use `precision_bits = 12` so the cert pipeline
/// picks `s_w = 2^11 = s_x` and the Phase 3c gadget's scale-coupling
/// precondition holds.
fn prove_with_precision(
    net: Network,
    prop: Property,
    x_lower: Array1<f64>,
    x_upper: Array1<f64>,
    precision_bits: i32,
) -> Proved {
    let mut rng = test_rng();
    let stmt = SnarkStatement {
        network: net,
        property: prop,
        x_lower,
        x_upper,
    };
    let params =
        SnarkParams::setup(
        &stmt.network,
        &stmt.property,
        precision_bits,
        crate::snark::preprocess::test_shared(19, 19, 19),
        &mut rng,
    )
    .unwrap();
    let mut prover_sponge = fresh_sponge();
    let proof = prove_final_pass(&stmt, &params, &mut prover_sponge, &mut rng).unwrap();
    Proved {
        stmt,
        params,
        proof,
    }
}

/// Cached honest sigmoid/tanh proofs. Phase 3c prove takes ~30 seconds
/// in release; building it once per cargo-test process and cloning the
/// `Proved` for each tamper test brings the suite from ~10 min to ~1 min.
///
/// Cargo's test framework runs each `#[test]` on its own worker thread
/// (even with `--test-threads=1`), so a `thread_local!` cache would
/// miss every test. A process-wide `Mutex<Option<Proved>>` serializes
/// access across worker threads. Tamper tests must clone the returned
/// `Proved` and mutate the clone — never the cached value itself.
use std::sync::{Mutex, OnceLock};
fn process_cache_sigmoid() -> &'static Mutex<Option<Proved>> {
    static CACHE: OnceLock<Mutex<Option<Proved>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}
fn process_cache_tanh() -> &'static Mutex<Option<Proved>> {
    static CACHE: OnceLock<Mutex<Option<Proved>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

fn cached_or_prove<F: FnOnce() -> Proved>(
    cache: &'static Mutex<Option<Proved>>,
    build: F,
) -> Proved {
    let mut guard = cache.lock().unwrap();
    if guard.is_none() {
        *guard = Some(build());
    }
    let p = guard.as_ref().expect("just initialized");
    Proved {
        stmt: p.stmt.clone(),
        params: p.params.clone(),
        proof: p.proof.clone(),
    }
}

/// `prove_with_precision(small_sigmoid_2x3x2(), 12)`. Cached process-wide.
pub(super) fn prove_small_sigmoid() -> Proved {
    cached_or_prove(process_cache_sigmoid(), || {
        let (net, prop, x_l, x_u) = small_sigmoid_2x3x2();
        prove_with_precision(net, prop, x_l, x_u, 12)
    })
}

/// `prove_with_precision(small_tanh_2x3x2(), 12)`. Cached process-wide.
pub(super) fn prove_small_tanh() -> Proved {
    cached_or_prove(process_cache_tanh(), || {
        let (net, prop, x_l, x_u) = small_tanh_2x3x2();
        prove_with_precision(net, prop, x_l, x_u, 12)
    })
}

/// Run the verifier with a freshly-constructed sponge.
pub(super) fn verify_with_fresh_sponge(p: &Proved) -> Result<VerifiedBound, SnarkError> {
    let mut sponge = fresh_sponge();
    verify_final_pass(&p.stmt.to_verifier(), &p.proof, &p.params, &mut sponge)
}

/// Run the verifier and assert it returns `Err`. Returns the error so
/// callers can `matches!` against a specific variant. `ctx` is the
/// failure description used if the verifier unexpectedly accepts.
pub(super) fn expect_reject_after_tamper(p: &Proved, ctx: &str) -> SnarkError {
    match verify_with_fresh_sponge(p) {
        Err(e) => e,
        Ok(_) => panic!("verifier accepted tampered proof: {ctx}"),
    }
}
