//! Per-property certified L_inf radius under float and QUANTIZED (PANDA)
//! CROWN. Instead of the fixed epsilon baked into a fixture's
//! `x_lower`/`x_upper`, we bisect epsilon to the largest box that still
//! certifies robustness:
//!
//!   1. float CROWN pre-pass -> `r_float`, the largest radius the vanilla
//!      (float64) backward-CROWN lower bound stays > 0. Float CROWN is
//!      ~ms/call, so this brackets the radius cheaply even at big epsilon.
//!   2. quantized CROWN pass -> `r_star`, the largest radius the quantized
//!      (PANDA) backward-CROWN lower bound stays > 0. The quantized bound is
//!      looser (and a `QCrownError` from scale/range overflow counts as "not
//!      certified"), so `r_star <= r_float`; we search only the small window
//!      near `r_float` and, if it does not certify, lower epsilon until it
//!      does. `r_star` is the "epsilon computed by quantized CROWN".
//!
//! The robustness decision is `lower_bound > 0` on every row of a lower-side
//! spec (matches tests/benchmarks.rs).
//!
//! Bounded work: a property CROWN already certifies at `eps_hi` reports the
//! radius as saturated with zero search steps; every search is capped by
//! `*_iters`; and the (slow) quantized search window is capped by
//! `quant_eps_cap`, so certifying at a very large radius never drags the
//! quantized bound into the expensive large-epsilon regime.
//!
//! Two input schemas are auto-detected:
//!
//! 1. **Single external fixture** (`evaluation/benchmarks/**/property_*.json`,
//!    the schema `crown_float_eval` reads): reconstruct the un-clamped center
//!    x0 from `x_lower`/`x_upper` and bisect one property. Search knobs come
//!    from the environment (see below); this is the path
//!    `evaluation.crown_bin_search.runner` drives, one process per fixture.
//!
//! 2. **Grouped batch** (`{ "activations", "weights", "biases",
//!    "precision_bits", "clip_lo", "clip_hi", "eps_hi", "bisect_iters",
//!    "items": [{ "image_id", "x0", "spec_c", "spec_d" }] }`): shares one
//!    network across many items. Optional `<start> <count>` args shard items
//!    across processes.
//!
//! Environment knobs (single-fixture mode):
//!   BISECT_EPS_HI           radius search upper bound        (default 0.5)
//!   BISECT_FLOAT_ITERS      float crown_bin_search step cap         (default 35)
//!   BISECT_ITERS            quantized crown_bin_search step cap     (default 35)
//!   BISECT_QUANT_EPS_CAP    quantized search window cap      (default +inf)
//!   BISECT_PRECISION_BITS   fixed-point bits (overrides the fixture's
//!                           `precision_bits`; REQUIRED when the fixture
//!                           carries none — there is no built-in default)
//!
//! The SNARK-provability check behind `r_prov` runs at ONE fixed pair
//! of range budgets — there is no escalation ladder. The budgets are
//! RUNTIME parameters with no built-in value:
//! PANDA_OUT_BOUND_RANGE_BITS (the final-pass output-margin window) is
//! REQUIRED whenever the quantized pass runs (`bisect_iters > 0`) and
//! comes from the per-model `evaluation/quant_params/<model>.json`;
//! PANDA_GADGET_RANGE_BITS (the per-neuron gadget window) is optional
//! and falls back to the out-bound budget — exactly the historical
//! single-parameter behavior. A cert not provable at those budgets
//! lowers eps via crown_bin_search instead of widening the window
//! (which resolves endpoint-gadget rejections; an output-bound overflow
//! only worsens as eps shrinks, so it conservatively reports r_prov ~ 0);
//! the budgets land in each record's `out_bound_range_bits` /
//! `gadget_range_bits`, and the PANDA proof stage runs exactly those.
//!
//! Run: cargo run --release --bin crown_bin_search -- <input.json>

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use ndarray::{Array1, Array2};
use serde::Deserialize;
use serde_json::Value;

use panda::{
    backward_bound, quantized_backward_bound, quantized_cert_snark_provable, ActivationKind, Layer,
    Network, Property, Side,
};

/// One item in the grouped batch schema (network shared across items).
#[derive(Debug, Deserialize)]
struct Item {
    image_id: i64,
    x0: Vec<f64>,
    spec_c: Vec<Vec<f64>>,
    spec_d: Vec<f64>,
}

/// Grouped batch schema: one network, many centered-ball items.
#[derive(Debug, Deserialize)]
struct Input {
    activations: Vec<String>,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
    precision_bits: i32,
    clip_lo: f64,
    clip_hi: f64,
    eps_hi: f64,
    bisect_iters: u32,
    #[serde(default = "default_float_iters")]
    float_iters: u32,
    #[serde(default = "default_quant_cap")]
    quant_eps_cap: f64,
    items: Vec<Item>,
}

/// Single external fixture: the schema `crown_float_eval` reads, plus optional
/// scalar box metadata (`epsilon`, `clip_lo`, `clip_hi`) written by
/// `evaluation/benchmarks/mnist/generate_least_likely.py`. When those are absent we
/// reconstruct the center and radius from `x_lower`/`x_upper` directly.
#[derive(Debug, Deserialize)]
struct ExternalFixture {
    activations: Vec<String>,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
    x_lower: Vec<f64>,
    x_upper: Vec<f64>,
    spec_c: Vec<Vec<f64>>,
    spec_d: Vec<f64>,
    #[serde(default)]
    side: Option<String>,
    #[serde(default)]
    epsilon: Option<f64>,
    #[serde(default)]
    clip_lo: Option<f64>,
    #[serde(default)]
    clip_hi: Option<f64>,
    #[serde(default)]
    precision_bits: Option<i32>,
    #[serde(default)]
    image_id: Option<i64>,
}

fn default_float_iters() -> u32 {
    35
}

fn default_quant_cap() -> f64 {
    f64::INFINITY
}

/// Bounded search knobs. Shared by both input schemas.
struct SearchParams {
    eps_hi: f64,
    float_iters: u32,
    bisect_iters: u32,
    quant_eps_cap: f64,
    precision_bits: i32,
    /// The output-bound range budget the SNARK-provability check
    /// (`r_prov`) uses for the final-pass output margin — a fixed
    /// runtime parameter, never escalated. From
    /// PANDA_OUT_BOUND_RANGE_BITS; `None` when the environment carries
    /// no budget (allowed only for float-only runs).
    out_bound_range_bits: Option<usize>,
    /// The per-neuron gadget range budget the provability check uses
    /// for the sigmoid/tanh endpoint replicas. From
    /// PANDA_GADGET_RANGE_BITS; `None` falls back to the out-bound
    /// budget (the historical single-parameter behavior).
    gadget_range_bits: Option<usize>,
}

/// The fixed output-bound range budget from the runtime environment
/// (`PANDA_OUT_BOUND_RANGE_BITS`). Returns `None` when unset — valid
/// only for float-only runs (`bisect_iters == 0`); the quantized pass
/// fails loudly on a missing budget instead of inventing a default.
fn out_bound_range_bits_env() -> Option<usize> {
    let bits = env::var("PANDA_OUT_BOUND_RANGE_BITS").ok()?;
    Some(
        bits.parse()
            .expect("PANDA_OUT_BOUND_RANGE_BITS must be a positive integer"),
    )
}

/// The per-neuron gadget range budget from the runtime environment
/// (`PANDA_GADGET_RANGE_BITS`). Returns `None` when unset — the caller
/// falls back to the out-bound budget, which reproduces the historical
/// single-parameter behavior exactly.
fn gadget_range_bits_env() -> Option<usize> {
    let bits = env::var("PANDA_GADGET_RANGE_BITS").ok()?;
    Some(
        bits.parse()
            .expect("PANDA_GADGET_RANGE_BITS must be a positive integer"),
    )
}

fn parse_kind(s: &str) -> ActivationKind {
    match s {
        "relu" => ActivationKind::ReLU,
        "sigmoid" => ActivationKind::Sigmoid,
        "tanh" => ActivationKind::Tanh,
        other => panic!("unsupported activation kind: {other}"),
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
            layers.push(Layer::Activation { kind: activations[i] });
        }
    }
    Network::new(layers).unwrap()
}

fn make_box(x0: &[f64], eps: f64, clip_lo: f64, clip_hi: f64) -> (Array1<f64>, Array1<f64>) {
    let lo: Array1<f64> = x0.iter().map(|&v| (v - eps).max(clip_lo)).collect::<Vec<_>>().into();
    let hi: Array1<f64> = x0.iter().map(|&v| (v + eps).min(clip_hi)).collect::<Vec<_>>().into();
    (lo, hi)
}

/// Float (vanilla) CROWN robustness decision: every spec-row lower bound > 0.
fn certifies_float(net: &Network, prop: &Property, x0: &[f64], eps: f64, clo: f64, chi: f64) -> bool {
    let (lo, hi) = make_box(x0, eps, clo, chi);
    match backward_bound(net, prop, &lo, &hi) {
        Ok(c) => c.target_lower.as_ref().map(|v| v.iter().all(|&x| x > 0.0)).unwrap_or(false),
        Err(_) => false,
    }
}

/// Quantized (PANDA) CROWN robustness decision: every quantized lower bound > 0.
/// A `QCrownError` (scale/range overflow) counts as "not certified", which
/// caps the radius at what the prover can actually discharge.
fn certifies_quant(
    net: &Network, prop: &Property, x0: &[f64], eps: f64, clo: f64, chi: f64, precision_bits: i32,
) -> bool {
    let (lo, hi) = make_box(x0, eps, clo, chi);
    match quantized_backward_bound(net, prop, &lo, &hi, precision_bits) {
        Ok(cert) => cert
            .final_bound_real()
            .0
            .map(|v| v.iter().all(|&x| x > 0.0))
            .unwrap_or(false),
        Err(_) => false,
    }
}

/// Quantized robustness AND SNARK-provability: every quantized lower bound > 0
/// *and* the sigmoid/tanh endpoint gadgets would accept the cert (checked
/// crypto-free via `quantized_cert_snark_provable`). This is the predicate for
/// `r_prov` — the largest radius the PANDA prover will actually discharge.
/// It only *tightens* `certifies_quant` (the endpoint gadget's split-arith
/// `diff` goes out of range on wide boxes even when the bound is > 0), so
/// `r_prov <= r_star`. For ReLU-only nets and narrow boxes it equals `r_star`.
#[allow(clippy::too_many_arguments)]
fn certifies_provable(
    net: &Network, prop: &Property, x0: &[f64], eps: f64, clo: f64, chi: f64, precision_bits: i32,
    out_bound_range_bits: usize, gadget_range_bits: usize,
) -> bool {
    let (lo, hi) = make_box(x0, eps, clo, chi);
    // The provability check re-derives the σ tables; they must match the
    // scales the cert was built at (the default path uses
    // `default_sigma_scales`).
    let (sigma_x, sigma_v) = panda::default_sigma_scales(precision_bits);
    match quantized_backward_bound(net, prop, &lo, &hi, precision_bits) {
        Ok(cert) => {
            let bound_ok = cert
                .final_bound_real()
                .0
                .map(|v| v.iter().all(|&x| x > 0.0))
                .unwrap_or(false);
            bound_ok
                && quantized_cert_snark_provable(
                    net,
                    &cert,
                    out_bound_range_bits,
                    gadget_range_bits,
                    sigma_x,
                    sigma_v,
                )
        }
        Err(_) => false,
    }
}

/// Largest eps in [0, hi] that still certifies, via crown_bin_search. Assumes
/// certify(0)==true and certify(hi)==false (monotone decreasing).
fn bisect<F: Fn(f64) -> bool>(hi: f64, iters: u32, certify: F) -> f64 {
    let (mut lo, mut hi) = (0.0_f64, hi);
    for _ in 0..iters {
        let mid = 0.5 * (lo + hi);
        if certify(mid) { lo = mid } else { hi = mid }
    }
    lo
}

/// Reconstruct the un-clamped center x0 from a clamped `[x_lower, x_upper]`
/// box of radius `eps` over the domain `[clip_lo, clip_hi]`. A coordinate that
/// was not clamped recovers exactly from the midpoint; a coordinate clamped on
/// one side recovers from the un-clamped side (`x0 = hi - eps` when the lower
/// side hit `clip_lo`, `x0 = lo + eps` when the upper side hit `clip_hi`). Both
/// sides clamped only happens when `2*eps >= clip_hi - clip_lo`, far outside
/// the radii used here, and falls back to the midpoint.
fn reconstruct_center(lo: &[f64], hi: &[f64], clip_lo: f64, clip_hi: f64, eps: f64) -> Vec<f64> {
    let tol = 1e-9;
    lo.iter()
        .zip(hi.iter())
        .map(|(&l, &h)| {
            if l <= clip_lo + tol && h < clip_hi - tol {
                h - eps
            } else if h >= clip_hi - tol && l > clip_lo + tol {
                l + eps
            } else {
                0.5 * (l + h)
            }
        })
        .collect()
}

/// Full float+quantized radius search for one property. Returns the JSON
/// record `run_crown_bin_search` parses (`r_star`, `r_float`, saturation flags).
fn bisect_one(
    net: &Network,
    prop: &Property,
    x0: &[f64],
    clip_lo: f64,
    clip_hi: f64,
    image_id: i64,
    p: &SearchParams,
) -> Value {
    let (clo, chi) = (clip_lo, clip_hi);

    // Fast float pre-pass: certified float radius r_float over [0, eps_hi].
    let f0 = certifies_float(net, prop, x0, 0.0, clo, chi);
    let (r_float, f_sat) = if !f0 {
        (0.0, false)
    } else if certifies_float(net, prop, x0, p.eps_hi, clo, chi) {
        (p.eps_hi, true) // CROWN certifies even at eps_hi: no crown_bin_search needed.
    } else {
        (
            bisect(p.eps_hi, p.float_iters, |e| certifies_float(net, prop, x0, e, clo, chi)),
            false,
        )
    };

    // bisect_iters == 0 => float-only mode (skip the expensive quantized pass),
    // used to profile the radius scale before committing.
    if p.bisect_iters == 0 {
        return serde_json::json!({
            "image_id": image_id, "r_star": -1.0, "r_float": r_float, "r_prov": -1.0,
            "certified_at_zero": false, "float_certified_at_zero": f0,
            "saturated": false, "float_saturated": f_sat,
        });
    }

    // Quantized radius: search only the small-eps window [0, ~r_float]. The
    // quantized bound is looser, so r_quant <= r_float (+/- tiny drift); a 15%
    // margin above r_float brackets it while keeping every (expensive)
    // quantized eval at a small, fast epsilon. quant_eps_cap bounds the window
    // when r_float itself saturated at eps_hi.
    let q_hi = (r_float * 1.15).min(p.eps_hi).min(p.quant_eps_cap);
    // "Certified at zero" must NOT be probed at eps = 0.0 exactly: a radius
    // below one code unit quantizes to l == u, which forces the sigmoid/tanh
    // relaxation into its degenerate d = 0 envelope fallback — pathologically
    // loose for deep wide tanh nets (their fixed-eps sweeps certify fine at
    // eps = 0.005 while the zero-width probe reports a negative bound, which
    // used to short-circuit r_star to 0). Probe at two code units instead:
    // the smallest radius with a non-degenerate quantized box.
    let eps0 = (2.0f64).powi(1 - p.precision_bits);
    let q0 = certifies_quant(net, prop, x0, eps0, clo, chi, p.precision_bits);
    let (r_quant, q_sat) = if !q0 || q_hi <= 0.0 {
        // Quantized fails even at a point, or there is no float radius to
        // bracket: certified radius is zero.
        (0.0, false)
    } else if certifies_quant(net, prop, x0, q_hi, clo, chi, p.precision_bits) {
        (q_hi, true) // saturated against the r_float-derived / capped window
    } else {
        (
            bisect(q_hi, p.bisect_iters, |e| {
                certifies_quant(net, prop, x0, e, clo, chi, p.precision_bits)
            }),
            false,
        )
    };

    // r_prov: largest eps <= r_quant the SNARK will actually PROVE at the
    // ONE fixed range budget — never widened. No proof is run — each eval
    // is a quantized CROWN pass plus the crypto-free range checks (both
    // ~ms), and the budget is recorded so the PANDA proof stage runs
    // exactly it, first try. The eps crown_bin_search targets the sigmoid/tanh
    // endpoint-gadget rejection, which relaxes as the box narrows; an
    // output-bound window OVERFLOW is anti-monotone (the margin code
    // GROWS as eps shrinks), so when overflow is the blocker the search
    // conservatively lands at r_prov ~ 0 rather than finding a provable
    // middle band — with no ladder, the honest answer is "not provable
    // at this budget", not a wider window.
    let range_bits = p.out_bound_range_bits.unwrap_or_else(|| {
        panic!(
            "the quantized SNARK-provability pass needs an output-bound \
             budget: set PANDA_OUT_BOUND_RANGE_BITS (from the model's \
             evaluation/quant_params JSON)"
        )
    });
    let gadget_bits = p.gadget_range_bits.unwrap_or(range_bits);
    let provable = |e: f64| {
        certifies_provable(
            net,
            prop,
            x0,
            e,
            clo,
            chi,
            p.precision_bits,
            range_bits,
            gadget_bits,
        )
    };
    let (r_prov, prov_at_star) = if r_quant <= 0.0 {
        (0.0, false)
    } else if provable(r_quant) {
        (r_quant, true)
    } else {
        (bisect(r_quant, p.bisect_iters, provable), false)
    };

    serde_json::json!({
        "image_id": image_id,
        "r_star": r_quant,
        "r_prov": r_prov,
        "r_float": r_float,
        "certified_at_zero": q0,
        "float_certified_at_zero": f0,
        "provable_at_r_star": prov_at_star,
        "out_bound_range_bits": range_bits,
        "gadget_range_bits": gadget_bits,
        "saturated": q_sat,
        "float_saturated": f_sat,
    })
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

fn env_f64(key: &str, default: f64) -> f64 {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_u32(key: &str, default: u32) -> u32 {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_i32(key: &str, default: i32) -> i32 {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Single-fixture mode: reconstruct x0 and bisect one property. Prints one
/// record with the radii, saturation flags, and the search knobs used.
fn run_external(fix: ExternalFixture) -> Value {
    if let Some(side) = fix.side.as_deref() {
        if side != "lower" {
            panic!("radius search supports lower-side robustness specs only (got side={side:?})");
        }
    }
    let acts: Vec<ActivationKind> = fix.activations.iter().map(|s| parse_kind(s)).collect();
    let net = build_network(&fix.weights, &fix.biases, &acts);
    let n_out = net.output_dim();

    let clip_lo = fix.clip_lo.unwrap_or(-0.5);
    let clip_hi = fix.clip_hi.unwrap_or(0.5);
    // Un-clamped coordinates have width exactly 2*eps; clamped ones are
    // narrower, so max(width)/2 recovers the fixed radius when it is not stored.
    let eps = fix.epsilon.unwrap_or_else(|| {
        let mut w = 0.0_f64;
        for (l, h) in fix.x_lower.iter().zip(fix.x_upper.iter()) {
            w = w.max(h - l);
        }
        0.5 * w
    });
    let x0 = reconstruct_center(&fix.x_lower, &fix.x_upper, clip_lo, clip_hi, eps);

    let p = SearchParams {
        eps_hi: env_f64("BISECT_EPS_HI", 0.5),
        float_iters: env_u32("BISECT_FLOAT_ITERS", 35),
        bisect_iters: env_u32("BISECT_ITERS", 35),
        quant_eps_cap: env_f64("BISECT_QUANT_EPS_CAP", f64::INFINITY),
        precision_bits: env::var("BISECT_PRECISION_BITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .or(fix.precision_bits)
            .expect(
                "precision_bits: fixture carries none and BISECT_PRECISION_BITS \
                 is unset — supply it from the model's quant_params JSON",
            ),
        out_bound_range_bits: out_bound_range_bits_env(),
        gadget_range_bits: gadget_range_bits_env(),
    };

    let prop = property_from_spec(&fix.spec_c, &fix.spec_d, n_out);
    let image_id = fix.image_id.unwrap_or(-1);

    let t0 = Instant::now();
    let mut rec = bisect_one(&net, &prop, &x0, clip_lo, clip_hi, image_id, &p);
    let elapsed = t0.elapsed().as_secs_f64();
    let obj = rec.as_object_mut().unwrap();
    obj.insert("schema".into(), Value::String("external".into()));
    obj.insert("eps_hi".into(), serde_json::json!(p.eps_hi));
    obj.insert("float_iters".into(), serde_json::json!(p.float_iters));
    obj.insert("bisect_iters".into(), serde_json::json!(p.bisect_iters));
    obj.insert("quant_eps_cap".into(), serde_json::json!(p.quant_eps_cap));
    obj.insert("precision_bits".into(), serde_json::json!(p.precision_bits));
    obj.insert("fixed_epsilon".into(), serde_json::json!(eps));
    obj.insert("runtime_secs".into(), serde_json::json!(elapsed));
    rec
}

/// Grouped-batch mode: one shared network, many centered-ball items.
fn run_grouped(inp: Input, start: usize, count: usize) -> Value {
    let end = (start + count).min(inp.items.len());
    let acts: Vec<ActivationKind> = inp.activations.iter().map(|s| parse_kind(s)).collect();
    let net = build_network(&inp.weights, &inp.biases, &acts);
    // The search knobs are baked into the input file; env vars override them so
    // the sweep can be re-tuned without regenerating the (large) input.
    let p = SearchParams {
        eps_hi: env_f64("BISECT_EPS_HI", inp.eps_hi),
        float_iters: env_u32("BISECT_FLOAT_ITERS", inp.float_iters),
        bisect_iters: env_u32("BISECT_ITERS", inp.bisect_iters),
        quant_eps_cap: env_f64("BISECT_QUANT_EPS_CAP", inp.quant_eps_cap),
        precision_bits: env_i32("BISECT_PRECISION_BITS", inp.precision_bits),
        out_bound_range_bits: out_bound_range_bits_env(),
        gadget_range_bits: gadget_range_bits_env(),
    };
    let n_out = net.output_dim();

    let t0 = Instant::now();
    let mut out = Vec::with_capacity(end - start);
    for it in &inp.items[start..end] {
        let prop = property_from_spec(&it.spec_c, &it.spec_d, n_out);
        out.push(bisect_one(&net, &prop, &it.x0, inp.clip_lo, inp.clip_hi, it.image_id, &p));
    }
    let elapsed = t0.elapsed().as_secs_f64();
    serde_json::json!({
        "precision_bits": inp.precision_bits,
        "out_bound_range_bits": p.out_bound_range_bits,
        "gadget_range_bits": p.gadget_range_bits,
        "eps_hi": inp.eps_hi,
        "bisect_iters": inp.bisect_iters,
        "n_items": out.len(),
        "start": start,
        "end": end,
        "runtime_secs": elapsed,
        "radii": out,
    })
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: crown_bin_search <input.json> [start] [count]");
        std::process::exit(1);
    }
    let path = PathBuf::from(&args[1]);
    let body = fs::read_to_string(&path).expect("read input");
    // Schema detection: the grouped batch carries an `items` array; a single
    // external fixture does not (it carries a top-level `x_lower`).
    let v: Value = serde_json::from_str(&body).expect("parse input");
    let result = if v.get("items").is_some() {
        let inp: Input = serde_json::from_value(v).expect("parse grouped input");
        let start: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
        let count: usize = args
            .get(3)
            .and_then(|s| s.parse().ok())
            .unwrap_or(inp.items.len().saturating_sub(start));
        run_grouped(inp, start, count)
    } else {
        let fix: ExternalFixture = serde_json::from_value(v).expect("parse external fixture");
        run_external(fix)
    };
    println!("{}", serde_json::to_string(&result).unwrap());
}
