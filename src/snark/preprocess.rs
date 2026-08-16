//! Public lookup tables shared by the prover and verifier.
//!
//! Table sizes are RUNTIME parameters: [`Preprocessed::build`] takes the
//! signed-range-table half-width and the two unsigned range budgets —
//! `out_bound_range_bits` (the output-stage slack/property LogUps, used
//! once per proof at the final pass; 21 covers the very-robust models
//! whose output margins overflow 19 bits) and `gadget_range_bits`
//! (every per-neuron gadget range check: ReLU upper-endpoint,
//! sigmoid/tanh endpoint and critical-point split arithmetic, and the
//! hidden-pass preact-bound inequalities; 19 keeps those tables small)
//! — as explicit arguments, so the verifier can recompute the exact
//! same tables from the public parameters at verification time — no
//! compile-time table-size agreement (and no cargo feature) is needed.
//! Prover and verifier must simply agree on the three bit values, which
//! travel out-of-band as public parameters (the evaluation reads them
//! from `evaluation/quant_params/<model>.json`).
//!
//! The sigmoid/tanh envelope tables are different: their real-unit
//! domain is fixed (independent of the range-table width), but their
//! `(s_x, s_v)` scales are RUNTIME per-model PUBLIC parameters — the
//! input scale `s_x = 2^sigma_x_scale_log2` sets the table resolution
//! (and hence the sigmoid/tanh output-bound drift floor); the value
//! scale `s_v = 2^sigma_v_scale_log2` sets the σ-code magnitude. Both
//! travel with `Preprocessed`/`SnarkParams` alongside the range
//! budgets, the quantizer must use byte-identical envelopes to the
//! SNARK gadgets, and the verifier recomputes the tables from the same
//! public scales. Tables live in [`SigmaTables`], built once per
//! `(s_x_log2, s_v_log2)` pair and shared via [`SigmaTables::shared`].

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use ark_bn254::Fr;

use crate::snark_primitives::finite_field::signed_lift_to_fr;

use crate::snark::errors::SnarkError;

/// Cost-model cap on `sigma_x_scale_log2`: the σ/tanh half-tables hold
/// `128·2^sigma_x_scale_log2 = 2^(7 + sigma_x_scale_log2)` entries, so
/// larger `s_x` grows the table, the σ-envelope LogUp, and proof size.
/// Capping at 14 keeps `7 + 14 = 21 ≤ MAX_TABLE_BITS` with room to
/// spare. Used only by [`default_sigma_scales`] and the scale validator.
pub const SIGMA_X_SCALE_LOG2_MAX: i32 = 14;
/// Cost-model cap on `sigma_v_scale_log2`: `s_v` sets the σ-code
/// magnitude that flows into the sshape gadget split-arith. Capping at
/// 18 keeps `line_sigma_code ≈ s_v` inside a per-neuron gadget range
/// table wide enough for the checked-in budgets. Used only by
/// [`default_sigma_scales`] and the scale validator.
pub const SIGMA_V_SCALE_LOG2_MAX: i32 = 18;

/// The single source of truth for the default sigmoid/tanh table scales
/// as a function of `precision_bits`. Both optional per-set JSON keys
/// (`sigma_x_scale_log2`, `sigma_v_scale_log2`) fall back to these
/// formulas when absent. Bigger `s_x` tightens the sigmoid/tanh
/// output-bound drift (it is the lever) but grows the table; `s_v`
/// barely moves tanh drift and only inflates gadget magnitudes, so it
/// stays modest. Every literal scale cap lives here (and in
/// [`SIGMA_X_SCALE_LOG2_MAX`] / [`SIGMA_V_SCALE_LOG2_MAX`]) — nowhere
/// in gadget/quantizer/verifier logic.
///
/// **`s_x` interacts with `gadget_range_bits`.** The sshape endpoint
/// gadget indexes the σ half-table (which spans `2^(7 + s_x)` entries)
/// and range-checks that index against `2^gadget_range_bits`. So a
/// sigmoid/tanh preactivation of real magnitude `|x|` is provable only
/// when `|x|·2^s_x < 2^gadget_range_bits`, i.e. `|x| < 2^(gadget_range_bits
/// − s_x)`. At the default `s_x = 14` the full `|x| < 128 = 2^7` domain
/// needs `gadget_range_bits ≥ 21`; a narrower gadget budget shrinks the
/// provable preact domain (soundness-safe — an out-of-budget preact
/// records "unknown", never a false accept). Pair a larger `s_x` with a
/// gadget budget of at least `7 + s_x` for full sigmoid/tanh coverage.
pub fn default_sigma_scales(precision_bits: i32) -> (i32, i32) {
    (
        precision_bits.min(SIGMA_X_SCALE_LOG2_MAX),
        (precision_bits + 2).min(SIGMA_V_SCALE_LOG2_MAX),
    )
}

/// Half-table outer bound for sigmoid: covers real x ∈ [0, 128). Negative
/// inputs are recovered via the symmetry `σ(-x) = 1 − σ(x)`.
pub const SIGMOID_TABLE_X_BOUND_REAL: i64 = 128;
/// Beyond this bound the sigmoid table stores the saturation envelope
/// `(s_v - 1, s_v)` instead of a computed value. `σ(32)` differs from 1
/// by less than 1 LSB at `s_v = 2^16`.
pub const SIGMOID_NATURAL_BOUND_REAL: i64 = 32;
/// Half-table outer bound for tanh. Matches sigmoid for table-size parity.
pub const TANH_TABLE_X_BOUND_REAL: i64 = 128;
/// Beyond this bound the tanh table stores saturation. `tanh(16)`
/// differs from 1 by less than 1 LSB at `s_v = 2^16`.
pub const TANH_NATURAL_BOUND_REAL: i64 = 16;

/// Resource ceiling for either runtime table-size parameter. A table of
/// `2^(26+1)` Fr entries already costs ~4 GiB to materialize; anything
/// larger is a misconfiguration, and `1usize << bits` must stay far from
/// overflow. This is a sanity guard on resource use, not a tuning
/// default — the actual per-model values live in the evaluation's
/// quantization-parameter JSONs.
pub const MAX_TABLE_BITS: u32 = 26;

/// Saturation envelope used by the gadget when a sigmoid input is past
/// the natural-domain bound on the right (x ≫ 0), at value scale
/// `s_v = 2^s_v_log2`.
pub fn sigmoid_sat_right_lower(s_v_log2: i32) -> i128 {
    (1i128 << s_v_log2) - 1
}
pub fn sigmoid_sat_right_upper(s_v_log2: i32) -> i128 {
    1i128 << s_v_log2
}
// The left-saturation (x ≪ 0) envelopes below document the gadget's
// negative-domain contract, but the in-circuit lookup recovers them from
// the right half-table via the σ symmetries rather than calling these,
// so they are exercised only by the conservativeness tests.
/// Saturation envelope on the left (x ≪ 0). Sigmoid is near zero there.
#[allow(dead_code)]
pub fn sigmoid_sat_left_lower(_s_v_log2: i32) -> i128 {
    0
}
#[allow(dead_code)]
pub fn sigmoid_sat_left_upper(_s_v_log2: i32) -> i128 {
    1
}
/// Saturation envelope for tanh on the right (x ≫ 0). Tanh is near +1.
pub fn tanh_sat_right_lower(s_v_log2: i32) -> i128 {
    (1i128 << s_v_log2) - 1
}
pub fn tanh_sat_right_upper(s_v_log2: i32) -> i128 {
    1i128 << s_v_log2
}
/// Saturation envelope for tanh on the left (x ≪ 0). Tanh is near -1.
#[allow(dead_code)]
pub fn tanh_sat_left_lower(s_v_log2: i32) -> i128 {
    -(1i128 << s_v_log2)
}
#[allow(dead_code)]
pub fn tanh_sat_left_upper(s_v_log2: i32) -> i128 {
    -((1i128 << s_v_log2) - 1)
}

/// Sigmoid/tanh envelope half-tables built for one `(s_x_log2,
/// s_v_log2)` scale pair. The real-unit domain is fixed but the scales
/// are runtime public parameters, so instances are built and cached per
/// scale pair via [`SigmaTables::shared`]. The quantized-CROWN cert
/// generator and the SNARK gadgets must both read the instance built at
/// the proof's public scales (byte-identical envelopes).
#[derive(Clone)]
pub struct SigmaTables {
    /// Log2 of the input scale these tables were built at
    /// (`s_x = 2^s_x_log2`); the table index of real `x` is
    /// `round(x · s_x)`.
    pub s_x_log2: i32,
    /// Log2 of the value scale these tables were built at
    /// (`s_v = 2^s_v_log2`); envelope codes are `σ(x) · s_v`.
    pub s_v_log2: i32,
    /// Sigmoid half-table x-coordinates for `x_int ∈ [0, 128·s_x)`.
    /// Negative inputs are recovered in-circuit via the symmetry
    /// `σ(-x) = 1 − σ(x)`.
    pub sigmoid_x_fr: Vec<Fr>,
    /// Conservative lower envelope of σ on the half-table; saturates to
    /// `s_v - 1` past the natural-domain bound.
    pub sigmoid_lower_fr: Vec<Fr>,
    /// Conservative upper envelope of σ on the half-table; saturates to
    /// `s_v` past the natural-domain bound.
    pub sigmoid_upper_fr: Vec<Fr>,
    /// Tanh half-table x-coordinates. Negative inputs use the odd
    /// symmetry `tanh(-x) = −tanh(x)`.
    pub tanh_x_fr: Vec<Fr>,
    /// Conservative lower envelope of tanh on the half-table.
    pub tanh_lower_fr: Vec<Fr>,
    /// Conservative upper envelope of tanh on the half-table.
    pub tanh_upper_fr: Vec<Fr>,
}

impl SigmaTables {
    /// Allocate and populate the σ/tanh envelope tables at the given
    /// runtime scales `s_x = 2^s_x_log2`, `s_v = 2^s_v_log2`.
    ///
    /// Panics if `7 + s_x_log2 > MAX_TABLE_BITS` (the half-table would
    /// exceed the materialization ceiling): building an oversized table
    /// is a catastrophic misconfiguration, so it fails loudly here
    /// rather than attempting a multi-GiB allocation. Callers that take
    /// scales from untrusted input should validate first via
    /// [`validate_sigma_scales`].
    pub fn build(s_x_log2: i32, s_v_log2: i32) -> Self {
        assert!(
            s_x_log2 >= 0 && (7 + s_x_log2) as u32 <= MAX_TABLE_BITS,
            "sigma_x_scale_log2 = {s_x_log2}: 7 + s_x_log2 must be in [7, MAX_TABLE_BITS]"
        );
        assert!(s_v_log2 >= 0, "sigma_v_scale_log2 = {s_v_log2} must be >= 0");
        let (sigmoid_x_fr, sigmoid_lower_fr, sigmoid_upper_fr) =
            build_sigmoid_value_table(s_x_log2, s_v_log2);
        let (tanh_x_fr, tanh_lower_fr, tanh_upper_fr) = build_tanh_value_table(s_x_log2, s_v_log2);
        Self {
            s_x_log2,
            s_v_log2,
            sigmoid_x_fr,
            sigmoid_lower_fr,
            sigmoid_upper_fr,
            tanh_x_fr,
            tanh_lower_fr,
            tanh_upper_fr,
        }
    }

    /// Returns the cached instance for `(s_x_log2, s_v_log2)`, building
    /// it on first request for that scale pair. Subsequent calls at the
    /// same scales only clone the `Arc`, so prover, verifier, and the
    /// cert generator that agree on the public scales share one table.
    pub fn shared(s_x_log2: i32, s_v_log2: i32) -> Arc<Self> {
        let mut cache = SIGMA_CACHE.lock().unwrap();
        cache
            .entry((s_x_log2, s_v_log2))
            .or_insert_with(|| Arc::new(SigmaTables::build(s_x_log2, s_v_log2)))
            .clone()
    }
}

/// Process-wide cache of σ/tanh tables keyed by `(s_x_log2, s_v_log2)`,
/// so each distinct scale pair is built once per process.
static SIGMA_CACHE: LazyLock<Mutex<HashMap<(i32, i32), Arc<SigmaTables>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Sigmoid/tanh table scales the test suite pins by default. These
/// reproduce the historical hard-coded constants (`s_x = 2^11`,
/// `s_v = 2^16`), so every existing test builds byte-identical σ tables
/// and its accept/reject verdict is unchanged by the scales becoming
/// runtime parameters. Tests that exercise non-default scales use
/// [`test_shared_sigma`].
#[cfg(test)]
pub(crate) const TEST_SIGMA_X_SCALE_LOG2: i32 = 11;
#[cfg(test)]
pub(crate) const TEST_SIGMA_V_SCALE_LOG2: i32 = 16;

/// Test-only cache of `Preprocessed` instances keyed by their runtime
/// bit parameters (range/out-bound/gadget budgets), at the default test
/// σ scales. Production callers build (and share via `Arc`) their own
/// instance from the runtime parameters.
#[cfg(test)]
pub(crate) fn test_shared(
    range_table_half_bits: i32,
    out_bound_range_bits: usize,
    gadget_range_bits: usize,
) -> Arc<Preprocessed> {
    test_shared_sigma(
        range_table_half_bits,
        out_bound_range_bits,
        gadget_range_bits,
        TEST_SIGMA_X_SCALE_LOG2,
        TEST_SIGMA_V_SCALE_LOG2,
    )
}

/// Like [`test_shared`] but at explicit σ scales, so a test can prove
/// and verify at non-default `(s_x_log2, s_v_log2)`. Keyed by the full
/// five-tuple so each distinct configuration is built once.
#[cfg(test)]
pub(crate) fn test_shared_sigma(
    range_table_half_bits: i32,
    out_bound_range_bits: usize,
    gadget_range_bits: usize,
    sigma_x_scale_log2: i32,
    sigma_v_scale_log2: i32,
) -> Arc<Preprocessed> {
    static CACHE: LazyLock<Mutex<HashMap<(i32, usize, usize, i32, i32), Arc<Preprocessed>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    let mut cache = CACHE.lock().unwrap();
    cache
        .entry((
            range_table_half_bits,
            out_bound_range_bits,
            gadget_range_bits,
            sigma_x_scale_log2,
            sigma_v_scale_log2,
        ))
        .or_insert_with(|| {
            Arc::new(
                Preprocessed::build(
                    range_table_half_bits,
                    out_bound_range_bits,
                    gadget_range_bits,
                    sigma_x_scale_log2,
                    sigma_v_scale_log2,
                    None,
                )
                .expect("test_shared: valid table bits"),
            )
        })
        .clone()
}

/// Pre-built lookup tables for one `(range_table_half_bits,
/// out_bound_range_bits, gadget_range_bits)` parameter triple, shared
/// by the prover and verifier of proofs at those parameters.
///
/// All entries live in the BN254 scalar field `Fr`. The size-dependent
/// tables (signed range, positive out-bound range, ReLU coordinates)
/// are built from the runtime parameters; the σ/tanh envelopes are the
/// process-shared [`SigmaTables`].
#[derive(Clone)]
pub struct Preprocessed {
    /// Half-width of the signed range / ReLU tables: they cover
    /// `[-2^k, 2^k)` with `k = range_table_half_bits`. Every
    /// public-witness tensor (input-box codes, weights, biases,
    /// relaxation coefficients, hidden-layer preact codes) must fit in
    /// this range. Public parameter; prover and verifier must agree.
    pub range_table_half_bits: i32,
    /// Output-bound range budget: the FINAL-pass slack and property
    /// LogUps (the `c·y − d` output-margin checks, once per proof) use
    /// `[0, 2^out_bound_range_bits)`. Public parameter; prover and
    /// verifier must agree.
    pub out_bound_range_bits: usize,
    /// Per-neuron gadget range budget: every activation-gadget range
    /// check (ReLU upper-endpoint slack/epsilon, sigmoid/tanh endpoint
    /// and critical-point split arithmetic, chunk moduli, scale
    /// preconditions) and the hidden-pass preact-bound inequalities use
    /// `[0, 2^gadget_range_bits)`. Public parameter; prover and
    /// verifier must agree.
    pub gadget_range_bits: usize,
    /// Signed range table `{ -2^k, ..., 2^k - 1 }`, lifted to `Fr`.
    pub range_table_fr: Vec<Fr>,
    /// Unsigned range table `{ 0, ..., 2^out_bound_range_bits - 1 }`,
    /// used ONLY by the final-pass output-bound slack/property LogUps;
    /// select with [`Preprocessed::pos_range_table`].
    pub out_bound_pos_range_table: Vec<Fr>,
    /// Unsigned range table `{ 0, ..., 2^gadget_range_bits - 1 }`, used
    /// by every per-neuron gadget range LogUp and the hidden-pass
    /// inequalities; select with [`Preprocessed::pos_range_table`].
    /// Shares the out-bound table's allocation when the two budgets
    /// coincide.
    pub gadget_pos_range_table: Vec<Fr>,
    /// `x`-coordinates of the ReLU lookup table. Combined with a
    /// Fiat-Shamir challenge `α` at runtime via
    /// [`Preprocessed::relu_table_at`].
    pub relu_table_x_fr: Vec<Fr>,
    /// `ReLU(x)`-coordinates matching `relu_table_x_fr` index-for-index.
    pub relu_table_relu_fr: Vec<Fr>,
    /// Log2 of the sigmoid/tanh input scale (`s_x = 2^sigma_x_scale_log2`)
    /// these σ tables were built at. Public parameter; prover and
    /// verifier must agree, and the quantizer must build the cert at the
    /// same scale.
    pub sigma_x_scale_log2: i32,
    /// Log2 of the sigmoid/tanh value scale (`s_v = 2^sigma_v_scale_log2`).
    /// Public parameter; prover and verifier must agree.
    pub sigma_v_scale_log2: i32,
    /// σ/tanh envelope tables built for `(sigma_x_scale_log2,
    /// sigma_v_scale_log2)` and shared across instances at the same
    /// scales.
    pub sigma: Arc<SigmaTables>,
    /// Log2 of the input-box quantization scale (`input = 2^input_scale_log2`),
    /// or `None` to keep the default `pick_scale_pow2(x_box, precision_bits)`.
    /// Sizes NO lookup table (unlike `sigma_x_scale_log2`), but travels
    /// with the public params so the verifier's `derive_public_scales`
    /// recomputes the same input scale. Public parameter; prover and
    /// verifier must agree.
    pub input_scale_log2: Option<i32>,
}

/// Reject a runtime table-size parameter outside `[2, MAX_TABLE_BITS]`.
fn validate_table_bits(bits: u32, what: &'static str) -> Result<(), SnarkError> {
    if !(2..=MAX_TABLE_BITS).contains(&bits) {
        return Err(SnarkError::InvalidParameter { what });
    }
    Ok(())
}

/// Reject sigmoid/tanh table scales outside their valid ranges. `s_x`
/// is bounded by the cost cap [`SIGMA_X_SCALE_LOG2_MAX`] (which keeps
/// `7 + s_x ≤ MAX_TABLE_BITS`); `s_v` by [`SIGMA_V_SCALE_LOG2_MAX`].
/// Public parameters, so the verifier applies the identical bounds.
pub fn validate_sigma_scales(
    sigma_x_scale_log2: i32,
    sigma_v_scale_log2: i32,
) -> Result<(), SnarkError> {
    if !(1..=SIGMA_X_SCALE_LOG2_MAX).contains(&sigma_x_scale_log2) {
        return Err(SnarkError::InvalidParameter {
            what: "sigma_x_scale_log2 outside [1, SIGMA_X_SCALE_LOG2_MAX]",
        });
    }
    if !(1..=SIGMA_V_SCALE_LOG2_MAX).contains(&sigma_v_scale_log2) {
        return Err(SnarkError::InvalidParameter {
            what: "sigma_v_scale_log2 outside [1, SIGMA_V_SCALE_LOG2_MAX]",
        });
    }
    Ok(())
}

/// Reject an input-scale override that cannot fit the signed range table.
/// A magnitude-1.0 input at scale `2^e` produces code `2^e`, which must
/// lie strictly inside the range table `[-2^k, 2^k)` used for the x-box
/// LogUp (`k = range_table_half_bits`): `2^e < 2^k  ⟺  e ≤ k − 1`.
/// (Inputs with magnitude > 1.0 would overflow at a smaller `e` and are
/// still caught soundly — as a proof rejection, never a false accept —
/// by the x-box range LogUp at proof time; this is a friendly early
/// guard checking only the necessary unit-input condition.) `None`
/// disables the override and is always accepted. Public parameter, so
/// the verifier applies the identical bound.
pub fn validate_input_scale(
    input_scale_log2: Option<i32>,
    range_table_half_bits: i32,
) -> Result<(), SnarkError> {
    if let Some(e) = input_scale_log2 {
        if e < 1 || e > range_table_half_bits - 1 {
            return Err(SnarkError::InvalidParameter {
                what: "input_scale_log2 outside [1, range_table_half_bits - 1]",
            });
        }
    }
    Ok(())
}

impl Preprocessed {
    /// Allocate and populate the tables for one runtime parameter pair.
    ///
    /// Rejects bit widths outside `[2, MAX_TABLE_BITS]` (resource
    /// guard). Both the prover and the verifier construct this from the
    /// same public parameters, so the verifier recomputes the tables at
    /// runtime rather than trusting anything proof-embedded.
    pub fn build(
        range_table_half_bits: i32,
        out_bound_range_bits: usize,
        gadget_range_bits: usize,
        sigma_x_scale_log2: i32,
        sigma_v_scale_log2: i32,
        input_scale_log2: Option<i32>,
    ) -> Result<Self, SnarkError> {
        validate_table_bits(
            u32::try_from(range_table_half_bits).unwrap_or(u32::MAX),
            "range_table_half_bits outside [2, MAX_TABLE_BITS]",
        )?;
        validate_table_bits(
            u32::try_from(out_bound_range_bits).unwrap_or(u32::MAX),
            "out_bound_range_bits outside [2, MAX_TABLE_BITS]",
        )?;
        validate_table_bits(
            u32::try_from(gadget_range_bits).unwrap_or(u32::MAX),
            "gadget_range_bits outside [2, MAX_TABLE_BITS]",
        )?;
        validate_sigma_scales(sigma_x_scale_log2, sigma_v_scale_log2)?;
        validate_input_scale(input_scale_log2, range_table_half_bits)?;
        let range_table_fr = build_signed_range_table(range_table_half_bits);
        let out_bound_pos_range_table = build_pos_range_table(out_bound_range_bits);
        let gadget_pos_range_table = if gadget_range_bits == out_bound_range_bits {
            out_bound_pos_range_table.clone()
        } else {
            build_pos_range_table(gadget_range_bits)
        };
        let (relu_table_x_fr, relu_table_relu_fr) = build_relu_table_coords(range_table_half_bits);
        Ok(Self {
            range_table_half_bits,
            out_bound_range_bits,
            gadget_range_bits,
            range_table_fr,
            out_bound_pos_range_table,
            gadget_pos_range_table,
            relu_table_x_fr,
            relu_table_relu_fr,
            sigma_x_scale_log2,
            sigma_v_scale_log2,
            sigma: SigmaTables::shared(sigma_x_scale_log2, sigma_v_scale_log2),
            input_scale_log2,
        })
    }

    /// The positive range table `{ 0, ..., 2^bits - 1 }`. `bits` must
    /// equal one of the two budgets this instance was built for — the
    /// table choice is soundness-relevant, so a mismatched request is
    /// an error, not a rebuild. The out-bound budget is matched first;
    /// when the two budgets coincide the tables are identical anyway.
    pub fn pos_range_table(&self, bits: usize) -> Result<&[Fr], SnarkError> {
        if bits == self.out_bound_range_bits {
            Ok(self.out_bound_pos_range_table.as_slice())
        } else if bits == self.gadget_range_bits {
            Ok(self.gadget_pos_range_table.as_slice())
        } else {
            Err(SnarkError::InvalidParameter {
                what: "requested range bits match neither out_bound_range_bits \
                       nor gadget_range_bits",
            })
        }
    }

    /// Returns the per-step ReLU LogUp table `α · x + ReLU(x)`, combining
    /// the two pre-lifted coordinate vectors against the FS challenge `α`.
    pub fn relu_table_at(&self, alpha: Fr) -> Vec<Fr> {
        debug_assert_eq!(self.relu_table_x_fr.len(), self.relu_table_relu_fr.len());
        self.relu_table_x_fr
            .iter()
            .zip(self.relu_table_relu_fr.iter())
            .map(|(x, r)| alpha * x + r)
            .collect()
    }
}

/// Builds the signed range table `{ -2^k, ..., 2^k - 1 }` and lifts every
/// entry into `Fr`.
fn build_signed_range_table(half_bits: i32) -> Vec<Fr> {
    let lo: i128 = -(1i128 << half_bits);
    let hi: i128 = 1i128 << half_bits;
    let len = (hi - lo) as usize;
    let mut t = Vec::with_capacity(len);
    for v in lo..hi {
        t.push(signed_lift_to_fr(v));
    }
    t
}

/// Builds the unsigned range table `{ 0, ..., 2^bits - 1 }` as `Fr`.
fn build_pos_range_table(bits: usize) -> Vec<Fr> {
    let len = 1usize << bits;
    (0..len).map(|v| Fr::from(v as u64)).collect()
}

/// Builds the sigmoid half-table covering `x_int ∈ [0, 128·s_x)`.
///
/// The inner region `[0, 32·s_x)` stores the conservative envelope
/// `[floor(σ·s_v) − 1, ceil(σ·s_v) + 1]` clamped to `[0, s_v]`.
/// The outer region holds the saturation envelope `(s_v − 1, s_v)`.
fn build_sigmoid_value_table(s_x_log2: i32, s_v_log2: i32) -> (Vec<Fr>, Vec<Fr>, Vec<Fr>) {
    let s_x = (1i128 << s_x_log2) as f64;
    let s_v_int = 1i128 << s_v_log2;
    let s_v = s_v_int as f64;
    let bound = SIGMOID_TABLE_X_BOUND_REAL * (1i64 << s_x_log2);
    let natural_bound = SIGMOID_NATURAL_BOUND_REAL * (1i64 << s_x_log2);
    let hi = bound as i128;
    let nat_hi = natural_bound as i128;
    let len = hi as usize;
    let mut x = Vec::with_capacity(len);
    let mut sl = Vec::with_capacity(len);
    let mut su = Vec::with_capacity(len);
    let sat_right_lo = signed_lift_to_fr(sigmoid_sat_right_lower(s_v_log2));
    let sat_right_up = signed_lift_to_fr(sigmoid_sat_right_upper(s_v_log2));
    for v in 0..hi {
        x.push(signed_lift_to_fr(v));
        if v < nat_hi {
            let xr = (v as f64) / s_x;
            let s = (1.0 / (1.0 + (-xr).exp())) * s_v;
            let lower = ((s.floor() as i128).saturating_sub(1)).max(0);
            let upper = ((s.ceil() as i128).saturating_add(1)).min(s_v_int);
            sl.push(signed_lift_to_fr(lower));
            su.push(signed_lift_to_fr(upper));
        } else {
            sl.push(sat_right_lo);
            su.push(sat_right_up);
        }
    }
    (x, sl, su)
}

/// Builds the tanh half-table over `x_int ∈ [0, 128·s_x)`.
///
/// Inner region `[0, 16·s_x)` stores the codomain-clamped envelope of
/// `tanh(x_int / s_x) · s_v`. Outer region stores the saturation envelope.
fn build_tanh_value_table(s_x_log2: i32, s_v_log2: i32) -> (Vec<Fr>, Vec<Fr>, Vec<Fr>) {
    let s_x = (1i128 << s_x_log2) as f64;
    let s_v_int = 1i128 << s_v_log2;
    let s_v = s_v_int as f64;
    let bound = TANH_TABLE_X_BOUND_REAL * (1i64 << s_x_log2);
    let natural_bound = TANH_NATURAL_BOUND_REAL * (1i64 << s_x_log2);
    let hi = bound as i128;
    let nat_hi = natural_bound as i128;
    let len = hi as usize;
    let mut x = Vec::with_capacity(len);
    let mut tl = Vec::with_capacity(len);
    let mut tu = Vec::with_capacity(len);
    let sat_right_lo = signed_lift_to_fr(tanh_sat_right_lower(s_v_log2));
    let sat_right_up = signed_lift_to_fr(tanh_sat_right_upper(s_v_log2));
    for v in 0..hi {
        x.push(signed_lift_to_fr(v));
        if v < nat_hi {
            let xr = (v as f64) / s_x;
            let t = xr.tanh() * s_v;
            let lower = ((t.floor() as i128).saturating_sub(1)).max(0);
            let upper = ((t.ceil() as i128).saturating_add(1)).min(s_v_int);
            tl.push(signed_lift_to_fr(lower));
            tu.push(signed_lift_to_fr(upper));
        } else {
            tl.push(sat_right_lo);
            tu.push(sat_right_up);
        }
    }
    (x, tl, tu)
}

/// Builds the `(x, ReLU(x))` coordinate pair over `{ -2^k, ..., 2^k - 1 }`.
fn build_relu_table_coords(half_bits: i32) -> (Vec<Fr>, Vec<Fr>) {
    let lo: i128 = -(1i128 << half_bits);
    let hi: i128 = 1i128 << half_bits;
    let len = (hi - lo) as usize;
    let mut x = Vec::with_capacity(len);
    let mut r = Vec::with_capacity(len);
    for v in lo..hi {
        x.push(signed_lift_to_fr(v));
        r.push(signed_lift_to_fr(v.max(0)));
    }
    (x, r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snark_primitives::finite_field::fr_to_signed_i128;

    /// Test-local table sizing; the evaluation reads real values from
    /// the per-model quantization-parameter JSONs.
    const TEST_HALF_BITS: i32 = 19;
    const TEST_OB_BITS: usize = 19;
    const TEST_GADGET_BITS: usize = 19;
    /// Default σ scales for these tests: the historical hard-coded pair.
    const TEST_SX: i32 = TEST_SIGMA_X_SCALE_LOG2;
    const TEST_SV: i32 = TEST_SIGMA_V_SCALE_LOG2;

    /// Build a `Preprocessed` at the default test σ scales.
    fn pb(
        half: i32,
        ob: usize,
        gadget: usize,
    ) -> Result<Preprocessed, SnarkError> {
        Preprocessed::build(half, ob, gadget, TEST_SX, TEST_SV, None)
    }

    #[test]
    fn build_sizes_follow_runtime_parameters() {
        // (19, 19, 21) pins gadget > out_bound: the two budgets are
        // fully independent runtime parameters — neither ordering is
        // assumed anywhere, so both split directions must build and
        // resolve their tables.
        for (half, ob, gadget) in [
            (19i32, 19usize, 19usize),
            (21, 21, 19),
            (18, 20, 17),
            (19, 19, 21),
        ] {
            let pre = pb(half, ob, gadget).unwrap();
            assert_eq!(pre.range_table_fr.len(), 1usize << (half + 1));
            assert_eq!(pre.relu_table_x_fr.len(), 1usize << (half + 1));
            assert_eq!(pre.relu_table_relu_fr.len(), 1usize << (half + 1));
            assert_eq!(pre.pos_range_table(ob).unwrap().len(), 1usize << ob);
            assert_eq!(pre.pos_range_table(gadget).unwrap().len(), 1usize << gadget);
            // A request matching neither budget is an error, not a rebuild.
            assert!(pre.pos_range_table(ob + 1).is_err());
        }
    }

    #[test]
    fn default_sigma_scales_follow_precision_with_caps() {
        // s_x tracks precision up to its cap; s_v is precision+2 up to its.
        assert_eq!(default_sigma_scales(8), (8, 10));
        assert_eq!(default_sigma_scales(11), (11, 13));
        assert_eq!(default_sigma_scales(14), (14, 16));
        // Above the caps, both saturate.
        assert_eq!(
            default_sigma_scales(20),
            (SIGMA_X_SCALE_LOG2_MAX, SIGMA_V_SCALE_LOG2_MAX)
        );
        // Every default is inside the validator's accepted range.
        for p in 2..=24 {
            let (sx, sv) = default_sigma_scales(p);
            assert!(validate_sigma_scales(sx, sv).is_ok(), "precision {p}");
        }
    }

    #[test]
    fn build_carries_and_caches_by_sigma_scales() {
        // Same σ scales ⇒ the same cached table Arc, even at different
        // range budgets; different σ scales ⇒ a distinct table.
        let a = pb(19, 19, 19).unwrap();
        let b = pb(21, 21, 19).unwrap();
        assert_eq!(a.sigma_x_scale_log2, TEST_SX);
        assert_eq!(a.sigma_v_scale_log2, TEST_SV);
        assert!(Arc::ptr_eq(&a.sigma, &b.sigma));
        assert!(Arc::ptr_eq(&a.sigma, &SigmaTables::shared(TEST_SX, TEST_SV)));

        let bigger = Preprocessed::build(19, 19, 19, TEST_SX + 1, TEST_SV, None).unwrap();
        assert_eq!(bigger.sigma_x_scale_log2, TEST_SX + 1);
        assert!(!Arc::ptr_eq(&a.sigma, &bigger.sigma));
        // Bigger s_x ⇒ a strictly larger half-table (2^(7 + s_x)).
        assert_eq!(bigger.sigma.sigmoid_x_fr.len(), a.sigma.sigmoid_x_fr.len() * 2);
    }

    #[test]
    fn gadget_table_shares_out_bound_allocation_when_equal() {
        let pre = pb(19, 19, 19).unwrap();
        assert_eq!(
            pre.gadget_pos_range_table.len(),
            pre.out_bound_pos_range_table.len()
        );
        assert_eq!(pre.gadget_pos_range_table, pre.out_bound_pos_range_table);
        // Both budgets resolve to a 2^19 table.
        assert_eq!(pre.pos_range_table(19).unwrap().len(), 1usize << 19);
    }

    #[test]
    fn build_rejects_out_of_range_bits() {
        assert!(pb(1, TEST_OB_BITS, TEST_GADGET_BITS).is_err());
        assert!(pb(-3, TEST_OB_BITS, TEST_GADGET_BITS).is_err());
        assert!(pb(MAX_TABLE_BITS as i32 + 1, TEST_OB_BITS, TEST_GADGET_BITS).is_err());
        assert!(pb(TEST_HALF_BITS, 1, TEST_GADGET_BITS).is_err());
        assert!(pb(TEST_HALF_BITS, MAX_TABLE_BITS as usize + 1, TEST_GADGET_BITS).is_err());
        assert!(pb(TEST_HALF_BITS, TEST_OB_BITS, 1).is_err());
        assert!(pb(TEST_HALF_BITS, TEST_OB_BITS, MAX_TABLE_BITS as usize + 1).is_err());
    }

    #[test]
    fn build_rejects_out_of_range_sigma_scales() {
        // s_x above the cost cap (would blow up the table); s_x/s_v ≤ 0.
        assert!(Preprocessed::build(
            TEST_HALF_BITS,
            TEST_OB_BITS,
            TEST_GADGET_BITS,
            SIGMA_X_SCALE_LOG2_MAX + 1,
            TEST_SV,
            None
        )
        .is_err());
        assert!(Preprocessed::build(
            TEST_HALF_BITS,
            TEST_OB_BITS,
            TEST_GADGET_BITS,
            0,
            TEST_SV,
            None
        )
        .is_err());
        assert!(Preprocessed::build(
            TEST_HALF_BITS,
            TEST_OB_BITS,
            TEST_GADGET_BITS,
            TEST_SX,
            SIGMA_V_SCALE_LOG2_MAX + 1,
            None
        )
        .is_err());
        assert!(Preprocessed::build(
            TEST_HALF_BITS,
            TEST_OB_BITS,
            TEST_GADGET_BITS,
            TEST_SX,
            0,
            None
        )
        .is_err());
        assert!(validate_sigma_scales(TEST_SX, TEST_SV).is_ok());
    }

    #[test]
    fn build_rejects_out_of_range_input_scale() {
        // A finer input scale in [1, half_bits - 1] is accepted; equal to
        // or above range_table_half_bits hits the exclusive table edge.
        assert!(Preprocessed::build(
            TEST_HALF_BITS,
            TEST_OB_BITS,
            TEST_GADGET_BITS,
            TEST_SX,
            TEST_SV,
            Some(TEST_HALF_BITS - 1)
        )
        .is_ok());
        assert!(Preprocessed::build(
            TEST_HALF_BITS,
            TEST_OB_BITS,
            TEST_GADGET_BITS,
            TEST_SX,
            TEST_SV,
            Some(TEST_HALF_BITS)
        )
        .is_err());
        assert!(Preprocessed::build(
            TEST_HALF_BITS,
            TEST_OB_BITS,
            TEST_GADGET_BITS,
            TEST_SX,
            TEST_SV,
            Some(0)
        )
        .is_err());
        assert!(validate_input_scale(None, TEST_HALF_BITS).is_ok());
        assert!(validate_input_scale(Some(TEST_HALF_BITS - 1), TEST_HALF_BITS).is_ok());
        assert!(validate_input_scale(Some(TEST_HALF_BITS), TEST_HALF_BITS).is_err());
    }

    #[test]
    fn sigmoid_table_lengths_match_spec() {
        let pre = SigmaTables::shared(TEST_SX, TEST_SV);
        let s_x = 1i64 << TEST_SX;
        let expected_len = (SIGMOID_TABLE_X_BOUND_REAL * s_x) as usize;
        assert_eq!(pre.sigmoid_x_fr.len(), expected_len);
        assert!(pre.sigmoid_x_fr.len().is_power_of_two());
        assert_eq!(pre.sigmoid_lower_fr.len(), expected_len);
        assert_eq!(pre.sigmoid_upper_fr.len(), expected_len);
    }

    #[test]
    fn tanh_table_lengths_match_spec() {
        let pre = SigmaTables::shared(TEST_SX, TEST_SV);
        let s_x = 1i64 << TEST_SX;
        let expected_len = (TANH_TABLE_X_BOUND_REAL * s_x) as usize;
        assert_eq!(pre.tanh_x_fr.len(), expected_len);
        assert!(pre.tanh_x_fr.len().is_power_of_two());
        assert_eq!(pre.tanh_lower_fr.len(), expected_len);
        assert_eq!(pre.tanh_upper_fr.len(), expected_len);
    }

    #[test]
    fn sigmoid_envelope_is_conservative() {
        let pre = SigmaTables::shared(TEST_SX, TEST_SV);
        let s_x = (1u64 << TEST_SX) as f64;
        let s_v = (1u64 << TEST_SV) as f64;
        let bound_int = SIGMOID_TABLE_X_BOUND_REAL * (1i64 << TEST_SX);
        for v in [
            0i128,
            1,
            bound_int as i128 / 4,
            bound_int as i128 / 2,
            bound_int as i128 - 1,
        ] {
            let idx = v as usize;
            let xr = (v as f64) / s_x;
            let s_real = 1.0 / (1.0 + (-xr).exp());
            let s_real_q = s_real * s_v;
            let lo = fr_to_signed_i128(pre.sigmoid_lower_fr[idx]).unwrap();
            let up = fr_to_signed_i128(pre.sigmoid_upper_fr[idx]).unwrap();
            assert!(
                (lo as f64) <= s_real_q + 1e-9,
                "sigmoid lower not ≤ true at idx {idx}"
            );
            assert!(
                (up as f64) >= s_real_q - 1e-9,
                "sigmoid upper not ≥ true at idx {idx}"
            );
            assert!(up >= lo, "sigmoid upper ≥ lower at idx {idx}");
        }
    }

    #[test]
    fn sigmoid_symmetry_recovery() {
        // The gadget recovers σ(-x) from the half-table via the relation
        // σ_lo(-x) = s_v − σ_up(x), σ_up(-x) = s_v − σ_lo(x). The test
        // checks the recovered envelope still brackets the true σ(-x).
        let pre = SigmaTables::shared(TEST_SX, TEST_SV);
        let s_v_int = 1i128 << TEST_SV;
        let s_v = s_v_int as f64;
        let s_x = (1u64 << TEST_SX) as f64;
        let nat_int = SIGMOID_NATURAL_BOUND_REAL * (1i64 << TEST_SX);
        for v_pos in [
            1i128,
            (1i128 << TEST_SX),
            (nat_int as i128) / 2,
            (nat_int as i128) - 1,
            (nat_int as i128) + 1,
        ] {
            let lo_pos = fr_to_signed_i128(pre.sigmoid_lower_fr[v_pos as usize]).unwrap();
            let up_pos = fr_to_signed_i128(pre.sigmoid_upper_fr[v_pos as usize]).unwrap();
            let lo_neg_recovered = s_v_int - up_pos;
            let up_neg_recovered = s_v_int - lo_pos;
            let xr_neg = -(v_pos as f64) / s_x;
            let sigma_neg = (1.0 / (1.0 + (-xr_neg).exp())) * s_v;
            assert!(
                (lo_neg_recovered as f64) <= sigma_neg + 1e-9,
                "recovered σ_lo(-{v_pos}) = {lo_neg_recovered} > σ(-x)·s_v = {sigma_neg}"
            );
            assert!(
                (up_neg_recovered as f64) >= sigma_neg - 1e-9,
                "recovered σ_up(-{v_pos}) = {up_neg_recovered} < σ(-x)·s_v = {sigma_neg}"
            );
        }
    }

    #[test]
    fn tanh_envelope_is_conservative() {
        let pre = SigmaTables::shared(TEST_SX, TEST_SV);
        let s_x = (1u64 << TEST_SX) as f64;
        let s_v = (1u64 << TEST_SV) as f64;
        let bound_int = TANH_TABLE_X_BOUND_REAL * (1i64 << TEST_SX);
        for v in [
            0i128,
            bound_int as i128 / 4,
            bound_int as i128 / 2,
            bound_int as i128 - 1,
        ] {
            let idx = v as usize;
            let xr = (v as f64) / s_x;
            let t_real = xr.tanh();
            let t_real_q = t_real * s_v;
            let lo = fr_to_signed_i128(pre.tanh_lower_fr[idx]).unwrap();
            let up = fr_to_signed_i128(pre.tanh_upper_fr[idx]).unwrap();
            assert!(
                (lo as f64) <= t_real_q + 1e-9,
                "tanh lower not ≤ true at idx {idx}"
            );
            assert!(
                (up as f64) >= t_real_q - 1e-9,
                "tanh upper not ≥ true at idx {idx}"
            );
            assert!(up >= lo, "tanh upper ≥ lower at idx {idx}");
        }
    }

    #[test]
    fn saturation_constants_are_conservative() {
        let s_v = (1u64 << TEST_SV) as f64;
        for x in [32.0_f64, 64.0, 128.0, 256.0] {
            let s_pos = (1.0 / (1.0 + (-x).exp())) * s_v;
            assert!(
                (sigmoid_sat_right_lower(TEST_SV) as f64) <= s_pos + 1e-9,
                "sigmoid sat right_lower not ≤ σ({x})·s_v = {s_pos}"
            );
            assert!(
                (sigmoid_sat_right_upper(TEST_SV) as f64) >= s_pos - 1e-9,
                "sigmoid sat right_upper not ≥ σ({x})·s_v = {s_pos}"
            );
            let s_neg = (1.0 / (1.0 + x.exp())) * s_v;
            assert!(
                (sigmoid_sat_left_lower(TEST_SV) as f64) <= s_neg + 1e-9,
                "sigmoid sat left_lower not ≤ σ(-{x})·s_v = {s_neg}"
            );
            assert!(
                (sigmoid_sat_left_upper(TEST_SV) as f64) >= s_neg - 1e-9,
                "sigmoid sat left_upper not ≥ σ(-{x})·s_v = {s_neg}"
            );
        }
        for x in [16.0_f64, 32.0, 64.0, 128.0, 256.0] {
            let t = x.tanh() * s_v;
            assert!(
                (tanh_sat_right_lower(TEST_SV) as f64) <= t + 1e-9,
                "tanh sat right_lower not ≤ tanh({x})·s_v = {t}"
            );
            assert!(
                (tanh_sat_right_upper(TEST_SV) as f64) >= t - 1e-9,
                "tanh sat right_upper not ≥ tanh({x})·s_v = {t}"
            );
        }
    }

    #[test]
    fn extended_table_saturation_regions_match_constants() {
        let pre = SigmaTables::shared(TEST_SX, TEST_SV);
        let bound_int = SIGMOID_TABLE_X_BOUND_REAL * (1i64 << TEST_SX);
        let nat_int = SIGMOID_NATURAL_BOUND_REAL * (1i64 << TEST_SX);
        for x_int in [
            nat_int as i128,
            bound_int as i128 / 2,
            bound_int as i128 - 1,
        ] {
            let lo = fr_to_signed_i128(pre.sigmoid_lower_fr[x_int as usize]).unwrap();
            let up = fr_to_signed_i128(pre.sigmoid_upper_fr[x_int as usize]).unwrap();
            assert_eq!(lo, sigmoid_sat_right_lower(TEST_SV));
            assert_eq!(up, sigmoid_sat_right_upper(TEST_SV));
        }
        let tanh_bound_int = TANH_TABLE_X_BOUND_REAL * (1i64 << TEST_SX);
        let tanh_nat_int = TANH_NATURAL_BOUND_REAL * (1i64 << TEST_SX);
        for x_int in [
            tanh_nat_int as i128,
            tanh_bound_int as i128 / 2,
            tanh_bound_int as i128 - 1,
        ] {
            let lo = fr_to_signed_i128(pre.tanh_lower_fr[x_int as usize]).unwrap();
            let up = fr_to_signed_i128(pre.tanh_upper_fr[x_int as usize]).unwrap();
            assert_eq!(lo, tanh_sat_right_lower(TEST_SV));
            assert_eq!(up, tanh_sat_right_upper(TEST_SV));
        }
    }

    #[test]
    fn extended_table_natural_region_matches_computed_sigma() {
        let pre = SigmaTables::shared(TEST_SX, TEST_SV);
        let nat_int = SIGMOID_NATURAL_BOUND_REAL * (1i64 << TEST_SX);
        let s_x = (1u64 << TEST_SX) as f64;
        let s_v_int = 1i128 << TEST_SV;
        let s_v = s_v_int as f64;
        for x_int in [
            0i128,
            1,
            nat_int as i128 / 4,
            nat_int as i128 / 2,
            nat_int as i128 - 1,
        ] {
            let idx = x_int as usize;
            let xr = (x_int as f64) / s_x;
            let s = (1.0 / (1.0 + (-xr).exp())) * s_v;
            let want_lo = ((s.floor() as i128).saturating_sub(1)).max(0);
            let want_up = ((s.ceil() as i128).saturating_add(1)).min(s_v_int);
            let lo = fr_to_signed_i128(pre.sigmoid_lower_fr[idx]).unwrap();
            let up = fr_to_signed_i128(pre.sigmoid_upper_fr[idx]).unwrap();
            assert_eq!(lo, want_lo);
            assert_eq!(up, want_up);
        }
    }

    #[test]
    fn sigmoid_envelope_in_codomain() {
        let pre = SigmaTables::shared(TEST_SX, TEST_SV);
        let s_v = 1i128 << TEST_SV;
        for i in 0..pre.sigmoid_lower_fr.len() {
            let lo = fr_to_signed_i128(pre.sigmoid_lower_fr[i]).unwrap();
            let up = fr_to_signed_i128(pre.sigmoid_upper_fr[i]).unwrap();
            assert!(lo >= 0);
            assert!(up <= s_v);
            assert!(lo <= up);
        }
    }

    #[test]
    fn tanh_envelope_in_codomain() {
        let pre = SigmaTables::shared(TEST_SX, TEST_SV);
        let s_v = 1i128 << TEST_SV;
        for i in 0..pre.tanh_lower_fr.len() {
            let lo = fr_to_signed_i128(pre.tanh_lower_fr[i]).unwrap();
            let up = fr_to_signed_i128(pre.tanh_upper_fr[i]).unwrap();
            assert!(lo >= 0);
            assert!(up <= s_v);
            assert!(lo <= up);
        }
    }

    #[test]
    fn sshape_tables_are_cross_platform_deterministic() {
        // The σ/tanh tables are computed with `f64::exp` and `f64::tanh`,
        // and the verifier re-derives them. Pin a handful of spot points
        // so a libm change that shifts rounded outputs is caught here
        // instead of breaking otherwise-valid proofs across machines. The
        // expected envelopes at the interior probe (x_real = 1.0) are
        // regenerated from the same build formula at the test scales, so
        // the pin travels with `(s_x, s_v)` instead of baking magic
        // numbers for one fixed scale.
        let pre = SigmaTables::shared(TEST_SX, TEST_SV);
        let bound_int = SIGMOID_TABLE_X_BOUND_REAL * (1i64 << TEST_SX);
        let s_v_int = 1i128 << TEST_SV;
        let s_v = s_v_int as f64;
        let s_x_f = (1u64 << TEST_SX) as f64;
        // Interior envelope regenerated from the scales (matches
        // `build_sigmoid_value_table`).
        let sig_at = |x_int: i128| -> (i128, i128) {
            let s = (1.0 / (1.0 + (-((x_int as f64) / s_x_f)).exp())) * s_v;
            (
                ((s.floor() as i128).saturating_sub(1)).max(0),
                ((s.ceil() as i128).saturating_add(1)).min(s_v_int),
            )
        };
        let one = 1i128 << TEST_SX;
        let (sig_one_lo, sig_one_up) = sig_at(one);
        let sigmoid_spots: [(i128, i128, i128); 4] = [
            (0, (s_v_int / 2) - 1, (s_v_int / 2) + 1),
            (one, sig_one_lo, sig_one_up),
            (
                32 * (1i128 << TEST_SX),
                sigmoid_sat_right_lower(TEST_SV),
                sigmoid_sat_right_upper(TEST_SV),
            ),
            (
                bound_int as i128 - 1,
                sigmoid_sat_right_lower(TEST_SV),
                sigmoid_sat_right_upper(TEST_SV),
            ),
        ];
        for &(x_int, want_lo, want_up) in sigmoid_spots.iter() {
            let idx = x_int as usize;
            let lo = fr_to_signed_i128(pre.sigmoid_lower_fr[idx]).unwrap();
            let up = fr_to_signed_i128(pre.sigmoid_upper_fr[idx]).unwrap();
            assert_eq!(lo, want_lo);
            assert_eq!(up, want_up);
        }

        let tanh_bound = TANH_TABLE_X_BOUND_REAL * (1i64 << TEST_SX);
        let tanh_at = |x_int: i128| -> (i128, i128) {
            let t = ((x_int as f64) / s_x_f).tanh() * s_v;
            (
                ((t.floor() as i128).saturating_sub(1)).max(0),
                ((t.ceil() as i128).saturating_add(1)).min(s_v_int),
            )
        };
        let (tanh_one_lo, tanh_one_up) = tanh_at(one);
        let tanh_spots: [(i128, i128, i128); 3] = [
            (0, 0, 1),
            (one, tanh_one_lo, tanh_one_up),
            (
                tanh_bound as i128 - 1,
                tanh_sat_right_lower(TEST_SV),
                tanh_sat_right_upper(TEST_SV),
            ),
        ];
        for &(x_int, want_lo, want_up) in tanh_spots.iter() {
            let idx = x_int as usize;
            let lo = fr_to_signed_i128(pre.tanh_lower_fr[idx]).unwrap();
            let up = fr_to_signed_i128(pre.tanh_upper_fr[idx]).unwrap();
            assert_eq!(lo, want_lo);
            assert_eq!(up, want_up);
        }
    }

    #[test]
    fn extended_table_build_smoke() {
        use std::time::Instant;
        let start = Instant::now();
        let pre = pb(TEST_HALF_BITS, TEST_OB_BITS, TEST_GADGET_BITS).unwrap();
        let elapsed = start.elapsed();
        let total_entries = pre.range_table_fr.len()
            + pre.out_bound_pos_range_table.len()
            + pre.gadget_pos_range_table.len()
            + pre.relu_table_x_fr.len()
            + pre.relu_table_relu_fr.len()
            + pre.sigma.sigmoid_x_fr.len()
            + pre.sigma.sigmoid_lower_fr.len()
            + pre.sigma.sigmoid_upper_fr.len()
            + pre.sigma.tanh_x_fr.len()
            + pre.sigma.tanh_lower_fr.len()
            + pre.sigma.tanh_upper_fr.len();
        eprintln!(
            "Preprocessed::build() built {total_entries} Fr entries in {:?}",
            elapsed
        );
        assert!(elapsed.as_secs() < 60);
    }

    #[test]
    fn relu_table_at_matches_inline_formula() {
        let pre = pb(TEST_HALF_BITS, TEST_OB_BITS, TEST_GADGET_BITS).unwrap();
        let alpha = Fr::from(7u64);
        let combined = pre.relu_table_at(alpha);
        let half_bits = pre.range_table_half_bits;
        let lo: i128 = -(1i128 << half_bits);
        let zero_idx = 1usize << half_bits;
        let last = combined.len() - 1;
        let hi_minus_one: i128 = (1i128 << half_bits) - 1;
        for (idx, v) in [(0usize, lo), (zero_idx, 0i128), (last, hi_minus_one)] {
            let expected = alpha * signed_lift_to_fr(v) + signed_lift_to_fr(v.max(0));
            assert_eq!(combined[idx], expected, "mismatch at idx {idx}");
        }
    }
}
