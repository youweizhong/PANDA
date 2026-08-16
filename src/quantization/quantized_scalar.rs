//! Single quantized scalar: an integer code with attached `Scale`
//! (`c · 2^e`).
//!
//! Codes are bounded by the runtime `precision_bits` (validated as
//! `precision_bits < range_table_half_bits` at `SnarkParams::setup`),
//! and `precision_bits` is capped by `PRECISION_BITS_ARITH_CEILING`
//! (`Code::BITS / 2`) so pairwise products stay representable in
//! `Code`. The arithmetic here matches
//! the field-level operations the SNARK will perform after lifting `Code`
//! into `Fr`, with rescales recorded as [`RescaleEntry`] witnesses that
//! the SNARK rescale gadget consumes.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::quantization::scale::{RescaleRatio, Scale, ScaleError};

/// Integer code type. The arithmetic semantics match the "signed integer
/// in `(-r/2, r/2]`" view of an Fr element so a future swap to `ark_bn254::Fr`
/// is localised to this alias.
pub type Code = i128;

/// Quantized number: integer code at scale `S = c · 2^e`. The real value
/// represented is `code / S`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qf {
    pub code: Code,
    pub scale: Scale,
}

// `add`/`sub`/`mul`/`neg` are fallible so they cannot be `core::ops`
// traits without silently overflowing or panicking.
#[allow(clippy::should_implement_trait)]
impl Qf {
    pub fn new(code: Code, scale: Scale) -> Self {
        Self { code, scale }
    }

    /// Quantize a real value at `scale` with banker's rounding.
    pub fn from_real(value: f64, scale: Scale) -> Self {
        let target = value * scale.to_real();
        Self {
            code: round_half_to_even(target),
            scale,
        }
    }

    /// Outward-rounding quantization toward `-∞`. Used for `x_lower` and
    /// `b_lower` so the dequantized bound is at most the real value,
    /// preserving the "quantized box ⊇ real box" invariant.
    pub fn from_real_floor(value: f64, scale: Scale) -> Self {
        let target = value * scale.to_real();
        Self {
            code: round_floor(target),
            scale,
        }
    }

    /// Outward-rounding quantization toward `+∞`. Used for `x_upper` and
    /// `b_upper` so the dequantized bound is at least the real value.
    pub fn from_real_ceil(value: f64, scale: Scale) -> Self {
        let target = value * scale.to_real();
        Self {
            code: round_ceil(target),
            scale,
        }
    }

    pub fn to_real(self) -> f64 {
        self.code as f64 / self.scale.to_real()
    }

    /// Add two scalars at the same scale; the result keeps that scale.
    pub fn add(self, other: Self) -> Result<Self, QfError> {
        if self.scale != other.scale {
            return Err(QfError::ScaleMismatch {
                lhs: self.scale,
                rhs: other.scale,
            });
        }
        let code = self
            .code
            .checked_add(other.code)
            .ok_or(QfError::OverflowOnAdd)?;
        Ok(Self {
            code,
            scale: self.scale,
        })
    }

    pub fn sub(self, other: Self) -> Result<Self, QfError> {
        if self.scale != other.scale {
            return Err(QfError::ScaleMismatch {
                lhs: self.scale,
                rhs: other.scale,
            });
        }
        let code = self
            .code
            .checked_sub(other.code)
            .ok_or(QfError::OverflowOnAdd)?;
        Ok(Self {
            code,
            scale: self.scale,
        })
    }

    pub fn neg(self) -> Result<Self, QfError> {
        let code = self.code.checked_neg().ok_or(QfError::OverflowOnAdd)?;
        Ok(Self {
            code,
            scale: self.scale,
        })
    }

    /// Multiply two scalars; the result is at the composed scale
    /// `s_x · s_y`. Does not rescale; callers must call [`Qf::rescale`]
    /// (or use [`Qf::mul_then_rescale`]).
    pub fn mul(self, other: Self) -> Result<Self, QfError> {
        let code = self
            .code
            .checked_mul(other.code)
            .ok_or(QfError::OverflowOnMul)?;
        let scale = self
            .scale
            .compose(other.scale)
            .map_err(QfError::ScaleCompose)?;
        Ok(Self { code, scale })
    }

    /// Pure rescale to `target` (`qy = 1`) with banker's rounding.
    /// Returns the new `Qf` and the witness [`RescaleEntry`] capturing
    /// the boxed-inequality slacks. Use [`Qf::rescale_dir`] for
    /// directional modes.
    pub fn rescale(self, target: Scale) -> Result<(Self, RescaleEntry), QfError> {
        self.rescale_dir(target, RoundDir::HalfAway)
    }

    /// Rescale with an explicit rounding direction.
    pub fn rescale_dir(
        self,
        target: Scale,
        dir: RoundDir,
    ) -> Result<(Self, RescaleEntry), QfError> {
        rescale_div_inner_dir(self.code, self.scale, 1, Scale::from_pow2(0), target, dir)
    }

    /// Multiply then rescale to `target`. Only useful for a single
    /// product; accumulators should sum first and rescale once.
    pub fn mul_then_rescale(
        self,
        other: Self,
        target: Scale,
    ) -> Result<(Self, RescaleEntry), QfError> {
        self.mul(other)?.rescale(target)
    }
}

/// Rounding direction for a rescale event.
///
/// The SNARK rescale gadget verifies a different identity per direction;
/// `crate::snark::rescaling::verify_rescale_event` derives the expected
/// slack offset from the `dir` field of the proof.
///
/// Despite the name, `HalfAway` rounds half-up (toward `+∞`), not "away
/// from zero": for an exact half the result is `k + 1` regardless of
/// sign. The name is a historical artifact.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundDir {
    /// Round-half-up toward `+∞`:
    /// `qz = floor((2·c1·qx + c2·qy) / (2·c2·qy))`.
    HalfAway,
    /// `qz = floor(c1·qx / (c2·qy))`. Used by Lower-direction `b_acc`
    /// and concretize rescales so the dequantized lower bound is a
    /// sound under-approximation.
    Floor,
    /// `qz = ceil(c1·qx / (c2·qy))`. Upper-direction counterpart of
    /// `Floor`.
    Ceil,
}

impl RoundDir {
    /// Byte tag used by the SNARK rescale gadget when serialising
    /// per-event direction into the FS sponge.
    pub fn tag(self) -> u8 {
        match self {
            RoundDir::HalfAway => 0,
            RoundDir::Floor => 1,
            RoundDir::Ceil => 2,
        }
    }

    pub fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(RoundDir::HalfAway),
            1 => Some(RoundDir::Floor),
            2 => Some(RoundDir::Ceil),
            _ => None,
        }
    }
}

/// Boxed-inequality witness recorded for every rescale gate.
///
/// Both slacks are non-negative when the rescale is honest; the SNARK
/// gadget proves that with a Lasso range lookup.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RescaleEntry {
    pub s_in: Scale,
    pub s_out: Scale,
    /// Input integer code (dividend).
    pub qx: Code,
    /// Divisor: `1` for pure rescale, witnessed for [`rescale_div`]
    /// (e.g. the ReLU upper-chord ratio uses `qy = u - l`).
    pub qy: Code,
    /// Output integer code at `s_out`.
    pub qz: Code,
    /// `C1 / C2 = S_in / S_out` as a positive integer ratio.
    pub c1: Code,
    pub c2: Code,
    /// Boxed-inequality slacks; both ≥ 0 iff the rescale is honest.
    pub slack_lo: Code,
    pub slack_hi: Code,
    /// Rounding direction; selects the matching identity in the SNARK
    /// rescale gadget.
    pub dir: RoundDir,
}

/// Rescale with an explicit witnessed divisor `qy`:
/// `qz = round_half_up((C1 · qx) / (C2 · qy))`.
///
/// Emits a witness with `qy != 1` so the SNARK gadget can prove the
/// identity without folding `qy` into `C2`. Used by the ReLU upper-chord
/// `u / (u - l)` math.
pub fn rescale_div(
    qx: Code,
    s_in: Scale,
    qy: Code,
    s_y: Scale,
    target: Scale,
) -> Result<(Qf, RescaleEntry), QfError> {
    if qy == 0 {
        return Err(QfError::DivByZero);
    }
    rescale_div_inner(qx, s_in, qy, s_y, target)
}

fn rescale_div_inner(
    qx: Code,
    s_in: Scale,
    qy: Code,
    s_y: Scale,
    target: Scale,
) -> Result<(Qf, RescaleEntry), QfError> {
    rescale_div_inner_dir(qx, s_in, qy, s_y, target, RoundDir::HalfAway)
}

fn rescale_div_inner_dir(
    qx: Code,
    s_in: Scale,
    qy: Code,
    s_y: Scale,
    target: Scale,
    dir: RoundDir,
) -> Result<(Qf, RescaleEntry), QfError> {
    // C1 / C2 = (S_y · S_out) / S_in; from qz · S_out = qx · S_y / (qy · S_in).
    let num_scale = s_y.compose(target).map_err(QfError::ScaleCompose)?;
    let RescaleRatio { c1, c2 } = num_scale
        .ratio_as_c1_c2(s_in)
        .map_err(QfError::ScaleCompose)?;

    // Normalise qy positive so the gadget always sees a positive denom.
    let (qx_eff, qy_pos) = if qy < 0 { (-qx, -qy) } else { (qx, qy) };
    let two_c1_qx = 2i128
        .checked_mul(c1)
        .and_then(|v| v.checked_mul(qx_eff))
        .ok_or(QfError::OverflowOnMul)?;
    let denom = c2.checked_mul(qy_pos).ok_or(QfError::OverflowOnMul)?;
    if denom == 0 {
        return Err(QfError::DivByZero);
    }
    // Unified slack form across the three directions, all in `[0, 2·denom)`:
    //   slack_lo = 2·c1·qx − 2·denom·qz + offset(dir, denom)
    //   offset: HalfAway → denom, Floor → 0, Ceil → 2·denom − 2.
    let two_denom = denom.checked_mul(2).ok_or(QfError::OverflowOnMul)?;
    let qz = match dir {
        RoundDir::HalfAway => {
            let numerator = two_c1_qx.checked_add(denom).ok_or(QfError::OverflowOnAdd)?;
            euclid_floor_div(numerator, two_denom)
        }
        RoundDir::Floor => {
            euclid_floor_div_signed(c1.checked_mul(qx_eff).ok_or(QfError::OverflowOnMul)?, denom)
        }
        RoundDir::Ceil => {
            // ceil(n/d) = -floor(-n/d) for positive d.
            let num = c1.checked_mul(qx_eff).ok_or(QfError::OverflowOnMul)?;
            -euclid_floor_div_signed(-num, denom)
        }
    };

    let offset: Code = match dir {
        RoundDir::HalfAway => denom,
        RoundDir::Floor => 0,
        RoundDir::Ceil => two_denom - 2,
    };
    let two_denom_qz = two_denom.checked_mul(qz).ok_or(QfError::OverflowOnMul)?;
    let slack_lo = two_c1_qx
        .checked_sub(two_denom_qz)
        .and_then(|v| v.checked_add(offset))
        .ok_or(QfError::OverflowOnAdd)?;
    let slack_hi = two_denom
        .checked_sub(1)
        .and_then(|v| v.checked_sub(slack_lo))
        .ok_or(QfError::OverflowOnAdd)?;
    debug_assert!(slack_lo >= 0, "slack_lo = {slack_lo}, dir={:?}", dir);
    debug_assert!(slack_hi >= 0, "slack_hi = {slack_hi}, dir={:?}", dir);
    debug_assert!(
        slack_lo + slack_hi == two_denom - 1,
        "slack invariant: lo + hi = 2·denom − 1, got lo={slack_lo} hi={slack_hi} 2denom={two_denom}"
    );

    let entry = RescaleEntry {
        s_in,
        s_out: target,
        qx,
        qy,
        qz,
        c1,
        c2,
        slack_lo,
        slack_hi,
        dir,
    };
    Ok((
        Qf {
            code: qz,
            scale: target,
        },
        entry,
    ))
}

/// Euclidean floor division for signed dividend and positive divisor;
/// matches Python's `//`. Used for the `Floor`/`Ceil` directions.
fn euclid_floor_div_signed(numer: Code, denom: Code) -> Code {
    debug_assert!(denom > 0, "euclid_floor_div_signed assumes positive denom");
    let q = numer / denom;
    let r = numer % denom;
    if r < 0 {
        q - 1
    } else {
        q
    }
}

#[cfg(test)]
#[test]
fn directional_rescale_identity_holds() {
    let s_in = Scale { c: 1, e: 28 };
    let s_out = Scale { c: 1, e: 14 };
    for &qx in &[
        -100000i128,
        -32768,
        -1000,
        -1,
        0,
        1,
        1000,
        32768,
        65536,
        100000,
    ] {
        let qf = Qf::new(qx, s_in);
        let (qz_floor, e_floor) = qf.rescale_dir(s_out, RoundDir::Floor).unwrap();
        let (qz_ceil, e_ceil) = qf.rescale_dir(s_out, RoundDir::Ceil).unwrap();
        let (qz_half, e_half) = qf.rescale_dir(s_out, RoundDir::HalfAway).unwrap();
        for (label, qz, slack, dir) in [
            ("Floor", qz_floor.code, e_floor.slack_lo, RoundDir::Floor),
            ("Ceil", qz_ceil.code, e_ceil.slack_lo, RoundDir::Ceil),
            ("HalfA", qz_half.code, e_half.slack_lo, RoundDir::HalfAway),
        ] {
            let c1 = e_floor.c1;
            let c2 = e_floor.c2;
            let offset = match dir {
                RoundDir::HalfAway => c2,
                RoundDir::Floor => 0,
                RoundDir::Ceil => 2 * c2 - 2,
            };
            let expected = 2 * c1 * qx - 2 * c2 * qz + offset;
            assert_eq!(slack, expected, "{label} qx={qx} qz={qz} c1={c1} c2={c2}");
            assert!(slack >= 0, "{label} negative slack");
            assert!(slack < 2 * c2, "{label} slack out of range");
        }
    }
}

/// Floor toward `-∞` for f64 → i128 quantization.
pub(crate) fn round_floor(x: f64) -> Code {
    x.floor() as i128
}

/// Ceil toward `+∞` for f64 → i128 quantization.
pub(crate) fn round_ceil(x: f64) -> Code {
    x.ceil() as i128
}

/// Banker's rounding (round half to even) for f64 → i128.
fn round_half_to_even(x: f64) -> Code {
    let floor = x.floor();
    let diff = x - floor;
    let base = floor as i128;
    if diff < 0.5 {
        base
    } else if diff > 0.5 {
        base + 1
    } else if base % 2 == 0 {
        base
    } else {
        base + 1
    }
}

/// Euclidean floor division: floors toward `-∞` (Rust's `/` truncates
/// toward zero). Required so the boxed inequality's `qz` is well-defined
/// for negative numerators.
fn euclid_floor_div(numer: i128, denom: i128) -> Code {
    debug_assert!(denom > 0, "euclid_floor_div assumes positive denom");
    let q = numer / denom;
    let r = numer % denom;
    if r < 0 {
        q - 1
    } else {
        q
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum QfError {
    #[error("Qf scale mismatch: lhs={lhs:?} rhs={rhs:?}")]
    ScaleMismatch { lhs: Scale, rhs: Scale },
    #[error("Qf addition overflowed i128")]
    OverflowOnAdd,
    #[error("Qf multiplication overflowed i128")]
    OverflowOnMul,
    #[error("rescale divisor (qy) was zero")]
    DivByZero,
    #[error("scale composition failed: {0}")]
    ScaleCompose(#[source] ScaleError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pow2(q: i32) -> Scale {
        Scale::from_pow2(q)
    }

    #[test]
    fn from_real_round_trips_pow2() {
        let s = pow2(8);
        let q = Qf::from_real(1.5, s);
        // 1.5 · 256 = 384
        assert_eq!(q.code, 384);
        assert!((q.to_real() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn from_real_banker_rounds_at_half() {
        let s = pow2(0);
        // 0.5 → 0 (round to even), 1.5 → 2 (round to even).
        assert_eq!(Qf::from_real(0.5, s).code, 0);
        assert_eq!(Qf::from_real(1.5, s).code, 2);
        assert_eq!(Qf::from_real(2.5, s).code, 2);
        assert_eq!(Qf::from_real(-0.5, s).code, 0);
        assert_eq!(Qf::from_real(-1.5, s).code, -2);
    }

    #[test]
    fn add_requires_same_scale() {
        let a = Qf::from_real(1.0, pow2(8));
        let b = Qf::from_real(1.0, pow2(4));
        assert!(matches!(a.add(b), Err(QfError::ScaleMismatch { .. })));
    }

    #[test]
    fn add_works_at_matching_scale() {
        let a = Qf::from_real(1.5, pow2(8));
        let b = Qf::from_real(0.25, pow2(8));
        let c = a.add(b).unwrap();
        assert_eq!(c.code, 384 + 64);
        assert_eq!(c.scale, pow2(8));
    }

    #[test]
    fn mul_composes_scales() {
        let a = Qf::from_real(3.0, pow2(4));
        let b = Qf::from_real(0.5, pow2(8));
        let c = a.mul(b).unwrap();
        // codes: 48 · 128 = 6144; scale = pow2(12); to_real = 6144 / 4096 = 1.5.
        assert_eq!(c.code, 6144);
        assert_eq!(c.scale, pow2(12));
        assert!((c.to_real() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn rescale_pow2_pure_round_trip() {
        let a = Qf::from_real(1.5, pow2(8));
        let (b, w) = a.rescale(pow2(4)).unwrap();
        assert_eq!(b.scale, pow2(4));
        assert_eq!(b.code, 24); // 1.5 · 16
        assert_eq!(w.qy, 1);
        assert_eq!(w.s_in, pow2(8));
        assert_eq!(w.s_out, pow2(4));
        assert!(w.slack_lo >= 0 && w.slack_hi >= 0);
    }

    #[test]
    fn rescale_pow2_round_half_up_at_exact_half() {
        // qx = 7 at S_in = 2, target S_out = 1 ⇒ value = 3.5 ⇒ qz = 4
        // (boxed-inequality round-half-up rule).
        let (out, w) = Qf::new(7, pow2(1)).rescale(pow2(0)).unwrap();
        assert_eq!(out.code, 4);
        assert_eq!(w.qy, 1);
        assert!(w.slack_lo >= 0 && w.slack_hi >= 0);
        // qx = 5 at S_in = 2 ⇒ value = 2.5 ⇒ qz = 3 (also rounds up).
        let (out, _) = Qf::new(5, pow2(1)).rescale(pow2(0)).unwrap();
        assert_eq!(out.code, 3);
        // qx = -7 at S_in = 2 ⇒ value = -3.5. Round-half-up away from
        // zero on negatives lands on -3 because the boxed inequality
        // breaks ties toward +∞.
        let (out, _) = Qf::new(-7, pow2(1)).rescale(pow2(0)).unwrap();
        assert_eq!(out.code, -3);
    }

    #[test]
    fn rescale_semantics_match_real_value() {
        // Pick a handful of (s_in, s_out) and verify the rescaled real
        // value is within 1 ULP of the pre-rescale real value at the
        // output scale.
        for &(s_in, s_out) in &[
            (pow2(8), pow2(4)),
            (pow2(4), pow2(8)),
            (pow2(8), pow2(0)),
            (Scale::new(3, 2).unwrap(), Scale::new(3, -1).unwrap()),
            (Scale::new(7, -3).unwrap(), Scale::new(5, 2).unwrap()),
        ] {
            // Pick a real value that fits comfortably in i32 at both
            // scales.
            for &v in &[0.0, 0.25, 0.75, 1.0, -0.5, 3.125, -7.0] {
                let a = Qf::from_real(v, s_in);
                let (b, w) = a.rescale(s_out).unwrap();
                assert_eq!(b.scale, s_out);
                let v_round = b.to_real();
                // Worst-case drift: the original `from_real` quantises to
                // S_in's grid (error ≤ 1/(2·S_in)); rescale rounds again
                // (error ≤ 1/(2·S_out)). The triangle-inequality bound is
                // the sum of half-grid sizes.
                let max_step = 0.5 * (1.0 / s_in.to_real() + 1.0 / s_out.to_real());
                assert!(
                    (v - v_round).abs() <= max_step + 1e-12,
                    "rescale drift {v} -> {v_round} > {max_step} (s_in={s_in:?}, s_out={s_out:?})"
                );
                assert!(w.slack_lo >= 0);
                assert!(w.slack_hi >= 0);
            }
        }
    }

    #[test]
    fn rescale_div_relu_upper_chord_shape() {
        // ReLU upper chord ratio: u / (u - l). Take l = -2, u = 6 in
        // unit-scale codes. Expected qz = 6 / 8 = 0.75. At target scale
        // pow2(2), qz = round(0.75 · 4) = 3.
        let s = Scale::from_pow2(0);
        let (qf, w) = rescale_div(/*qx=*/ 6, s, /*qy=*/ 8, s, Scale::from_pow2(2)).unwrap();
        assert_eq!(qf.code, 3);
        assert_eq!(w.qy, 8);
        assert!(w.slack_lo >= 0 && w.slack_hi >= 0);
    }

    #[test]
    fn rescale_div_handles_negative_qy() {
        // qy < 0: gadget normalizes the sign internally and returns the
        // same qz as if qy were positive with -qx.
        let s = Scale::from_pow2(0);
        let (a, _) = rescale_div(6, s, 8, s, Scale::from_pow2(2)).unwrap();
        let (b, _) = rescale_div(-6, s, -8, s, Scale::from_pow2(2)).unwrap();
        assert_eq!(a.code, b.code);
    }

    #[test]
    fn rescale_div_rejects_zero_divisor() {
        let s = Scale::from_pow2(0);
        assert!(matches!(
            rescale_div(1, s, 0, s, Scale::from_pow2(0)),
            Err(QfError::DivByZero)
        ));
    }
}
