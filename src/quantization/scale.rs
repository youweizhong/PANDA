//! Fixed-point scale family `S = c · 2^e`.
//!
//! Every quantized tensor carries one of these. The `(c, e)` form covers
//! both the legacy power-of-two case (`c = 1`) and the non-pow2 scales
//! emitted by [`Scale::search`].
//!
//! Search rectangle and tie-breaking rules mirror zkGPT's
//! `pair<int,int> search(double)`. [`Scale::search`] is the O(E) variant;
//! [`Scale::search_bruteforce`] is kept for cross-validation in tests.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Smallest `c` considered by [`Scale::search`].
pub const SEARCH_C_MIN: i64 = 1;
/// Largest `c` considered by [`Scale::search`].
pub const SEARCH_C_MAX: i64 = 800;
/// Smallest `e` considered by [`Scale::search`].
pub const SEARCH_E_MIN: i32 = -10;
/// Largest `e` considered by [`Scale::search`].
pub const SEARCH_E_MAX: i32 = 10;
/// Verifier-side bound on `|e|` for any scale used in the SNARK. Keeps
/// the shifts in [`Scale::ratio_as_c1_c2`] inside `i128` so a hostile
/// composed scale cannot trigger a shift wrap before the binding check.
pub const MAX_SCALE_E_ABS: i32 = 32;
/// Initial sentinel for the search loop. Any finite positive target
/// falls strictly below this on the first candidate.
pub const SEARCH_SENTINEL: f64 = 1e9;

/// Scale `c · 2^e` with positive integer mantissa.
///
/// Search-emitted scales live in
/// `c ∈ [SEARCH_C_MIN, SEARCH_C_MAX], e ∈ [SEARCH_E_MIN, SEARCH_E_MAX]`.
/// Constructors do not enforce the search rectangle, so composed scales
/// (via [`Scale::compose`]) may land outside it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scale {
    pub c: i64,
    pub e: i32,
}

impl Scale {
    pub fn new(c: i64, e: i32) -> Result<Self, ScaleError> {
        if c <= 0 {
            return Err(ScaleError::NonPositiveC { c });
        }
        Ok(Self { c, e })
    }

    /// Verifier-side validation: reject scales that would cause
    /// [`Scale::ratio_as_c1_c2`] to overflow `i128` or shift beyond
    /// the type width. Any scale stored in a public proof component
    /// must pass this check before the SNARK driver consumes it.
    pub fn validate_for_pin(self) -> Result<(), ScaleError> {
        if self.c <= 0 {
            return Err(ScaleError::NonPositiveC { c: self.c });
        }
        // `unsigned_abs` (vs `e.abs()`) handles `e = i32::MIN` without
        // panicking in debug or wrapping in release.
        if self.e.unsigned_abs() > MAX_SCALE_E_ABS as u32 {
            return Err(ScaleError::ExponentOutOfRange { e: self.e });
        }
        Ok(())
    }

    /// Power-of-two scale `c = 1, e = q` — the "real value = integer / 2^q"
    /// convention.
    pub fn from_pow2(q: i32) -> Self {
        Self { c: 1, e: q }
    }

    /// Returns the real-valued scale `c · 2^e`.
    pub fn to_real(self) -> f64 {
        let mantissa = self.c as f64;
        mantissa * 2f64.powi(self.e)
    }

    /// True iff `c == 1` (the power-of-two case).
    pub fn is_pow2(self) -> bool {
        self.c == 1
    }

    /// Returns `e` if `c == 1`, error otherwise.
    pub fn pow2_exponent(self) -> Result<i32, ScaleError> {
        if self.c != 1 {
            return Err(ScaleError::NotPow2 {
                c: self.c,
                e: self.e,
            });
        }
        Ok(self.e)
    }

    /// Compose two scales:
    /// `(c_x · 2^{e_x}) · (c_y · 2^{e_y}) = (c_x · c_y) · 2^{e_x + e_y}`.
    /// Does not normalize; trailing factors of two are preserved so an
    /// honest composer can replay the bit-exact ratio later.
    pub fn compose(self, other: Self) -> Result<Self, ScaleError> {
        let c = self
            .c
            .checked_mul(other.c)
            .ok_or(ScaleError::OverflowOnCompose)?;
        let e = self
            .e
            .checked_add(other.e)
            .ok_or(ScaleError::OverflowOnCompose)?;
        Ok(Self { c, e })
    }

    /// Strip trailing factors of two from `c`, incrementing `e` to preserve
    /// the real value. After `normalize`, `c` is odd.
    pub fn normalize(self) -> Self {
        let mut c = self.c;
        let mut e = self.e;
        while c != 0 && c & 1 == 0 {
            c >>= 1;
            e += 1;
        }
        Self { c, e }
    }

    /// Best `(c, e)` in the search rectangle approximating `target`.
    /// O(E) probes; at fixed `e` the optimal `c` is one of
    /// `floor(target / 2^e)` and that plus one, clamped to the
    /// rectangle. Tie-breaking matches [`Scale::search_bruteforce`].
    pub fn search(target: f64) -> Result<Self, ScaleError> {
        if !(target.is_finite() && target > 0.0) {
            return Err(ScaleError::SearchInputInvalid { target });
        }
        let mut best = Self {
            c: SEARCH_C_MIN,
            e: SEARCH_E_MIN,
        };
        let mut min_diff = SEARCH_SENTINEL;
        for e in SEARCH_E_MIN..=SEARCH_E_MAX {
            let s_unit = 2f64.powi(e);
            let c_real = target / s_unit;
            let c_floor = c_real.floor() as i64;
            for &c_cand in &[c_floor, c_floor + 1] {
                let c = c_cand.clamp(SEARCH_C_MIN, SEARCH_C_MAX);
                let s = s_unit * c as f64;
                let diff = (s - target).abs();
                if diff < min_diff {
                    min_diff = diff;
                    best = Self { c, e };
                }
            }
        }
        Ok(best)
    }

    /// Brute-force reference search. Identical output to [`Scale::search`];
    /// kept for cross-validation in tests.
    pub fn search_bruteforce(target: f64) -> Result<Self, ScaleError> {
        if !(target.is_finite() && target > 0.0) {
            return Err(ScaleError::SearchInputInvalid { target });
        }
        let mut best = Self {
            c: SEARCH_C_MIN,
            e: SEARCH_E_MIN,
        };
        let mut min_diff = SEARCH_SENTINEL;
        for e in SEARCH_E_MIN..=SEARCH_E_MAX {
            for c in SEARCH_C_MIN..=SEARCH_C_MAX {
                let s = 2f64.powi(e) * c as f64;
                let diff = (s - target).abs();
                if diff < min_diff {
                    min_diff = diff;
                    best = Self { c, e };
                }
            }
        }
        Ok(best)
    }

    /// Multiplier `S_in / S_out` expressed as a positive-integer ratio
    /// `(C1, C2)` with `C1 / C2 = S_in / S_out`. Does not reduce by
    /// `gcd(c_in, c_out)`; the bare `c_in`, `c_out` (separated by their
    /// pow-2 difference) match the boxed-inequality math.
    ///
    /// Returns [`ScaleError::ExponentOutOfRange`] if a hostile composed
    /// scale would push the internal shift past the `i128` width.
    pub fn ratio_as_c1_c2(self, other: Self) -> Result<RescaleRatio, ScaleError> {
        let delta = self
            .e
            .checked_sub(other.e)
            .ok_or(ScaleError::ExponentOutOfRange { e: self.e })?;
        if delta >= 0 {
            let shift: u32 = (delta as i64)
                .try_into()
                .map_err(|_| ScaleError::ExponentOutOfRange { e: delta })?;
            let c1 = (self.c as i128)
                .checked_shl(shift)
                .ok_or(ScaleError::ExponentOutOfRange { e: delta })?;
            let c2 = other.c as i128;
            Ok(RescaleRatio { c1, c2 })
        } else {
            let shift: u32 = (-(delta as i64))
                .try_into()
                .map_err(|_| ScaleError::ExponentOutOfRange { e: delta })?;
            let c1 = self.c as i128;
            let c2 = (other.c as i128)
                .checked_shl(shift)
                .ok_or(ScaleError::ExponentOutOfRange { e: delta })?;
            Ok(RescaleRatio { c1, c2 })
        }
    }
}

/// Rescale multiplier as a positive ratio of integers.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RescaleRatio {
    pub c1: i128,
    pub c2: i128,
}

#[derive(Debug, Error, PartialEq)]
pub enum ScaleError {
    #[error("scale c must be positive (got {c})")]
    NonPositiveC { c: i64 },
    #[error("scale c={c}, e={e} is not a power of two (c != 1)")]
    NotPow2 { c: i64, e: i32 },
    #[error("scale composition overflowed i64/i32")]
    OverflowOnCompose,
    #[error("Scale::search input must be finite and positive (got {target})")]
    SearchInputInvalid { target: f64 },
    #[error("scale e={e} exceeds verifier-side bound MAX_SCALE_E_ABS")]
    ExponentOutOfRange { e: i32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pow2_round_trip() {
        for q in -8..=8 {
            let s = Scale::from_pow2(q);
            assert!(s.is_pow2());
            assert_eq!(s.pow2_exponent().unwrap(), q);
            assert!((s.to_real() - 2f64.powi(q)).abs() < 1e-12);
        }
    }

    #[test]
    fn search_matches_bruteforce_on_random_targets() {
        // A reasonable spread across the search rectangle. We avoid 0 and
        // numbers too far below the rectangle (those collapse to {1, e_min}
        // and the test would be trivial) but otherwise hit a mix.
        let targets = [
            0.001, 0.1, 0.5, 1.0, 1.7, 3.7, 7.5, 13.0, 77.0, 200.0, 511.5, 800.0, 1023.5, 2048.0,
            1e6,
        ];
        for &t in &targets {
            let opt = Scale::search(t).unwrap();
            let brute = Scale::search_bruteforce(t).unwrap();
            assert_eq!(opt, brute, "search(target={t}) diverged from bruteforce");
        }
    }

    #[test]
    fn search_rejects_nonfinite_or_nonpositive() {
        assert!(matches!(
            Scale::search(0.0),
            Err(ScaleError::SearchInputInvalid { .. })
        ));
        assert!(matches!(
            Scale::search(-1.0),
            Err(ScaleError::SearchInputInvalid { .. })
        ));
        assert!(matches!(
            Scale::search(f64::NAN),
            Err(ScaleError::SearchInputInvalid { .. })
        ));
        assert!(matches!(
            Scale::search(f64::INFINITY),
            Err(ScaleError::SearchInputInvalid { .. })
        ));
    }

    #[test]
    fn compose_does_not_normalize() {
        let s = Scale::new(2, 3)
            .unwrap()
            .compose(Scale::new(2, 0).unwrap())
            .unwrap();
        assert_eq!(s, Scale { c: 4, e: 3 });
    }

    #[test]
    fn normalize_strips_powers_of_two() {
        let s = Scale::new(12, 5).unwrap().normalize();
        assert_eq!(s, Scale { c: 3, e: 7 });
        // Idempotent on already-normalized scales.
        let t = Scale::new(7, -2).unwrap().normalize();
        assert_eq!(t, Scale { c: 7, e: -2 });
    }

    #[test]
    fn ratio_pow2_only() {
        let in_s = Scale::from_pow2(8); // 256
        let out_s = Scale::from_pow2(3); // 8
        let r = in_s.ratio_as_c1_c2(out_s).unwrap();
        // S_in / S_out = 32 / 1 = 32 / 1
        assert_eq!(r.c1, 32);
        assert_eq!(r.c2, 1);
    }

    #[test]
    fn ratio_with_nontrivial_c() {
        let in_s = Scale::new(3, 2).unwrap(); // 12
        let out_s = Scale::new(5, -1).unwrap(); // 2.5
        let r = in_s.ratio_as_c1_c2(out_s).unwrap();
        // delta = 2 - (-1) = 3, so c1 = 3 << 3 = 24, c2 = 5
        assert_eq!(r.c1, 24);
        assert_eq!(r.c2, 5);
        // Sanity: (24/5) * 2^0 = 4.8 = 12 / 2.5.
        let s_in_real = 3.0 * 4.0;
        let s_out_real = 5.0 * 0.5_f64;
        assert!((s_in_real / s_out_real - 24.0 / 5.0).abs() < 1e-12);
    }

    #[test]
    fn validate_for_pin_handles_i32_min_without_panic() {
        // `e = i32::MIN` must reject cleanly: an `e.abs()` would panic in
        // debug and wrap in release, both unsound for verifier preflight.
        let s = Scale { c: 1, e: i32::MIN };
        assert!(matches!(
            s.validate_for_pin(),
            Err(ScaleError::ExponentOutOfRange { e: _ })
        ));
    }

    #[test]
    fn ratio_rejects_huge_delta_instead_of_panicking() {
        // A composed scale whose `e` would push `delta` past the i128
        // shift width must return an Err rather than panicking. Here
        // delta = 200 is well outside the type width.
        let a = Scale { c: 1, e: 100 };
        let b = Scale { c: 1, e: -100 };
        // delta = 100 - (-100) = 200, which is past i128 width.
        assert!(matches!(
            a.ratio_as_c1_c2(b),
            Err(ScaleError::ExponentOutOfRange { .. })
        ));
        // Reverse direction also rejects.
        assert!(matches!(
            b.ratio_as_c1_c2(a),
            Err(ScaleError::ExponentOutOfRange { .. })
        ));
    }
}
