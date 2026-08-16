//! BN254 `Fr` encoding helpers.
//!
//! The cert generator works in `i128` integer codes; the SNARK driver
//! lifts those codes into `ark_bn254::Fr`. This module defines the
//! canonical signed lift, its inverse on values that fit in 128 bits,
//! and a serde-friendly `Fr` wrapper used in JSON fixtures.

use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::quantization::quantized_scalar::{Code, RescaleEntry};

/// Lift a signed integer into `Fr`. Negatives map to `r - |x|`.
pub fn signed_lift_to_fr(x: Code) -> Fr {
    if x >= 0 {
        Fr::from(x as u128)
    } else {
        let pos = Fr::from((-x) as u128);
        -pos
    }
}

/// Recover the canonical signed representative of an `Fr` element when it
/// fits in `i128`. Returns `None` if the canonical lift overflows `i128`;
/// honest cert codes stay inside the runtime range table and never trigger this.
pub fn fr_to_signed_i128(x: Fr) -> Option<Code> {
    let modulus = Fr::MODULUS;
    let half_modulus = {
        let mut m = modulus;
        m.div2();
        m
    };
    let bigint = x.into_bigint();
    if bigint <= half_modulus {
        bigint_to_i128_unsigned(bigint).map(|v| v as Code)
    } else {
        let neg = (-x).into_bigint();
        bigint_to_i128_unsigned(neg).and_then(|v| (v as i128).checked_neg())
    }
}

fn bigint_to_i128_unsigned(b: <Fr as PrimeField>::BigInt) -> Option<u128> {
    let bytes = b.to_bytes_le();
    if bytes.iter().skip(16).any(|&b| b != 0) {
        return None;
    }
    let mut buf = [0u8; 16];
    buf.copy_from_slice(&bytes[..16]);
    Some(u128::from_le_bytes(buf))
}

/// Field-encoded [`RescaleEntry`]. Each field is the signed-lift of the
/// corresponding `i128` code; consumed directly as an `Fr` witness.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrRescaleEntry {
    pub qx: SerFr,
    pub qy: SerFr,
    pub qz: SerFr,
    pub c1: SerFr,
    pub c2: SerFr,
    pub slack_lo: SerFr,
    pub slack_hi: SerFr,
}

impl FrRescaleEntry {
    pub fn from_i128(entry: &RescaleEntry) -> Self {
        Self {
            qx: SerFr::from(signed_lift_to_fr(entry.qx)),
            qy: SerFr::from(signed_lift_to_fr(entry.qy)),
            qz: SerFr::from(signed_lift_to_fr(entry.qz)),
            c1: SerFr::from(signed_lift_to_fr(entry.c1)),
            c2: SerFr::from(signed_lift_to_fr(entry.c2)),
            slack_lo: SerFr::from(signed_lift_to_fr(entry.slack_lo)),
            slack_hi: SerFr::from(signed_lift_to_fr(entry.slack_hi)),
        }
    }
}

/// Serde wrapper around `Fr`. Round-trips through the canonical decimal
/// representation so JSON fixtures stay human-readable. `Fr` itself does
/// not implement serde's traits.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SerFr(pub Fr);

impl From<Fr> for SerFr {
    fn from(f: Fr) -> Self {
        SerFr(f)
    }
}

impl Serialize for SerFr {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        // Unsigned lift in [0, r) — the on-wire shape the verifier reads.
        let s = self.0.into_bigint().to_string();
        ser.serialize_str(&s)
    }
}

impl<'de> Deserialize<'de> for SerFr {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let s = String::deserialize(de)?;
        let bigint = parse_decimal_to_bigint(&s).map_err(D::Error::custom)?;
        let fr =
            Fr::from_bigint(bigint).ok_or_else(|| D::Error::custom("Fr from_bigint failed"))?;
        Ok(SerFr(fr))
    }
}

fn parse_decimal_to_bigint(s: &str) -> Result<<Fr as PrimeField>::BigInt, FieldError> {
    // Repeated *10 + digit; inputs are always in [0, r).
    let mut acc = Fr::from(0u64);
    let ten = Fr::from(10u64);
    for ch in s.chars() {
        let d = ch.to_digit(10).ok_or(FieldError::ParseDigit)?;
        acc = acc * ten + Fr::from(d as u64);
    }
    Ok(acc.into_bigint())
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FieldError {
    #[error("invalid digit in decimal Fr")]
    ParseDigit,
    #[error("bigint exceeded field modulus")]
    BigIntOutOfField,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_positive_round_trip() {
        for x in [0i128, 1, 2, 7, 1000, i64::MAX as i128] {
            let f = signed_lift_to_fr(x);
            let back = fr_to_signed_i128(f).expect("fits in i128");
            assert_eq!(back, x);
        }
    }

    #[test]
    fn small_negative_round_trip() {
        for x in [-1i128, -7, -1000, i64::MIN as i128] {
            let f = signed_lift_to_fr(x);
            let back = fr_to_signed_i128(f).expect("fits in i128");
            assert_eq!(back, x);
        }
    }

    #[test]
    fn fr_addition_matches_signed_arithmetic() {
        for &(a, b) in &[
            (3i128, 4i128),
            (-3, 5),
            (-7, -10),
            (100_000, -50),
            (i64::MAX as i128, 1),
        ] {
            let s = signed_lift_to_fr(a) + signed_lift_to_fr(b);
            let recovered = fr_to_signed_i128(s).unwrap();
            assert_eq!(recovered, a + b);
        }
    }

    #[test]
    fn fr_multiplication_matches_signed_arithmetic() {
        for &(a, b) in &[
            (3i128, 4i128),
            (-7, 5),
            (-1000, -1000),
            (123456789, -987654321),
        ] {
            let s = signed_lift_to_fr(a) * signed_lift_to_fr(b);
            let recovered = fr_to_signed_i128(s).unwrap();
            assert_eq!(recovered, a * b);
        }
    }

    #[test]
    fn ser_fr_round_trip_through_json() {
        let f = signed_lift_to_fr(-12345);
        let wrapped = SerFr(f);
        let s = serde_json::to_string(&wrapped).unwrap();
        let back: SerFr = serde_json::from_str(&s).unwrap();
        assert_eq!(wrapped, back);
        assert_eq!(fr_to_signed_i128(back.0).unwrap(), -12345);
    }

    #[test]
    fn fr_rescale_entry_round_trips_through_json() {
        let entry = RescaleEntry {
            s_in: crate::quantization::scale::Scale::from_pow2(8),
            s_out: crate::quantization::scale::Scale::from_pow2(4),
            qx: 384,
            qy: 1,
            qz: 24,
            c1: 1,
            c2: 16,
            slack_lo: 16,
            slack_hi: 15,
            dir: crate::quantization::quantized_scalar::RoundDir::HalfAway,
        };
        let fr = FrRescaleEntry::from_i128(&entry);
        let json = serde_json::to_string(&fr).unwrap();
        let back: FrRescaleEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(fr, back);
        assert_eq!(fr_to_signed_i128(back.qz.0).unwrap(), 24);
        assert_eq!(fr_to_signed_i128(back.slack_lo.0).unwrap(), 16);
    }
}
