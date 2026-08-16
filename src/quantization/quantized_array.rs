//! Quantized arrays: `ndarray` of integer codes with one attached
//! [`Scale`].
//!
//! "Single scale per array" means every value in a quantized matrix or
//! vector lives on the same grid, so addition requires matching scales
//! and matmuls compose scales the obvious way. This file collects the
//! 1-D and 2-D wrappers, the quantize/dequantize helpers, the per-element
//! rescale helpers (with directional rounding modes), and the integer
//! matmul/matvec used by the bound generator.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::quantization::quantized_scalar::{Code, QfError, RescaleEntry};
use crate::quantization::scale::{Scale, ScaleError};

/// Arithmetic ceiling on `precision_bits`, derived from the [`Code`]
/// integer width: pairwise products of codes bounded by
/// `2^(precision_bits - 1)` must stay representable in `Code = i128`,
/// so the ceiling is `Code::BITS / 2`. This is an overflow guard on the
/// integer arithmetic, NOT a tuning value — the SNARK-facing headroom
/// requirement (`precision_bits < range_table_half_bits`, so every
/// honest code stays strictly inside the runtime signed range table) is
/// validated where the runtime table parameters are known:
/// `SnarkParams::setup` on both the prover and verifier sides.
pub const PRECISION_BITS_ARITH_CEILING: i32 = (Code::BITS / 2) as i32;

/// 1-D array of integer codes at a shared scale.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QArray1 {
    pub codes: Array1<Code>,
    pub scale: Scale,
}

/// 2-D array of integer codes at a shared scale.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QArray2 {
    pub codes: Array2<Code>,
    pub scale: Scale,
}

impl QArray1 {
    pub fn new(codes: Array1<Code>, scale: Scale) -> Self {
        Self { codes, scale }
    }
    pub fn len(&self) -> usize {
        self.codes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }
    pub fn to_real(&self) -> Array1<f64> {
        let inv = 1.0 / self.scale.to_real();
        self.codes.mapv(|c| c as f64 * inv)
    }
    pub fn add(&self, other: &Self) -> Result<Self, QArrayError> {
        if self.scale != other.scale {
            return Err(QArrayError::ScaleMismatch {
                lhs: self.scale,
                rhs: other.scale,
            });
        }
        if self.codes.len() != other.codes.len() {
            return Err(QArrayError::ShapeMismatch {
                lhs: vec![self.codes.len()],
                rhs: vec![other.codes.len()],
            });
        }
        Ok(Self {
            codes: &self.codes + &other.codes,
            scale: self.scale,
        })
    }
}

impl QArray2 {
    pub fn new(codes: Array2<Code>, scale: Scale) -> Self {
        Self { codes, scale }
    }
    pub fn nrows(&self) -> usize {
        self.codes.nrows()
    }
    pub fn ncols(&self) -> usize {
        self.codes.ncols()
    }
    pub fn to_real(&self) -> Array2<f64> {
        let inv = 1.0 / self.scale.to_real();
        self.codes.mapv(|c| c as f64 * inv)
    }
}

/// Quantize a real-valued vector at `scale` with banker's rounding.
pub fn quantize_vector(values: &Array1<f64>, scale: Scale) -> QArray1 {
    let codes =
        values.mapv(|v| crate::quantization::quantized_scalar::Qf::from_real(v, scale).code);
    QArray1 { codes, scale }
}

/// Outward-quantize a vector with floor rounding. Used for `x_lower`
/// so the dequantized lower bound is at most the real lower bound and
/// the quantized box strictly contains the original real box.
pub fn quantize_vector_floor(values: &Array1<f64>, scale: Scale) -> QArray1 {
    let codes =
        values.mapv(|v| crate::quantization::quantized_scalar::Qf::from_real_floor(v, scale).code);
    QArray1 { codes, scale }
}

/// Outward-quantize a vector with ceil rounding. Used for `x_upper`
/// so the dequantized upper bound is at least the real upper bound.
pub fn quantize_vector_ceil(values: &Array1<f64>, scale: Scale) -> QArray1 {
    let codes =
        values.mapv(|v| crate::quantization::quantized_scalar::Qf::from_real_ceil(v, scale).code);
    QArray1 { codes, scale }
}

/// Quantize a real-valued matrix at `scale` with banker's rounding.
pub fn quantize_matrix(values: &Array2<f64>, scale: Scale) -> QArray2 {
    let codes =
        values.mapv(|v| crate::quantization::quantized_scalar::Qf::from_real(v, scale).code);
    QArray2 { codes, scale }
}

/// Pick a power-of-two scale so the maximum-magnitude code fits just
/// under `precision_bits`. Power-of-two only keeps the chain-ReLU
/// working buffers simple; call sites that want a non-pow2 search can
/// use [`Scale::search`] directly.
pub fn pick_scale_pow2(values: &[f64], precision_bits: i32) -> Scale {
    let max_abs = values
        .iter()
        .copied()
        .map(f64::abs)
        .filter(|v| v.is_finite())
        .fold(0.0f64, f64::max);
    if max_abs == 0.0 {
        return Scale::from_pow2(precision_bits.saturating_sub(1));
    }
    // Largest `q` with `max_abs · 2^q < 2^{precision_bits}` is
    // `q < precision_bits - log2(max_abs)`. Floor and shave one bit of
    // slack to cover the `max_abs = 2^k` corner case.
    let log2_max = max_abs.log2();
    let q_real = (precision_bits as f64) - log2_max;
    let q = (q_real.floor() as i32) - 1;
    Scale::from_pow2(q)
}

/// Rescale every element of `arr` to `target` with banker's rounding.
/// Returns the rescaled array and the per-element witness table.
pub fn rescale_vector(
    arr: &QArray1,
    target: Scale,
) -> Result<(QArray1, Vec<RescaleEntry>), QArrayError> {
    rescale_vector_dir(
        arr,
        target,
        crate::quantization::quantized_scalar::RoundDir::HalfAway,
    )
}

/// Directional version of [`rescale_vector`]. Used by the cert generator
/// to emit `b_acc` / concretize rescales that keep the dequantized bound
/// a sound under-/over-approximation of the float CROWN value.
pub fn rescale_vector_dir(
    arr: &QArray1,
    target: Scale,
    dir: crate::quantization::quantized_scalar::RoundDir,
) -> Result<(QArray1, Vec<RescaleEntry>), QArrayError> {
    let mut codes = Array1::<Code>::zeros(arr.codes.len());
    let mut witnesses = Vec::with_capacity(arr.codes.len());
    for (i, &code) in arr.codes.iter().enumerate() {
        let qf = crate::quantization::quantized_scalar::Qf::new(code, arr.scale);
        let (out, w) = qf.rescale_dir(target, dir).map_err(QArrayError::Rescale)?;
        codes[i] = out.code;
        witnesses.push(w);
    }
    Ok((
        QArray1 {
            codes,
            scale: target,
        },
        witnesses,
    ))
}

/// Matrix analogue of [`rescale_vector`]. The witness table is
/// row-major flattened.
pub fn rescale_matrix(
    arr: &QArray2,
    target: Scale,
) -> Result<(QArray2, Vec<RescaleEntry>), QArrayError> {
    rescale_matrix_dir(
        arr,
        target,
        crate::quantization::quantized_scalar::RoundDir::HalfAway,
    )
}

/// Directional version of [`rescale_matrix`].
pub fn rescale_matrix_dir(
    arr: &QArray2,
    target: Scale,
    dir: crate::quantization::quantized_scalar::RoundDir,
) -> Result<(QArray2, Vec<RescaleEntry>), QArrayError> {
    let mut codes = Array2::<Code>::zeros((arr.nrows(), arr.ncols()));
    let mut witnesses = Vec::with_capacity(arr.codes.len());
    for ((i, j), &code) in arr.codes.indexed_iter() {
        let qf = crate::quantization::quantized_scalar::Qf::new(code, arr.scale);
        let (out, w) = qf.rescale_dir(target, dir).map_err(QArrayError::Rescale)?;
        codes[[i, j]] = out.code;
        witnesses.push(w);
    }
    Ok((
        QArray2 {
            codes,
            scale: target,
        },
        witnesses,
    ))
}

/// Per-element away-from-zero rounding for chain-matrix `A` rescales:
/// ceil for positives, floor for negatives, giving `|A| ≥ |A_exact|`.
/// This keeps both `A_pos · x_lower` and `A_pos · x_upper` on the sound
/// side. Each returned witness records the per-element direction.
pub fn rescale_matrix_away_from_zero(
    arr: &QArray2,
    target: Scale,
) -> Result<(QArray2, Vec<RescaleEntry>), QArrayError> {
    let mut codes = Array2::<Code>::zeros((arr.nrows(), arr.ncols()));
    let mut witnesses = Vec::with_capacity(arr.codes.len());
    for ((i, j), &code) in arr.codes.indexed_iter() {
        let qf = crate::quantization::quantized_scalar::Qf::new(code, arr.scale);
        let dir = if code >= 0 {
            crate::quantization::quantized_scalar::RoundDir::Ceil
        } else {
            crate::quantization::quantized_scalar::RoundDir::Floor
        };
        let (out, w) = qf.rescale_dir(target, dir).map_err(QArrayError::Rescale)?;
        codes[[i, j]] = out.code;
        witnesses.push(w);
    }
    Ok((
        QArray2 {
            codes,
            scale: target,
        },
        witnesses,
    ))
}

/// Accumulate-first matmul: `out[i,k] = sum_j A[i,j] * B[j,k]` in `i128`
/// with no per-product rescale. The result is at the composed scale
/// `s_a · s_b`; callers follow up with a single [`rescale_matrix`] per
/// output element.
pub fn matmul(a: &QArray2, b: &QArray2) -> Result<QArray2, QArrayError> {
    if a.ncols() != b.nrows() {
        return Err(QArrayError::ShapeMismatch {
            lhs: vec![a.nrows(), a.ncols()],
            rhs: vec![b.nrows(), b.ncols()],
        });
    }
    let composed = a
        .scale
        .compose(b.scale)
        .map_err(QArrayError::ScaleCompose)?;
    let mut codes = Array2::<Code>::zeros((a.nrows(), b.ncols()));
    for i in 0..a.nrows() {
        for k in 0..b.ncols() {
            let mut acc: Code = 0;
            for j in 0..a.ncols() {
                let prod = a.codes[[i, j]]
                    .checked_mul(b.codes[[j, k]])
                    .ok_or(QArrayError::OverflowOnMul)?;
                acc = acc.checked_add(prod).ok_or(QArrayError::OverflowOnAdd)?;
            }
            codes[[i, k]] = acc;
        }
    }
    Ok(QArray2 {
        codes,
        scale: composed,
    })
}

/// Matrix-vector analogue of [`matmul`]: `out[i] = sum_j A[i,j] * b[j]`
/// at the composed scale `s_a · s_b`.
pub fn matvec(a: &QArray2, b: &QArray1) -> Result<QArray1, QArrayError> {
    if a.ncols() != b.len() {
        return Err(QArrayError::ShapeMismatch {
            lhs: vec![a.nrows(), a.ncols()],
            rhs: vec![b.len()],
        });
    }
    let composed = a
        .scale
        .compose(b.scale)
        .map_err(QArrayError::ScaleCompose)?;
    let mut codes = Array1::<Code>::zeros(a.nrows());
    for i in 0..a.nrows() {
        let mut acc: Code = 0;
        for j in 0..a.ncols() {
            let prod = a.codes[[i, j]]
                .checked_mul(b.codes[j])
                .ok_or(QArrayError::OverflowOnMul)?;
            acc = acc.checked_add(prod).ok_or(QArrayError::OverflowOnAdd)?;
        }
        codes[i] = acc;
    }
    Ok(QArray1 {
        codes,
        scale: composed,
    })
}

#[derive(Debug, Error, PartialEq)]
pub enum QArrayError {
    #[error("QArray scale mismatch: lhs={lhs:?} rhs={rhs:?}")]
    ScaleMismatch { lhs: Scale, rhs: Scale },
    #[error("QArray shape mismatch: lhs={lhs:?} rhs={rhs:?}")]
    ShapeMismatch { lhs: Vec<usize>, rhs: Vec<usize> },
    #[error("QArray multiplication overflowed i128")]
    OverflowOnMul,
    #[error("QArray addition overflowed i128")]
    OverflowOnAdd,
    #[error("scale composition failed: {0}")]
    ScaleCompose(#[source] ScaleError),
    #[error("element-wise rescale failed: {0}")]
    Rescale(#[source] QfError),
}

#[cfg(test)]
mod tests {
    /// Test-local code width (the evaluation supplies real widths at runtime).
    const TEST_MAX_BITS: i32 = 18;
    use super::*;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn quantize_dequantize_round_trip_within_grid() {
        let v: Array1<f64> = array![0.0, 0.25, -1.5, 3.125];
        let s = Scale::from_pow2(8);
        let q = quantize_vector(&v, s);
        let back = q.to_real();
        for (orig, recon) in v.iter().zip(back.iter()) {
            assert!(
                (orig - recon).abs() <= 0.5 / s.to_real(),
                "drift exceeds half-grid: {orig} vs {recon}"
            );
        }
    }

    #[test]
    fn quantize_floor_rounds_toward_neg_infinity() {
        // Grid step = 1/8 at scale 8. 0.05 should floor to 0; -0.05
        // should floor to -1/8 (i.e., code -1); 0.625 = 5/8 (exact)
        // should remain 5; -0.625 should remain -5.
        let v: Array1<f64> = array![0.0, 0.05, -0.05, 0.625, -0.625, 0.99];
        let s = Scale::from_pow2(3); // = 8
        let q = quantize_vector_floor(&v, s);
        // floor(0 * 8) = 0
        assert_eq!(q.codes[0], 0);
        // floor(0.05 * 8) = floor(0.4) = 0
        assert_eq!(q.codes[1], 0);
        // floor(-0.05 * 8) = floor(-0.4) = -1
        assert_eq!(q.codes[2], -1);
        // floor(0.625 * 8) = 5
        assert_eq!(q.codes[3], 5);
        // floor(-0.625 * 8) = -5
        assert_eq!(q.codes[4], -5);
        // floor(0.99 * 8) = floor(7.92) = 7
        assert_eq!(q.codes[5], 7);
    }

    #[test]
    fn quantize_ceil_rounds_toward_pos_infinity() {
        let v: Array1<f64> = array![0.0, 0.05, -0.05, 0.625, -0.625, -0.99];
        let s = Scale::from_pow2(3);
        let q = quantize_vector_ceil(&v, s);
        assert_eq!(q.codes[0], 0);
        // ceil(0.4) = 1
        assert_eq!(q.codes[1], 1);
        // ceil(-0.4) = 0
        assert_eq!(q.codes[2], 0);
        // ceil(5.0) = 5
        assert_eq!(q.codes[3], 5);
        // ceil(-5.0) = -5
        assert_eq!(q.codes[4], -5);
        // ceil(-7.92) = -7
        assert_eq!(q.codes[5], -7);
    }

    #[test]
    fn outward_input_box_quantization_contains_real_box() {
        // Invariant: dequantized[x_lower] ≤ real x_lower and
        // dequantized[x_upper] ≥ real x_upper for every coordinate.
        let real_lower: Array1<f64> = array![-1.0, -0.5, 0.0, 0.123];
        let real_upper: Array1<f64> = array![1.0, 0.75, 0.5, 0.876];
        // Use a non-trivial scale where the real values do *not* land
        // on the integer grid (so rounding direction matters).
        let scale = Scale::from_pow2(4); // = 16
        let q_lower = quantize_vector_floor(&real_lower, scale);
        let q_upper = quantize_vector_ceil(&real_upper, scale);
        let inv = 1.0 / scale.to_real();
        for (i, real) in real_lower.iter().enumerate() {
            let recon = q_lower.codes[i] as f64 * inv;
            assert!(
                recon <= *real + 1e-12,
                "x_lower[{i}]: dequantized {recon} > real {real}",
            );
        }
        for (i, real) in real_upper.iter().enumerate() {
            let recon = q_upper.codes[i] as f64 * inv;
            assert!(
                recon >= *real - 1e-12,
                "x_upper[{i}]: dequantized {recon} < real {real}",
            );
        }
    }

    #[test]
    fn pick_scale_keeps_codes_under_precision() {
        let v: Array1<f64> = array![0.0, 0.5, -1.5, 3.0, -7.25];
        let scale = pick_scale_pow2(v.as_slice().unwrap(), TEST_MAX_BITS);
        let q = quantize_vector(&v, scale);
        let limit: Code = 1 << TEST_MAX_BITS;
        for &c in q.codes.iter() {
            assert!(c.abs() < limit, "code {c} exceeded precision_bits");
        }
    }

    #[test]
    fn pick_scale_handles_zero_array() {
        let v: Array1<f64> = array![0.0, 0.0, 0.0];
        let scale = pick_scale_pow2(v.as_slice().unwrap(), TEST_MAX_BITS);
        // Should still produce a valid scale (no panic / no infinity).
        assert!(scale.to_real().is_finite());
    }

    #[test]
    fn matmul_composes_scales_and_matches_float() {
        let a_real: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
        let b_real: Array2<f64> = array![[0.5, -0.25], [0.125, 0.75]];
        let s_a = Scale::from_pow2(4);
        let s_b = Scale::from_pow2(8);
        let a = quantize_matrix(&a_real, s_a);
        let b = quantize_matrix(&b_real, s_b);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.scale, Scale::from_pow2(12));
        let c_recon = c.to_real();
        let c_real = a_real.dot(&b_real);
        for ((i, j), &r) in c_recon.indexed_iter() {
            assert!(
                (c_real[[i, j]] - r).abs() < 1e-3,
                "matmul drift at ({i},{j}): {} vs {r}",
                c_real[[i, j]]
            );
        }
    }

    #[test]
    fn matmul_rejects_inner_dim_mismatch() {
        let a = quantize_matrix(&Array2::<f64>::zeros((2, 3)), Scale::from_pow2(0));
        let b = quantize_matrix(&Array2::<f64>::zeros((4, 2)), Scale::from_pow2(0));
        assert!(matches!(
            matmul(&a, &b),
            Err(QArrayError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn rescale_vector_records_one_witness_per_element() {
        let v: Array1<f64> = array![1.5, -2.0, 0.125];
        let s_in = Scale::from_pow2(8);
        let q = quantize_vector(&v, s_in);
        let s_out = Scale::from_pow2(4);
        let (rescaled, ws) = rescale_vector(&q, s_out).unwrap();
        assert_eq!(rescaled.scale, s_out);
        assert_eq!(ws.len(), v.len());
        for w in &ws {
            assert!(w.slack_lo >= 0 && w.slack_hi >= 0);
        }
    }

    #[test]
    fn matvec_matches_float() {
        let a_real: Array2<f64> = array![[1.0, -2.0, 0.5], [0.25, 1.0, -1.0]];
        let b_real: Array1<f64> = array![1.0, 0.5, 2.0];
        let s_a = Scale::from_pow2(8);
        let s_b = Scale::from_pow2(8);
        let a = quantize_matrix(&a_real, s_a);
        let b = quantize_vector(&b_real, s_b);
        let c = matvec(&a, &b).unwrap();
        let c_real = a_real.dot(&b_real);
        for (i, &r) in c.to_real().iter().enumerate() {
            assert!(
                (c_real[i] - r).abs() < 1e-3,
                "matvec drift at {i}: {} vs {r}",
                c_real[i]
            );
        }
    }
}
