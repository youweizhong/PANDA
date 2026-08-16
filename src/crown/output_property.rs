//! Output linear property `C @ y + d` and which side(s) of it the
//! certificate must bound.
//!
//! The full matrix `C` enters CROWN in a single backward run rather than
//! one row at a time, which is what lets the prover certify a vector
//! property (e.g. all class-margin scores at once) with one sweep.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Which side(s) of `C @ y + d` the certificate must bound.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Lower,
    Upper,
    Both,
}

impl Side {
    pub fn needs_lower(self) -> bool {
        matches!(self, Side::Lower | Side::Both)
    }
    pub fn needs_upper(self) -> bool {
        matches!(self, Side::Upper | Side::Both)
    }
}

/// Output linear property the prover wants to certify.
///
/// The certified statement has the form `C·y + d ≥ lower_threshold` and/or
/// `C·y + d ≤ upper_threshold`, depending on `side`. The optional
/// thresholds let the in-SNARK property check compare the (private)
/// claimed bound against a public threshold rather than revealing the
/// bound itself; both default to zero.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Property {
    pub c_matrix: Array2<f64>,
    pub d_vector: Array1<f64>,
    pub side: Side,
    #[serde(default)]
    pub lower_threshold: Option<Array1<f64>>,
    #[serde(default)]
    pub upper_threshold: Option<Array1<f64>>,
}

impl Property {
    /// Construct a property with default (zero) thresholds.
    ///
    /// Errors if `C.nrows() != d.len()` or if `C` is empty in either
    /// dimension.
    pub fn new(
        c_matrix: Array2<f64>,
        d_vector: Array1<f64>,
        side: Side,
    ) -> Result<Self, PropertyError> {
        if c_matrix.nrows() != d_vector.len() {
            return Err(PropertyError::ShapeMismatch {
                rows: c_matrix.nrows(),
                d_len: d_vector.len(),
            });
        }
        if c_matrix.nrows() == 0 || c_matrix.ncols() == 0 {
            return Err(PropertyError::Empty);
        }
        Ok(Self {
            c_matrix,
            d_vector,
            side,
            lower_threshold: None,
            upper_threshold: None,
        })
    }

    /// Construct a property with explicit per-direction thresholds.
    ///
    /// Each provided threshold must have length `C.nrows()`.
    pub fn new_with_thresholds(
        c_matrix: Array2<f64>,
        d_vector: Array1<f64>,
        side: Side,
        lower_threshold: Option<Array1<f64>>,
        upper_threshold: Option<Array1<f64>>,
    ) -> Result<Self, PropertyError> {
        let mut p = Self::new(c_matrix, d_vector, side)?;
        if let Some(ref t) = lower_threshold {
            if t.len() != p.c_matrix.nrows() {
                return Err(PropertyError::ShapeMismatch {
                    rows: p.c_matrix.nrows(),
                    d_len: t.len(),
                });
            }
        }
        if let Some(ref t) = upper_threshold {
            if t.len() != p.c_matrix.nrows() {
                return Err(PropertyError::ShapeMismatch {
                    rows: p.c_matrix.nrows(),
                    d_len: t.len(),
                });
            }
        }
        p.lower_threshold = lower_threshold;
        p.upper_threshold = upper_threshold;
        Ok(p)
    }

    pub fn n_spec(&self) -> usize {
        self.c_matrix.nrows()
    }

    pub fn output_dim(&self) -> usize {
        self.c_matrix.ncols()
    }

    /// Resolve the lower-direction threshold (defaults to zeros).
    pub fn lower_threshold_or_zero(&self) -> Array1<f64> {
        self.lower_threshold
            .clone()
            .unwrap_or_else(|| Array1::zeros(self.c_matrix.nrows()))
    }

    /// Resolve the upper-direction threshold (defaults to zeros).
    pub fn upper_threshold_or_zero(&self) -> Array1<f64> {
        self.upper_threshold
            .clone()
            .unwrap_or_else(|| Array1::zeros(self.c_matrix.nrows()))
    }
}

/// Errors raised while constructing a [`Property`].
#[derive(Debug, Error)]
pub enum PropertyError {
    #[error("C has {rows} rows but d has length {d_len}")]
    ShapeMismatch { rows: usize, d_len: usize },
    #[error("property matrix must be non-empty")]
    Empty,
}
