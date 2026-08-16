//! Typed verifier-side accessor for per-layer `(c, e)` scale
//! parameters.
//!
//! Wraps a verified `LayerScalesCommit` (already opened and decoded
//! by `verify_layer_scale_opens`), runs an extra out-of-range check
//! on every exponent at construction, and hands out concrete
//! `Scale` values. Downstream gadgets consume `Scale` through this
//! accessor instead of indexing the raw per-class Vecs directly.

use crate::quantization::scale::Scale;

use crate::snark::errors::SnarkError;
use crate::snark::proof::{LayerScalesCommit, ScaleClass};

/// Sanity bound for `|e|` on any per-layer scale exponent. Honest
/// networks at `precision_bits ∈ [7, 16]` stay well under 64; the
/// looser `100` cap is wide enough to never reject an honest
/// configuration yet still rejects attacker-supplied exponents that
/// would overflow `1i128 << e` shifts downstream.
const SCALE_E_ABS_BOUND: i32 = 100;

/// Typed accessor over a verified `LayerScalesCommit`. Holds a
/// borrow of the verified commit; the only constructor runs the
/// extra range check, so every `Scale` returned is inside the
/// protocol's admissible exponent range.
pub struct LayerScaleAccessor<'a> {
    inner: &'a LayerScalesCommit,
    n_layers: usize,
}

impl<'a> LayerScaleAccessor<'a> {
    /// Wrap a verified `LayerScalesCommit` and range-check every
    /// per-layer exponent against `SCALE_E_ABS_BOUND`. Returns
    /// `Err(...)` on any out-of-range exponent or per-class Vec
    /// length mismatch.
    pub fn new(inner: &'a LayerScalesCommit) -> Result<Self, SnarkError> {
        let n_layers = inner.weight_c.len();
        if inner.weight_e.len() != n_layers
            || inner.bias_c.len() != n_layers
            || inner.bias_e.len() != n_layers
            || inner.relax_d_c.len() != n_layers
            || inner.relax_d_e.len() != n_layers
            || inner.relax_b_c.len() != n_layers
            || inner.relax_b_e.len() != n_layers
        {
            return Err(SnarkError::ArchitectureMismatch {
                what: "LayerScaleAccessor: per-class Vec length mismatch",
            });
        }
        let check_e = |e: i32| -> Result<(), SnarkError> {
            if e.abs() > SCALE_E_ABS_BOUND {
                return Err(SnarkError::ArchitectureMismatch {
                    what: "LayerScaleAccessor: scale exponent out of the |e| ≤ SCALE_E_ABS_BOUND range",
                });
            }
            Ok(())
        };
        for &e in inner
            .weight_e
            .iter()
            .chain(inner.bias_e.iter())
            .chain(inner.relax_d_e.iter())
            .chain(inner.relax_b_e.iter())
        {
            check_e(e)?;
        }
        Ok(Self { inner, n_layers })
    }

    /// Number of layers covered by the underlying commit.
    #[allow(dead_code)]
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Fetch the `Scale` for `(class, layer_idx)`. Returns
    /// `Err(...)` if `layer_idx` is out of range or if `class` is
    /// one of the auxiliary `E` indices.
    pub fn scale_for(&self, class: ScaleClass, layer_idx: usize) -> Result<Scale, SnarkError> {
        if layer_idx >= self.n_layers {
            return Err(SnarkError::ArchitectureMismatch {
                what: "LayerScaleAccessor::scale_for: layer_idx out of bounds",
            });
        }
        let (c, e) = match class {
            ScaleClass::WeightC => (
                self.inner.weight_c[layer_idx],
                self.inner.weight_e[layer_idx],
            ),
            ScaleClass::WeightE => {
                return Err(SnarkError::ArchitectureMismatch {
                    what: "LayerScaleAccessor::scale_for: pass a 'C' class (e.g. WeightC); the 'E' classes are auxiliary indices",
                });
            }
            ScaleClass::BiasC => (self.inner.bias_c[layer_idx], self.inner.bias_e[layer_idx]),
            ScaleClass::BiasE => {
                return Err(SnarkError::ArchitectureMismatch {
                    what: "LayerScaleAccessor::scale_for: pass a 'C' class",
                });
            }
            ScaleClass::RelaxDC => (
                self.inner.relax_d_c[layer_idx],
                self.inner.relax_d_e[layer_idx],
            ),
            ScaleClass::RelaxDE => {
                return Err(SnarkError::ArchitectureMismatch {
                    what: "LayerScaleAccessor::scale_for: pass a 'C' class",
                });
            }
            ScaleClass::RelaxBC => (
                self.inner.relax_b_c[layer_idx],
                self.inner.relax_b_e[layer_idx],
            ),
            ScaleClass::RelaxBE => {
                return Err(SnarkError::ArchitectureMismatch {
                    what: "LayerScaleAccessor::scale_for: pass a 'C' class",
                });
            }
        };
        Ok(Scale { c, e })
    }

    /// Per-layer weight scale.
    #[allow(dead_code)]
    pub fn weight_scale(&self, layer_idx: usize) -> Result<Scale, SnarkError> {
        self.scale_for(ScaleClass::WeightC, layer_idx)
    }

    /// Per-layer bias scale.
    #[allow(dead_code)]
    pub fn bias_scale(&self, layer_idx: usize) -> Result<Scale, SnarkError> {
        self.scale_for(ScaleClass::BiasC, layer_idx)
    }

    /// Per-layer relaxation `d` scale.
    pub fn relax_d_scale(&self, layer_idx: usize) -> Result<Scale, SnarkError> {
        self.scale_for(ScaleClass::RelaxDC, layer_idx)
    }

    /// Per-layer relaxation `b` scale.
    pub fn relax_b_scale(&self, layer_idx: usize) -> Result<Scale, SnarkError> {
        self.scale_for(ScaleClass::RelaxBC, layer_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_n(n: usize) -> LayerScalesCommit {
        LayerScalesCommit {
            weight_c: vec![1; n],
            weight_e: vec![0; n],
            bias_c: vec![1; n],
            bias_e: vec![0; n],
            relax_d_c: vec![1; n],
            relax_d_e: vec![0; n],
            relax_b_c: vec![1; n],
            relax_b_e: vec![0; n],
        }
    }

    #[test]
    fn accessor_returns_scale_for_valid_inputs() {
        let inner = empty_n(3);
        let acc = LayerScaleAccessor::new(&inner).unwrap();
        let s = acc.weight_scale(2).unwrap();
        assert_eq!(s.c, 1);
        assert_eq!(s.e, 0);
    }

    #[test]
    fn accessor_rejects_out_of_range_exponent() {
        let mut inner = empty_n(3);
        inner.relax_d_e[1] = SCALE_E_ABS_BOUND + 1;
        let r = LayerScaleAccessor::new(&inner);
        assert!(matches!(r, Err(SnarkError::ArchitectureMismatch { .. })));
    }

    #[test]
    fn accessor_rejects_negative_out_of_range_exponent() {
        let mut inner = empty_n(2);
        inner.bias_e[0] = -(SCALE_E_ABS_BOUND + 1);
        let r = LayerScaleAccessor::new(&inner);
        assert!(matches!(r, Err(SnarkError::ArchitectureMismatch { .. })));
    }

    #[test]
    fn accessor_rejects_out_of_bounds_layer_idx() {
        let inner = empty_n(2);
        let acc = LayerScaleAccessor::new(&inner).unwrap();
        let r = acc.weight_scale(5);
        assert!(matches!(r, Err(SnarkError::ArchitectureMismatch { .. })));
    }

    #[test]
    fn accessor_rejects_e_class_in_scale_for() {
        let inner = empty_n(2);
        let acc = LayerScaleAccessor::new(&inner).unwrap();
        let r = acc.scale_for(ScaleClass::WeightE, 0);
        assert!(matches!(r, Err(SnarkError::ArchitectureMismatch { .. })));
    }

    #[test]
    fn accessor_rejects_mismatched_vec_lengths() {
        let mut inner = empty_n(3);
        inner.weight_c.push(0);
        let r = LayerScaleAccessor::new(&inner);
        assert!(matches!(r, Err(SnarkError::ArchitectureMismatch { .. })));
    }
}
