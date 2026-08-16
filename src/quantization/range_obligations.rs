//! Range-check obligations collected from a quantized cert.
//!
//! Every value the SNARK exposes as a "small signed integer" needs an
//! explicit range proof when the cert is lifted into `Fr`, otherwise a
//! malicious prover could stash an `Fr` element above `r/2` that decodes
//! as a small negative integer.
//!
//! Two kinds of obligation:
//!
//! * [`RangeKind::Signed`] — `|value| < 2^{bits - 1}`. Every code in the
//!   cert (weights, biases, relaxation slopes, working A / b_acc, target
//!   bounds) is signed at the caller-supplied code width.
//! * [`RangeKind::Unsigned`] — `value ∈ [0, 2^bits)`. Every rescale
//!   `slack_lo` / `slack_hi` is unsigned, reusing the same 16-bit table
//!   as the codes.
//!
//! Lasso wiring sits in a later step; this file just collects the typed
//! list so a tampered cert can be detected by replaying each obligation.

use serde::{Deserialize, Serialize};

use crate::quantized_crown::QuantCert;
use crate::snark_primitives::finite_field::{fr_to_signed_i128, signed_lift_to_fr, SerFr};

/// Single range-check obligation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RangeObligation {
    pub value: SerFr,
    pub kind: RangeKind,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RangeKind {
    /// `|signed_lift(value)| < 2^{bits - 1}`.
    Signed { bits: u32 },
    /// `signed_lift(value)` is non-negative and `< 2^bits`.
    Unsigned { bits: u32 },
}

impl RangeObligation {
    /// True iff the obligation holds for its `value`. The SNARK side
    /// replaces this with a Lasso lookup; in tests it acts as the
    /// reference oracle for tamper checks.
    pub fn check(&self) -> bool {
        let Some(signed) = fr_to_signed_i128(self.value.0) else {
            return false;
        };
        match self.kind {
            RangeKind::Signed { bits } => {
                if bits == 0 {
                    return signed == 0;
                }
                let limit: i128 = 1i128 << (bits - 1);
                signed > -limit && signed < limit
            }
            RangeKind::Unsigned { bits } => {
                if signed < 0 {
                    return false;
                }
                if bits >= 127 {
                    return true;
                }
                let limit: i128 = 1i128 << bits;
                signed < limit
            }
        }
    }
}

/// Collect every range-check obligation a quantized cert imposes on the
/// SNARK gadget.
///
/// Order is deterministic: codes first (input box, weights, biases,
/// relaxations, target bounds), then rescale slacks in emission order.
/// `bits` is the signed/unsigned obligation width — the caller supplies
/// it from the runtime quantization parameters (the SNARK's authoritative
/// range checks are the per-tensor LogUps against the runtime table; this
/// module is a reference oracle used by its own tests).
pub fn collect_range_obligations(cert: &QuantCert, bits: u32) -> Vec<RangeObligation> {
    let mut out: Vec<RangeObligation> = Vec::new();
    let push_signed = |out: &mut Vec<RangeObligation>, code: i128| {
        out.push(RangeObligation {
            value: SerFr(signed_lift_to_fr(code)),
            kind: RangeKind::Signed { bits },
        });
    };
    let push_unsigned = |out: &mut Vec<RangeObligation>, code: i128| {
        out.push(RangeObligation {
            value: SerFr(signed_lift_to_fr(code)),
            kind: RangeKind::Unsigned { bits },
        });
    };

    for code in cert.x_lower.codes.iter() {
        push_signed(&mut out, *code);
    }
    for code in cert.x_upper.codes.iter() {
        push_signed(&mut out, *code);
    }
    for w in cert.weights.iter().flatten() {
        for code in w.codes.iter() {
            push_signed(&mut out, *code);
        }
    }
    for b in cert.biases.iter().flatten() {
        for code in b.codes.iter() {
            push_signed(&mut out, *code);
        }
    }
    for rel in cert.relaxations.iter().flatten() {
        for code in rel.d_lower.codes.iter() {
            push_signed(&mut out, *code);
        }
        for code in rel.d_upper.codes.iter() {
            push_signed(&mut out, *code);
        }
        for code in rel.b_lower.codes.iter() {
            push_signed(&mut out, *code);
        }
        for code in rel.b_upper.codes.iter() {
            push_signed(&mut out, *code);
        }
    }
    if let Some(t) = &cert.target_lower {
        for code in t.codes.iter() {
            push_signed(&mut out, *code);
        }
    }
    if let Some(t) = &cert.target_upper {
        for code in t.codes.iter() {
            push_signed(&mut out, *code);
        }
    }
    for w in &cert.witnesses {
        push_unsigned(&mut out, w.slack_lo);
        push_unsigned(&mut out, w.slack_hi);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crown::network::Layer;
    use crate::crown::network::Network;
    use crate::crown::output_property::{Property, Side};
    use crate::quantized_crown::quantized_backward_bound;
    use ndarray::{array, Array1, Array2};

    fn small_relu_net() -> Network {
        let w1: Array2<f64> = array![[1.0, 2.0], [-1.0, 1.0], [0.5, -0.5]];
        let b1: Array1<f64> = array![0.0, 0.5, -0.25];
        let w2: Array2<f64> = array![[1.0, -1.0, 2.0], [0.0, 1.0, 1.0]];
        let b2: Array1<f64> = array![0.1, -0.2];
        Network::new(vec![
            Layer::linear(w1, b1).unwrap(),
            Layer::relu(),
            Layer::linear(w2, b2).unwrap(),
        ])
        .unwrap()
    }

    fn cert() -> QuantCert {
        let net = small_relu_net();
        let prop = Property::new(
            Array2::eye(net.output_dim()),
            Array1::zeros(net.output_dim()),
            Side::Both,
        )
        .unwrap();
        let x_l = array![-1.0, -0.5];
        let x_u = array![1.0, 0.75];
        quantized_backward_bound(&net, &prop, &x_l, &x_u, 14).unwrap()
    }

    #[test]
    fn obligations_pass_on_honest_cert() {
        let c = cert();
        let obs = collect_range_obligations(&c, 18);
        assert!(!obs.is_empty(), "expected at least one obligation");
        for (i, ob) in obs.iter().enumerate() {
            assert!(ob.check(), "obligation {i} failed: {ob:?}");
        }
    }

    #[test]
    fn obligations_count_matches_cert_shape() {
        let c = cert();
        let obs = collect_range_obligations(&c, 18);
        // Codes per tensor type, then 2 slacks per witness.
        let weight_codes: usize = c.weights.iter().flatten().map(|w| w.codes.len()).sum();
        let bias_codes: usize = c.biases.iter().flatten().map(|b| b.codes.len()).sum();
        let relax_codes: usize = c
            .relaxations
            .iter()
            .flatten()
            .map(|r| r.d_lower.len() + r.d_upper.len() + r.b_lower.len() + r.b_upper.len())
            .sum();
        let target_codes: usize = c.target_lower.as_ref().map(|v| v.len()).unwrap_or(0)
            + c.target_upper.as_ref().map(|v| v.len()).unwrap_or(0);
        let box_codes = c.x_lower.len() + c.x_upper.len();
        let slack_count = c.witnesses.len() * 2;
        let expected =
            weight_codes + bias_codes + relax_codes + target_codes + box_codes + slack_count;
        assert_eq!(obs.len(), expected, "obligation count drifted");
    }

    #[test]
    fn tampered_code_breaks_obligation() {
        let c = cert();
        let mut obs = collect_range_obligations(&c, 18);
        // Tamper one signed obligation by lifting an out-of-range code.
        let big_code: i128 = 1i128 << 18;
        obs[0] = RangeObligation {
            value: SerFr(signed_lift_to_fr(big_code)),
            kind: RangeKind::Signed {
                bits: 18,
            },
        };
        let bad = obs.iter().filter(|o| !o.check()).count();
        assert_eq!(bad, 1);
    }

    #[test]
    fn negative_unsigned_breaks_obligation() {
        let bad = RangeObligation {
            value: SerFr(signed_lift_to_fr(-1)),
            kind: RangeKind::Unsigned { bits: 16 },
        };
        assert!(!bad.check());
    }

    #[test]
    fn obligation_round_trips_through_json() {
        let ob = RangeObligation {
            value: SerFr(signed_lift_to_fr(-12345)),
            kind: RangeKind::Signed { bits: 16 },
        };
        let s = serde_json::to_string(&ob).unwrap();
        let back: RangeObligation = serde_json::from_str(&s).unwrap();
        assert_eq!(ob, back);
    }
}
