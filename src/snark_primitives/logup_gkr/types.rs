//! LogUp-GKR data structures: per-position fraction, layered circuit,
//! per-layer sumcheck output, full proof envelope. Also the shared
//! `absorb_round_poly` transcript helper.

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use thiserror::Error;

use crate::snark_primitives::sumcheck::RoundPoly3;

/// Un-reduced rational `numerator / denominator`. The value at every
/// node of the LogUp circuit.
#[derive(Copy, Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Fraction<F: Field> {
    pub numerator: F,
    pub denominator: F,
}

impl<F: Field> Fraction<F> {
    pub fn new(numerator: F, denominator: F) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    pub fn as_tuple(&self) -> (F, F) {
        (self.numerator, self.denominator)
    }
}

impl<F: Field> std::ops::Add for Fraction<F> {
    type Output = Self;

    /// `(a/b) + (c/d) = (a·d + b·c) / (b·d)`. No reduction.
    fn add(self, other: Self) -> Self {
        Self {
            numerator: self.numerator * other.denominator + self.denominator * other.numerator,
            denominator: self.denominator * other.denominator,
        }
    }
}

/// One layer of a LogUp-GKR circuit. The bottom layer is either
/// `InitialLookup` (numerators implicitly `-1`) or `InitialTable`
/// (numerators are the multiplicities); every layer above is `Generic`.
#[derive(Clone, Debug)]
pub enum LogUpLayer<F: Field> {
    Generic {
        numerator: Vec<F>,
        denominator: Vec<F>,
    },
    InitialLookup {
        denominator: Vec<F>,
    },
    InitialTable {
        numerator: Vec<F>,
        denominator: Vec<F>,
    },
}

impl<F: Field> LogUpLayer<F> {
    /// Number of variables in the MLE of each half of this layer. Equals
    /// `log2(len/2)`: this layer's vectors split into a low and a high
    /// half of `2^num_vars()` entries each.
    pub fn num_vars(&self) -> usize {
        let n = match self {
            LogUpLayer::Generic { denominator, .. }
            | LogUpLayer::InitialTable { denominator, .. }
            | LogUpLayer::InitialLookup { denominator } => denominator.len(),
        };
        debug_assert!(n.is_power_of_two() && n >= 2, "denominator len = {n}");
        (n / 2).trailing_zeros() as usize
    }

    pub fn is_top(&self) -> bool {
        self.num_vars() == 0
    }

    /// Build the next layer up by pairwise-adding low- and high-half
    /// fractions. Returns `None` if this layer is already the size-2 top.
    pub fn merge_up(&self) -> Option<LogUpLayer<F>> {
        if self.is_top() {
            return None;
        }
        let half = 1usize << self.num_vars();
        let next = match self {
            LogUpLayer::Generic {
                numerator,
                denominator,
            }
            | LogUpLayer::InitialTable {
                numerator,
                denominator,
            } => {
                let (n_lo, n_hi) = numerator.split_at(half);
                let (d_lo, d_hi) = denominator.split_at(half);
                let mut next_num = Vec::with_capacity(half);
                let mut next_den = Vec::with_capacity(half);
                for i in 0..half {
                    let f = Fraction::new(n_lo[i], d_lo[i]) + Fraction::new(n_hi[i], d_hi[i]);
                    next_num.push(f.numerator);
                    next_den.push(f.denominator);
                }
                LogUpLayer::Generic {
                    numerator: next_num,
                    denominator: next_den,
                }
            }
            LogUpLayer::InitialLookup { denominator } => {
                let (d_lo, d_hi) = denominator.split_at(half);
                let mut next_num = Vec::with_capacity(half);
                let mut next_den = Vec::with_capacity(half);
                let neg_one = -F::one();
                for i in 0..half {
                    let f = Fraction::new(neg_one, d_lo[i]) + Fraction::new(neg_one, d_hi[i]);
                    next_num.push(f.numerator);
                    next_den.push(f.denominator);
                }
                LogUpLayer::Generic {
                    numerator: next_num,
                    denominator: next_den,
                }
            }
        };
        Some(next)
    }

    /// Split this layer into `[n_lo, n_hi, d_lo, d_hi]` for the per-layer
    /// sumcheck. `InitialLookup` expands its implicit `-1` numerators
    /// into explicit vectors so prover/verifier code stays uniform.
    pub fn into_halves(&self) -> [Vec<F>; 4] {
        let half = 1usize << self.num_vars();
        match self {
            LogUpLayer::Generic {
                numerator,
                denominator,
            }
            | LogUpLayer::InitialTable {
                numerator,
                denominator,
            } => {
                let (n_lo, n_hi) = numerator.split_at(half);
                let (d_lo, d_hi) = denominator.split_at(half);
                [n_lo.to_vec(), n_hi.to_vec(), d_lo.to_vec(), d_hi.to_vec()]
            }
            LogUpLayer::InitialLookup { denominator } => {
                let (d_lo, d_hi) = denominator.split_at(half);
                let neg_one = -F::one();
                [
                    vec![neg_one; half],
                    vec![neg_one; half],
                    d_lo.to_vec(),
                    d_hi.to_vec(),
                ]
            }
        }
    }
}

/// Layered LogUp circuit. `layers[0]` is the initial bottom layer,
/// `layers.last()` the size-2 top.
#[derive(Clone, Debug)]
pub struct LogUpCircuit<F: Field> {
    pub layers: Vec<LogUpLayer<F>>,
}

impl<F: Field> LogUpCircuit<F> {
    /// Build a circuit from an initial layer by repeatedly merging up.
    pub fn new(initial: LogUpLayer<F>) -> Result<Self, LogUpError> {
        let mut layers = vec![initial];
        loop {
            let last = layers.last().expect("non-empty");
            match last.merge_up() {
                Some(merged) => layers.push(merged),
                None => break,
            }
        }
        Ok(Self { layers })
    }

    /// Build the lookup-side (witness) circuit: `denominator[i] = w[i] - α`,
    /// numerators implicit `-1`.
    pub fn lookup(witness: &[F], alpha: F) -> Result<Self, LogUpError> {
        if witness.is_empty() || !witness.len().is_power_of_two() {
            return Err(LogUpError::NonPowerOfTwoLen { len: witness.len() });
        }
        let denominator: Vec<F> = witness.iter().map(|w| *w - alpha).collect();
        Self::new(LogUpLayer::InitialLookup { denominator })
    }

    /// Build the table-side circuit: `numerator[j] = m[j]`,
    /// `denominator[j] = t[j] - α`.
    pub fn table(table: &[F], multiplicities: &[F], alpha: F) -> Result<Self, LogUpError> {
        if table.len() != multiplicities.len() {
            return Err(LogUpError::ShapeMismatch {
                table: table.len(),
                mults: multiplicities.len(),
            });
        }
        if table.is_empty() || !table.len().is_power_of_two() {
            return Err(LogUpError::NonPowerOfTwoLen { len: table.len() });
        }
        let denominator: Vec<F> = table.iter().map(|t| *t - alpha).collect();
        Self::new(LogUpLayer::InitialTable {
            numerator: multiplicities.to_vec(),
            denominator,
        })
    }

    /// Aggregated top-of-circuit fraction.
    pub fn output(&self) -> Fraction<F> {
        let top = self.layers.last().expect("≥ 1 layer");
        let (n0, n1, d0, d1) = match top {
            LogUpLayer::Generic {
                numerator,
                denominator,
            } => {
                debug_assert_eq!(numerator.len(), 2);
                debug_assert_eq!(denominator.len(), 2);
                (numerator[0], numerator[1], denominator[0], denominator[1])
            }
            LogUpLayer::InitialLookup { denominator } => {
                debug_assert_eq!(denominator.len(), 2);
                (-F::one(), -F::one(), denominator[0], denominator[1])
            }
            LogUpLayer::InitialTable {
                numerator,
                denominator,
            } => {
                debug_assert_eq!(numerator.len(), 2);
                debug_assert_eq!(denominator.len(), 2);
                (numerator[0], numerator[1], denominator[0], denominator[1])
            }
        };
        Fraction::new(n0 * d1 + n1 * d0, d0 * d1)
    }

    pub fn num_vars(&self) -> usize {
        self.layers.first().expect("≥ 1 layer").num_vars()
    }
}

/// One layer's sumcheck output: the round polynomials plus the four
/// final MLE evaluations `[num_lo, num_hi, denom_lo, denom_hi](r)`.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct LayerProof<F: Field> {
    pub rounds: Vec<RoundPoly3<F>>,
    pub final_evals: [F; 4],
}

/// LogUp-GKR proof of a single circuit.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct LogUpProof<F: Field> {
    /// Top-of-circuit fraction. Initial claim for the verifier.
    pub top_numerator: F,
    pub top_denominator: F,
    /// One sumcheck per layer transition, top→bottom. Length equals
    /// `circuit.num_vars()`.
    pub layers: Vec<LayerProof<F>>,
    /// Final `(num, denom)` claims at the bottom-layer challenge point;
    /// the SNARK driver wires these to PCS openings.
    pub bottom_num: F,
    pub bottom_denom: F,
    /// Bottom-layer challenge point at which `bottom_num` /
    /// `bottom_denom` are evaluated. Length `circuit.num_vars()`.
    pub bottom_point: Vec<F>,
}

#[derive(Debug, Error, PartialEq)]
pub enum LogUpError {
    #[error("input length {len} is not a power of two ≥ 2")]
    NonPowerOfTwoLen { len: usize },
    #[error("table/multiplicities length mismatch: table={table}, mults={mults}")]
    ShapeMismatch { table: usize, mults: usize },
    #[error("layer count mismatch in proof: expected {expected}, got {got}")]
    LayerCountMismatch { expected: usize, got: usize },
    #[error("per-layer sumcheck split check failed at layer {layer}")]
    SumcheckSplitMismatch { layer: usize },
    #[error("final round-poly value didn't match the claimed combination")]
    SumcheckFinalMismatch { layer: usize },
    #[error("top-of-circuit fraction inconsistent with proof claims")]
    TopFractionMismatch,
}

/// Absorb a degree-3 round polynomial and squeeze a challenge. Must be
/// called in the same transcript position on prover and verifier sides.
pub(super) fn absorb_round_poly<F, S>(sponge: &mut S, p: &RoundPoly3<F>) -> F
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
{
    sponge.absorb(&p.at_zero);
    sponge.absorb(&p.at_one);
    sponge.absorb(&p.at_two);
    sponge.absorb(&p.at_three);
    sponge.squeeze_field_elements::<F>(1)[0]
}
