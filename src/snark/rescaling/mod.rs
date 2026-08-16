//! Rescale gadget. Binds a pre-rescale tensor `qx` to a post-rescale
//! tensor `qz` via a boxed-inequality identity on a per-cell
//! `slack_lo` and a LogUp range check `slack_lo ⊆ [0, 2c2)`.
//!
//! The verifier opens `(qx, qz, slack_lo)` at a Fiat-Shamir random
//! `r` and checks the linear identity; `slack_hi ≥ 0` follows from
//! the range on `slack_lo`.
//!
//! Submodules: [`prove`] / [`verify`] hold the per-event prover and
//! verifier; this facade exposes the proof types and helpers
//! ([`absorb_commitment`], [`build_range_table`],
//! [`build_multiplicities`], [`top_halves`]). The [`driver`] module
//! walks a backward pass and applies the per-event prover/verifier
//! in canonical order.

mod prove;
mod verify;

#[cfg(test)]
mod tests;

pub mod driver;

pub use prove::prove_rescale_event;
pub use verify::verify_rescale_event;

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::snark_primitives::logup_gkr::{LogUpCircuit, LogUpLayer, LogUpProof};
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

/// Public description of one rescale event in a backward pass.
///
/// `c1`/`c2` are the boxed-inequality coefficients
/// (`c1/c2 = s_y · s_out / s_in`); `dir` selects the rounding mode
/// for `qz` (Floor/Ceil for Lower/Upper b_acc rescales, HalfAway
/// elsewhere). The verifier picks the slack identity offset to match.
#[derive(Clone, Debug)]
pub struct RescaleEventDesc {
    pub c1: i128,
    pub c2: i128,
    /// MLE variables for the cell index of this event.
    pub n_vars: usize,
    pub dir: crate::quantization::quantized_scalar::RoundDir,
}

/// Per-event rescale proof.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct RescaleEventProof {
    pub c1_fr: Fr,
    pub c2_fr: Fr,
    pub n_vars: usize,
    /// `RoundDir::tag()` selecting the slack-identity offset.
    pub dir_tag: u8,
    pub slack_lo_commit: <HyraxBn254 as MlPcs>::Commitment,
    /// LogUp range proof for `slack_lo ⊆ [0, 2c2)`.
    pub logup_alpha: Fr,
    pub logup_beta: Fr,
    pub lookup_proof: LogUpProof<Fr>,
    pub table_proof: LogUpProof<Fr>,
    pub lookup_top: [Fr; 4],
    pub table_top: [Fr; 4],
    pub lookup_n_vars: usize,
    pub table_n_vars: usize,
    pub witness_len: usize,
    pub table_len: usize,
    /// Open binding `lookup.bottom_denom = slack_lo(r) − β`.
    pub slack_lo_logup_open: <HyraxBn254 as MlPcs>::Proof,
    pub slack_lo_logup_eval: Fr,
    pub r_identity: Vec<Fr>,
    /// Batched Hyrax open at `r_identity` for `(qx, qz, slack_lo)`.
    pub r_identity_open: <HyraxBn254 as MlPcs>::Proof,
    pub qx_eval: Fr,
    pub qz_eval: Fr,
    pub slack_lo_eval: Fr,
    /// Multiplicity commit (absorbed before β); verifier checks the
    /// open at `table_proof.bottom_point` matches `bottom_num`.
    pub mult_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub mult_open: <HyraxBn254 as MlPcs>::Proof,
    pub mult_n_vars: usize,
}

/// Absorb a Hyrax commitment into the FS sponge by canonical-serialising it.
pub(crate) fn absorb_commitment(
    sponge: &mut impl CryptographicSponge,
    commitment: &<HyraxBn254 as MlPcs>::Commitment,
) {
    let mut buf = Vec::new();
    commitment
        .serialize_compressed(&mut buf)
        .expect("serialize commitment");
    sponge.absorb(&buf);
}

/// Build the table `T = {0, 1, ..., 2c2 − 1}` lifted to `Fr`, padded
/// with zeros up to the next power of two.
pub(crate) fn build_range_table(two_c2: u128) -> Vec<Fr> {
    let len = two_c2 as usize;
    let target = len.next_power_of_two().max(2);
    let mut t = Vec::with_capacity(target);
    for v in 0..len {
        t.push(Fr::from(v as u64));
    }
    while t.len() < target {
        t.push(Fr::from(0u64));
    }
    t
}

/// Multiplicity vector counting occurrences of each value of
/// `slack_lo` in `[0, 2c2)`. Out-of-range cells are dropped; the
/// LogUp identity then fails on them.
pub(crate) fn build_multiplicities(slack_lo_padded: &[i128], two_c2: u128) -> Vec<u64> {
    let table_len = (two_c2 as usize).next_power_of_two().max(2);
    let mut mults = vec![0u64; table_len];
    for &v in slack_lo_padded {
        if v >= 0 && (v as u128) < two_c2 {
            mults[v as usize] += 1;
        }
    }
    mults
}

/// Extract the four field elements making up the top fractions of a
/// LogUp circuit's last layer.
pub(crate) fn top_halves(circuit: &LogUpCircuit<Fr>) -> [Fr; 4] {
    use ark_ff::One;
    let top = circuit.layers.last().expect("non-empty");
    match top {
        LogUpLayer::Generic {
            numerator,
            denominator,
        } => [numerator[0], numerator[1], denominator[0], denominator[1]],
        LogUpLayer::InitialLookup { denominator } => {
            [-Fr::one(), -Fr::one(), denominator[0], denominator[1]]
        }
        LogUpLayer::InitialTable {
            numerator,
            denominator,
        } => [numerator[0], numerator[1], denominator[0], denominator[1]],
    }
}
