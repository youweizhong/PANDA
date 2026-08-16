//! Hyrax PCS helpers wrapping the primitive open / verify routines.
//!
//! Provides a per-commit `hyrax_open_at` / `hyrax_verify_at` pair
//! that lifts a small big-endian point up to the commit's native
//! `n_vars`, plus a batched-at-one-point variant
//! ([`hyrax_open_batched_at`] / [`hyrax_verify_batched_at`]) that
//! produces a single proof for the random-linear-combination commit
//! `C_batch = Σ_k ρ^k · C_k` — saving `K − 1` opens when several
//! polynomials share an evaluation point.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::One;
use ark_std::rand::RngCore;

use crate::snark_primitives::hyrax_pcs;
use crate::snark_primitives::logup_gkr::{LogUpCircuit, LogUpLayer};
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use crate::snark::commitment::commit::CommittedAux;
use crate::snark::errors::SnarkError;

/// Open a previously committed padded poly at a small BE point.
/// The point is lifted to the commit's native `n_vars` (derived
/// from `aux.0.len()`) and reversed for arkworks's LE convention.
pub(crate) fn hyrax_open_at(
    ck: &<HyraxBn254 as MlPcs>::CommitterKey,
    aux: &CommittedAux,
    commitment: &<HyraxBn254 as MlPcs>::Commitment,
    point_be: &[Fr],
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<(Fr, <HyraxBn254 as MlPcs>::Proof), SnarkError> {
    if !aux.0.len().is_power_of_two() || aux.0.is_empty() {
        return Err(SnarkError::ShapeMismatch {
            what: "hyrax_open_at: aux MLE length must be a non-zero power of two",
        });
    }
    let n_vars = aux.0.len().trailing_zeros() as usize;
    if point_be.len() > n_vars {
        return Err(SnarkError::ShapeMismatch {
            what: "hyrax_open_at: opening point has more vars than the commit",
        });
    }
    let lifted = lift_point_to_max(point_be, n_vars);
    let (val, proof) = HyraxBn254::open(ck, &aux.0, commitment, &aux.1, &lifted, sponge, Some(rng))
        .map_err(SnarkError::Pcs)?;
    Ok((val, proof))
}

/// Verify a Hyrax open at a small BE point. The verifier passes
/// the commit's native `commit_n_vars` (derived from public
/// dimensions); the point is lifted to that size with zero-padding.
pub(crate) fn hyrax_verify_at(
    vk: &<HyraxBn254 as MlPcs>::VerifierKey,
    commitment: &<HyraxBn254 as MlPcs>::Commitment,
    point_be: &[Fr],
    value: Fr,
    proof: &<HyraxBn254 as MlPcs>::Proof,
    commit_n_vars: usize,
    sponge: &mut impl CryptographicSponge,
) -> Result<bool, SnarkError> {
    if point_be.len() > commit_n_vars {
        return Err(SnarkError::ShapeMismatch {
            what: "hyrax_verify_at: verify point has more vars than the commit",
        });
    }
    let lifted = lift_point_to_max(point_be, commit_n_vars);
    HyraxBn254::verify(vk, commitment, &lifted, value, proof, sponge).map_err(SnarkError::Pcs)
}

/// Convert a small big-endian point to a little-endian, zero-padded
/// point of length `max_num_vars` for arkworks's MLE evaluator.
/// Reverses the BE point and appends zeros for the high-order LE
/// variables (which select the zero-padded tail of the eval buffer).
pub(crate) fn lift_point_to_max(small_point_be: &[Fr], max_num_vars: usize) -> Vec<Fr> {
    let mut out: Vec<Fr> = small_point_be.iter().rev().copied().collect();
    out.resize(max_num_vars, Fr::from(0u64));
    out
}

/// Prover-side batched-open input. Every item must share the same
/// `commit_n_vars` so the homomorphic combination `C_batch` is
/// well-defined.
pub(crate) struct BatchOpenSpec<'a> {
    pub aux: &'a CommittedAux,
    pub commitment: &'a <HyraxBn254 as MlPcs>::Commitment,
    pub commit_n_vars: usize,
}

/// Verifier-side batched-open input. Same `commit_n_vars` constraint
/// as [`BatchOpenSpec`].
pub(crate) struct BatchVerifySpec<'a> {
    pub commitment: &'a <HyraxBn254 as MlPcs>::Commitment,
    pub value: Fr,
    pub commit_n_vars: usize,
}

/// Batched open of multiple commits at a single shared BE point.
/// All items must share the same `commit_n_vars`. Returns the
/// per-item claimed values plus a single Hyrax proof.
pub(crate) fn hyrax_open_batched_at(
    ck: &<HyraxBn254 as MlPcs>::CommitterKey,
    items: &[BatchOpenSpec<'_>],
    point_be: &[Fr],
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<(Vec<Fr>, <HyraxBn254 as MlPcs>::Proof), SnarkError> {
    if items.is_empty() {
        return Err(SnarkError::ShapeMismatch {
            what: "hyrax_open_batched_at: empty item list",
        });
    }
    let n_vars = items[0].commit_n_vars;
    if point_be.len() > n_vars {
        return Err(SnarkError::ShapeMismatch {
            what: "hyrax_open_batched_at: opening point has more vars than the commit",
        });
    }
    for item in items.iter().skip(1) {
        if item.commit_n_vars != n_vars {
            return Err(SnarkError::ShapeMismatch {
                what: "hyrax_open_batched_at: heterogeneous commit_n_vars",
            });
        }
    }
    for item in items {
        if !item.aux.0.len().is_power_of_two() || item.aux.0.is_empty() {
            return Err(SnarkError::ShapeMismatch {
                what: "hyrax_open_batched_at: aux MLE length must be a non-zero power of two",
            });
        }
        if item.aux.0.len().trailing_zeros() as usize != n_vars {
            return Err(SnarkError::ShapeMismatch {
                what: "hyrax_open_batched_at: aux MLE n_vars != commit_n_vars",
            });
        }
    }

    let lifted = lift_point_to_max(point_be, n_vars);
    let native_items: Vec<hyrax_pcs::BatchOpenItem> = items
        .iter()
        .map(|spec| hyrax_pcs::BatchOpenItem {
            com: spec.commitment,
            evals: &spec.aux.0,
            state: &spec.aux.1,
        })
        .collect();
    let (values, proof) = hyrax_pcs::open_batched_at(ck, &native_items, &lifted, sponge, rng)
        .map_err(|e| SnarkError::Pcs(e.into()))?;
    Ok((values, proof))
}

/// Verifier counterpart to [`hyrax_open_batched_at`].
pub(crate) fn hyrax_verify_batched_at(
    vk: &<HyraxBn254 as MlPcs>::VerifierKey,
    items: &[BatchVerifySpec<'_>],
    point_be: &[Fr],
    proof: &<HyraxBn254 as MlPcs>::Proof,
    sponge: &mut impl CryptographicSponge,
) -> Result<bool, SnarkError> {
    if items.is_empty() {
        return Err(SnarkError::ShapeMismatch {
            what: "hyrax_verify_batched_at: empty item list",
        });
    }
    let n_vars = items[0].commit_n_vars;
    if point_be.len() > n_vars {
        return Err(SnarkError::ShapeMismatch {
            what: "hyrax_verify_batched_at: verify point has more vars than the commit",
        });
    }
    for item in items.iter().skip(1) {
        if item.commit_n_vars != n_vars {
            return Err(SnarkError::ShapeMismatch {
                what: "hyrax_verify_batched_at: heterogeneous commit_n_vars",
            });
        }
    }

    let lifted = lift_point_to_max(point_be, n_vars);
    let native_items: Vec<hyrax_pcs::BatchVerifyItem> = items
        .iter()
        .map(|spec| hyrax_pcs::BatchVerifyItem {
            com: spec.commitment,
            value: spec.value,
        })
        .collect();
    hyrax_pcs::verify_batched_at(vk, &native_items, &lifted, proof, sponge)
        .map_err(|e| SnarkError::Pcs(e.into()))
}

/// Extract `[num0, num1, denom0, denom1]` from a LogUp circuit's
/// top layer.
pub(crate) fn top_halves(circuit: &LogUpCircuit<Fr>) -> [Fr; 4] {
    let top = circuit.layers.last().expect("≥ 1 layer");
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
