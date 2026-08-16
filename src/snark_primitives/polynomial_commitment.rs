//! Multilinear polynomial commitment abstraction.
//!
//! Every commitment in PANDA is to the multilinear extension of a
//! `2^k`-length evaluation table. The default backend is the native
//! Hyrax re-implementation in [`crate::snark_primitives::hyrax_pcs`],
//! which is wire-compatible with `ark_poly_commit::hyrax` but exposes
//! per-row Pedersen randomness so the SNARK driver can batch many
//! single-point opens into one. The trait keeps the driver independent
//! of the backend so we can swap in Dory or another scheme later.

use ark_bn254::{Fr, G1Affine};
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_poly_commit::hyrax::HyraxPC;
use ark_poly_commit::PolynomialCommitment;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;
use thiserror::Error;

use crate::snark_primitives::hyrax_pcs;

/// Multilinear PCS trait. `Field` is the underlying SNARK scalar field.
pub trait MlPcs {
    type Field: PrimeField;
    type Params: Clone;
    type CommitterKey: Clone;
    type VerifierKey: Clone;
    type Commitment: Clone + CanonicalSerialize + CanonicalDeserialize;
    /// Prover state retained between commit and open (e.g. row blinders).
    type CommitmentState: Clone;
    type Proof: Clone + CanonicalSerialize + CanonicalDeserialize;
    type Error: std::fmt::Debug;

    /// One-time setup for MLEs of `num_vars` variables.
    fn setup(
        num_vars: usize,
        rng: &mut impl RngCore,
    ) -> Result<(Self::CommitterKey, Self::VerifierKey), Self::Error>;

    /// Commit to one MLE given as its `2^num_vars`-entry evaluation table.
    /// Returns the commitment and the prover state needed to open later.
    fn commit(
        ck: &Self::CommitterKey,
        evals: &[Self::Field],
        rng: Option<&mut dyn RngCore>,
    ) -> Result<(Self::Commitment, Self::CommitmentState), Self::Error>;

    /// Open the committed MLE at `point`.
    fn open(
        ck: &Self::CommitterKey,
        evals: &[Self::Field],
        commitment: &Self::Commitment,
        state: &Self::CommitmentState,
        point: &[Self::Field],
        sponge: &mut impl CryptographicSponge,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<(Self::Field, Self::Proof), Self::Error>;

    /// Verify an opening.
    fn verify(
        vk: &Self::VerifierKey,
        commitment: &Self::Commitment,
        point: &[Self::Field],
        value: Self::Field,
        proof: &Self::Proof,
        sponge: &mut impl CryptographicSponge,
    ) -> Result<bool, Self::Error>;
}

/// Default backend: Hyrax over BN254's G1. Pedersen-row construction;
/// no pairings, no trusted setup beyond a hash-derived generator vector.
pub struct HyraxBn254;

type Inner = HyraxPC<G1Affine, DenseMultilinearExtension<Fr>>;

#[derive(Debug, Error)]
pub enum HyraxError {
    #[error("polynomial-commitment error: {0:?}")]
    Pc(ark_poly_commit::Error),
    #[error("internal: produced an empty commitment list")]
    EmptyCommitment,
    #[error("internal: produced an empty opening")]
    EmptyOpening,
    #[error("evaluation-table size {got} is not 2^num_vars (= {expected})")]
    BadEvalTable { got: usize, expected: usize },
    #[error("evaluation point has {got} variables but the MLE has {expected}")]
    BadPointDim { got: usize, expected: usize },
    #[error("Hyrax commit/open requires randomness but caller passed None")]
    MissingRandomness,
    #[error("native PCS error: {0:?}")]
    Native(hyrax_pcs::HyraxError),
}

impl From<ark_poly_commit::Error> for HyraxError {
    fn from(e: ark_poly_commit::Error) -> Self {
        HyraxError::Pc(e)
    }
}

impl From<hyrax_pcs::HyraxError> for HyraxError {
    fn from(e: hyrax_pcs::HyraxError) -> Self {
        // Preserve the variants the test suite pattern-matches on
        // (BadEvalTable / BadPointDim); bucket the rest under `Native`.
        match e {
            hyrax_pcs::HyraxError::BadEvalTable { got } => {
                HyraxError::BadEvalTable { got, expected: 0 }
            }
            hyrax_pcs::HyraxError::BadPointDim { got, expected } => {
                HyraxError::BadPointDim { got, expected }
            }
            other => HyraxError::Native(other),
        }
    }
}

impl MlPcs for HyraxBn254 {
    type Field = Fr;
    type Params = (
        <Inner as PolynomialCommitment<Fr, DenseMultilinearExtension<Fr>>>::CommitterKey,
        <Inner as PolynomialCommitment<Fr, DenseMultilinearExtension<Fr>>>::VerifierKey,
    );
    type CommitterKey = hyrax_pcs::CommitterKey;
    type VerifierKey = hyrax_pcs::VerifierKey;
    type Commitment = hyrax_pcs::Commitment;
    /// Per-row Pedersen blinders. Public so the SNARK driver can
    /// homomorphically combine states for batched single-point opens via
    /// [`hyrax_pcs::open_batched_at`].
    type CommitmentState = hyrax_pcs::CommitState;
    type Proof = hyrax_pcs::Proof;
    type Error = HyraxError;

    fn setup(
        num_vars: usize,
        rng: &mut impl RngCore,
    ) -> Result<(Self::CommitterKey, Self::VerifierKey), Self::Error> {
        // Hyrax's sqrt-MLE layout needs an even number of variables (one
        // half to rows, the other to columns). Reuse upstream's
        // hash-derived setup so the public parameters match across the
        // two implementations.
        let padded = if num_vars % 2 == 1 {
            num_vars + 1
        } else {
            num_vars
        };
        let pp = Inner::setup(1, Some(padded), rng)?;
        let (ck, vk) = Inner::trim(&pp, 1, 0, None)?;
        Ok((ck, vk))
    }

    fn commit(
        ck: &Self::CommitterKey,
        evals: &[Self::Field],
        rng: Option<&mut dyn RngCore>,
    ) -> Result<(Self::Commitment, Self::CommitmentState), Self::Error> {
        let rng = rng.ok_or(HyraxError::MissingRandomness)?;
        Ok(hyrax_pcs::commit(ck, evals, rng)?)
    }

    fn open(
        ck: &Self::CommitterKey,
        evals: &[Self::Field],
        commitment: &Self::Commitment,
        state: &Self::CommitmentState,
        point: &[Self::Field],
        sponge: &mut impl CryptographicSponge,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<(Self::Field, Self::Proof), Self::Error> {
        // Check the point dimension here: arkworks'
        // `DenseMultilinearExtension::evaluate` panics on a mismatch,
        // and we'd rather return a structured error.
        let n_vars = nvars_from_len(evals.len())?;
        if point.len() != n_vars {
            return Err(HyraxError::BadPointDim {
                got: point.len(),
                expected: n_vars,
            });
        }
        // Absorb the claimed value into the transcript before delegating
        // to the dot-product argument. `hyrax_pcs::open_at` does NOT
        // absorb internally; mismatched verifier absorption flips the
        // FS challenge `c` and the round equations diverge.
        let value = pcs_native_eval(evals, point);
        sponge.absorb(&value);
        let rng = rng.ok_or(HyraxError::MissingRandomness)?;
        let (val, proof) = hyrax_pcs::open_at(ck, evals, state, commitment, point, sponge, rng)?;
        debug_assert_eq!(val, value, "hyrax_pcs::open_at returned a divergent eval");
        Ok((val, proof))
    }

    fn verify(
        vk: &Self::VerifierKey,
        commitment: &Self::Commitment,
        point: &[Self::Field],
        value: Self::Field,
        proof: &Self::Proof,
        sponge: &mut impl CryptographicSponge,
    ) -> Result<bool, Self::Error> {
        sponge.absorb(&value);
        Ok(hyrax_pcs::verify_at(vk, commitment, point, proof, sponge)?)
    }
}

/// Evaluate the MLE at `point` using the LE-indexing convention shared
/// by every downstream module. Used by the prover wrapper so the
/// claimed value matches what the test suite computes.
fn pcs_native_eval(evals: &[Fr], point: &[Fr]) -> Fr {
    let mle = DenseMultilinearExtension::from_evaluations_slice(point.len(), evals);
    <DenseMultilinearExtension<Fr> as ark_poly::Polynomial<Fr>>::evaluate(&mle, &point.to_vec())
}

fn nvars_from_len(len: usize) -> Result<usize, HyraxError> {
    if !len.is_power_of_two() || len == 0 {
        return Err(HyraxError::BadEvalTable {
            got: len,
            expected: 0,
        });
    }
    Ok(len.trailing_zeros() as usize)
}

/// Construct a fresh Merlin-backed sponge under a label. Used by tests.
pub fn fresh_sponge(label: &'static [u8]) -> ark_crypto_primitives::sponge::merlin::Transcript {
    use ark_crypto_primitives::sponge::CryptographicSponge;
    <ark_crypto_primitives::sponge::merlin::Transcript as CryptographicSponge>::new(&label)
}

// Re-exports for downstream sumcheck combinators.
pub use ark_crypto_primitives::sponge::{
    Absorb as SpongeAbsorb, CryptographicSponge as SpongeTrait,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::test_rng;
    use ark_std::UniformRand;

    fn rand_evals(n_vars: usize, rng: &mut impl RngCore) -> Vec<Fr> {
        (0..(1usize << n_vars)).map(|_| Fr::rand(rng)).collect()
    }

    fn mle_evaluate(evals: &[Fr], point: &[Fr]) -> Fr {
        let mle = DenseMultilinearExtension::from_evaluations_slice(point.len(), evals);
        <DenseMultilinearExtension<Fr> as ark_poly::Polynomial<Fr>>::evaluate(&mle, &point.to_vec())
    }

    #[test]
    fn round_trip_commit_open_verify() {
        let mut rng = test_rng();
        // Hyrax requires an even number of variables.
        for n_vars in [2usize, 4, 6] {
            let (ck, vk) = HyraxBn254::setup(n_vars, &mut rng).unwrap();
            let evals = rand_evals(n_vars, &mut rng);
            let (com, state) = HyraxBn254::commit(&ck, &evals, Some(&mut rng)).unwrap();
            let point: Vec<Fr> = (0..n_vars).map(|_| Fr::rand(&mut rng)).collect();
            let mut prover_sponge = fresh_sponge(b"panda-pcs-test");
            let (value, proof) = HyraxBn254::open(
                &ck,
                &evals,
                &com,
                &state,
                &point,
                &mut prover_sponge,
                Some(&mut rng),
            )
            .unwrap();
            let expected = mle_evaluate(&evals, &point);
            assert_eq!(value, expected, "open returned the wrong evaluation");
            let mut verifier_sponge = fresh_sponge(b"panda-pcs-test");
            let ok =
                HyraxBn254::verify(&vk, &com, &point, value, &proof, &mut verifier_sponge).unwrap();
            assert!(ok, "honest opening rejected");
        }
    }

    #[test]
    fn lying_evaluation_rejected() {
        let mut rng = test_rng();
        let n_vars = 4;
        let (ck, vk) = HyraxBn254::setup(n_vars, &mut rng).unwrap();
        let evals = rand_evals(n_vars, &mut rng);
        let (com, state) = HyraxBn254::commit(&ck, &evals, Some(&mut rng)).unwrap();
        let point: Vec<Fr> = (0..n_vars).map(|_| Fr::rand(&mut rng)).collect();
        let mut prover_sponge = fresh_sponge(b"panda-pcs-test");
        let (value, proof) = HyraxBn254::open(
            &ck,
            &evals,
            &com,
            &state,
            &point,
            &mut prover_sponge,
            Some(&mut rng),
        )
        .unwrap();
        let mut verifier_sponge = fresh_sponge(b"panda-pcs-test");
        let bogus = value + Fr::from(1u64);
        let verdict =
            HyraxBn254::verify(&vk, &com, &point, bogus, &proof, &mut verifier_sponge).unwrap();
        assert!(!verdict, "verifier accepted lying opening");
    }

    #[test]
    fn size_mismatch_rejected() {
        let mut rng = test_rng();
        let (ck, _vk) = HyraxBn254::setup(4, &mut rng).unwrap();
        // 7 isn't a power of two.
        let evals: Vec<Fr> = (0..7).map(Fr::from).collect();
        let err = HyraxBn254::commit(&ck, &evals, Some(&mut rng));
        assert!(matches!(err, Err(HyraxError::BadEvalTable { .. })));
    }

    #[test]
    fn point_dim_mismatch_rejected() {
        let mut rng = test_rng();
        let n_vars = 4;
        let (ck, _vk) = HyraxBn254::setup(n_vars, &mut rng).unwrap();
        let evals = rand_evals(n_vars, &mut rng);
        let (com, state) = HyraxBn254::commit(&ck, &evals, Some(&mut rng)).unwrap();
        let mut prover_sponge = fresh_sponge(b"panda-pcs-test");
        let bad_point: Vec<Fr> = (0..3).map(Fr::from).collect(); // wrong length
        let err = HyraxBn254::open(
            &ck,
            &evals,
            &com,
            &state,
            &bad_point,
            &mut prover_sponge,
            Some(&mut rng),
        );
        assert!(matches!(err, Err(HyraxError::BadPointDim { .. })));
    }
}
