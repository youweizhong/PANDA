//! Inner-product sumcheck over BN254's scalar field, plus the round-poly
//! types (degree 2/3/4) and an MLE-evaluation helper shared with the
//! LogUp-GKR module.
//!
//! The inner-product sumcheck proves `Σ_i a[i] · b[i] = claim` for two
//! length-`2^k` vectors treated as MLEs. Each round emits a degree-2
//! polynomial (the bilinear summand `a~ · b~`); the SNARK driver
//! supplies the final `(a~(r), b~(r))` pair via Hyrax PCS openings.
//!
//! Fiat-Shamir runs through an `ark-crypto-primitives`
//! `CryptographicSponge` (Merlin-backed by default), shared with the
//! PCS so transcripts compose cleanly. We avoid depending on
//! `ark-linear-sumcheck` because it pins arkworks 0.4 and would
//! duplicate the arkworks stack here.

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use thiserror::Error;

/// Degree-2 round polynomial, transmitted as its evaluations at
/// `0, 1, 2`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct RoundPoly<F: Field> {
    pub at_zero: F,
    pub at_one: F,
    pub at_two: F,
}

impl<F: Field> RoundPoly<F> {
    /// Lagrange-interpolate through `(0, 1, 2)` and evaluate at `x`.
    pub fn evaluate(&self, x: F) -> F {
        let x_minus_1 = x - F::one();
        let x_minus_2 = x - F::from(2u64);
        let two_inv = F::from(2u64).inverse().expect("2 is invertible in Fr");
        let l0 = x_minus_1 * x_minus_2 * two_inv;
        let l1 = -(x * x_minus_2);
        let l2 = x * x_minus_1 * two_inv;
        self.at_zero * l0 + self.at_one * l1 + self.at_two * l2
    }
}

/// Degree-3 round polynomial, transmitted as evaluations at `0, 1, 2, 3`.
/// Used by the LogUp-GKR per-layer sumcheck (summand is `eq · q1 · q2`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct RoundPoly3<F: Field> {
    pub at_zero: F,
    pub at_one: F,
    pub at_two: F,
    pub at_three: F,
}

impl<F: Field> RoundPoly3<F> {
    /// Lagrange-interpolate the unique cubic through `0..=3` and
    /// evaluate at `x`.
    pub fn evaluate(&self, x: F) -> F {
        let two = F::from(2u64);
        let three = F::from(3u64);
        let six_inv = F::from(6u64).inverse().expect("6 invertible in Fr");
        let two_inv = two.inverse().expect("2 invertible in Fr");
        let x0 = x;
        let x1 = x - F::one();
        let x2 = x - two;
        let x3 = x - three;
        // L_0 = (x-1)(x-2)(x-3) / -6,  L_1 = x(x-2)(x-3) / 2,
        // L_2 = x(x-1)(x-3) / -2,      L_3 = x(x-1)(x-2) / 6.
        let l0 = -x1 * x2 * x3 * six_inv;
        let l1 = x0 * x2 * x3 * two_inv;
        let l2 = -x0 * x1 * x3 * two_inv;
        let l3 = x0 * x1 * x2 * six_inv;
        self.at_zero * l0 + self.at_one * l1 + self.at_two * l2 + self.at_three * l3
    }
}

/// Degree-4 round polynomial, transmitted as evaluations at `0..=4`.
/// Used by sumchecks whose summand has degree four in one variable
/// (e.g. `eq · is_real · q1 · q2` where each `q_i` is bilinear).
#[derive(Copy, Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct RoundPoly4<F: Field> {
    pub at_zero: F,
    pub at_one: F,
    pub at_two: F,
    pub at_three: F,
    pub at_four: F,
}

impl<F: Field> RoundPoly4<F> {
    /// Lagrange-interpolate the unique quartic through `0..=4` and
    /// evaluate at `x`.
    pub fn evaluate(&self, x: F) -> F {
        let two = F::from(2u64);
        let three = F::from(3u64);
        let four = F::from(4u64);
        let inv_24 = F::from(24u64).inverse().expect("24 invertible in Fr");
        let inv_6_neg = -F::from(6u64).inverse().expect("6 invertible in Fr");
        let inv_4 = F::from(4u64).inverse().expect("4 invertible in Fr");
        let x0 = x;
        let x1 = x - F::one();
        let x2 = x - two;
        let x3 = x - three;
        let x4 = x - four;
        let l0 = x1 * x2 * x3 * x4 * inv_24;
        let l1 = x0 * x2 * x3 * x4 * inv_6_neg;
        let l2 = x0 * x1 * x3 * x4 * inv_4;
        let l3 = x0 * x1 * x2 * x4 * inv_6_neg;
        let l4 = x0 * x1 * x2 * x3 * inv_24;
        self.at_zero * l0
            + self.at_one * l1
            + self.at_two * l2
            + self.at_three * l3
            + self.at_four * l4
    }
}

/// Inner-product sumcheck proof: one [`RoundPoly`] per variable.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct InnerProductProof<F: Field> {
    pub rounds: Vec<RoundPoly<F>>,
}

/// Evaluate the multilinear extension of a `2^k`-entry table at
/// `r ∈ F^k`. Uses the same in-place bookkeeping pattern as
/// `prove_inner_product` — `O(2^k)` field multiplications total.
pub fn eval_multilinear<F: PrimeField>(evals: &[F], r: &[F]) -> F {
    assert!(
        evals.len().is_power_of_two(),
        "eval_multilinear: table size must be a power of two (got {})",
        evals.len()
    );
    assert_eq!(
        evals.len(),
        1 << r.len(),
        "eval_multilinear: r has {} entries but table has {} (= 2^{})",
        r.len(),
        evals.len(),
        r.len()
    );
    let mut tab: Vec<F> = evals.to_vec();
    for &ri in r {
        let half = tab.len() / 2;
        for i in 0..half {
            let delta = tab[half + i] - tab[i];
            tab[i] += ri * delta;
        }
        tab.truncate(half);
    }
    tab[0]
}

/// Absorb the round polynomial and squeeze one challenge. Used by both
/// prover and verifier to keep transcripts in lockstep.
fn squeeze_round_challenge<F, S>(sponge: &mut S, poly: &RoundPoly<F>) -> F
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
{
    sponge.absorb(&poly.at_zero);
    sponge.absorb(&poly.at_one);
    sponge.absorb(&poly.at_two);
    sponge.squeeze_field_elements::<F>(1)[0]
}

/// Prove `Σ_i a[i] · b[i] = claim` against a Fiat-Shamir sponge. The
/// caller absorbs all public context (claim, commitment hashes, public
/// parameters) into `sponge` first; sumcheck only absorbs round polys.
pub fn prove_inner_product_with_sponge<F, S>(
    a: &[F],
    b: &[F],
    sponge: &mut S,
) -> Result<(InnerProductProof<F>, Vec<F>), SumcheckError>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
{
    prove_inner_product(a, b, |poly| squeeze_round_challenge(sponge, poly))
}

/// Verify a sumcheck proof using the same transcript shape as the
/// prover.
pub fn verify_inner_product_with_sponge<F, S>(
    claim: F,
    n_vars: usize,
    proof: &InnerProductProof<F>,
    expected_final: F,
    sponge: &mut S,
) -> Result<Vec<F>, SumcheckError>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
{
    verify_inner_product(claim, n_vars, proof, expected_final, |poly| {
        squeeze_round_challenge(sponge, poly)
    })
}

/// Prove `Σ_i a[i] · b[i] = claim`. Per-round challenges come from
/// `next_challenge`, so tests can inject deterministic challenges.
/// Production callers should use [`prove_inner_product_with_sponge`].
///
/// Runs Thaler's linear-time prover (Thaler 2022, §4.2.1) with in-place
/// bookkeeping-table updates: `O(2^n)` field multiplications total, one
/// `Vec` per side, `truncate` after each round.
pub fn prove_inner_product<F: PrimeField>(
    a: &[F],
    b: &[F],
    mut next_challenge: impl FnMut(&RoundPoly<F>) -> F,
) -> Result<(InnerProductProof<F>, Vec<F>), SumcheckError> {
    if a.len() != b.len() {
        return Err(SumcheckError::ShapeMismatch {
            a_len: a.len(),
            b_len: b.len(),
        });
    }
    if a.is_empty() || !a.len().is_power_of_two() {
        return Err(SumcheckError::NonPowerOfTwoLen { len: a.len() });
    }
    let k = a.len().trailing_zeros() as usize;
    let mut a_tab: Vec<F> = a.to_vec();
    let mut b_tab: Vec<F> = b.to_vec();
    let mut rounds: Vec<RoundPoly<F>> = Vec::with_capacity(k);
    let mut challenges: Vec<F> = Vec::with_capacity(k);
    for _ in 0..k {
        let half = a_tab.len() / 2;
        let mut s0 = F::zero();
        let mut s1 = F::zero();
        let mut s2 = F::zero();
        for i in 0..half {
            let a0 = a_tab[i];
            let a1 = a_tab[half + i];
            let b0 = b_tab[i];
            let b1 = b_tab[half + i];
            s0 += a0 * b0;
            s1 += a1 * b1;
            // Affine extension: (a(2), b(2)) = (2·a1 - a0, 2·b1 - b0).
            let a2 = a1.double() - a0;
            let b2 = b1.double() - b0;
            s2 += a2 * b2;
        }
        let poly = RoundPoly {
            at_zero: s0,
            at_one: s1,
            at_two: s2,
        };
        let r = next_challenge(&poly);
        // In-place bookkeeping: `a[i] ← a[i] + r·(a[half+i] - a[i])`.
        for i in 0..half {
            let delta_a = a_tab[half + i] - a_tab[i];
            a_tab[i] += r * delta_a;
            let delta_b = b_tab[half + i] - b_tab[i];
            b_tab[i] += r * delta_b;
        }
        a_tab.truncate(half);
        b_tab.truncate(half);
        rounds.push(poly);
        challenges.push(r);
    }
    Ok((InnerProductProof { rounds }, challenges))
}

/// Verify an inner-product sumcheck proof. Returns the challenge
/// vector `r` so the caller can independently check `(a~(r), b~(r))`.
/// `expected_final` is the value the last round-poly should agree with
/// — tests pass `a~(r) · b~(r)` directly; the SNARK driver supplies it
/// via PCS openings.
pub fn verify_inner_product<F: PrimeField>(
    claim: F,
    n_vars: usize,
    proof: &InnerProductProof<F>,
    expected_final: F,
    mut next_challenge: impl FnMut(&RoundPoly<F>) -> F,
) -> Result<Vec<F>, SumcheckError> {
    if proof.rounds.len() != n_vars {
        return Err(SumcheckError::WrongRoundCount {
            expected: n_vars,
            got: proof.rounds.len(),
        });
    }
    let mut current_claim = claim;
    let mut challenges: Vec<F> = Vec::with_capacity(n_vars);
    for round in &proof.rounds {
        let split = round.at_zero + round.at_one;
        if split != current_claim {
            return Err(SumcheckError::SplitMismatch);
        }
        let r = next_challenge(round);
        current_claim = round.evaluate(r);
        challenges.push(r);
    }
    if current_claim != expected_final {
        return Err(SumcheckError::FinalMismatch);
    }
    Ok(challenges)
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SumcheckError {
    #[error("input vectors have mismatched lengths a={a_len}, b={b_len}")]
    ShapeMismatch { a_len: usize, b_len: usize },
    #[error("input length {len} is not a power of two")]
    NonPowerOfTwoLen { len: usize },
    #[error("expected {expected} sumcheck rounds, got {got}")]
    WrongRoundCount { expected: usize, got: usize },
    #[error("round-poly split (g_i(0) + g_i(1)) did not match incoming claim")]
    SplitMismatch,
    #[error("final round-poly value did not match a~(r) · b~(r)")]
    FinalMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use ark_std::UniformRand;

    fn inner_product<F: PrimeField>(a: &[F], b: &[F]) -> F {
        a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
    }

    #[test]
    fn round_poly_evaluate_matches_lagrange() {
        // Pick g(X) = (X - 1)·X = X^2 - X. g(0) = 0, g(1) = 0, g(2) = 2.
        let p = RoundPoly {
            at_zero: Fr::from(0u64),
            at_one: Fr::from(0u64),
            at_two: Fr::from(2u64),
        };
        // Should evaluate as X^2 - X at any X. Try X = 3 → 6.
        assert_eq!(p.evaluate(Fr::from(3u64)), Fr::from(6u64));
        // X = 5 → 20.
        assert_eq!(p.evaluate(Fr::from(5u64)), Fr::from(20u64));
    }

    #[test]
    fn eval_multilinear_at_corner_recovers_table_entry() {
        // Convention: `r[0]` binds the highest-order index bit, `r[k-1]`
        // the lowest. For index `i`, `r` is its bits in MSB-first order.
        let evals: Vec<Fr> = (0..8).map(|i| Fr::from(i as u64 * 3 + 1)).collect();
        for i in 0..8u32 {
            let bits: Vec<Fr> = (0..3)
                .rev()
                .map(|b| Fr::from(((i >> b) & 1) as u64))
                .collect();
            assert_eq!(eval_multilinear(&evals, &bits), evals[i as usize]);
        }
    }

    fn round_robin_challenges<F: PrimeField>(seeds: &[u64]) -> impl FnMut(&RoundPoly<F>) -> F + '_ {
        let mut idx = 0;
        move |_round: &RoundPoly<F>| {
            let v = F::from(seeds[idx]);
            idx += 1;
            v
        }
    }

    #[test]
    fn prove_then_verify_roundtrips() {
        let mut rng = test_rng();
        for n_vars in 1..=5 {
            let len = 1usize << n_vars;
            let a: Vec<Fr> = (0..len).map(|_| Fr::rand(&mut rng)).collect();
            let b: Vec<Fr> = (0..len).map(|_| Fr::rand(&mut rng)).collect();
            let claim = inner_product(&a, &b);

            let seeds: Vec<u64> = (0..n_vars).map(|i| (i as u64 + 1) * 17).collect();
            let prover_challenges = seeds.clone();
            let verifier_challenges = seeds.clone();

            let (proof, r_p) =
                prove_inner_product(&a, &b, round_robin_challenges::<Fr>(&prover_challenges))
                    .unwrap();
            let final_a = eval_multilinear(&a, &r_p);
            let final_b = eval_multilinear(&b, &r_p);
            let expected = final_a * final_b;
            let r_v = verify_inner_product(
                claim,
                n_vars,
                &proof,
                expected,
                round_robin_challenges::<Fr>(&verifier_challenges),
            )
            .unwrap();
            assert_eq!(r_p, r_v);
        }
    }

    #[test]
    fn tampered_round_poly_breaks_verifier() {
        let a: Vec<Fr> = (0..4).map(|i| Fr::from(i as u64 + 1)).collect();
        let b: Vec<Fr> = (0..4).map(|i| Fr::from(7 - i as u64)).collect();
        let claim = inner_product(&a, &b);
        let seeds = vec![5u64, 9];
        let (mut proof, r) =
            prove_inner_product(&a, &b, round_robin_challenges::<Fr>(&seeds)).unwrap();
        // Bumping round 0's at_zero breaks g(0)+g(1) = claim.
        proof.rounds[0].at_zero += Fr::from(1u64);
        let expected = eval_multilinear(&a, &r) * eval_multilinear(&b, &r);
        let verdict = verify_inner_product(
            claim,
            2,
            &proof,
            expected,
            round_robin_challenges::<Fr>(&seeds),
        );
        assert!(matches!(verdict, Err(SumcheckError::SplitMismatch)));
    }

    #[test]
    fn wrong_final_evaluation_breaks_verifier() {
        let a: Vec<Fr> = (0..8).map(|i| Fr::from(i as u64 + 1)).collect();
        let b: Vec<Fr> = (0..8).map(|i| Fr::from(2 * i as u64 + 3)).collect();
        let claim = inner_product(&a, &b);
        let seeds = vec![3u64, 11, 17];
        let (proof, r) = prove_inner_product(&a, &b, round_robin_challenges::<Fr>(&seeds)).unwrap();
        let bogus = eval_multilinear(&a, &r) * eval_multilinear(&b, &r) + Fr::from(1u64);
        let verdict = verify_inner_product(
            claim,
            3,
            &proof,
            bogus,
            round_robin_challenges::<Fr>(&seeds),
        );
        assert!(matches!(verdict, Err(SumcheckError::FinalMismatch)));
    }

    #[test]
    fn round_poly4_evaluate_matches_lagrange() {
        // g(X) = X·(X-1)·(X-2)·(X-3): zero at 0..=3, g(4) = 24.
        let p = RoundPoly4 {
            at_zero: Fr::from(0u64),
            at_one: Fr::from(0u64),
            at_two: Fr::from(0u64),
            at_three: Fr::from(0u64),
            at_four: Fr::from(24u64),
        };
        // g(5) = 5·4·3·2 = 120
        assert_eq!(p.evaluate(Fr::from(5u64)), Fr::from(120u64));
        // g(6) = 6·5·4·3 = 360
        assert_eq!(p.evaluate(Fr::from(6u64)), Fr::from(360u64));
    }

    #[test]
    fn round_poly4_interpolates_arbitrary_quartic() {
        // p(X) = 2·X^4 - 3·X^3 + X - 7.
        let coef = |x: u64| {
            let xi = Fr::from(x);
            let two = Fr::from(2u64);
            let three = Fr::from(3u64);
            let one = Fr::from(1u64);
            let seven = Fr::from(7u64);
            two * xi * xi * xi * xi - three * xi * xi * xi + one * xi - seven
        };
        let p = RoundPoly4 {
            at_zero: coef(0),
            at_one: coef(1),
            at_two: coef(2),
            at_three: coef(3),
            at_four: coef(4),
        };
        assert_eq!(p.evaluate(Fr::from(7u64)), coef(7));
        assert_eq!(p.evaluate(Fr::from(11u64)), coef(11));
    }

    #[test]
    fn rejects_non_power_of_two() {
        let a: Vec<Fr> = (0..3).map(Fr::from).collect();
        let b: Vec<Fr> = (0..3).map(Fr::from).collect();
        let err = prove_inner_product(&a, &b, |_| Fr::from(1u64));
        assert!(matches!(err, Err(SumcheckError::NonPowerOfTwoLen { .. })));
    }

    #[test]
    fn shape_mismatch_rejected() {
        let a: Vec<Fr> = (0..4).map(Fr::from).collect();
        let b: Vec<Fr> = (0..2).map(Fr::from).collect();
        let err = prove_inner_product(&a, &b, |_| Fr::from(1u64));
        assert!(matches!(err, Err(SumcheckError::ShapeMismatch { .. })));
    }

    fn fresh_transcript() -> ark_crypto_primitives::sponge::merlin::Transcript {
        ark_crypto_primitives::sponge::merlin::Transcript::new(b"panda-sumcheck-test")
    }

    #[test]
    fn sponge_prover_verifier_agree_end_to_end() {
        use ark_crypto_primitives::sponge::CryptographicSponge;
        let mut rng = test_rng();
        for n_vars in 1..=5 {
            let len = 1usize << n_vars;
            let a: Vec<Fr> = (0..len).map(|_| Fr::rand(&mut rng)).collect();
            let b: Vec<Fr> = (0..len).map(|_| Fr::rand(&mut rng)).collect();
            let claim = inner_product(&a, &b);

            // Both sides absorb the same public context so transcripts
            // diverge only on tampered round polys.
            let mut prover_sponge = fresh_transcript();
            prover_sponge.absorb(&claim);
            prover_sponge.absorb(&(n_vars as u64));
            let (proof, r_p) =
                prove_inner_product_with_sponge::<Fr, _>(&a, &b, &mut prover_sponge).unwrap();

            let mut verifier_sponge = fresh_transcript();
            verifier_sponge.absorb(&claim);
            verifier_sponge.absorb(&(n_vars as u64));
            let final_a = eval_multilinear(&a, &r_p);
            let final_b = eval_multilinear(&b, &r_p);
            let r_v = verify_inner_product_with_sponge::<Fr, _>(
                claim,
                n_vars,
                &proof,
                final_a * final_b,
                &mut verifier_sponge,
            )
            .unwrap();
            assert_eq!(r_p, r_v);
        }
    }

    #[test]
    fn sponge_diverging_context_rejects_proof() {
        use ark_crypto_primitives::sponge::CryptographicSponge;
        // Diverging absorbed context flips r_0; the recomputed claim
        // into round 1 then fails the split check.
        let a: Vec<Fr> = (0..8).map(|i| Fr::from(i as u64 + 1)).collect();
        let b: Vec<Fr> = (0..8).map(|i| Fr::from(2 * i as u64 + 3)).collect();
        let claim = inner_product(&a, &b);

        let mut prover_sponge = fresh_transcript();
        prover_sponge.absorb(&claim);
        let (proof, r_p) =
            prove_inner_product_with_sponge::<Fr, _>(&a, &b, &mut prover_sponge).unwrap();

        let mut verifier_sponge = fresh_transcript();
        verifier_sponge.absorb(&(claim + Fr::from(1u64)));
        let final_a = eval_multilinear(&a, &r_p);
        let final_b = eval_multilinear(&b, &r_p);
        let verdict = verify_inner_product_with_sponge::<Fr, _>(
            claim,
            3,
            &proof,
            final_a * final_b,
            &mut verifier_sponge,
        );
        assert!(
            verdict.is_err(),
            "expected verifier rejection on diverging transcript"
        );
    }
}
