//! Per-ReLU-layer gadget proving `d_lower[j] ∈ {0, s_d}` for every
//! neuron `j`.
//!
//! The cell-wise constraint `d_lower[j] · (s_d − d_lower[j]) = 0`
//! holds for every Boolean index `j` iff the multilinear extension
//! of the per-cell product is the zero polynomial. A degree-3
//! sumcheck on `Σ_j eq(j, r) · d(j) · (s_d − d(j)) = 0` reduces this
//! to one Hyrax open of `d_lower` at the sumcheck-final point, with
//! `r` squeezed from the FS sponge after binding
//! `(layer_idx, n_vars, d_lower commit, s_d)`.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::AdditiveGroup;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;

use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};
use crate::snark_primitives::sumcheck::{eval_multilinear, RoundPoly3};

use crate::snark::commitment::commit::CommittedAux;
use crate::snark::commitment::multilinear_extensions::build_eq_table;
use crate::snark::commitment::pcs_helpers::{hyrax_open_at, hyrax_verify_at};
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;

/// Per-ReLU-layer proof that `d_lower[j] · (s_d − d_lower[j]) = 0`
/// for every neuron `j`.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ReluDBooleanProof {
    pub layer_idx: usize,
    pub n_vars: usize,
    /// Per-round univariate, transmitted as evaluations at 0,1,2,3.
    pub rounds: Vec<RoundPoly3<Fr>>,
    /// FS-derived test point `r` (squeezed before round 1).
    pub r: Vec<Fr>,
    /// Sumcheck-final point `r'` (concatenated round challenges).
    pub r_final: Vec<Fr>,
    /// Hyrax open of `d_lower` at `r_final`.
    pub d_lower_eval: Fr,
    pub d_lower_open: <HyraxBn254 as MlPcs>::Proof,
}

fn squeeze_round_challenge_3(sponge: &mut impl CryptographicSponge, poly: &RoundPoly3<Fr>) -> Fr {
    let mut buf = Vec::new();
    poly.serialize_compressed(&mut buf)
        .expect("serialize round poly");
    sponge.absorb(&buf);
    sponge.squeeze_field_elements::<Fr>(1)[0]
}

/// Prove `d_lower[j] · (s_d − d_lower[j]) = 0` for every neuron `j`.
pub(crate) fn prove_relu_d_boolean(
    layer_idx: usize,
    d_lower_codes: &[Fr],
    d_lower_aux: &CommittedAux,
    d_lower_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s_d: Fr,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<ReluDBooleanProof, SnarkError> {
    let _timing = crate::timing::scope("relu_gadget");
    let n = d_lower_codes.len();
    if !n.is_power_of_two() || n == 0 {
        return Err(SnarkError::ShapeMismatch {
            what: "relu_d_boolean: d_lower length must be a positive power of two",
        });
    }
    if d_lower_aux.0.len() != n {
        return Err(SnarkError::ShapeMismatch {
            what: "relu_d_boolean: d_lower codes length != d_lower aux length",
        });
    }
    let n_vars = n.trailing_zeros() as usize;

    sponge.absorb(&(layer_idx as u64));
    sponge.absorb(&(n_vars as u64));
    let mut buf = Vec::new();
    d_lower_commit
        .serialize_compressed(&mut buf)
        .expect("serialize commitment");
    sponge.absorb(&buf);
    sponge.absorb(&s_d);
    let r: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(n_vars);

    let mut eq = build_eq_table(&r);
    let mut d = d_lower_codes.to_vec();
    let mut current_sum = Fr::ZERO;
    let mut rounds: Vec<RoundPoly3<Fr>> = Vec::with_capacity(n_vars);
    let mut r_final: Vec<Fr> = Vec::with_capacity(n_vars);

    for _ in 0..n_vars {
        let half = d.len() / 2;
        let (mut e0, mut e1, mut e2, mut e3) = (Fr::ZERO, Fr::ZERO, Fr::ZERO, Fr::ZERO);
        for i in 0..half {
            let d0 = d[i];
            let d1 = d[half + i];
            let d2 = d1.double() - d0;
            let d3 = d1.double() + d1 - d0.double();
            let q0 = eq[i];
            let q1 = eq[half + i];
            let q2 = q1.double() - q0;
            let q3 = q1.double() + q1 - q0.double();
            // summand = eq · d · (s_d − d)
            e0 += q0 * d0 * (s_d - d0);
            e1 += q1 * d1 * (s_d - d1);
            e2 += q2 * d2 * (s_d - d2);
            e3 += q3 * d3 * (s_d - d3);
        }
        let poly = RoundPoly3 {
            at_zero: e0,
            at_one: e1,
            at_two: e2,
            at_three: e3,
        };
        // Fail fast if the witness is malformed — otherwise we'd
        // emit an Ok proof that the verifier later rejects.
        if poly.at_zero + poly.at_one != current_sum {
            return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
                which: "relu_d_boolean::sumcheck_round_invariant",
            });
        }

        let r_i = squeeze_round_challenge_3(sponge, &poly);
        rounds.push(poly);
        r_final.push(r_i);
        current_sum = rounds.last().unwrap().evaluate(r_i);

        for i in 0..half {
            let d_lo = d[i];
            let d_hi = d[half + i];
            d[i] = d_lo + r_i * (d_hi - d_lo);
            let e_lo = eq[i];
            let e_hi = eq[half + i];
            eq[i] = e_lo + r_i * (e_hi - e_lo);
        }
        d.truncate(half);
        eq.truncate(half);
    }

    let (d_lower_eval, d_lower_open) = hyrax_open_at(
        &params.committer_key,
        d_lower_aux,
        d_lower_commit,
        &r_final,
        sponge,
        rng,
    )?;

    // Fail fast on a malformed witness so we don't emit an Ok proof
    // that the verifier later rejects.
    let eq_eval = eval_multilinear(&build_eq_table(&r), &r_final);
    if eq_eval * d_lower_eval * (s_d - d_lower_eval) != current_sum {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "relu_d_boolean::final_identity",
        });
    }

    Ok(ReluDBooleanProof {
        layer_idx,
        n_vars,
        rounds,
        r,
        r_final,
        d_lower_eval,
        d_lower_open,
    })
}

/// Verify a [`ReluDBooleanProof`] against the public `d_lower`
/// commit and the public scale code `s_d`.
pub(crate) fn verify_relu_d_boolean(
    proof: &ReluDBooleanProof,
    expected_layer_idx: usize,
    expected_n_vars: usize,
    s_d: Fr,
    d_lower_commit: &<HyraxBn254 as MlPcs>::Commitment,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    if proof.layer_idx != expected_layer_idx {
        return Err(SnarkError::ArchitectureMismatch {
            what: "relu_d_boolean: layer_idx mismatch",
        });
    }
    if proof.n_vars != expected_n_vars {
        return Err(SnarkError::ArchitectureMismatch {
            what: "relu_d_boolean: n_vars mismatch with public architecture",
        });
    }
    if proof.r.len() != proof.n_vars
        || proof.r_final.len() != proof.n_vars
        || proof.rounds.len() != proof.n_vars
    {
        return Err(SnarkError::ShapeMismatch {
            what: "relu_d_boolean: r/r_final/rounds length must match n_vars",
        });
    }

    sponge.absorb(&(proof.layer_idx as u64));
    sponge.absorb(&(proof.n_vars as u64));
    let mut buf = Vec::new();
    d_lower_commit
        .serialize_compressed(&mut buf)
        .expect("serialize commitment");
    sponge.absorb(&buf);
    sponge.absorb(&s_d);
    let expected_r = sponge.squeeze_field_elements::<Fr>(proof.n_vars);
    if expected_r != proof.r {
        return Err(SnarkError::TranscriptMismatch);
    }

    let mut claim = Fr::ZERO;
    for (round_idx, round_poly) in proof.rounds.iter().enumerate() {
        if round_poly.at_zero + round_poly.at_one != claim {
            return Err(SnarkError::SumcheckRoundCheckFailed { round: round_idx });
        }
        let r_i = squeeze_round_challenge_3(sponge, round_poly);
        if r_i != proof.r_final[round_idx] {
            return Err(SnarkError::TranscriptMismatch);
        }
        claim = round_poly.evaluate(r_i);
    }

    let ok = hyrax_verify_at(
        &params.verifier_key,
        d_lower_commit,
        &proof.r_final,
        proof.d_lower_eval,
        &proof.d_lower_open,
        proof.n_vars,
        sponge,
    )?;
    if !ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "relu_d_boolean::d_lower",
        });
    }

    let eq_eval = eval_multilinear(&build_eq_table(&proof.r), &proof.r_final);
    let lhs = eq_eval * proof.d_lower_eval * (s_d - proof.d_lower_eval);
    if lhs != claim {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "relu_d_boolean::final_identity",
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_crypto_primitives::sponge::merlin::Transcript;
    use ark_crypto_primitives::sponge::CryptographicSponge;
    use ark_std::test_rng;

    fn fresh_sponge() -> Transcript {
        <Transcript as CryptographicSponge>::new(&b"relu-d-boolean-test".as_slice())
    }

    fn setup_params(n_vars: usize) -> SnarkParams {
        // Build a minimal SnarkParams just for the Hyrax keys
        // (preprocessed tables aren't used by this gadget). Hyrax
        // requires even n_vars; the caller passes the **already-
        // bumped** even n_vars.
        use crate::snark_primitives::polynomial_commitment::HyraxBn254;
        assert!(n_vars.is_multiple_of(2) && n_vars >= 2, "n_vars must be even ≥ 2");
        let mut rng = test_rng();
        let (ck, vk) = HyraxBn254::setup(n_vars, &mut rng).expect("hyrax setup");
        SnarkParams {
            committer_key: ck,
            verifier_key: vk,
            max_num_vars: n_vars,
            precision_bits: 14,
            out_bound_range_bits: 19,
            gadget_range_bits: 19,
            sigma_x_scale_log2: crate::snark::preprocess::TEST_SIGMA_X_SCALE_LOG2,
            sigma_v_scale_log2: crate::snark::preprocess::TEST_SIGMA_V_SCALE_LOG2,
            input_scale_log2: None,
            preprocessed: crate::snark::preprocess::test_shared(19, 19, 19),
        }
    }

    fn commit_d(
        d: &[Fr],
        params: &SnarkParams,
    ) -> (CommittedAux, <HyraxBn254 as MlPcs>::Commitment) {
        use crate::snark_primitives::polynomial_commitment::HyraxBn254;
        let mut rng = test_rng();
        let (commit, state) =
            HyraxBn254::commit(&params.committer_key, d, Some(&mut rng)).expect("commit");
        ((d.to_vec(), state), commit)
    }

    #[test]
    fn valid_d_lower_in_zero_or_sd_accepts() {
        // d_lower vector with cells in {0, s_d}: the gadget must
        // both prove and verify without error.
        let s_d = Fr::from(1u64 << 14);
        // 16 cells (n_vars = 4) of {0, s_d}.
        let mut d: Vec<Fr> = Vec::with_capacity(16);
        for i in 0..16 {
            d.push(if i % 2 == 0 { Fr::from(0u64) } else { s_d });
        }
        let params = setup_params(4);
        let (aux, commit) = commit_d(&d, &params);

        let mut prover_sponge = fresh_sponge();
        let proof = prove_relu_d_boolean(
            7,
            &d,
            &aux,
            &commit,
            s_d,
            &params,
            &mut prover_sponge,
            &mut test_rng(),
        )
        .expect("valid {0, s_d} witness must prove");

        let mut verifier_sponge = fresh_sponge();
        verify_relu_d_boolean(&proof, 7, 4, s_d, &commit, &params, &mut verifier_sponge)
            .expect("valid proof must verify");
    }

    #[test]
    fn malformed_d_lower_with_non_boolean_cell_prover_rejects() {
        // d_lower with one cell at d = s_d/2 (real-valued slope 0.5,
        // forbidden for canonical ReLU). The prover must fail fast —
        // it should NOT emit an Ok proof that the verifier later
        // rejects.
        let s_d = Fr::from(1u64 << 14);
        let half_sd = Fr::from(1u64 << 13); // s_d/2 — invalid
        let mut d: Vec<Fr> = Vec::with_capacity(16);
        for _ in 0..16 {
            d.push(Fr::from(0u64));
        }
        d[5] = half_sd; // single invalid cell
        d[12] = s_d; // one valid s_d to make it look like a real witness
        let params = setup_params(4);
        let (aux, commit) = commit_d(&d, &params);

        let mut prover_sponge = fresh_sponge();
        let result = prove_relu_d_boolean(
            7,
            &d,
            &aux,
            &commit,
            s_d,
            &params,
            &mut prover_sponge,
            &mut test_rng(),
        );
        // Either the round-invariant check or the final-identity
        // check inside the prover must fire.
        match result {
            Err(SnarkError::RelaxationSoundnessFinalCheckFailed { which }) => {
                assert!(
                    which == "relu_d_boolean::sumcheck_round_invariant"
                        || which == "relu_d_boolean::final_identity",
                    "unexpected which: {}",
                    which
                );
            }
            Err(other) => panic!("expected RelaxationSoundnessFinalCheckFailed, got {other:?}"),
            Ok(_) => panic!(
                "prover must fail fast on a non-{{0, s_d}} d_lower cell — \
                 instead it produced an Ok proof"
            ),
        }
    }

    #[test]
    fn forged_proof_with_non_boolean_d_lower_verifier_rejects() {
        // Adversarial scenario: the prover ALMOST honestly produces
        // a proof with malformed d_lower but bypasses the prover-side
        // fail-fast check. We simulate this by constructing a proof
        // honestly, then tampering the d_lower_eval to a forced 0.
        // The verifier must reject (final-identity check).
        let s_d = Fr::from(1u64 << 14);
        // Honest valid witness so the prover succeeds.
        let mut d: Vec<Fr> = Vec::with_capacity(16);
        for i in 0..16 {
            d.push(if i % 2 == 0 { Fr::from(0u64) } else { s_d });
        }
        let params = setup_params(4);
        let (aux, commit) = commit_d(&d, &params);

        let mut prover_sponge = fresh_sponge();
        let mut proof = prove_relu_d_boolean(
            42,
            &d,
            &aux,
            &commit,
            s_d,
            &params,
            &mut prover_sponge,
            &mut test_rng(),
        )
        .expect("honest valid witness");

        // Tamper: forge a different d_lower_eval. With overwhelming
        // probability (Schwartz-Zippel), the actual eval at the
        // sumcheck-final r' is non-zero (cells include both 0 and
        // s_d), so forcing it to 0 breaks the final identity.
        proof.d_lower_eval = Fr::from(0u64);

        let mut verifier_sponge = fresh_sponge();
        let err = verify_relu_d_boolean(&proof, 42, 4, s_d, &commit, &params, &mut verifier_sponge)
            .expect_err("tampered eval must be rejected");
        // Either Hyrax open verify or final-identity check rejects.
        // Both are valid rejection paths.
        assert!(
            matches!(
                err,
                SnarkError::PcsOpenRejected { .. }
                    | SnarkError::RelaxationSoundnessFinalCheckFailed { .. }
            ),
            "unexpected rejection error: {err:?}"
        );
    }
}
