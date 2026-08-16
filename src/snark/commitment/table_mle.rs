//! Closed-form MLE evaluations of public lookup tables.
//!
//! Each LogUp-GKR verifier site binds the table-side bottom
//! denominator to the canonical table by checking
//! `proof.table_proof.bottom_denom == T_canonical_mle(bottom_point) − α`.
//! Helpers here provide `T_canonical_mle` for the range, signed
//! range, ReLU, and sigmoid/tanh value tables — verifier-side O(n)
//! computations with no commit required for the table.

use ark_bn254::Fr;

/// MLE evaluation of the identity table `T[i] = i` over
/// `i ∈ [0, 2^n)`, with `n = point.len()`. Big-endian convention:
/// `point[0]` carries bit-weight `2^{n-1}`, so the closed form is
/// `Σ_j 2^{n-1-j} · point[j]`.
pub(crate) fn identity_mle_eval(point: &[Fr]) -> Fr {
    let mut acc = Fr::from(0u64);
    let mut pow = Fr::from(1u64);
    let two = Fr::from(2u64);
    for &r_j in point.iter().rev() {
        acc += pow * r_j;
        pow *= two;
    }
    acc
}

/// MLE evaluation of the signed-centered range table
/// `T[i] = i − 2^{n-1}` over `i ∈ [0, 2^n)`. Used by every
/// per-tensor range LogUp.
pub(crate) fn signed_centered_range_mle_eval(point: &[Fr]) -> Fr {
    let n = point.len();
    let half = pow2_fr(n.saturating_sub(1));
    identity_mle_eval(point) - half
}

/// MLE evaluation of the non-negative range table `T[i] = i`.
/// Aliases [`identity_mle_eval`] for readability at call sites
/// (output-bound positive-range table; rescale table).
pub(crate) fn pos_range_mle_eval(point: &[Fr]) -> Fr {
    identity_mle_eval(point)
}

/// MLE evaluation of the ReLU lookup table
/// `T[i] = α · x[i] + ReLU(x[i])` with `x[i] = i − 2^k` over
/// `i ∈ [0, 2^{k+1})`. `point[0]` is the high bit selecting the
/// non-negative half; `point[1..]` are the lower `k` bits.
///
/// Closed form: split `i = h · 2^k + y`, then `x = y − 2^k (1 − h)`
/// and `ReLU(x) = h · y`, giving
/// `T_mle = α (Y_mle − 2^k (1 − r_high)) + r_high · Y_mle` with
/// `Y_mle = identity_mle_eval(point[1..])`.
pub(crate) fn relu_table_mle_eval(point: &[Fr], alpha: Fr) -> Fr {
    let n = point.len();
    debug_assert!(n >= 1, "ReLU table must have ≥ 1 variable");
    let r_high = point[0];
    let r_low = &point[1..];
    let y_mle = identity_mle_eval(r_low);
    let two_k = pow2_fr(n - 1);
    let x_mle = y_mle - two_k * (Fr::from(1u64) - r_high);
    let relu_mle = r_high * y_mle;
    alpha * x_mle + relu_mle
}

/// MLE evaluation of the combined sigmoid/tanh table
/// `T[i] = α · i + σ_value[i]` over the positive half-table.
///
/// Uses MLE linearity to avoid materializing the combined table:
/// returns `α · identity_mle_eval(point) + eval_multilinear_full(sigma_values, point)`.
/// `sigma_values` must have length `2^point.len()` and is typically
/// one of `Preprocessed::{sigmoid,tanh}_{lower,upper}_fr`.
#[allow(dead_code)] // first user lands with the Phase 3b/3c gadgets.
pub(crate) fn sigma_value_table_mle_eval(point: &[Fr], alpha: Fr, sigma_values: &[Fr]) -> Fr {
    debug_assert_eq!(
        sigma_values.len(),
        1usize << point.len(),
        "σ_values length must equal 2^point.len()"
    );
    alpha * identity_mle_eval(point)
        + crate::snark::commitment::multilinear_extensions::eval_multilinear_full(
            sigma_values,
            point,
        )
}

/// `2^k` lifted to `Fr`. Used for the `−2^{n-1}` shift in the
/// signed range table and the `−2^k` shift in the ReLU table.
pub(crate) fn pow2_fr(k: usize) -> Fr {
    if k < 64 {
        Fr::from(1u64 << k)
    } else {
        // Slow path for `k >= 64`. Never reached at our table sizes
        // (`half_bits <= MAX_TABLE_BITS = 26`), kept for completeness.
        let mut acc = Fr::from(1u64);
        let two = Fr::from(2u64);
        for _ in 0..k {
            acc *= two;
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_poly::{DenseMultilinearExtension, Polynomial};
    use ark_std::{test_rng, UniformRand};

    /// Brute-force MLE eval via arkworks's `DenseMultilinearExtension`.
    /// Arkworks uses LE variable ordering, so the BE point is
    /// reversed before evaluation.
    fn ground_truth_mle_be(table: &[Fr], be_point: &[Fr]) -> Fr {
        debug_assert_eq!(table.len(), 1usize << be_point.len());
        let mle = DenseMultilinearExtension::from_evaluations_slice(be_point.len(), table);
        let le_point: Vec<Fr> = be_point.iter().rev().copied().collect();
        mle.evaluate(&le_point)
    }

    #[test]
    fn identity_mle_matches_ground_truth() {
        let mut rng = test_rng();
        for n in 1..=8usize {
            let table: Vec<Fr> = (0..(1usize << n)).map(|i| Fr::from(i as u64)).collect();
            let point: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let closed = identity_mle_eval(&point);
            let gt = ground_truth_mle_be(&table, &point);
            assert_eq!(closed, gt, "n={n}");
        }
    }

    #[test]
    fn signed_centered_range_mle_matches_ground_truth() {
        let mut rng = test_rng();
        for n in 1..=8usize {
            let half = 1i128 << (n - 1);
            let table: Vec<Fr> = (0..(1usize << n))
                .map(|i| {
                    // i ∈ [0, 2^n); signed value = i - 2^{n-1}.
                    let signed = i as i128 - half;
                    if signed >= 0 {
                        Fr::from(signed as u64)
                    } else {
                        -Fr::from((-signed) as u64)
                    }
                })
                .collect();
            let point: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let closed = signed_centered_range_mle_eval(&point);
            let gt = ground_truth_mle_be(&table, &point);
            assert_eq!(closed, gt, "n={n}");
        }
    }

    #[test]
    fn relu_table_mle_matches_ground_truth() {
        let mut rng = test_rng();
        for k in 1..=6usize {
            // Half-width `k` ⇒ table size 2^{k+1}, n_vars = k + 1.
            let n = k + 1;
            let half = 1i128 << k;
            let alpha = Fr::rand(&mut rng);
            // Build the canonical T[i] = α · x[i] + ReLU(x[i]) where
            // x[i] = i − 2^k.
            let table: Vec<Fr> = (0..(1usize << n))
                .map(|i| {
                    let x = i as i128 - half;
                    let x_fr = if x >= 0 {
                        Fr::from(x as u64)
                    } else {
                        -Fr::from((-x) as u64)
                    };
                    let relu = if x >= 0 {
                        Fr::from(x as u64)
                    } else {
                        Fr::from(0u64)
                    };
                    alpha * x_fr + relu
                })
                .collect();
            let point: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let closed = relu_table_mle_eval(&point, alpha);
            let gt = ground_truth_mle_be(&table, &point);
            assert_eq!(closed, gt, "k={k}");
        }
    }

    #[test]
    fn sigma_value_table_mle_matches_materialized_eval() {
        // The virtualization `α · identity_mle(r) + σ_mle(r)` must
        // equal the MLE of the materialized combined table
        // `T[i] = α · i + σ_value[i]` evaluated at the same point.
        // Tests with several pseudo-σ vectors at small n.
        let mut rng = test_rng();
        for n in 1..=6usize {
            let len = 1usize << n;
            let alpha = Fr::rand(&mut rng);
            // Pick a σ vector — anything works for the linearity check.
            let sigma_values: Vec<Fr> = (0..len).map(|_| Fr::rand(&mut rng)).collect();
            // Materialize the combined table.
            let combined: Vec<Fr> = (0..len)
                .map(|i| alpha * Fr::from(i as u64) + sigma_values[i])
                .collect();
            // Random FS point.
            let point: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let virtual_eval = sigma_value_table_mle_eval(&point, alpha, &sigma_values);
            let materialized_eval = ground_truth_mle_be(&combined, &point);
            assert_eq!(virtual_eval, materialized_eval, "n={n}");
        }
    }

    #[test]
    fn sigma_value_table_mle_works_at_real_sigmoid_table_size() {
        // Smoke test at the actual half-table size: 2^18. We don't
        // run the full materialized comparison (would allocate 8 MiB)
        // but sanity-check that the helper doesn't panic and returns
        // a value distinct from a freshly-derandomized run.
        use crate::snark::SigmaTables;
        let pre = SigmaTables::shared(
            crate::snark::preprocess::TEST_SIGMA_X_SCALE_LOG2,
            crate::snark::preprocess::TEST_SIGMA_V_SCALE_LOG2,
        );
        let n = pre.sigmoid_lower_fr.len().trailing_zeros() as usize;
        assert!(n >= 17, "sigmoid half-table must be at least 2^17 entries");
        let mut rng = test_rng();
        let alpha = Fr::rand(&mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let v = sigma_value_table_mle_eval(&point, alpha, &pre.sigmoid_lower_fr);
        // Different alpha must (with overwhelming probability) yield
        // a different value — confirms the eval actually depends on α.
        let alpha2 = alpha + Fr::from(1u64);
        let v2 = sigma_value_table_mle_eval(&point, alpha2, &pre.sigmoid_lower_fr);
        assert_ne!(v, v2);
    }

    #[test]
    fn pow2_fr_matches_naive() {
        for k in 0..=30usize {
            let v = pow2_fr(k);
            let naive: Fr = (0..k).fold(Fr::from(1u64), |acc, _| acc * Fr::from(2u64));
            assert_eq!(v, naive, "k={k}");
        }
    }

    /// Confirms that without the multiplicity-commit binding a
    /// prover can pick `m` after seeing β so the LogUp identity
    /// passes for an out-of-table witness. Justifies the
    /// commit-before-α step in [`super::super::range_per_tensor`].
    #[test]
    fn forge_test_unbound_multiplicity_gap() {
        use crate::snark_primitives::logup_gkr::{
            prove_circuit, verify_circuit_with_top, LogUpCircuit, LogUpLayer,
        };
        use ark_crypto_primitives::sponge::{merlin::Transcript, CryptographicSponge};
        use ark_ff::{Field, One};

        // Canonical table: T[i] = i for i ∈ [0, 4). Public, structured.
        // Witness contains 5 — which is NOT in T.
        let canonical_t: Vec<Fr> = (0..4u64).map(Fr::from).collect();
        let witness: Vec<Fr> = vec![
            Fr::from(0u64),
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(5u64), // <-- out-of-table value; an honest m doesn't exist.
        ];

        let mut sponge = <Transcript as CryptographicSponge>::new(&b"forge-mult".as_slice());
        sponge.absorb(&(witness.len() as u64));
        sponge.absorb(&(canonical_t.len() as u64));
        let beta = sponge.squeeze_field_elements::<Fr>(1)[0];
        sponge.absorb(&beta);

        // Forge: pick m so the LogUp identity holds at THIS β. We
        // need Σ_i 1/(W[i] − β) = Σ_j m[j] / (T[j] − β). Set
        // m[0] = (T[0] − β) · Σ_i 1/(W[i] − β); other m_j = 0.
        let lookup_sum: Fr = witness.iter().map(|w| (*w - beta).inverse().unwrap()).sum();
        let fake_m_0 = (canonical_t[0] - beta) * lookup_sum;
        let fake_m: Vec<Fr> = vec![fake_m_0, Fr::from(0u64), Fr::from(0u64), Fr::from(0u64)];

        // Build & prove the LogUp circuits with the forged m.
        let lookup_circuit = LogUpCircuit::lookup(&witness, beta).unwrap();
        let table_circuit = LogUpCircuit::table(&canonical_t, &fake_m, beta).unwrap();

        let lookup_proof = prove_circuit(&lookup_circuit, &mut sponge).unwrap();
        let table_proof = prove_circuit(&table_circuit, &mut sponge).unwrap();

        // Top fractions.
        let top_halves = |c: &LogUpCircuit<Fr>| -> [Fr; 4] {
            let top = c.layers.last().unwrap();
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
        };
        let lookup_top = top_halves(&lookup_circuit);
        let table_top = top_halves(&table_circuit);

        // ----- Run the verifier the same way the SNARK driver does -----
        let mut v_sponge = <Transcript as CryptographicSponge>::new(&b"forge-mult".as_slice());
        v_sponge.absorb(&(witness.len() as u64));
        v_sponge.absorb(&(canonical_t.len() as u64));
        let beta_check = v_sponge.squeeze_field_elements::<Fr>(1)[0];
        assert_eq!(beta, beta_check);
        v_sponge.absorb(&beta);

        // Step 1: GKR-internal verification of both circuits passes
        // by construction (each circuit is internally consistent).
        let lookup_top_claim = lookup_top[0] * lookup_top[3] + lookup_top[1] * lookup_top[2];
        verify_circuit_with_top(
            &lookup_proof,
            lookup_circuit.num_vars(),
            lookup_top,
            lookup_top_claim,
            &mut v_sponge,
        )
        .expect("lookup GKR consistency holds");
        let table_top_claim = table_top[0] * table_top[3] + table_top[1] * table_top[2];
        verify_circuit_with_top(
            &table_proof,
            table_circuit.num_vars(),
            table_top,
            table_top_claim,
            &mut v_sponge,
        )
        .expect("table GKR consistency holds");

        // Step 2: top-fraction cancellation passes — by construction
        // of the forged m_0.
        let lookup_frac = (
            lookup_top[0] * lookup_top[3] + lookup_top[1] * lookup_top[2],
            lookup_top[2] * lookup_top[3],
        );
        let table_frac = (
            table_top[0] * table_top[3] + table_top[1] * table_top[2],
            table_top[2] * table_top[3],
        );
        let combined = lookup_frac.0 * table_frac.1 + lookup_frac.1 * table_frac.0;
        assert_eq!(
            combined,
            Fr::from(0u64),
            "LogUp identity holds at THIS β (the whole point of the forgery)"
        );

        // Step 3: table-denominator canonical binding passes — T is
        // canonical, just m is forged.
        let canonical_t_mle = pos_range_mle_eval(&table_proof.bottom_point);
        let expected_bottom_denom = canonical_t_mle - beta;
        assert_eq!(
            table_proof.bottom_denom, expected_bottom_denom,
            "table-side bottom_denom matches canonical T_mle (T is honest, m is forged)"
        );

        // Step 4: HERE is the soundness gap. The verifier accepts up
        // to this point with the forged m. Only a multiplicity-side
        // binding would catch it. The forged m_mle at bottom_point
        // does NOT equal what an honest count function would yield —
        // and an honest count function doesn't even exist here
        // because witness contains 5 ∉ T.
        let forged_m_mle = ground_truth_mle_be(&fake_m, &table_proof.bottom_point);
        assert_eq!(
            table_proof.bottom_num, forged_m_mle,
            "GKR consistency: bottom_num matches forged m_mle (current code accepts this)"
        );
        // What the binding fix will check:
        //     table_proof.bottom_num == m_commit.open(bottom_point)
        // The prover would have to commit to `fake_m` BEFORE β is
        // squeezed. That's impossible: by construction, `fake_m_0`
        // depends on β. So the binding catches this attack.
    }

    /// Confirms a forged-table LogUp (table set equal to the
    /// witness, multiplicities all `1`) is internally consistent
    /// but produces a `bottom_denom` that differs from the
    /// canonical-MLE-derived expected value, so the binding check
    /// rejects.
    #[test]
    fn canonical_binding_rejects_forged_table() {
        use crate::snark_primitives::logup_gkr::{prove_circuit, LogUpCircuit};
        use ark_crypto_primitives::sponge::{merlin::Transcript, CryptographicSponge};

        // Fix a small witness: 4 cells, all distinct values that are
        // NOT the canonical range. Use fresh field elements so the
        // forged table = witness diverges from "i - 2^{n-1}".
        let witness: Vec<Fr> = vec![
            Fr::from(7u64),
            Fr::from(11u64),
            Fr::from(13u64),
            Fr::from(17u64),
        ];

        // Forged scenario: the prover claims the table is `witness`
        // itself, with multiplicity 1 everywhere.
        let mut prover_sponge =
            <Transcript as CryptographicSponge>::new(&b"forged-table-test".as_slice());
        // Squeeze the LogUp denominator shift α the same way the
        // production gadget does (after a witness-len + table-len
        // absorb).
        prover_sponge.absorb(&(witness.len() as u64));
        prover_sponge.absorb(&(witness.len() as u64));
        let alpha = prover_sponge.squeeze_field_elements::<Fr>(1)[0];
        prover_sponge.absorb(&alpha);

        let lookup_circuit = LogUpCircuit::lookup(&witness, alpha).unwrap();
        let forged_table = witness.clone();
        let mults: Vec<Fr> = vec![Fr::from(1u64); witness.len()];
        let table_circuit = LogUpCircuit::table(&forged_table, &mults, alpha).unwrap();

        let _lookup_proof = prove_circuit(&lookup_circuit, &mut prover_sponge).unwrap();
        let table_proof = prove_circuit(&table_circuit, &mut prover_sponge).unwrap();

        // The GKR table sumcheck verification by itself ACCEPTS the
        // forged table (the circuit is internally consistent) — we
        // would re-verify here for completeness, but the point of the
        // test is the next check.
        let table_top_arr = {
            use crate::snark_primitives::logup_gkr::LogUpLayer;
            use ark_ff::One;
            let top = table_circuit.layers.last().unwrap();
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
        };
        let mut verifier_sponge =
            <Transcript as CryptographicSponge>::new(&b"forged-table-test".as_slice());
        verifier_sponge.absorb(&(witness.len() as u64));
        verifier_sponge.absorb(&(witness.len() as u64));
        let alpha_check = verifier_sponge.squeeze_field_elements::<Fr>(1)[0];
        assert_eq!(alpha, alpha_check);
        verifier_sponge.absorb(&alpha);
        // Skip the lookup-side verify; just verify the forged table
        // circuit so we know the GKR by itself accepts.
        // (No-op for this test; we focus on the canonical-binding
        // check.)
        let _ = (table_top_arr, &table_proof);

        // The actual soundness check: my binding helper applied to
        // the forged table's bottom_denom MUST produce a different
        // value than the canonical T_mle(bottom_point) - α. If it
        // matched, the binding would NOT catch the forge.
        let canonical_t_mle = signed_centered_range_mle_eval(&table_proof.bottom_point);
        let canonical_expected = canonical_t_mle - alpha;
        assert_ne!(
            table_proof.bottom_denom, canonical_expected,
            "forged table's bottom_denom unexpectedly equals canonical T_mle - α; \
             the binding would NOT catch this forge"
        );

        // For sanity, the forged bottom_denom DOES match the forged
        // table's MLE eval (since GKR is internally consistent). We
        // recompute it via the same closed-form-on-witness-as-table.
        let forged_table_mle_eval = ground_truth_mle_be(&forged_table, &table_proof.bottom_point);
        assert_eq!(
            table_proof.bottom_denom,
            forged_table_mle_eval - alpha,
            "GKR consistency: bottom_denom must equal forged_T_mle - α"
        );
    }
}
