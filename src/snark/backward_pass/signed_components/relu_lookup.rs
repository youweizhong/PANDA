//! ReLU-decomposition lookup gadget.
//!
//! Proves `A_pos[i, j] = ReLU(A[i, j])` cell-wise via a LogUp into
//! the public table
//!
//! ```text
//! T_ReLU = { α · x + ReLU(x)  :  x ∈ [-2^k, 2^k) }
//! ```
//!
//! with witness `α · A + A_pos` over an FS-derived `α`. The gadget
//! runs once per `(A, A_pos)` pair, binds the bottom-layer LogUp
//! denominator to the committed tensors via batched Hyrax opens at
//! the LogUp final point, and binds the multiplicity vector via a
//! prove-time commit absorbed before the LogUp `β` is squeezed.
//!
//! The file also exports the eq-weighted two-product sumcheck
//! `claim = Σ_x eq[x] · (p1·q1 + p2·q2)`, which `activation_step`,
//! `activation_matrix`, and `concretize` all reuse.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::AdditiveGroup;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;

use crate::snark_primitives::sumcheck::{RoundPoly3, SumcheckError};

use crate::snark_primitives::finite_field::signed_lift_to_fr;
use crate::snark_primitives::logup_gkr::{
    prove_circuit as prove_logup_circuit, verify_circuit_with_top, LogUpCircuit, LogUpLayer,
    LogUpProof,
};
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use crate::snark::commitment::commit::CommittedAux;
use crate::snark::commitment::pcs_helpers::{
    hyrax_open_at, hyrax_open_batched_at, hyrax_verify_at, hyrax_verify_batched_at, BatchOpenSpec,
    BatchVerifySpec,
};
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;

// ---------------------------------------------------------------------------
// `eq · (p1·q1 + p2·q2)` sumcheck — used by activation_step,
// activation_matrix, and concretize.
// ---------------------------------------------------------------------------

/// Degree-3 sumcheck proof for `claim = Σ_x eq[x] · (p1[x]·q1[x] + p2[x]·q2[x])`.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct EqTwoProductProof {
    pub rounds: Vec<RoundPoly3<Fr>>,
    pub r_full: Vec<Fr>,
    pub eq_eval: Fr,
    pub p1_eval: Fr,
    pub q1_eval: Fr,
    pub p2_eval: Fr,
    pub q2_eval: Fr,
}

/// Prove `claim = Σ_x eq[x] · (p1[x] · q1[x] + p2[x] · q2[x])` via a
/// degree-3 sumcheck with Thaler-style in-place bookkeeping.
pub fn prove_eq_two_product_sumcheck(
    eq: &[Fr],
    p1: &[Fr],
    q1: &[Fr],
    p2: &[Fr],
    q2: &[Fr],
    claim: Fr,
    sponge: &mut impl CryptographicSponge,
) -> Result<EqTwoProductProof, crate::snark::errors::SnarkError> {
    let n = eq.len();
    if !n.is_power_of_two() || n == 0 {
        return Err(crate::snark::errors::SnarkError::ShapeMismatch {
            what: "eq-two-product: non-pow2 length",
        });
    }
    if p1.len() != n || q1.len() != n || p2.len() != n || q2.len() != n {
        return Err(crate::snark::errors::SnarkError::ShapeMismatch {
            what: "eq-two-product: vector length mismatch",
        });
    }
    let k = n.trailing_zeros() as usize;
    let mut eq_t = eq.to_vec();
    let mut p1_t = p1.to_vec();
    let mut q1_t = q1.to_vec();
    let mut p2_t = p2.to_vec();
    let mut q2_t = q2.to_vec();
    let mut current_sum = claim;
    let mut rounds: Vec<RoundPoly3<Fr>> = Vec::with_capacity(k);
    let mut r_full: Vec<Fr> = Vec::with_capacity(k);

    for _ in 0..k {
        let half = eq_t.len() / 2;
        let (mut e0, mut e1, mut e2, mut e3) = (
            Fr::from(0u64),
            Fr::from(0u64),
            Fr::from(0u64),
            Fr::from(0u64),
        );
        for i in 0..half {
            let q0 = eq_t[i];
            let q1v = eq_t[half + i];
            let q2v = q1v.double() - q0;
            let q3v = q1v.double() + q1v - q0.double();

            let pa0 = p1_t[i];
            let pa1 = p1_t[half + i];
            let pa2 = pa1.double() - pa0;
            let pa3 = pa1.double() + pa1 - pa0.double();

            let qa0 = q1_t[i];
            let qa1 = q1_t[half + i];
            let qa2 = qa1.double() - qa0;
            let qa3 = qa1.double() + qa1 - qa0.double();

            let pb0 = p2_t[i];
            let pb1 = p2_t[half + i];
            let pb2 = pb1.double() - pb0;
            let pb3 = pb1.double() + pb1 - pb0.double();

            let qb0 = q2_t[i];
            let qb1 = q2_t[half + i];
            let qb2 = qb1.double() - qb0;
            let qb3 = qb1.double() + qb1 - qb0.double();

            e0 += q0 * (pa0 * qa0 + pb0 * qb0);
            e1 += q1v * (pa1 * qa1 + pb1 * qb1);
            e2 += q2v * (pa2 * qa2 + pb2 * qb2);
            e3 += q3v * (pa3 * qa3 + pb3 * qb3);
        }
        let poly = RoundPoly3 {
            at_zero: e0,
            at_one: e1,
            at_two: e2,
            at_three: e3,
        };
        debug_assert_eq!(
            poly.at_zero + poly.at_one,
            current_sum,
            "eq-two-product: g(0)+g(1) ≠ incoming claim"
        );
        sponge.absorb(&poly.at_zero);
        sponge.absorb(&poly.at_one);
        sponge.absorb(&poly.at_two);
        sponge.absorb(&poly.at_three);
        let r = sponge.squeeze_field_elements::<Fr>(1)[0];
        for i in 0..half {
            let dq = eq_t[half + i] - eq_t[i];
            eq_t[i] += r * dq;
            let d1 = p1_t[half + i] - p1_t[i];
            p1_t[i] += r * d1;
            let d2 = q1_t[half + i] - q1_t[i];
            q1_t[i] += r * d2;
            let d3 = p2_t[half + i] - p2_t[i];
            p2_t[i] += r * d3;
            let d4 = q2_t[half + i] - q2_t[i];
            q2_t[i] += r * d4;
        }
        eq_t.truncate(half);
        p1_t.truncate(half);
        q1_t.truncate(half);
        p2_t.truncate(half);
        q2_t.truncate(half);
        current_sum = poly.evaluate(r);
        r_full.push(r);
        rounds.push(poly);
    }

    Ok(EqTwoProductProof {
        rounds,
        r_full,
        eq_eval: eq_t[0],
        p1_eval: p1_t[0],
        q1_eval: q1_t[0],
        p2_eval: p2_t[0],
        q2_eval: q2_t[0],
    })
}

/// Verify a proof produced by [`prove_eq_two_product_sumcheck`].
pub fn verify_eq_two_product_sumcheck(
    proof: &EqTwoProductProof,
    n_vars: usize,
    claim: Fr,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), crate::snark::errors::SnarkError> {
    if proof.rounds.len() != n_vars || proof.r_full.len() != n_vars {
        return Err(crate::snark::errors::SnarkError::ShapeMismatch {
            what: "eq-two-product: round count mismatch",
        });
    }
    let mut current_sum = claim;
    let mut r_full: Vec<Fr> = Vec::with_capacity(n_vars);
    for round in &proof.rounds {
        if round.at_zero + round.at_one != current_sum {
            return Err(crate::snark::errors::SnarkError::Sumcheck(
                SumcheckError::SplitMismatch,
            ));
        }
        sponge.absorb(&round.at_zero);
        sponge.absorb(&round.at_one);
        sponge.absorb(&round.at_two);
        sponge.absorb(&round.at_three);
        let r = sponge.squeeze_field_elements::<Fr>(1)[0];
        current_sum = round.evaluate(r);
        r_full.push(r);
    }
    if r_full != proof.r_full {
        return Err(crate::snark::errors::SnarkError::TranscriptMismatch);
    }
    let final_value =
        proof.eq_eval * (proof.p1_eval * proof.q1_eval + proof.p2_eval * proof.q2_eval);
    if final_value != current_sum {
        return Err(crate::snark::errors::SnarkError::Sumcheck(
            SumcheckError::FinalMismatch,
        ));
    }
    Ok(())
}

/// Per-step ReLU lookup proof.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ReluStepProof {
    pub combine_alpha: Fr,
    pub lookup_proof: LogUpProof<Fr>,
    pub table_proof: LogUpProof<Fr>,
    pub lookup_top: [Fr; 4],
    pub table_top: [Fr; 4],
    pub logup_beta: Fr,
    pub lookup_n_vars: usize,
    pub table_n_vars: usize,
    pub witness_len: usize,
    pub table_len: usize,
    /// Batched Hyrax open of `(A, A_pos)` at the LogUp final point.
    pub a_batched_open: <HyraxBn254 as MlPcs>::Proof,
    pub a_eval_at_r: Fr,
    pub a_pos_eval_at_r: Fr,
    /// Hyrax commit and open of the multiplicity MLE; absorbed
    /// before `β` is squeezed and opened at `table_proof.bottom_point`
    /// to bind `bottom_num` to a fixed witness.
    pub mult_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub mult_open: <HyraxBn254 as MlPcs>::Proof,
    pub mult_n_vars: usize,
}

/// Build the witness `α · A + A_pos` at the 2D-padded layout used by
/// `mle_table_from_matrix`. Padding cells contribute `(α·0 + 0) = 0`,
/// matching the table entry at value 0.
fn build_relu_witness_padded(
    a: &ndarray::Array2<i128>,
    a_pos: &ndarray::Array2<i128>,
    half_bits: i32,
    alpha: Fr,
) -> (Vec<Fr>, Vec<u64>) {
    debug_assert_eq!(a.shape(), a_pos.shape());
    let rows = a.nrows();
    let cols = a.ncols();
    let log_rows = crate::snark::commitment::multilinear_extensions::next_pow2_log(rows);
    let log_cols = crate::snark::commitment::multilinear_extensions::next_pow2_log(cols);
    let pow_rows = 1usize << log_rows;
    let pow_cols = 1usize << log_cols;
    let table_len = 1usize << (half_bits + 1);
    let mut mults = vec![0u64; table_len];
    let mut witness = vec![Fr::from(0u64); pow_rows * pow_cols];
    for i in 0..pow_rows {
        for j in 0..pow_cols {
            let (a_v, a_pos_v) = if i < rows && j < cols {
                (a[[i, j]], a_pos[[i, j]])
            } else {
                (0i128, 0i128)
            };
            witness[i * pow_cols + j] = alpha * signed_lift_to_fr(a_v) + signed_lift_to_fr(a_pos_v);
            let table_idx_i128 = a_v + (1i128 << half_bits);
            if (0..table_len as i128).contains(&table_idx_i128) {
                mults[table_idx_i128 as usize] += 1;
            }
        }
    }
    (witness, mults)
}

/// Extract the four-element top fraction `(n0, n1, d0, d1)` from a
/// LogUp circuit's final layer.
fn top_halves(circuit: &LogUpCircuit<Fr>) -> [Fr; 4] {
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

/// Prove `A_pos = ReLU(A)` for one matrix pair via the ReLU LogUp.
#[allow(clippy::too_many_arguments)]
pub fn prove_relu_step(
    a_codes: &ndarray::Array2<i128>,
    a_pos_codes: &ndarray::Array2<i128>,
    a_aux: &CommittedAux,
    a_commitment: &<HyraxBn254 as MlPcs>::Commitment,
    a_pos_aux: &CommittedAux,
    a_pos_commitment: &<HyraxBn254 as MlPcs>::Commitment,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<ReluStepProof, SnarkError> {
    if a_codes.shape() != a_pos_codes.shape() {
        return Err(SnarkError::ShapeMismatch {
            what: "ReLU-lookup: A vs A_pos shape",
        });
    }
    let n_padded = {
        let lr = crate::snark::commitment::multilinear_extensions::next_pow2_log(a_codes.nrows());
        let lc = crate::snark::commitment::multilinear_extensions::next_pow2_log(a_codes.ncols());
        1usize << (lr + lc)
    };
    sponge.absorb(&(n_padded as u64));
    let combine_alpha = sponge.squeeze_field_elements::<Fr>(1)[0];

    let (witness, mults) = build_relu_witness_padded(
        a_codes,
        a_pos_codes,
        params.range_table_half_bits(),
        combine_alpha,
    );
    // The preprocessed `(x, ReLU(x))` pairs blend with α here so we
    // avoid per-call i128→Fr conversions.
    let table = params.preprocessed.relu_table_at(combine_alpha);
    let mults_fr: Vec<Fr> = mults.iter().map(|&m| Fr::from(m)).collect();

    // Hyrax-commit the multiplicity vector BEFORE β is squeezed and
    // absorb the commit. Pins `m` to a witness the verifier later
    // opens against `table_proof.bottom_num`.
    let mult_n_vars = {
        let nv = (table.len() as f64).log2().round() as usize; // = 20 default
        let nv = if nv % 2 == 1 { nv + 1 } else { nv };
        nv.max(2)
    };
    let mult_padded_len = 1usize << mult_n_vars;
    debug_assert!(mults_fr.len() <= mult_padded_len);
    let mut mults_padded: Vec<Fr> = Vec::with_capacity(mult_padded_len);
    mults_padded.extend_from_slice(&mults_fr);
    mults_padded.resize(mult_padded_len, Fr::from(0u64));
    let (mult_commit, mult_state) =
        HyraxBn254::commit(&params.committer_key, &mults_padded, Some(rng))
            .map_err(SnarkError::Pcs)?;
    let mult_aux: CommittedAux = (mults_padded, mult_state);
    {
        let mut buf = Vec::new();
        ark_serialize::CanonicalSerialize::serialize_compressed(&mult_commit, &mut buf)
            .expect("serialize commitment");
        sponge.absorb(&buf);
    }

    sponge.absorb(&(witness.len() as u64));
    sponge.absorb(&(table.len() as u64));
    let logup_beta = sponge.squeeze_field_elements::<Fr>(1)[0];
    sponge.absorb(&logup_beta);

    let lookup_circuit = LogUpCircuit::lookup(&witness, logup_beta).map_err(SnarkError::LogUp)?;
    let table_circuit =
        LogUpCircuit::table(&table, &mults_fr, logup_beta).map_err(SnarkError::LogUp)?;
    let lookup_top = top_halves(&lookup_circuit);
    let table_top = top_halves(&table_circuit);
    let lookup_proof = prove_logup_circuit(&lookup_circuit, sponge).map_err(SnarkError::LogUp)?;
    let table_proof = prove_logup_circuit(&table_circuit, sponge).map_err(SnarkError::LogUp)?;

    // (A, A_pos) share shape and query point r_final, so batch them.
    let r_final = lookup_proof.bottom_point.clone();
    let a_n_vars = a_aux.0.len().trailing_zeros() as usize;
    debug_assert_eq!(
        a_n_vars,
        a_pos_aux.0.len().trailing_zeros() as usize,
        "ReLU lookup: A and A_pos must share commit n_vars"
    );
    let r_final_items = [
        BatchOpenSpec {
            aux: a_aux,
            commitment: a_commitment,
            commit_n_vars: a_n_vars,
        },
        BatchOpenSpec {
            aux: a_pos_aux,
            commitment: a_pos_commitment,
            commit_n_vars: a_n_vars,
        },
    ];
    let (r_final_vals, a_batched_open) =
        hyrax_open_batched_at(&params.committer_key, &r_final_items, &r_final, sponge, rng)?;
    let a_eval_at_r = r_final_vals[0];
    let a_pos_eval_at_r = r_final_vals[1];

    debug_assert_eq!(
        lookup_proof.bottom_denom,
        combine_alpha * a_eval_at_r + a_pos_eval_at_r - logup_beta,
        "ReLU-lookup: bottom denom must equal α·A + A_pos − β at r_final"
    );

    let (mult_eval_check, mult_open) = hyrax_open_at(
        &params.committer_key,
        &mult_aux,
        &mult_commit,
        &table_proof.bottom_point,
        sponge,
        rng,
    )?;
    debug_assert_eq!(
        mult_eval_check, table_proof.bottom_num,
        "ReLU-lookup: mult open eval must equal table_proof.bottom_num"
    );

    Ok(ReluStepProof {
        combine_alpha,
        lookup_proof,
        table_proof,
        lookup_top,
        table_top,
        logup_beta,
        lookup_n_vars: witness.len().trailing_zeros() as usize - 1,
        table_n_vars: table.len().trailing_zeros() as usize - 1,
        witness_len: witness.len(),
        table_len: table.len(),
        a_batched_open,
        a_eval_at_r,
        a_pos_eval_at_r,
        mult_commit,
        mult_open,
        mult_n_vars,
    })
}

/// Verify a [`ReluStepProof`] against the committed `(A, A_pos)`
/// tensors. `commit_n_vars` is the native commit size derived by the
/// caller from the public per-step shape.
pub fn verify_relu_step(
    proof: &ReluStepProof,
    a_commitment: &<HyraxBn254 as MlPcs>::Commitment,
    a_pos_commitment: &<HyraxBn254 as MlPcs>::Commitment,
    params: &SnarkParams,
    n_padded: usize,
    commit_n_vars: usize,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    sponge.absorb(&(n_padded as u64));
    let alpha_check = sponge.squeeze_field_elements::<Fr>(1)[0];
    if alpha_check != proof.combine_alpha {
        return Err(SnarkError::TranscriptMismatch);
    }
    // Pin the prover-claimed table width to the runtime public
    // parameter: with table size a runtime value, a prover must not be
    // able to range-check against a wider canonical table than the
    // statement's range_table_half_bits.
    let expected_table_len = 1usize << (params.range_table_half_bits() + 1);
    if proof.table_len != expected_table_len
        || proof.table_n_vars != expected_table_len.trailing_zeros() as usize - 1
    {
        return Err(SnarkError::LogUpTableNotCanonical {
            which: "relu_table: table_len != 2^(range_table_half_bits + 1)",
        });
    }
    // Absorb the multiplicity commit before β (mirrors prover).
    {
        let mut buf = Vec::new();
        ark_serialize::CanonicalSerialize::serialize_compressed(&proof.mult_commit, &mut buf)
            .expect("serialize commitment");
        sponge.absorb(&buf);
    }
    sponge.absorb(&(proof.witness_len as u64));
    sponge.absorb(&(proof.table_len as u64));
    let beta_check = sponge.squeeze_field_elements::<Fr>(1)[0];
    if beta_check != proof.logup_beta {
        return Err(SnarkError::TranscriptMismatch);
    }
    sponge.absorb(&proof.logup_beta);

    verify_circuit_with_top(
        &proof.lookup_proof,
        proof.lookup_n_vars,
        proof.lookup_top,
        proof.lookup_top[0] * proof.lookup_top[3] + proof.lookup_top[1] * proof.lookup_top[2],
        sponge,
    )
    .map_err(SnarkError::LogUp)?;
    verify_circuit_with_top(
        &proof.table_proof,
        proof.table_n_vars,
        proof.table_top,
        proof.table_top[0] * proof.table_top[3] + proof.table_top[1] * proof.table_top[2],
        sponge,
    )
    .map_err(SnarkError::LogUp)?;

    // Bind the table-side bottom denominator to the canonical ReLU
    // table MLE (closed-form, O(n) field ops; no precommit).
    let canonical_t_mle = crate::snark::commitment::table_mle::relu_table_mle_eval(
        &proof.table_proof.bottom_point,
        proof.combine_alpha,
    );
    let expected_table_bottom_denom = canonical_t_mle - proof.logup_beta;
    if proof.table_proof.bottom_denom != expected_table_bottom_denom {
        return Err(SnarkError::LogUpTableNotCanonical {
            which: "relu_table",
        });
    }

    let lookup_top_frac = (
        proof.lookup_top[0] * proof.lookup_top[3] + proof.lookup_top[1] * proof.lookup_top[2],
        proof.lookup_top[2] * proof.lookup_top[3],
    );
    let table_top_frac = (
        proof.table_top[0] * proof.table_top[3] + proof.table_top[1] * proof.table_top[2],
        proof.table_top[2] * proof.table_top[3],
    );
    let combined = lookup_top_frac.0 * table_top_frac.1 + lookup_top_frac.1 * table_top_frac.0;
    if combined != Fr::from(0u64) {
        return Err(SnarkError::ReluLookupIdentityFailed);
    }

    let r_final = proof.lookup_proof.bottom_point.clone();
    let r_final_items = [
        BatchVerifySpec {
            commitment: a_commitment,
            value: proof.a_eval_at_r,
            commit_n_vars,
        },
        BatchVerifySpec {
            commitment: a_pos_commitment,
            value: proof.a_pos_eval_at_r,
            commit_n_vars,
        },
    ];
    let r_final_ok = hyrax_verify_batched_at(
        &params.verifier_key,
        &r_final_items,
        &r_final,
        &proof.a_batched_open,
        sponge,
    )?;
    if !r_final_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "relu_A_A_pos_batch",
        });
    }

    let expected_bottom_denom =
        proof.combine_alpha * proof.a_eval_at_r + proof.a_pos_eval_at_r - proof.logup_beta;
    if proof.lookup_proof.bottom_denom != expected_bottom_denom {
        return Err(SnarkError::ReluLookupBindingFailed);
    }

    // Bind table-side bottom_num to the multiplicity commitment.
    let mult_ok = hyrax_verify_at(
        &params.verifier_key,
        &proof.mult_commit,
        &proof.table_proof.bottom_point,
        proof.table_proof.bottom_num,
        &proof.mult_open,
        proof.mult_n_vars,
        sponge,
    )?;
    if !mult_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "relu_lookup_mult",
        });
    }

    Ok(())
}
