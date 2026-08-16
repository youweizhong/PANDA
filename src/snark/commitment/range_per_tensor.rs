//! Per-committed-tensor range LogUp.
//!
//! Each public-witness tensor (input box, weights, biases, ReLU
//! relaxation coefficients) gets its own LogUp instance proving
//! that every committed cell lies in the signed-centered range
//! `T[i] = i − 2^k` over `i ∈ [0, 2^{k+1})`, with
//! `k = params.range_table_half_bits()` (a runtime public parameter).
//!
//! Both the witness column and the multiplicity column are bound at
//! the GKR-derived `bottom_point` (witness via the existing tensor
//! commit, multiplicities via a fresh per-instance commit). The
//! table side is closed-form via
//! [`crate::snark::commitment::table_mle::signed_centered_range_mle_eval`]
//! — no commit needed.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;

use crate::snark_primitives::logup_gkr::{
    prove_circuit as prove_logup_circuit, verify_circuit_with_top, LogUpCircuit, LogUpProof,
};
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use crate::snark::commitment::commit::CommittedAux;
use crate::snark::commitment::pcs_helpers::{hyrax_open_at, hyrax_verify_at, top_halves};
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;

/// Per-tensor range LogUp proof. The verifier supplies the
/// matching `tensor_commit` (already in `proof.commitments`) and
/// the architecture-derived `tensor_n_vars`; this proof does not
/// carry the tensor commit.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct TensorRangeProof {
    pub lookup_proof: LogUpProof<Fr>,
    pub table_proof: LogUpProof<Fr>,
    pub lookup_top: [Fr; 4],
    pub table_top: [Fr; 4],
    pub alpha: Fr,
    pub lookup_n_vars: usize,
    pub table_n_vars: usize,
    pub witness_len: usize,
    pub table_len: usize,
    /// Tensor-commit open at `lookup_proof.bottom_point`, binding
    /// the lookup-side `bottom_denom` to committed data.
    pub tensor_open: <HyraxBn254 as MlPcs>::Proof,
    pub tensor_eval: Fr,
    /// Multiplicity commit (committed before α is squeezed) and
    /// its open at `table_proof.bottom_point`.
    pub mult_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub mult_open: <HyraxBn254 as MlPcs>::Proof,
    pub mult_n_vars: usize,
}

fn absorb_commitment(
    sponge: &mut impl CryptographicSponge,
    commitment: &<HyraxBn254 as MlPcs>::Commitment,
) {
    let mut buf = Vec::new();
    commitment
        .serialize_compressed(&mut buf)
        .expect("serialize commitment");
    sponge.absorb(&buf);
}

/// Prove that every cell of a committed tensor lies in the
/// canonical signed range. The LogUp witness is the tensor's padded
/// MLE table (zero-pad cells trivially satisfy the range).
pub(crate) fn prove_tensor_range_logup(
    tensor_aux: &CommittedAux,
    tensor_commit: &<HyraxBn254 as MlPcs>::Commitment,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<TensorRangeProof, SnarkError> {
    let witness: &[Fr] = &tensor_aux.0;
    let table_fr: &[Fr] = &params.preprocessed.range_table_fr;
    let half = 1i128 << params.range_table_half_bits();

    // Cells outside the canonical range are dropped here; the
    // LogUp identity then fails for them, catching the misbehaviour.
    let mut mults = vec![0u64; table_fr.len()];
    for w in witness.iter() {
        let signed_i128 =
            crate::snark_primitives::finite_field::fr_to_signed_i128(*w).unwrap_or(i128::MAX);
        let idx = signed_i128 + half;
        if (0..(table_fr.len() as i128)).contains(&idx) {
            mults[idx as usize] += 1;
        }
    }
    let mults_fr: Vec<Fr> = mults.iter().map(|&m| Fr::from(m)).collect();

    // Commit multiplicities before α is squeezed. Hyrax needs an
    // even n_vars, so pad to one more variable when the table size
    // has odd log; the LogUp circuit still uses unpadded `mults_fr`.
    let mult_n_vars = {
        let nv = table_fr.len().trailing_zeros() as usize;
        if nv % 2 == 1 {
            nv + 1
        } else {
            nv
        }
    };
    let mut mults_padded = mults_fr.clone();
    mults_padded.resize(1usize << mult_n_vars, Fr::from(0u64));
    let (mult_commit, mult_state) =
        HyraxBn254::commit(&params.committer_key, &mults_padded, Some(rng))
            .map_err(SnarkError::Pcs)?;
    let mult_aux: CommittedAux = (mults_padded, mult_state);
    absorb_commitment(sponge, &mult_commit);

    sponge.absorb(&(witness.len() as u64));
    sponge.absorb(&(table_fr.len() as u64));
    let alpha = sponge.squeeze_field_elements::<Fr>(1)[0];
    sponge.absorb(&alpha);

    let lookup_circuit = LogUpCircuit::lookup(witness, alpha)?;
    let table_circuit = LogUpCircuit::table(table_fr, &mults_fr, alpha)?;
    let lookup_top = top_halves(&lookup_circuit);
    let table_top = top_halves(&table_circuit);
    let lookup_proof = prove_logup_circuit(&lookup_circuit, sponge)?;
    let table_proof = prove_logup_circuit(&table_circuit, sponge)?;

    // Bind lookup-side bottom_denom to the committed tensor.
    let (tensor_eval, tensor_open) = hyrax_open_at(
        &params.committer_key,
        tensor_aux,
        tensor_commit,
        &lookup_proof.bottom_point,
        sponge,
        rng,
    )?;
    debug_assert_eq!(
        lookup_proof.bottom_denom,
        tensor_eval - alpha,
        "per-tensor range: lookup bottom_denom must equal tensor_eval − α"
    );

    // Bind table-side bottom_num to the multiplicity commit.
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
        "per-tensor range: mult open eval must equal table_proof.bottom_num"
    );

    Ok(TensorRangeProof {
        lookup_proof,
        table_proof,
        lookup_top,
        table_top,
        alpha,
        lookup_n_vars: witness.len().trailing_zeros() as usize - 1,
        table_n_vars: table_fr.len().trailing_zeros() as usize - 1,
        witness_len: witness.len(),
        table_len: table_fr.len(),
        tensor_open,
        tensor_eval,
        mult_commit,
        mult_open,
        mult_n_vars,
    })
}

/// Verify a per-tensor range LogUp against `tensor_commit`.
/// `tensor_n_vars` is the native commit size derived from the
/// public architecture.
pub(crate) fn verify_tensor_range_logup(
    proof: &TensorRangeProof,
    tensor_commit: &<HyraxBn254 as MlPcs>::Commitment,
    tensor_n_vars: usize,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    if proof.witness_len != 1usize << tensor_n_vars {
        return Err(SnarkError::ArchitectureMismatch {
            what: "per-tensor range: witness_len does not match tensor_n_vars",
        });
    }
    // Pin the prover-claimed table width to the runtime public
    // parameter — the range check must run against exactly the
    // statement's signed table, never a wider prover-chosen one.
    let expected_table_len = 1usize << (params.range_table_half_bits() + 1);
    if proof.table_len != expected_table_len
        || proof.table_n_vars != expected_table_len.trailing_zeros() as usize - 1
    {
        return Err(SnarkError::LogUpTableNotCanonical {
            which: "per_tensor_range: table_len != 2^(range_table_half_bits + 1)",
        });
    }

    // Mirror prover transcript order: absorb mult_commit before α.
    absorb_commitment(sponge, &proof.mult_commit);
    sponge.absorb(&(proof.witness_len as u64));
    sponge.absorb(&(proof.table_len as u64));
    let alpha_check = sponge.squeeze_field_elements::<Fr>(1)[0];
    if alpha_check != proof.alpha {
        return Err(SnarkError::TranscriptMismatch);
    }
    sponge.absorb(&proof.alpha);

    verify_circuit_with_top(
        &proof.lookup_proof,
        proof.lookup_n_vars,
        proof.lookup_top,
        proof.lookup_top[0] * proof.lookup_top[3] + proof.lookup_top[1] * proof.lookup_top[2],
        sponge,
    )?;
    verify_circuit_with_top(
        &proof.table_proof,
        proof.table_n_vars,
        proof.table_top,
        proof.table_top[0] * proof.table_top[3] + proof.table_top[1] * proof.table_top[2],
        sponge,
    )?;

    // Top-fraction cancellation: lookup-side and table-side sums
    // must be exact negatives of one another.
    let lookup_frac = (
        proof.lookup_top[0] * proof.lookup_top[3] + proof.lookup_top[1] * proof.lookup_top[2],
        proof.lookup_top[2] * proof.lookup_top[3],
    );
    let table_frac = (
        proof.table_top[0] * proof.table_top[3] + proof.table_top[1] * proof.table_top[2],
        proof.table_top[2] * proof.table_top[3],
    );
    let combined = lookup_frac.0 * table_frac.1 + lookup_frac.1 * table_frac.0;
    if combined != Fr::from(0u64) {
        return Err(SnarkError::LogUpIdentityFailed);
    }

    // Table-side bottom_denom from the closed-form canonical MLE.
    let canonical_t_mle = crate::snark::commitment::table_mle::signed_centered_range_mle_eval(
        &proof.table_proof.bottom_point,
    );
    let expected_table_bottom_denom = canonical_t_mle - proof.alpha;
    if proof.table_proof.bottom_denom != expected_table_bottom_denom {
        return Err(SnarkError::LogUpTableNotCanonical {
            which: "per_tensor_range",
        });
    }

    // Lookup-side bottom_denom from the tensor commit open.
    let tensor_ok = hyrax_verify_at(
        &params.verifier_key,
        tensor_commit,
        &proof.lookup_proof.bottom_point,
        proof.tensor_eval,
        &proof.tensor_open,
        tensor_n_vars,
        sponge,
    )?;
    if !tensor_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "per_tensor_range_witness",
        });
    }
    if proof.lookup_proof.bottom_denom != proof.tensor_eval - proof.alpha {
        return Err(SnarkError::PerTensorRangeWitnessNotBound);
    }

    // Table-side bottom_num from the multiplicity commit open.
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
            which: "per_tensor_range_mult",
        });
    }

    Ok(())
}
