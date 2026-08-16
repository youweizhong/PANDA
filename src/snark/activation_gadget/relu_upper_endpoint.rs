//! Per-ReLU-layer gadget proving the upper-line `d_upper · x +
//! b_upper` dominates `ReLU(x)` at the two preactivation endpoints
//! `(preact_lower, preact_upper)` of every neuron. ReLU's convexity
//! plus the line's affineness then extends validity to the whole
//! interval `[l, u]`.
//!
//! The slack at the working scale is
//!
//! ```text
//!     slack[j] = (d_int[j] · preact_int[j]) / s_d
//!              + b_int[j] · (s_w / s_b)
//!              − relu_int[j]
//! ```
//!
//! which is integer-valued in `i128` under the layer-scale
//! precondition `s_b | s_w` (the gadget rejects with
//! `RelaxationSoundnessFinalCheckFailed` otherwise). For honest
//! lines the slack is `≥ 0`.
//!
//! The gadget bundles, per endpoint side:
//!
//! * a `[0, 2^GADGET_RANGE_BITS)` LogUp range check on `slack`
//!   and `epsilon` (multiplicities committed before β is squeezed,
//!   table-side bottom_denom bound against the canonical table MLE),
//! * a `(preact, relu) ⊆ T_ReLU` LogUp that binds the committed
//!   `relu` witness to `ReLU(preact)` without revealing preact codes,
//! * a degree-3 sumcheck on the slack identity, closed by a six-way
//!   batched Hyrax open of `(d, b, slack, epsilon, preact, relu)` at
//!   the sumcheck-final point.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::AdditiveGroup;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;

use crate::quantization::scale::Scale;
use crate::snark_primitives::finite_field::{fr_to_signed_i128, signed_lift_to_fr};
use crate::snark_primitives::logup_gkr::{
    prove_circuit as prove_logup_circuit, verify_circuit_with_top, LogUpCircuit, LogUpProof,
};
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};
use crate::snark_primitives::sumcheck::RoundPoly3;

use crate::snark::commitment::commit::CommittedAux;
use crate::snark::commitment::multilinear_extensions::{build_eq_table, eval_multilinear_full};
use crate::snark::commitment::pcs_helpers::{
    hyrax_open_at, hyrax_open_batched_at, hyrax_verify_at, hyrax_verify_batched_at, top_halves,
    BatchOpenSpec, BatchVerifySpec,
};
use crate::snark::errors::SnarkError;
use crate::snark::output_bound::{absorb_commitment, build_pos_multiplicities};
use crate::snark::params::SnarkParams;

/// LogUp range proof bundle for one Hyrax-committed witness vector
/// constrained to `[0, 2^GADGET_RANGE_BITS)`. Mirrors the shape
/// used in `output_bound::inequality`.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PosRangeLogUp {
    pub logup_alpha: Fr,
    pub logup_beta: Fr,
    pub lookup_proof: LogUpProof<Fr>,
    pub table_proof: LogUpProof<Fr>,
    pub lookup_top: [Fr; 4],
    pub table_top: [Fr; 4],
    pub witness_len: usize,
    pub table_len: usize,
    pub mult_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub mult_open: <HyraxBn254 as MlPcs>::Proof,
    pub mult_n_vars: usize,
    /// `witness(bottom_point) − β = lookup.bottom_denom`.
    pub witness_logup_open: <HyraxBn254 as MlPcs>::Proof,
    pub witness_logup_eval: Fr,
}

/// 1-D `(preact, relu) ⊆ T_ReLU` LogUp lookup proof. Binds the
/// committed `preact` and `relu` witnesses to the canonical ReLU
/// table `{(x, ReLU(x)) : x ∈ [-2^k, 2^k)}` so the verifier never
/// reads raw preact codes.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ReluLookup1dProof {
    pub combine_alpha: Fr,
    pub logup_beta: Fr,
    pub lookup_proof: LogUpProof<Fr>,
    pub table_proof: LogUpProof<Fr>,
    pub lookup_top: [Fr; 4],
    pub table_top: [Fr; 4],
    pub witness_len: usize,
    pub table_len: usize,
    pub mult_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub mult_open: <HyraxBn254 as MlPcs>::Proof,
    pub mult_n_vars: usize,
    /// Batched open of `(preact, relu)` at `lookup_proof.bottom_point`.
    pub witness_batched_open: <HyraxBn254 as MlPcs>::Proof,
    pub preact_logup_eval: Fr,
    pub relu_logup_eval: Fr,
}

/// Per-endpoint half of the ReLU upper-line validity proof.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ReluUpperEndpointHalf {
    pub n_vars: usize,
    /// Committed `relu_fr[j] = max(preact[j], 0)`, bound to
    /// `ReLU(preact)` via `relu_lookup`.
    pub relu_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub slack_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub epsilon_commit: <HyraxBn254 as MlPcs>::Commitment,
    /// `(preact, relu_fr) ⊆ T_ReLU` LogUp lookup.
    pub relu_lookup: ReluLookup1dProof,
    pub slack_range: PosRangeLogUp,
    pub epsilon_range: PosRangeLogUp,
    pub r_test: Vec<Fr>,
    pub rounds: Vec<RoundPoly3<Fr>>,
    pub r_final: Vec<Fr>,
    /// Batched open of `(d_upper, b_upper, slack, epsilon, preact,
    /// relu)` at `r_final` — all six evals are consumed by the
    /// slack-identity final check.
    pub r_batched_open: <HyraxBn254 as MlPcs>::Proof,
    pub d_upper_eval: Fr,
    pub b_upper_eval: Fr,
    pub slack_eval: Fr,
    pub epsilon_eval: Fr,
    pub preact_eval: Fr,
    pub relu_eval: Fr,
}

/// Per-ReLU-layer proof: `lo` (preact_lower side) + `hi`
/// (preact_upper side).
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ReluUpperEndpointProof {
    pub layer_idx: usize,
    pub n_vars: usize,
    pub lo: ReluUpperEndpointHalf,
    pub hi: ReluUpperEndpointHalf,
}

/// Prove `(preact[i], relu[i]) ⊆ T_ReLU` for all `i` in `0..n_padded`,
/// where the canonical table is
/// `{(x, ReLU(x)) : x ∈ [-2^k, 2^k)}` with `k =
/// params.range_table_half_bits()` (runtime public parameter). The
/// multiplicity commit is absorbed before β is squeezed, and the
/// lookup-side bottom denom is bound to `α · preact + relu − β` via a
/// batched Hyrax open.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_relu_lookup_1d(
    preact_padded: &[Fr],
    relu_padded: &[Fr],
    preact_aux: &CommittedAux,
    preact_commit: &<HyraxBn254 as MlPcs>::Commitment,
    relu_aux: &CommittedAux,
    relu_commit: &<HyraxBn254 as MlPcs>::Commitment,
    n_vars: usize,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<ReluLookup1dProof, SnarkError> {
    let _timing = crate::timing::lu_scope();
    let n_padded = 1usize << n_vars;
    if preact_padded.len() != n_padded
        || relu_padded.len() != n_padded
        || preact_aux.0.len() != n_padded
        || relu_aux.0.len() != n_padded
    {
        return Err(SnarkError::ShapeMismatch {
            what: "relu_lookup_1d: padded length != 2^n_vars",
        });
    }
    sponge.absorb(&(n_padded as u64));
    let combine_alpha = sponge.squeeze_field_elements::<Fr>(1)[0];

    let half_range = 1i128 << params.range_table_half_bits();
    let table = params.preprocessed.relu_table_at(combine_alpha);
    let table_len = table.len();

    // Fail-close on out-of-domain preact or `relu != ReLU(preact)`
    // per cell — soundness still rests on the LogUp identity, but
    // failing early surfaces clearer errors than a GKR-tree
    // disagreement would.
    let mut witness: Vec<Fr> = Vec::with_capacity(n_padded);
    let mut mults = vec![0u64; table_len];
    for i in 0..n_padded {
        let p_i128 =
            fr_to_signed_i128(preact_padded[i]).ok_or(SnarkError::FieldDecodeOutOfRange {
                which: "relu_lookup_1d: preact lift",
            })?;
        let r_i128 =
            fr_to_signed_i128(relu_padded[i]).ok_or(SnarkError::FieldDecodeOutOfRange {
                which: "relu_lookup_1d: relu lift",
            })?;
        if p_i128 < -half_range || p_i128 >= half_range {
            return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
                which: "relu_lookup_1d: preact out of T_ReLU domain",
            });
        }
        let expected_relu = if p_i128 > 0 { p_i128 } else { 0 };
        if r_i128 != expected_relu {
            return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
                which: "relu_lookup_1d: relu != ReLU(preact) per cell",
            });
        }
        let table_idx = (p_i128 + half_range) as usize;
        if table_idx >= table_len {
            return Err(SnarkError::ShapeMismatch {
                what: "relu_lookup_1d: table_idx >= table_len (internal)",
            });
        }
        mults[table_idx] += 1;
        witness.push(combine_alpha * preact_padded[i] + relu_padded[i]);
    }
    let mults_fr: Vec<Fr> = mults.iter().map(|&m| Fr::from(m)).collect();

    // Commit multiplicities BEFORE β is squeezed (m-after-β forge
    // defense).
    let mult_n_vars = {
        let nv = (table_len as f64).log2().round() as usize;
        let nv = if nv % 2 == 1 { nv + 1 } else { nv };
        nv.max(2)
    };
    let mult_padded_len = 1usize << mult_n_vars;
    let mut mults_padded: Vec<Fr> = mults_fr.clone();
    mults_padded.resize(mult_padded_len, Fr::from(0u64));
    let (mult_commit, mult_state) =
        HyraxBn254::commit(&params.committer_key, &mults_padded, Some(rng))
            .map_err(SnarkError::Pcs)?;
    let mult_aux: CommittedAux = (mults_padded, mult_state);
    absorb_commitment(sponge, &mult_commit);

    sponge.absorb(&(witness.len() as u64));
    sponge.absorb(&(table_len as u64));
    let logup_beta = sponge.squeeze_field_elements::<Fr>(1)[0];
    sponge.absorb(&logup_beta);

    let lookup_circuit = LogUpCircuit::lookup(&witness, logup_beta).map_err(SnarkError::LogUp)?;
    let table_circuit =
        LogUpCircuit::table(&table, &mults_fr, logup_beta).map_err(SnarkError::LogUp)?;
    let lookup_top = top_halves(&lookup_circuit);
    let table_top = top_halves(&table_circuit);
    let lookup_proof = prove_logup_circuit(&lookup_circuit, sponge).map_err(SnarkError::LogUp)?;
    let table_proof = prove_logup_circuit(&table_circuit, sponge).map_err(SnarkError::LogUp)?;

    let r_logup = lookup_proof.bottom_point.clone();
    let items = [
        BatchOpenSpec {
            aux: preact_aux,
            commitment: preact_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: relu_aux,
            commitment: relu_commit,
            commit_n_vars: n_vars,
        },
    ];
    let (vals, witness_batched_open) =
        hyrax_open_batched_at(&params.committer_key, &items, &r_logup, sponge, rng)?;
    let preact_logup_eval = vals[0];
    let relu_logup_eval = vals[1];

    // Prover self-check; the verifier re-checks this identity against
    // its own opened evals.
    if lookup_proof.bottom_denom != combine_alpha * preact_logup_eval + relu_logup_eval - logup_beta
    {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "relu_lookup_1d: prover-side bottom_denom vs (α·p + r − β) drift",
        });
    }

    let mult_bottom_pt = table_proof.bottom_point.clone();
    let (mult_eval_check, mult_open) = hyrax_open_at(
        &params.committer_key,
        &mult_aux,
        &mult_commit,
        &mult_bottom_pt,
        sponge,
        rng,
    )?;
    if mult_eval_check != table_proof.bottom_num {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "relu_lookup_1d: prover-side mult open vs table.bottom_num drift",
        });
    }

    Ok(ReluLookup1dProof {
        combine_alpha,
        logup_beta,
        lookup_proof,
        table_proof,
        lookup_top,
        table_top,
        witness_len: witness.len(),
        table_len,
        mult_commit,
        mult_open,
        mult_n_vars,
        witness_batched_open,
        preact_logup_eval,
        relu_logup_eval,
    })
}

/// Verifier for [`prove_relu_lookup_1d`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_relu_lookup_1d(
    proof: &ReluLookup1dProof,
    preact_commit: &<HyraxBn254 as MlPcs>::Commitment,
    relu_commit: &<HyraxBn254 as MlPcs>::Commitment,
    n_vars: usize,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    let n_padded = 1usize << n_vars;
    sponge.absorb(&(n_padded as u64));
    let combine_alpha = sponge.squeeze_field_elements::<Fr>(1)[0];
    if combine_alpha != proof.combine_alpha {
        return Err(SnarkError::TranscriptMismatch);
    }
    // Reject non-canonical / oddly-sized tables here — otherwise a
    // forged table could pass the GKR sumcheck against a forged top.
    let canonical_table_len = params.preprocessed.relu_table_at(combine_alpha).len();
    if proof.table_len != canonical_table_len {
        return Err(SnarkError::LogUpTableNotCanonical {
            which: "relu_lookup_1d: table_len != canonical T_ReLU.len()",
        });
    }
    if !proof.table_len.is_power_of_two() || proof.table_len < 4 {
        return Err(SnarkError::ShapeMismatch {
            what: "relu_lookup_1d: table_len must be a power of two ≥ 4",
        });
    }
    // LogUp's bottom_point length is exactly `log2(table_len)`.
    let expected_bottom_n = proof.table_len.trailing_zeros() as usize;
    if proof.table_proof.bottom_point.len() != expected_bottom_n {
        return Err(SnarkError::ShapeMismatch {
            what: "relu_lookup_1d: table_proof.bottom_point.len() != log2(table_len)",
        });
    }
    absorb_commitment(sponge, &proof.mult_commit);
    sponge.absorb(&(proof.witness_len as u64));
    sponge.absorb(&(proof.table_len as u64));
    let logup_beta = sponge.squeeze_field_elements::<Fr>(1)[0];
    sponge.absorb(&logup_beta);
    if logup_beta != proof.logup_beta {
        return Err(SnarkError::TranscriptMismatch);
    }
    if proof.witness_len != n_padded {
        return Err(SnarkError::ShapeMismatch {
            what: "relu_lookup_1d: witness_len != 2^n_vars",
        });
    }

    let lookup_n = (proof.witness_len.trailing_zeros() as usize).saturating_sub(1);
    let table_n = (proof.table_len.trailing_zeros() as usize).saturating_sub(1);
    let lookup_top_num =
        proof.lookup_top[0] * proof.lookup_top[3] + proof.lookup_top[1] * proof.lookup_top[2];
    let table_top_num =
        proof.table_top[0] * proof.table_top[3] + proof.table_top[1] * proof.table_top[2];
    verify_circuit_with_top(
        &proof.lookup_proof,
        lookup_n,
        proof.lookup_top,
        lookup_top_num,
        sponge,
    )
    .map_err(SnarkError::LogUp)?;
    verify_circuit_with_top(
        &proof.table_proof,
        table_n,
        proof.table_top,
        table_top_num,
        sponge,
    )
    .map_err(SnarkError::LogUp)?;

    let r_logup = proof.lookup_proof.bottom_point.clone();
    let items = [
        BatchVerifySpec {
            commitment: preact_commit,
            commit_n_vars: n_vars,
            value: proof.preact_logup_eval,
        },
        BatchVerifySpec {
            commitment: relu_commit,
            commit_n_vars: n_vars,
            value: proof.relu_logup_eval,
        },
    ];
    let r_open_ok = hyrax_verify_batched_at(
        &params.verifier_key,
        &items,
        &r_logup,
        &proof.witness_batched_open,
        sponge,
    )?;
    if !r_open_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "relu_lookup_1d: witness batched open",
        });
    }

    if proof.lookup_proof.bottom_denom
        != combine_alpha * proof.preact_logup_eval + proof.relu_logup_eval - logup_beta
    {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "relu_lookup_1d: bottom_denom != α·preact + relu − β at bottom_point",
        });
    }

    let mult_bottom_pt = proof.table_proof.bottom_point.clone();
    let mult_open_ok = hyrax_verify_at(
        &params.verifier_key,
        &proof.mult_commit,
        &mult_bottom_pt,
        proof.table_proof.bottom_num,
        &proof.mult_open,
        proof.mult_n_vars,
        sponge,
    )?;
    if !mult_open_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "relu_lookup_1d: mult open",
        });
    }

    let canonical_t_mle = crate::snark::commitment::table_mle::relu_table_mle_eval(
        &proof.table_proof.bottom_point,
        combine_alpha,
    );
    if proof.table_proof.bottom_denom != canonical_t_mle - logup_beta {
        return Err(SnarkError::LogUpTableNotCanonical {
            which: "relu_lookup_1d: table not canonical",
        });
    }

    Ok(())
}

pub(crate) fn squeeze_round_challenge_3(
    sponge: &mut impl CryptographicSponge,
    poly: &RoundPoly3<Fr>,
) -> Fr {
    let mut buf = Vec::new();
    poly.serialize_compressed(&mut buf)
        .expect("serialize round poly");
    sponge.absorb(&buf);
    sponge.squeeze_field_elements::<Fr>(1)[0]
}

pub(crate) fn squeeze_round_challenge_4(
    sponge: &mut impl CryptographicSponge,
    poly: &crate::snark_primitives::sumcheck::RoundPoly4<Fr>,
) -> Fr {
    let mut buf = Vec::new();
    poly.serialize_compressed(&mut buf)
        .expect("serialize round poly");
    sponge.absorb(&buf);
    sponge.squeeze_field_elements::<Fr>(1)[0]
}

/// Compute `(slack, epsilon)` at scale `s_w` in `i128`, where
///
/// ```text
///     slack[j] * s_d + epsilon[j] = d[j] * preact[j]
///                                 + b[j] * (s_w / s_b) * s_d
///                                 - relu[j] * s_d
/// ```
///
/// and `epsilon[j] ∈ [0, s_d)` is the floor-division remainder.
/// Caller precondition: `s_w % s_b == 0`. For valid relaxation lines
/// the RHS is non-negative, so `slack ≥ 0`.
fn compute_slack_and_epsilon(
    d_int: i128,
    b_int: i128,
    preact_int: i128,
    relu_int: i128,
    s_d_code: i128,
    s_w_code: i128,
    s_b_code: i128,
) -> Result<(i128, i128), SnarkError> {
    if s_w_code % s_b_code != 0 {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "relu_upper_endpoint: s_b ∤ s_w (precondition q_b > q_w)",
        });
    }
    let s_w_over_s_b = s_w_code / s_b_code;

    let prod_d_preact = d_int
        .checked_mul(preact_int)
        .ok_or(SnarkError::ShapeMismatch {
            what: "relu_upper_endpoint: d*preact i128 overflow",
        })?;
    let term_b_scaled = b_int
        .checked_mul(s_w_over_s_b)
        .and_then(|v| v.checked_mul(s_d_code))
        .ok_or(SnarkError::ShapeMismatch {
            what: "relu_upper_endpoint: b*(s_w/s_b)*s_d i128 overflow",
        })?;
    let term_relu_scaled = relu_int
        .checked_mul(s_d_code)
        .ok_or(SnarkError::ShapeMismatch {
            what: "relu_upper_endpoint: relu*s_d i128 overflow",
        })?;

    // Numerator at scale s_d · s_w — must be ≥ 0 for valid lines.
    let numerator = prod_d_preact
        .checked_add(term_b_scaled)
        .ok_or(SnarkError::ShapeMismatch {
            what: "relu_upper_endpoint: numerator add overflow",
        })?
        .checked_sub(term_relu_scaled)
        .ok_or(SnarkError::ShapeMismatch {
            what: "relu_upper_endpoint: numerator sub overflow",
        })?;

    if numerator < 0 {
        return Err(SnarkError::RelaxationSoundnessReluUpperEndpointInvalid {
            layer_idx: 0, // caller patches this with the actual neuron idx
            endpoint: "numerator < 0 ⇒ committed line < ReLU at this neuron's endpoint",
        });
    }

    let slack = numerator / s_d_code;
    let epsilon = numerator - slack * s_d_code;
    debug_assert!(epsilon >= 0 && epsilon < s_d_code);
    Ok((slack, epsilon))
}

/// Run a LogUp range check `witness ⊆ [0, 2^GADGET_RANGE_BITS)`
/// and bind the multiplicity commit plus the witness-side bottom
/// denom in one go. This is the shared per-neuron range check used by
/// every activation gadget; it always runs at the narrow gadget
/// budget, never the wide out-bound one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_pos_range(
    witness_fr: &[Fr],
    witness_i128: &[i128],
    witness_aux: &CommittedAux,
    witness_commit: &<HyraxBn254 as MlPcs>::Commitment,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<PosRangeLogUp, SnarkError> {
    let _timing = crate::timing::lu_scope();
    // The gadget budget is this proof's runtime parameter (19
    // default); the matching table was prebuilt.
    let table_bits = params.gadget_range_bits;
    let table: &[Fr] = params.preprocessed.pos_range_table(table_bits)?;
    let n_padded = witness_fr.len();
    let bound = 1i128 << table_bits;
    for &v in witness_i128.iter() {
        if v < 0 || v >= bound {
            return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
                which: "relu_upper_endpoint: witness out of [0, 2^GADGET_RANGE_BITS)",
            });
        }
    }
    let mults = build_pos_multiplicities(witness_i128, table_bits);
    let mults_fr: Vec<Fr> = mults.iter().map(|&m| Fr::from(m)).collect();

    let mult_n_vars = {
        let nv = table_bits;
        if nv % 2 == 1 { nv + 1 } else { nv }.max(2)
    };
    let mult_padded_len = 1usize << mult_n_vars;
    let mut mults_padded: Vec<Fr> = Vec::with_capacity(mult_padded_len);
    mults_padded.extend_from_slice(&mults_fr);
    mults_padded.resize(mult_padded_len, Fr::from(0u64));
    let (mult_commit, mult_state) =
        HyraxBn254::commit(&params.committer_key, &mults_padded, Some(rng))
            .map_err(SnarkError::Pcs)?;
    let mult_aux: CommittedAux = (mults_padded, mult_state);
    absorb_commitment(sponge, &mult_commit);

    sponge.absorb(&(n_padded as u64));
    sponge.absorb(&(table.len() as u64));
    let logup_alpha = sponge.squeeze_field_elements::<Fr>(1)[0];
    let logup_beta = sponge.squeeze_field_elements::<Fr>(1)[0];
    sponge.absorb(&logup_alpha);
    sponge.absorb(&logup_beta);

    let lookup_circuit = LogUpCircuit::lookup(witness_fr, logup_beta).map_err(SnarkError::LogUp)?;
    let table_circuit =
        LogUpCircuit::table(table, &mults_fr, logup_beta).map_err(SnarkError::LogUp)?;
    let lookup_top = top_halves(&lookup_circuit);
    let table_top = top_halves(&table_circuit);
    let lookup_proof = prove_logup_circuit(&lookup_circuit, sponge).map_err(SnarkError::LogUp)?;
    let table_proof = prove_logup_circuit(&table_circuit, sponge).map_err(SnarkError::LogUp)?;

    let mult_bottom_pt = table_proof.bottom_point.clone();
    let (mult_eval_check, mult_open) = hyrax_open_at(
        &params.committer_key,
        &mult_aux,
        &mult_commit,
        &mult_bottom_pt,
        sponge,
        rng,
    )?;
    if mult_eval_check != table_proof.bottom_num {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "relu_upper_endpoint: mult open mismatch",
        });
    }

    let logup_point = lookup_proof.bottom_point.clone();
    let (witness_logup_eval, witness_logup_open) = hyrax_open_at(
        &params.committer_key,
        witness_aux,
        witness_commit,
        &logup_point,
        sponge,
        rng,
    )?;
    if lookup_proof.bottom_denom != witness_logup_eval - logup_beta {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "relu_upper_endpoint: witness logup bottom_denom mismatch",
        });
    }

    Ok(PosRangeLogUp {
        logup_alpha,
        logup_beta,
        lookup_proof,
        table_proof,
        lookup_top,
        table_top,
        witness_len: n_padded,
        table_len: table.len(),
        mult_commit,
        mult_open,
        mult_n_vars,
        witness_logup_open,
        witness_logup_eval,
    })
}

pub(crate) fn verify_pos_range(
    range: &PosRangeLogUp,
    expected_n_vars: usize,
    witness_commit: &<HyraxBn254 as MlPcs>::Commitment,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    let table: &[Fr] = params
        .preprocessed
        .pos_range_table(params.gadget_range_bits)?;
    let n_padded = 1usize << expected_n_vars;
    if range.witness_len != n_padded || range.table_len != table.len() {
        return Err(SnarkError::ShapeMismatch {
            what: "relu_upper_endpoint: range LogUp len",
        });
    }
    absorb_commitment(sponge, &range.mult_commit);
    sponge.absorb(&(range.witness_len as u64));
    sponge.absorb(&(range.table_len as u64));
    let logup_alpha = sponge.squeeze_field_elements::<Fr>(1)[0];
    let logup_beta = sponge.squeeze_field_elements::<Fr>(1)[0];
    sponge.absorb(&logup_alpha);
    sponge.absorb(&logup_beta);
    if logup_alpha != range.logup_alpha || logup_beta != range.logup_beta {
        return Err(SnarkError::TranscriptMismatch);
    }

    let lookup_n = (range.witness_len.trailing_zeros() as usize).saturating_sub(1);
    let table_n = (range.table_len.trailing_zeros() as usize).saturating_sub(1);
    let lookup_top_num =
        range.lookup_top[0] * range.lookup_top[3] + range.lookup_top[1] * range.lookup_top[2];
    let table_top_num =
        range.table_top[0] * range.table_top[3] + range.table_top[1] * range.table_top[2];
    verify_circuit_with_top(
        &range.lookup_proof,
        lookup_n,
        range.lookup_top,
        lookup_top_num,
        sponge,
    )
    .map_err(SnarkError::LogUp)?;
    verify_circuit_with_top(
        &range.table_proof,
        table_n,
        range.table_top,
        table_top_num,
        sponge,
    )
    .map_err(SnarkError::LogUp)?;

    let mult_bottom_pt = range.table_proof.bottom_point.clone();
    let mult_open_ok = hyrax_verify_at(
        &params.verifier_key,
        &range.mult_commit,
        &mult_bottom_pt,
        range.table_proof.bottom_num,
        &range.mult_open,
        range.mult_n_vars,
        sponge,
    )?;
    if !mult_open_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "relu_upper_endpoint::range_mult",
        });
    }

    let logup_point = range.lookup_proof.bottom_point.clone();
    let witness_logup_ok = hyrax_verify_at(
        &params.verifier_key,
        witness_commit,
        &logup_point,
        range.witness_logup_eval,
        &range.witness_logup_open,
        expected_n_vars,
        sponge,
    )?;
    if !witness_logup_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "relu_upper_endpoint::range_witness_logup",
        });
    }
    if range.lookup_proof.bottom_denom != range.witness_logup_eval - logup_beta {
        return Err(SnarkError::PerTensorRangeWitnessNotBound);
    }

    let canonical_table_eval =
        crate::snark::commitment::table_mle::pos_range_mle_eval(&range.table_proof.bottom_point);
    if range.table_proof.bottom_denom != canonical_table_eval - logup_beta {
        return Err(SnarkError::LogUpTableNotCanonical {
            which: "relu_upper_endpoint::pos_range_table",
        });
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prove_half(
    n_vars: usize,
    preact_aux: &CommittedAux,
    preact_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_aux: &CommittedAux,
    d_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_aux: &CommittedAux,
    b_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s_d: Fr,
    c_b_sd: Fr,
    s_d_code: i128,
    s_w_code: i128,
    s_b_code: i128,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<ReluUpperEndpointHalf, SnarkError> {
    let n_padded = 1usize << n_vars;
    let preact_padded = &preact_aux.0;
    if preact_padded.len() != n_padded || d_aux.0.len() != n_padded || b_aux.0.len() != n_padded {
        return Err(SnarkError::ShapeMismatch {
            what: "relu_upper_endpoint: padded length != 2^n_vars",
        });
    }

    // Compute (relu, slack, epsilon) per cell from the committed
    // preact aux. The verifier never sees raw preact codes; relu_fr
    // is committed and bound via the LogUp lookup below.
    let mut slack_i128: Vec<i128> = Vec::with_capacity(n_padded);
    let mut epsilon_i128: Vec<i128> = Vec::with_capacity(n_padded);
    let mut slack_fr: Vec<Fr> = Vec::with_capacity(n_padded);
    let mut epsilon_fr: Vec<Fr> = Vec::with_capacity(n_padded);
    let mut relu_fr: Vec<Fr> = Vec::with_capacity(n_padded);
    let d_padded = &d_aux.0;
    let b_padded = &b_aux.0;
    for i in 0..n_padded {
        let preact_i128 =
            fr_to_signed_i128(preact_padded[i]).ok_or(SnarkError::FieldDecodeOutOfRange {
                which: "relu_upper_endpoint: preact lift",
            })?;
        let d_i128 = fr_to_signed_i128(d_padded[i]).ok_or(SnarkError::FieldDecodeOutOfRange {
            which: "relu_upper_endpoint: d_upper lift",
        })?;
        let b_i128 = fr_to_signed_i128(b_padded[i]).ok_or(SnarkError::FieldDecodeOutOfRange {
            which: "relu_upper_endpoint: b_upper lift",
        })?;
        let relu_i128 = if preact_i128 > 0 { preact_i128 } else { 0 };
        let (s, eps) = compute_slack_and_epsilon(
            d_i128,
            b_i128,
            preact_i128,
            relu_i128,
            s_d_code,
            s_w_code,
            s_b_code,
        )
        .map_err(|e| match e {
            SnarkError::RelaxationSoundnessReluUpperEndpointInvalid { endpoint, .. } => {
                SnarkError::RelaxationSoundnessReluUpperEndpointInvalid {
                    layer_idx: i,
                    endpoint,
                }
            }
            other => other,
        })?;
        slack_i128.push(s);
        epsilon_i128.push(eps);
        slack_fr.push(signed_lift_to_fr(s));
        epsilon_fr.push(signed_lift_to_fr(eps));
        relu_fr.push(signed_lift_to_fr(relu_i128));
    }

    sponge.absorb(&(n_vars as u64));
    absorb_commitment(sponge, d_commit);
    absorb_commitment(sponge, b_commit);
    absorb_commitment(sponge, preact_commit);

    let (relu_commit, relu_state) =
        HyraxBn254::commit(&params.committer_key, &relu_fr, Some(rng)).map_err(SnarkError::Pcs)?;
    let relu_aux: CommittedAux = (relu_fr.clone(), relu_state);
    absorb_commitment(sponge, &relu_commit);

    let relu_lookup = prove_relu_lookup_1d(
        preact_padded,
        &relu_fr,
        preact_aux,
        preact_commit,
        &relu_aux,
        &relu_commit,
        n_vars,
        params,
        sponge,
        rng,
    )?;

    let (slack_commit, slack_state) =
        HyraxBn254::commit(&params.committer_key, &slack_fr, Some(rng)).map_err(SnarkError::Pcs)?;
    let slack_aux: CommittedAux = (slack_fr.clone(), slack_state);
    absorb_commitment(sponge, &slack_commit);
    let (epsilon_commit, epsilon_state) =
        HyraxBn254::commit(&params.committer_key, &epsilon_fr, Some(rng))
            .map_err(SnarkError::Pcs)?;
    let epsilon_aux: CommittedAux = (epsilon_fr.clone(), epsilon_state);
    absorb_commitment(sponge, &epsilon_commit);

    let slack_range = prove_pos_range(
        &slack_fr,
        &slack_i128,
        &slack_aux,
        &slack_commit,
        params,
        sponge,
        rng,
    )?;
    let epsilon_range = prove_pos_range(
        &epsilon_fr,
        &epsilon_i128,
        &epsilon_aux,
        &epsilon_commit,
        params,
        sponge,
        rng,
    )?;

    // Slack identity sumcheck:
    //   Σ_j eq(j, r) · (slack·s_d + epsilon
    //                   - d·preact - b·c_b_sd + relu·s_d) = 0
    let r_test: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(n_vars);
    let mut eq = build_eq_table(&r_test);
    let mut d = d_padded.clone();
    let mut b = b_padded.clone();
    let mut s = slack_fr.clone();
    let mut e = epsilon_fr.clone();
    let mut p = preact_padded.clone();
    let mut relu = relu_fr.clone();
    let mut current_sum = Fr::ZERO;
    let mut rounds: Vec<RoundPoly3<Fr>> = Vec::with_capacity(n_vars);
    let mut r_final: Vec<Fr> = Vec::with_capacity(n_vars);

    for _ in 0..n_vars {
        let half = d.len() / 2;
        let (mut e0, mut e1, mut e2, mut e3) = (Fr::ZERO, Fr::ZERO, Fr::ZERO, Fr::ZERO);
        for i in 0..half {
            let lin = |a0: Fr, a1: Fr| (a0, a1, a1.double() - a0, a1.double() + a1 - a0.double());
            let (q0, q1, q2, q3) = lin(eq[i], eq[half + i]);
            let (d0, d1, d2, d3) = lin(d[i], d[half + i]);
            let (bo0, bo1, bo2, bo3) = lin(b[i], b[half + i]);
            let (s0, s1, s2, s3) = lin(s[i], s[half + i]);
            let (eps0, eps1, eps2, eps3) = lin(e[i], e[half + i]);
            let (p0, p1, p2, p3) = lin(p[i], p[half + i]);
            let (r0, r1, r2, r3) = lin(relu[i], relu[half + i]);
            // summand = eq · (slack·s_d + eps - d·preact - b·c_b_sd + relu·s_d)
            e0 += q0 * (s0 * s_d + eps0 - d0 * p0 - bo0 * c_b_sd + r0 * s_d);
            e1 += q1 * (s1 * s_d + eps1 - d1 * p1 - bo1 * c_b_sd + r1 * s_d);
            e2 += q2 * (s2 * s_d + eps2 - d2 * p2 - bo2 * c_b_sd + r2 * s_d);
            e3 += q3 * (s3 * s_d + eps3 - d3 * p3 - bo3 * c_b_sd + r3 * s_d);
        }
        let poly = RoundPoly3 {
            at_zero: e0,
            at_one: e1,
            at_two: e2,
            at_three: e3,
        };
        if poly.at_zero + poly.at_one != current_sum {
            return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
                which: "relu_upper_endpoint: sumcheck round invariant",
            });
        }
        let r_i = squeeze_round_challenge_3(sponge, &poly);
        rounds.push(poly);
        r_final.push(r_i);
        current_sum = rounds.last().unwrap().evaluate(r_i);
        for i in 0..half {
            let bind = |lo: Fr, hi: Fr| lo + r_i * (hi - lo);
            d[i] = bind(d[i], d[half + i]);
            b[i] = bind(b[i], b[half + i]);
            s[i] = bind(s[i], s[half + i]);
            e[i] = bind(e[i], e[half + i]);
            p[i] = bind(p[i], p[half + i]);
            relu[i] = bind(relu[i], relu[half + i]);
            eq[i] = bind(eq[i], eq[half + i]);
        }
        d.truncate(half);
        b.truncate(half);
        s.truncate(half);
        e.truncate(half);
        p.truncate(half);
        relu.truncate(half);
        eq.truncate(half);
    }

    // Six-way batched open of (d, b, slack, epsilon, preact, relu)
    // at r_final — the slack identity consumes all six evals.
    let r_items = [
        BatchOpenSpec {
            aux: d_aux,
            commitment: d_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: b_aux,
            commitment: b_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &slack_aux,
            commitment: &slack_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &epsilon_aux,
            commitment: &epsilon_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: preact_aux,
            commitment: preact_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &relu_aux,
            commitment: &relu_commit,
            commit_n_vars: n_vars,
        },
    ];
    let (r_vals, r_batched_open) =
        hyrax_open_batched_at(&params.committer_key, &r_items, &r_final, sponge, rng)?;
    let d_eval = r_vals[0];
    let b_eval = r_vals[1];
    let slack_eval = r_vals[2];
    let epsilon_eval = r_vals[3];
    let preact_eval = r_vals[4];
    let relu_eval = r_vals[5];

    let eq_eval = eval_multilinear_full(&build_eq_table(&r_test), &r_final);
    let lhs = eq_eval
        * (slack_eval * s_d + epsilon_eval - d_eval * preact_eval - b_eval * c_b_sd
            + relu_eval * s_d);
    if lhs != current_sum {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "relu_upper_endpoint: final identity",
        });
    }

    Ok(ReluUpperEndpointHalf {
        n_vars,
        relu_commit,
        slack_commit,
        epsilon_commit,
        relu_lookup,
        slack_range,
        epsilon_range,
        r_test,
        rounds,
        r_final,
        r_batched_open,
        d_upper_eval: d_eval,
        b_upper_eval: b_eval,
        slack_eval,
        epsilon_eval,
        preact_eval,
        relu_eval,
    })
}

#[allow(clippy::too_many_arguments)]
fn verify_half(
    half: &ReluUpperEndpointHalf,
    expected_n_vars: usize,
    preact_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s_d: Fr,
    c_b_sd: Fr,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    if half.n_vars != expected_n_vars {
        return Err(SnarkError::ArchitectureMismatch {
            what: "relu_upper_endpoint: half.n_vars mismatch",
        });
    }

    sponge.absorb(&(half.n_vars as u64));
    absorb_commitment(sponge, d_commit);
    absorb_commitment(sponge, b_commit);
    absorb_commitment(sponge, preact_commit);
    absorb_commitment(sponge, &half.relu_commit);

    verify_relu_lookup_1d(
        &half.relu_lookup,
        preact_commit,
        &half.relu_commit,
        expected_n_vars,
        params,
        sponge,
    )?;

    absorb_commitment(sponge, &half.slack_commit);
    absorb_commitment(sponge, &half.epsilon_commit);

    verify_pos_range(
        &half.slack_range,
        expected_n_vars,
        &half.slack_commit,
        params,
        sponge,
    )?;
    verify_pos_range(
        &half.epsilon_range,
        expected_n_vars,
        &half.epsilon_commit,
        params,
        sponge,
    )?;

    let r_test = sponge.squeeze_field_elements::<Fr>(half.n_vars);
    if r_test != half.r_test {
        return Err(SnarkError::TranscriptMismatch);
    }
    if half.rounds.len() != half.n_vars || half.r_final.len() != half.n_vars {
        return Err(SnarkError::ShapeMismatch {
            what: "relu_upper_endpoint: sumcheck rounds length",
        });
    }
    let mut claim = Fr::ZERO;
    for (round_idx, round_poly) in half.rounds.iter().enumerate() {
        if round_poly.at_zero + round_poly.at_one != claim {
            return Err(SnarkError::SumcheckRoundCheckFailed { round: round_idx });
        }
        let r_i = squeeze_round_challenge_3(sponge, round_poly);
        if r_i != half.r_final[round_idx] {
            return Err(SnarkError::TranscriptMismatch);
        }
        claim = round_poly.evaluate(r_i);
    }

    let v_items = [
        BatchVerifySpec {
            commitment: d_commit,
            commit_n_vars: half.n_vars,
            value: half.d_upper_eval,
        },
        BatchVerifySpec {
            commitment: b_commit,
            commit_n_vars: half.n_vars,
            value: half.b_upper_eval,
        },
        BatchVerifySpec {
            commitment: &half.slack_commit,
            commit_n_vars: half.n_vars,
            value: half.slack_eval,
        },
        BatchVerifySpec {
            commitment: &half.epsilon_commit,
            commit_n_vars: half.n_vars,
            value: half.epsilon_eval,
        },
        BatchVerifySpec {
            commitment: preact_commit,
            commit_n_vars: half.n_vars,
            value: half.preact_eval,
        },
        BatchVerifySpec {
            commitment: &half.relu_commit,
            commit_n_vars: half.n_vars,
            value: half.relu_eval,
        },
    ];
    let r_open_ok = hyrax_verify_batched_at(
        &params.verifier_key,
        &v_items,
        &half.r_final,
        &half.r_batched_open,
        sponge,
    )?;
    if !r_open_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "relu_upper_endpoint::batched_open",
        });
    }

    let eq_eval = eval_multilinear_full(&build_eq_table(&half.r_test), &half.r_final);
    let lhs = eq_eval
        * (half.slack_eval * s_d + half.epsilon_eval
            - half.d_upper_eval * half.preact_eval
            - half.b_upper_eval * c_b_sd
            + half.relu_eval * s_d);
    if lhs != claim {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "relu_upper_endpoint::final_identity",
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_relu_upper_endpoint(
    layer_idx: usize,
    n_vars: usize,
    preact_lower_aux: &CommittedAux,
    preact_lower_commit: &<HyraxBn254 as MlPcs>::Commitment,
    preact_upper_aux: &CommittedAux,
    preact_upper_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_aux: &CommittedAux,
    d_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_aux: &CommittedAux,
    b_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s_d_scale: Scale,
    s_w_scale: Scale,
    s_b_scale: Scale,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<ReluUpperEndpointProof, SnarkError> {
    let _timing = crate::timing::scope("relu_gadget");
    let s_d_code = crate::quantization::quantized_scalar::Qf::from_real(1.0, s_d_scale).code;
    let s_w_code = crate::quantization::quantized_scalar::Qf::from_real(1.0, s_w_scale).code;
    let s_b_code = crate::quantization::quantized_scalar::Qf::from_real(1.0, s_b_scale).code;
    if s_d_code <= 0 || s_w_code <= 0 || s_b_code <= 0 {
        return Err(SnarkError::ShapeMismatch {
            what: "relu_upper_endpoint: non-positive scale code",
        });
    }
    if s_w_code % s_b_code != 0 {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "relu_upper_endpoint: precondition q_b > q_w",
        });
    }
    let s_d = signed_lift_to_fr(s_d_code);
    // c_b_sd = (s_w / s_b) · s_d, an integer constant.
    let c_b_sd_code =
        (s_w_code / s_b_code)
            .checked_mul(s_d_code)
            .ok_or(SnarkError::ShapeMismatch {
                what: "relu_upper_endpoint: c_b_sd code overflow",
            })?;
    let c_b_sd = signed_lift_to_fr(c_b_sd_code);

    sponge.absorb(&(layer_idx as u64));

    let lo = prove_half(
        n_vars,
        preact_lower_aux,
        preact_lower_commit,
        d_aux,
        d_commit,
        b_aux,
        b_commit,
        s_d,
        c_b_sd,
        s_d_code,
        s_w_code,
        s_b_code,
        params,
        sponge,
        rng,
    )?;
    let hi = prove_half(
        n_vars,
        preact_upper_aux,
        preact_upper_commit,
        d_aux,
        d_commit,
        b_aux,
        b_commit,
        s_d,
        c_b_sd,
        s_d_code,
        s_w_code,
        s_b_code,
        params,
        sponge,
        rng,
    )?;
    Ok(ReluUpperEndpointProof {
        layer_idx,
        n_vars,
        lo,
        hi,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_relu_upper_endpoint(
    proof: &ReluUpperEndpointProof,
    expected_layer_idx: usize,
    expected_n_vars: usize,
    preact_lower_commit: &<HyraxBn254 as MlPcs>::Commitment,
    preact_upper_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s_d_scale: Scale,
    s_w_scale: Scale,
    s_b_scale: Scale,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    if proof.layer_idx != expected_layer_idx {
        return Err(SnarkError::ArchitectureMismatch {
            what: "relu_upper_endpoint: layer_idx mismatch",
        });
    }
    if proof.n_vars != expected_n_vars {
        return Err(SnarkError::ArchitectureMismatch {
            what: "relu_upper_endpoint: n_vars mismatch",
        });
    }
    let s_d_code = crate::quantization::quantized_scalar::Qf::from_real(1.0, s_d_scale).code;
    let s_w_code = crate::quantization::quantized_scalar::Qf::from_real(1.0, s_w_scale).code;
    let s_b_code = crate::quantization::quantized_scalar::Qf::from_real(1.0, s_b_scale).code;
    if s_w_code % s_b_code != 0 {
        return Err(SnarkError::ArchitectureMismatch {
            what: "relu_upper_endpoint: precondition q_b > q_w",
        });
    }
    let s_d = signed_lift_to_fr(s_d_code);
    let c_b_sd_code =
        (s_w_code / s_b_code)
            .checked_mul(s_d_code)
            .ok_or(SnarkError::ShapeMismatch {
                what: "relu_upper_endpoint: c_b_sd code overflow",
            })?;
    let c_b_sd = signed_lift_to_fr(c_b_sd_code);
    sponge.absorb(&(expected_layer_idx as u64));

    verify_half(
        &proof.lo,
        expected_n_vars,
        preact_lower_commit,
        d_commit,
        b_commit,
        s_d,
        c_b_sd,
        params,
        sponge,
    )?;
    verify_half(
        &proof.hi,
        expected_n_vars,
        preact_upper_commit,
        d_commit,
        b_commit,
        s_d,
        c_b_sd,
        params,
        sponge,
    )?;
    Ok(())
}
