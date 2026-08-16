//! Inequality mode for the final-pass output. The claimed bound is a
//! private witness, Hyrax-committed inside the proof and opened only
//! at the FS-derived identity point.
//!
//! The gadget produces two per-cell non-negative slack tensors, each
//! range-checked via LogUp against `[0, 2^range_bits)` at the budget
//! the CALLER selects: the final pass runs at
//! `params.out_bound_range_bits` (the wide window that covers
//! very-robust output margins — the only place the wide table is used)
//! and the hidden-pass preact bounds run at `params.gadget_range_bits`.
//! `slack` ties the claimed bound to `b_acc_final + acc_w`, and
//! `prop_slack` (optional) ties the claimed bound to a public
//! threshold so the verifier learns accept/reject without seeing the
//! bound. `prop_slack` is `Option` so hidden-pass preact bounds can
//! skip the property identity.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;

use crate::quantized_crown::BoundDir;
use crate::snark_primitives::finite_field::signed_lift_to_fr;
use crate::snark_primitives::logup_gkr::{
    prove_circuit as prove_logup_circuit, verify_circuit_with_top, LogUpCircuit, LogUpProof,
};
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use super::{absorb_commitment, build_pos_multiplicities};
use crate::snark::commitment::commit::CommittedAux;
use crate::snark::commitment::pcs_helpers::{
    hyrax_open_at, hyrax_open_batched_at, hyrax_verify_at, hyrax_verify_batched_at, top_halves,
    BatchOpenSpec, BatchVerifySpec,
};
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;

/// Output-bound inequality proof carrying the claimed-bound commit,
/// the slack commit, the per-event LogUp range, and an optional
/// property-check sub-proof.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct OutputBoundIneqProof {
    /// 0 = lower-pass (`slack = computed − claimed`), 1 = upper-pass.
    pub direction: u64,
    pub n_vars: usize,
    /// Hyrax commit to the private claimed-bound codes; opened at the
    /// FS-derived `r` for the slack identity and property check.
    pub claimed_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub claimed_open_at_r: <HyraxBn254 as MlPcs>::Proof,
    pub claimed_eval: Fr,
    pub slack_commit: <HyraxBn254 as MlPcs>::Commitment,
    /// LogUp range proof for `slack ⊆ [0, 2^OUT_BOUND_RANGE_BITS)`.
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
    /// Multiplicity-binding commit (absorbed before β); the verifier
    /// checks the open at `table_proof.bottom_point` matches
    /// `bottom_num`.
    pub mult_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub mult_open: <HyraxBn254 as MlPcs>::Proof,
    pub mult_n_vars: usize,
    /// Bottom-bind: `slack(bottom_point) − β = lookup.bottom_denom`.
    pub slack_logup_open: <HyraxBn254 as MlPcs>::Proof,
    pub slack_logup_eval: Fr,
    pub r: Vec<Fr>,
    /// Batched Hyrax open at `r` for `(b_acc_final, acc_w, slack)`.
    pub r_batched_open: <HyraxBn254 as MlPcs>::Proof,
    pub b_acc_final_eval: Fr,
    pub acc_w_eval: Fr,
    pub slack_eval: Fr,

    /// Optional in-SNARK property check binding the private claimed
    /// bound to a public threshold. `Some` on the final pass, `None`
    /// on hidden-pass preact bounds.
    pub property_check: Option<PropertyCheckProof>,
}

/// Sub-proof for the in-SNARK property check on `prop_slack`.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PropertyCheckProof {
    pub prop_slack_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub prop_logup_alpha: Fr,
    pub prop_logup_beta: Fr,
    pub prop_lookup_proof: LogUpProof<Fr>,
    pub prop_table_proof: LogUpProof<Fr>,
    pub prop_lookup_top: [Fr; 4],
    pub prop_table_top: [Fr; 4],
    pub prop_lookup_n_vars: usize,
    pub prop_table_n_vars: usize,
    pub prop_witness_len: usize,
    pub prop_table_len: usize,
    pub prop_mult_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub prop_mult_open: <HyraxBn254 as MlPcs>::Proof,
    pub prop_mult_n_vars: usize,
    pub prop_slack_logup_open: <HyraxBn254 as MlPcs>::Proof,
    pub prop_slack_logup_eval: Fr,
    pub prop_slack_eval_at_r: Fr,
    pub prop_slack_open_at_r: <HyraxBn254 as MlPcs>::Proof,
}

/// Prove the output-bound inequality for one direction.
///
/// Caller must pre-pad every input to `2^n_vars` and use `n_vars >= 2`
/// (LogUp requires this). `external_claimed`, when `Some`, makes the
/// gadget reuse the caller's existing `(aux, commit)` for the claimed
/// witness; required by the hidden pass so downstream gadgets see the
/// same MLE the inequality binds.
///
/// `range_bits` selects the slack range budget and must be one of the
/// two budgets `params.preprocessed` was built for: the final pass
/// passes `params.out_bound_range_bits`, hidden-pass preact bounds
/// pass `params.gadget_range_bits`. The verifier derives the same
/// value from its call site, so the choice is architecture-determined,
/// never prover-controlled.
#[allow(clippy::too_many_arguments)]
pub fn prove_output_bound_inequality(
    direction: BoundDir,
    range_bits: usize,
    n_vars: usize,
    claimed_codes_padded: &[i128],
    b_acc_final_codes: &[i128],
    acc_w_codes: &[i128],
    // Public threshold codes at the working scale, padded to
    // `1 << n_vars`. `Some` enables the in-SNARK property check;
    // `None` skips it (hidden-pass preact bounds).
    threshold_codes_padded: Option<&[i128]>,
    b_acc_final_aux: &CommittedAux,
    b_acc_final_com: &<HyraxBn254 as MlPcs>::Commitment,
    acc_w_aux: &CommittedAux,
    acc_w_com: &<HyraxBn254 as MlPcs>::Commitment,
    external_claimed: Option<(&CommittedAux, &<HyraxBn254 as MlPcs>::Commitment)>,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<OutputBoundIneqProof, SnarkError> {
    if n_vars < 2 {
        return Err(SnarkError::ShapeMismatch {
            what: "output_bound: caller must bump n_vars >= 2",
        });
    }
    let n_padded = 1usize << n_vars;
    if claimed_codes_padded.len() != n_padded
        || b_acc_final_codes.len() != n_padded
        || acc_w_codes.len() != n_padded
    {
        return Err(SnarkError::ShapeMismatch {
            what: "output_bound ineq: input length != 2^n_vars",
        });
    }
    if let Some(t) = threshold_codes_padded {
        if t.len() != n_padded {
            return Err(SnarkError::ShapeMismatch {
                what: "output_bound ineq: threshold padded length != 2^n_vars",
            });
        }
    }

    // Slack per cell. Upper: claimed - computed; Lower: computed -
    // claimed. A dishonest claim yields negative slack, rejected by
    // the LogUp range check.
    let mut slack: Vec<i128> = Vec::with_capacity(n_padded);
    for i in 0..n_padded {
        let computed =
            b_acc_final_codes[i]
                .checked_add(acc_w_codes[i])
                .ok_or(SnarkError::ShapeMismatch {
                    what: "output_bound ineq: computed sum overflow",
                })?;
        let s = match direction {
            BoundDir::Upper => claimed_codes_padded[i] - computed,
            BoundDir::Lower => computed - claimed_codes_padded[i],
        };
        slack.push(s);
    }

    sponge.absorb(&(direction as u64));
    sponge.absorb(&(n_vars as u64));
    absorb_commitment(sponge, b_acc_final_com);
    absorb_commitment(sponge, acc_w_com);
    // Reuse the caller-supplied claimed commit when present so the
    // hidden pass binds the same MLE downstream gadgets consume.
    let claimed_padded_fr: Vec<Fr> = claimed_codes_padded
        .iter()
        .map(|&c| signed_lift_to_fr(c))
        .collect();
    let (claimed_commit, claimed_aux): (<HyraxBn254 as MlPcs>::Commitment, CommittedAux) =
        if let Some((ext_aux, ext_commit)) = external_claimed {
            if ext_aux.0.len() != n_padded {
                return Err(SnarkError::ShapeMismatch {
                    what: "output_bound: external_claimed aux length != 2^n_vars",
                });
            }
            if ext_aux.0 != claimed_padded_fr {
                return Err(SnarkError::ShapeMismatch {
                    what: "output_bound: external_claimed aux MLE mismatch with claimed_codes",
                });
            }
            (ext_commit.clone(), ext_aux.clone())
        } else {
            let (commit, state) =
                HyraxBn254::commit(&params.committer_key, &claimed_padded_fr, Some(rng))
                    .map_err(SnarkError::Pcs)?;
            let aux: CommittedAux = (claimed_padded_fr.clone(), state);
            (commit, aux)
        };
    absorb_commitment(sponge, &claimed_commit);

    // Slack commits at native size; caller must bump n_vars to
    // even ≥ 2 for Hyrax.
    debug_assert!(
        n_vars.is_multiple_of(2) && n_vars >= 2,
        "output_bound: caller must bump n_vars to even ≥ 2 for Hyrax"
    );
    let slack_padded: Vec<Fr> = slack.iter().map(|&v| signed_lift_to_fr(v)).collect();
    debug_assert_eq!(slack_padded.len(), 1usize << n_vars);
    let (slack_commit, slack_state) =
        HyraxBn254::commit(&params.committer_key, &slack_padded, Some(rng))
            .map_err(SnarkError::Pcs)?;
    let slack_aux: CommittedAux = (slack_padded, slack_state);
    absorb_commitment(sponge, &slack_commit);

    // LogUp range slack ⊆ [0, 2^bits) at the caller-selected budget.
    // Tables for both supported budgets are preprocessed; only
    // multiplicities are built online.
    let logup_witness: Vec<Fr> = slack.iter().map(|&v| signed_lift_to_fr(v)).collect();
    let table: &[Fr] = params.preprocessed.pos_range_table(range_bits)?;
    let mults = build_pos_multiplicities(&slack, range_bits);
    let mults_fr: Vec<Fr> = mults.iter().map(|&m| Fr::from(m)).collect();

    // Commit mults BEFORE β is squeezed so the prover can't adapt
    // them to β. Hyrax requires even n_vars; pad up if needed.
    let mult_n_vars = {
        let nv = range_bits;
        if nv % 2 == 1 { nv + 1 } else { nv }.max(2)
    };
    let mult_padded_len = 1usize << mult_n_vars;
    let mut mults_padded: Vec<Fr> = Vec::with_capacity(mult_padded_len);
    mults_padded.extend_from_slice(&mults_fr);
    mults_padded.resize(mult_padded_len, Fr::from(0u64));
    let lu_timing = crate::timing::lu_scope();
    let (mult_commit, mult_state) =
        HyraxBn254::commit(&params.committer_key, &mults_padded, Some(rng))
            .map_err(SnarkError::Pcs)?;
    let mult_aux: CommittedAux = (mults_padded, mult_state);
    absorb_commitment(sponge, &mult_commit);

    sponge.absorb(&(logup_witness.len() as u64));
    sponge.absorb(&(table.len() as u64));
    let logup_alpha = sponge.squeeze_field_elements::<Fr>(1)[0];
    let logup_beta = sponge.squeeze_field_elements::<Fr>(1)[0];
    sponge.absorb(&logup_alpha);
    sponge.absorb(&logup_beta);

    let lookup_circuit =
        LogUpCircuit::lookup(&logup_witness, logup_beta).map_err(SnarkError::LogUp)?;
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
    debug_assert_eq!(
        mult_eval_check, table_proof.bottom_num,
        "output_bound: mult open eval must equal table_proof.bottom_num"
    );
    drop(lu_timing);

    let logup_point = lookup_proof.bottom_point.clone();
    let (slack_logup_eval, slack_logup_open) = hyrax_open_at(
        &params.committer_key,
        &slack_aux,
        &slack_commit,
        &logup_point,
        sponge,
        rng,
    )?;
    debug_assert_eq!(
        lookup_proof.bottom_denom,
        slack_logup_eval - logup_beta,
        "output_bound: LogUp bottom_denom must equal slack(r) − β"
    );

    // Identity at random r; the three n_spec-shaped commits share
    // the same n_vars and batch into a single Hyrax open.
    let r: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(n_vars);
    let r_items = [
        BatchOpenSpec {
            aux: b_acc_final_aux,
            commitment: b_acc_final_com,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: acc_w_aux,
            commitment: acc_w_com,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &slack_aux,
            commitment: &slack_commit,
            commit_n_vars: n_vars,
        },
    ];
    let (r_vals, r_batched_open) =
        hyrax_open_batched_at(&params.committer_key, &r_items, &r, sponge, rng)?;
    let b_acc_final_eval = r_vals[0];
    let acc_w_eval = r_vals[1];
    let slack_eval = r_vals[2];

    let (claimed_eval, claimed_open_at_r) = hyrax_open_at(
        &params.committer_key,
        &claimed_aux,
        &claimed_commit,
        &r,
        sponge,
        rng,
    )?;
    let expected_slack = match direction {
        BoundDir::Upper => claimed_eval - b_acc_final_eval - acc_w_eval,
        BoundDir::Lower => b_acc_final_eval + acc_w_eval - claimed_eval,
    };
    debug_assert_eq!(
        slack_eval, expected_slack,
        "output_bound ineq: identity must hold at r"
    );

    // In-SNARK property check: binds claimed to a public threshold
    // via a range-checked prop_slack and an identity at the same `r`.
    let property_check = if let Some(threshold_codes_padded) = threshold_codes_padded {
        let mut prop_slack: Vec<i128> = Vec::with_capacity(n_padded);
        for i in 0..n_padded {
            let v = match direction {
                BoundDir::Lower => claimed_codes_padded[i] - threshold_codes_padded[i],
                BoundDir::Upper => threshold_codes_padded[i] - claimed_codes_padded[i],
            };
            prop_slack.push(v);
        }
        let prop_slack_padded: Vec<Fr> = prop_slack.iter().map(|&v| signed_lift_to_fr(v)).collect();
        let (prop_slack_commit, prop_slack_state) =
            HyraxBn254::commit(&params.committer_key, &prop_slack_padded, Some(rng))
                .map_err(SnarkError::Pcs)?;
        let prop_slack_aux: CommittedAux = (prop_slack_padded.clone(), prop_slack_state);
        absorb_commitment(sponge, &prop_slack_commit);

        let prop_logup_witness: Vec<Fr> =
            prop_slack.iter().map(|&v| signed_lift_to_fr(v)).collect();
        let prop_table: &[Fr] = params.preprocessed.pos_range_table(range_bits)?;
        let prop_mults = build_pos_multiplicities(&prop_slack, range_bits);
        let prop_mults_fr: Vec<Fr> = prop_mults.iter().map(|&m| Fr::from(m)).collect();

        let prop_mult_n_vars = {
            let nv = range_bits;
            if nv % 2 == 1 { nv + 1 } else { nv }.max(2)
        };
        let prop_mult_padded_len = 1usize << prop_mult_n_vars;
        let mut prop_mults_padded: Vec<Fr> = Vec::with_capacity(prop_mult_padded_len);
        prop_mults_padded.extend_from_slice(&prop_mults_fr);
        prop_mults_padded.resize(prop_mult_padded_len, Fr::from(0u64));
        let prop_lu_timing = crate::timing::lu_scope();
        let (prop_mult_commit, prop_mult_state) =
            HyraxBn254::commit(&params.committer_key, &prop_mults_padded, Some(rng))
                .map_err(SnarkError::Pcs)?;
        let prop_mult_aux: CommittedAux = (prop_mults_padded, prop_mult_state);
        absorb_commitment(sponge, &prop_mult_commit);

        sponge.absorb(&(prop_logup_witness.len() as u64));
        sponge.absorb(&(prop_table.len() as u64));
        let prop_logup_alpha = sponge.squeeze_field_elements::<Fr>(1)[0];
        let prop_logup_beta = sponge.squeeze_field_elements::<Fr>(1)[0];
        sponge.absorb(&prop_logup_alpha);
        sponge.absorb(&prop_logup_beta);

        let prop_lookup_circuit = LogUpCircuit::lookup(&prop_logup_witness, prop_logup_beta)
            .map_err(SnarkError::LogUp)?;
        let prop_table_circuit = LogUpCircuit::table(prop_table, &prop_mults_fr, prop_logup_beta)
            .map_err(SnarkError::LogUp)?;
        let prop_lookup_top = top_halves(&prop_lookup_circuit);
        let prop_table_top = top_halves(&prop_table_circuit);
        let prop_lookup_proof =
            prove_logup_circuit(&prop_lookup_circuit, sponge).map_err(SnarkError::LogUp)?;
        let prop_table_proof =
            prove_logup_circuit(&prop_table_circuit, sponge).map_err(SnarkError::LogUp)?;

        let prop_mult_bottom_pt = prop_table_proof.bottom_point.clone();
        let (prop_mult_eval_check, prop_mult_open) = hyrax_open_at(
            &params.committer_key,
            &prop_mult_aux,
            &prop_mult_commit,
            &prop_mult_bottom_pt,
            sponge,
            rng,
        )?;
        debug_assert_eq!(prop_mult_eval_check, prop_table_proof.bottom_num);
        drop(prop_lu_timing);

        let prop_slack_bottom_pt = prop_lookup_proof.bottom_point.clone();
        let (prop_slack_logup_eval, prop_slack_logup_open) = hyrax_open_at(
            &params.committer_key,
            &prop_slack_aux,
            &prop_slack_commit,
            &prop_slack_bottom_pt,
            sponge,
            rng,
        )?;

        // prop_slack opens at the same `r` as the slack identity.
        let (prop_slack_eval_at_r, prop_slack_open_at_r) = hyrax_open_at(
            &params.committer_key,
            &prop_slack_aux,
            &prop_slack_commit,
            &r,
            sponge,
            rng,
        )?;
        // threshold(r) is the verifier-recomputable canonical MLE of
        // the public threshold codes at r.
        let threshold_padded_fr: Vec<Fr> = threshold_codes_padded
            .iter()
            .map(|&v| signed_lift_to_fr(v))
            .collect();
        let threshold_eval_at_r =
            crate::snark::commitment::multilinear_extensions::eval_multilinear_full(
                &threshold_padded_fr,
                &r,
            );
        let expected_prop_slack_eval = match direction {
            BoundDir::Lower => claimed_eval - threshold_eval_at_r,
            BoundDir::Upper => threshold_eval_at_r - claimed_eval,
        };
        debug_assert_eq!(
            prop_slack_eval_at_r, expected_prop_slack_eval,
            "output_bound prop: prop_slack(r) must equal claimed(r) ± threshold(r)"
        );
        Some(PropertyCheckProof {
            prop_slack_commit,
            prop_logup_alpha,
            prop_logup_beta,
            prop_lookup_proof,
            prop_table_proof,
            prop_lookup_top,
            prop_table_top,
            prop_lookup_n_vars: prop_logup_witness.len().trailing_zeros() as usize - 1,
            prop_table_n_vars: prop_table.len().trailing_zeros() as usize - 1,
            prop_witness_len: prop_logup_witness.len(),
            prop_table_len: prop_table.len(),
            prop_mult_commit,
            prop_mult_open,
            prop_mult_n_vars,
            prop_slack_logup_open,
            prop_slack_logup_eval,
            prop_slack_eval_at_r,
            prop_slack_open_at_r,
        })
    } else {
        None
    };

    Ok(OutputBoundIneqProof {
        direction: direction as u64,
        n_vars,
        claimed_commit,
        claimed_open_at_r,
        claimed_eval,
        slack_commit,
        logup_alpha,
        logup_beta,
        lookup_proof,
        table_proof,
        lookup_top,
        table_top,
        lookup_n_vars: logup_witness.len().trailing_zeros() as usize - 1,
        table_n_vars: table.len().trailing_zeros() as usize - 1,
        witness_len: logup_witness.len(),
        table_len: table.len(),
        mult_commit,
        mult_open,
        mult_n_vars,
        slack_logup_open,
        slack_logup_eval,
        r,
        r_batched_open,
        b_acc_final_eval,
        acc_w_eval,
        slack_eval,
        property_check,
    })
}

/// Verify an output-bound inequality proof. `threshold_codes_padded`
/// must be `Some` for the final pass and `None` for hidden-pass
/// preact bounds; presence must match `proof.property_check`.
///
/// `range_bits` must mirror the prover's call site: the final pass
/// passes `params.out_bound_range_bits`, hidden-pass preact bounds
/// pass `params.gadget_range_bits`.
#[allow(clippy::too_many_arguments)]
pub fn verify_output_bound_inequality(
    proof: &OutputBoundIneqProof,
    direction: BoundDir,
    range_bits: usize,
    expected_n_vars: usize,
    threshold_codes_padded: Option<&[Fr]>,
    b_acc_final_com: &<HyraxBn254 as MlPcs>::Commitment,
    acc_w_com: &<HyraxBn254 as MlPcs>::Commitment,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    let n_padded_v = 1usize << expected_n_vars;
    if let Some(t) = threshold_codes_padded {
        if t.len() != n_padded_v {
            return Err(SnarkError::ShapeMismatch {
                what: "output_bound prop: threshold padded length != 2^n_vars",
            });
        }
    }
    if proof.direction != direction as u64 {
        return Err(SnarkError::ShapeMismatch {
            what: "output_bound ineq: direction tag mismatch",
        });
    }
    if proof.n_vars != expected_n_vars {
        return Err(SnarkError::ArchitectureMismatch {
            what: "output_bound ineq: proof.n_vars != native_vector_n_vars(n_spec)",
        });
    }
    // Pin the range-table width to this call site's budget: a prover
    // must not range-check against a wider window than the statement's
    // budget for this pass. The budget itself must be one of the two
    // the preprocessed tables were built for.
    if params.preprocessed.pos_range_table(range_bits).is_err() {
        return Err(SnarkError::InvalidParameter {
            what: "output_bound ineq: range_bits matches neither preprocessed budget",
        });
    }
    if proof.table_len != 1usize << range_bits {
        return Err(SnarkError::ShapeMismatch {
            what: "output_bound ineq: range table length != 2^range_bits",
        });
    }

    sponge.absorb(&(direction as u64));
    sponge.absorb(&(expected_n_vars as u64));
    absorb_commitment(sponge, b_acc_final_com);
    absorb_commitment(sponge, acc_w_com);
    absorb_commitment(sponge, &proof.claimed_commit);
    absorb_commitment(sponge, &proof.slack_commit);
    // mult_commit must be absorbed before β is squeezed.
    absorb_commitment(sponge, &proof.mult_commit);

    sponge.absorb(&(proof.witness_len as u64));
    sponge.absorb(&(proof.table_len as u64));
    let alpha = sponge.squeeze_field_elements::<Fr>(1)[0];
    let beta = sponge.squeeze_field_elements::<Fr>(1)[0];
    if alpha != proof.logup_alpha || beta != proof.logup_beta {
        return Err(SnarkError::TranscriptMismatch);
    }
    sponge.absorb(&proof.logup_alpha);
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

    // Bind table-side bottom_denom to the canonical positive-range MLE.
    let canonical_t_mle =
        crate::snark::commitment::table_mle::pos_range_mle_eval(&proof.table_proof.bottom_point);
    let expected_table_bottom_denom = canonical_t_mle - beta;
    if proof.table_proof.bottom_denom != expected_table_bottom_denom {
        return Err(SnarkError::LogUpTableNotCanonical {
            which: "output_bound_pos_range",
        });
    }

    // Bind table-side bottom_num to the multiplicity commit.
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
            which: "output_bound_mult",
        });
    }

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
        return Err(SnarkError::OutputBoundRangeFailed);
    }

    let logup_point = proof.lookup_proof.bottom_point.clone();
    let cnv = expected_n_vars;
    let ok = hyrax_verify_at(
        &params.verifier_key,
        &proof.slack_commit,
        &logup_point,
        proof.slack_logup_eval,
        &proof.slack_logup_open,
        cnv,
        sponge,
    )?;
    if !ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "output_bound_slack_logup",
        });
    }
    if proof.lookup_proof.bottom_denom != proof.slack_logup_eval - proof.logup_beta {
        return Err(SnarkError::OutputBoundRangeFailed);
    }

    let r: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(expected_n_vars);
    if r != proof.r {
        return Err(SnarkError::TranscriptMismatch);
    }

    let r_items = [
        BatchVerifySpec {
            commitment: b_acc_final_com,
            value: proof.b_acc_final_eval,
            commit_n_vars: cnv,
        },
        BatchVerifySpec {
            commitment: acc_w_com,
            value: proof.acc_w_eval,
            commit_n_vars: cnv,
        },
        BatchVerifySpec {
            commitment: &proof.slack_commit,
            value: proof.slack_eval,
            commit_n_vars: cnv,
        },
    ];
    let r_ok = hyrax_verify_batched_at(
        &params.verifier_key,
        &r_items,
        &r,
        &proof.r_batched_open,
        sponge,
    )?;
    if !r_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "output_bound_r_batch",
        });
    }

    // Open the (private) claimed commit at `r` to recover claimed_eval.
    let claimed_open_ok = hyrax_verify_at(
        &params.verifier_key,
        &proof.claimed_commit,
        &r,
        proof.claimed_eval,
        &proof.claimed_open_at_r,
        cnv,
        sponge,
    )?;
    if !claimed_open_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "output_bound_claimed_at_r",
        });
    }
    let claimed_eval = proof.claimed_eval;
    let expected_slack = match direction {
        BoundDir::Upper => claimed_eval - proof.b_acc_final_eval - proof.acc_w_eval,
        BoundDir::Lower => proof.b_acc_final_eval + proof.acc_w_eval - claimed_eval,
    };
    if proof.slack_eval != expected_slack {
        return Err(SnarkError::OutputBoundIdentityFailed);
    }

    // In-SNARK property check; presence on both sides must agree.
    match (threshold_codes_padded, &proof.property_check) {
        (None, None) => {}
        (Some(_), None) | (None, Some(_)) => {
            return Err(SnarkError::ShapeMismatch {
                what: "output_bound prop: threshold/property_check presence mismatch",
            });
        }
        (Some(threshold_codes_padded), Some(pc)) => {
            if pc.prop_table_len != 1usize << range_bits {
                return Err(SnarkError::ShapeMismatch {
                    what: "output_bound prop: range table length != 2^range_bits",
                });
            }
            absorb_commitment(sponge, &pc.prop_slack_commit);
            absorb_commitment(sponge, &pc.prop_mult_commit);
            sponge.absorb(&(pc.prop_witness_len as u64));
            sponge.absorb(&(pc.prop_table_len as u64));
            let prop_alpha = sponge.squeeze_field_elements::<Fr>(1)[0];
            let prop_beta = sponge.squeeze_field_elements::<Fr>(1)[0];
            if prop_alpha != pc.prop_logup_alpha || prop_beta != pc.prop_logup_beta {
                return Err(SnarkError::TranscriptMismatch);
            }
            sponge.absorb(&pc.prop_logup_alpha);
            sponge.absorb(&pc.prop_logup_beta);

            verify_circuit_with_top(
                &pc.prop_lookup_proof,
                pc.prop_lookup_n_vars,
                pc.prop_lookup_top,
                pc.prop_lookup_top[0] * pc.prop_lookup_top[3]
                    + pc.prop_lookup_top[1] * pc.prop_lookup_top[2],
                sponge,
            )
            .map_err(SnarkError::LogUp)?;
            verify_circuit_with_top(
                &pc.prop_table_proof,
                pc.prop_table_n_vars,
                pc.prop_table_top,
                pc.prop_table_top[0] * pc.prop_table_top[3]
                    + pc.prop_table_top[1] * pc.prop_table_top[2],
                sponge,
            )
            .map_err(SnarkError::LogUp)?;

            let prop_canonical_t_mle = crate::snark::commitment::table_mle::pos_range_mle_eval(
                &pc.prop_table_proof.bottom_point,
            );
            let prop_expected_table_bottom_denom = prop_canonical_t_mle - prop_beta;
            if pc.prop_table_proof.bottom_denom != prop_expected_table_bottom_denom {
                return Err(SnarkError::LogUpTableNotCanonical {
                    which: "output_bound_prop_pos_range",
                });
            }

            let prop_mult_ok = hyrax_verify_at(
                &params.verifier_key,
                &pc.prop_mult_commit,
                &pc.prop_table_proof.bottom_point,
                pc.prop_table_proof.bottom_num,
                &pc.prop_mult_open,
                pc.prop_mult_n_vars,
                sponge,
            )?;
            if !prop_mult_ok {
                return Err(SnarkError::PcsOpenRejected {
                    which: "output_bound_prop_mult",
                });
            }

            let prop_lookup_frac = (
                pc.prop_lookup_top[0] * pc.prop_lookup_top[3]
                    + pc.prop_lookup_top[1] * pc.prop_lookup_top[2],
                pc.prop_lookup_top[2] * pc.prop_lookup_top[3],
            );
            let prop_table_frac = (
                pc.prop_table_top[0] * pc.prop_table_top[3]
                    + pc.prop_table_top[1] * pc.prop_table_top[2],
                pc.prop_table_top[2] * pc.prop_table_top[3],
            );
            let prop_combined =
                prop_lookup_frac.0 * prop_table_frac.1 + prop_lookup_frac.1 * prop_table_frac.0;
            if prop_combined != Fr::from(0u64) {
                return Err(SnarkError::OutputBoundRangeFailed);
            }

            let prop_logup_point = pc.prop_lookup_proof.bottom_point.clone();
            let prop_slack_logup_ok = hyrax_verify_at(
                &params.verifier_key,
                &pc.prop_slack_commit,
                &prop_logup_point,
                pc.prop_slack_logup_eval,
                &pc.prop_slack_logup_open,
                cnv,
                sponge,
            )?;
            if !prop_slack_logup_ok {
                return Err(SnarkError::PcsOpenRejected {
                    which: "output_bound_prop_slack_logup",
                });
            }
            if pc.prop_lookup_proof.bottom_denom != pc.prop_slack_logup_eval - pc.prop_logup_beta {
                return Err(SnarkError::OutputBoundRangeFailed);
            }

            let prop_at_r_ok = hyrax_verify_at(
                &params.verifier_key,
                &pc.prop_slack_commit,
                &r,
                pc.prop_slack_eval_at_r,
                &pc.prop_slack_open_at_r,
                cnv,
                sponge,
            )?;
            if !prop_at_r_ok {
                return Err(SnarkError::PcsOpenRejected {
                    which: "output_bound_prop_slack_at_r",
                });
            }
            let threshold_eval_at_r =
                crate::snark::commitment::multilinear_extensions::eval_multilinear_full(
                    threshold_codes_padded,
                    &r,
                );
            let expected_prop_slack = match direction {
                BoundDir::Lower => claimed_eval - threshold_eval_at_r,
                BoundDir::Upper => threshold_eval_at_r - claimed_eval,
            };
            if pc.prop_slack_eval_at_r != expected_prop_slack {
                return Err(SnarkError::OutputBoundIdentityFailed);
            }
        }
    }

    Ok(())
}
