//! Per-event rescale prover. Commits `slack_lo`, runs the LogUp
//! range proof on `slack_lo ⊆ [0, 2c2)`, and opens
//! `(qx, qz, slack_lo)` at a Fiat-Shamir random `r_identity` so the
//! verifier can replay the boxed-inequality identity.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_std::rand::RngCore;

use crate::snark_primitives::finite_field::signed_lift_to_fr;
use crate::snark_primitives::logup_gkr::{prove_circuit as prove_logup_circuit, LogUpCircuit};
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use super::{
    absorb_commitment, build_multiplicities, build_range_table, top_halves, RescaleEventDesc,
    RescaleEventProof,
};
use crate::snark::commitment::commit::CommittedAux;
use crate::snark::commitment::pcs_helpers::{hyrax_open_at, hyrax_open_batched_at, BatchOpenSpec};
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;

/// Prove one rescale event.
///
/// Caller pre-pads `slack_lo_codes`, `qx_evals`, and `qz_evals` to
/// `2^desc.n_vars` and supplies the existing Hyrax `(aux, commit)`
/// pairs for `qx` and `qz`.
#[allow(clippy::too_many_arguments)]
pub fn prove_rescale_event(
    desc: &RescaleEventDesc,
    slack_lo_codes: &[i128],
    qx_evals: &[Fr],
    qz_evals: &[Fr],
    qx_aux: &CommittedAux,
    qx_commitment: &<HyraxBn254 as MlPcs>::Commitment,
    qz_aux: &CommittedAux,
    qz_commitment: &<HyraxBn254 as MlPcs>::Commitment,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<RescaleEventProof, SnarkError> {
    let n_padded = 1usize << desc.n_vars;
    if qx_evals.len() != n_padded || qz_evals.len() != n_padded {
        return Err(SnarkError::ShapeMismatch {
            what: "rescale: qx/qz length != 2^n_vars",
        });
    }
    if slack_lo_codes.len() != n_padded {
        return Err(SnarkError::ShapeMismatch {
            what: "rescale: slack_lo length != 2^n_vars (caller must pre-pad)",
        });
    }
    if desc.c2 <= 0 {
        return Err(SnarkError::ShapeMismatch {
            what: "rescale: c2 must be positive",
        });
    }
    let two_c2: u128 = (desc.c2 as u128)
        .checked_mul(2)
        .ok_or(SnarkError::ShapeMismatch {
            what: "rescale: 2*c2 overflow",
        })?;

    sponge.absorb(&(desc.n_vars as u64));
    sponge.absorb(&signed_lift_to_fr(desc.c1));
    sponge.absorb(&signed_lift_to_fr(desc.c2));
    sponge.absorb(&(desc.dir.tag() as u64));
    absorb_commitment(sponge, qx_commitment);
    absorb_commitment(sponge, qz_commitment);

    // Commit slack_lo at native size; caller padded to 2^n_vars and
    // bumped n_vars even ≥ 2 for Hyrax.
    let slack_padded: Vec<Fr> = slack_lo_codes
        .iter()
        .map(|&v| signed_lift_to_fr(v))
        .collect();
    debug_assert_eq!(slack_padded.len(), 1usize << desc.n_vars);
    debug_assert!(
        desc.n_vars.is_multiple_of(2) && desc.n_vars >= 2,
        "rescale: caller must bump n_vars to even ≥ 2 for Hyrax"
    );
    let (slack_lo_commit, slack_lo_state) =
        HyraxBn254::commit(&params.committer_key, &slack_padded, Some(rng))
            .map_err(SnarkError::Pcs)?;
    let slack_lo_aux: CommittedAux = (slack_padded, slack_lo_state);
    absorb_commitment(sponge, &slack_lo_commit);

    // LogUp range for slack_lo ⊆ [0, 2c2).
    let logup_witness: Vec<Fr> = slack_lo_codes
        .iter()
        .map(|&v| signed_lift_to_fr(v))
        .collect();
    let table = build_range_table(two_c2);
    let mults = build_multiplicities(slack_lo_codes, two_c2);
    let mults_fr: Vec<Fr> = mults.iter().map(|&m| Fr::from(m)).collect();

    // Commit mults BEFORE β is squeezed.
    let mult_n_vars = {
        let nv = (table.len() as f64).log2().round() as usize;
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
        LogUpCircuit::table(&table, &mults_fr, logup_beta).map_err(SnarkError::LogUp)?;
    let lookup_top = top_halves(&lookup_circuit);
    let table_top = top_halves(&table_circuit);
    let lookup_proof = prove_logup_circuit(&lookup_circuit, sponge).map_err(SnarkError::LogUp)?;
    let table_proof = prove_logup_circuit(&table_circuit, sponge).map_err(SnarkError::LogUp)?;

    // Bottom-bind slack_lo to the committed tensor at the LogUp
    // bottom_point (length `n_vars − 1` per LogUp's convention).
    let logup_point = lookup_proof.bottom_point.clone();
    let (slack_lo_logup_eval, slack_lo_logup_open) = hyrax_open_at(
        &params.committer_key,
        &slack_lo_aux,
        &slack_lo_commit,
        &logup_point,
        sponge,
        rng,
    )?;
    debug_assert_eq!(
        lookup_proof.bottom_denom,
        slack_lo_logup_eval - logup_beta,
        "rescale: LogUp bottom denom must equal slack_lo(r) − β at bottom_point"
    );

    let r_identity: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(desc.n_vars);
    let r_identity_items = [
        BatchOpenSpec {
            aux: qx_aux,
            commitment: qx_commitment,
            commit_n_vars: desc.n_vars,
        },
        BatchOpenSpec {
            aux: qz_aux,
            commitment: qz_commitment,
            commit_n_vars: desc.n_vars,
        },
        BatchOpenSpec {
            aux: &slack_lo_aux,
            commitment: &slack_lo_commit,
            commit_n_vars: desc.n_vars,
        },
    ];
    let (r_identity_vals, r_identity_open) = hyrax_open_batched_at(
        &params.committer_key,
        &r_identity_items,
        &r_identity,
        sponge,
        rng,
    )?;
    let qx_eval = r_identity_vals[0];
    let qz_eval = r_identity_vals[1];
    let slack_lo_eval = r_identity_vals[2];

    // Direction-specific identity offset must match the verifier.
    let c1_fr = signed_lift_to_fr(desc.c1);
    let c2_fr = signed_lift_to_fr(desc.c2);
    let two = Fr::from(2u64);
    let offset_fr = match desc.dir {
        crate::quantization::quantized_scalar::RoundDir::HalfAway => c2_fr,
        crate::quantization::quantized_scalar::RoundDir::Floor => Fr::from(0u64),
        crate::quantization::quantized_scalar::RoundDir::Ceil => two * c2_fr - two,
    };
    debug_assert_eq!(
        slack_lo_eval,
        two * c1_fr * qx_eval - two * c2_fr * qz_eval + offset_fr,
        "rescale: per-direction identity must hold at r_identity"
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
        "rescale: mult open eval must equal table_proof.bottom_num"
    );

    Ok(RescaleEventProof {
        c1_fr,
        c2_fr,
        n_vars: desc.n_vars,
        dir_tag: desc.dir.tag(),
        slack_lo_commit,
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
        slack_lo_logup_open,
        slack_lo_logup_eval,
        r_identity,
        r_identity_open,
        qx_eval,
        qz_eval,
        slack_lo_eval,
        mult_commit,
        mult_open,
        mult_n_vars,
    })
}
