//! Per-event rescale verifier. Replays the transcript, runs the
//! LogUp range check (with canonical-MLE + multiplicity binding),
//! and checks the boxed-inequality identity at `r_identity` after
//! Hyrax-opening `(qx, qz, slack_lo)` at that point.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;

use crate::snark_primitives::finite_field::signed_lift_to_fr;
use crate::snark_primitives::logup_gkr::verify_circuit_with_top;
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use super::{absorb_commitment, RescaleEventDesc, RescaleEventProof};
use crate::snark::commitment::pcs_helpers::{
    hyrax_verify_at, hyrax_verify_batched_at, BatchVerifySpec,
};
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;

/// Verify one rescale-event proof.
pub fn verify_rescale_event(
    proof: &RescaleEventProof,
    desc: &RescaleEventDesc,
    qx_commitment: &<HyraxBn254 as MlPcs>::Commitment,
    qz_commitment: &<HyraxBn254 as MlPcs>::Commitment,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    if desc.n_vars != proof.n_vars {
        return Err(SnarkError::ShapeMismatch {
            what: "rescale: n_vars mismatch",
        });
    }
    let c1_fr = signed_lift_to_fr(desc.c1);
    let c2_fr = signed_lift_to_fr(desc.c2);
    if c1_fr != proof.c1_fr || c2_fr != proof.c2_fr {
        return Err(SnarkError::RescaleScaleMismatch);
    }

    sponge.absorb(&(desc.n_vars as u64));
    sponge.absorb(&c1_fr);
    sponge.absorb(&c2_fr);
    if proof.dir_tag != desc.dir.tag() {
        return Err(SnarkError::RescaleScaleMismatch);
    }
    sponge.absorb(&(proof.dir_tag as u64));
    absorb_commitment(sponge, qx_commitment);
    absorb_commitment(sponge, qz_commitment);
    absorb_commitment(sponge, &proof.slack_lo_commit);
    // mult_commit absorbed before β.
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

    // Bind table-side bottom_denom to canonical T_mle. Only the case
    // where `2c2` is itself a power of two is supported (default pow2
    // scale family); otherwise the canonical MLE no longer collapses
    // to the closed-form `Σ 2^j r_j`.
    let two_c2 = (desc.c2 as u128)
        .checked_mul(2)
        .ok_or(SnarkError::ShapeMismatch {
            what: "rescale: 2c2 overflow during verifier table-binding",
        })?;
    let table_len_u128 = proof.table_len as u128;
    if two_c2 != table_len_u128 || !table_len_u128.is_power_of_two() {
        return Err(SnarkError::LogUpTableNotCanonical {
            which: "rescale_per_event_range (non-pow2 2c2 unsupported by closed-form binding)",
        });
    }
    let canonical_t_mle =
        crate::snark::commitment::table_mle::pos_range_mle_eval(&proof.table_proof.bottom_point);
    let expected_table_bottom_denom = canonical_t_mle - beta;
    if proof.table_proof.bottom_denom != expected_table_bottom_denom {
        return Err(SnarkError::LogUpTableNotCanonical {
            which: "rescale_per_event_range",
        });
    }

    // Top-fraction cancellation: lookup ⊆ table iff sum is 0.
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
        return Err(SnarkError::RescaleRangeIdentityFailed);
    }

    // Bottom-bind slack_lo to lookup.bottom_denom.
    let logup_point = proof.lookup_proof.bottom_point.clone();
    let cnv = proof.n_vars;
    let ok = hyrax_verify_at(
        &params.verifier_key,
        &proof.slack_lo_commit,
        &logup_point,
        proof.slack_lo_logup_eval,
        &proof.slack_lo_logup_open,
        cnv,
        sponge,
    )?;
    if !ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "rescale_slack_lo_logup",
        });
    }
    if proof.lookup_proof.bottom_denom != proof.slack_lo_logup_eval - proof.logup_beta {
        return Err(SnarkError::RescaleRangeBindingFailed);
    }

    let r_identity: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(proof.n_vars);
    if r_identity != proof.r_identity {
        return Err(SnarkError::TranscriptMismatch);
    }

    let r_identity_items = [
        BatchVerifySpec {
            commitment: qx_commitment,
            value: proof.qx_eval,
            commit_n_vars: cnv,
        },
        BatchVerifySpec {
            commitment: qz_commitment,
            value: proof.qz_eval,
            commit_n_vars: cnv,
        },
        BatchVerifySpec {
            commitment: &proof.slack_lo_commit,
            value: proof.slack_lo_eval,
            commit_n_vars: cnv,
        },
    ];
    let r_identity_ok = hyrax_verify_batched_at(
        &params.verifier_key,
        &r_identity_items,
        &r_identity,
        &proof.r_identity_open,
        sponge,
    )?;
    if !r_identity_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "rescale_r_identity_batch",
        });
    }

    // slack_lo == 2·c1·qx − 2·c2·qz + offset(dir).
    let two = Fr::from(2u64);
    let two_c1_qx = two * c1_fr * proof.qx_eval;
    let two_c2_qz = two * c2_fr * proof.qz_eval;
    let offset_fr = match crate::quantization::quantized_scalar::RoundDir::from_tag(proof.dir_tag) {
        Some(crate::quantization::quantized_scalar::RoundDir::HalfAway) => c2_fr,
        Some(crate::quantization::quantized_scalar::RoundDir::Floor) => Fr::from(0u64),
        Some(crate::quantization::quantized_scalar::RoundDir::Ceil) => two * c2_fr - two,
        None => return Err(SnarkError::RescaleScaleMismatch),
    };
    let expected = two_c1_qx - two_c2_qz + offset_fr;
    if proof.slack_lo_eval != expected {
        return Err(SnarkError::RescaleIdentityFailed);
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
            which: "rescale_mult",
        });
    }

    Ok(())
}
