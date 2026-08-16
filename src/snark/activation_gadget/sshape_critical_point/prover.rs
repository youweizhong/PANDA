//! Prover for the critical-point gadget. Builds per-neuron witnesses,
//! runs the σ-envelope LogUp at three sample points, and folds all
//! per-cell identities into one degree-4 sumcheck.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::AdditiveGroup;
use ark_std::rand::RngCore;

use crate::crown::network::ActivationKind;
use crate::quantization::scale::Scale;
use crate::snark_primitives::finite_field::signed_lift_to_fr;
use crate::snark_primitives::logup_gkr::{prove_circuit as prove_logup_circuit, LogUpCircuit};
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};
use crate::snark_primitives::sumcheck::RoundPoly4;

use super::super::relu_upper_endpoint::{prove_pos_range, squeeze_round_challenge_4};
use super::super::sshape_endpoint::SshapeLineKind;
use crate::snark::commitment::commit::CommittedAux;
use crate::snark::commitment::multilinear_extensions::{build_eq_table, eval_multilinear_full};
use crate::snark::commitment::pcs_helpers::{hyrax_open_at, hyrax_open_batched_at, BatchOpenSpec};
use crate::snark::errors::SnarkError;
use crate::snark::output_bound::absorb_commitment;
use crate::snark::params::SnarkParams;

use super::types::SshapeCriticalPointProof;
use super::witness::{compute_witnesses, fd_scale_precondition_holds, lift};

/// Prove critical-point validity for one (sigmoid/tanh layer, line
/// direction). `preact_lower_codes` / `preact_upper_codes` are the
/// `[l, u]` interval at scale `s_w`; the gadget binds the relaxation
/// `(d_line, b_line)` commits at the chosen line direction. The
/// hidden-pass preact aux commits are opened at this gadget's
/// `r_final` so the verifier never reads raw preact codes.
#[allow(clippy::too_many_arguments)]
pub fn prove_sshape_critical_point(
    layer_idx: usize,
    kind: ActivationKind,
    line: SshapeLineKind,
    preact_lower_codes: &[i128],
    preact_upper_codes: &[i128],
    preact_lower_aux: &CommittedAux,
    preact_lower_commit: &<HyraxBn254 as MlPcs>::Commitment,
    preact_upper_aux: &CommittedAux,
    preact_upper_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_line_aux: &CommittedAux,
    d_line_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_line_aux: &CommittedAux,
    b_line_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s_d: Scale,
    s_b: Scale,
    s_w: Scale,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<SshapeCriticalPointProof, SnarkError> {
    let _timing = crate::timing::scope("sshape_critical");
    let kind_tag = match kind {
        ActivationKind::Sigmoid => 0u8,
        ActivationKind::Tanh => 1u8,
        ActivationKind::ReLU => {
            return Err(SnarkError::ShapeMismatch {
                what: "sshape3c called for ReLU layer",
            });
        }
    };
    let line_tag = line.tag();

    if !fd_scale_precondition_holds(s_d, s_w, params.gadget_range_bits) {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape3c: FD scale precondition s_d·s_x·s_v ≤ 2^GADGET_RANGE_BITS",
        });
    }
    // Each remainder in the factor_b chain is bounded by one of
    // {s_d, s_b, s_w}, so each scale must fit in one chunk.
    let s_d_e = s_d.pow2_exponent().map_err(|_| SnarkError::ShapeMismatch {
        what: "sshape3c: s_d not pow2",
    })?;
    let s_b_e = s_b.pow2_exponent().map_err(|_| SnarkError::ShapeMismatch {
        what: "sshape3c: s_b not pow2",
    })?;
    let s_w_e = s_w.pow2_exponent().map_err(|_| SnarkError::ShapeMismatch {
        what: "sshape3c: s_w not pow2",
    })?;
    if s_d_e < 0 || s_b_e < 0 || s_w_e < 0 {
        return Err(SnarkError::ShapeMismatch {
            what: "sshape3c: scales must have non-neg pow2 exponent",
        });
    }
    let bits = params.gadget_range_bits as i32;
    if s_d_e > bits || s_b_e > bits || s_w_e > bits {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape3c: per-scale precondition (each of s_d, s_b, s_w must be ≤ 2^GADGET_RANGE_BITS)",
        });
    }
    let n = preact_lower_codes.len();
    if n == 0 {
        return Err(SnarkError::Reserved {
            what: "sshape3c: requires n ≥ 1",
        });
    }
    if n != preact_upper_codes.len() {
        return Err(SnarkError::ShapeMismatch {
            what: "sshape3c: l/u length mismatch",
        });
    }
    // Padding rows are masked by the public is_real MLE in the
    // combined sumcheck.
    let n_vars = crate::snark::commitment::commit::native_vector_n_vars(n);
    let n_padded = 1usize << n_vars;
    if d_line_aux.0.len() != n_padded || b_line_aux.0.len() != n_padded {
        return Err(SnarkError::ShapeMismatch {
            what: "sshape3c: d/b commit n_vars mismatch",
        });
    }
    // Require s_x == s_w so z and u share scale in the factor_a
    // identity. Lifting this needs an abs/shift witness.
    if s_w_e != params.sigma_x_scale_log2 {
        return Err(SnarkError::Reserved {
            what: "sshape3c: requires s_w = s_x (= 2^sigma_x_scale_log2) — abs/shift extension TBD",
        });
    }

    let s_d_code = 1i128 << s_d_e;
    let s_b_code = 1i128 << s_b_e;
    let s_w_code = 1i128 << s_w_e;
    let s_x_code = 1i128 << params.sigma_x_scale_log2;
    let s_v_code = 1i128 << params.sigma_v_scale_log2;

    sponge.absorb(&(layer_idx as u64));
    sponge.absorb(&kind_tag);
    sponge.absorb(&line_tag);
    sponge.absorb(&(n_vars as u64));
    sponge.absorb(&(n as u64));
    absorb_commitment(sponge, d_line_commit);
    absorb_commitment(sponge, b_line_commit);
    // Bind preact codes via their hidden-pass commits. The padded
    // vectors below are local to the prover; the verifier consumes
    // these values only through `r_final` opens.
    absorb_commitment(sponge, preact_lower_commit);
    absorb_commitment(sponge, preact_upper_commit);
    let preact_l_padded: Vec<Fr> = preact_lower_codes
        .iter()
        .map(|&v| signed_lift_to_fr(v))
        .chain(std::iter::repeat_n(Fr::from(0u64), n_padded - n))
        .collect();
    let preact_u_padded: Vec<Fr> = preact_upper_codes
        .iter()
        .map(|&v| signed_lift_to_fr(v))
        .chain(std::iter::repeat_n(Fr::from(0u64), n_padded - n))
        .collect();
    if preact_lower_aux.0.len() != n_padded || preact_upper_aux.0.len() != n_padded {
        return Err(SnarkError::ShapeMismatch {
            what: "sshape3c: preact aux length mismatch with n_padded",
        });
    }

    let d_padded: Vec<i128> = d_line_aux
        .0
        .iter()
        .map(|f| crate::snark_primitives::finite_field::fr_to_signed_i128(*f).unwrap_or(0))
        .collect();
    let b_padded: Vec<i128> = b_line_aux
        .0
        .iter()
        .map(|f| crate::snark_primitives::finite_field::fr_to_signed_i128(*f).unwrap_or(0))
        .collect();
    let nws = compute_witnesses(
        &params.preprocessed.sigma,
        kind,
        line,
        layer_idx,
        preact_lower_codes,
        preact_upper_codes,
        &d_padded,
        &b_padded,
        s_d_code,
        s_b_code,
        s_w_code,
        s_x_code,
        s_v_code,
        n,
        n_padded,
        params.gadget_range_bits,
    )?;
    let z_codes: Vec<i128> = nws.iter().map(|w| w.z).collect();
    let sigma_lo_z_codes: Vec<i128> = nws.iter().map(|w| w.sigma_lo_z).collect();
    let sigma_up_z_codes: Vec<i128> = nws.iter().map(|w| w.sigma_up_z).collect();
    let sigma_lo_zmd_codes: Vec<i128> = nws.iter().map(|w| w.sigma_lo_zmd).collect();
    let sigma_up_zmd_codes: Vec<i128> = nws.iter().map(|w| w.sigma_up_zmd).collect();
    let sigma_lo_zpd_codes: Vec<i128> = nws.iter().map(|w| w.sigma_lo_zpd).collect();
    let sigma_up_zpd_codes: Vec<i128> = nws.iter().map(|w| w.sigma_up_zpd).collect();
    let slack_fd1_codes: Vec<i128> = nws.iter().map(|w| w.slack_fd1).collect();
    let slack_fd2_codes: Vec<i128> = nws.iter().map(|w| w.slack_fd2).collect();
    let slack_fd1_high_codes: Vec<i128> = nws.iter().map(|w| w.slack_fd1_high).collect();
    let slack_fd1_low_codes: Vec<i128> = nws.iter().map(|w| w.slack_fd1_low).collect();
    let slack_fd2_high_codes: Vec<i128> = nws.iter().map(|w| w.slack_fd2_high).collect();
    let slack_fd2_low_codes: Vec<i128> = nws.iter().map(|w| w.slack_fd2_low).collect();
    let factor_a_codes: Vec<i128> = nws.iter().map(|w| w.factor_a).collect();
    let factor_b_codes: Vec<i128> = nws.iter().map(|w| w.factor_b).collect();
    let dz_step_1_codes: Vec<i128> = nws.iter().map(|w| w.dz_step_1).collect();
    let dz_step_1_rem_codes: Vec<i128> = nws.iter().map(|w| w.dz_step_1_rem).collect();
    let dz_sigma_code_codes: Vec<i128> = nws.iter().map(|w| w.dz_sigma_code).collect();
    let dz_sigma_rem_codes: Vec<i128> = nws.iter().map(|w| w.dz_sigma_rem).collect();
    let b_sigma_code_codes: Vec<i128> = nws.iter().map(|w| w.b_sigma_code).collect();
    let b_sigma_rem_codes: Vec<i128> = nws.iter().map(|w| w.b_sigma_rem).collect();
    let is_active_codes: Vec<i128> = nws.iter().map(|w| w.is_active).collect();
    let inside_codes: Vec<i128> = nws.iter().map(|w| w.inside_bit).collect();
    let slack_pos_codes: Vec<i128> = nws.iter().map(|w| w.slack_pos).collect();
    let slack_pos_high_codes: Vec<i128> = nws.iter().map(|w| w.slack_pos_high).collect();
    let slack_pos_low_codes: Vec<i128> = nws.iter().map(|w| w.slack_pos_low).collect();
    let gated_gap_codes: Vec<i128> = nws.iter().map(|w| w.gated_gap).collect();
    let gated_gap_high_codes: Vec<i128> = nws.iter().map(|w| w.gated_gap_high).collect();
    let gated_gap_low_codes: Vec<i128> = nws.iter().map(|w| w.gated_gap_low).collect();
    let bound = 1i128 << params.gadget_range_bits;
    let check_pos = |label: &'static str, vs: &[i128]| -> Result<(), SnarkError> {
        for &v in vs {
            if v < 0 || v >= bound {
                return Err(SnarkError::RelaxationSoundnessFinalCheckFailed { which: label });
            }
        }
        Ok(())
    };
    check_pos("sshape3c: dz_step_1_rem out of range", &dz_step_1_rem_codes)?;
    check_pos("sshape3c: dz_sigma_rem out of range", &dz_sigma_rem_codes)?;
    check_pos("sshape3c: b_sigma_rem out of range", &b_sigma_rem_codes)?;
    check_pos(
        "sshape3c: slack_pos_high out of range",
        &slack_pos_high_codes,
    )?;
    check_pos("sshape3c: slack_pos_low out of range", &slack_pos_low_codes)?;
    check_pos(
        "sshape3c: gated_gap_high out of range",
        &gated_gap_high_codes,
    )?;
    check_pos("sshape3c: gated_gap_low out of range", &gated_gap_low_codes)?;
    check_pos(
        "sshape3c: slack_fd1_high out of range",
        &slack_fd1_high_codes,
    )?;
    check_pos("sshape3c: slack_fd1_low out of range", &slack_fd1_low_codes)?;
    check_pos(
        "sshape3c: slack_fd2_high out of range",
        &slack_fd2_high_codes,
    )?;
    check_pos("sshape3c: slack_fd2_low out of range", &slack_fd2_low_codes)?;
    for &v in &inside_codes {
        if v != 0 && v != 1 {
            return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
                which: "sshape3c: inside_bit not in {0, 1}",
            });
        }
    }
    for &v in &is_active_codes {
        if v != 0 && v != 1 {
            return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
                which: "sshape3c: is_active not in {0, 1}",
            });
        }
    }

    let z_fr = lift(&z_codes);
    let sigma_lo_z_fr = lift(&sigma_lo_z_codes);
    let sigma_up_z_fr = lift(&sigma_up_z_codes);
    let sigma_lo_zmd_fr = lift(&sigma_lo_zmd_codes);
    let sigma_up_zmd_fr = lift(&sigma_up_zmd_codes);
    let sigma_lo_zpd_fr = lift(&sigma_lo_zpd_codes);
    let sigma_up_zpd_fr = lift(&sigma_up_zpd_codes);
    let slack_fd1_fr = lift(&slack_fd1_codes);
    let slack_fd2_fr = lift(&slack_fd2_codes);
    let slack_fd1_high_fr = lift(&slack_fd1_high_codes);
    let slack_fd1_low_fr = lift(&slack_fd1_low_codes);
    let slack_fd2_high_fr = lift(&slack_fd2_high_codes);
    let slack_fd2_low_fr = lift(&slack_fd2_low_codes);
    let factor_a_fr = lift(&factor_a_codes);
    let factor_b_fr = lift(&factor_b_codes);
    let dz_step_1_fr = lift(&dz_step_1_codes);
    let dz_step_1_rem_fr = lift(&dz_step_1_rem_codes);
    let dz_sigma_code_fr = lift(&dz_sigma_code_codes);
    let dz_sigma_rem_fr = lift(&dz_sigma_rem_codes);
    let b_sigma_code_fr = lift(&b_sigma_code_codes);
    let b_sigma_rem_fr = lift(&b_sigma_rem_codes);
    let is_active_fr = lift(&is_active_codes);
    let inside_fr = lift(&inside_codes);
    let slack_pos_fr = lift(&slack_pos_codes);
    let slack_pos_high_fr = lift(&slack_pos_high_codes);
    let slack_pos_low_fr = lift(&slack_pos_low_codes);
    let gated_gap_fr = lift(&gated_gap_codes);
    let gated_gap_high_fr = lift(&gated_gap_high_codes);
    let gated_gap_low_fr = lift(&gated_gap_low_codes);

    let commit_one = |v: &[Fr],
                      rng: &mut dyn RngCore|
     -> Result<(<HyraxBn254 as MlPcs>::Commitment, CommittedAux), SnarkError> {
        let (commit, state) =
            HyraxBn254::commit(&params.committer_key, v, Some(rng)).map_err(SnarkError::Pcs)?;
        Ok((commit, (v.to_vec(), state)))
    };
    let (z_commit, z_aux) = commit_one(&z_fr, rng)?;
    absorb_commitment(sponge, &z_commit);
    let (sigma_lo_z_commit, sigma_lo_z_aux) = commit_one(&sigma_lo_z_fr, rng)?;
    absorb_commitment(sponge, &sigma_lo_z_commit);
    let (sigma_up_z_commit, sigma_up_z_aux) = commit_one(&sigma_up_z_fr, rng)?;
    absorb_commitment(sponge, &sigma_up_z_commit);
    let (sigma_lo_zmd_commit, sigma_lo_zmd_aux) = commit_one(&sigma_lo_zmd_fr, rng)?;
    absorb_commitment(sponge, &sigma_lo_zmd_commit);
    let (sigma_up_zmd_commit, sigma_up_zmd_aux) = commit_one(&sigma_up_zmd_fr, rng)?;
    absorb_commitment(sponge, &sigma_up_zmd_commit);
    let (sigma_lo_zpd_commit, sigma_lo_zpd_aux) = commit_one(&sigma_lo_zpd_fr, rng)?;
    absorb_commitment(sponge, &sigma_lo_zpd_commit);
    let (sigma_up_zpd_commit, sigma_up_zpd_aux) = commit_one(&sigma_up_zpd_fr, rng)?;
    absorb_commitment(sponge, &sigma_up_zpd_commit);
    let (slack_fd1_commit, slack_fd1_aux) = commit_one(&slack_fd1_fr, rng)?;
    absorb_commitment(sponge, &slack_fd1_commit);
    let (slack_fd2_commit, slack_fd2_aux) = commit_one(&slack_fd2_fr, rng)?;
    absorb_commitment(sponge, &slack_fd2_commit);
    let (slack_fd1_high_commit, slack_fd1_high_aux) = commit_one(&slack_fd1_high_fr, rng)?;
    absorb_commitment(sponge, &slack_fd1_high_commit);
    let (slack_fd1_low_commit, slack_fd1_low_aux) = commit_one(&slack_fd1_low_fr, rng)?;
    absorb_commitment(sponge, &slack_fd1_low_commit);
    let (slack_fd2_high_commit, slack_fd2_high_aux) = commit_one(&slack_fd2_high_fr, rng)?;
    absorb_commitment(sponge, &slack_fd2_high_commit);
    let (slack_fd2_low_commit, slack_fd2_low_aux) = commit_one(&slack_fd2_low_fr, rng)?;
    absorb_commitment(sponge, &slack_fd2_low_commit);
    let (factor_a_commit, factor_a_aux) = commit_one(&factor_a_fr, rng)?;
    absorb_commitment(sponge, &factor_a_commit);
    let (factor_b_commit, factor_b_aux) = commit_one(&factor_b_fr, rng)?;
    absorb_commitment(sponge, &factor_b_commit);
    let (dz_step_1_commit, dz_step_1_aux) = commit_one(&dz_step_1_fr, rng)?;
    absorb_commitment(sponge, &dz_step_1_commit);
    let (dz_step_1_rem_commit, dz_step_1_rem_aux) = commit_one(&dz_step_1_rem_fr, rng)?;
    absorb_commitment(sponge, &dz_step_1_rem_commit);
    let (dz_sigma_code_commit, dz_sigma_code_aux) = commit_one(&dz_sigma_code_fr, rng)?;
    absorb_commitment(sponge, &dz_sigma_code_commit);
    let (dz_sigma_rem_commit, dz_sigma_rem_aux) = commit_one(&dz_sigma_rem_fr, rng)?;
    absorb_commitment(sponge, &dz_sigma_rem_commit);
    let (b_sigma_code_commit, b_sigma_code_aux) = commit_one(&b_sigma_code_fr, rng)?;
    absorb_commitment(sponge, &b_sigma_code_commit);
    let (b_sigma_rem_commit, b_sigma_rem_aux) = commit_one(&b_sigma_rem_fr, rng)?;
    absorb_commitment(sponge, &b_sigma_rem_commit);
    let (is_active_commit, is_active_aux) = commit_one(&is_active_fr, rng)?;
    absorb_commitment(sponge, &is_active_commit);
    let (inside_commit, inside_aux) = commit_one(&inside_fr, rng)?;
    absorb_commitment(sponge, &inside_commit);
    let (slack_pos_commit, slack_pos_aux) = commit_one(&slack_pos_fr, rng)?;
    absorb_commitment(sponge, &slack_pos_commit);
    let (slack_pos_high_commit, slack_pos_high_aux) = commit_one(&slack_pos_high_fr, rng)?;
    absorb_commitment(sponge, &slack_pos_high_commit);
    let (slack_pos_low_commit, slack_pos_low_aux) = commit_one(&slack_pos_low_fr, rng)?;
    absorb_commitment(sponge, &slack_pos_low_commit);
    let (gated_gap_commit, gated_gap_aux) = commit_one(&gated_gap_fr, rng)?;
    absorb_commitment(sponge, &gated_gap_commit);
    let (gated_gap_high_commit, gated_gap_high_aux) = commit_one(&gated_gap_high_fr, rng)?;
    absorb_commitment(sponge, &gated_gap_high_commit);
    let (gated_gap_low_commit, gated_gap_low_aux) = commit_one(&gated_gap_low_fr, rng)?;
    absorb_commitment(sponge, &gated_gap_low_commit);

    let z_range = prove_pos_range(&z_fr, &z_codes, &z_aux, &z_commit, params, sponge, rng)?;
    let slack_fd1_high_range = prove_pos_range(
        &slack_fd1_high_fr,
        &slack_fd1_high_codes,
        &slack_fd1_high_aux,
        &slack_fd1_high_commit,
        params,
        sponge,
        rng,
    )?;
    let slack_fd1_low_range = prove_pos_range(
        &slack_fd1_low_fr,
        &slack_fd1_low_codes,
        &slack_fd1_low_aux,
        &slack_fd1_low_commit,
        params,
        sponge,
        rng,
    )?;
    let slack_fd2_high_range = prove_pos_range(
        &slack_fd2_high_fr,
        &slack_fd2_high_codes,
        &slack_fd2_high_aux,
        &slack_fd2_high_commit,
        params,
        sponge,
        rng,
    )?;
    let slack_fd2_low_range = prove_pos_range(
        &slack_fd2_low_fr,
        &slack_fd2_low_codes,
        &slack_fd2_low_aux,
        &slack_fd2_low_commit,
        params,
        sponge,
        rng,
    )?;
    let slack_pos_high_range = prove_pos_range(
        &slack_pos_high_fr,
        &slack_pos_high_codes,
        &slack_pos_high_aux,
        &slack_pos_high_commit,
        params,
        sponge,
        rng,
    )?;
    let slack_pos_low_range = prove_pos_range(
        &slack_pos_low_fr,
        &slack_pos_low_codes,
        &slack_pos_low_aux,
        &slack_pos_low_commit,
        params,
        sponge,
        rng,
    )?;
    let gated_gap_high_range = prove_pos_range(
        &gated_gap_high_fr,
        &gated_gap_high_codes,
        &gated_gap_high_aux,
        &gated_gap_high_commit,
        params,
        sponge,
        rng,
    )?;
    let gated_gap_low_range = prove_pos_range(
        &gated_gap_low_fr,
        &gated_gap_low_codes,
        &gated_gap_low_aux,
        &gated_gap_low_commit,
        params,
        sponge,
        rng,
    )?;
    let dz_step_1_rem_range = prove_pos_range(
        &dz_step_1_rem_fr,
        &dz_step_1_rem_codes,
        &dz_step_1_rem_aux,
        &dz_step_1_rem_commit,
        params,
        sponge,
        rng,
    )?;
    let dz_sigma_rem_range = prove_pos_range(
        &dz_sigma_rem_fr,
        &dz_sigma_rem_codes,
        &dz_sigma_rem_aux,
        &dz_sigma_rem_commit,
        params,
        sponge,
        rng,
    )?;
    let b_sigma_rem_range = prove_pos_range(
        &b_sigma_rem_fr,
        &b_sigma_rem_codes,
        &b_sigma_rem_aux,
        &b_sigma_rem_commit,
        params,
        sponge,
        rng,
    )?;

    // 3-column σ-envelope LogUp at rows (x, σ_lo, σ_up) for
    // x ∈ {z, z−δ, z+δ} plus one padding row per cell.
    sponge.absorb(&((3 * n_padded) as u64));
    let envelope_combine_alpha_1 = sponge.squeeze_field_elements::<Fr>(1)[0];
    let envelope_combine_alpha_2 = sponge.squeeze_field_elements::<Fr>(1)[0];

    let table_upper = match kind {
        ActivationKind::Sigmoid => &params.preprocessed.sigma.sigmoid_upper_fr,
        ActivationKind::Tanh => &params.preprocessed.sigma.tanh_upper_fr,
        ActivationKind::ReLU => unreachable!(),
    };
    let table_lower = match kind {
        ActivationKind::Sigmoid => &params.preprocessed.sigma.sigmoid_lower_fr,
        ActivationKind::Tanh => &params.preprocessed.sigma.tanh_lower_fr,
        ActivationKind::ReLU => unreachable!(),
    };
    let table_len = table_upper.len();
    // Lay out 4 rows per cell so the LogUp bottom_point decomposes
    // cleanly into (cell index, row-high, row-low). Rows 0/1/2 hold
    // (z, z−δ, z+δ); row 3 is table[0] padding (a valid lookup that
    // the witness-bind reconstruction skips).
    let s_lo_pad_int =
        crate::snark_primitives::finite_field::fr_to_signed_i128(table_lower[0]).unwrap_or(0);
    let s_up_pad_int =
        crate::snark_primitives::finite_field::fr_to_signed_i128(table_upper[0]).unwrap_or(0);
    let s_lo_pad_fr = signed_lift_to_fr(s_lo_pad_int);
    let s_up_pad_fr = signed_lift_to_fr(s_up_pad_int);
    let mut envelope_witness: Vec<Fr> = Vec::with_capacity(4 * n_padded);
    let mut envelope_index_codes: Vec<i128> = Vec::with_capacity(4 * n_padded);
    for j in 0..n_padded {
        let x = z_codes[j];
        let s_lo = sigma_lo_z_codes[j];
        let s_up = sigma_up_z_codes[j];
        envelope_index_codes.push(x);
        envelope_witness.push(
            envelope_combine_alpha_1 * signed_lift_to_fr(x)
                + envelope_combine_alpha_2 * signed_lift_to_fr(s_up)
                + signed_lift_to_fr(s_lo),
        );
        let x = z_codes[j] - 1;
        let s_lo = sigma_lo_zmd_codes[j];
        let s_up = sigma_up_zmd_codes[j];
        envelope_index_codes.push(x);
        envelope_witness.push(
            envelope_combine_alpha_1 * signed_lift_to_fr(x)
                + envelope_combine_alpha_2 * signed_lift_to_fr(s_up)
                + signed_lift_to_fr(s_lo),
        );
        let x = z_codes[j] + 1;
        let s_lo = sigma_lo_zpd_codes[j];
        let s_up = sigma_up_zpd_codes[j];
        envelope_index_codes.push(x);
        envelope_witness.push(
            envelope_combine_alpha_1 * signed_lift_to_fr(x)
                + envelope_combine_alpha_2 * signed_lift_to_fr(s_up)
                + signed_lift_to_fr(s_lo),
        );
        envelope_index_codes.push(0);
        envelope_witness.push(
            envelope_combine_alpha_1 * Fr::from(0u64)
                + envelope_combine_alpha_2 * s_up_pad_fr
                + s_lo_pad_fr,
        );
    }
    let env_pow2 = envelope_witness.len();
    debug_assert!(env_pow2.is_power_of_two() && env_pow2 == 4 * n_padded);
    let mut envelope_mults_u64 = vec![0u64; table_len];
    for &x in envelope_index_codes.iter() {
        if x >= 0 && (x as usize) < table_len {
            envelope_mults_u64[x as usize] += 1;
        }
    }
    let envelope_mults_fr: Vec<Fr> = envelope_mults_u64.iter().map(|&m| Fr::from(m)).collect();
    let envelope_mult_n_vars = {
        let nv = (table_len as f64).log2().round() as usize;
        let nv = if nv % 2 == 1 { nv + 1 } else { nv };
        nv.max(2)
    };
    let envelope_mult_padded_len = 1usize << envelope_mult_n_vars;
    let mut envelope_mults_padded: Vec<Fr> = Vec::with_capacity(envelope_mult_padded_len);
    envelope_mults_padded.extend_from_slice(&envelope_mults_fr);
    envelope_mults_padded.resize(envelope_mult_padded_len, Fr::from(0u64));
    let lu_timing = crate::timing::lu_scope();
    let (envelope_mult_commit, envelope_mult_state) =
        HyraxBn254::commit(&params.committer_key, &envelope_mults_padded, Some(rng))
            .map_err(SnarkError::Pcs)?;
    let envelope_mult_aux: CommittedAux = (envelope_mults_padded, envelope_mult_state);
    absorb_commitment(sponge, &envelope_mult_commit);

    sponge.absorb(&(envelope_witness.len() as u64));
    sponge.absorb(&(table_len as u64));
    let envelope_logup_beta = sponge.squeeze_field_elements::<Fr>(1)[0];
    sponge.absorb(&envelope_combine_alpha_1);
    sponge.absorb(&envelope_combine_alpha_2);
    sponge.absorb(&envelope_logup_beta);

    let envelope_table: Vec<Fr> = (0..table_len)
        .map(|i| {
            envelope_combine_alpha_1 * Fr::from(i as u64)
                + envelope_combine_alpha_2 * table_upper[i]
                + table_lower[i]
        })
        .collect();
    let envelope_lookup_circuit =
        LogUpCircuit::lookup(&envelope_witness, envelope_logup_beta).map_err(SnarkError::LogUp)?;
    let envelope_table_circuit =
        LogUpCircuit::table(&envelope_table, &envelope_mults_fr, envelope_logup_beta)
            .map_err(SnarkError::LogUp)?;
    let envelope_lookup_top = top_halves_logup(&envelope_lookup_circuit);
    let envelope_table_top = top_halves_logup(&envelope_table_circuit);
    let envelope_lookup_proof =
        prove_logup_circuit(&envelope_lookup_circuit, sponge).map_err(SnarkError::LogUp)?;
    let envelope_table_proof =
        prove_logup_circuit(&envelope_table_circuit, sponge).map_err(SnarkError::LogUp)?;

    // Single-point bind: decompose the LogUp bottom_point as
    // (cell index, row-high, row-low) and batch-open the 7 per-cell
    // σ commits at the cell index. The verifier reconstructs W(bp)
    // from those 7 evals plus the 4-row eq selector and checks
    // W(bp) == bottom_denom + β. Order matters: mult open first,
    // then σ batched open, so verifier sponge state stays aligned.
    let mult_bottom_pt = envelope_table_proof.bottom_point.clone();
    let (envelope_mult_eval_check, envelope_mult_open) = hyrax_open_at(
        &params.committer_key,
        &envelope_mult_aux,
        &envelope_mult_commit,
        &mult_bottom_pt,
        sponge,
        rng,
    )?;
    if envelope_mult_eval_check != envelope_table_proof.bottom_num {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape3c: envelope mult open mismatch",
        });
    }
    drop(lu_timing);

    let bp_full = envelope_lookup_proof.bottom_point.clone();
    if bp_full.len() < n_vars + 2 {
        return Err(SnarkError::ShapeMismatch {
            what: "sshape3c: LogUp bottom_point too short for row decomposition",
        });
    }
    let bp_high = &bp_full[..n_vars];
    let envelope_witness_open_items = [
        BatchOpenSpec {
            aux: &z_aux,
            commitment: &z_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &sigma_lo_z_aux,
            commitment: &sigma_lo_z_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &sigma_up_z_aux,
            commitment: &sigma_up_z_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &sigma_lo_zmd_aux,
            commitment: &sigma_lo_zmd_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &sigma_up_zmd_aux,
            commitment: &sigma_up_zmd_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &sigma_lo_zpd_aux,
            commitment: &sigma_lo_zpd_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &sigma_up_zpd_aux,
            commitment: &sigma_up_zpd_commit,
            commit_n_vars: n_vars,
        },
    ];
    let (envelope_evals_at_bp_high, envelope_witness_batched_open) = hyrax_open_batched_at(
        &params.committer_key,
        &envelope_witness_open_items,
        bp_high,
        sponge,
        rng,
    )?;
    let envelope_z_eval = envelope_evals_at_bp_high[0];
    let envelope_sigma_lo_z_eval = envelope_evals_at_bp_high[1];
    let envelope_sigma_up_z_eval = envelope_evals_at_bp_high[2];
    let envelope_sigma_lo_zmd_eval = envelope_evals_at_bp_high[3];
    let envelope_sigma_up_zmd_eval = envelope_evals_at_bp_high[4];
    let envelope_sigma_lo_zpd_eval = envelope_evals_at_bp_high[5];
    let envelope_sigma_up_zpd_eval = envelope_evals_at_bp_high[6];
    {
        let bp_row_high = bp_full[n_vars];
        let bp_row_low = bp_full[n_vars + 1];
        let one = Fr::from(1u64);
        let e_row0 = (one - bp_row_high) * (one - bp_row_low);
        let e_row1 = (one - bp_row_high) * bp_row_low;
        let e_row2 = bp_row_high * (one - bp_row_low);
        let e_row3 = bp_row_high * bp_row_low;
        let s_up_pad =
            crate::snark_primitives::finite_field::fr_to_signed_i128(table_upper[0]).unwrap_or(0);
        let s_lo_pad =
            crate::snark_primitives::finite_field::fr_to_signed_i128(table_lower[0]).unwrap_or(0);
        let s_up_pad_fr = signed_lift_to_fr(s_up_pad);
        let s_lo_pad_fr = signed_lift_to_fr(s_lo_pad);
        let w_row0 = envelope_combine_alpha_1 * envelope_z_eval
            + envelope_combine_alpha_2 * envelope_sigma_up_z_eval
            + envelope_sigma_lo_z_eval;
        let w_row1 = envelope_combine_alpha_1 * (envelope_z_eval - one)
            + envelope_combine_alpha_2 * envelope_sigma_up_zmd_eval
            + envelope_sigma_lo_zmd_eval;
        let w_row2 = envelope_combine_alpha_1 * (envelope_z_eval + one)
            + envelope_combine_alpha_2 * envelope_sigma_up_zpd_eval
            + envelope_sigma_lo_zpd_eval;
        let w_row3 = envelope_combine_alpha_2 * s_up_pad_fr + s_lo_pad_fr;
        let w_at_bp = e_row0 * w_row0 + e_row1 * w_row1 + e_row2 * w_row2 + e_row3 * w_row3;
        debug_assert_eq!(
            envelope_lookup_proof.bottom_denom + envelope_logup_beta,
            w_at_bp,
            "sshape3c: prover-side LogUp single-point bind self-check mismatch"
        );
    }

    // Combined sumcheck: FD1 carries an implicit coefficient 1; the
    // remaining 15 identities are folded with ρ_a..ρ_o.
    let combined_rho_a = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_b = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_c = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_d = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_e = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_f = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_g = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_h = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_i = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_j = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_k = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_l = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_m = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_n = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_o = sponge.squeeze_field_elements::<Fr>(1)[0];

    let c_fd_sigma_fr = signed_lift_to_fr(s_d_code * s_x_code);
    let c_fd_d_fr = signed_lift_to_fr(s_v_code);
    let epsilon_crit_fr = signed_lift_to_fr(
        super::super::sshape_helpers::fd_epsilon_crit(s_d_code, s_x_code).ok_or(
            SnarkError::ShapeMismatch {
                what: "sshape3c: ε_crit overflow",
            },
        )?,
    );
    let s_d_fr = signed_lift_to_fr(s_d_code);
    let s_b_fr = signed_lift_to_fr(s_b_code);
    let s_w_fr = signed_lift_to_fr(s_w_code);
    let s_v_fr = signed_lift_to_fr(s_v_code);
    let chunk_modulus_fr = signed_lift_to_fr(1i128 << params.gadget_range_bits);
    let line_sign_fr: Fr = match line {
        SshapeLineKind::Upper => Fr::from(1u64),
        SshapeLineKind::Lower => -Fr::from(1u64),
    };

    let r_test: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(n_vars);
    let mut eq = build_eq_table(&r_test);
    let is_real_table: Vec<Fr> = (0..n_padded)
        .map(|j| {
            if j < n {
                Fr::from(1u64)
            } else {
                Fr::from(0u64)
            }
        })
        .collect();
    let mut ir_v = is_real_table.clone();

    let mut d_v = d_padded
        .iter()
        .map(|&v| signed_lift_to_fr(v))
        .collect::<Vec<_>>();
    let mut b_v = b_padded
        .iter()
        .map(|&v| signed_lift_to_fr(v))
        .collect::<Vec<_>>();
    let mut l_v = preact_l_padded.clone();
    let mut u_v = preact_u_padded.clone();
    let mut z_v = z_fr.clone();
    let mut s_lo_z_v = sigma_lo_z_fr.clone();
    let mut s_up_z_v = sigma_up_z_fr.clone();
    let mut s_lo_zmd_v = sigma_lo_zmd_fr.clone();
    let mut s_up_zmd_v = sigma_up_zmd_fr.clone();
    let mut s_lo_zpd_v = sigma_lo_zpd_fr.clone();
    let mut s_up_zpd_v = sigma_up_zpd_fr.clone();
    let mut sk1_v = slack_fd1_fr.clone();
    let mut sk2_v = slack_fd2_fr.clone();
    let mut fa_v = factor_a_fr.clone();
    let mut fb_v = factor_b_fr.clone();
    let mut q_dz_v = dz_step_1_fr.clone();
    let mut r_dz_v = dz_step_1_rem_fr.clone();
    let mut q_dzs_v = dz_sigma_code_fr.clone();
    let mut r_dzs_v = dz_sigma_rem_fr.clone();
    let mut q_bs_v = b_sigma_code_fr.clone();
    let mut r_bs_v = b_sigma_rem_fr.clone();
    let mut s1h_v = slack_fd1_high_fr.clone();
    let mut s1l_v = slack_fd1_low_fr.clone();
    let mut s2h_v = slack_fd2_high_fr.clone();
    let mut s2l_v = slack_fd2_low_fr.clone();
    let mut ia_v = is_active_fr.clone();
    let mut ins_v = inside_fr.clone();
    let mut sp_v = slack_pos_fr.clone();
    let mut sp_h_v = slack_pos_high_fr.clone();
    let mut sp_l_v = slack_pos_low_fr.clone();
    let mut gg_v = gated_gap_fr.clone();
    let mut gg_h_v = gated_gap_high_fr.clone();
    let mut gg_l_v = gated_gap_low_fr.clone();

    let one_fr = Fr::from(1u64);
    // Cells packed in this order:
    //   [eq, d, bo, l, uu, z, s_lo_z, s_up_z, s_lo_zmd, s_up_zmd,
    //    s_lo_zpd, s_up_zpd, sk1, sk2, fa, fb,
    //    q_dz, r_dz, q_dzs, r_dzs, q_bs, r_bs,
    //    s1h, s1l, s2h, s2l,
    //    ins, sp, sp_h, sp_l, gg, gg_h, gg_l, ia]
    let inner_eval = |ir: Fr, c: &[Fr; 34]| -> Fr {
        let q = c[0];
        let d = c[1];
        let bo = c[2];
        let l = c[3];
        let uu = c[4];
        let z = c[5];
        let s_lo_z = c[6];
        let s_up_z = c[7];
        let _s_lo_zmd = c[8];
        let s_up_zmd = c[9];
        let _s_lo_zpd = c[10];
        let s_up_zpd = c[11];
        let sk1 = c[12];
        let sk2 = c[13];
        let fa = c[14];
        let fb = c[15];
        let q_dz = c[16];
        let r_dz = c[17];
        let q_dzs = c[18];
        let r_dzs = c[19];
        let q_bs = c[20];
        let r_bs = c[21];
        let s1h = c[22];
        let s1l = c[23];
        let s2h = c[24];
        let s2l = c[25];
        let ins = c[26];
        let sp = c[27];
        let sp_h = c[28];
        let sp_l = c[29];
        let gg = c[30];
        let gg_h = c[31];
        let gg_l = c[32];
        let ia = c[33];
        // FD slack identities, gated by is_active.
        let fd1_raw = sk1 - (s_lo_z - s_up_zmd) * c_fd_sigma_fr + d * c_fd_d_fr - epsilon_crit_fr;
        let fd2_raw = sk2 - d * c_fd_d_fr + (s_up_zpd - s_lo_z) * c_fd_sigma_fr - epsilon_crit_fr;
        let fd1_id = ia * fd1_raw;
        let fd2_id = ia * fd2_raw;
        let factor_a_id = match line {
            SshapeLineKind::Upper => fa - (uu - z),
            SshapeLineKind::Lower => fa - (-z - l),
        };
        let id_dz = d * z - q_dz * s_d_fr - r_dz;
        let id_dzs = q_dz * s_v_fr - q_dzs * s_w_fr - r_dzs;
        let id_bs = line_sign_fr * (bo * s_v_fr - q_bs * s_b_fr) - r_bs;
        let factor_b_id = match line {
            SshapeLineKind::Upper => fb - q_dzs - q_bs + s_up_z,
            SshapeLineKind::Lower => {
                let sigma_lo_neg_z = if matches!(kind, ActivationKind::Sigmoid) {
                    s_v_fr - s_up_z
                } else {
                    -s_up_z
                };
                fb - sigma_lo_neg_z - q_dzs + q_bs
            }
        };
        let id_slack1_chunk = sk1 - s1h * chunk_modulus_fr - s1l;
        let id_slack2_chunk = sk2 - s2h * chunk_modulus_fr - s2l;
        let id_boolean = ins * (one_fr - ins);
        let id_slack_pos = sp - ins.double() * fa + fa + one_fr - ins;
        let id_slack_pos_chunk = sp - sp_h * chunk_modulus_fr - sp_l;
        let id_gated_gap = gg - ins * fb;
        let id_gated_gap_chunk = gg - gg_h * chunk_modulus_fr - gg_l;
        let id_active_boolean = ia * (one_fr - ia);
        let id_active_zero = (one_fr - ia) * d;
        q * ir
            * (fd1_id
                + combined_rho_a * fd2_id
                + combined_rho_b * factor_a_id
                + combined_rho_c * factor_b_id
                + combined_rho_d * id_dz
                + combined_rho_e * id_dzs
                + combined_rho_f * id_bs
                + combined_rho_g * id_slack1_chunk
                + combined_rho_h * id_slack2_chunk
                + combined_rho_i * id_boolean
                + combined_rho_j * id_slack_pos
                + combined_rho_k * id_slack_pos_chunk
                + combined_rho_l * id_gated_gap
                + combined_rho_m * id_gated_gap_chunk
                + combined_rho_n * id_active_boolean
                + combined_rho_o * id_active_zero)
    };

    let mut current_sum = Fr::ZERO;
    let mut rounds: Vec<RoundPoly4<Fr>> = Vec::with_capacity(n_vars);
    let mut r_final: Vec<Fr> = Vec::with_capacity(n_vars);

    let _ = one_fr;

    for _ in 0..n_vars {
        let half = eq.len() / 2;
        let lin = |a0: Fr, a1: Fr| {
            let d = a1 - a0;
            [a0, a1, a1 + d, a1 + d.double(), a1 + d + d.double()]
        };
        let (mut e0, mut e1, mut e2, mut e3, mut e4) =
            (Fr::ZERO, Fr::ZERO, Fr::ZERO, Fr::ZERO, Fr::ZERO);
        for i in 0..half {
            let q = lin(eq[i], eq[half + i]);
            let ir = lin(ir_v[i], ir_v[half + i]);
            let d_ = lin(d_v[i], d_v[half + i]);
            let b_ = lin(b_v[i], b_v[half + i]);
            let l_ = lin(l_v[i], l_v[half + i]);
            let u_ = lin(u_v[i], u_v[half + i]);
            let z_ = lin(z_v[i], z_v[half + i]);
            let sloz = lin(s_lo_z_v[i], s_lo_z_v[half + i]);
            let supz = lin(s_up_z_v[i], s_up_z_v[half + i]);
            let slozmd = lin(s_lo_zmd_v[i], s_lo_zmd_v[half + i]);
            let supzmd = lin(s_up_zmd_v[i], s_up_zmd_v[half + i]);
            let slozpd = lin(s_lo_zpd_v[i], s_lo_zpd_v[half + i]);
            let supzpd = lin(s_up_zpd_v[i], s_up_zpd_v[half + i]);
            let sk1 = lin(sk1_v[i], sk1_v[half + i]);
            let sk2 = lin(sk2_v[i], sk2_v[half + i]);
            let fa = lin(fa_v[i], fa_v[half + i]);
            let fb = lin(fb_v[i], fb_v[half + i]);
            let q_dz = lin(q_dz_v[i], q_dz_v[half + i]);
            let r_dz = lin(r_dz_v[i], r_dz_v[half + i]);
            let q_dzs = lin(q_dzs_v[i], q_dzs_v[half + i]);
            let r_dzs = lin(r_dzs_v[i], r_dzs_v[half + i]);
            let q_bs = lin(q_bs_v[i], q_bs_v[half + i]);
            let r_bs = lin(r_bs_v[i], r_bs_v[half + i]);
            let s1h = lin(s1h_v[i], s1h_v[half + i]);
            let s1l = lin(s1l_v[i], s1l_v[half + i]);
            let s2h = lin(s2h_v[i], s2h_v[half + i]);
            let s2l = lin(s2l_v[i], s2l_v[half + i]);
            let ins = lin(ins_v[i], ins_v[half + i]);
            let sp = lin(sp_v[i], sp_v[half + i]);
            let sp_h = lin(sp_h_v[i], sp_h_v[half + i]);
            let sp_l = lin(sp_l_v[i], sp_l_v[half + i]);
            let gg = lin(gg_v[i], gg_v[half + i]);
            let gg_h = lin(gg_h_v[i], gg_h_v[half + i]);
            let gg_l = lin(gg_l_v[i], gg_l_v[half + i]);
            let ia = lin(ia_v[i], ia_v[half + i]);
            for k in 0..5usize {
                let cell: [Fr; 34] = [
                    q[k], d_[k], b_[k], l_[k], u_[k], z_[k], sloz[k], supz[k], slozmd[k],
                    supzmd[k], slozpd[k], supzpd[k], sk1[k], sk2[k], fa[k], fb[k], q_dz[k],
                    r_dz[k], q_dzs[k], r_dzs[k], q_bs[k], r_bs[k], s1h[k], s1l[k], s2h[k], s2l[k],
                    ins[k], sp[k], sp_h[k], sp_l[k], gg[k], gg_h[k], gg_l[k], ia[k],
                ];
                let v = inner_eval(ir[k], &cell);
                match k {
                    0 => e0 += v,
                    1 => e1 += v,
                    2 => e2 += v,
                    3 => e3 += v,
                    4 => e4 += v,
                    _ => unreachable!(),
                }
            }
        }
        let poly = RoundPoly4 {
            at_zero: e0,
            at_one: e1,
            at_two: e2,
            at_three: e3,
            at_four: e4,
        };
        if poly.at_zero + poly.at_one != current_sum {
            return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
                which: "sshape3c: sumcheck round invariant",
            });
        }
        let r_i = squeeze_round_challenge_4(sponge, &poly);
        rounds.push(poly);
        r_final.push(r_i);
        current_sum = rounds.last().unwrap().evaluate(r_i);
        for i in 0..half {
            let bind = |lo: Fr, hi: Fr| lo + r_i * (hi - lo);
            ir_v[i] = bind(ir_v[i], ir_v[half + i]);
            d_v[i] = bind(d_v[i], d_v[half + i]);
            b_v[i] = bind(b_v[i], b_v[half + i]);
            l_v[i] = bind(l_v[i], l_v[half + i]);
            u_v[i] = bind(u_v[i], u_v[half + i]);
            z_v[i] = bind(z_v[i], z_v[half + i]);
            s_lo_z_v[i] = bind(s_lo_z_v[i], s_lo_z_v[half + i]);
            s_up_z_v[i] = bind(s_up_z_v[i], s_up_z_v[half + i]);
            s_lo_zmd_v[i] = bind(s_lo_zmd_v[i], s_lo_zmd_v[half + i]);
            s_up_zmd_v[i] = bind(s_up_zmd_v[i], s_up_zmd_v[half + i]);
            s_lo_zpd_v[i] = bind(s_lo_zpd_v[i], s_lo_zpd_v[half + i]);
            s_up_zpd_v[i] = bind(s_up_zpd_v[i], s_up_zpd_v[half + i]);
            sk1_v[i] = bind(sk1_v[i], sk1_v[half + i]);
            sk2_v[i] = bind(sk2_v[i], sk2_v[half + i]);
            fa_v[i] = bind(fa_v[i], fa_v[half + i]);
            fb_v[i] = bind(fb_v[i], fb_v[half + i]);
            q_dz_v[i] = bind(q_dz_v[i], q_dz_v[half + i]);
            r_dz_v[i] = bind(r_dz_v[i], r_dz_v[half + i]);
            q_dzs_v[i] = bind(q_dzs_v[i], q_dzs_v[half + i]);
            r_dzs_v[i] = bind(r_dzs_v[i], r_dzs_v[half + i]);
            q_bs_v[i] = bind(q_bs_v[i], q_bs_v[half + i]);
            r_bs_v[i] = bind(r_bs_v[i], r_bs_v[half + i]);
            s1h_v[i] = bind(s1h_v[i], s1h_v[half + i]);
            s1l_v[i] = bind(s1l_v[i], s1l_v[half + i]);
            s2h_v[i] = bind(s2h_v[i], s2h_v[half + i]);
            s2l_v[i] = bind(s2l_v[i], s2l_v[half + i]);
            ins_v[i] = bind(ins_v[i], ins_v[half + i]);
            sp_v[i] = bind(sp_v[i], sp_v[half + i]);
            sp_h_v[i] = bind(sp_h_v[i], sp_h_v[half + i]);
            sp_l_v[i] = bind(sp_l_v[i], sp_l_v[half + i]);
            gg_v[i] = bind(gg_v[i], gg_v[half + i]);
            gg_h_v[i] = bind(gg_h_v[i], gg_h_v[half + i]);
            gg_l_v[i] = bind(gg_l_v[i], gg_l_v[half + i]);
            ia_v[i] = bind(ia_v[i], ia_v[half + i]);
            eq[i] = bind(eq[i], eq[half + i]);
        }
        s1h_v.truncate(half);
        s1l_v.truncate(half);
        s2h_v.truncate(half);
        s2l_v.truncate(half);
        ir_v.truncate(half);
        d_v.truncate(half);
        b_v.truncate(half);
        l_v.truncate(half);
        u_v.truncate(half);
        z_v.truncate(half);
        s_lo_z_v.truncate(half);
        s_up_z_v.truncate(half);
        s_lo_zmd_v.truncate(half);
        s_up_zmd_v.truncate(half);
        s_lo_zpd_v.truncate(half);
        s_up_zpd_v.truncate(half);
        sk1_v.truncate(half);
        sk2_v.truncate(half);
        fa_v.truncate(half);
        fb_v.truncate(half);
        q_dz_v.truncate(half);
        r_dz_v.truncate(half);
        q_dzs_v.truncate(half);
        r_dzs_v.truncate(half);
        q_bs_v.truncate(half);
        r_bs_v.truncate(half);
        ins_v.truncate(half);
        sp_v.truncate(half);
        sp_h_v.truncate(half);
        sp_l_v.truncate(half);
        gg_v.truncate(half);
        gg_h_v.truncate(half);
        gg_l_v.truncate(half);
        ia_v.truncate(half);
        eq.truncate(half);
    }

    let r_items = [
        BatchOpenSpec {
            aux: d_line_aux,
            commitment: d_line_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: b_line_aux,
            commitment: b_line_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &z_aux,
            commitment: &z_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &sigma_lo_z_aux,
            commitment: &sigma_lo_z_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &sigma_up_z_aux,
            commitment: &sigma_up_z_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &sigma_lo_zmd_aux,
            commitment: &sigma_lo_zmd_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &sigma_up_zmd_aux,
            commitment: &sigma_up_zmd_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &sigma_lo_zpd_aux,
            commitment: &sigma_lo_zpd_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &sigma_up_zpd_aux,
            commitment: &sigma_up_zpd_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &slack_fd1_aux,
            commitment: &slack_fd1_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &slack_fd2_aux,
            commitment: &slack_fd2_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &factor_a_aux,
            commitment: &factor_a_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &factor_b_aux,
            commitment: &factor_b_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &dz_step_1_aux,
            commitment: &dz_step_1_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &dz_step_1_rem_aux,
            commitment: &dz_step_1_rem_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &dz_sigma_code_aux,
            commitment: &dz_sigma_code_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &dz_sigma_rem_aux,
            commitment: &dz_sigma_rem_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &b_sigma_code_aux,
            commitment: &b_sigma_code_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &b_sigma_rem_aux,
            commitment: &b_sigma_rem_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &slack_fd1_high_aux,
            commitment: &slack_fd1_high_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &slack_fd1_low_aux,
            commitment: &slack_fd1_low_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &slack_fd2_high_aux,
            commitment: &slack_fd2_high_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &slack_fd2_low_aux,
            commitment: &slack_fd2_low_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &inside_aux,
            commitment: &inside_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &slack_pos_aux,
            commitment: &slack_pos_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &slack_pos_high_aux,
            commitment: &slack_pos_high_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &slack_pos_low_aux,
            commitment: &slack_pos_low_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &gated_gap_aux,
            commitment: &gated_gap_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &gated_gap_high_aux,
            commitment: &gated_gap_high_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &gated_gap_low_aux,
            commitment: &gated_gap_low_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &is_active_aux,
            commitment: &is_active_commit,
            commit_n_vars: n_vars,
        },
    ];
    let (r_vals, batched_open_at_r) =
        hyrax_open_batched_at(&params.committer_key, &r_items, &r_final, sponge, rng)?;
    let d_eval = r_vals[0];
    let b_eval = r_vals[1];
    let z_eval = r_vals[2];
    let sigma_lo_z_eval = r_vals[3];
    let sigma_up_z_eval = r_vals[4];
    let sigma_lo_zmd_eval = r_vals[5];
    let sigma_up_zmd_eval = r_vals[6];
    let sigma_lo_zpd_eval = r_vals[7];
    let sigma_up_zpd_eval = r_vals[8];
    let slack_fd1_eval = r_vals[9];
    let slack_fd2_eval = r_vals[10];
    let factor_a_eval = r_vals[11];
    let factor_b_eval = r_vals[12];
    let dz_step_1_eval = r_vals[13];
    let dz_step_1_rem_eval = r_vals[14];
    let dz_sigma_code_eval = r_vals[15];
    let dz_sigma_rem_eval = r_vals[16];
    let b_sigma_code_eval = r_vals[17];
    let b_sigma_rem_eval = r_vals[18];
    let slack_fd1_high_eval = r_vals[19];
    let slack_fd1_low_eval = r_vals[20];
    let slack_fd2_high_eval = r_vals[21];
    let slack_fd2_low_eval = r_vals[22];
    let inside_eval = r_vals[23];
    let slack_pos_eval = r_vals[24];
    let slack_pos_high_eval = r_vals[25];
    let slack_pos_low_eval = r_vals[26];
    let gated_gap_eval = r_vals[27];
    let gated_gap_high_eval = r_vals[28];
    let gated_gap_low_eval = r_vals[29];
    let is_active_eval = r_vals[30];

    // Open hidden-pass preact commits at r_final. The verifier
    // checks the same opens and consumes the evals in factor_a_id.
    let (preact_l_eval, preact_l_open_at_r_final) =
        crate::snark::commitment::pcs_helpers::hyrax_open_at(
            &params.committer_key,
            preact_lower_aux,
            preact_lower_commit,
            &r_final,
            sponge,
            rng,
        )?;
    let (preact_u_eval, preact_u_open_at_r_final) =
        crate::snark::commitment::pcs_helpers::hyrax_open_at(
            &params.committer_key,
            preact_upper_aux,
            preact_upper_commit,
            &r_final,
            sponge,
            rng,
        )?;
    debug_assert_eq!(
        preact_l_eval,
        eval_multilinear_full(&preact_l_padded, &r_final),
        "sshape3c: prover-side preact_l_eval drift"
    );
    debug_assert_eq!(
        preact_u_eval,
        eval_multilinear_full(&preact_u_padded, &r_final),
        "sshape3c: prover-side preact_u_eval drift"
    );

    let eq_eval = eval_multilinear_full(&build_eq_table(&r_test), &r_final);
    let fd1_id = slack_fd1_eval - (sigma_lo_z_eval - sigma_up_zmd_eval) * c_fd_sigma_fr
        + d_eval * c_fd_d_fr
        - epsilon_crit_fr;
    let fd2_id = slack_fd2_eval - d_eval * c_fd_d_fr
        + (sigma_up_zpd_eval - sigma_lo_z_eval) * c_fd_sigma_fr
        - epsilon_crit_fr;
    let factor_a_id = match line {
        SshapeLineKind::Upper => factor_a_eval - (preact_u_eval - z_eval),
        SshapeLineKind::Lower => factor_a_eval - (-z_eval - preact_l_eval),
    };
    let factor_b_id = match line {
        SshapeLineKind::Upper => {
            factor_b_eval - dz_sigma_code_eval - b_sigma_code_eval + sigma_up_z_eval
        }
        SshapeLineKind::Lower => {
            let sigma_lo_neg_z = if matches!(kind, ActivationKind::Sigmoid) {
                s_v_fr - sigma_up_z_eval
            } else {
                -sigma_up_z_eval
            };
            factor_b_eval - sigma_lo_neg_z - dz_sigma_code_eval + b_sigma_code_eval
        }
    };
    let id_dz = d_eval * z_eval - dz_step_1_eval * s_d_fr - dz_step_1_rem_eval;
    let id_dzs = dz_step_1_eval * s_v_fr - dz_sigma_code_eval * s_w_fr - dz_sigma_rem_eval;
    let id_bs = line_sign_fr * (b_eval * s_v_fr - b_sigma_code_eval * s_b_fr) - b_sigma_rem_eval;
    let id_slack1_chunk =
        slack_fd1_eval - slack_fd1_high_eval * chunk_modulus_fr - slack_fd1_low_eval;
    let id_slack2_chunk =
        slack_fd2_eval - slack_fd2_high_eval * chunk_modulus_fr - slack_fd2_low_eval;
    let one_fr_local = Fr::from(1u64);
    let id_boolean = inside_eval * (one_fr_local - inside_eval);
    let id_slack_pos =
        slack_pos_eval - inside_eval.double() * factor_a_eval + factor_a_eval + one_fr_local
            - inside_eval;
    let id_slack_pos_chunk =
        slack_pos_eval - slack_pos_high_eval * chunk_modulus_fr - slack_pos_low_eval;
    let id_gated_gap = gated_gap_eval - inside_eval * factor_b_eval;
    let id_gated_gap_chunk =
        gated_gap_eval - gated_gap_high_eval * chunk_modulus_fr - gated_gap_low_eval;
    let fd1_id_gated = is_active_eval * fd1_id;
    let fd2_id_gated = is_active_eval * fd2_id;
    let id_active_boolean = is_active_eval * (one_fr_local - is_active_eval);
    let id_active_zero = (one_fr_local - is_active_eval) * d_eval;
    let is_real_eval = eval_multilinear_full(&is_real_table, &r_final);
    let lhs = eq_eval
        * is_real_eval
        * (fd1_id_gated
            + combined_rho_a * fd2_id_gated
            + combined_rho_b * factor_a_id
            + combined_rho_c * factor_b_id
            + combined_rho_d * id_dz
            + combined_rho_e * id_dzs
            + combined_rho_f * id_bs
            + combined_rho_g * id_slack1_chunk
            + combined_rho_h * id_slack2_chunk
            + combined_rho_i * id_boolean
            + combined_rho_j * id_slack_pos
            + combined_rho_k * id_slack_pos_chunk
            + combined_rho_l * id_gated_gap
            + combined_rho_m * id_gated_gap_chunk
            + combined_rho_n * id_active_boolean
            + combined_rho_o * id_active_zero);
    if lhs != current_sum {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape3c: final identity at r_final",
        });
    }
    let _ = one_fr;

    Ok(SshapeCriticalPointProof {
        layer_idx,
        kind_tag,
        line_tag,
        n_vars,
        n_real: n,
        z_commit,
        sigma_lo_z_commit,
        sigma_up_z_commit,
        sigma_lo_zmd_commit,
        sigma_up_zmd_commit,
        sigma_lo_zpd_commit,
        sigma_up_zpd_commit,
        slack_fd1_commit,
        slack_fd2_commit,
        factor_a_commit,
        factor_b_commit,
        dz_step_1_commit,
        dz_step_1_rem_commit,
        dz_sigma_code_commit,
        dz_sigma_rem_commit,
        b_sigma_code_commit,
        b_sigma_rem_commit,
        is_active_commit,
        inside_commit,
        slack_pos_commit,
        gated_gap_commit,
        z_range,
        slack_fd1_high_commit,
        slack_fd1_low_commit,
        slack_fd2_high_commit,
        slack_fd2_low_commit,
        slack_fd1_high_range,
        slack_fd1_low_range,
        slack_fd2_high_range,
        slack_fd2_low_range,
        slack_pos_high_commit,
        slack_pos_low_commit,
        slack_pos_high_range,
        slack_pos_low_range,
        gated_gap_high_commit,
        gated_gap_low_commit,
        gated_gap_high_range,
        gated_gap_low_range,
        dz_step_1_rem_range,
        dz_sigma_rem_range,
        b_sigma_rem_range,
        envelope_combine_alpha_1,
        envelope_combine_alpha_2,
        envelope_logup_beta,
        envelope_lookup_proof,
        envelope_table_proof,
        envelope_lookup_top,
        envelope_table_top,
        envelope_witness_len: env_pow2,
        envelope_table_len: table_len,
        envelope_mult_commit,
        envelope_mult_open,
        envelope_mult_n_vars,
        envelope_sigma_lo_z_eval,
        envelope_sigma_up_z_eval,
        envelope_sigma_lo_zmd_eval,
        envelope_sigma_up_zmd_eval,
        envelope_sigma_lo_zpd_eval,
        envelope_sigma_up_zpd_eval,
        envelope_z_eval,
        envelope_witness_batched_open,
        combined_rho_a,
        combined_rho_b,
        combined_rho_c,
        combined_rho_d,
        combined_rho_e,
        combined_rho_f,
        combined_rho_g,
        combined_rho_h,
        combined_rho_i,
        combined_rho_j,
        combined_rho_k,
        combined_rho_l,
        combined_rho_m,
        combined_rho_n,
        combined_rho_o,
        r_test,
        rounds,
        r_final,
        d_eval,
        b_eval,
        preact_l_eval,
        preact_u_eval,
        z_eval,
        sigma_lo_z_eval,
        sigma_up_z_eval,
        sigma_lo_zmd_eval,
        sigma_up_zmd_eval,
        sigma_lo_zpd_eval,
        sigma_up_zpd_eval,
        slack_fd1_eval,
        slack_fd2_eval,
        factor_a_eval,
        factor_b_eval,
        dz_step_1_eval,
        dz_step_1_rem_eval,
        dz_sigma_code_eval,
        dz_sigma_rem_eval,
        b_sigma_code_eval,
        b_sigma_rem_eval,
        slack_fd1_high_eval,
        slack_fd1_low_eval,
        slack_fd2_high_eval,
        slack_fd2_low_eval,
        inside_eval,
        slack_pos_eval,
        slack_pos_high_eval,
        slack_pos_low_eval,
        gated_gap_eval,
        gated_gap_high_eval,
        gated_gap_low_eval,
        is_active_eval,
        batched_open_at_r,
        preact_l_open_at_r_final,
        preact_u_open_at_r_final,
    })
}

/// Top halves of a LogUp circuit.
fn top_halves_logup(circuit: &LogUpCircuit<Fr>) -> [Fr; 4] {
    use crate::snark_primitives::logup_gkr::LogUpLayer;
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
