//! Prover for the endpoint gadget. Builds per-cell split-arithmetic
//! witnesses, runs the σ-envelope LogUp, and folds every identity into
//! one degree-4 sumcheck.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::AdditiveGroup;
use ark_std::rand::RngCore;

use crate::crown::network::ActivationKind;
use crate::quantization::scale::Scale;
use crate::snark_primitives::finite_field::{fr_to_signed_i128, signed_lift_to_fr};
use crate::snark_primitives::logup_gkr::{prove_circuit as prove_logup_circuit, LogUpCircuit};
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};
use crate::snark_primitives::sumcheck::RoundPoly4;

use super::super::relu_upper_endpoint::{prove_pos_range, squeeze_round_challenge_4};
use crate::snark::commitment::commit::CommittedAux;
use crate::snark::commitment::multilinear_extensions::{build_eq_table, eval_multilinear_full};
use crate::snark::commitment::pcs_helpers::{hyrax_open_at, hyrax_open_batched_at, BatchOpenSpec};
use crate::snark::errors::SnarkError;
use crate::snark::output_bound::absorb_commitment;
use crate::snark::params::SnarkParams;

use super::envelope_logup::top_halves_logup;
use super::types::{SshapeEndpointKind, SshapeEndpointProof, SshapeLineKind};
use super::witness::{compute_sigma_used_fr, compute_witnesses, scale_precondition_holds};

/// Prove endpoint validity for one `(kind, line, endpoint)` triple.
/// `preact_aux` / `preact_commit` come from the hidden pass and are
/// opened at this gadget's `r_final` so the verifier never sees raw
/// preact codes.
#[allow(clippy::too_many_arguments)]
pub fn prove_sshape_at_endpoint(
    layer_idx: usize,
    kind: ActivationKind,
    line: SshapeLineKind,
    endpoint: SshapeEndpointKind,
    preact_codes: &[i128],
    preact_aux: &CommittedAux,
    preact_commit: &<HyraxBn254 as MlPcs>::Commitment,
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
) -> Result<SshapeEndpointProof, SnarkError> {
    let _timing = crate::timing::scope("sshape_endpoint");
    let kind_tag = match kind {
        ActivationKind::Sigmoid => 0u8,
        ActivationKind::Tanh => 1u8,
        ActivationKind::ReLU => {
            return Err(SnarkError::ShapeMismatch {
                what: "sshape_endpoint called for ReLU layer",
            });
        }
    };
    if !scale_precondition_holds(s_d, s_b, s_w, params.gadget_range_bits) {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape_endpoint: DENOM ≤ 2^GADGET_RANGE_BITS precondition",
        });
    }
    let n = preact_codes.len();
    if n == 0 {
        return Err(SnarkError::Reserved {
            what: "sshape_endpoint: requires n (neuron count) ≥ 1",
        });
    }
    // Padding rows are masked by the is_real MLE in the combined
    // sumcheck.
    let n_vars = crate::snark::commitment::commit::native_vector_n_vars(n);
    let n_padded = 1usize << n_vars;
    if d_line_aux.0.len() != n_padded || b_line_aux.0.len() != n_padded {
        return Err(SnarkError::ShapeMismatch {
            what: "sshape_endpoint: d/b commit n_vars doesn't match preact n_vars",
        });
    }
    let s_d_code = 1i128 << s_d.pow2_exponent().unwrap();
    let s_b_code = 1i128 << s_b.pow2_exponent().unwrap();
    let s_w_code = 1i128 << s_w.pow2_exponent().unwrap();
    let s_v_code = 1i128 << params.sigma_v_scale_log2;
    // The abs identity needs s_x ≥ s_w (abs_l is at s_x, l at s_w).
    let s_x_log2 = params.sigma_x_scale_log2;
    let s_w_log2_i = s_w.pow2_exponent().unwrap();
    if s_x_log2 < s_w_log2_i {
        return Err(SnarkError::Reserved {
            what: "sshape_endpoint: s_w > s_x not yet supported (would need rounding witness)",
        });
    }
    let s_x_over_s_w_code: i128 = 1i128 << (s_x_log2 - s_w_log2_i);

    let d_padded: Vec<i128> = d_line_aux
        .0
        .iter()
        .map(|fr| fr_to_signed_i128(*fr).unwrap_or(0))
        .collect();
    let b_padded: Vec<i128> = b_line_aux
        .0
        .iter()
        .map(|fr| fr_to_signed_i128(*fr).unwrap_or(0))
        .collect();

    let cells = compute_witnesses(
        &params.preprocessed.sigma,
        kind,
        line,
        preact_codes,
        &d_padded,
        &b_padded,
        s_d_code,
        s_b_code,
        s_w_code,
        s_v_code,
        n_padded,
    )?;

    let abs_l_codes: Vec<i128> = cells.iter().map(|c| c.abs_l).collect();
    let sign_codes: Vec<i128> = cells.iter().map(|c| c.sign).collect();
    let sigma_upper_at_abs_codes: Vec<i128> = cells.iter().map(|c| c.sigma_upper_at_abs).collect();
    let sigma_lower_at_abs_codes: Vec<i128> = cells.iter().map(|c| c.sigma_lower_at_abs).collect();
    let dx_step_1_codes: Vec<i128> = cells.iter().map(|c| c.dx_step_1).collect();
    let dx_step_1_rem_codes: Vec<i128> = cells.iter().map(|c| c.dx_step_1_rem).collect();
    let dx_sigma_code_codes: Vec<i128> = cells.iter().map(|c| c.dx_sigma_code).collect();
    let dx_sigma_rem_codes: Vec<i128> = cells.iter().map(|c| c.dx_sigma_rem).collect();
    let b_sigma_code_codes: Vec<i128> = cells.iter().map(|c| c.b_sigma_code).collect();
    let b_sigma_rem_codes: Vec<i128> = cells.iter().map(|c| c.b_sigma_rem).collect();
    let diff_codes: Vec<i128> = cells.iter().map(|c| c.diff).collect();

    let bound = 1i128 << params.gadget_range_bits;
    let check_pos = |label: &'static str, vs: &[i128]| -> Result<(), SnarkError> {
        for &v in vs {
            if v < 0 || v >= bound {
                return Err(SnarkError::RelaxationSoundnessFinalCheckFailed { which: label });
            }
        }
        Ok(())
    };
    check_pos("sshape_endpoint: abs_l out of range", &abs_l_codes)?;
    check_pos(
        "sshape_endpoint: dx_step_1_rem out of range",
        &dx_step_1_rem_codes,
    )?;
    check_pos(
        "sshape_endpoint: dx_sigma_rem out of range",
        &dx_sigma_rem_codes,
    )?;
    check_pos(
        "sshape_endpoint: b_sigma_rem out of range",
        &b_sigma_rem_codes,
    )?;
    check_pos("sshape_endpoint: diff out of range", &diff_codes)?;

    let abs_l_fr: Vec<Fr> = abs_l_codes.iter().map(|&v| signed_lift_to_fr(v)).collect();
    let sign_fr: Vec<Fr> = sign_codes.iter().map(|&v| signed_lift_to_fr(v)).collect();
    let sigma_upper_at_abs_fr: Vec<Fr> = sigma_upper_at_abs_codes
        .iter()
        .map(|&v| signed_lift_to_fr(v))
        .collect();
    let sigma_lower_at_abs_fr: Vec<Fr> = sigma_lower_at_abs_codes
        .iter()
        .map(|&v| signed_lift_to_fr(v))
        .collect();
    let dx_step_1_fr: Vec<Fr> = dx_step_1_codes
        .iter()
        .map(|&v| signed_lift_to_fr(v))
        .collect();
    let dx_step_1_rem_fr: Vec<Fr> = dx_step_1_rem_codes
        .iter()
        .map(|&v| signed_lift_to_fr(v))
        .collect();
    let dx_sigma_code_fr: Vec<Fr> = dx_sigma_code_codes
        .iter()
        .map(|&v| signed_lift_to_fr(v))
        .collect();
    let dx_sigma_rem_fr: Vec<Fr> = dx_sigma_rem_codes
        .iter()
        .map(|&v| signed_lift_to_fr(v))
        .collect();
    let b_sigma_code_fr: Vec<Fr> = b_sigma_code_codes
        .iter()
        .map(|&v| signed_lift_to_fr(v))
        .collect();
    let b_sigma_rem_fr: Vec<Fr> = b_sigma_rem_codes
        .iter()
        .map(|&v| signed_lift_to_fr(v))
        .collect();
    let diff_fr: Vec<Fr> = diff_codes.iter().map(|&v| signed_lift_to_fr(v)).collect();

    let endpoint_tag = endpoint.tag();
    let line_tag = line.tag();
    sponge.absorb(&(layer_idx as u64));
    sponge.absorb(&kind_tag);
    sponge.absorb(&endpoint_tag);
    sponge.absorb(&line_tag);
    sponge.absorb(&(n_vars as u64));
    // Bind n so a malicious prover cannot move the is_real mask
    // boundary.
    sponge.absorb(&(n as u64));
    absorb_commitment(sponge, d_line_commit);
    absorb_commitment(sponge, b_line_commit);
    // Bind preact codes via the hidden-pass commit; the verifier
    // consumes the value through an `r_final` open.
    absorb_commitment(sponge, preact_commit);
    let preact_codes_padded: Vec<Fr> = preact_codes
        .iter()
        .map(|&v| signed_lift_to_fr(v))
        .chain(std::iter::repeat_n(Fr::from(0u64), n_padded - n))
        .collect();

    let (abs_l_commit, abs_l_state) =
        HyraxBn254::commit(&params.committer_key, &abs_l_fr, Some(rng)).map_err(SnarkError::Pcs)?;
    let abs_l_aux: CommittedAux = (abs_l_fr.clone(), abs_l_state);
    absorb_commitment(sponge, &abs_l_commit);

    let (sign_commit, sign_state) =
        HyraxBn254::commit(&params.committer_key, &sign_fr, Some(rng)).map_err(SnarkError::Pcs)?;
    let sign_aux: CommittedAux = (sign_fr.clone(), sign_state);
    absorb_commitment(sponge, &sign_commit);

    let (sigma_upper_at_abs_commit, sigma_upper_at_abs_state) =
        HyraxBn254::commit(&params.committer_key, &sigma_upper_at_abs_fr, Some(rng))
            .map_err(SnarkError::Pcs)?;
    let sigma_upper_at_abs_aux: CommittedAux =
        (sigma_upper_at_abs_fr.clone(), sigma_upper_at_abs_state);
    absorb_commitment(sponge, &sigma_upper_at_abs_commit);

    let (sigma_lower_at_abs_commit, sigma_lower_at_abs_state) =
        HyraxBn254::commit(&params.committer_key, &sigma_lower_at_abs_fr, Some(rng))
            .map_err(SnarkError::Pcs)?;
    let sigma_lower_at_abs_aux: CommittedAux =
        (sigma_lower_at_abs_fr.clone(), sigma_lower_at_abs_state);
    absorb_commitment(sponge, &sigma_lower_at_abs_commit);

    let (dx_step_1_commit, st) =
        HyraxBn254::commit(&params.committer_key, &dx_step_1_fr, Some(rng))
            .map_err(SnarkError::Pcs)?;
    let dx_step_1_aux: CommittedAux = (dx_step_1_fr.clone(), st);
    absorb_commitment(sponge, &dx_step_1_commit);
    let (dx_step_1_rem_commit, st) =
        HyraxBn254::commit(&params.committer_key, &dx_step_1_rem_fr, Some(rng))
            .map_err(SnarkError::Pcs)?;
    let dx_step_1_rem_aux: CommittedAux = (dx_step_1_rem_fr.clone(), st);
    absorb_commitment(sponge, &dx_step_1_rem_commit);
    let (dx_sigma_code_commit, st) =
        HyraxBn254::commit(&params.committer_key, &dx_sigma_code_fr, Some(rng))
            .map_err(SnarkError::Pcs)?;
    let dx_sigma_code_aux: CommittedAux = (dx_sigma_code_fr.clone(), st);
    absorb_commitment(sponge, &dx_sigma_code_commit);
    let (dx_sigma_rem_commit, st) =
        HyraxBn254::commit(&params.committer_key, &dx_sigma_rem_fr, Some(rng))
            .map_err(SnarkError::Pcs)?;
    let dx_sigma_rem_aux: CommittedAux = (dx_sigma_rem_fr.clone(), st);
    absorb_commitment(sponge, &dx_sigma_rem_commit);
    let (b_sigma_code_commit, st) =
        HyraxBn254::commit(&params.committer_key, &b_sigma_code_fr, Some(rng))
            .map_err(SnarkError::Pcs)?;
    let b_sigma_code_aux: CommittedAux = (b_sigma_code_fr.clone(), st);
    absorb_commitment(sponge, &b_sigma_code_commit);
    let (b_sigma_rem_commit, st) =
        HyraxBn254::commit(&params.committer_key, &b_sigma_rem_fr, Some(rng))
            .map_err(SnarkError::Pcs)?;
    let b_sigma_rem_aux: CommittedAux = (b_sigma_rem_fr.clone(), st);
    absorb_commitment(sponge, &b_sigma_rem_commit);
    let (diff_commit, st) =
        HyraxBn254::commit(&params.committer_key, &diff_fr, Some(rng)).map_err(SnarkError::Pcs)?;
    let diff_aux: CommittedAux = (diff_fr.clone(), st);
    absorb_commitment(sponge, &diff_commit);

    let abs_l_range = prove_pos_range(
        &abs_l_fr,
        &abs_l_codes,
        &abs_l_aux,
        &abs_l_commit,
        params,
        sponge,
        rng,
    )?;
    let dx_step_1_rem_range = prove_pos_range(
        &dx_step_1_rem_fr,
        &dx_step_1_rem_codes,
        &dx_step_1_rem_aux,
        &dx_step_1_rem_commit,
        params,
        sponge,
        rng,
    )?;
    let dx_sigma_rem_range = prove_pos_range(
        &dx_sigma_rem_fr,
        &dx_sigma_rem_codes,
        &dx_sigma_rem_aux,
        &dx_sigma_rem_commit,
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
    let diff_range = prove_pos_range(
        &diff_fr,
        &diff_codes,
        &diff_aux,
        &diff_commit,
        params,
        sponge,
        rng,
    )?;

    sponge.absorb(&(n_padded as u64));
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
    let envelope_witness: Vec<Fr> = (0..n_padded)
        .map(|j| {
            envelope_combine_alpha_1 * abs_l_fr[j]
                + envelope_combine_alpha_2 * sigma_upper_at_abs_fr[j]
                + sigma_lower_at_abs_fr[j]
        })
        .collect();
    let mut envelope_mults_u64 = vec![0u64; table_len];
    for j in 0..n_padded {
        let idx = abs_l_codes[j] as usize;
        if idx < table_len {
            envelope_mults_u64[idx] += 1;
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

    let logup_point = envelope_lookup_proof.bottom_point.clone();
    let logup_items = [
        BatchOpenSpec {
            aux: &abs_l_aux,
            commitment: &abs_l_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &sigma_upper_at_abs_aux,
            commitment: &sigma_upper_at_abs_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &sigma_lower_at_abs_aux,
            commitment: &sigma_lower_at_abs_commit,
            commit_n_vars: n_vars,
        },
    ];
    let (logup_evals, envelope_witness_batched_open) = hyrax_open_batched_at(
        &params.committer_key,
        &logup_items,
        &logup_point,
        sponge,
        rng,
    )?;
    let envelope_abs_l_eval = logup_evals[0];
    let envelope_sigma_upper_at_abs_eval = logup_evals[1];
    let envelope_sigma_lower_at_abs_eval = logup_evals[2];

    debug_assert_eq!(
        envelope_lookup_proof.bottom_denom,
        envelope_combine_alpha_1 * envelope_abs_l_eval
            + envelope_combine_alpha_2 * envelope_sigma_upper_at_abs_eval
            + envelope_sigma_lower_at_abs_eval
            - envelope_logup_beta,
        "envelope LogUp witness bottom_denom mismatch"
    );

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
            which: "sshape_endpoint: envelope mult open mismatch",
        });
    }
    drop(lu_timing);

    // Combined sumcheck: six identities folded with ρ_a..ρ_e
    // (id_1 carries an implicit coefficient 1). The line_sign factor
    // is +1 (upper, floor) or −1 (lower, ceil).
    //   id_1: d·l − dx_step_1·s_d − line_sign·dx_step_1_rem
    //   id_2: dx_step_1·s_v − dx_sigma_code·s_w − line_sign·dx_sigma_rem
    //   id_3: b·s_v − b_sigma_code·s_b − line_sign·b_sigma_rem
    //   id_4: line_sign·(dx_sigma_code + b_sigma_code − σ_used) − diff
    //   id_5: sign·(1 − sign)
    //   id_6: l·(s_x/s_w) − abs_l + 2·sign·abs_l
    let combined_rho_a = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_b = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_c = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_d = sponge.squeeze_field_elements::<Fr>(1)[0];
    let combined_rho_e = sponge.squeeze_field_elements::<Fr>(1)[0];

    let s_d_fr = signed_lift_to_fr(s_d_code);
    let s_b_fr = signed_lift_to_fr(s_b_code);
    let s_w_fr = signed_lift_to_fr(s_w_code);
    let s_v_fr = signed_lift_to_fr(s_v_code);
    let s_x_over_s_w_fr = signed_lift_to_fr(s_x_over_s_w_code);
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
    let mut l_v = preact_codes_padded.clone();
    let mut abs_l_v = abs_l_fr.clone();
    let mut sign_v = sign_fr.clone();
    let mut sup_v = sigma_upper_at_abs_fr.clone();
    let mut slo_v = sigma_lower_at_abs_fr.clone();
    let mut q1_v = dx_step_1_fr.clone();
    let mut r1_v = dx_step_1_rem_fr.clone();
    let mut q2_v = dx_sigma_code_fr.clone();
    let mut r2_v = dx_sigma_rem_fr.clone();
    let mut q3_v = b_sigma_code_fr.clone();
    let mut r3_v = b_sigma_rem_fr.clone();
    let mut diff_v = diff_fr.clone();
    let mut current_sum = Fr::ZERO;
    let mut rounds: Vec<RoundPoly4<Fr>> = Vec::with_capacity(n_vars);
    let mut r_final: Vec<Fr> = Vec::with_capacity(n_vars);

    let inner_eval = |q: Fr,
                      ir: Fr,
                      d: Fr,
                      bo: Fr,
                      l: Fr,
                      abs_l: Fr,
                      sign: Fr,
                      sup: Fr,
                      slo: Fr,
                      q1: Fr,
                      r1: Fr,
                      q2: Fr,
                      r2: Fr,
                      q3: Fr,
                      r3: Fr,
                      diff: Fr|
     -> Fr {
        let s_used = compute_sigma_used_fr(kind, line, sup, slo, sign, s_v_fr);
        let one = Fr::from(1u64);
        let two = Fr::from(2u64);
        let id_1 = d * l - q1 * s_d_fr - line_sign_fr * r1;
        let id_2 = q1 * s_v_fr - q2 * s_w_fr - line_sign_fr * r2;
        let id_3 = bo * s_v_fr - q3 * s_b_fr - line_sign_fr * r3;
        let id_4 = line_sign_fr * (q2 + q3 - s_used) - diff;
        let id_5 = sign * (one - sign);
        let id_6 = l * s_x_over_s_w_fr - abs_l + two * sign * abs_l;
        q * ir
            * (id_1
                + combined_rho_a * id_2
                + combined_rho_b * id_3
                + combined_rho_c * id_4
                + combined_rho_d * id_5
                + combined_rho_e * id_6)
    };

    for _ in 0..n_vars {
        let half = eq.len() / 2;
        let (mut e0, mut e1, mut e2, mut e3, mut e4) =
            (Fr::ZERO, Fr::ZERO, Fr::ZERO, Fr::ZERO, Fr::ZERO);
        for i in 0..half {
            let lin = |a0: Fr, a1: Fr| {
                let d = a1 - a0;
                (a0, a1, a1 + d, a1 + d.double(), a1 + d + d.double())
            };
            let (q0, q1_, q2_, q3_, q4_) = lin(eq[i], eq[half + i]);
            let (ir0, ir1_, ir2_, ir3_, ir4_) = lin(ir_v[i], ir_v[half + i]);
            let (d0, d1, d2, d3, d4) = lin(d_v[i], d_v[half + i]);
            let (bo0, bo1, bo2, bo3, bo4) = lin(b_v[i], b_v[half + i]);
            let (l0, l1, l2, l3, l4) = lin(l_v[i], l_v[half + i]);
            let (al0, al1, al2, al3, al4) = lin(abs_l_v[i], abs_l_v[half + i]);
            let (sg0, sg1, sg2, sg3, sg4) = lin(sign_v[i], sign_v[half + i]);
            let (su0, su1, su2, su3, su4) = lin(sup_v[i], sup_v[half + i]);
            let (sl0, sl1, sl2, sl3, sl4) = lin(slo_v[i], slo_v[half + i]);
            let (qa0, qa1, qa2, qa3, qa4) = lin(q1_v[i], q1_v[half + i]);
            let (ra0, ra1, ra2, ra3, ra4) = lin(r1_v[i], r1_v[half + i]);
            let (qb0, qb1, qb2, qb3, qb4) = lin(q2_v[i], q2_v[half + i]);
            let (rb0, rb1, rb2, rb3, rb4) = lin(r2_v[i], r2_v[half + i]);
            let (qc0, qc1, qc2, qc3, qc4) = lin(q3_v[i], q3_v[half + i]);
            let (rc0, rc1, rc2, rc3, rc4) = lin(r3_v[i], r3_v[half + i]);
            let (df0, df1, df2, df3, df4) = lin(diff_v[i], diff_v[half + i]);
            e0 += inner_eval(
                q0, ir0, d0, bo0, l0, al0, sg0, su0, sl0, qa0, ra0, qb0, rb0, qc0, rc0, df0,
            );
            e1 += inner_eval(
                q1_, ir1_, d1, bo1, l1, al1, sg1, su1, sl1, qa1, ra1, qb1, rb1, qc1, rc1, df1,
            );
            e2 += inner_eval(
                q2_, ir2_, d2, bo2, l2, al2, sg2, su2, sl2, qa2, ra2, qb2, rb2, qc2, rc2, df2,
            );
            e3 += inner_eval(
                q3_, ir3_, d3, bo3, l3, al3, sg3, su3, sl3, qa3, ra3, qb3, rb3, qc3, rc3, df3,
            );
            e4 += inner_eval(
                q4_, ir4_, d4, bo4, l4, al4, sg4, su4, sl4, qa4, ra4, qb4, rb4, qc4, rc4, df4,
            );
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
                which: "sshape_endpoint: sumcheck round invariant",
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
            abs_l_v[i] = bind(abs_l_v[i], abs_l_v[half + i]);
            sign_v[i] = bind(sign_v[i], sign_v[half + i]);
            sup_v[i] = bind(sup_v[i], sup_v[half + i]);
            slo_v[i] = bind(slo_v[i], slo_v[half + i]);
            q1_v[i] = bind(q1_v[i], q1_v[half + i]);
            r1_v[i] = bind(r1_v[i], r1_v[half + i]);
            q2_v[i] = bind(q2_v[i], q2_v[half + i]);
            r2_v[i] = bind(r2_v[i], r2_v[half + i]);
            q3_v[i] = bind(q3_v[i], q3_v[half + i]);
            r3_v[i] = bind(r3_v[i], r3_v[half + i]);
            diff_v[i] = bind(diff_v[i], diff_v[half + i]);
            eq[i] = bind(eq[i], eq[half + i]);
        }
        ir_v.truncate(half);
        d_v.truncate(half);
        b_v.truncate(half);
        l_v.truncate(half);
        abs_l_v.truncate(half);
        sign_v.truncate(half);
        sup_v.truncate(half);
        slo_v.truncate(half);
        q1_v.truncate(half);
        r1_v.truncate(half);
        q2_v.truncate(half);
        r2_v.truncate(half);
        q3_v.truncate(half);
        r3_v.truncate(half);
        diff_v.truncate(half);
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
            aux: &abs_l_aux,
            commitment: &abs_l_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &sign_aux,
            commitment: &sign_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &sigma_upper_at_abs_aux,
            commitment: &sigma_upper_at_abs_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &sigma_lower_at_abs_aux,
            commitment: &sigma_lower_at_abs_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &dx_step_1_aux,
            commitment: &dx_step_1_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &dx_step_1_rem_aux,
            commitment: &dx_step_1_rem_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &dx_sigma_code_aux,
            commitment: &dx_sigma_code_commit,
            commit_n_vars: n_vars,
        },
        BatchOpenSpec {
            aux: &dx_sigma_rem_aux,
            commitment: &dx_sigma_rem_commit,
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
            aux: &diff_aux,
            commitment: &diff_commit,
            commit_n_vars: n_vars,
        },
    ];
    let (r_vals, batched_open_at_r) =
        hyrax_open_batched_at(&params.committer_key, &r_items, &r_final, sponge, rng)?;
    let d_line_eval = r_vals[0];
    let b_line_eval = r_vals[1];
    let abs_l_eval = r_vals[2];
    let sign_eval = r_vals[3];
    let sigma_upper_at_abs_eval = r_vals[4];
    let sigma_lower_at_abs_eval = r_vals[5];
    let dx_step_1_eval = r_vals[6];
    let dx_step_1_rem_eval = r_vals[7];
    let dx_sigma_code_eval = r_vals[8];
    let dx_sigma_rem_eval = r_vals[9];
    let b_sigma_code_eval = r_vals[10];
    let b_sigma_rem_eval = r_vals[11];
    let diff_eval = r_vals[12];

    // Open the hidden-pass preact commit at r_final so the verifier
    // never reads raw preact codes.
    let (preact_eval_at_r_final, preact_open_at_r_final) =
        crate::snark::commitment::pcs_helpers::hyrax_open_at(
            &params.committer_key,
            preact_aux,
            preact_commit,
            &r_final,
            sponge,
            rng,
        )?;
    let l_eval_check = eval_multilinear_full(&preact_codes_padded, &r_final);
    debug_assert_eq!(
        preact_eval_at_r_final, l_eval_check,
        "sshape_endpoint: preact open eval ≠ MLE — Hyrax bug or wrong aux"
    );
    let l_eval = preact_eval_at_r_final;
    let eq_eval = eval_multilinear_full(&build_eq_table(&r_test), &r_final);
    let s_used_eval = compute_sigma_used_fr(
        kind,
        line,
        sigma_upper_at_abs_eval,
        sigma_lower_at_abs_eval,
        sign_eval,
        s_v_fr,
    );
    let one = Fr::from(1u64);
    let two = Fr::from(2u64);
    let id_1_eval =
        d_line_eval * l_eval - dx_step_1_eval * s_d_fr - line_sign_fr * dx_step_1_rem_eval;
    let id_2_eval =
        dx_step_1_eval * s_v_fr - dx_sigma_code_eval * s_w_fr - line_sign_fr * dx_sigma_rem_eval;
    let id_3_eval =
        b_line_eval * s_v_fr - b_sigma_code_eval * s_b_fr - line_sign_fr * b_sigma_rem_eval;
    let id_4_eval =
        line_sign_fr * (dx_sigma_code_eval + b_sigma_code_eval - s_used_eval) - diff_eval;
    let id_5_eval = sign_eval * (one - sign_eval);
    let id_6_eval = l_eval * s_x_over_s_w_fr - abs_l_eval + two * sign_eval * abs_l_eval;
    let is_real_eval = eval_multilinear_full(&is_real_table, &r_final);
    let lhs = eq_eval
        * is_real_eval
        * (id_1_eval
            + combined_rho_a * id_2_eval
            + combined_rho_b * id_3_eval
            + combined_rho_c * id_4_eval
            + combined_rho_d * id_5_eval
            + combined_rho_e * id_6_eval);
    if lhs != current_sum {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape_endpoint: final identity at r' (split-arith)",
        });
    }

    Ok(SshapeEndpointProof {
        layer_idx,
        kind_tag,
        endpoint_tag,
        line_tag,
        n_vars,
        n_real: n,
        abs_l_commit,
        sign_commit,
        sigma_upper_at_abs_commit,
        sigma_lower_at_abs_commit,
        dx_step_1_commit,
        dx_step_1_rem_commit,
        dx_sigma_code_commit,
        dx_sigma_rem_commit,
        b_sigma_code_commit,
        b_sigma_rem_commit,
        diff_commit,
        abs_l_range,
        dx_step_1_rem_range,
        dx_sigma_rem_range,
        b_sigma_rem_range,
        diff_range,
        envelope_combine_alpha_1,
        envelope_combine_alpha_2,
        envelope_logup_beta,
        envelope_lookup_proof,
        envelope_table_proof,
        envelope_lookup_top,
        envelope_table_top,
        envelope_witness_len: n_padded,
        envelope_table_len: table_len,
        envelope_mult_commit,
        envelope_mult_open,
        envelope_mult_n_vars,
        envelope_abs_l_eval,
        envelope_sigma_upper_at_abs_eval,
        envelope_sigma_lower_at_abs_eval,
        envelope_witness_batched_open,
        combined_rho_a,
        combined_rho_b,
        combined_rho_c,
        combined_rho_d,
        combined_rho_e,
        r_test,
        rounds,
        r_final,
        d_line_eval,
        b_line_eval,
        abs_l_eval,
        sign_eval,
        sigma_upper_at_abs_eval,
        sigma_lower_at_abs_eval,
        dx_step_1_eval,
        dx_step_1_rem_eval,
        dx_sigma_code_eval,
        dx_sigma_rem_eval,
        b_sigma_code_eval,
        b_sigma_rem_eval,
        diff_eval,
        batched_open_at_r,
        preact_eval_at_r_final,
        preact_open_at_r_final,
    })
}
