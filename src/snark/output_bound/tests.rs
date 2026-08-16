//! End-to-end tests for the output-bound gadget: the inequality
//! mode plus reject-after-tamper.

use ark_bn254::Fr;
use ark_std::{rand::RngCore, test_rng};

use crate::quantized_crown::BoundDir;
use crate::snark_primitives::finite_field::signed_lift_to_fr;
use crate::snark_primitives::polynomial_commitment::{fresh_sponge, HyraxBn254, MlPcs};

use super::{
    prove_output_bound_inequality, verify_output_bound_inequality,
};
use crate::snark::commitment::commit::CommittedAux;
use crate::snark::params::SnarkParams;

/// Test-local runtime table parameters.
const TEST_HALF_BITS: i32 = 19;
const TEST_OB_BITS: usize = 19;
const TEST_GADGET_BITS: usize = 19;

fn make_params(rng: &mut impl RngCore) -> SnarkParams {
    // Mult-binding commits at TEST_OB_BITS = 19 padded to 20 vars;
    // the Hyrax key must be ≥ 20 vars even for tiny tests.
    let num_vars = TEST_OB_BITS + 1;
    let num_vars = if num_vars % 2 == 1 {
        num_vars + 1
    } else {
        num_vars
    };
    let (ck, vk) = HyraxBn254::setup(num_vars, rng).unwrap();
    SnarkParams {
        committer_key: ck,
        verifier_key: vk,
        max_num_vars: num_vars,
        precision_bits: 12,
        out_bound_range_bits: TEST_OB_BITS,
        gadget_range_bits: TEST_GADGET_BITS,
        sigma_x_scale_log2: crate::snark::preprocess::TEST_SIGMA_X_SCALE_LOG2,
        sigma_v_scale_log2: crate::snark::preprocess::TEST_SIGMA_V_SCALE_LOG2,
        input_scale_log2: None,
        preprocessed: crate::snark::preprocess::test_shared(
            TEST_HALF_BITS,
            TEST_OB_BITS,
            TEST_GADGET_BITS,
        ),
    }
}

fn pad_to_max(codes: &[i128], n_vars: usize, _max_num_vars: usize) -> Vec<Fr> {
    // Commits are sized per-tensor; pad to `1 << n_vars`.
    assert!(codes.len() <= 1 << n_vars);
    let mut out = vec![Fr::from(0u64); 1 << n_vars];
    for (slot, &c) in out.iter_mut().zip(codes.iter()) {
        *slot = signed_lift_to_fr(c);
    }
    out
}

fn pad_to_n(codes: &[i128], n_vars: usize) -> Vec<i128> {
    let mut out = vec![0i128; 1 << n_vars];
    out[..codes.len()].copy_from_slice(codes);
    out
}

/// Permissive threshold (±1000) keeping `prop_slack` inside
/// `[0, 2^OUT_BOUND_RANGE_BITS)` for the test claim magnitudes.
fn test_threshold(direction: BoundDir, n_vars: usize) -> Vec<i128> {
    let v: i128 = match direction {
        BoundDir::Lower => -1000,
        BoundDir::Upper => 1000,
    };
    vec![v; 1 << n_vars]
}

fn test_threshold_fr(direction: BoundDir, n_vars: usize) -> Vec<Fr> {
    test_threshold(direction, n_vars)
        .iter()
        .map(|&v| signed_lift_to_fr(v))
        .collect()
}

#[test]
fn ineq_upper_zero_slack_roundtrip() {
    // claimed = computed exactly. Slack = 0 everywhere. Should pass.
    let mut rng = test_rng();
    let params = make_params(&mut rng);
    let n_vars = 2usize;
    let b = pad_to_n(&[3, 5, 7, 11], n_vars);
    let aw = pad_to_n(&[2, 4, 6, 8], n_vars);
    let claimed: Vec<i128> = b.iter().zip(aw.iter()).map(|(x, y)| x + y).collect();

    let b_padded = pad_to_max(&b, n_vars, params.max_num_vars);
    let aw_padded = pad_to_max(&aw, n_vars, params.max_num_vars);
    let (b_com, b_state) =
        HyraxBn254::commit(&params.committer_key, &b_padded, Some(&mut rng)).unwrap();
    let (aw_com, aw_state) =
        HyraxBn254::commit(&params.committer_key, &aw_padded, Some(&mut rng)).unwrap();
    let b_aux: CommittedAux = (b_padded, b_state);
    let aw_aux: CommittedAux = (aw_padded, aw_state);

    let mut sp = fresh_sponge(b"ob-ineq-zero");
    let threshold = test_threshold(BoundDir::Upper, n_vars);
    let proof = prove_output_bound_inequality(
        BoundDir::Upper,
        TEST_OB_BITS,
        n_vars,
        &claimed,
        &b,
        &aw,
        Some(&threshold),
        &b_aux,
        &b_com,
        &aw_aux,
        &aw_com,
        None,
        &params,
        &mut sp,
        &mut rng,
    )
    .unwrap();

    let _ = &claimed; // claimed is committed inside the proof now.
    let mut sv = fresh_sponge(b"ob-ineq-zero");
    let threshold_fr = test_threshold_fr(BoundDir::Upper, n_vars);
    verify_output_bound_inequality(
        &proof,
        BoundDir::Upper,
        TEST_OB_BITS,
        n_vars,
        Some(&threshold_fr),
        &b_com,
        &aw_com,
        &params,
        &mut sv,
    )
    .unwrap();
}

#[test]
fn ineq_upper_loose_claim_accepted() {
    // claimed > computed everywhere. Looser bound is sound.
    let mut rng = test_rng();
    let params = make_params(&mut rng);
    let n_vars = 2usize;
    let b = pad_to_n(&[3, 5, 7, 11], n_vars);
    let aw = pad_to_n(&[2, 4, 6, 8], n_vars);
    let claimed: Vec<i128> = b
        .iter()
        .zip(aw.iter())
        .map(|(x, y)| x + y + 100) // looser by +100
        .collect();

    let b_padded = pad_to_max(&b, n_vars, params.max_num_vars);
    let aw_padded = pad_to_max(&aw, n_vars, params.max_num_vars);
    let (b_com, b_state) =
        HyraxBn254::commit(&params.committer_key, &b_padded, Some(&mut rng)).unwrap();
    let (aw_com, aw_state) =
        HyraxBn254::commit(&params.committer_key, &aw_padded, Some(&mut rng)).unwrap();
    let b_aux: CommittedAux = (b_padded, b_state);
    let aw_aux: CommittedAux = (aw_padded, aw_state);

    let mut sp = fresh_sponge(b"ob-ineq-loose");
    let threshold = test_threshold(BoundDir::Upper, n_vars);
    let proof = prove_output_bound_inequality(
        BoundDir::Upper,
        TEST_OB_BITS,
        n_vars,
        &claimed,
        &b,
        &aw,
        Some(&threshold),
        &b_aux,
        &b_com,
        &aw_aux,
        &aw_com,
        None,
        &params,
        &mut sp,
        &mut rng,
    )
    .unwrap();

    let _ = &claimed;
    let mut sv = fresh_sponge(b"ob-ineq-loose");
    let threshold_fr = test_threshold_fr(BoundDir::Upper, n_vars);
    verify_output_bound_inequality(
        &proof,
        BoundDir::Upper,
        TEST_OB_BITS,
        n_vars,
        Some(&threshold_fr),
        &b_com,
        &aw_com,
        &params,
        &mut sv,
    )
    .unwrap();
}

#[test]
fn ineq_upper_under_claim_rejected() {
    // claimed < computed for some i. Slack would be negative ⇒
    // range-check fails ⇒ verifier rejects. The prover's
    // debug_assert may fire first in debug; we run the prover
    // under catch_unwind and only verify if it returned.
    let mut rng = test_rng();
    let params = make_params(&mut rng);
    let n_vars = 2usize;
    let b = pad_to_n(&[3, 5, 7, 11], n_vars);
    let aw = pad_to_n(&[2, 4, 6, 8], n_vars);
    let mut claimed: Vec<i128> = b.iter().zip(aw.iter()).map(|(x, y)| x + y).collect();
    claimed[0] -= 50; // under-claim

    let b_padded = pad_to_max(&b, n_vars, params.max_num_vars);
    let aw_padded = pad_to_max(&aw, n_vars, params.max_num_vars);
    let (b_com, b_state) =
        HyraxBn254::commit(&params.committer_key, &b_padded, Some(&mut rng)).unwrap();
    let (aw_com, aw_state) =
        HyraxBn254::commit(&params.committer_key, &aw_padded, Some(&mut rng)).unwrap();
    let b_aux: CommittedAux = (b_padded, b_state);
    let aw_aux: CommittedAux = (aw_padded, aw_state);

    let threshold = test_threshold(BoundDir::Upper, n_vars);
    let mut sp = fresh_sponge(b"ob-ineq-under");
    let proof_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        prove_output_bound_inequality(
            BoundDir::Upper,
            TEST_OB_BITS,
            n_vars,
            &claimed,
            &b,
            &aw,
            Some(&threshold),
            &b_aux,
            &b_com,
            &aw_aux,
            &aw_com,
            None,
            &params,
            &mut sp,
            &mut rng,
        )
    }));
    if let Ok(Ok(proof)) = proof_res {
        let _ = &claimed;
        let mut sv = fresh_sponge(b"ob-ineq-under");
        let threshold_fr = test_threshold_fr(BoundDir::Upper, n_vars);
        let verdict = verify_output_bound_inequality(
            &proof,
            BoundDir::Upper,
            TEST_OB_BITS,
            n_vars,
            Some(&threshold_fr),
            &b_com,
            &aw_com,
            &params,
            &mut sv,
        );
        assert!(
            verdict.is_err(),
            "verifier must reject under-claim (slack negative)"
        );
    }
}

#[test]
fn ineq_lower_zero_slack_roundtrip() {
    let mut rng = test_rng();
    let params = make_params(&mut rng);
    let n_vars = 2usize;
    let b = pad_to_n(&[3, 5, 7, 11], n_vars);
    let aw = pad_to_n(&[2, 4, 6, 8], n_vars);
    let claimed: Vec<i128> = b.iter().zip(aw.iter()).map(|(x, y)| x + y).collect();

    let b_padded = pad_to_max(&b, n_vars, params.max_num_vars);
    let aw_padded = pad_to_max(&aw, n_vars, params.max_num_vars);
    let (b_com, b_state) =
        HyraxBn254::commit(&params.committer_key, &b_padded, Some(&mut rng)).unwrap();
    let (aw_com, aw_state) =
        HyraxBn254::commit(&params.committer_key, &aw_padded, Some(&mut rng)).unwrap();
    let b_aux: CommittedAux = (b_padded, b_state);
    let aw_aux: CommittedAux = (aw_padded, aw_state);

    let mut sp = fresh_sponge(b"ob-ineq-lower");
    let threshold = test_threshold(BoundDir::Lower, n_vars);
    let proof = prove_output_bound_inequality(
        BoundDir::Lower,
        TEST_OB_BITS,
        n_vars,
        &claimed,
        &b,
        &aw,
        Some(&threshold),
        &b_aux,
        &b_com,
        &aw_aux,
        &aw_com,
        None,
        &params,
        &mut sp,
        &mut rng,
    )
    .unwrap();

    let _ = &claimed;
    let mut sv = fresh_sponge(b"ob-ineq-lower");
    let threshold_fr = test_threshold_fr(BoundDir::Lower, n_vars);
    verify_output_bound_inequality(
        &proof,
        BoundDir::Lower,
        TEST_OB_BITS,
        n_vars,
        Some(&threshold_fr),
        &b_com,
        &aw_com,
        &params,
        &mut sv,
    )
    .unwrap();
}

#[test]
fn ineq_tampered_slack_eval_rejected() {
    let mut rng = test_rng();
    let params = make_params(&mut rng);
    let n_vars = 2usize;
    let b = pad_to_n(&[3, 5, 7, 11], n_vars);
    let aw = pad_to_n(&[2, 4, 6, 8], n_vars);
    let claimed: Vec<i128> = b.iter().zip(aw.iter()).map(|(x, y)| x + y + 5).collect();

    let b_padded = pad_to_max(&b, n_vars, params.max_num_vars);
    let aw_padded = pad_to_max(&aw, n_vars, params.max_num_vars);
    let (b_com, b_state) =
        HyraxBn254::commit(&params.committer_key, &b_padded, Some(&mut rng)).unwrap();
    let (aw_com, aw_state) =
        HyraxBn254::commit(&params.committer_key, &aw_padded, Some(&mut rng)).unwrap();
    let b_aux: CommittedAux = (b_padded, b_state);
    let aw_aux: CommittedAux = (aw_padded, aw_state);

    let mut sp = fresh_sponge(b"ob-ineq-tamper");
    let threshold = test_threshold(BoundDir::Upper, n_vars);
    let mut proof = prove_output_bound_inequality(
        BoundDir::Upper,
        TEST_OB_BITS,
        n_vars,
        &claimed,
        &b,
        &aw,
        Some(&threshold),
        &b_aux,
        &b_com,
        &aw_aux,
        &aw_com,
        None,
        &params,
        &mut sp,
        &mut rng,
    )
    .unwrap();
    proof.slack_eval += Fr::from(1u64);

    let _ = &claimed;
    let mut sv = fresh_sponge(b"ob-ineq-tamper");
    let threshold_fr = test_threshold_fr(BoundDir::Upper, n_vars);
    let verdict = verify_output_bound_inequality(
        &proof,
        BoundDir::Upper,
        TEST_OB_BITS,
        n_vars,
        Some(&threshold_fr),
        &b_com,
        &aw_com,
        &params,
        &mut sv,
    );
    assert!(verdict.is_err());
}
