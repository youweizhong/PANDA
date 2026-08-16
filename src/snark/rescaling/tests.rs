//! End-to-end tests for the rescale gadget. Cover happy-path
//! round-trip and tamper rejections.

use ark_bn254::Fr;
use ark_std::{rand::RngCore, test_rng};

use crate::quantization::quantized_scalar::Qf;
use crate::quantization::scale::Scale;
use crate::snark_primitives::finite_field::signed_lift_to_fr;
use crate::snark_primitives::polynomial_commitment::{fresh_sponge, HyraxBn254, MlPcs};

use super::{prove_rescale_event, verify_rescale_event, RescaleEventDesc};
use crate::snark::commitment::commit::CommittedAux;
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;

/// Build a small rescale event by running real `Qf::rescale`, so the
/// `slack_lo`/`qz` pair satisfies the boxed inequality cell-wise.
fn build_event(qx_codes: &[i128], s_in: Scale, s_out: Scale) -> (Vec<i128>, Vec<i128>, i128, i128) {
    let mut qz = Vec::with_capacity(qx_codes.len());
    let mut slack_lo = Vec::with_capacity(qx_codes.len());
    let mut c1 = 0i128;
    let mut c2 = 0i128;
    for &qx in qx_codes {
        let (out, w) = Qf::new(qx, s_in).rescale(s_out).unwrap();
        qz.push(out.code);
        slack_lo.push(w.slack_lo);
        c1 = w.c1;
        c2 = w.c2;
    }
    (qz, slack_lo, c1, c2)
}

/// Test-local runtime table parameters.
const TEST_HALF_BITS: i32 = 19;
const TEST_OB_BITS: usize = 19;
const TEST_GADGET_BITS: usize = 19;

fn make_params(rng: &mut impl RngCore) -> SnarkParams {
    let num_vars = 12;
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
    assert!(codes.len() <= 1 << n_vars);
    let mut out = vec![Fr::from(0u64); 1 << n_vars];
    for (slot, &v) in out.iter_mut().zip(codes.iter()) {
        *slot = signed_lift_to_fr(v);
    }
    out
}

#[test]
fn round_trip_2d_matrix_with_c2_padding() {
    // 2×3 matrix padded to 2×4; padding cells have qx=qz=0, slack_lo=c2.
    let mut rng = test_rng();
    let params = make_params(&mut rng);
    let s_in = Scale::from_pow2(8);
    let s_out = Scale::from_pow2(2);
    let rows = 2usize;
    let cols = 3usize;
    let pow_cols = 4usize;
    // log2(2) + log2(4) = 3, bumped to even for Hyrax.
    let n_vars = 4usize;
    let n_padded = 1usize << n_vars;
    let mut slack_lo: Vec<i128> = Vec::with_capacity(n_padded);
    let mut qx_codes: Vec<i128> = vec![0; n_padded];
    let mut qz_codes: Vec<i128> = vec![0; n_padded];
    let mut c1 = 0i128;
    let mut c2 = 0i128;
    for i in 0..rows {
        for j in 0..pow_cols {
            let qx = if j < cols {
                (i as i128 * 7 + j as i128 * 11 - 5) * 13
            } else {
                0
            };
            let (out, w) = Qf::new(qx, s_in).rescale(s_out).unwrap();
            qx_codes[i * pow_cols + j] = qx;
            qz_codes[i * pow_cols + j] = out.code;
            slack_lo.push(w.slack_lo);
            c1 = w.c1;
            c2 = w.c2;
        }
    }
    // Padding cells (qx=qz=0) get slack_lo=c2 from Qf::rescale.
    slack_lo.resize(n_padded, c2);
    let qx_padded = pad_to_max(&qx_codes, n_vars, params.max_num_vars);
    let qz_padded = pad_to_max(&qz_codes, n_vars, params.max_num_vars);
    let (qx_com, qx_state) =
        HyraxBn254::commit(&params.committer_key, &qx_padded, Some(&mut rng)).unwrap();
    let (qz_com, qz_state) =
        HyraxBn254::commit(&params.committer_key, &qz_padded, Some(&mut rng)).unwrap();
    let qx_aux: CommittedAux = (qx_padded.clone(), qx_state);
    let qz_aux: CommittedAux = (qz_padded.clone(), qz_state);

    let desc = RescaleEventDesc {
        c1,
        c2,
        n_vars,
        dir: crate::quantization::quantized_scalar::RoundDir::HalfAway,
    };
    let mut sp = fresh_sponge(b"rescale-2d-matrix");
    let proof = prove_rescale_event(
        &desc,
        &slack_lo,
        &qx_padded[..(1 << n_vars)],
        &qz_padded[..(1 << n_vars)],
        &qx_aux,
        &qx_com,
        &qz_aux,
        &qz_com,
        &params,
        &mut sp,
        &mut rng,
    )
    .unwrap();
    let mut sv = fresh_sponge(b"rescale-2d-matrix");
    verify_rescale_event(&proof, &desc, &qx_com, &qz_com, &params, &mut sv).unwrap();
}

#[test]
fn round_trip_pure_pow2_rescale() {
    let mut rng = test_rng();
    let params = make_params(&mut rng);
    let s_in = Scale::from_pow2(8);
    let s_out = Scale::from_pow2(2);
    let qx_codes: Vec<i128> = (0..16).map(|i| (i as i128 - 8) * 13).collect();
    let (qz_codes, slack_lo, c1, c2) = build_event(&qx_codes, s_in, s_out);

    let n_vars = 4;
    let qx_padded = pad_to_max(&qx_codes, n_vars, params.max_num_vars);
    let qz_padded = pad_to_max(&qz_codes, n_vars, params.max_num_vars);
    let (qx_com, qx_state) =
        HyraxBn254::commit(&params.committer_key, &qx_padded, Some(&mut rng)).unwrap();
    let (qz_com, qz_state) =
        HyraxBn254::commit(&params.committer_key, &qz_padded, Some(&mut rng)).unwrap();
    let qx_aux: CommittedAux = (qx_padded.clone(), qx_state);
    let qz_aux: CommittedAux = (qz_padded.clone(), qz_state);

    let desc = RescaleEventDesc {
        c1,
        c2,
        n_vars,
        dir: crate::quantization::quantized_scalar::RoundDir::HalfAway,
    };
    let mut sp = fresh_sponge(b"rescale-roundtrip");
    let proof = prove_rescale_event(
        &desc,
        &slack_lo,
        &qx_padded[..(1 << n_vars)],
        &qz_padded[..(1 << n_vars)],
        &qx_aux,
        &qx_com,
        &qz_aux,
        &qz_com,
        &params,
        &mut sp,
        &mut rng,
    )
    .unwrap();

    let mut sv = fresh_sponge(b"rescale-roundtrip");
    verify_rescale_event(&proof, &desc, &qx_com, &qz_com, &params, &mut sv).unwrap();
}

#[test]
fn tampered_qz_evaluation_rejected() {
    let mut rng = test_rng();
    let params = make_params(&mut rng);
    let s_in = Scale::from_pow2(6);
    let s_out = Scale::from_pow2(2);
    let qx_codes: Vec<i128> = (0..8).map(|i| (i as i128 - 4) * 11).collect();
    let (qz_codes, mut slack_lo, c1, c2) = build_event(&qx_codes, s_in, s_out);

    // 8 cells → log2=3, bumped to 4 for Hyrax; trailing slack = c2.
    let n_vars = 4;
    slack_lo.resize(1usize << n_vars, c2);
    let qx_padded = pad_to_max(&qx_codes, n_vars, params.max_num_vars);
    let qz_padded = pad_to_max(&qz_codes, n_vars, params.max_num_vars);
    let (qx_com, qx_state) =
        HyraxBn254::commit(&params.committer_key, &qx_padded, Some(&mut rng)).unwrap();
    let (qz_com, qz_state) =
        HyraxBn254::commit(&params.committer_key, &qz_padded, Some(&mut rng)).unwrap();
    let qx_aux: CommittedAux = (qx_padded.clone(), qx_state);
    let qz_aux: CommittedAux = (qz_padded.clone(), qz_state);

    let desc = RescaleEventDesc {
        c1,
        c2,
        n_vars,
        dir: crate::quantization::quantized_scalar::RoundDir::HalfAway,
    };
    let mut sp = fresh_sponge(b"rescale-tamper-qz");
    let mut proof = prove_rescale_event(
        &desc,
        &slack_lo,
        &qx_padded[..(1 << n_vars)],
        &qz_padded[..(1 << n_vars)],
        &qx_aux,
        &qx_com,
        &qz_aux,
        &qz_com,
        &params,
        &mut sp,
        &mut rng,
    )
    .unwrap();
    proof.qz_eval += Fr::from(1u64);
    let mut sv = fresh_sponge(b"rescale-tamper-qz");
    let err = verify_rescale_event(&proof, &desc, &qx_com, &qz_com, &params, &mut sv);
    assert!(matches!(
        err,
        Err(SnarkError::PcsOpenRejected { .. }) | Err(SnarkError::RescaleIdentityFailed)
    ));
}

#[test]
fn tampered_slack_lo_breaks_range() {
    let mut rng = test_rng();
    let params = make_params(&mut rng);
    let s_in = Scale::from_pow2(6);
    let s_out = Scale::from_pow2(2);
    let qx_codes: Vec<i128> = vec![1, 2, 3, 4];
    let (qz_codes, mut slack_lo, c1, c2) = build_event(&qx_codes, s_in, s_out);
    // Force slack_lo[0] above 2c2 to violate the range.
    slack_lo[0] = 2 * c2;
    let n_vars = 2;
    let qx_padded = pad_to_max(&qx_codes, n_vars, params.max_num_vars);
    let qz_padded = pad_to_max(&qz_codes, n_vars, params.max_num_vars);
    let (qx_com, qx_state) =
        HyraxBn254::commit(&params.committer_key, &qx_padded, Some(&mut rng)).unwrap();
    let (qz_com, qz_state) =
        HyraxBn254::commit(&params.committer_key, &qz_padded, Some(&mut rng)).unwrap();
    let qx_aux: CommittedAux = (qx_padded.clone(), qx_state);
    let qz_aux: CommittedAux = (qz_padded.clone(), qz_state);

    let desc = RescaleEventDesc {
        c1,
        c2,
        n_vars,
        dir: crate::quantization::quantized_scalar::RoundDir::HalfAway,
    };
    let mut sp = fresh_sponge(b"rescale-tamper-slack");
    // In debug builds prove fires the identity assertion; in release
    // a proof is produced and verify rejects via the range check.
    let proof_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        prove_rescale_event(
            &desc,
            &slack_lo,
            &qx_padded[..(1 << n_vars)],
            &qz_padded[..(1 << n_vars)],
            &qx_aux,
            &qx_com,
            &qz_aux,
            &qz_com,
            &params,
            &mut sp,
            &mut rng,
        )
    }));
    if let Ok(Ok(proof)) = proof_res {
        let mut sv = fresh_sponge(b"rescale-tamper-slack");
        let err = verify_rescale_event(&proof, &desc, &qx_com, &qz_com, &params, &mut sv);
        assert!(err.is_err());
    }
}

#[test]
fn directional_floor_round_trip() {
    let mut rng = test_rng();
    let params = make_params(&mut rng);
    let s_in = Scale::from_pow2(10);
    let s_out = Scale::from_pow2(4);
    let n_vars = 2usize;
    let n_padded = 1usize << n_vars;

    let mut qx_codes = vec![0i128; n_padded];
    let mut qz_codes = vec![0i128; n_padded];
    let mut slack_lo = vec![0i128; n_padded];
    let mut c1 = 0i128;
    let mut c2 = 0i128;
    let inputs = [97i128, -45, 12, -3];
    for (i, &qx) in inputs.iter().enumerate() {
        let (out, w) = Qf::new(qx, s_in)
            .rescale_dir(
                s_out,
                crate::quantization::quantized_scalar::RoundDir::Floor,
            )
            .unwrap();
        qx_codes[i] = qx;
        qz_codes[i] = out.code;
        slack_lo[i] = w.slack_lo;
        c1 = w.c1;
        c2 = w.c2;
    }

    let qx_padded = pad_to_max(&qx_codes, n_vars, params.max_num_vars);
    let qz_padded = pad_to_max(&qz_codes, n_vars, params.max_num_vars);
    let (qx_com, qx_state) =
        HyraxBn254::commit(&params.committer_key, &qx_padded, Some(&mut rng)).unwrap();
    let (qz_com, qz_state) =
        HyraxBn254::commit(&params.committer_key, &qz_padded, Some(&mut rng)).unwrap();
    let qx_aux: CommittedAux = (qx_padded.clone(), qx_state);
    let qz_aux: CommittedAux = (qz_padded.clone(), qz_state);

    let desc = RescaleEventDesc {
        c1,
        c2,
        n_vars,
        dir: crate::quantization::quantized_scalar::RoundDir::Floor,
    };
    let mut sp = fresh_sponge(b"rescale-floor");
    let proof = prove_rescale_event(
        &desc,
        &slack_lo,
        &qx_padded[..(1 << n_vars)],
        &qz_padded[..(1 << n_vars)],
        &qx_aux,
        &qx_com,
        &qz_aux,
        &qz_com,
        &params,
        &mut sp,
        &mut rng,
    )
    .unwrap();
    let mut sv = fresh_sponge(b"rescale-floor");
    verify_rescale_event(&proof, &desc, &qx_com, &qz_com, &params, &mut sv).unwrap();
}

#[test]
fn directional_ceil_round_trip() {
    let mut rng = test_rng();
    let params = make_params(&mut rng);
    let s_in = Scale::from_pow2(10);
    let s_out = Scale::from_pow2(4);
    let n_vars = 2usize;
    let n_padded = 1usize << n_vars;

    let mut qx_codes = vec![0i128; n_padded];
    let mut qz_codes = vec![0i128; n_padded];
    let mut slack_lo = vec![0i128; n_padded];
    let mut c1 = 0i128;
    let mut c2 = 0i128;
    let inputs = [97i128, -45, 12, -3];
    for (i, &qx) in inputs.iter().enumerate() {
        let (out, w) = Qf::new(qx, s_in)
            .rescale_dir(s_out, crate::quantization::quantized_scalar::RoundDir::Ceil)
            .unwrap();
        qx_codes[i] = qx;
        qz_codes[i] = out.code;
        slack_lo[i] = w.slack_lo;
        c1 = w.c1;
        c2 = w.c2;
    }

    let qx_padded = pad_to_max(&qx_codes, n_vars, params.max_num_vars);
    let qz_padded = pad_to_max(&qz_codes, n_vars, params.max_num_vars);
    let (qx_com, qx_state) =
        HyraxBn254::commit(&params.committer_key, &qx_padded, Some(&mut rng)).unwrap();
    let (qz_com, qz_state) =
        HyraxBn254::commit(&params.committer_key, &qz_padded, Some(&mut rng)).unwrap();
    let qx_aux: CommittedAux = (qx_padded.clone(), qx_state);
    let qz_aux: CommittedAux = (qz_padded.clone(), qz_state);

    let desc = RescaleEventDesc {
        c1,
        c2,
        n_vars,
        dir: crate::quantization::quantized_scalar::RoundDir::Ceil,
    };
    let mut sp = fresh_sponge(b"rescale-ceil");
    let proof = prove_rescale_event(
        &desc,
        &slack_lo,
        &qx_padded[..(1 << n_vars)],
        &qz_padded[..(1 << n_vars)],
        &qx_aux,
        &qx_com,
        &qz_aux,
        &qz_com,
        &params,
        &mut sp,
        &mut rng,
    )
    .unwrap();
    let mut sv = fresh_sponge(b"rescale-ceil");
    verify_rescale_event(&proof, &desc, &qx_com, &qz_com, &params, &mut sv).unwrap();
}

#[test]
fn directional_dir_mismatch_rejected() {
    // Prove Floor, verify HalfAway → reject.
    let mut rng = test_rng();
    let params = make_params(&mut rng);
    let s_in = Scale::from_pow2(10);
    let s_out = Scale::from_pow2(4);
    let n_vars = 2usize;
    let n_padded = 1usize << n_vars;

    let mut qx_codes = vec![0i128; n_padded];
    let mut qz_codes = vec![0i128; n_padded];
    let mut slack_lo = vec![0i128; n_padded];
    let (mut c1, mut c2) = (0i128, 0i128);
    let inputs = [97i128, -45, 12, -3];
    for (i, &qx) in inputs.iter().enumerate() {
        let (out, w) = Qf::new(qx, s_in)
            .rescale_dir(
                s_out,
                crate::quantization::quantized_scalar::RoundDir::Floor,
            )
            .unwrap();
        qx_codes[i] = qx;
        qz_codes[i] = out.code;
        slack_lo[i] = w.slack_lo;
        c1 = w.c1;
        c2 = w.c2;
    }

    let qx_padded = pad_to_max(&qx_codes, n_vars, params.max_num_vars);
    let qz_padded = pad_to_max(&qz_codes, n_vars, params.max_num_vars);
    let (qx_com, qx_state) =
        HyraxBn254::commit(&params.committer_key, &qx_padded, Some(&mut rng)).unwrap();
    let (qz_com, qz_state) =
        HyraxBn254::commit(&params.committer_key, &qz_padded, Some(&mut rng)).unwrap();
    let qx_aux: CommittedAux = (qx_padded.clone(), qx_state);
    let qz_aux: CommittedAux = (qz_padded.clone(), qz_state);

    let prove_desc = RescaleEventDesc {
        c1,
        c2,
        n_vars,
        dir: crate::quantization::quantized_scalar::RoundDir::Floor,
    };
    let mut sp = fresh_sponge(b"rescale-mismatch");
    let proof = prove_rescale_event(
        &prove_desc,
        &slack_lo,
        &qx_padded[..(1 << n_vars)],
        &qz_padded[..(1 << n_vars)],
        &qx_aux,
        &qx_com,
        &qz_aux,
        &qz_com,
        &params,
        &mut sp,
        &mut rng,
    )
    .unwrap();

    let verify_desc = RescaleEventDesc {
        c1,
        c2,
        n_vars,
        dir: crate::quantization::quantized_scalar::RoundDir::HalfAway,
    };
    let mut sv = fresh_sponge(b"rescale-mismatch");
    let res = verify_rescale_event(&proof, &verify_desc, &qx_com, &qz_com, &params, &mut sv);
    assert!(res.is_err(), "verifier must reject dir mismatch");
}
