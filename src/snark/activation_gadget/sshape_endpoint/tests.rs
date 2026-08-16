//! Unit tests for the endpoint gadget: completeness, tamper,
//! scale-precondition, n-padding, cross-endpoint/line replay, and
//! out-of-domain coverage.

use super::witness::scale_precondition_holds;
use super::{
    prove_sshape_upper_at_lower, prove_sshape_upper_at_upper, verify_sshape_upper_at_lower,
    verify_sshape_upper_at_upper, SshapeEndpointProof,
};
use crate::crown::network::ActivationKind;
use crate::quantization::scale::Scale;
use crate::snark::commitment::commit::{native_vector_n_vars, CommittedAux};
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;
use crate::snark_primitives::finite_field::signed_lift_to_fr;
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};
use ark_bn254::Fr;
use ark_crypto_primitives::sponge::merlin::Transcript;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_std::rand::RngCore;
use ark_std::test_rng;

fn test_params() -> SnarkParams {
    use crate::crown::network::{Layer, Network};
    use crate::crown::output_property::{Property, Side};
    use ndarray::{array, Array1, Array2};
    let net = Network::new(vec![Layer::linear(
        array![[1.0]] as Array2<f64>,
        array![0.0] as Array1<f64>,
    )
    .unwrap()])
    .unwrap();
    let prop = Property::new(Array2::eye(1), Array1::zeros(1), Side::Both).unwrap();
    let mut rng = test_rng();
    SnarkParams::setup(
        &net,
        &prop,
        14,
        crate::snark::preprocess::test_shared(19, 19, 19),
        &mut rng,
    )
    .unwrap()
}

fn commit_vec(
    params: &SnarkParams,
    codes: &[i128],
    n_vars: usize,
    rng: &mut impl RngCore,
) -> (CommittedAux, <HyraxBn254 as MlPcs>::Commitment) {
    let n_padded = 1usize << n_vars;
    let mut padded: Vec<Fr> = codes.iter().map(|&v| signed_lift_to_fr(v)).collect();
    padded.resize(n_padded, Fr::from(0u64));
    let (commit, state) = HyraxBn254::commit(&params.committer_key, &padded, Some(rng)).unwrap();
    ((padded, state), commit)
}

/// Build a proof for given preact codes (any signs) with d=0, b=2,
/// s_d=s_b=s_w=1.
fn build_proof(
    kind: ActivationKind,
    preact: &[i128],
) -> (
    SshapeEndpointProof,
    SnarkParams,
    usize,                             // n_real
    <HyraxBn254 as MlPcs>::Commitment, // preact_commit
    <HyraxBn254 as MlPcs>::Commitment, // d_commit
    <HyraxBn254 as MlPcs>::Commitment, // b_commit
    Scale,
) {
    let params = test_params();
    let mut rng = test_rng();
    let n = preact.len();
    let n_vars = native_vector_n_vars(n);
    let n_padded = 1usize << n_vars;
    let d_codes = vec![0i128; n_padded];
    let b_codes = vec![2i128; n_padded];
    let (preact_aux, preact_commit) = commit_vec(&params, preact, n_vars, &mut rng);
    let (d_aux, d_commit) = commit_vec(&params, &d_codes, n_vars, &mut rng);
    let (b_aux, b_commit) = commit_vec(&params, &b_codes, n_vars, &mut rng);
    let s = Scale::from_pow2(0);
    let mut sponge = <Transcript as CryptographicSponge>::new(&b"sshape-test".as_slice());
    let proof = prove_sshape_upper_at_lower(
        7,
        kind,
        preact,
        &preact_aux,
        &preact_commit,
        &d_aux,
        &d_commit,
        &b_aux,
        &b_commit,
        s,
        s,
        s,
        &params,
        &mut sponge,
        &mut rng,
    )
    .expect("prove succeeds");
    (proof, params, n, preact_commit, d_commit, b_commit, s)
}

fn verify_with(
    proof: &SshapeEndpointProof,
    params: &SnarkParams,
    kind: ActivationKind,
    layer_idx: usize,
    n_real: usize,
    preact_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s: Scale,
) -> Result<(), SnarkError> {
    let mut sponge = <Transcript as CryptographicSponge>::new(&b"sshape-test".as_slice());
    verify_sshape_upper_at_lower(
        proof,
        layer_idx,
        kind,
        n_real,
        preact_commit,
        d_commit,
        b_commit,
        s,
        s,
        s,
        params,
        &mut sponge,
    )
}

#[test]
fn completeness_sigmoid_all_positive() {
    let (proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Sigmoid, &[0, 0, 0, 0]);
    verify_with(
        &proof,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s,
    )
    .expect("verify accepts honest all-positive sigmoid proof");
}

#[test]
fn completeness_sigmoid_mixed_pos_neg() {
    let (proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    verify_with(
        &proof,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s,
    )
    .expect("verify accepts mixed-sign sigmoid proof");
}

#[test]
fn completeness_tanh_all_positive() {
    let (proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Tanh, &[0, 0, 0, 0]);
    verify_with(
        &proof,
        &params,
        ActivationKind::Tanh,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s,
    )
    .expect("verify accepts all-positive tanh proof");
}

#[test]
fn completeness_tanh_mixed_pos_neg() {
    let (proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Tanh, &[0, -1, 0, -2]);
    verify_with(
        &proof,
        &params,
        ActivationKind::Tanh,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s,
    )
    .expect("verify accepts mixed-sign tanh proof");
}

#[test]
fn completeness_sigmoid_at_minus_table_bound() {
    let (proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Sigmoid, &[0, -16, 0, 0]);
    verify_with(
        &proof,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s,
    )
    .expect("verify accepts l = -16 sigmoid proof");
}

#[test]
fn completeness_tanh_at_minus_natural_bound() {
    let (proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Tanh, &[0, -16, 0, 0]);
    verify_with(
        &proof,
        &params,
        ActivationKind::Tanh,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s,
    )
    .expect("verify accepts l = -16 tanh proof");
}

#[test]
fn tamper_sign_eval_rejects() {
    let (mut proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    proof.sign_eval += Fr::from(1u64);
    assert!(
        verify_with(
            &proof,
            &params,
            ActivationKind::Sigmoid,
            7,
            n_real,
            &preact_commit,
            &d_commit,
            &b_commit,
            s
        )
        .is_err(),
        "tampered sign_eval must reject"
    );
}

#[test]
fn tamper_abs_l_eval_rejects() {
    let (mut proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    proof.abs_l_eval += Fr::from(1u64);
    assert!(verify_with(
        &proof,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s
    )
    .is_err());
}

#[test]
fn tamper_sigma_upper_at_abs_rejects() {
    let (mut proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    proof.sigma_upper_at_abs_eval += Fr::from(1u64);
    assert!(verify_with(
        &proof,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s
    )
    .is_err());
}

#[test]
fn tamper_sigma_lower_at_abs_rejects() {
    let (mut proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    proof.sigma_lower_at_abs_eval += Fr::from(1u64);
    assert!(verify_with(
        &proof,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s
    )
    .is_err());
}

#[test]
fn tamper_slack_eval_rejects() {
    let (mut proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    proof.diff_eval += Fr::from(1u64);
    assert!(verify_with(
        &proof,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s
    )
    .is_err());
}

#[test]
fn tamper_epsilon_eval_rejects() {
    let (mut proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    proof.dx_step_1_rem_eval += Fr::from(1u64);
    assert!(verify_with(
        &proof,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s
    )
    .is_err());
}

#[test]
fn tamper_d_upper_eval_rejects() {
    let (mut proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    proof.d_line_eval += Fr::from(1u64);
    assert!(verify_with(
        &proof,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s
    )
    .is_err());
}

#[test]
fn tamper_b_upper_eval_rejects() {
    let (mut proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    proof.b_line_eval += Fr::from(1u64);
    assert!(verify_with(
        &proof,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s
    )
    .is_err());
}

#[test]
fn tamper_layer_idx_rejects() {
    let (proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    let r = verify_with(
        &proof,
        &params,
        ActivationKind::Sigmoid,
        99, // wrong layer idx
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s,
    );
    assert!(r.is_err(), "wrong layer_idx must reject; got {r:?}");
}

#[test]
fn scale_precondition_accepts_typical_pow2_scales() {
    // The budget is an explicit runtime parameter; a scale just past
    // it must reject, at representative budgets.
    for bits in [19usize, 21] {
        assert!(scale_precondition_holds(
            Scale::from_pow2(11),
            Scale::from_pow2(11),
            Scale::from_pow2(11),
            bits,
        ));
        assert!(scale_precondition_holds(
            Scale::from_pow2(bits as i32),
            Scale::from_pow2(11),
            Scale::from_pow2(11),
            bits,
        ));
        assert!(!scale_precondition_holds(
            Scale::from_pow2(bits as i32 + 1),
            Scale::from_pow2(11),
            Scale::from_pow2(11),
            bits,
        ));
    }
}

#[test]
fn tamper_r_final_rejects() {
    let (mut proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    if let Some(last) = proof.r_final.last_mut() {
        *last += Fr::from(1u64);
    }
    let r = verify_with(
        &proof,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s,
    );
    assert!(r.is_err(), "tampered r_final must reject; got {r:?}");
}

#[test]
fn tamper_envelope_witness_len_rejects() {
    let (mut proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    proof.envelope_witness_len = 2;
    let r = verify_with(
        &proof,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s,
    );
    assert!(
        r.is_err(),
        "tampered envelope_witness_len must reject; got {r:?}"
    );
}

#[test]
fn tamper_envelope_table_len_rejects() {
    let (mut proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    proof.envelope_table_len = 1024;
    let r = verify_with(
        &proof,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s,
    );
    assert!(
        r.is_err(),
        "tampered envelope_table_len must reject; got {r:?}"
    );
}

#[test]
fn tamper_envelope_lookup_top_rejects() {
    let (mut proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    proof.envelope_lookup_top[0] += Fr::from(1u64);
    let r = verify_with(
        &proof,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s,
    );
    assert!(
        r.is_err(),
        "tampered envelope_lookup_top must reject; got {r:?}"
    );
}

#[test]
fn tamper_envelope_table_top_rejects() {
    let (mut proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    proof.envelope_table_top[0] += Fr::from(1u64);
    let r = verify_with(
        &proof,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s,
    );
    assert!(
        r.is_err(),
        "tampered envelope_table_top must reject; got {r:?}"
    );
}

/// Isolates the top-fraction cancellation check: tampers a
/// denominator entry the per-side GKR verify wouldn't otherwise
/// flag.
#[test]
fn tamper_envelope_lookup_top_denom_rejects() {
    let (mut proof, params, n_real, preact_commit, d_commit, b_commit, s) =
        build_proof(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    proof.envelope_lookup_top[3] += Fr::from(1u64);
    let r = verify_with(
        &proof,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &preact_commit,
        &d_commit,
        &b_commit,
        s,
    );
    assert!(
        r.is_err(),
        "tampered envelope_lookup_top[3] must reject; got {r:?}"
    );
}

// U(u) tests cover the upper-line at the upper endpoint; the
// `endpoint_tag` binding rules out cross-endpoint replay.

fn build_proof_at_u(
    kind: ActivationKind,
    preact_u: &[i128],
) -> (
    SshapeEndpointProof,
    SnarkParams,
    usize,                             // n_real
    <HyraxBn254 as MlPcs>::Commitment, // preact_commit
    <HyraxBn254 as MlPcs>::Commitment, // d_commit
    <HyraxBn254 as MlPcs>::Commitment, // b_commit
    Scale,
) {
    let params = test_params();
    let mut rng = test_rng();
    let n = preact_u.len();
    let n_vars = native_vector_n_vars(n);
    let n_padded = 1usize << n_vars;
    let d_codes = vec![0i128; n_padded];
    let b_codes = vec![2i128; n_padded];
    let (preact_aux, preact_commit) = commit_vec(&params, preact_u, n_vars, &mut rng);
    let (d_aux, d_commit) = commit_vec(&params, &d_codes, n_vars, &mut rng);
    let (b_aux, b_commit) = commit_vec(&params, &b_codes, n_vars, &mut rng);
    let s = Scale::from_pow2(0);
    let mut sponge = <Transcript as CryptographicSponge>::new(&b"sshape-test".as_slice());
    let proof = prove_sshape_upper_at_upper(
        7,
        kind,
        preact_u,
        &preact_aux,
        &preact_commit,
        &d_aux,
        &d_commit,
        &b_aux,
        &b_commit,
        s,
        s,
        s,
        &params,
        &mut sponge,
        &mut rng,
    )
    .expect("U(u) prove succeeds");
    (proof, params, n, preact_commit, d_commit, b_commit, s)
}

fn verify_with_u(
    proof: &SshapeEndpointProof,
    params: &SnarkParams,
    kind: ActivationKind,
    layer_idx: usize,
    n_real: usize,
    preact_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s: Scale,
) -> Result<(), SnarkError> {
    let mut sponge = <Transcript as CryptographicSponge>::new(&b"sshape-test".as_slice());
    verify_sshape_upper_at_upper(
        proof,
        layer_idx,
        kind,
        n_real,
        preact_commit,
        d_commit,
        b_commit,
        s,
        s,
        s,
        params,
        &mut sponge,
    )
}

#[test]
fn u_completeness_sigmoid_all_positive() {
    let (p, params, n_real, pc, dc, bc, s) =
        build_proof_at_u(ActivationKind::Sigmoid, &[0, 0, 0, 0]);
    verify_with_u(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s,
    )
    .expect("U(u) verify accepts honest all-positive sigmoid proof");
}

#[test]
fn u_completeness_sigmoid_mixed_pos_neg() {
    let (p, params, n_real, pc, dc, bc, s) =
        build_proof_at_u(ActivationKind::Sigmoid, &[1, -1, 0, -2]);
    verify_with_u(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s,
    )
    .expect("U(u) verify accepts mixed-sign sigmoid proof");
}

#[test]
fn u_completeness_tanh_all_positive() {
    let (p, params, n_real, pc, dc, bc, s) = build_proof_at_u(ActivationKind::Tanh, &[0, 0, 0, 0]);
    verify_with_u(
        &p,
        &params,
        ActivationKind::Tanh,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s,
    )
    .expect("U(u) verify accepts honest all-positive tanh proof");
}

#[test]
fn u_completeness_tanh_mixed_pos_neg() {
    let (p, params, n_real, pc, dc, bc, s) =
        build_proof_at_u(ActivationKind::Tanh, &[2, -1, 1, -3]);
    verify_with_u(
        &p,
        &params,
        ActivationKind::Tanh,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s,
    )
    .expect("U(u) verify accepts mixed-sign tanh proof");
}

#[test]
fn u_tamper_sign_eval_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_at_u(ActivationKind::Sigmoid, &[1, -1, 0, -2]);
    p.sign_eval += Fr::from(1u64);
    assert!(verify_with_u(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn u_tamper_abs_l_eval_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_at_u(ActivationKind::Sigmoid, &[1, -1, 0, -2]);
    p.abs_l_eval += Fr::from(1u64);
    assert!(verify_with_u(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn u_tamper_sigma_upper_at_abs_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_at_u(ActivationKind::Sigmoid, &[1, -1, 0, -2]);
    p.sigma_upper_at_abs_eval += Fr::from(1u64);
    assert!(verify_with_u(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn u_tamper_sigma_lower_at_abs_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_at_u(ActivationKind::Sigmoid, &[1, -1, 0, -2]);
    p.sigma_lower_at_abs_eval += Fr::from(1u64);
    assert!(verify_with_u(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn u_tamper_slack_eval_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_at_u(ActivationKind::Sigmoid, &[1, -1, 0, -2]);
    p.diff_eval += Fr::from(1u64);
    assert!(verify_with_u(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn u_tamper_epsilon_eval_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_at_u(ActivationKind::Sigmoid, &[1, -1, 0, -2]);
    p.dx_step_1_rem_eval += Fr::from(1u64);
    assert!(verify_with_u(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn u_tamper_d_upper_eval_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_at_u(ActivationKind::Sigmoid, &[1, -1, 0, -2]);
    p.d_line_eval += Fr::from(1u64);
    assert!(verify_with_u(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn u_tamper_b_upper_eval_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_at_u(ActivationKind::Sigmoid, &[1, -1, 0, -2]);
    p.b_line_eval += Fr::from(1u64);
    assert!(verify_with_u(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn u_tamper_r_final_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_at_u(ActivationKind::Sigmoid, &[1, -1, 0, -2]);
    if let Some(last) = p.r_final.last_mut() {
        *last += Fr::from(1u64);
    }
    assert!(verify_with_u(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn u_tamper_envelope_witness_len_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_at_u(ActivationKind::Sigmoid, &[1, -1, 0, -2]);
    p.envelope_witness_len = 2;
    assert!(verify_with_u(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn u_tamper_envelope_table_len_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_at_u(ActivationKind::Sigmoid, &[1, -1, 0, -2]);
    p.envelope_table_len = 1024;
    assert!(verify_with_u(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn u_tamper_envelope_lookup_top_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_at_u(ActivationKind::Sigmoid, &[1, -1, 0, -2]);
    p.envelope_lookup_top[0] += Fr::from(1u64);
    assert!(verify_with_u(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn u_tamper_envelope_table_top_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_at_u(ActivationKind::Sigmoid, &[1, -1, 0, -2]);
    p.envelope_table_top[0] += Fr::from(1u64);
    assert!(verify_with_u(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn u_tamper_layer_idx_rejects() {
    let (p, params, n_real, pc, dc, bc, s) =
        build_proof_at_u(ActivationKind::Sigmoid, &[1, -1, 0, -2]);
    let r = verify_with_u(
        &p,
        &params,
        ActivationKind::Sigmoid,
        99,
        n_real,
        &pc,
        &dc,
        &bc,
        s,
    );
    assert!(r.is_err(), "wrong layer_idx must reject; got {r:?}");
}

/// Out-of-domain `|u| ≥ 128·s_w` must be rejected at the prover.
#[test]
fn u_out_of_domain_rejects() {
    let params = test_params();
    let mut rng = test_rng();
    let u = vec![0i128, 0, 0, 128];
    let n_vars = native_vector_n_vars(u.len());
    let n_padded = 1usize << n_vars;
    let d_codes = vec![0i128; n_padded];
    let b_codes = vec![2i128; n_padded];
    let (preact_aux, preact_commit) = commit_vec(&params, &u, n_vars, &mut rng);
    let (d_aux, d_commit) = commit_vec(&params, &d_codes, n_vars, &mut rng);
    let (b_aux, b_commit) = commit_vec(&params, &b_codes, n_vars, &mut rng);
    let s = Scale::from_pow2(0);
    let mut sponge = <Transcript as CryptographicSponge>::new(&b"sshape-test".as_slice());
    let r = prove_sshape_upper_at_upper(
        7,
        ActivationKind::Sigmoid,
        &u,
        &preact_aux,
        &preact_commit,
        &d_aux,
        &d_commit,
        &b_aux,
        &b_commit,
        s,
        s,
        s,
        &params,
        &mut sponge,
        &mut rng,
    );
    assert!(
        r.is_err(),
        "out-of-domain u must reject at prove; got {r:?}"
    );
}

/// Any `n ≥ 1` round-trips under the is_real mask.
#[test]
fn small_n_with_padding_now_accepted_via_is_real() {
    let params = test_params();
    let mut rng = test_rng();
    for n in [1usize, 2, 3, 8, 32] {
        let preact = vec![0i128; n];
        let n_vars = native_vector_n_vars(n);
        let n_padded = 1usize << n_vars;
        let d_codes = vec![0i128; n_padded];
        let b_codes = vec![2i128; n_padded];
        let (preact_aux, preact_commit) = commit_vec(&params, &preact, n_vars, &mut rng);
        let (d_aux, d_commit) = commit_vec(&params, &d_codes, n_vars, &mut rng);
        let (b_aux, b_commit) = commit_vec(&params, &b_codes, n_vars, &mut rng);
        let s = Scale::from_pow2(0);
        let mut sponge_p = <Transcript as CryptographicSponge>::new(&b"sshape-test".as_slice());
        let proof = prove_sshape_upper_at_lower(
            7,
            ActivationKind::Sigmoid,
            &preact,
            &preact_aux,
            &preact_commit,
            &d_aux,
            &d_commit,
            &b_aux,
            &b_commit,
            s,
            s,
            s,
            &params,
            &mut sponge_p,
            &mut rng,
        )
        .unwrap_or_else(|e| panic!("n={n} prove must succeed under is_real, got {e:?}"));

        let mut sponge_v = <Transcript as CryptographicSponge>::new(&b"sshape-test".as_slice());
        verify_sshape_upper_at_lower(
            &proof,
            7,
            ActivationKind::Sigmoid,
            n,
            &preact_commit,
            &d_commit,
            &b_commit,
            s,
            s,
            s,
            &params,
            &mut sponge_v,
        )
        .unwrap_or_else(|e| panic!("n={n} verify must accept under is_real, got {e:?}"));
    }
}

/// Shrinking `n_real` must be rejected by the FS binding.
#[test]
fn tamper_n_real_rejects() {
    let params = test_params();
    let mut rng = test_rng();
    let n: usize = 12;
    let preact: Vec<i128> = (0..n).map(|i| i as i128).collect();
    let n_vars = native_vector_n_vars(n);
    let n_padded = 1usize << n_vars;
    let d_codes = vec![0i128; n_padded];
    let b_codes = vec![2i128; n_padded];
    let (preact_aux, preact_commit) = commit_vec(&params, &preact, n_vars, &mut rng);
    let (d_aux, d_commit) = commit_vec(&params, &d_codes, n_vars, &mut rng);
    let (b_aux, b_commit) = commit_vec(&params, &b_codes, n_vars, &mut rng);
    let s = Scale::from_pow2(0);
    let mut sponge_p = <Transcript as CryptographicSponge>::new(&b"sshape-test".as_slice());
    let mut proof = prove_sshape_upper_at_lower(
        7,
        ActivationKind::Sigmoid,
        &preact,
        &preact_aux,
        &preact_commit,
        &d_aux,
        &d_commit,
        &b_aux,
        &b_commit,
        s,
        s,
        s,
        &params,
        &mut sponge_p,
        &mut rng,
    )
    .expect("honest prove must succeed");
    proof.n_real = n - 1;
    let mut sponge_v = <Transcript as CryptographicSponge>::new(&b"sshape-test".as_slice());
    let r = verify_sshape_upper_at_lower(
        &proof,
        7,
        ActivationKind::Sigmoid,
        n,
        &preact_commit,
        &d_commit,
        &b_commit,
        s,
        s,
        s,
        &params,
        &mut sponge_v,
    );
    assert!(r.is_err(), "tampered n_real must reject; got {r:?}");
}

#[test]
fn supported_n_values_succeed() {
    let preact = vec![0i128; 16];
    let params = test_params();
    let mut rng = test_rng();
    let n_vars = native_vector_n_vars(16);
    let n_padded = 1usize << n_vars;
    assert_eq!(n_padded, 16, "n=16 should not pad");
    let d_codes = vec![0i128; n_padded];
    let b_codes = vec![2i128; n_padded];
    let (preact_aux, preact_commit) = commit_vec(&params, &preact, n_vars, &mut rng);
    let (d_aux, d_commit) = commit_vec(&params, &d_codes, n_vars, &mut rng);
    let (b_aux, b_commit) = commit_vec(&params, &b_codes, n_vars, &mut rng);
    let s = Scale::from_pow2(0);
    let mut sponge = <Transcript as CryptographicSponge>::new(&b"sshape-test".as_slice());
    let proof = prove_sshape_upper_at_lower(
        7,
        ActivationKind::Sigmoid,
        &preact,
        &preact_aux,
        &preact_commit,
        &d_aux,
        &d_commit,
        &b_aux,
        &b_commit,
        s,
        s,
        s,
        &params,
        &mut sponge,
        &mut rng,
    )
    .expect("n=16 should prove");
    let mut verify_sponge = <Transcript as CryptographicSponge>::new(&b"sshape-test".as_slice());
    verify_sshape_upper_at_lower(
        &proof,
        7,
        ActivationKind::Sigmoid,
        16,
        &preact_commit,
        &d_commit,
        &b_commit,
        s,
        s,
        s,
        &params,
        &mut verify_sponge,
    )
    .expect("n=16 should verify");
}

/// A U(l) proof must not verify as U(u) (and vice versa).
#[test]
fn cross_endpoint_replay_rejects() {
    let (proof_l, params, n_real, pc, dc, bc, s) =
        build_proof(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    let r = verify_with_u(
        &proof_l,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s,
    );
    assert!(
        r.is_err(),
        "U(l) proof replayed as U(u) must reject; got {r:?}"
    );

    let (proof_u, params2, n_real2, pc2, dc2, bc2, s2) =
        build_proof_at_u(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    let r2 = verify_with(
        &proof_u,
        &params2,
        ActivationKind::Sigmoid,
        7,
        n_real2,
        &pc2,
        &dc2,
        &bc2,
        s2,
    );
    assert!(
        r2.is_err(),
        "U(u) proof replayed as U(l) must reject; got {r2:?}"
    );
}

// Lower-line endpoint tests: prove `σ_lower(x) ≥ L(x)` (sign-flipped
// slack identity).

use super::{
    prove_sshape_lower_at_lower, prove_sshape_lower_at_upper, verify_sshape_lower_at_lower,
    verify_sshape_lower_at_upper,
};

/// Build a `σ_lower ≥ L` proof at the lower endpoint with
/// `d_lower = 0, b_lower = -2` so `L(x) = -2`.
fn build_proof_lower_at_l(
    kind: ActivationKind,
    preact_l: &[i128],
) -> (
    SshapeEndpointProof,
    SnarkParams,
    usize,                             // n_real
    <HyraxBn254 as MlPcs>::Commitment, // preact_commit
    <HyraxBn254 as MlPcs>::Commitment, // d_commit
    <HyraxBn254 as MlPcs>::Commitment, // b_commit
    Scale,
) {
    let params = test_params();
    let mut rng = test_rng();
    let n = preact_l.len();
    let n_vars = native_vector_n_vars(n);
    let n_padded = 1usize << n_vars;
    let d_codes = vec![0i128; n_padded];
    let b_codes = vec![-2i128; n_padded];
    let (preact_aux, preact_commit) = commit_vec(&params, preact_l, n_vars, &mut rng);
    let (d_aux, d_commit) = commit_vec(&params, &d_codes, n_vars, &mut rng);
    let (b_aux, b_commit) = commit_vec(&params, &b_codes, n_vars, &mut rng);
    let s = Scale::from_pow2(0);
    let mut sponge = <Transcript as CryptographicSponge>::new(&b"sshape-test".as_slice());
    let proof = prove_sshape_lower_at_lower(
        7,
        kind,
        preact_l,
        &preact_aux,
        &preact_commit,
        &d_aux,
        &d_commit,
        &b_aux,
        &b_commit,
        s,
        s,
        s,
        &params,
        &mut sponge,
        &mut rng,
    )
    .expect("lower-l prove succeeds");
    (proof, params, n, preact_commit, d_commit, b_commit, s)
}

fn build_proof_lower_at_u(
    kind: ActivationKind,
    preact_u: &[i128],
) -> (
    SshapeEndpointProof,
    SnarkParams,
    usize,                             // n_real
    <HyraxBn254 as MlPcs>::Commitment, // preact_commit
    <HyraxBn254 as MlPcs>::Commitment, // d_commit
    <HyraxBn254 as MlPcs>::Commitment, // b_commit
    Scale,
) {
    let params = test_params();
    let mut rng = test_rng();
    let n = preact_u.len();
    let n_vars = native_vector_n_vars(n);
    let n_padded = 1usize << n_vars;
    let d_codes = vec![0i128; n_padded];
    let b_codes = vec![-2i128; n_padded];
    let (preact_aux, preact_commit) = commit_vec(&params, preact_u, n_vars, &mut rng);
    let (d_aux, d_commit) = commit_vec(&params, &d_codes, n_vars, &mut rng);
    let (b_aux, b_commit) = commit_vec(&params, &b_codes, n_vars, &mut rng);
    let s = Scale::from_pow2(0);
    let mut sponge = <Transcript as CryptographicSponge>::new(&b"sshape-test".as_slice());
    let proof = prove_sshape_lower_at_upper(
        7,
        kind,
        preact_u,
        &preact_aux,
        &preact_commit,
        &d_aux,
        &d_commit,
        &b_aux,
        &b_commit,
        s,
        s,
        s,
        &params,
        &mut sponge,
        &mut rng,
    )
    .expect("lower-u prove succeeds");
    (proof, params, n, preact_commit, d_commit, b_commit, s)
}

fn verify_with_lower_l(
    proof: &SshapeEndpointProof,
    params: &SnarkParams,
    kind: ActivationKind,
    layer_idx: usize,
    n_real: usize,
    preact_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s: Scale,
) -> Result<(), SnarkError> {
    let mut sponge = <Transcript as CryptographicSponge>::new(&b"sshape-test".as_slice());
    verify_sshape_lower_at_lower(
        proof,
        layer_idx,
        kind,
        n_real,
        preact_commit,
        d_commit,
        b_commit,
        s,
        s,
        s,
        params,
        &mut sponge,
    )
}

fn verify_with_lower_u(
    proof: &SshapeEndpointProof,
    params: &SnarkParams,
    kind: ActivationKind,
    layer_idx: usize,
    n_real: usize,
    preact_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s: Scale,
) -> Result<(), SnarkError> {
    let mut sponge = <Transcript as CryptographicSponge>::new(&b"sshape-test".as_slice());
    verify_sshape_lower_at_upper(
        proof,
        layer_idx,
        kind,
        n_real,
        preact_commit,
        d_commit,
        b_commit,
        s,
        s,
        s,
        params,
        &mut sponge,
    )
}

#[test]
fn lower_l_completeness_sigmoid_mixed() {
    let (p, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_l(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    verify_with_lower_l(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s,
    )
    .expect("L(l) sigmoid mixed should verify");
}

#[test]
fn lower_l_completeness_tanh_mixed() {
    let (p, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_l(ActivationKind::Tanh, &[1, -1, 0, -2]);
    verify_with_lower_l(
        &p,
        &params,
        ActivationKind::Tanh,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s,
    )
    .expect("L(l) tanh mixed should verify");
}

#[test]
fn lower_u_completeness_sigmoid_mixed() {
    let (p, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_u(ActivationKind::Sigmoid, &[1, -1, 0, -2]);
    verify_with_lower_u(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s,
    )
    .expect("L(u) sigmoid mixed should verify");
}

#[test]
fn lower_u_completeness_tanh_mixed() {
    let (p, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_u(ActivationKind::Tanh, &[2, -1, 1, -3]);
    verify_with_lower_u(
        &p,
        &params,
        ActivationKind::Tanh,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s,
    )
    .expect("L(u) tanh mixed should verify");
}

#[test]
fn lower_l_completeness_all_positive() {
    let (p, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_l(ActivationKind::Sigmoid, &[0, 1, 2, 3]);
    verify_with_lower_l(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s,
    )
    .expect("L(l) all-positive should verify");
}

#[test]
fn lower_l_tamper_sign_eval_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_l(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    p.sign_eval += Fr::from(1u64);
    assert!(verify_with_lower_l(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn lower_l_tamper_abs_l_eval_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_l(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    p.abs_l_eval += Fr::from(1u64);
    assert!(verify_with_lower_l(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn lower_l_tamper_sigma_upper_at_abs_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_l(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    p.sigma_upper_at_abs_eval += Fr::from(1u64);
    assert!(verify_with_lower_l(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn lower_l_tamper_sigma_lower_at_abs_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_l(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    p.sigma_lower_at_abs_eval += Fr::from(1u64);
    assert!(verify_with_lower_l(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn lower_l_tamper_slack_eval_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_l(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    p.diff_eval += Fr::from(1u64);
    assert!(verify_with_lower_l(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn lower_l_tamper_epsilon_eval_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_l(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    p.dx_step_1_rem_eval += Fr::from(1u64);
    assert!(verify_with_lower_l(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn lower_l_tamper_d_line_eval_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_l(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    p.d_line_eval += Fr::from(1u64);
    assert!(verify_with_lower_l(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn lower_l_tamper_b_line_eval_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_l(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    p.b_line_eval += Fr::from(1u64);
    assert!(verify_with_lower_l(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn lower_l_tamper_r_final_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_l(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    if let Some(last) = p.r_final.last_mut() {
        *last += Fr::from(1u64);
    }
    assert!(verify_with_lower_l(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn lower_l_tamper_envelope_witness_len_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_l(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    p.envelope_witness_len = 2;
    assert!(verify_with_lower_l(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn lower_l_tamper_envelope_lookup_top_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_l(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    p.envelope_lookup_top[0] += Fr::from(1u64);
    assert!(verify_with_lower_l(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn lower_l_tamper_envelope_table_top_rejects() {
    let (mut p, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_l(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    p.envelope_table_top[0] += Fr::from(1u64);
    assert!(verify_with_lower_l(
        &p,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s
    )
    .is_err());
}

#[test]
fn lower_l_tamper_layer_idx_rejects() {
    let (p, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_l(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    let r = verify_with_lower_l(
        &p,
        &params,
        ActivationKind::Sigmoid,
        99,
        n_real,
        &pc,
        &dc,
        &bc,
        s,
    );
    assert!(r.is_err(), "wrong layer_idx must reject; got {r:?}");
}

#[test]
fn lower_l_out_of_domain_rejects() {
    let params = test_params();
    let mut rng = test_rng();
    let l = vec![0i128, 0, 0, 128];
    let n_vars = native_vector_n_vars(l.len());
    let n_padded = 1usize << n_vars;
    let d_codes = vec![0i128; n_padded];
    let b_codes = vec![-2i128; n_padded];
    let (preact_aux, preact_commit) = commit_vec(&params, &l, n_vars, &mut rng);
    let (d_aux, d_commit) = commit_vec(&params, &d_codes, n_vars, &mut rng);
    let (b_aux, b_commit) = commit_vec(&params, &b_codes, n_vars, &mut rng);
    let s = Scale::from_pow2(0);
    let mut sponge = <Transcript as CryptographicSponge>::new(&b"sshape-test".as_slice());
    let r = prove_sshape_lower_at_lower(
        7,
        ActivationKind::Sigmoid,
        &l,
        &preact_aux,
        &preact_commit,
        &d_aux,
        &d_commit,
        &b_aux,
        &b_commit,
        s,
        s,
        s,
        &params,
        &mut sponge,
        &mut rng,
    );
    assert!(
        r.is_err(),
        "out-of-domain l must reject at prove; got {r:?}"
    );
}

/// A lower-line proof must not verify under the upper-line wrapper.
#[test]
fn cross_line_replay_rejects() {
    let (proof_lower, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_l(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    let r = verify_with(
        &proof_lower,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s,
    );
    assert!(
        r.is_err(),
        "lower-line proof verified as upper-line must reject; got {r:?}"
    );

    let (proof_upper, params2, n_real2, pc2, dc2, bc2, s2) =
        build_proof(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    let r2 = verify_with_lower_l(
        &proof_upper,
        &params2,
        ActivationKind::Sigmoid,
        7,
        n_real2,
        &pc2,
        &dc2,
        &bc2,
        s2,
    );
    assert!(
        r2.is_err(),
        "upper-line proof verified as lower-line must reject; got {r2:?}"
    );
}

/// L(l) proof must not verify as L(u) and vice versa.
#[test]
fn lower_cross_endpoint_replay_rejects() {
    let (proof_lower_l, params, n_real, pc, dc, bc, s) =
        build_proof_lower_at_l(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    let r = verify_with_lower_u(
        &proof_lower_l,
        &params,
        ActivationKind::Sigmoid,
        7,
        n_real,
        &pc,
        &dc,
        &bc,
        s,
    );
    assert!(
        r.is_err(),
        "L(l) proof replayed as L(u) must reject; got {r:?}"
    );

    let (proof_lower_u, params2, n_real2, pc2, dc2, bc2, s2) =
        build_proof_lower_at_u(ActivationKind::Sigmoid, &[0, -1, 0, -2]);
    let r2 = verify_with_lower_l(
        &proof_lower_u,
        &params2,
        ActivationKind::Sigmoid,
        7,
        n_real2,
        &pc2,
        &dc2,
        &bc2,
        s2,
    );
    assert!(
        r2.is_err(),
        "L(u) proof replayed as L(l) must reject; got {r2:?}"
    );
}
