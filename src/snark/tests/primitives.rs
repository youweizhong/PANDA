//! Sub-protocol primitives tested in isolation:
//! * `linear_backward` — per-layer matmul + matvec sumcheck +
//!   bound-to-commit gadget.
//!
//! Per-tensor commits are made at native sizes (`n_vars_from_logs` per
//! shape), so the per-tensor sizing refactor is covered too.

use ark_bn254::Fr;
use ark_std::{test_rng, UniformRand};

use super::fixtures::fresh_sponge;
use crate::snark::backward_pass::linear_step::{
    prove_linear_backward, verify_linear_backward, LinearBackwardCommitContext,
    LinearBackwardVerifyContext,
};
use crate::snark::commitment::commit::{n_vars_from_logs, CommittedAux};
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

#[allow(clippy::type_complexity)]
fn primitive_test_commit_at_sizes(
    max_num_vars: usize,
    sized: &[&[Fr]],
) -> (
    <HyraxBn254 as MlPcs>::CommitterKey,
    <HyraxBn254 as MlPcs>::VerifierKey,
    Vec<(<HyraxBn254 as MlPcs>::Commitment, CommittedAux)>,
) {
    let mut rng = test_rng();
    let (ck, vk) = HyraxBn254::setup(max_num_vars, &mut rng).unwrap();
    let mut auxes = Vec::with_capacity(sized.len());
    for evals in sized {
        let (com, st) = HyraxBn254::commit(&ck, evals, Some(&mut rng)).unwrap();
        auxes.push((com, (evals.to_vec(), st)));
    }
    (ck, vk, auxes)
}

fn pad(small: &[Fr], target_len: usize) -> Vec<Fr> {
    let mut out = vec![Fr::from(0u64); target_len];
    out[..small.len()].copy_from_slice(small);
    out
}

/// Build the random `(a, w, b)` inputs and the canonical `(a_w, a_b)`
/// outputs for the per-layer linear backward gadget.
fn build_linear_backward_witness(
    lns: usize,
    lni: usize,
    lno: usize,
) -> (Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>) {
    let mut rng = test_rng();
    let n_spec = 1 << lns;
    let n_inner = 1 << lni;
    let n_out = 1 << lno;
    let a: Vec<Fr> = (0..n_spec * n_inner).map(|_| Fr::rand(&mut rng)).collect();
    let w: Vec<Fr> = (0..n_inner * n_out).map(|_| Fr::rand(&mut rng)).collect();
    let b: Vec<Fr> = (0..n_inner).map(|_| Fr::rand(&mut rng)).collect();
    let mut a_w = vec![Fr::from(0u64); n_spec * n_out];
    for i in 0..n_spec {
        for k in 0..n_out {
            let mut acc = Fr::from(0u64);
            for j in 0..n_inner {
                acc += a[i * n_inner + j] * w[j * n_out + k];
            }
            a_w[i * n_out + k] = acc;
        }
    }
    let mut a_b = vec![Fr::from(0u64); n_spec];
    for i in 0..n_spec {
        let mut acc = Fr::from(0u64);
        for j in 0..n_inner {
            acc += a[i * n_inner + j] * b[j];
        }
        a_b[i] = acc;
    }
    (a, w, b, a_w, a_b)
}

/// Pad each input tensor to its native commit size and return per-
/// tensor sizes alongside the padded buffers and the sup-size used to
/// set up the Hyrax key.
#[allow(clippy::type_complexity)]
fn pad_to_native_sizes(
    a: &[Fr],
    w: &[Fr],
    b: &[Fr],
    a_w: &[Fr],
    a_b: &[Fr],
    lns: usize,
    lni: usize,
    lno: usize,
) -> (Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>, usize) {
    let w_nv = n_vars_from_logs(&[lni, lno]);
    let b_nv = n_vars_from_logs(&[lni]);
    let a_old_nv = n_vars_from_logs(&[lns, lni]);
    let a_w_nv = n_vars_from_logs(&[lns, lno]);
    let a_b_nv = n_vars_from_logs(&[lns]);
    let max_nv = w_nv.max(b_nv).max(a_old_nv).max(a_w_nv).max(a_b_nv);
    (
        pad(w, 1 << w_nv),
        pad(b, 1 << b_nv),
        pad(a, 1 << a_old_nv),
        pad(a_w, 1 << a_w_nv),
        pad(a_b, 1 << a_b_nv),
        max_nv,
    )
}

#[test]
fn linear_backward_primitive_roundtrip() {
    let (lns, lni, lno) = (1, 2, 1);
    let (a, w, b, a_w, a_b) = build_linear_backward_witness(lns, lni, lno);
    let (w_padded, b_padded, a_old_padded, a_w_padded, a_b_padded, max_nv) =
        pad_to_native_sizes(&a, &w, &b, &a_w, &a_b, lns, lni, lno);
    let (ck, vk, mut auxes) = primitive_test_commit_at_sizes(
        max_nv,
        &[
            &w_padded,
            &b_padded,
            &a_old_padded,
            &a_w_padded,
            &a_b_padded,
        ],
    );
    let (a_b_com, a_b_aux) = auxes.pop().unwrap();
    let (a_w_com, a_w_aux) = auxes.pop().unwrap();
    let (a_old_com, a_old_aux) = auxes.pop().unwrap();
    let (b_com, b_aux) = auxes.pop().unwrap();
    let (w_com, w_aux) = auxes.pop().unwrap();
    let commit_ctx = LinearBackwardCommitContext {
        ck: &ck,
        w_aux: &w_aux,
        w_commitment: &w_com,
        b_aux: &b_aux,
        b_commitment: &b_com,
        a_old_aux: &a_old_aux,
        a_old_commitment: &a_old_com,
        a_w_aux: &a_w_aux,
        a_w_commitment: &a_w_com,
        a_b_aux: &a_b_aux,
        a_b_commitment: &a_b_com,
        max_num_vars: max_nv,
    };

    let mut prover_sponge = fresh_sponge();
    let mut rng_box = test_rng();
    let proof = prove_linear_backward(
        &a,
        (lns, lni),
        &w,
        (lni, lno),
        &b,
        &a_w,
        &a_b,
        &commit_ctx,
        &mut prover_sponge,
        &mut rng_box,
    )
    .unwrap();

    let verify_ctx = LinearBackwardVerifyContext {
        vk: &vk,
        w_commitment: &w_com,
        b_commitment: &b_com,
        a_old_commitment: &a_old_com,
        a_w_commitment: &a_w_com,
        a_b_commitment: &a_b_com,
        max_num_vars: max_nv,
    };
    let mut verifier_sponge = fresh_sponge();
    let openings = verify_linear_backward(
        &proof,
        (lns, lni),
        (lni, lno),
        &verify_ctx,
        &mut verifier_sponge,
    )
    .unwrap();
    assert_eq!(openings.a_w.1, proof.matmul_claim);
    assert_eq!(openings.a_b.1, proof.matvec_claim);
}

#[test]
fn linear_backward_primitive_rejects_tampered_a_w_claim() {
    // Tamper one a_w cell before prove; verifier must reject.
    let (lns, lni, lno) = (1, 2, 1);
    let (a, w, b, mut a_w, a_b) = build_linear_backward_witness(lns, lni, lno);
    a_w[0] += Fr::from(1u64);
    let (w_padded, b_padded, a_old_padded, a_w_padded, a_b_padded, max_nv) =
        pad_to_native_sizes(&a, &w, &b, &a_w, &a_b, lns, lni, lno);
    let (ck, vk, mut auxes) = primitive_test_commit_at_sizes(
        max_nv,
        &[
            &w_padded,
            &b_padded,
            &a_old_padded,
            &a_w_padded,
            &a_b_padded,
        ],
    );
    let (a_b_com, a_b_aux) = auxes.pop().unwrap();
    let (a_w_com, a_w_aux) = auxes.pop().unwrap();
    let (a_old_com, a_old_aux) = auxes.pop().unwrap();
    let (b_com, b_aux) = auxes.pop().unwrap();
    let (w_com, w_aux) = auxes.pop().unwrap();
    let commit_ctx = LinearBackwardCommitContext {
        ck: &ck,
        w_aux: &w_aux,
        w_commitment: &w_com,
        b_aux: &b_aux,
        b_commitment: &b_com,
        a_old_aux: &a_old_aux,
        a_old_commitment: &a_old_com,
        a_w_aux: &a_w_aux,
        a_w_commitment: &a_w_com,
        a_b_aux: &a_b_aux,
        a_b_commitment: &a_b_com,
        max_num_vars: max_nv,
    };
    let mut prover_sponge = fresh_sponge();
    let mut rng_box = test_rng();
    let proof = prove_linear_backward(
        &a,
        (lns, lni),
        &w,
        (lni, lno),
        &b,
        &a_w,
        &a_b,
        &commit_ctx,
        &mut prover_sponge,
        &mut rng_box,
    )
    .unwrap();
    let verify_ctx = LinearBackwardVerifyContext {
        vk: &vk,
        w_commitment: &w_com,
        b_commitment: &b_com,
        a_old_commitment: &a_old_com,
        a_w_commitment: &a_w_com,
        a_b_commitment: &a_b_com,
        max_num_vars: max_nv,
    };
    let mut verifier_sponge = fresh_sponge();
    let verdict = verify_linear_backward(
        &proof,
        (lns, lni),
        (lni, lno),
        &verify_ctx,
        &mut verifier_sponge,
    );
    assert!(verdict.is_err(), "verifier should reject tampered A_W");
}

