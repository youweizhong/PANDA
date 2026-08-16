//! End-to-end tamper tests on `prove_final_pass`'s output.
//!
//! Each test mutates exactly one field on a freshly-produced
//! `SnarkProof` and asserts the verifier rejects. Where a specific
//! `SnarkError` variant is the natural rejection (e.g. `RangeRejected`
//! for tampered `range.alpha`), the test pins it via `matches!`; for
//! cross-cutting tampers a plain `is_err()` is used. This is the
//! soundness backbone — every committed value and every claimed
//! sumcheck claim should have at least one tamper test here.

use ark_bn254::Fr;

use super::fixtures::{
    expect_reject_after_tamper, prove_small_relu, prove_small_sigmoid, prove_small_tanh,
};
use crate::snark::errors::SnarkError;

#[test]
fn tampered_alpha_breaks_verifier() {
    // Tamper first per-tensor range LogUp `α`; transcript mismatch on
    // the first per-tensor LogUp must reject.
    let mut p = prove_small_relu();
    assert!(!p.proof.tensor_range_proofs.is_empty());
    p.proof.tensor_range_proofs[0].alpha += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "tampered first per-tensor range α");
}

#[test]
fn tampered_output_bound_claimed_eval_rejects() {
    // Tamper `output_bound_upper.claimed_eval` (the value the verifier
    // uses to anchor the slack identity); LogUp/sumcheck identity check
    // must reject.
    let mut p = prove_small_relu();
    if let Some(ob) = p.proof.output_bound_upper.as_mut() {
        ob.claimed_eval += Fr::from(1u64);
    }
    let _ = expect_reject_after_tamper(&p, "tampered output_bound_upper.claimed_eval");
}

#[test]
fn full_pass_chain_rejects_tampered_activation_claim() {
    // Tamper `delta_b_claim` on the first activation step.
    let mut p = prove_small_relu();
    if let Some(chain) = p.proof.activation_backward_lower.as_mut() {
        assert!(!chain.is_empty(), "small_relu has 1 activation step");
        chain[0].delta_b_claim += Fr::from(1u64);
    }
    let err = expect_reject_after_tamper(&p, "tampered activation delta_b_claim");
    assert!(matches!(err, SnarkError::ActivationLayerRejected { .. }));
}

#[test]
fn full_pass_chain_rejects_tampered_relu_lookup() {
    // Tamper ReLU lookup `a_pos_eval_at_r`.
    let mut p = prove_small_relu();
    if let Some(relu) = p.proof.relu_lower_activation.as_mut() {
        assert!(!relu.is_empty());
        relu[0].a_pos_eval_at_r += Fr::from(1u64);
    }
    let _ = expect_reject_after_tamper(&p, "tampered ReLU a_pos_eval_at_r");
}

#[test]
fn full_pass_chain_rejects_tampered_rescale_qz_eval() {
    // Tamper rescale `qz_eval`.
    let mut p = prove_small_relu();
    if let Some(rescale) = p.proof.rescale_lower.as_mut() {
        assert!(!rescale.is_empty());
        rescale[0].qz_eval += Fr::from(1u64);
    }
    let _ = expect_reject_after_tamper(&p, "tampered rescale qz_eval");
}

#[test]
fn full_pass_chain_rejects_tampered_rescale_slack_eval() {
    // Tamper rescale `slack_lo_eval`.
    let mut p = prove_small_relu();
    if let Some(rescale) = p.proof.rescale_lower.as_mut() {
        rescale[0].slack_lo_eval += Fr::from(1u64);
    }
    let _ = expect_reject_after_tamper(&p, "tampered rescale slack_lo_eval");
}

#[test]
fn full_pass_chain_rejects_tampered_concretize_claim() {
    // Tamper concretize `target_doubled_claim`.
    let mut p = prove_small_relu();
    if let Some(c) = p.proof.concretize_lower.as_mut() {
        c.target_doubled_claim += Fr::from(1u64);
    }
    let err = expect_reject_after_tamper(&p, "tampered concretize target_doubled_claim");
    assert!(matches!(err, SnarkError::ConcretizeRejected));
}

#[test]
fn full_pass_chain_rejects_tampered_a_old_eval() {
    // Tamper linear `a_old_eval`.
    let mut p = prove_small_relu();
    if let Some(chain) = p.proof.linear_backward_lower.as_mut() {
        chain[0].proof.a_old_eval += Fr::from(1u64);
    }
    let _ = expect_reject_after_tamper(&p, "tampered linear a_old_eval");
}

#[test]
fn full_pass_chain_rejects_tampered_a_w_eval() {
    // Tamper linear `matmul_claim`.
    let mut p = prove_small_relu();
    if let Some(chain) = p.proof.linear_backward_lower.as_mut() {
        chain[0].proof.matmul_claim += Fr::from(1u64);
    }
    let _ = expect_reject_after_tamper(&p, "tampered linear matmul_claim");
}

#[test]
fn full_pass_chain_rejects_tampered_w_eval() {
    // Tamper linear `w_eval`.
    let mut p = prove_small_relu();
    if let Some(chain) = p.proof.linear_backward_lower.as_mut() {
        chain[0].proof.w_eval += Fr::from(1u64);
    }
    let _ = expect_reject_after_tamper(&p, "tampered linear w_eval");
}

#[test]
fn full_pass_chain_rejects_tampered_per_layer_claim() {
    // Same field as `tampered_a_w_eval` but pins
    // `LinearLayerRejected` (the per-layer subprotocol path).
    let mut p = prove_small_relu();
    if let Some(chain) = p.proof.linear_backward_lower.as_mut() {
        chain[0].proof.matmul_claim += Fr::from(1u64);
    }
    let err = expect_reject_after_tamper(&p, "tampered linear matmul_claim (variant)");
    assert!(matches!(err, SnarkError::LinearLayerRejected { .. }));
}

#[test]
fn full_pass_chain_rejects_tampered_chain_init_a() {
    // Tamper chain_init `chain_a_eval`.
    let mut p = prove_small_relu();
    if let Some(ci) = p.proof.chain_init_lower.as_mut() {
        ci.chain_a_eval += Fr::from(1u64);
    }
    let _ = expect_reject_after_tamper(&p, "tampered chain_init chain_a_eval");
}

#[test]
fn full_pass_chain_rejects_tampered_chain_init_b() {
    // Tamper chain_init `spec_d_eval`.
    let mut p = prove_small_relu();
    if let Some(ci) = p.proof.chain_init_lower.as_mut() {
        ci.spec_d_eval += Fr::from(1u64);
    }
    let _ = expect_reject_after_tamper(&p, "tampered chain_init spec_d_eval");
}

#[test]
fn full_pass_chain_rejects_tampered_b_acc_step() {
    // Tamper b_acc_step `b_new_eval`.
    let mut p = prove_small_relu();
    if let Some(steps) = p.proof.b_acc_step_lower.as_mut() {
        assert!(!steps.is_empty(), "expected b_acc step proofs");
        steps[0].b_new_eval += Fr::from(1u64);
    }
    let _ = expect_reject_after_tamper(&p, "tampered b_acc_step b_new_eval");
}

#[test]
fn full_pass_chain_rejects_tampered_activation_matrix_claim() {
    // Tamper activation_matrix `a_d_claim`.
    let mut p = prove_small_relu();
    if let Some(steps) = p.proof.activation_matrix_lower.as_mut() {
        assert!(!steps.is_empty());
        steps[0].a_d_claim += Fr::from(1u64);
    }
    let _ = expect_reject_after_tamper(&p, "tampered activation_matrix a_d_claim");
}

// LogUp table-side binding: the verifier independently computes
// `T_canonical_mle(bottom_point) - α` and checks against
// `proof.X.table_proof.bottom_denom`. Tampering causes a reject —
// either at the GKR-internal sumcheck or at the canonical-table
// binding. A more targeted forged-table negative-case lives in
// `snark::commitment::table_mle::tests`.

#[test]
fn full_pass_chain_rejects_tampered_global_range_table_bottom_denom() {
    // Tamper first per-tensor range proof's `bottom_denom`.
    let mut p = prove_small_relu();
    assert!(!p.proof.tensor_range_proofs.is_empty());
    p.proof.tensor_range_proofs[0].table_proof.bottom_denom += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "tampered first per-tensor table_proof.bottom_denom");
}

#[test]
fn full_pass_chain_rejects_tampered_output_bound_table_bottom_denom() {
    // Tamper output_bound `table_proof.bottom_denom`.
    let mut p = prove_small_relu();
    if let Some(ob) = p.proof.output_bound_lower.as_mut() {
        ob.table_proof.bottom_denom += Fr::from(1u64);
    }
    let _ = expect_reject_after_tamper(&p, "tampered output_bound table_proof.bottom_denom");
}

#[test]
fn full_pass_chain_rejects_tampered_rescale_table_bottom_denom() {
    // Tamper rescale `table_proof.bottom_denom`.
    let mut p = prove_small_relu();
    if let Some(rescale) = p.proof.rescale_lower.as_mut() {
        assert!(!rescale.is_empty());
        rescale[0].table_proof.bottom_denom += Fr::from(1u64);
    }
    let _ = expect_reject_after_tamper(&p, "tampered rescale table_proof.bottom_denom");
}

#[test]
fn full_pass_chain_rejects_tampered_layer_scales_weight_e_open() {
    // Tamper `layer_scale_opens.weight[0].e_eval`; Hyrax `verify_at`
    // must reject because the eval no longer matches the commit.
    let mut p = prove_small_relu();
    if let Some(open) = p.proof.layer_scale_opens.weight[0].as_mut() {
        open.e_eval += Fr::from(1u64);
    }
    let _ = expect_reject_after_tamper(&p, "tampered weight[0].e_eval");
}

#[test]
fn full_pass_chain_rejects_layer_scales_e_out_of_range() {
    // Tamper `layer_scale_opens.weight[0].e_eval` to a value beyond
    // `MAX_SCALE_E_ABS` (= 32); the synthetic `LayerScalesCommit`
    // range guard in `check_layer_scales_shape` must reject.
    use crate::snark_primitives::finite_field::signed_lift_to_fr;
    use ark_bn254::Fr as ArkFr;
    let mut p = prove_small_relu();
    if let Some(open) = p.proof.layer_scale_opens.weight[0].as_mut() {
        open.e_eval = signed_lift_to_fr(64);
        let _ = ArkFr::from(0u64);
    }
    let _ = expect_reject_after_tamper(&p, "weight[0].e_eval = 64 out of [-32, 32]");
}

#[test]
fn full_pass_chain_rejects_tampered_output_bound_lower_claimed_eval() {
    // Tamper `output_bound_lower.claimed_eval`; slack identity at the
    // FS-derived random point breaks.
    let mut p = prove_small_relu();
    if let Some(ob) = p.proof.output_bound_lower.as_mut() {
        ob.claimed_eval += Fr::from(1u64);
    }
    let _ = expect_reject_after_tamper(&p, "tampered output_bound_lower.claimed_eval");
}

#[test]
fn full_pass_chain_rejects_tampered_hidden_chain_init_a_open_proof() {
    // Swap hidden_pass `chain_init_lower.chain_a_open` with the upper
    // pass's open; the verifier's re-derived `r_a` diverges from the
    // swapped proof's `r_a` and Hyrax verify rejects.
    let mut p = prove_small_relu();
    if let Some(hp) = p.proof.hidden_passes.first_mut() {
        hp.chain_init_lower.chain_a_open = hp.chain_init_upper.chain_a_open.clone();
    }
    let _ = expect_reject_after_tamper(&p, "swapped hidden_pass chain_init lower a_open");
}

#[test]
fn full_pass_chain_rejects_tampered_hidden_chain_init_a() {
    // Tamper hidden_pass `chain_init_lower.chain_a_eval`; the identity
    // MLE check (chain_a bound to canonical identity) rejects.
    let mut p = prove_small_relu();
    if let Some(hp) = p.proof.hidden_passes.first_mut() {
        hp.chain_init_lower.chain_a_eval += Fr::from(1u64);
    }
    let _ = expect_reject_after_tamper(&p, "tampered hidden_pass chain_init lower a_eval");
}

#[test]
fn full_pass_chain_rejects_tampered_output_bound_slack_eval() {
    // Tamper `output_bound_lower.slack_eval`; slack identity at the
    // FS-derived random point breaks.
    let mut p = prove_small_relu();
    if let Some(ob) = p.proof.output_bound_lower.as_mut() {
        ob.slack_eval += Fr::from(1u64);
    }
    let _ = expect_reject_after_tamper(&p, "tampered output_bound_lower.slack_eval");
}

#[test]
fn full_pass_chain_rejects_tampered_property_check_prop_slack_eval() {
    // Tamper the property-check `prop_slack_eval_at_r`; identity
    // `prop_slack(r) = claimed(r) ± threshold(r)` breaks.
    let mut p = prove_small_relu();
    if let Some(ob) = p.proof.output_bound_lower.as_mut() {
        if let Some(pc) = ob.property_check.as_mut() {
            pc.prop_slack_eval_at_r += Fr::from(1u64);
        }
    }
    let _ = expect_reject_after_tamper(
        &p,
        "tampered output_bound_lower.property_check.prop_slack_eval_at_r",
    );
}

#[test]
fn full_pass_chain_rejects_tampered_property_check_logup_eval() {
    // Tamper the property-check LogUp `prop_slack_logup_eval`; LogUp
    // bottom-denom binding at the LogUp final point breaks.
    let mut p = prove_small_relu();
    if let Some(ob) = p.proof.output_bound_upper.as_mut() {
        if let Some(pc) = ob.property_check.as_mut() {
            pc.prop_slack_logup_eval += Fr::from(1u64);
        }
    }
    let _ = expect_reject_after_tamper(
        &p,
        "tampered output_bound_upper.property_check.prop_slack_logup_eval",
    );
}

#[test]
fn full_pass_chain_rejects_tampered_rescale_c1_fr() {
    // Tamper rescale `c1_fr`; the binding check
    // `expected_c1_fr == proof.c1_fr` rejects.
    let mut p = prove_small_relu();
    if let Some(rescale) = p.proof.rescale_lower.as_mut() {
        assert!(!rescale.is_empty());
        rescale[0].c1_fr += Fr::from(1u64);
    }
    let _ = expect_reject_after_tamper(&p, "tampered rescale c1_fr");
}

#[test]
fn full_pass_chain_rejects_relu_lower_offset_nonzero_eval() {
    // Tamper `relu_lower_offset_proofs[0].b_lower_eval` to non-zero.
    // Either the Hyrax open verify or the `eval == 0` check rejects.
    let mut p = prove_small_relu();
    assert!(
        !p.proof.relu_lower_offset_proofs.is_empty(),
        "small_relu has at least one ReLU layer"
    );
    p.proof.relu_lower_offset_proofs[0].b_lower_eval += Fr::from(1u64);
    let err = expect_reject_after_tamper(&p, "non-zero b_lower_eval");
    assert!(matches!(
        err,
        SnarkError::PcsOpenRejected {
            which: "crate::snark::activation_gadget::b_lower"
        } | SnarkError::RelaxationSoundnessReluLowerOffsetNonZero { .. }
            | SnarkError::TranscriptMismatch
    ));
}

#[test]
fn full_pass_chain_rejects_relu_lower_offset_layer_idx() {
    // Tamper `layer_idx`; architecture-binding rejects before any
    // cryptographic verification.
    let mut p = prove_small_relu();
    assert!(!p.proof.relu_lower_offset_proofs.is_empty());
    p.proof.relu_lower_offset_proofs[0].layer_idx += 100;
    let err = expect_reject_after_tamper(&p, "wrong relu_lower_offset layer_idx");
    assert!(matches!(err, SnarkError::ArchitectureMismatch { .. }));
}

#[test]
fn full_pass_chain_rejects_dropped_relu_lower_offset_proof() {
    // Drop all ReLU lower-offset proofs; count mismatch with the
    // public ReLU layer count must reject.
    let mut p = prove_small_relu();
    assert!(!p.proof.relu_lower_offset_proofs.is_empty());
    p.proof.relu_lower_offset_proofs.clear();
    let _ = expect_reject_after_tamper(&p, "dropped relu_lower_offset_proofs");
}

#[test]
fn full_pass_chain_rejects_tampered_relu_d_boolean_eval() {
    // Tamper `relu_d_boolean_proofs[0].d_lower_eval`; either batched
    // Hyrax verify or the `eq · d · (s_d − d) = claim` identity
    // rejects.
    let mut p = prove_small_relu();
    assert!(
        !p.proof.relu_d_boolean_proofs.is_empty(),
        "small_relu has at least one ReLU layer"
    );
    p.proof.relu_d_boolean_proofs[0].d_lower_eval += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "tampered relu_d_boolean d_lower_eval");
}

#[test]
fn full_pass_chain_rejects_tampered_relu_d_boolean_round_poly() {
    // Tamper sumcheck round poly `at_zero`; per-round invariant
    // `p(0) + p(1) == claim` rejects.
    let mut p = prove_small_relu();
    assert!(!p.proof.relu_d_boolean_proofs.is_empty());
    assert!(!p.proof.relu_d_boolean_proofs[0].rounds.is_empty());
    p.proof.relu_d_boolean_proofs[0].rounds[0].at_zero += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "tampered relu_d_boolean round_poly.at_zero");
}

#[test]
fn full_pass_chain_rejects_dropped_relu_d_boolean_proof() {
    // Drop all ReLU d_boolean proofs; count mismatch rejects.
    let mut p = prove_small_relu();
    assert!(!p.proof.relu_d_boolean_proofs.is_empty());
    p.proof.relu_d_boolean_proofs.clear();
    let _ = expect_reject_after_tamper(&p, "dropped relu_d_boolean_proofs");
}

// ReLU upper-line endpoint validity (Phase 2) tampers.

#[test]
fn full_pass_chain_rejects_tampered_relu_upper_endpoint_slack_eval() {
    // Tamper `relu_upper_endpoint_proofs[0].lo.slack_eval`; either
    // the batched Hyrax open or the final identity at r' rejects.
    let mut p = prove_small_relu();
    assert!(
        !p.proof.relu_upper_endpoint_proofs.is_empty(),
        "small_relu has at least one ReLU layer"
    );
    p.proof.relu_upper_endpoint_proofs[0].lo.slack_eval += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "tampered relu_upper_endpoint slack_eval");
}

#[test]
fn full_pass_chain_rejects_tampered_relu_upper_endpoint_d_eval() {
    // Tamper `relu_upper_endpoint_proofs[0].hi.d_upper_eval`; batched
    // open rejects.
    let mut p = prove_small_relu();
    assert!(!p.proof.relu_upper_endpoint_proofs.is_empty());
    p.proof.relu_upper_endpoint_proofs[0].hi.d_upper_eval += Fr::from(7u64);
    let _ = expect_reject_after_tamper(&p, "tampered relu_upper_endpoint d_upper_eval (hi half)");
}

#[test]
fn full_pass_chain_rejects_tampered_relu_upper_endpoint_round_poly() {
    // Tamper first round poly `at_zero`; per-round invariant rejects.
    let mut p = prove_small_relu();
    assert!(!p.proof.relu_upper_endpoint_proofs.is_empty());
    assert!(!p.proof.relu_upper_endpoint_proofs[0].lo.rounds.is_empty());
    p.proof.relu_upper_endpoint_proofs[0].lo.rounds[0].at_zero += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "tampered relu_upper_endpoint round_poly.at_zero");
}

#[test]
fn full_pass_chain_rejects_dropped_relu_upper_endpoint_proof() {
    // Drop all ReLU upper-endpoint proofs; count mismatch rejects.
    let mut p = prove_small_relu();
    assert!(!p.proof.relu_upper_endpoint_proofs.is_empty());
    p.proof.relu_upper_endpoint_proofs.clear();
    let _ = expect_reject_after_tamper(&p, "dropped relu_upper_endpoint_proofs");
}

#[test]
fn full_pass_chain_rejects_relu_upper_endpoint_layer_idx() {
    // Tamper `layer_idx`; architecture-binding rejects.
    let mut p = prove_small_relu();
    assert!(!p.proof.relu_upper_endpoint_proofs.is_empty());
    p.proof.relu_upper_endpoint_proofs[0].layer_idx += 100;
    let err = expect_reject_after_tamper(&p, "wrong relu_upper_endpoint layer_idx");
    assert!(matches!(err, SnarkError::ArchitectureMismatch { .. }));
}

// ReLU upper-endpoint privacy tampers (committed `preact` / `relu_fr`).

#[test]
fn full_pass_chain_rejects_tampered_relu_upper_endpoint_preact_eval() {
    // Tamper `preact_eval`; either batched Hyrax open at r_final or
    // the final algebraic identity rejects.
    let mut p = prove_small_relu();
    assert!(!p.proof.relu_upper_endpoint_proofs.is_empty());
    p.proof.relu_upper_endpoint_proofs[0].lo.preact_eval += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "tampered relu_upper_endpoint preact_eval");
}

#[test]
fn full_pass_chain_rejects_tampered_relu_upper_endpoint_relu_eval() {
    // Tamper `relu_eval`; same rejection chain as `preact_eval`.
    let mut p = prove_small_relu();
    assert!(!p.proof.relu_upper_endpoint_proofs.is_empty());
    p.proof.relu_upper_endpoint_proofs[0].hi.relu_eval += Fr::from(2u64);
    let _ = expect_reject_after_tamper(&p, "tampered relu_upper_endpoint relu_eval (hi half)");
}

#[test]
fn full_pass_chain_rejects_tampered_relu_upper_endpoint_relu_lookup_alpha() {
    // Tamper `relu_lookup.combine_alpha`; verifier's re-squeezed value
    // diverges and the transcript mismatches.
    let mut p = prove_small_relu();
    assert!(!p.proof.relu_upper_endpoint_proofs.is_empty());
    p.proof.relu_upper_endpoint_proofs[0]
        .lo
        .relu_lookup
        .combine_alpha += Fr::from(3u64);
    let err =
        expect_reject_after_tamper(&p, "tampered relu_upper_endpoint relu_lookup.combine_alpha");
    assert!(matches!(err, SnarkError::TranscriptMismatch | _));
}

#[test]
fn full_pass_chain_rejects_tampered_relu_upper_endpoint_relu_lookup_logup_eval() {
    // Tamper `preact_logup_eval`; binding
    // `bottom_denom = α·preact_logup_eval + relu_logup_eval − β`
    // rejects.
    let mut p = prove_small_relu();
    assert!(!p.proof.relu_upper_endpoint_proofs.is_empty());
    p.proof.relu_upper_endpoint_proofs[0]
        .lo
        .relu_lookup
        .preact_logup_eval += Fr::from(5u64);
    let _ = expect_reject_after_tamper(
        &p,
        "tampered relu_upper_endpoint relu_lookup.preact_logup_eval",
    );
}

#[test]
fn sigmoid_network_fails_closed_at_prove_and_verify() {
    // Smoke test for sigmoid at `precision_bits = 14`. The cert
    // pipeline now forces `working = s_x = 2^11` before the Phase 3c
    // gadget runs (cert-pipeline `s_w → s_x` override), so the prover
    // succeeds — this test pins that behavior.
    use crate::crown::network::{Layer, Network};
    use crate::crown::output_property::{Property, Side};
    use crate::snark::{prove_final_pass, SnarkParams, SnarkStatement};
    use ark_crypto_primitives::sponge::merlin::Transcript;
    use ark_crypto_primitives::sponge::CryptographicSponge;
    use ark_std::{sync::Arc, test_rng};
    use ndarray::{array, Array1, Array2};

    let w1: Array2<f64> = array![[0.5, -0.5], [-0.5, 0.25], [0.25, 0.5]];
    let b1: Array1<f64> = array![0.0, 0.1, -0.05];
    let w2: Array2<f64> = array![[1.0, -1.0, 0.5], [-0.5, 0.5, 1.0]];
    let b2: Array1<f64> = array![0.0, 0.05];
    let net = Network::new(vec![
        Layer::linear(w1, b1).unwrap(),
        Layer::sigmoid(),
        Layer::linear(w2, b2).unwrap(),
    ])
    .unwrap();
    let prop = Property::new_with_thresholds(
        Array2::eye(2),
        Array1::zeros(2),
        Side::Both,
        Some(Array1::from_elem(2, -10.0)),
        Some(Array1::from_elem(2, 10.0)),
    )
    .unwrap();
    let stmt = SnarkStatement {
        network: net,
        property: prop,
        x_lower: array![-0.5, -0.5],
        x_upper: array![0.5, 0.5],
    };

    let mut rng = test_rng();
    let preprocessed = crate::snark::preprocess::test_shared(19, 19, 19);
    let params = SnarkParams::setup(
        &stmt.network,
        &stmt.property,
        14,
        Arc::clone(&preprocessed),
        &mut rng,
    )
    .unwrap();

    let mut prover_sponge = <Transcript as CryptographicSponge>::new(&b"sshape-test".as_slice());
    let _ = prove_final_pass(&stmt, &params, &mut prover_sponge, &mut rng)
        .expect("sigmoid prove must succeed under the s_w → s_x cert-pipeline override");
}

#[test]
fn tanh_network_fails_closed_at_prove_and_verify() {
    // Smoke test for tanh at `precision_bits = 14`. Mirrors the
    // sigmoid sibling above.
    use crate::crown::network::{Layer, Network};
    use crate::crown::output_property::{Property, Side};
    use crate::snark::{prove_final_pass, SnarkParams, SnarkStatement};
    use ark_crypto_primitives::sponge::merlin::Transcript;
    use ark_crypto_primitives::sponge::CryptographicSponge;
    use ark_std::{sync::Arc, test_rng};
    use ndarray::{array, Array1, Array2};

    let w1: Array2<f64> = array![[0.3, -0.2], [-0.1, 0.4], [0.2, 0.3]];
    let b1: Array1<f64> = array![0.0, 0.05, -0.02];
    let w2: Array2<f64> = array![[1.0, -1.0, 0.5], [-0.5, 0.5, 1.0]];
    let b2: Array1<f64> = array![0.0, 0.05];
    let net = Network::new(vec![
        Layer::linear(w1, b1).unwrap(),
        Layer::tanh(),
        Layer::linear(w2, b2).unwrap(),
    ])
    .unwrap();
    let prop = Property::new_with_thresholds(
        Array2::eye(2),
        Array1::zeros(2),
        Side::Both,
        Some(Array1::from_elem(2, -10.0)),
        Some(Array1::from_elem(2, 10.0)),
    )
    .unwrap();
    let stmt = SnarkStatement {
        network: net,
        property: prop,
        x_lower: array![-0.5, -0.5],
        x_upper: array![0.5, 0.5],
    };

    let mut rng = test_rng();
    let preprocessed = crate::snark::preprocess::test_shared(19, 19, 19);
    let params = SnarkParams::setup(
        &stmt.network,
        &stmt.property,
        14,
        Arc::clone(&preprocessed),
        &mut rng,
    )
    .unwrap();
    let mut prover_sponge =
        <Transcript as CryptographicSponge>::new(&b"sshape-tanh-test".as_slice());
    let _ = prove_final_pass(&stmt, &params, &mut prover_sponge, &mut rng)
        .expect("tanh prove must succeed under the s_w → s_x cert-pipeline override");
}

/// Honest end-to-end sigmoid round-trip at `precision_bits = 12` (cert
/// pipeline lands `s_w = 2^11 = s_x`, matching the Phase 3a half-table
/// and the Phase 3c gadget's scale-coupling constraint).
#[test]
fn sigmoid_network_honest_roundtrip() {
    use crate::crown::network::{Layer, Network};
    use crate::crown::output_property::{Property, Side};
    use crate::snark::{
        prove_final_pass, verify_final_pass, SnarkParams, SnarkStatement,
    };
    use ark_crypto_primitives::sponge::merlin::Transcript;
    use ark_crypto_primitives::sponge::CryptographicSponge;
    use ark_std::{sync::Arc, test_rng};
    use ndarray::{array, Array1, Array2};

    let w1: Array2<f64> = array![[0.5, -0.5], [-0.5, 0.25], [0.25, 0.5]];
    let b1: Array1<f64> = array![0.0, 0.1, -0.05];
    let w2: Array2<f64> = array![[1.0, -1.0, 0.5], [-0.5, 0.5, 1.0]];
    let b2: Array1<f64> = array![0.0, 0.05];
    let net = Network::new(vec![
        Layer::linear(w1, b1).unwrap(),
        Layer::sigmoid(),
        Layer::linear(w2, b2).unwrap(),
    ])
    .unwrap();
    let prop = Property::new_with_thresholds(
        Array2::eye(2),
        Array1::zeros(2),
        Side::Both,
        Some(Array1::from_elem(2, -10.0)),
        Some(Array1::from_elem(2, 10.0)),
    )
    .unwrap();
    let stmt = SnarkStatement {
        network: net,
        property: prop,
        x_lower: array![-0.5, -0.5],
        x_upper: array![0.5, 0.5],
    };

    let mut rng = test_rng();
    let preprocessed = crate::snark::preprocess::test_shared(19, 19, 19);
    let params = SnarkParams::setup(
        &stmt.network,
        &stmt.property,
        12,
        Arc::clone(&preprocessed),
        &mut rng,
    )
    .unwrap();

    let mut prover_sponge = <Transcript as CryptographicSponge>::new(&b"sigmoid-honest".as_slice());
    let proof = prove_final_pass(&stmt, &params, &mut prover_sponge, &mut rng)
        .expect("honest sigmoid prove must succeed");
    let mut verifier_sponge =
        <Transcript as CryptographicSponge>::new(&b"sigmoid-honest".as_slice());
    verify_final_pass(&stmt.to_verifier(), &proof, &params, &mut verifier_sponge)
        .expect("honest sigmoid verify must succeed");
}

/// Honest sigmoid roundtrip at `precision_bits = 14`, where the cert
/// pipeline's `pick_scale_pow2` returns `s_w = 2^13 ≠ s_x = 2^11`.
/// Exercises the public `s_w → s_x` rescale (via
/// `rescale_preacts_to_sx`).
#[test]
fn sigmoid_network_honest_roundtrip_pb14() {
    use crate::crown::network::{Layer, Network};
    use crate::crown::output_property::{Property, Side};
    use crate::snark::{
        prove_final_pass, verify_final_pass, SnarkParams, SnarkStatement,
    };
    use ark_crypto_primitives::sponge::merlin::Transcript;
    use ark_crypto_primitives::sponge::CryptographicSponge;
    use ark_std::{sync::Arc, test_rng};
    use ndarray::{array, Array1, Array2};

    let w1: Array2<f64> = array![[0.5, -0.5], [-0.5, 0.25], [0.25, 0.5]];
    let b1: Array1<f64> = array![0.0, 0.1, -0.05];
    let w2: Array2<f64> = array![[1.0, -1.0, 0.5], [-0.5, 0.5, 1.0]];
    let b2: Array1<f64> = array![0.0, 0.05];
    let net = Network::new(vec![
        Layer::linear(w1, b1).unwrap(),
        Layer::sigmoid(),
        Layer::linear(w2, b2).unwrap(),
    ])
    .unwrap();
    let prop = Property::new_with_thresholds(
        Array2::eye(2),
        Array1::zeros(2),
        Side::Both,
        Some(Array1::from_elem(2, -10.0)),
        Some(Array1::from_elem(2, 10.0)),
    )
    .unwrap();
    let stmt = SnarkStatement {
        network: net,
        property: prop,
        x_lower: array![-0.5, -0.5],
        x_upper: array![0.5, 0.5],
    };

    let mut rng = test_rng();
    let preprocessed = crate::snark::preprocess::test_shared(19, 19, 19);
    let params = SnarkParams::setup(
        &stmt.network,
        &stmt.property,
        14,
        Arc::clone(&preprocessed),
        &mut rng,
    )
    .unwrap();

    let mut prover_sponge =
        <Transcript as CryptographicSponge>::new(&b"sigmoid-honest-pb14".as_slice());
    let proof = prove_final_pass(&stmt, &params, &mut prover_sponge, &mut rng)
        .expect("honest sigmoid prove must succeed at precision_bits = 14");
    let mut verifier_sponge =
        <Transcript as CryptographicSponge>::new(&b"sigmoid-honest-pb14".as_slice());
    verify_final_pass(&stmt.to_verifier(), &proof, &params, &mut verifier_sponge)
        .expect("honest sigmoid verify must succeed at precision_bits = 14");
}

/// Regression pin for the Phase 3b/3c transcript order: the prover
/// emits all S-shape endpoint proofs for every activation layer before
/// all critical-point proofs, so the verifier must not interleave
/// endpoint + critical-point verification per layer.
#[test]
fn two_sigmoid_layers_honest_roundtrip() {
    use crate::crown::network::{Layer, Network};
    use crate::crown::output_property::{Property, Side};
    use crate::snark::{
        prove_final_pass, verify_final_pass, SnarkParams, SnarkStatement,
    };
    use ark_crypto_primitives::sponge::merlin::Transcript;
    use ark_crypto_primitives::sponge::CryptographicSponge;
    use ark_std::{sync::Arc, test_rng};
    use ndarray::{array, Array1, Array2};

    let w1: Array2<f64> = array![[0.35, -0.20], [-0.15, 0.25]];
    let b1: Array1<f64> = array![0.02, -0.01];
    let w2: Array2<f64> = array![[0.40, -0.10], [0.15, 0.30]];
    let b2: Array1<f64> = array![0.01, 0.02];
    let w3: Array2<f64> = array![[0.45, -0.20], [-0.25, 0.35]];
    let b3: Array1<f64> = array![0.0, 0.01];
    let net = Network::new(vec![
        Layer::linear(w1, b1).unwrap(),
        Layer::sigmoid(),
        Layer::linear(w2, b2).unwrap(),
        Layer::sigmoid(),
        Layer::linear(w3, b3).unwrap(),
    ])
    .unwrap();
    let prop = Property::new_with_thresholds(
        Array2::eye(2),
        Array1::zeros(2),
        Side::Both,
        Some(Array1::from_elem(2, -10.0)),
        Some(Array1::from_elem(2, 10.0)),
    )
    .unwrap();
    let stmt = SnarkStatement {
        network: net,
        property: prop,
        x_lower: array![-0.25, -0.25],
        x_upper: array![0.25, 0.25],
    };

    let mut rng = test_rng();
    let preprocessed = crate::snark::preprocess::test_shared(19, 19, 19);
    let params = SnarkParams::setup(
        &stmt.network,
        &stmt.property,
        12,
        Arc::clone(&preprocessed),
        &mut rng,
    )
    .unwrap();

    let mut prover_sponge =
        <Transcript as CryptographicSponge>::new(&b"two-sigmoid-honest".as_slice());
    let proof = prove_final_pass(&stmt, &params, &mut prover_sponge, &mut rng)
        .expect("two-sigmoid honest prove must succeed");
    let mut verifier_sponge =
        <Transcript as CryptographicSponge>::new(&b"two-sigmoid-honest".as_slice());
    verify_final_pass(&stmt.to_verifier(), &proof, &params, &mut verifier_sponge)
        .expect("two-sigmoid honest verify must succeed");
}

/// Honest end-to-end tanh round-trip. Same setup as the sigmoid
/// roundtrip — only the activation kind differs.
#[test]
fn tanh_network_honest_roundtrip() {
    use crate::crown::network::{Layer, Network};
    use crate::crown::output_property::{Property, Side};
    use crate::snark::{
        prove_final_pass, verify_final_pass, SnarkParams, SnarkStatement,
    };
    use ark_crypto_primitives::sponge::merlin::Transcript;
    use ark_crypto_primitives::sponge::CryptographicSponge;
    use ark_std::{sync::Arc, test_rng};
    use ndarray::{array, Array1, Array2};

    let w1: Array2<f64> = array![[0.3, -0.2], [-0.1, 0.4], [0.2, 0.3]];
    let b1: Array1<f64> = array![0.0, 0.05, -0.02];
    let w2: Array2<f64> = array![[1.0, -1.0, 0.5], [-0.5, 0.5, 1.0]];
    let b2: Array1<f64> = array![0.0, 0.05];
    let net = Network::new(vec![
        Layer::linear(w1, b1).unwrap(),
        Layer::tanh(),
        Layer::linear(w2, b2).unwrap(),
    ])
    .unwrap();
    let prop = Property::new_with_thresholds(
        Array2::eye(2),
        Array1::zeros(2),
        Side::Both,
        Some(Array1::from_elem(2, -10.0)),
        Some(Array1::from_elem(2, 10.0)),
    )
    .unwrap();
    let stmt = SnarkStatement {
        network: net,
        property: prop,
        x_lower: array![-0.5, -0.5],
        x_upper: array![0.5, 0.5],
    };

    let mut rng = test_rng();
    let preprocessed = crate::snark::preprocess::test_shared(19, 19, 19);
    let params = SnarkParams::setup(
        &stmt.network,
        &stmt.property,
        12,
        Arc::clone(&preprocessed),
        &mut rng,
    )
    .unwrap();

    let mut prover_sponge = <Transcript as CryptographicSponge>::new(&b"tanh-honest".as_slice());
    let proof = prove_final_pass(&stmt, &params, &mut prover_sponge, &mut rng)
        .expect("honest tanh prove must succeed");
    let mut verifier_sponge = <Transcript as CryptographicSponge>::new(&b"tanh-honest".as_slice());
    verify_final_pass(&stmt.to_verifier(), &proof, &params, &mut verifier_sponge)
        .expect("honest tanh verify must succeed");
}

#[test]
fn verify_final_pass_rejects_sshape_statement_at_entry() {
    // Prove a ReLU statement, then call verify with a sigmoid statement
    // (same shape) against that proof. The verifier walks the
    // per-sigmoid/tanh layer Phase 3b loop and finds zero `sshape_*`
    // proofs (vs the 1 expected by the architecture) and rejects with
    // `MissingRequiredComponent` (or an architecture mismatch).
    use crate::crown::network::{Layer, Network};
    use crate::crown::output_property::{Property, Side};
    use crate::snark::{
        prove_final_pass, verify_final_pass, SnarkParams, SnarkStatement,
    };
    use ark_crypto_primitives::sponge::merlin::Transcript;
    use ark_crypto_primitives::sponge::CryptographicSponge;
    use ark_std::{sync::Arc, test_rng};
    use ndarray::{array, Array1, Array2};

    let w1: Array2<f64> = array![[0.5, -0.5], [-0.5, 0.25], [0.25, 0.5]];
    let b1: Array1<f64> = array![0.0, 0.1, -0.05];
    let w2: Array2<f64> = array![[1.0, -1.0, 0.5], [-0.5, 0.5, 1.0]];
    let b2: Array1<f64> = array![0.0, 0.05];
    let net_relu = Network::new(vec![
        Layer::linear(w1.clone(), b1.clone()).unwrap(),
        Layer::relu(),
        Layer::linear(w2.clone(), b2.clone()).unwrap(),
    ])
    .unwrap();
    let prop = Property::new_with_thresholds(
        Array2::eye(2),
        Array1::zeros(2),
        Side::Both,
        Some(Array1::from_elem(2, -10.0)),
        Some(Array1::from_elem(2, 10.0)),
    )
    .unwrap();
    let stmt_relu = SnarkStatement {
        network: net_relu,
        property: prop.clone(),
        x_lower: array![-0.5, -0.5],
        x_upper: array![0.5, 0.5],
    };
    let mut rng = test_rng();
    let preprocessed = crate::snark::preprocess::test_shared(19, 19, 19);
    let params_relu = SnarkParams::setup(
        &stmt_relu.network,
        &stmt_relu.property,
        14,
        Arc::clone(&preprocessed),
        &mut rng,
    )
    .unwrap();
    let mut prover_sponge =
        <Transcript as CryptographicSponge>::new(&b"verify-gate-test".as_slice());
    let proof = prove_final_pass(&stmt_relu, &params_relu, &mut prover_sponge, &mut rng)
        .expect("ReLU prove succeeds");

    let net_sig = Network::new(vec![
        Layer::linear(w1, b1).unwrap(),
        Layer::sigmoid(),
        Layer::linear(w2, b2).unwrap(),
    ])
    .unwrap();
    let stmt_sig = SnarkStatement {
        network: net_sig,
        property: prop,
        x_lower: array![-0.5, -0.5],
        x_upper: array![0.5, 0.5],
    };
    let params_sig = SnarkParams::setup(
        &stmt_sig.network,
        &stmt_sig.property,
        14,
        Arc::clone(&preprocessed),
        &mut rng,
    )
    .unwrap();
    let mut verifier_sponge =
        <Transcript as CryptographicSponge>::new(&b"verify-gate-test".as_slice());
    let err = verify_final_pass(
        &stmt_sig.to_verifier(),
        &proof,
        &params_sig,
        &mut verifier_sponge,
    )
    .expect_err("verify must reject sigmoid statement against a non-sigmoid proof");
    assert!(
        matches!(
            err,
            SnarkError::MissingRequiredComponent { .. }
                | SnarkError::ArchitectureMismatch { .. }
                | SnarkError::ShapeMismatch { .. }
                | SnarkError::RelaxationSoundnessSshapeInvalid {
                    activation: "sigmoid",
                    ..
                }
                | SnarkError::RelaxationSoundnessFinalCheckFailed { .. }
        ),
        "expected verify-side rejection of sigmoid statement; got {err:?}"
    );
}


// Batched-open soundness tampers. Multiple per-tensor (commit, claimed
// value) tuples are bound into a single Hyrax dot-product proof under
// a FS-derived ρ. Tampering any single claimed value diverges the
// verifier's ρ from the prover's, so the batched verify rejects.

#[test]
fn full_pass_chain_rejects_tampered_b_acc_step_b_old_eval() {
    // Tamper b_acc_step `b_old_eval` (batched at shared `r` with
    // b_new, delta); ρ diverges and batched verify rejects.
    let mut p = prove_small_relu();
    if let Some(steps) = p.proof.b_acc_step_lower.as_mut() {
        assert!(!steps.is_empty());
        steps[0].b_old_eval += Fr::from(1u64);
    } else {
        return;
    }
    let _ = expect_reject_after_tamper(&p, "tampered b_acc_step b_old_eval");
}

#[test]
fn full_pass_chain_rejects_tampered_chain_init_spec_c_eval() {
    // Tamper chain_init `spec_c_eval`; either ρ diverges or the
    // `chain_a_eval == spec_c_eval` equality rejects.
    let mut p = prove_small_relu();
    if let Some(ci) = p.proof.chain_init_lower.as_mut() {
        ci.spec_c_eval += Fr::from(1u64);
    } else {
        return;
    }
    let _ = expect_reject_after_tamper(&p, "tampered chain_init spec_c_eval");
}

#[test]
fn full_pass_chain_rejects_tampered_public_x_upper_eval() {
    // Tamper public_binding `x_upper_eval`; batched open AND canonical
    // input-box-MLE check reject.
    let mut p = prove_small_relu();
    if let Some(pb) = p.proof.public_binding.as_mut() {
        pb.x_upper_eval += Fr::from(1u64);
    } else {
        return;
    }
    let _ = expect_reject_after_tamper(&p, "tampered public_binding x_upper_eval");
}

// Phase 3c sigmoid/tanh critical-point tamper coverage. Each test
// starts from an honest sigmoid (or tanh) proof at
// `precision_bits = 12`, mutates a single Phase 3c proof field, and
// asserts `verify_final_pass` rejects. The honest proof builds once
// per process and is cloned by each tamper test (see `fixtures.rs`).

#[test]
fn phase3c_tampered_missing_critical_point_proofs_rejects() {
    // Drop all upper-direction Phase 3c proofs; architecture-vs-proof-
    // count check rejects.
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs.clear();
    let err = expect_reject_after_tamper(&p, "phase3c missing upper critical-point proofs");
    assert!(
        matches!(
            err,
            SnarkError::ArchitectureMismatch { .. } | SnarkError::MissingRequiredComponent { .. }
        ),
        "expected architecture/missing rejection, got {err:?}"
    );
}

#[test]
fn phase3c_tampered_layer_idx_rejects() {
    // Bump `layer_idx`; architecture binding rejects.
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].layer_idx += 100;
    let _ = expect_reject_after_tamper(&p, "phase3c tampered layer_idx");
}

#[test]
fn phase3c_tampered_line_tag_rejects() {
    // Flip `line_tag` upper(0) ↔ lower(1). Bound as a u8 sponge absorb,
    // so any mutation diverges the prover/verifier transcript at the
    // first public-input absorb of the sub-protocol.
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].line_tag ^= 1;
    let _ = expect_reject_after_tamper(&p, "phase3c tampered line_tag");
}

#[test]
fn phase3c_tampered_r_final_rejects() {
    // Tamper one `r_final` entry; verifier accumulates its own
    // challenges from the round polys and rejects the mismatch.
    let mut p = prove_small_sigmoid();
    let r0 = p.proof.sshape_critical_point_upper_proofs[0].r_final[0];
    p.proof.sshape_critical_point_upper_proofs[0].r_final[0] = r0 + Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "phase3c tampered r_final[0]");
}

#[test]
fn phase3c_tampered_envelope_sigma_lo_z_eval_rejects() {
    // Tamper `envelope_sigma_lo_z_eval`; batched Hyrax open at the
    // LogUp witness `bp_high` rejects.
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].envelope_sigma_lo_z_eval += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "phase3c tampered envelope_sigma_lo_z_eval");
}

#[test]
fn phase3c_tampered_envelope_sigma_up_zmd_eval_rejects() {
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].envelope_sigma_up_zmd_eval += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "phase3c tampered envelope_sigma_up_zmd_eval");
}

#[test]
fn phase3c_tampered_envelope_z_eval_rejects() {
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].envelope_z_eval += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "phase3c tampered envelope_z_eval");
}

#[test]
fn phase3c_tampered_envelope_mult_n_vars_rejects() {
    // Tamper `envelope_mult_n_vars`; the multiplicities open at
    // `envelope_table_proof.bottom_point` is lifted differently and
    // Hyrax verify (or an earlier shape check) rejects.
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].envelope_mult_n_vars =
        p.proof.sshape_critical_point_upper_proofs[0]
            .envelope_mult_n_vars
            .wrapping_add(2);
    let _ = expect_reject_after_tamper(&p, "phase3c tampered envelope_mult_n_vars");
}

#[test]
fn phase3c_tampered_envelope_table_bottom_num_rejects() {
    // Tamper `envelope_table_proof.bottom_num`; cross-check with the
    // multiplicities Hyrax open eval (or the LogUp top-fraction
    // cancellation) rejects.
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0]
        .envelope_table_proof
        .bottom_num += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "phase3c tampered envelope_table_proof.bottom_num");
}

#[test]
fn phase3c_tampered_slack_fd1_high_eval_rejects() {
    // Tamper slack chunk-binding high half; chunk-binding identity
    // `slack_fdN − slack_fdN_high · 2^19 − slack_fdN_low = 0` breaks
    // at r_final.
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].slack_fd1_high_eval += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "phase3c tampered slack_fd1_high_eval");
}

#[test]
fn phase3c_tampered_slack_fd2_low_eval_rejects() {
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].slack_fd2_low_eval += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "phase3c tampered slack_fd2_low_eval");
}

#[test]
fn phase3c_tampered_gated_gap_high_eval_rejects() {
    // Tamper gated-gap chunk-binding high half; the chunked-range
    // identity at r_final breaks.
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].gated_gap_high_eval += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "phase3c tampered gated_gap_high_eval");
}

#[test]
fn phase3c_tampered_gated_gap_low_eval_rejects() {
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].gated_gap_low_eval += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "phase3c tampered gated_gap_low_eval");
}

#[test]
fn phase3c_tampered_inside_eval_rejects() {
    // Tamper `inside_eval`; the booleanity identity
    // `inside · (1 − inside) = 0` at r_final breaks.
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].inside_eval += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "phase3c tampered inside_eval (booleanity)");
}

#[test]
fn phase3c_tampered_slack_pos_eval_rejects() {
    // Tamper `slack_pos_eval`; the slack_pos definition identity at
    // r_final breaks.
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].slack_pos_eval += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "phase3c tampered slack_pos_eval");
}

#[test]
fn phase3c_tampered_gated_gap_eval_rejects() {
    // Tamper `gated_gap_eval`; the gated_gap definition identity
    // (gated_gap = inside · factor_b) at r_final breaks.
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].gated_gap_eval += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "phase3c tampered gated_gap_eval");
}

#[test]
fn phase3c_tampered_is_active_eval_rejects() {
    // Tamper `is_active_eval` to non-{0, 1}; booleanity
    // `is_active · (1 − is_active) = 0` at r_final breaks.
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].is_active_eval += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "phase3c tampered is_active_eval (booleanity)");
}

#[test]
fn phase3c_tampered_factor_a_eval_huge_rejects() {
    // Tamper `factor_a_eval` to a value beyond the slack_pos chunked
    // range (here 2^50 > 2^38); slack_pos chunked-range LogUp rejects.
    // Pins the delta-bound soundness: BN254 wraparound must not let a
    // huge `delta` fake a sign.
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].factor_a_eval += Fr::from(1u128 << 50);
    let _ = expect_reject_after_tamper(&p, "phase3c factor_a_eval huge (delta out of bound)");
}

#[test]
fn phase3c_tampered_batched_open_at_r_rejects() {
    // Corrupt one byte of the batched Hyrax dot-product proof; without
    // a verified bool from `hyrax_verify_batched_at`, the claimed evals
    // would be unbound to their commits. This test pins that the bool
    // is checked.
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    let mut p = prove_small_sigmoid();
    let proof = &mut p.proof.sshape_critical_point_upper_proofs[0];
    let mut buf = Vec::new();
    proof
        .batched_open_at_r
        .serialize_compressed(&mut buf)
        .expect("serialize batched_open_at_r");
    let i = buf.len() / 2;
    buf[i] ^= 0x01;
    let tampered = <crate::snark_primitives::polynomial_commitment::HyraxBn254 as crate::snark_primitives::polynomial_commitment::MlPcs>::Proof
        ::deserialize_compressed(&buf[..])
        .expect("deserialize tampered batched_open_at_r");
    proof.batched_open_at_r = tampered;
    let err = expect_reject_after_tamper(
        &p,
        "phase3c tampered batched_open_at_r (Hyrax dot-product proof)",
    );
    assert!(matches!(err, SnarkError::PcsOpenRejected { .. } | _));
}

#[test]
fn phase3c_tampered_dz_step_1_rem_eval_rejects() {
    // Tamper split-arith remainder `dz_step_1_rem_eval`; the
    // id_dz identity `d·z − dz_step_1·s_d − dz_step_1_rem = 0` at
    // r_final breaks.
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].dz_step_1_rem_eval += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "phase3c tampered dz_step_1_rem_eval");
}

#[test]
fn phase3c_tampered_dz_sigma_rem_eval_rejects() {
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].dz_sigma_rem_eval += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "phase3c tampered dz_sigma_rem_eval");
}

#[test]
fn phase3c_tampered_b_sigma_rem_eval_rejects() {
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].b_sigma_rem_eval += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "phase3c tampered b_sigma_rem_eval");
}

#[test]
fn phase3c_tampered_combined_sumcheck_round_rejects() {
    // Tamper first round poly `at_zero`; split-check
    // `g_0(0) + g_0(1) == claim` rejects.
    let mut p = prove_small_sigmoid();
    p.proof.sshape_critical_point_upper_proofs[0].rounds[0].at_zero += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "phase3c tampered rounds[0].at_zero");
}

// Tanh siblings: same identity skeleton (FD slacks at scale
// `s_d·s_x·s_v`, slack/gated-gap chunked, σ-evals batched-bound). A
// pair of representative tanh-side tampers ensures the tanh code path
// is exercised.

#[test]
fn phase3c_tanh_tampered_layer_idx_rejects() {
    // Tanh sibling: bump `layer_idx`.
    let mut p = prove_small_tanh();
    p.proof.sshape_critical_point_upper_proofs[0].layer_idx += 100;
    let _ = expect_reject_after_tamper(&p, "phase3c tanh tampered layer_idx");
}

#[test]
fn phase3c_tanh_tampered_combined_sumcheck_round_rejects() {
    // Tanh sibling: tamper first round poly `at_zero`.
    let mut p = prove_small_tanh();
    p.proof.sshape_critical_point_upper_proofs[0].rounds[0].at_zero += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "phase3c tanh tampered rounds[0].at_zero");
}

/// The hidden-pass `output_bound_*.claimed_commit` MUST be byte-equal
/// to `preact_*_commit`. Without this binding the inequality could be
/// proven against a different MLE than the one downstream relaxation
/// gadgets consume.
#[test]
fn full_pass_chain_rejects_hidden_output_bound_commit_swap() {
    // Swap hidden_pass `output_bound_lower.claimed_commit` with the
    // upper one; byte-equality against `preact_lower_commit` rejects.
    let mut p = prove_small_relu();
    assert!(!p.proof.hidden_passes.is_empty());
    let hp = &mut p.proof.hidden_passes[0];
    let other = hp.output_bound_upper.claimed_commit.clone();
    hp.output_bound_lower.claimed_commit = other;
    let err = expect_reject_after_tamper(
        &p,
        "swapped hidden output_bound_lower.claimed_commit with upper's",
    );
    assert!(matches!(err, SnarkError::ArchitectureMismatch { .. } | _));
}

#[test]
fn full_pass_chain_rejects_hidden_output_bound_commit_vs_preact_divergence() {
    // Replace hidden_pass `output_bound_upper.claimed_commit` with the
    // lower one; byte-equality against `preact_upper_commit` rejects.
    let mut p = prove_small_relu();
    assert!(!p.proof.hidden_passes.is_empty());
    let hp = &mut p.proof.hidden_passes[0];
    let stolen = hp.output_bound_lower.claimed_commit.clone();
    hp.output_bound_upper.claimed_commit = stolen;
    let _ = expect_reject_after_tamper(
        &p,
        "hidden output_bound_upper.claimed_commit replaced with lower's",
    );
}

// Layer-scale typed accessor + per-cell open tampers. The
// `LayerScaleAccessor::new` step runs a global out-of-range exponent
// check on every per-class × per-layer cell; per-class opens
// (`LayerScaleOpenCE`) are individually verified inside
// `verify_layer_scale_opens`.

#[test]
fn full_pass_chain_rejects_tampered_layer_scale_open_c_eval() {
    // Tamper `layer_scale_opens.weight[0].c_eval`; Hyrax verify
    // dot-product fails (open proof was generated for the honest
    // value).
    let mut p = prove_small_relu();
    let opens = &mut p.proof.layer_scale_opens;
    let w0 = opens
        .weight
        .iter_mut()
        .find_map(|o| o.as_mut())
        .expect("ReLU network has at least one Linear layer with a weight scale open");
    w0.c_eval += Fr::from(1u64);
    let _ = expect_reject_after_tamper(&p, "tampered layer_scale_opens.weight[0].c_eval");
}

#[test]
fn full_pass_chain_rejects_tampered_layer_scale_open_proof_swap() {
    // Swap two layers' weight `e_open` proofs. Each is valid for its
    // own layer, but at the wrong unit-vector point the verifier's
    // dot-product fails.
    let mut p = prove_small_relu();
    let weight_opens = &mut p.proof.layer_scale_opens.weight;
    let mut idxs: Vec<usize> = weight_opens
        .iter()
        .enumerate()
        .filter_map(|(i, o)| if o.is_some() { Some(i) } else { None })
        .collect();
    assert!(
        idxs.len() >= 2,
        "ReLU network should have ≥ 2 Linear layers (test fixture invariant)"
    );
    let i0 = idxs.remove(0);
    let i1 = idxs.remove(0);
    let p0 = weight_opens[i0].as_ref().unwrap().e_open.clone();
    let p1 = weight_opens[i1].as_ref().unwrap().e_open.clone();
    weight_opens[i0].as_mut().unwrap().e_open = p1;
    weight_opens[i1].as_mut().unwrap().e_open = p0;
    let _ = expect_reject_after_tamper(
        &p,
        "swapped layer_scale_opens weight[i0].e_open <-> weight[i1].e_open",
    );
}

#[test]
fn full_pass_chain_rejects_layer_scale_open_at_wrong_class_slot() {
    // Move the Linear-layer weight open to the relax_d slot at the
    // same layer index. Architecture binding ("relax_* present only
    // at Activation layers") rejects.
    let mut p = prove_small_relu();
    let i = p
        .proof
        .layer_scale_opens
        .weight
        .iter()
        .position(|o| o.is_some())
        .expect("at least one Linear layer");
    let stolen = p.proof.layer_scale_opens.weight[i].clone();
    p.proof.layer_scale_opens.relax_d[i] = stolen;
    let _ =
        expect_reject_after_tamper(&p, "wrong-class layer_scale_opens slot (weight in relax_d)");
}

#[test]
fn full_pass_chain_rejects_layer_scale_accessor_out_of_range_exponent() {
    // Tamper `bias[i].e_eval` to a huge field element decoded as a
    // huge i32. The Hyrax open verify rejects first (proof was
    // generated for the honest eval); the accessor's range gate would
    // catch it otherwise.
    let mut p = prove_small_relu();
    let opens = &mut p.proof.layer_scale_opens;
    let bias0 = opens
        .bias
        .iter_mut()
        .find_map(|o| o.as_mut())
        .expect("ReLU network has at least one Linear layer with a bias scale open");
    bias0.e_eval = crate::snark_primitives::finite_field::signed_lift_to_fr(1_000_000_000i128);
    let _ = expect_reject_after_tamper(&p, "out-of-range layer_scale e_eval (bias)");
}
