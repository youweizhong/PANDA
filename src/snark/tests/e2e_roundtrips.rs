//! Full prove + verify round-trips on small CROWN fixtures. Happy-path
//! tests: assert the verifier accepts an honest proof.

use super::fixtures::{
    fresh_sponge, prove_deeper_relu, prove_multi_spec_odd_ceil_log2, prove_small_relu,
    verify_with_fresh_sponge, Proved,
};

#[test]
fn roundtrip_small_relu() {
    // The verifier no longer returns the dequantized bound; the
    // in-SNARK property check binds the (private) claimed bound to the
    // public threshold. Accept means "the property holds". We then
    // forward all input-box corners and confirm they satisfy the
    // threshold the prover proved against.
    let p = prove_small_relu();
    let bound = verify_with_fresh_sponge(&p).unwrap();
    assert!(bound.lower.is_none(), "bound is private now");
    assert!(bound.upper.is_none(), "bound is private now");
    let lo_t = p.stmt.property.lower_threshold_or_zero();
    let up_t = p.stmt.property.upper_threshold_or_zero();
    for mask in 0..4 {
        let x = ndarray::array![
            if mask & 1 == 1 {
                p.stmt.x_upper[0]
            } else {
                p.stmt.x_lower[0]
            },
            if mask & 2 == 2 {
                p.stmt.x_upper[1]
            } else {
                p.stmt.x_lower[1]
            }
        ];
        let y = p.stmt.network.forward(&x);
        for k in 0..2 {
            assert!(
                y[k] >= lo_t[k] && y[k] <= up_t[k],
                "corner {mask} spec {k} should satisfy the public threshold"
            );
        }
    }
}

#[test]
fn roundtrip_deeper_relu() {
    let p = prove_deeper_relu();
    let _bound = verify_with_fresh_sponge(&p).unwrap();
}

#[test]
fn roundtrip_runtime_table_parameters_in_one_process() {
    // Table sizes are RUNTIME parameters: one process can prove and
    // verify at different output-bound budgets — including budgets
    // (like 20) that previously had no prebuilt table — by building
    // `Preprocessed` from the public parameter pair per proof.
    use super::super::{prove_final_pass, SnarkParams, SnarkStatement};
    use super::fixtures::small_relu_2x3x2;
    use ark_std::test_rng;

    let mut proved_at_21 = None;
    for (ob_bits, gadget_bits) in [(19usize, 19usize), (20, 20), (21, 19)] {
        let mut rng = test_rng();
        let (net, prop, x_l, x_u) = small_relu_2x3x2();
        let stmt = SnarkStatement {
            network: net,
            property: prop,
            x_lower: x_l,
            x_upper: x_u,
        };
        let pre = crate::snark::preprocess::test_shared(19, ob_bits, gadget_bits);
        let params = SnarkParams::setup(&stmt.network, &stmt.property, 14, pre, &mut rng).unwrap();
        assert_eq!(params.out_bound_range_bits, ob_bits);
        assert_eq!(params.gadget_range_bits, gadget_bits);
        let mut prover_sponge = fresh_sponge();
        let proof = prove_final_pass(&stmt, &params, &mut prover_sponge, &mut rng).unwrap();
        let p = Proved {
            stmt,
            params,
            proof,
        };
        verify_with_fresh_sponge(&p).unwrap();
        if ob_bits == 21 {
            proved_at_21 = Some(p);
        }
    }

    // A proof produced at out-bound 21 must NOT verify against a
    // statement whose budget is 19: the verifier pins the range-table
    // width of the final-pass LogUps.
    let p21 = proved_at_21.unwrap();
    let mut params_19 = p21.params.clone();
    params_19.out_bound_range_bits = 19;
    params_19.gadget_range_bits = 19;
    params_19.preprocessed = crate::snark::preprocess::test_shared(19, 19, 19);
    let mismatched = Proved {
        stmt: p21.stmt.clone(),
        params: params_19,
        proof: p21.proof.clone(),
    };
    assert!(
        verify_with_fresh_sponge(&mismatched).is_err(),
        "21-bit proof must reject under a 19-bit statement"
    );
}

#[test]
fn proof_rejects_under_mismatched_gadget_budget() {
    // The per-neuron gadget budget is its own runtime public
    // parameter. A proof whose gadget range checks ran at 19 bits must
    // reject under a verifier statement that says 18 — the verifier
    // pins every per-neuron PosRangeLogUp table length against its own
    // gadget parameter.
    use super::super::{prove_final_pass, SnarkParams, SnarkStatement};
    use super::fixtures::small_relu_2x3x2;
    use ark_std::test_rng;

    let mut rng = test_rng();
    let (net, prop, x_l, x_u) = small_relu_2x3x2();
    let stmt = SnarkStatement {
        network: net,
        property: prop,
        x_lower: x_l,
        x_upper: x_u,
    };
    let pre = crate::snark::preprocess::test_shared(19, 21, 19);
    let params = SnarkParams::setup(&stmt.network, &stmt.property, 14, pre, &mut rng).unwrap();
    let mut prover_sponge = fresh_sponge();
    let proof = prove_final_pass(&stmt, &params, &mut prover_sponge, &mut rng).unwrap();

    let mut params_g18 = params.clone();
    params_g18.gadget_range_bits = 18;
    params_g18.preprocessed = crate::snark::preprocess::test_shared(19, 21, 18);
    let mismatched = Proved {
        stmt,
        params: params_g18,
        proof,
    };
    assert!(
        verify_with_fresh_sponge(&mismatched).is_err(),
        "gadget-19 proof must reject under a gadget-18 statement"
    );
}

#[test]
fn sshape_gadgets_use_the_narrow_budget_while_output_stage_uses_the_wide_one() {
    // The whole point of the split: a sigmoid/tanh net proved at
    // out_bound=21, gadget=19 must run every PER-NEURON sshape range
    // check against the 2^19 table while the FINAL-pass output bound
    // uses the 2^21 table. This is the ONLY committed coverage of the
    // sshape gadgets at out_bound != gadget — the ReLU split-budget
    // tests exercise only the shared pos-range path, not the
    // sshape-specific budget-selection sites. Asserting the concrete
    // table widths catches BOTH a one-sided prover/verifier
    // disagreement (honest reject) AND a both-sides flip of an sshape
    // site to the wide budget (wrong table_len, still "sound" but the
    // performance regression the split exists to prevent).
    use super::super::{prove_final_pass, SnarkParams, SnarkStatement};
    use super::fixtures::{small_sigmoid_2x3x2, small_tanh_2x3x2};
    use ark_std::test_rng;

    for (label, build) in [
        ("sigmoid", small_sigmoid_2x3x2 as fn() -> (_, _, _, _)),
        ("tanh", small_tanh_2x3x2 as fn() -> (_, _, _, _)),
    ] {
        let mut rng = test_rng();
        let (net, prop, x_l, x_u) = build();
        let stmt = SnarkStatement {
            network: net,
            property: prop,
            x_lower: x_l,
            x_upper: x_u,
        };
        // out_bound=21, gadget=19 — the production __ob21_g19 setting.
        let pre = crate::snark::preprocess::test_shared(19, 21, 19);
        let params = SnarkParams::setup(&stmt.network, &stmt.property, 12, pre, &mut rng).unwrap();
        let mut prover_sponge = fresh_sponge();
        let proof = prove_final_pass(&stmt, &params, &mut prover_sponge, &mut rng).unwrap();

        // Final-pass output bound rides the WIDE table.
        for ob in [
            proof.output_bound_lower.as_ref(),
            proof.output_bound_upper.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            assert_eq!(
                ob.table_len,
                1usize << 21,
                "{label}: output-bound LogUp must use the 2^21 out-bound table"
            );
        }

        // Every per-neuron sshape endpoint range check rides the NARROW
        // table. These vecs are non-empty for a one-activation-layer net.
        let endpoint_proofs = [
            &proof.sshape_upper_at_lower_proofs,
            &proof.sshape_upper_at_upper_proofs,
            &proof.sshape_lower_at_lower_proofs,
            &proof.sshape_lower_at_upper_proofs,
        ];
        let mut saw_endpoint = false;
        for group in endpoint_proofs {
            for ep in group.iter() {
                saw_endpoint = true;
                for (which, r) in [
                    ("abs_l", &ep.abs_l_range),
                    ("dx_step_1_rem", &ep.dx_step_1_rem_range),
                    ("dx_sigma_rem", &ep.dx_sigma_rem_range),
                    ("b_sigma_rem", &ep.b_sigma_rem_range),
                    ("diff", &ep.diff_range),
                ] {
                    assert_eq!(
                        r.table_len,
                        1usize << 19,
                        "{label}: sshape endpoint {which} range must use the 2^19 gadget table"
                    );
                }
            }
        }
        assert!(
            saw_endpoint,
            "{label}: expected at least one sshape endpoint proof"
        );

        // Honest roundtrip still verifies (prover and verifier agree on
        // every budget); a one-sided flip would have rejected here.
        let p = Proved {
            stmt,
            params,
            proof,
        };
        verify_with_fresh_sponge(&p).unwrap();

        // A statement claiming gadget=18 must reject the gadget=19
        // proof: the sshape verifier pins each PosRangeLogUp table
        // length against its OWN gadget budget.
        let mut params_g18 = p.params.clone();
        params_g18.gadget_range_bits = 18;
        params_g18.preprocessed = crate::snark::preprocess::test_shared(19, 21, 18);
        let mismatched = Proved {
            stmt: p.stmt.clone(),
            params: params_g18,
            proof: p.proof.clone(),
        };
        assert!(
            verify_with_fresh_sponge(&mismatched).is_err(),
            "{label}: gadget-19 proof must reject under a gadget-18 statement"
        );
    }
}

#[test]
fn sshape_roundtrip_at_non_default_sigma_scales_and_rejects_mismatch() {
    // The sigmoid/tanh table scales `(sigma_x_scale_log2,
    // sigma_v_scale_log2)` are runtime PUBLIC parameters. An honest
    // proof produced at NON-DEFAULT scales (here s_x = 2^13, s_v = 2^16;
    // the default for precision 12 would be s_x = 2^12, s_v = 2^14) must
    // verify, and a verifier statement claiming DIFFERENT scales must
    // reject — the verifier rebuilds the σ tables from its own scales
    // and pins the sigmoid/tanh working scale through the public
    // binding, so neither an s_x nor an s_v disagreement can slip
    // through. ReLU nets never touch the σ tables, so this uses an
    // sshape net (sigmoid) where the scales are load-bearing.
    use super::super::{prove_final_pass, SnarkParams, SnarkStatement};
    use super::fixtures::small_sigmoid_2x3x2;
    use ark_std::test_rng;

    const SX: i32 = 13; // non-default for precision 12 (default s_x = 12)
    const SV: i32 = 16; // non-default for precision 12 (default s_v = 14)

    let mut rng = test_rng();
    let (net, prop, x_l, x_u) = small_sigmoid_2x3x2();
    let stmt = SnarkStatement {
        network: net,
        property: prop,
        x_lower: x_l,
        x_upper: x_u,
    };
    let pre = crate::snark::preprocess::test_shared_sigma(19, 21, 19, SX, SV);
    let params = SnarkParams::setup(&stmt.network, &stmt.property, 12, pre, &mut rng).unwrap();
    assert_eq!(params.sigma_x_scale_log2, SX);
    assert_eq!(params.sigma_v_scale_log2, SV);
    let mut prover_sponge = fresh_sponge();
    let proof = prove_final_pass(&stmt, &params, &mut prover_sponge, &mut rng).unwrap();
    let p = Proved {
        stmt,
        params,
        proof,
    };
    // Honest roundtrip at the non-default scales verifies.
    verify_with_fresh_sponge(&p).unwrap();

    // A verifier claiming a different s_x (with the σ tables rebuilt at
    // that s_x) must reject: the forced sigmoid/tanh working scale no
    // longer matches the proof's target scale.
    let mut params_sx = p.params.clone();
    params_sx.sigma_x_scale_log2 = 11;
    params_sx.preprocessed = crate::snark::preprocess::test_shared_sigma(19, 21, 19, 11, SV);
    let mismatched_sx = Proved {
        stmt: p.stmt.clone(),
        params: params_sx,
        proof: p.proof.clone(),
    };
    assert!(
        verify_with_fresh_sponge(&mismatched_sx).is_err(),
        "s_x=13 proof must reject under an s_x=11 statement"
    );

    // A verifier claiming a different s_v (σ tables rebuilt at that s_v)
    // must reject: the σ envelope codes differ, so the sshape gadget
    // sumchecks no longer close.
    let mut params_sv = p.params.clone();
    params_sv.sigma_v_scale_log2 = 15;
    params_sv.preprocessed = crate::snark::preprocess::test_shared_sigma(19, 21, 19, SX, 15);
    let mismatched_sv = Proved {
        stmt: p.stmt.clone(),
        params: params_sv,
        proof: p.proof.clone(),
    };
    assert!(
        verify_with_fresh_sponge(&mismatched_sv).is_err(),
        "s_v=16 proof must reject under an s_v=15 statement"
    );
}

#[test]
fn proof_rejects_under_mismatched_range_table_width() {
    // The signed range / ReLU table half-width is also a runtime public
    // parameter. A proof produced at half-width 19 must reject under a
    // verifier whose statement says 20 — the verifier pins every
    // prover-claimed table length against its own parameter.
    let p = prove_small_relu();
    verify_with_fresh_sponge(&p).unwrap();
    let mut params_wider = p.params.clone();
    params_wider.preprocessed = crate::snark::preprocess::test_shared(
        20,
        p.params.out_bound_range_bits,
        p.params.gadget_range_bits,
    );
    let mismatched = Proved {
        stmt: p.stmt.clone(),
        params: params_wider,
        proof: p.proof.clone(),
    };
    assert!(
        verify_with_fresh_sponge(&mismatched).is_err(),
        "proof at range-table half-width 19 must reject under a 20-bit-wide statement"
    );
}

#[test]
fn setup_rejects_precision_without_table_headroom() {
    // precision_bits must leave at least one bit of headroom inside the
    // signed range table: setup rejects precision_bits >= half_bits.
    use super::super::SnarkParams;
    use super::fixtures::small_relu_2x3x2;
    use ark_std::test_rng;
    let mut rng = test_rng();
    let (net, prop, ..) = small_relu_2x3x2();
    let pre = crate::snark::preprocess::test_shared(19, 19, 19);
    assert!(
        SnarkParams::setup(&net, &prop, 19, pre.clone(), &mut rng).is_err(),
        "precision_bits == range_table_half_bits must fail setup"
    );
    assert!(
        SnarkParams::setup(&net, &prop, 1, pre, &mut rng).is_err(),
        "precision_bits <= 1 must fail setup"
    );
}

#[test]
#[ignore = "five-layer net's CROWN bound diverges enough to push slacks above a 2^19 LogUp range table; run with a wider runtime table (heavy) or Lasso decomposition. Tracked separately."]
fn roundtrip_five_layer() {
    use super::super::{prove_final_pass, SnarkParams, SnarkStatement};
    use super::fixtures::five_layer_relu;
    use ark_std::test_rng;
    let mut rng = test_rng();
    let (net, prop, x_l, x_u) = five_layer_relu();
    let stmt = SnarkStatement {
        network: net,
        property: prop,
        x_lower: x_l,
        x_upper: x_u,
    };
    let params = SnarkParams::setup(
        &stmt.network,
        &stmt.property,
        14,
        crate::snark::preprocess::test_shared(21, 21, 21),
        &mut rng,
    )
    .unwrap();
    let mut prover_sponge = fresh_sponge();
    let proof = prove_final_pass(&stmt, &params, &mut prover_sponge, &mut rng).unwrap();
    let p = Proved {
        stmt,
        params,
        proof,
    };
    let _bound = verify_with_fresh_sponge(&p).unwrap();
}

#[test]
fn roundtrip_multi_spec_odd_ceil_log2_n_spec() {
    // Exercises `native_vector_n_vars(5) == 4` (the even-bumped path):
    // `ceil_log2(5) = 3` is odd, and Hyrax requires even, so the
    // architecture-derived helper must yield 4. Sanity-checks the
    // pipeline and that output_bound proofs record the bumped n_vars.
    let n_spec = 5usize;
    let raw_log = (5usize).next_power_of_two().trailing_zeros() as usize;
    assert_eq!(raw_log, 3, "ceil_log2(5) is 3 (odd) — that's the point");
    assert_eq!(
        crate::snark::commitment::commit::native_vector_n_vars(n_spec),
        4
    );

    let p = prove_multi_spec_odd_ceil_log2();
    if let Some(ob) = p.proof.output_bound_lower.as_ref() {
        assert_eq!(ob.n_vars, 4);
    }
    if let Some(ob) = p.proof.output_bound_upper.as_ref() {
        assert_eq!(ob.n_vars, 4);
    }
    let _bound = verify_with_fresh_sponge(&p).unwrap();
}
