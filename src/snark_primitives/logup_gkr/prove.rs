//! LogUp-GKR prover. Drives one degree-3 sumcheck per layer transition
//! from the top fraction down to the bottom layer's MLE.

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::{Field, PrimeField};

use crate::snark_primitives::sumcheck::RoundPoly3;

use super::types::{
    absorb_round_poly, LayerProof, LogUpCircuit, LogUpError, LogUpLayer, LogUpProof,
};

/// Prove a single LogUp-GKR circuit. The top fraction is shipped in
/// the proof so the verifier can use it as the initial claim without
/// evaluating the circuit themselves.
pub fn prove_circuit<F, S>(
    circuit: &LogUpCircuit<F>,
    sponge: &mut S,
) -> Result<LogUpProof<F>, LogUpError>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
{
    let n_layers = circuit.num_vars(); // == number of layer transitions
    if n_layers == 0 {
        // Trivial circuit (size-2 layer, no sumchecks). The verifier
        // still squeezes β (and λ) and expects `bottom_*` to be the
        // β-folded top — not the aggregated fraction — so mirror its
        // protocol exactly.
        let top_layer = circuit.layers.last().expect("≥ 1 layer");
        let [n0_top, n1_top, d0_top, d1_top] = match top_layer {
            LogUpLayer::Generic {
                numerator,
                denominator,
            } => [numerator[0], numerator[1], denominator[0], denominator[1]],
            LogUpLayer::InitialLookup { denominator } => {
                [-F::one(), -F::one(), denominator[0], denominator[1]]
            }
            LogUpLayer::InitialTable {
                numerator,
                denominator,
            } => [numerator[0], numerator[1], denominator[0], denominator[1]],
        };
        sponge.absorb(&n0_top);
        sponge.absorb(&n1_top);
        sponge.absorb(&d0_top);
        sponge.absorb(&d1_top);
        let beta = sponge.squeeze_field_elements::<F>(1)[0];
        // Squeeze λ to keep the sponge in sync with the verifier even
        // though no sumcheck rounds consume it here.
        let _lambda = sponge.squeeze_field_elements::<F>(1)[0];
        let bottom_num = (F::one() - beta) * n0_top + beta * n1_top;
        let bottom_denom = (F::one() - beta) * d0_top + beta * d1_top;
        return Ok(LogUpProof {
            top_numerator: n0_top,
            top_denominator: d0_top,
            layers: vec![],
            bottom_num,
            bottom_denom,
            bottom_point: vec![beta],
        });
    }

    // Top layer (size 2): emit (n0, n1, d0, d1). Verifier squeezes β₀
    // and λ to fold the two halves into a (num + λ·denom) claim at β₀.
    let top_layer = circuit.layers.last().expect("≥ 1 layer");
    let [n0_top, n1_top, d0_top, d1_top] = match top_layer {
        LogUpLayer::Generic {
            numerator,
            denominator,
        } => [numerator[0], numerator[1], denominator[0], denominator[1]],
        LogUpLayer::InitialLookup { denominator } => {
            [-F::one(), -F::one(), denominator[0], denominator[1]]
        }
        LogUpLayer::InitialTable {
            numerator,
            denominator,
        } => [numerator[0], numerator[1], denominator[0], denominator[1]],
    };
    sponge.absorb(&n0_top);
    sponge.absorb(&n1_top);
    sponge.absorb(&d0_top);
    sponge.absorb(&d1_top);

    let beta = sponge.squeeze_field_elements::<F>(1)[0];
    let lambda = sponge.squeeze_field_elements::<F>(1)[0];

    // Initial point r₀ = (β,); initial claim about the next layer down
    // is `num(r₀) + λ · denom(r₀)`, with each part folded from the two
    // top halves via `(1-β)·*_lo + β·*_hi`.
    let mut current_point: Vec<F> = vec![beta];
    let mut current_num = (F::one() - beta) * n0_top + beta * n1_top;
    let mut current_denom = (F::one() - beta) * d0_top + beta * d1_top;

    // Walk top → bottom. `circuit.layers` is bottom-up, so iterate
    // from last to second-from-bottom.
    let mut layer_proofs: Vec<LayerProof<F>> = Vec::with_capacity(n_layers);
    let total = circuit.layers.len();
    for layer_idx in (0..total - 1).rev() {
        let layer = &circuit.layers[layer_idx];
        let [n_lo, n_hi, d_lo, d_hi] = layer.into_halves();
        let nv = layer.num_vars();
        debug_assert_eq!(1 << nv, n_lo.len());
        debug_assert_eq!(current_point.len(), nv);

        // Sumcheck `g(x) = eq(x, current_point) · [n_lo·d_hi + n_hi·d_lo
        // + λ·d_lo·d_hi]`; total sum = current_num + λ · current_denom.
        let claim = current_num + lambda * current_denom;
        let (round_polys, sc_point, finals) = prove_layer_sumcheck(
            &n_lo,
            &n_hi,
            &d_lo,
            &d_hi,
            &current_point,
            lambda,
            claim,
            sponge,
        );

        layer_proofs.push(LayerProof {
            rounds: round_polys,
            final_evals: finals,
        });

        // Squeeze the next-layer batching challenge β'; the next
        // claim lives at point `(β', r)` and folds the four
        // `final_evals` via the standard `(1-β')·*_lo + β'·*_hi`.
        let beta_next = sponge.squeeze_field_elements::<F>(1)[0];
        let mut new_point = Vec::with_capacity(nv + 1);
        new_point.push(beta_next);
        new_point.extend_from_slice(&sc_point);
        current_point = new_point;
        current_num = (F::one() - beta_next) * finals[0] + beta_next * finals[1];
        current_denom = (F::one() - beta_next) * finals[2] + beta_next * finals[3];
    }

    Ok(LogUpProof {
        top_numerator: top_layer
            .into_halves()
            .into_iter()
            .next()
            .map(|_| n0_top)
            .unwrap_or_else(|| panic!("unreachable: layer always has halves")),
        top_denominator: d0_top, // see verifier — we ship raw n*/d* via
        // the four-element top_* below
        layers: layer_proofs,
        bottom_num: current_num,
        bottom_denom: current_denom,
        bottom_point: current_point,
    })
}

/// Run one layer's degree-3 sumcheck. Returns the round polynomials,
/// the challenge point, and the four MLE evaluations at that point.
#[allow(clippy::too_many_arguments)]
fn prove_layer_sumcheck<F, S>(
    n_lo: &[F],
    n_hi: &[F],
    d_lo: &[F],
    d_hi: &[F],
    eq_anchor: &[F],
    lambda: F,
    expected_sum: F,
    sponge: &mut S,
) -> (Vec<RoundPoly3<F>>, Vec<F>, [F; 4])
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
{
    let nv = eq_anchor.len();
    debug_assert_eq!(1 << nv, n_lo.len());

    // Build the eq table once; update in place each round.
    let mut eq_tab: Vec<F> = build_eq_table(eq_anchor);
    let mut nlo: Vec<F> = n_lo.to_vec();
    let mut nhi: Vec<F> = n_hi.to_vec();
    let mut dlo: Vec<F> = d_lo.to_vec();
    let mut dhi: Vec<F> = d_hi.to_vec();
    let mut rounds: Vec<RoundPoly3<F>> = Vec::with_capacity(nv);
    let mut challenges: Vec<F> = Vec::with_capacity(nv);
    let mut current_sum = expected_sum;

    for _ in 0..nv {
        let half = eq_tab.len() / 2;
        // Affine-extend each MLE table to X ∈ {0, 1, 2, 3} and sum
        // `eq(X)·[nlo·dhi + nhi·dlo + λ·dlo·dhi]` over the rest.
        let (mut e0, mut e1, mut e2, mut e3) = (F::zero(), F::zero(), F::zero(), F::zero());
        for i in 0..half {
            let eq0 = eq_tab[i];
            let eq1 = eq_tab[half + i];
            let eq2 = eq1.double() - eq0;
            let eq3 = eq1.double() + eq1 - eq0.double();

            let nl0 = nlo[i];
            let nl1 = nlo[half + i];
            let nl2 = nl1.double() - nl0;
            let nl3 = nl1.double() + nl1 - nl0.double();

            let nh0 = nhi[i];
            let nh1 = nhi[half + i];
            let nh2 = nh1.double() - nh0;
            let nh3 = nh1.double() + nh1 - nh0.double();

            let dl0 = dlo[i];
            let dl1 = dlo[half + i];
            let dl2 = dl1.double() - dl0;
            let dl3 = dl1.double() + dl1 - dl0.double();

            let dh0 = dhi[i];
            let dh1 = dhi[half + i];
            let dh2 = dh1.double() - dh0;
            let dh3 = dh1.double() + dh1 - dh0.double();

            e0 += eq0 * (nl0 * dh0 + nh0 * dl0 + lambda * dl0 * dh0);
            e1 += eq1 * (nl1 * dh1 + nh1 * dl1 + lambda * dl1 * dh1);
            e2 += eq2 * (nl2 * dh2 + nh2 * dl2 + lambda * dl2 * dh2);
            e3 += eq3 * (nl3 * dh3 + nh3 * dl3 + lambda * dl3 * dh3);
        }
        let poly = RoundPoly3 {
            at_zero: e0,
            at_one: e1,
            at_two: e2,
            at_three: e3,
        };
        debug_assert_eq!(
            poly.at_zero + poly.at_one,
            current_sum,
            "internal: round-poly split should match incoming claim"
        );
        let r = absorb_round_poly(sponge, &poly);
        bind_in_place(&mut eq_tab, r, half);
        bind_in_place(&mut nlo, r, half);
        bind_in_place(&mut nhi, r, half);
        bind_in_place(&mut dlo, r, half);
        bind_in_place(&mut dhi, r, half);
        current_sum = poly.evaluate(r);
        challenges.push(r);
        rounds.push(poly);
    }
    let finals = [nlo[0], nhi[0], dlo[0], dhi[0]];
    (rounds, challenges, finals)
}

/// In-place bookkeeping bind: `tab[i] ← tab[i] + r·(tab[half+i] - tab[i])`,
/// then `truncate(half)`.
fn bind_in_place<F: Field>(tab: &mut Vec<F>, r: F, half: usize) {
    for i in 0..half {
        let delta = tab[half + i] - tab[i];
        tab[i] += r * delta;
    }
    tab.truncate(half);
}

/// Build the eq-polynomial evaluation table on `{0,1}^nv` using the
/// "first variable = highest-order bit" convention shared by the rest
/// of this module.
fn build_eq_table<F: Field>(anchor: &[F]) -> Vec<F> {
    let nv = anchor.len();
    let mut tab = vec![F::one(); 1 << nv];
    for (k, &a) in anchor.iter().enumerate() {
        let stride = 1usize << (nv - 1 - k);
        for block in (0..(1 << nv)).step_by(stride * 2) {
            for i in 0..stride {
                let lo_idx = block + i;
                let hi_idx = block + stride + i;
                let v_lo = tab[lo_idx];
                let v_hi = tab[hi_idx];
                tab[lo_idx] = v_lo * (F::one() - a);
                tab[hi_idx] = v_hi * a;
            }
        }
    }
    tab
}

/// Build the lookup and table circuits for the given (witness, table,
/// multiplicities, α) and prove each. The verifier must additionally
/// check that `lookup_top + table_top = 0` (the LogUp identity).
pub fn prove_lookup<F, S>(
    witness: &[F],
    table: &[F],
    multiplicities: &[F],
    alpha: F,
    sponge: &mut S,
) -> Result<(LogUpProof<F>, LogUpProof<F>, F, F), LogUpError>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
{
    let lookup = LogUpCircuit::lookup(witness, alpha)?;
    let table_circ = LogUpCircuit::table(table, multiplicities, alpha)?;
    sponge.absorb(&alpha);
    let lookup_top = lookup.output();
    let table_top = table_circ.output();
    // The verifier checks the identity via cross-multiplication on the
    // top fractions; this `combined` is only the local sanity value.
    let combined =
        lookup_top.numerator * table_top.denominator + lookup_top.denominator * table_top.numerator;
    let lookup_proof = prove_circuit(&lookup, sponge)?;
    let table_proof = prove_circuit(&table_circ, sponge)?;
    let _ = combined;
    Ok((
        lookup_proof,
        table_proof,
        lookup_top.denominator,
        table_top.denominator,
    ))
}
