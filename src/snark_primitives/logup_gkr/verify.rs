//! LogUp-GKR verifier. Replays the per-layer Fiat-Shamir transcript,
//! checks each round-poly's split identity, confirms the layer's final
//! identity, and walks `(num, denom)` down to the bottom-layer claim
//! that the SNARK driver wires to a PCS opening.

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::{Field, PrimeField};

use super::types::{absorb_round_poly, LogUpError, LogUpProof};

/// Stub entry point: the protocol needs all four top halves to start
/// the transcript, but this signature only carries a single fraction.
/// Always errors out — use [`verify_circuit_with_top`].
pub fn verify_circuit<F, S>(
    proof: &LogUpProof<F>,
    n_vars: usize,
    expected_top_numerator: F,
    _sponge: &mut S,
) -> Result<(), LogUpError>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
{
    if proof.layers.len() != n_vars {
        return Err(LogUpError::LayerCountMismatch {
            expected: n_vars,
            got: proof.layers.len(),
        });
    }

    if proof.top_numerator != expected_top_numerator {
        return Err(LogUpError::TopFractionMismatch);
    }

    Err(LogUpError::TopFractionMismatch)
}

/// Verify a `LogUpProof`. `top = (n0, n1, d0, d1)` is the four top-layer
/// values the SNARK driver supplies (either as a public constant or via
/// a PCS opening); `expected_top_numerator` is what the aggregated top
/// numerator should equal (zero for the LogUp identity).
pub fn verify_circuit_with_top<F, S>(
    proof: &LogUpProof<F>,
    n_vars: usize,
    top: [F; 4],
    expected_top_numerator: F,
    sponge: &mut S,
) -> Result<(), LogUpError>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
{
    if proof.layers.len() != n_vars {
        return Err(LogUpError::LayerCountMismatch {
            expected: n_vars,
            got: proof.layers.len(),
        });
    }
    let [n0_top, n1_top, d0_top, d1_top] = top;
    // Aggregate via `(a/b) + (c/d) = (a·d + c·b) / (b·d)`.
    let top_num_check = n0_top * d1_top + n1_top * d0_top;
    if top_num_check != expected_top_numerator {
        return Err(LogUpError::TopFractionMismatch);
    }

    sponge.absorb(&n0_top);
    sponge.absorb(&n1_top);
    sponge.absorb(&d0_top);
    sponge.absorb(&d1_top);
    let beta = sponge.squeeze_field_elements::<F>(1)[0];
    let lambda = sponge.squeeze_field_elements::<F>(1)[0];

    let mut current_point: Vec<F> = vec![beta];
    let mut current_num = (F::one() - beta) * n0_top + beta * n1_top;
    let mut current_denom = (F::one() - beta) * d0_top + beta * d1_top;

    for (layer_idx, lp) in proof.layers.iter().enumerate() {
        let nv = current_point.len();
        if lp.rounds.len() != nv {
            return Err(LogUpError::LayerCountMismatch {
                expected: nv,
                got: lp.rounds.len(),
            });
        }
        let mut claim = current_num + lambda * current_denom;
        let mut new_point = Vec::with_capacity(nv);
        for round in &lp.rounds {
            if round.at_zero + round.at_one != claim {
                return Err(LogUpError::SumcheckSplitMismatch { layer: layer_idx });
            }
            let r = absorb_round_poly(sponge, round);
            claim = round.evaluate(r);
            new_point.push(r);
        }
        // Layer-end identity: the last-round claim must equal
        // `eq(new_point, current_point) · [n_lo·d_hi + n_hi·d_lo + λ·d_lo·d_hi]`.
        let [nl, nh, dl, dh] = lp.final_evals;
        let eq_val = eval_eq(&current_point, &new_point);
        let expected = eq_val * (nl * dh + nh * dl + lambda * dl * dh);
        if expected != claim {
            return Err(LogUpError::SumcheckFinalMismatch { layer: layer_idx });
        }

        let beta_next = sponge.squeeze_field_elements::<F>(1)[0];
        let mut next_point = Vec::with_capacity(nv + 1);
        next_point.push(beta_next);
        next_point.extend_from_slice(&new_point);
        current_point = next_point;
        current_num = (F::one() - beta_next) * nl + beta_next * nh;
        current_denom = (F::one() - beta_next) * dl + beta_next * dh;
    }

    if proof.bottom_num != current_num || proof.bottom_denom != current_denom {
        return Err(LogUpError::SumcheckFinalMismatch {
            layer: n_vars.saturating_sub(1),
        });
    }
    if proof.bottom_point != current_point {
        return Err(LogUpError::SumcheckFinalMismatch {
            layer: n_vars.saturating_sub(1),
        });
    }
    Ok(())
}

/// Evaluate the multilinear `eq` polynomial:
/// `eq(a, b) = ∏_i (a_i·b_i + (1-a_i)·(1-b_i))`.
fn eval_eq<F: Field>(a: &[F], b: &[F]) -> F {
    debug_assert_eq!(a.len(), b.len());
    let one = F::one();
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| *ai * *bi + (one - *ai) * (one - *bi))
        .product()
}
