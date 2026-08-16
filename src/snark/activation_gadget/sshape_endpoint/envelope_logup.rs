//! Shared σ-envelope LogUp helpers: top-fraction extractor and the
//! 3-column canonical-table MLE evaluation.

use ark_bn254::Fr;

use crate::snark_primitives::logup_gkr::LogUpCircuit;

use crate::snark::commitment::multilinear_extensions::eval_multilinear_full;
use crate::snark::commitment::table_mle::identity_mle_eval;

/// Top-layer halves of a LogUp circuit.
pub(crate) fn top_halves_logup(circuit: &LogUpCircuit<Fr>) -> [Fr; 4] {
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

/// MLE eval of the 3-column σ-envelope canonical table at `point`:
/// `α₁ · identity_mle(point) + α₂ · σ_upper_MLE(point) + σ_lower_MLE(point)`.
pub(crate) fn sigma_envelope_3col_table_mle_eval(
    point: &[Fr],
    alpha_1: Fr,
    alpha_2: Fr,
    sigma_upper_table: &[Fr],
    sigma_lower_table: &[Fr],
) -> Fr {
    debug_assert_eq!(sigma_upper_table.len(), sigma_lower_table.len());
    debug_assert_eq!(sigma_upper_table.len(), 1usize << point.len());
    alpha_1 * identity_mle_eval(point)
        + alpha_2 * eval_multilinear_full(sigma_upper_table, point)
        + eval_multilinear_full(sigma_lower_table, point)
}
