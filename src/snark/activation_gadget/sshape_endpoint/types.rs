//! Proof types for the endpoint gadget: line/endpoint tag enums and
//! the per-direction `SshapeEndpointProof` carrier.

use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::snark_primitives::logup_gkr::LogUpProof;
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};
use crate::snark_primitives::sumcheck::RoundPoly4;

use super::super::relu_upper_endpoint::PosRangeLogUp;

/// Which preact endpoint a proof covers (lower or upper). Bound into
/// the FS sponge so a proof at one endpoint can't be replayed at the
/// other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SshapeEndpointKind {
    Lower,
    Upper,
}

impl SshapeEndpointKind {
    pub fn tag(self) -> u8 {
        match self {
            Self::Lower => 0,
            Self::Upper => 1,
        }
    }
}

/// Which relaxation line a proof covers. Upper proves
/// `U(x) ≥ σ_upper(x)`; lower proves `σ_lower(x) ≥ L(x)`. Bound into
/// the FS sponge so the two directions can't be replayed against
/// each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SshapeLineKind {
    Upper,
    Lower,
}

impl SshapeLineKind {
    pub fn tag(self) -> u8 {
        match self {
            Self::Upper => 0,
            Self::Lower => 1,
        }
    }
}

/// Per-(line, endpoint) sub-proof.
///
/// The line value at scale `s_v` is reconstructed from a split-arith
/// chain:
///
/// ```text
///     d · x = dx_step_1 · s_d + dx_step_1_rem
///     dx_step_1 · s_v = dx_sigma_code · s_w + dx_sigma_rem
///     b · s_v = b_sigma_code · s_b + b_sigma_rem
///     line_sigma_code = dx_sigma_code + b_sigma_code
///     diff = line_sigma_code − σ_used   (upper line, ≥ 0)
///     diff = σ_used − line_sigma_code   (lower line, ≥ 0)
/// ```
///
/// Each remainder is range-checked `≥ 0`; the tighter `< scale`
/// bound is implied by `diff ≥ 0`.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SshapeEndpointProof {
    pub layer_idx: usize,
    /// `0 = sigmoid, 1 = tanh`.
    pub kind_tag: u8,
    /// `0 = preact lower endpoint l[j]`, `1 = preact upper endpoint u[j]`.
    pub endpoint_tag: u8,
    /// `0 = upper line, 1 = lower line`.
    pub line_tag: u8,
    pub n_vars: usize,
    /// Real cell count. The verifier reconstructs the public `is_real`
    /// MLE from `(n_real, n_vars)` and evaluates it at `r_final`.
    pub n_real: usize,
    pub abs_l_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub sign_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub sigma_upper_at_abs_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub sigma_lower_at_abs_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub dx_step_1_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub dx_step_1_rem_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub dx_sigma_code_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub dx_sigma_rem_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub b_sigma_code_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub b_sigma_rem_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub diff_commit: <HyraxBn254 as MlPcs>::Commitment,
    /// Range checks (`≥ 0`) on `abs_l`, the three split-arith
    /// remainders, and `diff`.
    pub abs_l_range: PosRangeLogUp,
    pub dx_step_1_rem_range: PosRangeLogUp,
    pub dx_sigma_rem_range: PosRangeLogUp,
    pub b_sigma_rem_range: PosRangeLogUp,
    pub diff_range: PosRangeLogUp,
    /// 3-column σ-envelope LogUp with combined key
    /// `α₁ · abs_l + α₂ · σ_upper_at_abs + σ_lower_at_abs`.
    pub envelope_combine_alpha_1: Fr,
    pub envelope_combine_alpha_2: Fr,
    pub envelope_logup_beta: Fr,
    pub envelope_lookup_proof: LogUpProof<Fr>,
    pub envelope_table_proof: LogUpProof<Fr>,
    pub envelope_lookup_top: [Fr; 4],
    pub envelope_table_top: [Fr; 4],
    pub envelope_witness_len: usize,
    pub envelope_table_len: usize,
    pub envelope_mult_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub envelope_mult_open: <HyraxBn254 as MlPcs>::Proof,
    pub envelope_mult_n_vars: usize,
    /// `(abs_l, σ_upper_at_abs, σ_lower_at_abs)` evals and one batched
    /// open at the LogUp witness bottom point.
    pub envelope_abs_l_eval: Fr,
    pub envelope_sigma_upper_at_abs_eval: Fr,
    pub envelope_sigma_lower_at_abs_eval: Fr,
    pub envelope_witness_batched_open: <HyraxBn254 as MlPcs>::Proof,
    /// Combined sumcheck folds six identities with FS-derived
    /// coefficients `(ρ_a..ρ_e)`.
    pub combined_rho_a: Fr,
    pub combined_rho_b: Fr,
    pub combined_rho_c: Fr,
    pub combined_rho_d: Fr,
    pub combined_rho_e: Fr,
    pub r_test: Vec<Fr>,
    pub rounds: Vec<RoundPoly4<Fr>>,
    pub r_final: Vec<Fr>,
    pub d_line_eval: Fr,
    pub b_line_eval: Fr,
    pub abs_l_eval: Fr,
    pub sign_eval: Fr,
    pub sigma_upper_at_abs_eval: Fr,
    pub sigma_lower_at_abs_eval: Fr,
    pub dx_step_1_eval: Fr,
    pub dx_step_1_rem_eval: Fr,
    pub dx_sigma_code_eval: Fr,
    pub dx_sigma_rem_eval: Fr,
    pub b_sigma_code_eval: Fr,
    pub b_sigma_rem_eval: Fr,
    pub diff_eval: Fr,
    pub batched_open_at_r: <HyraxBn254 as MlPcs>::Proof,
    /// Hyrax open of the hidden-pass preact commit at this gadget's
    /// `r_final`; the opened eval replaces the previous raw-codes
    /// MLE evaluation in the final identity.
    pub preact_eval_at_r_final: Fr,
    pub preact_open_at_r_final: <HyraxBn254 as MlPcs>::Proof,
}

#[allow(dead_code)]
pub type SshapeUpperEndpointProof = SshapeEndpointProof;

#[allow(dead_code)]
pub type SshapeUpperLowerEndpointProof = SshapeEndpointProof;
