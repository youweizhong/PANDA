//! Proof type for the Phase 3c critical-point gadget.

use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::snark_primitives::logup_gkr::LogUpProof;
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};
use crate::snark_primitives::sumcheck::RoundPoly4;

use super::super::relu_upper_endpoint::PosRangeLogUp;
use super::super::sshape_endpoint::SshapeLineKind;

/// Per-(layer, line) sub-proof for the finite-difference critical-point
/// check. See `mod.rs` for the high-level role; field comments describe
/// the individual witness commits, range proofs, σ-envelope LogUp, and
/// combined-sumcheck artefacts.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SshapeCriticalPointProof {
    pub layer_idx: usize,
    /// `0 = sigmoid`, `1 = tanh`.
    pub kind_tag: u8,
    /// `0 = upper line`, `1 = lower line`. Bound into the FS sponge so
    /// an upper-line proof can't be replayed as a lower-line proof.
    pub line_tag: u8,
    pub n_vars: usize,
    /// Real cell count. `n_padded = 1 << n_vars` can exceed `n_real`;
    /// padding rows are masked by the public `is_real` MLE that the
    /// verifier rebuilds and evaluates at `r_final`.
    pub n_real: usize,
    pub z_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub sigma_lo_z_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub sigma_up_z_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub sigma_lo_zmd_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub sigma_up_zmd_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub sigma_lo_zpd_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub sigma_up_zpd_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub slack_fd1_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub slack_fd2_commit: <HyraxBn254 as MlPcs>::Commitment,
    /// `factor_a = (u − z)` (upper line) or `((−z) − l)` (lower line).
    pub factor_a_commit: <HyraxBn254 as MlPcs>::Commitment,
    /// Line-σ gap at the chosen critical point, signed, at scale `s_v`.
    pub factor_b_commit: <HyraxBn254 as MlPcs>::Commitment,
    /// Split-arithmetic intermediates feeding `factor_b` at scale `s_v`.
    /// Each remainder is bounded by one of `{s_d, s_b, s_w}` separately.
    ///     d·z = dz_step_1·s_d + dz_step_1_rem
    ///     dz_step_1·s_v = dz_sigma_code·s_w + dz_sigma_rem
    ///     Upper: b·s_v = b_sigma_code·s_b + b_sigma_rem
    ///     Lower: b_sigma_code·s_b − b·s_v = b_sigma_rem
    pub dz_step_1_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub dz_step_1_rem_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub dz_sigma_code_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub dz_sigma_rem_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub b_sigma_code_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub b_sigma_rem_commit: <HyraxBn254 as MlPcs>::Commitment,
    /// `is_active ∈ {0, 1}` — 1 iff `d != 0`. Gates the FD identities
    /// so they are not enforced for degenerate `d = 0` cells.
    pub is_active_commit: <HyraxBn254 as MlPcs>::Commitment,
    /// `inside_bit ∈ {0, 1}` — 1 iff `factor_a (= delta) ≥ 0`.
    pub inside_commit: <HyraxBn254 as MlPcs>::Commitment,
    /// `slack_pos = inside·delta + (1−inside)·(−delta − 1)`. Combined
    /// with booleanity + range `slack_pos ≥ 0`, this binds
    /// `inside_bit` to the sign of `delta`.
    pub slack_pos_commit: <HyraxBn254 as MlPcs>::Commitment,
    /// `gated_gap = inside_bit · factor_b`. Range-checked `≥ 0`: the
    /// line-σ gap must be non-negative only when the critical point
    /// lies inside `[l, u]`.
    pub gated_gap_commit: <HyraxBn254 as MlPcs>::Commitment,
    /// Range check on `z`. Also covers the chunked halves of
    /// `slack_fd1`, `slack_fd2`, `slack_pos`, and `gated_gap` via the
    /// neighbouring `*_range` fields.
    pub z_range: PosRangeLogUp,
    /// Slack chunking: `slack_fdN = slack_fdN_high · 2¹⁹ + slack_fdN_low`
    /// with both halves in `[0, 2¹⁹)`.
    pub slack_fd1_high_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub slack_fd1_low_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub slack_fd2_high_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub slack_fd2_low_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub slack_fd1_high_range: PosRangeLogUp,
    pub slack_fd1_low_range: PosRangeLogUp,
    pub slack_fd2_high_range: PosRangeLogUp,
    pub slack_fd2_low_range: PosRangeLogUp,
    /// Chunked-range commits and proofs for `slack_pos` and `gated_gap`
    /// at base `2^19`.
    pub slack_pos_high_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub slack_pos_low_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub slack_pos_high_range: PosRangeLogUp,
    pub slack_pos_low_range: PosRangeLogUp,
    pub gated_gap_high_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub gated_gap_low_commit: <HyraxBn254 as MlPcs>::Commitment,
    pub gated_gap_high_range: PosRangeLogUp,
    pub gated_gap_low_range: PosRangeLogUp,
    /// Range checks (`≥ 0`) on the three split-arith remainders feeding
    /// `factor_b`.
    pub dz_step_1_rem_range: PosRangeLogUp,
    pub dz_sigma_rem_range: PosRangeLogUp,
    pub b_sigma_rem_range: PosRangeLogUp,
    /// 3-column σ-envelope LogUp: rows are `(x, σ_lo(x), σ_up(x))` for
    /// `x ∈ {z, z−δ, z+δ}` (and a padding row), looked up against the
    /// public Phase 3a half-table.
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
    /// Six σ evals (lo/up at z, z−δ, z+δ) plus `z` itself, bound in one
    /// batched Hyrax open at the LogUp witness bottom point.
    pub envelope_sigma_lo_z_eval: Fr,
    pub envelope_sigma_up_z_eval: Fr,
    pub envelope_sigma_lo_zmd_eval: Fr,
    pub envelope_sigma_up_zmd_eval: Fr,
    pub envelope_sigma_lo_zpd_eval: Fr,
    pub envelope_sigma_up_zpd_eval: Fr,
    pub envelope_z_eval: Fr,
    pub envelope_witness_batched_open: <HyraxBn254 as MlPcs>::Proof,
    /// Combined sumcheck: 16 identities folded with 15 ρ challenges
    /// (FD1 carries an implicit coefficient 1).
    pub combined_rho_a: Fr,
    pub combined_rho_b: Fr,
    pub combined_rho_c: Fr,
    pub combined_rho_d: Fr,
    pub combined_rho_e: Fr,
    pub combined_rho_f: Fr,
    pub combined_rho_g: Fr,
    pub combined_rho_h: Fr,
    pub combined_rho_i: Fr,
    pub combined_rho_j: Fr,
    pub combined_rho_k: Fr,
    pub combined_rho_l: Fr,
    pub combined_rho_m: Fr,
    pub combined_rho_n: Fr,
    pub combined_rho_o: Fr,
    pub r_test: Vec<Fr>,
    pub rounds: Vec<RoundPoly4<Fr>>,
    pub r_final: Vec<Fr>,
    pub d_eval: Fr,
    pub b_eval: Fr,
    pub preact_l_eval: Fr,
    pub preact_u_eval: Fr,
    pub z_eval: Fr,
    pub sigma_lo_z_eval: Fr,
    pub sigma_up_z_eval: Fr,
    pub sigma_lo_zmd_eval: Fr,
    pub sigma_up_zmd_eval: Fr,
    pub sigma_lo_zpd_eval: Fr,
    pub sigma_up_zpd_eval: Fr,
    pub slack_fd1_eval: Fr,
    pub slack_fd2_eval: Fr,
    pub factor_a_eval: Fr,
    pub factor_b_eval: Fr,
    pub dz_step_1_eval: Fr,
    pub dz_step_1_rem_eval: Fr,
    pub dz_sigma_code_eval: Fr,
    pub dz_sigma_rem_eval: Fr,
    pub b_sigma_code_eval: Fr,
    pub b_sigma_rem_eval: Fr,
    pub slack_fd1_high_eval: Fr,
    pub slack_fd1_low_eval: Fr,
    pub slack_fd2_high_eval: Fr,
    pub slack_fd2_low_eval: Fr,
    pub is_active_eval: Fr,
    pub inside_eval: Fr,
    pub slack_pos_eval: Fr,
    pub slack_pos_high_eval: Fr,
    pub slack_pos_low_eval: Fr,
    pub gated_gap_eval: Fr,
    pub gated_gap_high_eval: Fr,
    pub gated_gap_low_eval: Fr,
    pub batched_open_at_r: <HyraxBn254 as MlPcs>::Proof,
    /// Hyrax opens of the hidden-pass preact commits at this gadget's
    /// own `r_final`. The verifier consumes the opened evals in the
    /// `factor_a` identity instead of reading raw preact codes.
    pub preact_l_open_at_r_final: <HyraxBn254 as MlPcs>::Proof,
    pub preact_u_open_at_r_final: <HyraxBn254 as MlPcs>::Proof,
}

impl SshapeCriticalPointProof {
    pub fn line(&self) -> SshapeLineKind {
        match self.line_tag {
            0 => SshapeLineKind::Upper,
            _ => SshapeLineKind::Lower,
        }
    }
}
