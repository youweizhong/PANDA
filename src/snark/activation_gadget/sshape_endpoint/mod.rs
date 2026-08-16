//! Sigmoid/tanh relaxation-line endpoint validity (Phase 3b of the
//! PANDA SNARK).
//!
//! For each (sigmoid/tanh layer, line direction, preact endpoint),
//! the gadget proves `U(x) ≥ σ_upper(x)` on the upper line (and the
//! dual `σ_lower(x) ≥ L(x)` on the lower line) at one preact endpoint
//! `x ∈ {l[j], u[j]}`. Negative-x envelope values are recovered from
//! the half-table via the σ/tanh symmetries.
//!
//! The single generic [`prove_sshape_at_endpoint`] /
//! [`verify_sshape_at_endpoint`] pair handles all four
//! (line × endpoint) combinations. Tags for the line direction and
//! the endpoint side are absorbed into the FS sponge so a proof of
//! one configuration cannot be replayed under another. Convenience
//! wrappers below cover each direction by name.

mod envelope_logup;
mod prover;
#[cfg(test)]
mod tests;
mod types;
mod verifier;
mod witness;

pub use prover::prove_sshape_at_endpoint;
pub use types::{SshapeEndpointKind, SshapeEndpointProof, SshapeLineKind};
pub use verifier::verify_sshape_at_endpoint;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_std::rand::RngCore;

use crate::crown::network::{ActivationKind, Layer, Network};
use crate::quantization::scale::Scale;
use crate::quantized_crown::QuantCert;
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use crate::snark::commitment::commit::CommittedAux;
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;
use crate::snark::preprocess::SigmaTables;

/// Crypto-free replica of the endpoint range checks the prover runs at the
/// top of [`prove_sshape_at_endpoint`] (scale preconditions + `check_pos` on
/// every cell's `abs_l` / rems / `diff`), for one `(kind, line, preact)`
/// triple. Reuses the exact [`witness::compute_witnesses`] the prover uses, so
/// the accept/reject decision cannot drift from the real prover. Returns
/// `Ok(())` iff the prover would not reject this triple.
#[allow(clippy::too_many_arguments)]
fn endpoint_cells_in_range(
    pre: &SigmaTables,
    kind: ActivationKind,
    line: SshapeLineKind,
    preact_codes: &[i128],
    d_codes: &[i128],
    b_codes: &[i128],
    s_d: Scale,
    s_b: Scale,
    s_w: Scale,
    range_bits: usize,
) -> Result<(), SnarkError> {
    if !witness::scale_precondition_holds(s_d, s_b, s_w, range_bits) {
        return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
            which: "sshape_endpoint: DENOM ≤ 2^GADGET_RANGE_BITS precondition",
        });
    }
    // Match the prover's scale-code derivation (prover.rs). Any non-pow2 scale
    // means "not provable" here rather than a panic in the bisection driver.
    let (e_d, e_b, e_w) = match (s_d.pow2_exponent(), s_b.pow2_exponent(), s_w.pow2_exponent()) {
        (Ok(d), Ok(b), Ok(w)) => (d, b, w),
        _ => {
            return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
                which: "sshape_endpoint: non-pow2 scale",
            })
        }
    };
    if pre.s_x_log2 < e_w {
        return Err(SnarkError::Reserved {
            what: "sshape_endpoint: s_w > s_x not yet supported",
        });
    }
    let s_d_code = 1i128 << e_d;
    let s_b_code = 1i128 << e_b;
    let s_w_code = 1i128 << e_w;
    let s_v_code = 1i128 << pre.s_v_log2;
    // n_padded == n: real rows are computed identically regardless of padding,
    // and the prover's padding rows are all-zero (always in range), so this
    // reproduces the prover's check_pos verdict exactly.
    let n = preact_codes.len();
    let cells = witness::compute_witnesses(
        pre, kind, line, preact_codes, d_codes, b_codes, s_d_code, s_b_code, s_w_code, s_v_code, n,
    )?;
    let bound = 1i128 << range_bits;
    for c in &cells {
        for v in [c.abs_l, c.dx_step_1_rem, c.dx_sigma_rem, c.b_sigma_rem, c.diff] {
            if v < 0 || v >= bound {
                return Err(SnarkError::RelaxationSoundnessFinalCheckFailed {
                    which: "sshape_endpoint: diff out of range",
                });
            }
        }
    }
    Ok(())
}

/// Would the sigmoid/tanh endpoint gadgets in `prove_final_pass` accept this
/// quantized cert at the given output-bound range budget? Checks all four
/// `(line, endpoint)` inequalities on every sigmoid/tanh layer using only
/// cert data (relaxation `d`/`b` codes, preact endpoint codes, layer scales)
/// — no cryptography. `Ok(())` means every endpoint's split-arith witnesses
/// are in range, i.e. the prover will not reject with `RelaxationSoundness*`
/// for the endpoint gadgets when proving at `range_bits`.
pub(crate) fn quant_cert_sshape_endpoints_ok(
    network: &Network,
    cert: &QuantCert,
    range_bits: usize,
    sigma_x_scale_log2: i32,
    sigma_v_scale_log2: i32,
) -> Result<(), SnarkError> {
    let pre = SigmaTables::shared(sigma_x_scale_log2, sigma_v_scale_log2);
    let s_w = cert.scales.working;
    for (li, layer) in network.layers().iter().enumerate() {
        let kind = match layer {
            Layer::Activation {
                kind: k @ (ActivationKind::Sigmoid | ActivationKind::Tanh),
            } => *k,
            _ => continue,
        };
        let rel = cert.relaxations.get(li).and_then(|r| r.as_ref()).ok_or(
            SnarkError::ShapeMismatch {
                what: "provable check: missing sigmoid/tanh relaxation",
            },
        )?;
        // The preceding Linear layer's preact bounds (at the working scale).
        let lin = li.checked_sub(1).ok_or(SnarkError::ArchitectureMismatch {
            what: "sigmoid/tanh activation has no preceding Linear layer",
        })?;
        let pre_l = cert.preact_lower.get(lin).and_then(|p| p.as_ref()).ok_or(
            SnarkError::ShapeMismatch {
                what: "provable check: missing preact_lower for preceding Linear",
            },
        )?;
        let pre_u = cert.preact_upper.get(lin).and_then(|p| p.as_ref()).ok_or(
            SnarkError::ShapeMismatch {
                what: "provable check: missing preact_upper for preceding Linear",
            },
        )?;
        let pl = pre_l.codes.as_slice().expect("contiguous preact_lower");
        let pu = pre_u.codes.as_slice().expect("contiguous preact_upper");
        let du = rel.d_upper.codes.as_slice().expect("contiguous d_upper");
        let bu = rel.b_upper.codes.as_slice().expect("contiguous b_upper");
        let dl = rel.d_lower.codes.as_slice().expect("contiguous d_lower");
        let bl = rel.b_lower.codes.as_slice().expect("contiguous b_lower");
        let (sdu, sbu) = (rel.d_upper.scale, rel.b_upper.scale);
        let (sdl, sbl) = (rel.d_lower.scale, rel.b_lower.scale);
        // Upper line at both endpoints; lower line at both endpoints.
        let rb = range_bits;
        for (line, preact, d, b, sd, sb) in [
            (SshapeLineKind::Upper, pl, du, bu, sdu, sbu),
            (SshapeLineKind::Upper, pu, du, bu, sdu, sbu),
            (SshapeLineKind::Lower, pl, dl, bl, sdl, sbl),
            (SshapeLineKind::Lower, pu, dl, bl, sdl, sbl),
        ] {
            endpoint_cells_in_range(&pre, kind, line, preact, d, b, sd, sb, s_w, rb)?;
        }
    }
    Ok(())
}

/// Prove `U(l[j]) ≥ σ_upper(l[j])`.
#[allow(clippy::too_many_arguments)]
pub fn prove_sshape_upper_at_lower(
    layer_idx: usize,
    kind: ActivationKind,
    preact_lower_codes: &[i128],
    preact_aux: &CommittedAux,
    preact_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_upper_aux: &CommittedAux,
    d_upper_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_upper_aux: &CommittedAux,
    b_upper_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s_d: Scale,
    s_b: Scale,
    s_w: Scale,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<SshapeEndpointProof, SnarkError> {
    prove_sshape_at_endpoint(
        layer_idx,
        kind,
        SshapeLineKind::Upper,
        SshapeEndpointKind::Lower,
        preact_lower_codes,
        preact_aux,
        preact_commit,
        d_upper_aux,
        d_upper_commit,
        b_upper_aux,
        b_upper_commit,
        s_d,
        s_b,
        s_w,
        params,
        sponge,
        rng,
    )
}

/// Prove `U(u[j]) ≥ σ_upper(u[j])`.
#[allow(clippy::too_many_arguments)]
pub fn prove_sshape_upper_at_upper(
    layer_idx: usize,
    kind: ActivationKind,
    preact_upper_codes: &[i128],
    preact_aux: &CommittedAux,
    preact_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_upper_aux: &CommittedAux,
    d_upper_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_upper_aux: &CommittedAux,
    b_upper_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s_d: Scale,
    s_b: Scale,
    s_w: Scale,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<SshapeEndpointProof, SnarkError> {
    prove_sshape_at_endpoint(
        layer_idx,
        kind,
        SshapeLineKind::Upper,
        SshapeEndpointKind::Upper,
        preact_upper_codes,
        preact_aux,
        preact_commit,
        d_upper_aux,
        d_upper_commit,
        b_upper_aux,
        b_upper_commit,
        s_d,
        s_b,
        s_w,
        params,
        sponge,
        rng,
    )
}

/// Verify a proof of `U(l[j]) ≥ σ_upper(l[j])`.
#[allow(clippy::too_many_arguments)]
pub fn verify_sshape_upper_at_lower(
    proof: &SshapeEndpointProof,
    expected_layer_idx: usize,
    kind: ActivationKind,
    n_real_neurons: usize,
    preact_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_upper_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_upper_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s_d: Scale,
    s_b: Scale,
    s_w: Scale,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    verify_sshape_at_endpoint(
        proof,
        expected_layer_idx,
        kind,
        SshapeLineKind::Upper,
        SshapeEndpointKind::Lower,
        n_real_neurons,
        preact_commit,
        d_upper_commit,
        b_upper_commit,
        s_d,
        s_b,
        s_w,
        params,
        sponge,
    )
}

/// Verify a proof of `U(u[j]) ≥ σ_upper(u[j])`.
#[allow(clippy::too_many_arguments)]
pub fn verify_sshape_upper_at_upper(
    proof: &SshapeEndpointProof,
    expected_layer_idx: usize,
    kind: ActivationKind,
    n_real_neurons: usize,
    preact_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_upper_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_upper_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s_d: Scale,
    s_b: Scale,
    s_w: Scale,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    verify_sshape_at_endpoint(
        proof,
        expected_layer_idx,
        kind,
        SshapeLineKind::Upper,
        SshapeEndpointKind::Upper,
        n_real_neurons,
        preact_commit,
        d_upper_commit,
        b_upper_commit,
        s_d,
        s_b,
        s_w,
        params,
        sponge,
    )
}

/// Prove `σ_lower(l[j]) ≥ L(l[j])`.
#[allow(clippy::too_many_arguments)]
pub fn prove_sshape_lower_at_lower(
    layer_idx: usize,
    kind: ActivationKind,
    preact_lower_codes: &[i128],
    preact_aux: &CommittedAux,
    preact_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_lower_aux: &CommittedAux,
    d_lower_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_lower_aux: &CommittedAux,
    b_lower_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s_d: Scale,
    s_b: Scale,
    s_w: Scale,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<SshapeEndpointProof, SnarkError> {
    prove_sshape_at_endpoint(
        layer_idx,
        kind,
        SshapeLineKind::Lower,
        SshapeEndpointKind::Lower,
        preact_lower_codes,
        preact_aux,
        preact_commit,
        d_lower_aux,
        d_lower_commit,
        b_lower_aux,
        b_lower_commit,
        s_d,
        s_b,
        s_w,
        params,
        sponge,
        rng,
    )
}

/// Prove `σ_lower(u[j]) ≥ L(u[j])`.
#[allow(clippy::too_many_arguments)]
pub fn prove_sshape_lower_at_upper(
    layer_idx: usize,
    kind: ActivationKind,
    preact_upper_codes: &[i128],
    preact_aux: &CommittedAux,
    preact_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_lower_aux: &CommittedAux,
    d_lower_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_lower_aux: &CommittedAux,
    b_lower_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s_d: Scale,
    s_b: Scale,
    s_w: Scale,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<SshapeEndpointProof, SnarkError> {
    prove_sshape_at_endpoint(
        layer_idx,
        kind,
        SshapeLineKind::Lower,
        SshapeEndpointKind::Upper,
        preact_upper_codes,
        preact_aux,
        preact_commit,
        d_lower_aux,
        d_lower_commit,
        b_lower_aux,
        b_lower_commit,
        s_d,
        s_b,
        s_w,
        params,
        sponge,
        rng,
    )
}

/// Verify a proof of `σ_lower(l[j]) ≥ L(l[j])`.
#[allow(clippy::too_many_arguments)]
pub fn verify_sshape_lower_at_lower(
    proof: &SshapeEndpointProof,
    expected_layer_idx: usize,
    kind: ActivationKind,
    n_real_neurons: usize,
    preact_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_lower_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_lower_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s_d: Scale,
    s_b: Scale,
    s_w: Scale,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    verify_sshape_at_endpoint(
        proof,
        expected_layer_idx,
        kind,
        SshapeLineKind::Lower,
        SshapeEndpointKind::Lower,
        n_real_neurons,
        preact_commit,
        d_lower_commit,
        b_lower_commit,
        s_d,
        s_b,
        s_w,
        params,
        sponge,
    )
}

/// Verify a proof of `σ_lower(u[j]) ≥ L(u[j])`.
#[allow(clippy::too_many_arguments)]
pub fn verify_sshape_lower_at_upper(
    proof: &SshapeEndpointProof,
    expected_layer_idx: usize,
    kind: ActivationKind,
    n_real_neurons: usize,
    preact_commit: &<HyraxBn254 as MlPcs>::Commitment,
    d_lower_commit: &<HyraxBn254 as MlPcs>::Commitment,
    b_lower_commit: &<HyraxBn254 as MlPcs>::Commitment,
    s_d: Scale,
    s_b: Scale,
    s_w: Scale,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    verify_sshape_at_endpoint(
        proof,
        expected_layer_idx,
        kind,
        SshapeLineKind::Lower,
        SshapeEndpointKind::Upper,
        n_real_neurons,
        preact_commit,
        d_lower_commit,
        b_lower_commit,
        s_d,
        s_b,
        s_w,
        params,
        sponge,
    )
}
