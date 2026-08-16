//! Prover-side helpers used by [`super::prove_final_pass`]:
//! direction-specific wrappers for the output-bound inequality, the
//! per-tensor range LogUp loop, and the per-layer scales packing /
//! commit / open routines (plus their verifier counterparts).

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_std::rand::RngCore;

use crate::crown::network::{Layer, Network};
use crate::quantized_crown::BoundDir;
use crate::snark_primitives::finite_field::signed_lift_to_fr;

use crate::snark::commitment::commit::{native_vector_n_vars, ProverPolyStates, TensorCommitments};
use crate::snark::commitment::range_per_tensor::{prove_tensor_range_logup, TensorRangeProof};
use crate::snark::errors::SnarkError;
use crate::snark::output_bound::{prove_output_bound_inequality, OutputBoundIneqProof};
use crate::snark::params::SnarkParams;

/// Run `prove_output_bound_inequality` for one direction. Returns
/// `None` if the pass isn't requested (one-sided properties).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_output_bound_inequality(
    cert: &crate::quantized_crown::QuantCert,
    commitments: &TensorCommitments,
    prover_states: &ProverPolyStates,
    direction: BoundDir,
    // Public threshold codes (working scale), padded to
    // `1 << native_vector_n_vars(n_spec)`.
    threshold_codes_padded: &[i128],
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut impl RngCore,
) -> Result<Option<OutputBoundIneqProof>, SnarkError> {
    let _timing = crate::timing::scope("ob_final");
    let (claim, pass_com, pass_st) = match direction {
        BoundDir::Lower => (
            cert.target_lower.as_ref(),
            commitments.pass_lower.as_ref(),
            prover_states.pass_lower.as_ref(),
        ),
        BoundDir::Upper => (
            cert.target_upper.as_ref(),
            commitments.pass_upper.as_ref(),
            prover_states.pass_upper.as_ref(),
        ),
    };
    let (claim, pass_com, pass_st) = match (claim, pass_com, pass_st) {
        (Some(c), Some(pc), Some(ps)) => (c, pc, ps),
        _ => return Ok(None),
    };
    let acc_w_aux = match pass_st.concretize_acc_w.as_ref() {
        Some(a) => a,
        None => return Ok(None),
    };
    let acc_w_com = match pass_com.concretize_acc_w.as_ref() {
        Some(c) => c,
        None => return Ok(None),
    };
    let b_acc_final_aux = pass_st.chain_b_acc.first().expect("chain_b_acc non-empty");
    let b_acc_final_com = pass_com.chain_b_acc.first().expect("chain_b_acc non-empty");

    // n_vars matches the verifier's native vector sizing.
    let n_spec = claim.codes.len();
    let n_vars = native_vector_n_vars(n_spec);
    let n_padded = 1usize << n_vars;

    let mut claimed_codes = vec![0i128; n_padded];
    for (slot, &c) in claimed_codes.iter_mut().zip(claim.codes.iter()) {
        *slot = c;
    }

    // Decode b_acc_final / acc_w codes from their padded MLEs.
    let b_acc_final_codes: Vec<i128> = b_acc_final_aux
        .0
        .iter()
        .take(n_padded)
        .map(|f| crate::snark_primitives::finite_field::fr_to_signed_i128(*f).unwrap_or(0))
        .collect();
    let acc_w_codes: Vec<i128> = acc_w_aux
        .0
        .iter()
        .take(n_padded)
        .map(|f| crate::snark_primitives::finite_field::fr_to_signed_i128(*f).unwrap_or(0))
        .collect();

    if threshold_codes_padded.len() != n_padded {
        return Err(SnarkError::ShapeMismatch {
            what: "build_output_bound_inequality: threshold padded length mismatch",
        });
    }
    // Final pass: the output margin (c·y − d) can be large for very
    // robust properties, so this is the one place the wide
    // out_bound_range_bits window is used.
    let proof = prove_output_bound_inequality(
        direction,
        params.out_bound_range_bits,
        n_vars,
        &claimed_codes,
        &b_acc_final_codes,
        &acc_w_codes,
        Some(threshold_codes_padded),
        b_acc_final_aux,
        b_acc_final_com,
        acc_w_aux,
        acc_w_com,
        // Final pass: gadget creates its own claimed commit.
        None,
        params,
        sponge,
        rng,
    )?;
    let _ = signed_lift_to_fr;
    Ok(Some(proof))
}

/// Walk every range-checked public-witness tensor in canonical
/// order and emit a per-tensor range LogUp. The verifier walks the
/// same order to pair each `TensorRangeProof` with its commitment.
///
/// Order: `x_lower`, `x_upper`, then for each layer (network order)
/// `weight, bias` for Linear or `d_lower, d_upper, b_lower, b_upper`
/// for Activation.
pub(super) fn build_tensor_range_proofs(
    network: &Network,
    commitments: &TensorCommitments,
    prover_states: &ProverPolyStates,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<Vec<TensorRangeProof>, SnarkError> {
    let _timing = crate::timing::scope("tensor_range");
    let mut out: Vec<TensorRangeProof> = Vec::new();
    out.push(prove_tensor_range_logup(
        &prover_states.x_lower,
        &commitments.x_lower,
        params,
        sponge,
        rng,
    )?);
    out.push(prove_tensor_range_logup(
        &prover_states.x_upper,
        &commitments.x_upper,
        params,
        sponge,
        rng,
    )?);
    for (i, layer) in network.layers().iter().enumerate() {
        match layer {
            Layer::Linear { .. } => {
                let w_aux = prover_states.weight[i].as_ref().expect("linear weight aux");
                let w_com = commitments.weight[i]
                    .as_ref()
                    .expect("linear weight commit");
                out.push(prove_tensor_range_logup(w_aux, w_com, params, sponge, rng)?);
                let b_aux = prover_states.bias[i].as_ref().expect("linear bias aux");
                let b_com = commitments.bias[i].as_ref().expect("linear bias commit");
                out.push(prove_tensor_range_logup(b_aux, b_com, params, sponge, rng)?);
            }
            Layer::Activation { .. } => {
                let rs = prover_states.relaxation[i]
                    .as_ref()
                    .expect("activation relax states");
                let rc = commitments.relaxation[i]
                    .as_ref()
                    .expect("activation relax commits");
                out.push(prove_tensor_range_logup(
                    &rs.d_lower,
                    &rc.d_lower,
                    params,
                    sponge,
                    rng,
                )?);
                out.push(prove_tensor_range_logup(
                    &rs.d_upper,
                    &rc.d_upper,
                    params,
                    sponge,
                    rng,
                )?);
                out.push(prove_tensor_range_logup(
                    &rs.b_lower,
                    &rc.b_lower,
                    params,
                    sponge,
                    rng,
                )?);
                out.push(prove_tensor_range_logup(
                    &rs.b_upper,
                    &rc.b_upper,
                    params,
                    sponge,
                    rng,
                )?);
            }
        }
    }
    Ok(out)
}

/// Build the per-layer scale record from the cert. Linear layers
/// fill `weight`/`bias` (sentinel for relaxation); Activation layers
/// fill `relax_d`/`relax_b` (sentinel for weight/bias).
pub(super) fn build_layer_scales_commit(
    network: &Network,
    cert: &crate::quantized_crown::QuantCert,
) -> crate::snark::proof::LayerScalesCommit {
    let n = network.layers().len();
    let mut weight_c = vec![0i64; n];
    let mut weight_e = vec![0i32; n];
    let mut bias_c = vec![0i64; n];
    let mut bias_e = vec![0i32; n];
    let mut relax_d_c = vec![0i64; n];
    let mut relax_d_e = vec![0i32; n];
    let mut relax_b_c = vec![0i64; n];
    let mut relax_b_e = vec![0i32; n];
    for (i, layer) in network.layers().iter().enumerate() {
        match layer {
            Layer::Linear { .. } => {
                let s = &cert.scales.layers[i];
                let w = s.weight.expect("linear layer must have a weight scale");
                let b = s.bias.expect("linear layer must have a bias scale");
                weight_c[i] = w.c;
                weight_e[i] = w.e;
                bias_c[i] = b.c;
                bias_e[i] = b.e;
            }
            Layer::Activation { .. } => {
                let s = &cert.scales.layers[i];
                let d = s
                    .relax_d
                    .expect("activation layer must have a relax_d scale");
                let b = s
                    .relax_b
                    .expect("activation layer must have a relax_b scale");
                relax_d_c[i] = d.c;
                relax_d_e[i] = d.e;
                relax_b_c[i] = b.c;
                relax_b_e[i] = b.e;
            }
        }
    }
    crate::snark::proof::LayerScalesCommit {
        weight_c,
        weight_e,
        bias_c,
        bias_e,
        relax_d_c,
        relax_d_e,
        relax_b_c,
        relax_b_e,
    }
}

/// Pack a `LayerScalesCommit` into a single Fr column in canonical
/// `[weight_c, weight_e, bias_c, bias_e, relax_d_c, relax_d_e,
/// relax_b_c, relax_b_e]` order, padded to an even-vars Hyrax length.
/// Returns the packed column and `n_vars`.
pub(in crate::snark) fn pack_layer_scales_to_fr(
    scales: &crate::snark::proof::LayerScalesCommit,
) -> (Vec<ark_bn254::Fr>, usize) {
    let n_layers = scales.weight_c.len();
    let raw_len = 8 * n_layers;
    // Power-of-two with even n_vars ≥ 2 (Hyrax requirement).
    let padded_len = raw_len.next_power_of_two().max(4);
    let mut n_vars = padded_len.trailing_zeros() as usize;
    if n_vars % 2 == 1 {
        n_vars += 1;
    }
    let padded_len = 1usize << n_vars;
    let mut packed: Vec<ark_bn254::Fr> = Vec::with_capacity(padded_len);
    let push_c = |out: &mut Vec<ark_bn254::Fr>, v: &[i64]| {
        for &x in v {
            out.push(signed_lift_to_fr(x as i128));
        }
    };
    let push_e = |out: &mut Vec<ark_bn254::Fr>, v: &[i32]| {
        for &x in v {
            out.push(signed_lift_to_fr(x as i128));
        }
    };
    push_c(&mut packed, &scales.weight_c);
    push_e(&mut packed, &scales.weight_e);
    push_c(&mut packed, &scales.bias_c);
    push_e(&mut packed, &scales.bias_e);
    push_c(&mut packed, &scales.relax_d_c);
    push_e(&mut packed, &scales.relax_d_e);
    push_c(&mut packed, &scales.relax_b_c);
    push_e(&mut packed, &scales.relax_b_e);
    packed.resize(padded_len, ark_bn254::Fr::from(0u64));
    (packed, n_vars)
}

/// Canonical packed-column index for a `(class, layer)` pair. Class
/// order must match `pack_layer_scales_to_fr`.
pub(in crate::snark) fn scale_packed_index(
    class: crate::snark::proof::ScaleClass,
    layer_idx: usize,
    n_layers: usize,
) -> usize {
    (class as usize) * n_layers + layer_idx
}

/// Big-endian unit-vector multilinear point selecting `idx` in a
/// `1 << n_vars` MLE.
pub(in crate::snark) fn unit_point_be(idx: usize, n_vars: usize) -> Vec<ark_bn254::Fr> {
    debug_assert!(idx < (1usize << n_vars));
    let mut out = Vec::with_capacity(n_vars);
    for bit in (0..n_vars).rev() {
        let b = ((idx >> bit) & 1) == 1;
        out.push(if b {
            ark_bn254::Fr::from(1u64)
        } else {
            ark_bn254::Fr::from(0u64)
        });
    }
    out
}

/// Per-layer Hyrax opens of the packed scales column. Linear layers
/// open `(weight, bias)`; Activation layers open `(relax_d, relax_b)`.
/// Each open lands at the unit-vector index of the corresponding
/// `(class, layer_idx)`; the verifier replays the same order.
#[allow(clippy::too_many_arguments)]
pub(in crate::snark) fn build_layer_scale_opens(
    network: &Network,
    packed_fr: &[ark_bn254::Fr],
    state: &<crate::snark_primitives::polynomial_commitment::HyraxBn254 as crate::snark_primitives::polynomial_commitment::MlPcs>::CommitmentState,
    commit: &<crate::snark_primitives::polynomial_commitment::HyraxBn254 as crate::snark_primitives::polynomial_commitment::MlPcs>::Commitment,
    n_vars: usize,
    n_layers: usize,
    params: &crate::snark::params::SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<crate::snark::proof::LayerScaleOpens, crate::snark::errors::SnarkError> {
    let _timing = crate::timing::scope("layer_scale_opens");
    use crate::snark::proof::{LayerScaleOpenCE, LayerScaleOpens, ScaleClass};

    use crate::snark::commitment::commit::CommittedAux;
    let aux: CommittedAux = (packed_fr.to_vec(), state.clone());
    let mut weight: Vec<Option<LayerScaleOpenCE>> = vec![None; n_layers];
    let mut bias: Vec<Option<LayerScaleOpenCE>> = vec![None; n_layers];
    let mut relax_d: Vec<Option<LayerScaleOpenCE>> = vec![None; n_layers];
    let mut relax_b: Vec<Option<LayerScaleOpenCE>> = vec![None; n_layers];

    fn open_pair<S: CryptographicSponge>(
        c_class: crate::snark::proof::ScaleClass,
        e_class: crate::snark::proof::ScaleClass,
        layer_idx: usize,
        n_layers: usize,
        n_vars: usize,
        params: &crate::snark::params::SnarkParams,
        aux: &crate::snark::commitment::commit::CommittedAux,
        commit: &<crate::snark_primitives::polynomial_commitment::HyraxBn254 as crate::snark_primitives::polynomial_commitment::MlPcs>::Commitment,
        sponge: &mut S,
        rng: &mut dyn RngCore,
    ) -> Result<crate::snark::proof::LayerScaleOpenCE, crate::snark::errors::SnarkError> {
        let c_idx = scale_packed_index(c_class, layer_idx, n_layers);
        let e_idx = scale_packed_index(e_class, layer_idx, n_layers);
        let c_pt = unit_point_be(c_idx, n_vars);
        let e_pt = unit_point_be(e_idx, n_vars);
        let (c_eval, c_open) = crate::snark::commitment::pcs_helpers::hyrax_open_at(
            &params.committer_key,
            aux,
            commit,
            &c_pt,
            sponge,
            rng,
        )?;
        let (e_eval, e_open) = crate::snark::commitment::pcs_helpers::hyrax_open_at(
            &params.committer_key,
            aux,
            commit,
            &e_pt,
            sponge,
            rng,
        )?;
        Ok(crate::snark::proof::LayerScaleOpenCE {
            c_eval,
            e_eval,
            c_open,
            e_open,
        })
    }

    for (i, layer) in network.layers().iter().enumerate() {
        match layer {
            Layer::Linear { .. } => {
                weight[i] = Some(open_pair(
                    ScaleClass::WeightC,
                    ScaleClass::WeightE,
                    i,
                    n_layers,
                    n_vars,
                    params,
                    &aux,
                    commit,
                    sponge,
                    rng,
                )?);
                bias[i] = Some(open_pair(
                    ScaleClass::BiasC,
                    ScaleClass::BiasE,
                    i,
                    n_layers,
                    n_vars,
                    params,
                    &aux,
                    commit,
                    sponge,
                    rng,
                )?);
            }
            Layer::Activation { .. } => {
                relax_d[i] = Some(open_pair(
                    ScaleClass::RelaxDC,
                    ScaleClass::RelaxDE,
                    i,
                    n_layers,
                    n_vars,
                    params,
                    &aux,
                    commit,
                    sponge,
                    rng,
                )?);
                relax_b[i] = Some(open_pair(
                    ScaleClass::RelaxBC,
                    ScaleClass::RelaxBE,
                    i,
                    n_layers,
                    n_vars,
                    params,
                    &aux,
                    commit,
                    sponge,
                    rng,
                )?);
            }
        }
    }
    Ok(LayerScaleOpens {
        weight,
        bias,
        relax_d,
        relax_b,
    })
}

/// Verify per-layer scale opens and reconstruct a synthetic
/// `LayerScalesCommit` for downstream gadgets. Replays the prover's
/// loop order so the FS transcript stays aligned.
#[allow(clippy::too_many_arguments)]
pub(in crate::snark) fn verify_layer_scale_opens(
    arch: &crate::crown::network::NetworkArchitecture,
    layer_scales_commit: &crate::snark::proof::LayerScalesHyraxCommit,
    opens: &crate::snark::proof::LayerScaleOpens,
    params: &crate::snark::params::SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<crate::snark::proof::LayerScalesCommit, crate::snark::errors::SnarkError> {
    use crate::crown::network::LayerShape;
    use crate::snark::proof::ScaleClass;

    use crate::snark::errors::SnarkError;

    let n_layers = arch.layers().len();
    let n_vars = layer_scales_commit.n_vars as usize;
    if (layer_scales_commit.n_layers as usize) != n_layers {
        return Err(SnarkError::ArchitectureMismatch {
            what: "layer_scales: n_layers != arch.len()",
        });
    }
    if opens.weight.len() != n_layers
        || opens.bias.len() != n_layers
        || opens.relax_d.len() != n_layers
        || opens.relax_b.len() != n_layers
    {
        return Err(SnarkError::ArchitectureMismatch {
            what: "layer_scale_opens: per-class Vec length != n_layers",
        });
    }

    let mut weight_c = vec![0i64; n_layers];
    let mut weight_e = vec![0i32; n_layers];
    let mut bias_c = vec![0i64; n_layers];
    let mut bias_e = vec![0i32; n_layers];
    let mut relax_d_c = vec![0i64; n_layers];
    let mut relax_d_e = vec![0i32; n_layers];
    let mut relax_b_c = vec![0i64; n_layers];
    let mut relax_b_e = vec![0i32; n_layers];

    fn verify_pair<S: CryptographicSponge>(
        open: &crate::snark::proof::LayerScaleOpenCE,
        c_class: crate::snark::proof::ScaleClass,
        e_class: crate::snark::proof::ScaleClass,
        layer_idx: usize,
        n_layers: usize,
        n_vars: usize,
        params: &crate::snark::params::SnarkParams,
        layer_scales_commit: &crate::snark::proof::LayerScalesHyraxCommit,
        sponge: &mut S,
    ) -> Result<(i64, i32), crate::snark::errors::SnarkError> {
        let c_idx = scale_packed_index(c_class, layer_idx, n_layers);
        let e_idx = scale_packed_index(e_class, layer_idx, n_layers);
        let c_pt = unit_point_be(c_idx, n_vars);
        let e_pt = unit_point_be(e_idx, n_vars);
        if !crate::snark::commitment::pcs_helpers::hyrax_verify_at(
            &params.verifier_key,
            &layer_scales_commit.commit,
            &c_pt,
            open.c_eval,
            &open.c_open,
            n_vars,
            sponge,
        )? {
            return Err(crate::snark::errors::SnarkError::PcsOpenRejected {
                which: "layer_scale_opens: c-open Hyrax verify failed",
            });
        }
        if !crate::snark::commitment::pcs_helpers::hyrax_verify_at(
            &params.verifier_key,
            &layer_scales_commit.commit,
            &e_pt,
            open.e_eval,
            &open.e_open,
            n_vars,
            sponge,
        )? {
            return Err(crate::snark::errors::SnarkError::PcsOpenRejected {
                which: "layer_scale_opens: e-open Hyrax verify failed",
            });
        }
        let c = crate::snark_primitives::finite_field::fr_to_signed_i128(open.c_eval).ok_or(
            crate::snark::errors::SnarkError::ArchitectureMismatch {
                what: "layer_scale_opens: c does not fit in i128",
            },
        )?;
        let e = crate::snark_primitives::finite_field::fr_to_signed_i128(open.e_eval).ok_or(
            crate::snark::errors::SnarkError::ArchitectureMismatch {
                what: "layer_scale_opens: e does not fit in i128",
            },
        )?;
        if !(i64::MIN as i128..=i64::MAX as i128).contains(&c) {
            return Err(crate::snark::errors::SnarkError::ArchitectureMismatch {
                what: "layer_scale_opens: c out of i64 range",
            });
        }
        if !(i32::MIN as i128..=i32::MAX as i128).contains(&e) {
            return Err(crate::snark::errors::SnarkError::ArchitectureMismatch {
                what: "layer_scale_opens: e out of i32 range",
            });
        }
        Ok((c as i64, e as i32))
    }

    for (i, layer) in arch.layers().iter().enumerate() {
        match layer {
            LayerShape::Linear { .. } => {
                let w = opens.weight[i]
                    .as_ref()
                    .ok_or(SnarkError::ArchitectureMismatch {
                        what: "layer_scale_opens: missing weight open at Linear layer",
                    })?;
                let b = opens.bias[i]
                    .as_ref()
                    .ok_or(SnarkError::ArchitectureMismatch {
                        what: "layer_scale_opens: missing bias open at Linear layer",
                    })?;
                if opens.relax_d[i].is_some() || opens.relax_b[i].is_some() {
                    return Err(SnarkError::ArchitectureMismatch {
                        what: "layer_scale_opens: relax_* present at Linear layer",
                    });
                }
                let (c, e) = verify_pair(
                    w,
                    ScaleClass::WeightC,
                    ScaleClass::WeightE,
                    i,
                    n_layers,
                    n_vars,
                    params,
                    layer_scales_commit,
                    sponge,
                )?;
                weight_c[i] = c;
                weight_e[i] = e;
                let (c, e) = verify_pair(
                    b,
                    ScaleClass::BiasC,
                    ScaleClass::BiasE,
                    i,
                    n_layers,
                    n_vars,
                    params,
                    layer_scales_commit,
                    sponge,
                )?;
                bias_c[i] = c;
                bias_e[i] = e;
            }
            LayerShape::Activation { .. } => {
                let d = opens.relax_d[i]
                    .as_ref()
                    .ok_or(SnarkError::ArchitectureMismatch {
                        what: "layer_scale_opens: missing relax_d open at Activation layer",
                    })?;
                let bb = opens.relax_b[i]
                    .as_ref()
                    .ok_or(SnarkError::ArchitectureMismatch {
                        what: "layer_scale_opens: missing relax_b open at Activation layer",
                    })?;
                if opens.weight[i].is_some() || opens.bias[i].is_some() {
                    return Err(SnarkError::ArchitectureMismatch {
                        what: "layer_scale_opens: weight/bias present at Activation layer",
                    });
                }
                let (c, e) = verify_pair(
                    d,
                    ScaleClass::RelaxDC,
                    ScaleClass::RelaxDE,
                    i,
                    n_layers,
                    n_vars,
                    params,
                    layer_scales_commit,
                    sponge,
                )?;
                relax_d_c[i] = c;
                relax_d_e[i] = e;
                let (c, e) = verify_pair(
                    bb,
                    ScaleClass::RelaxBC,
                    ScaleClass::RelaxBE,
                    i,
                    n_layers,
                    n_vars,
                    params,
                    layer_scales_commit,
                    sponge,
                )?;
                relax_b_c[i] = c;
                relax_b_e[i] = e;
            }
        }
    }

    Ok(crate::snark::proof::LayerScalesCommit {
        weight_c,
        weight_e,
        bias_c,
        bias_e,
        relax_d_c,
        relax_d_e,
        relax_b_c,
        relax_b_e,
    })
}

/// Absorb the public layer-scales Hyrax commit into the FS sponge.
pub(in crate::snark) fn absorb_layer_scales(
    sponge: &mut impl CryptographicSponge,
    layer_scales_pub: &crate::snark::proof::LayerScalesHyraxCommit,
) {
    sponge.absorb(&(layer_scales_pub.n_layers as u64));
    sponge.absorb(&(layer_scales_pub.n_vars as u64));
    crate::snark::rescaling::absorb_commitment(sponge, &layer_scales_pub.commit);
}
