//! Public-statement-to-commit bindings.
//!
//! The verifier re-derives the working and input scales and the
//! canonical quantized codes (`spec_c`, `spec_d`, `x_lower`,
//! `x_upper`) from the public statement, then opens each committed
//! tensor at a Fiat-Shamir random point and checks the value
//! matches the canonical MLE eval at the same point. Without this
//! bind, the prover could commit to a different statement and the
//! chain's internal soundness would still hold.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;
use ndarray::Array1;

use crate::crown::output_property::Property;
use crate::quantization::quantized_array::{
    pick_scale_pow2, quantize_matrix, quantize_vector, quantize_vector_ceil, quantize_vector_floor,
};
use crate::quantization::scale::Scale;
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use crate::snark::commitment::commit::{
    native_matrix_n_vars, native_vector_n_vars, CommittedAux, TensorCommitments,
};
use crate::snark::commitment::multilinear_extensions::{
    eval_multilinear_full, mle_table_from_matrix, mle_table_from_vector,
};
use crate::snark::commitment::pcs_helpers::{
    hyrax_open_at, hyrax_open_batched_at, hyrax_verify_at, hyrax_verify_batched_at, BatchOpenSpec,
    BatchVerifySpec,
};
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;

/// Per-proof public-statement binding (independent of pass).
///
/// `(x_lower, x_upper)` share `r_x` and a common `commit_n_vars`,
/// so they are batched into a single Hyrax open; `spec_c` and
/// `spec_d` use distinct points and remain per-tensor opens.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PublicBindingProof {
    pub spec_c_n_vars: usize,
    pub spec_d_n_vars: usize,
    pub x_n_vars: usize,
    pub r_spec_c: Vec<Fr>,
    pub r_spec_d: Vec<Fr>,
    pub r_x: Vec<Fr>,
    pub spec_c_open: <HyraxBn254 as MlPcs>::Proof,
    pub spec_d_open: <HyraxBn254 as MlPcs>::Proof,
    /// Batched open of `(x_lower, x_upper)` at `r_x`.
    pub r_x_open: <HyraxBn254 as MlPcs>::Proof,
    pub spec_c_eval: Fr,
    pub spec_d_eval: Fr,
    pub x_lower_eval: Fr,
    pub x_upper_eval: Fr,
    /// Working scale claimed by the prover. The verifier re-derives
    /// the canonical value and checks consistency with
    /// `target_scale_c/_e`.
    pub working_c: i64,
    pub working_e: i32,
    /// Input scale claimed by the prover.
    pub input_c: i64,
    pub input_e: i32,
}

/// Re-derive `(working, input)` scales from the public statement
/// and `precision_bits`, mirroring the prover's scale picks in
/// `qcrown::run_quant_pass`.
///
/// Takes the architecture-only view (not the full `Network`) so
/// private weights cannot leak into this path; the architecture is
/// also where sigmoid/tanh layers are detected to apply the
/// `working = 2^sigma_x_scale_log2` override. `sigma_x_scale_log2` is
/// the runtime public σ input-scale (from `SnarkParams`), so a
/// prover/verifier disagreement on it surfaces here as a working-scale
/// mismatch against the claimed `target_scale`.
pub fn derive_public_scales(
    architecture: &crate::crown::network::NetworkArchitecture,
    property: &Property,
    x_lower: &Array1<f64>,
    x_upper: &Array1<f64>,
    precision_bits: i32,
    input_scale_log2: Option<i32>,
    sigma_x_scale_log2: i32,
) -> (Scale, Scale) {
    let spec_values: Vec<f64> = property
        .c_matrix
        .iter()
        .copied()
        .chain(property.d_vector.iter().copied())
        .collect();
    let mut working = pick_scale_pow2(&spec_values, precision_bits);
    // Mirror the cert pipeline's sigmoid/tanh override: force
    // `working = 2^sigma_x_scale_log2` so the SNARK Phase 3b/3c gadgets
    // see `s_w == s_x`.
    use crate::crown::network::{ActivationKind, LayerShape};
    let has_sshape = architecture.layers().iter().any(|l| {
        matches!(
            l,
            LayerShape::Activation {
                kind: ActivationKind::Sigmoid | ActivationKind::Tanh,
            }
        )
    });
    if has_sshape && working.e != sigma_x_scale_log2 {
        working = Scale::from_pow2(sigma_x_scale_log2);
    }
    let input_scale = match input_scale_log2 {
        Some(e) => Scale::from_pow2(e),
        None => {
            let mut input_concat: Vec<f64> = Vec::with_capacity(x_lower.len() + x_upper.len());
            input_concat.extend_from_slice(x_lower.as_slice().expect("x_lower contiguous"));
            input_concat.extend_from_slice(x_upper.as_slice().expect("x_upper contiguous"));
            pick_scale_pow2(&input_concat, precision_bits)
        }
    };
    (working, input_scale)
}

/// Native commit `n_vars` for `(spec_c, spec_d, x)` — matches the
/// sizes used by `commit_matrix` / `commit_vector` so FS-squeezed
/// points lift to the right dimension.
pub(crate) fn public_binding_n_vars(
    property: &Property,
    x_lower: &Array1<f64>,
) -> (usize, usize, usize) {
    let n_spec = property.c_matrix.nrows();
    let out_dim = property.c_matrix.ncols();
    let spec_c = native_matrix_n_vars(n_spec, out_dim);
    let spec_d = native_vector_n_vars(n_spec);
    let x = native_vector_n_vars(x_lower.len());
    (spec_c, spec_d, x)
}

/// Open `(spec_c, spec_d, x_lower, x_upper)` at FS-derived points
/// and bundle the evals into a [`PublicBindingProof`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_public_binding(
    network: &crate::crown::network::Network,
    property: &Property,
    x_lower: &Array1<f64>,
    x_upper: &Array1<f64>,
    precision_bits: i32,
    commitments: &TensorCommitments,
    spec_c_aux: &CommittedAux,
    spec_d_aux: &CommittedAux,
    x_lower_aux: &CommittedAux,
    x_upper_aux: &CommittedAux,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<PublicBindingProof, SnarkError> {
    let _timing = crate::timing::scope("public_binding");
    let arch = network.architecture();
    let (working, input_scale) = derive_public_scales(
        &arch,
        property,
        x_lower,
        x_upper,
        precision_bits,
        params.input_scale_log2,
        params.sigma_x_scale_log2,
    );
    let (spec_c_n_vars, spec_d_n_vars, x_n_vars) = public_binding_n_vars(property, x_lower);

    sponge.absorb(&(precision_bits as i64));
    sponge.absorb(&(spec_c_n_vars as u64));
    sponge.absorb(&(spec_d_n_vars as u64));
    sponge.absorb(&(x_n_vars as u64));
    let r_spec_c: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(spec_c_n_vars);
    let r_spec_d: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(spec_d_n_vars);
    let r_x: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(x_n_vars);

    let (spec_c_eval, spec_c_open) = hyrax_open_at(
        &params.committer_key,
        spec_c_aux,
        &commitments.spec_c,
        &r_spec_c,
        sponge,
        rng,
    )?;
    let (spec_d_eval, spec_d_open) = hyrax_open_at(
        &params.committer_key,
        spec_d_aux,
        &commitments.spec_d,
        &r_spec_d,
        sponge,
        rng,
    )?;

    let r_x_items = [
        BatchOpenSpec {
            aux: x_lower_aux,
            commitment: &commitments.x_lower,
            commit_n_vars: x_n_vars,
        },
        BatchOpenSpec {
            aux: x_upper_aux,
            commitment: &commitments.x_upper,
            commit_n_vars: x_n_vars,
        },
    ];
    let (x_vals, r_x_open) =
        hyrax_open_batched_at(&params.committer_key, &r_x_items, &r_x, sponge, rng)?;
    let x_lower_eval = x_vals[0];
    let x_upper_eval = x_vals[1];

    Ok(PublicBindingProof {
        spec_c_n_vars,
        spec_d_n_vars,
        x_n_vars,
        r_spec_c,
        r_spec_d,
        r_x,
        spec_c_open,
        spec_d_open,
        r_x_open,
        spec_c_eval,
        spec_d_eval,
        x_lower_eval,
        x_upper_eval,
        working_c: working.c,
        working_e: working.e,
        input_c: input_scale.c,
        input_e: input_scale.e,
    })
}

/// Re-derive scales, quantize the public statement, and assert
/// the canonical MLE evaluations match the committed opens.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_public_binding(
    proof: &PublicBindingProof,
    architecture: &crate::crown::network::NetworkArchitecture,
    property: &Property,
    x_lower: &Array1<f64>,
    x_upper: &Array1<f64>,
    precision_bits: i32,
    commitments: &TensorCommitments,
    target_scale_c: i64,
    target_scale_e: i32,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    let (working, input_scale) = derive_public_scales(
        architecture,
        property,
        x_lower,
        x_upper,
        precision_bits,
        params.input_scale_log2,
        params.sigma_x_scale_log2,
    );
    let (spec_c_n_vars, spec_d_n_vars, x_n_vars) = public_binding_n_vars(property, x_lower);

    if proof.spec_c_n_vars != spec_c_n_vars
        || proof.spec_d_n_vars != spec_d_n_vars
        || proof.x_n_vars != x_n_vars
    {
        return Err(SnarkError::ShapeMismatch {
            what: "public_binding: n_vars mismatch",
        });
    }
    if proof.working_c != working.c || proof.working_e != working.e {
        return Err(SnarkError::PublicBindingFailed);
    }
    if proof.input_c != input_scale.c || proof.input_e != input_scale.e {
        return Err(SnarkError::PublicBindingFailed);
    }
    // Cross-check with the prover's claimed target_scale, which
    // output_bound consumes downstream.
    if working.c != target_scale_c || working.e != target_scale_e {
        return Err(SnarkError::PublicBindingFailed);
    }

    sponge.absorb(&(precision_bits as i64));
    sponge.absorb(&(spec_c_n_vars as u64));
    sponge.absorb(&(spec_d_n_vars as u64));
    sponge.absorb(&(x_n_vars as u64));
    let r_spec_c: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(spec_c_n_vars);
    let r_spec_d: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(spec_d_n_vars);
    let r_x: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(x_n_vars);
    if r_spec_c != proof.r_spec_c || r_spec_d != proof.r_spec_d || r_x != proof.r_x {
        return Err(SnarkError::TranscriptMismatch);
    }

    let single_opens = [
        (
            "public_spec_c",
            &commitments.spec_c,
            &proof.r_spec_c,
            proof.spec_c_eval,
            &proof.spec_c_open,
            spec_c_n_vars,
        ),
        (
            "public_spec_d",
            &commitments.spec_d,
            &proof.r_spec_d,
            proof.spec_d_eval,
            &proof.spec_d_open,
            spec_d_n_vars,
        ),
    ];
    for (which, com, point, value, open_proof, nv) in single_opens {
        let ok = hyrax_verify_at(
            &params.verifier_key,
            com,
            point,
            value,
            open_proof,
            nv,
            sponge,
        )?;
        if !ok {
            return Err(SnarkError::PcsOpenRejected { which });
        }
    }

    // Batched (x_lower, x_upper) open at r_x.
    let r_x_items = [
        BatchVerifySpec {
            commitment: &commitments.x_lower,
            value: proof.x_lower_eval,
            commit_n_vars: x_n_vars,
        },
        BatchVerifySpec {
            commitment: &commitments.x_upper,
            value: proof.x_upper_eval,
            commit_n_vars: x_n_vars,
        },
    ];
    let r_x_ok = hyrax_verify_batched_at(
        &params.verifier_key,
        &r_x_items,
        &proof.r_x,
        &proof.r_x_open,
        sponge,
    )?;
    if !r_x_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "public_x_box_batch",
        });
    }

    // Pad each canonical table to the native commit n_vars so the
    // FS-squeezed point length lines up with the table size.
    fn pad_to_n_vars(mut evals: Vec<Fr>, n_vars: usize) -> Vec<Fr> {
        evals.resize(1usize << n_vars, Fr::from(0u64));
        evals
    }
    let canonical_spec_c = quantize_matrix(&property.c_matrix, working);
    let (spec_c_evals, _) = mle_table_from_matrix(&canonical_spec_c);
    let spec_c_evals = pad_to_n_vars(spec_c_evals, spec_c_n_vars);
    let expected_spec_c = eval_multilinear_full(&spec_c_evals, &proof.r_spec_c);
    if expected_spec_c != proof.spec_c_eval {
        return Err(SnarkError::PublicBindingFailed);
    }

    let canonical_spec_d = quantize_vector(&property.d_vector, working);
    let spec_d_evals = pad_to_n_vars(mle_table_from_vector(&canonical_spec_d), spec_d_n_vars);
    let expected_spec_d = eval_multilinear_full(&spec_d_evals, &proof.r_spec_d);
    if expected_spec_d != proof.spec_d_eval {
        return Err(SnarkError::PublicBindingFailed);
    }

    // Mirror the cert generator's outward rounding so the quantized
    // box strictly contains the real one.
    let canonical_x_lower = quantize_vector_floor(x_lower, input_scale);
    let x_lower_evals = pad_to_n_vars(mle_table_from_vector(&canonical_x_lower), x_n_vars);
    let expected_x_lower = eval_multilinear_full(&x_lower_evals, &proof.r_x);
    if expected_x_lower != proof.x_lower_eval {
        return Err(SnarkError::PublicBindingFailed);
    }

    let canonical_x_upper = quantize_vector_ceil(x_upper, input_scale);
    let x_upper_evals = pad_to_n_vars(mle_table_from_vector(&canonical_x_upper), x_n_vars);
    let expected_x_upper = eval_multilinear_full(&x_upper_evals, &proof.r_x);
    if expected_x_upper != proof.x_upper_eval {
        return Err(SnarkError::PublicBindingFailed);
    }

    Ok(())
}
