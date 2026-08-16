//! Chain-initialization binding for the backward pass.
//!
//! Pins the top of the chain to the public property: at the output
//! layer, `chain_a[L]` must equal `spec_c` and `chain_b_acc[L]` must
//! equal `spec_d`. Each side opens both commits at a single FS random
//! point (one for the matrix shape, one for the vector shape) and
//! the verifier checks equality on the opened evals.

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;

use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};

use crate::snark::commitment::commit::{
    native_matrix_n_vars, native_vector_n_vars, CommittedAux, PassCommitments, PassProverStates,
    TensorCommitments,
};
use crate::snark::commitment::pcs_helpers::{
    hyrax_open_batched_at, hyrax_verify_batched_at, BatchOpenSpec, BatchVerifySpec,
};
use crate::snark::errors::SnarkError;
use crate::snark::params::SnarkParams;

/// Absorb a Hyrax commitment into the FS sponge by canonical
/// serialization.
fn absorb_pcs_commit(
    sponge: &mut impl CryptographicSponge,
    commitment: &<HyraxBn254 as MlPcs>::Commitment,
) {
    let mut buf = Vec::new();
    commitment
        .serialize_compressed(&mut buf)
        .expect("serialize commitment");
    sponge.absorb(&buf);
}

/// Per-pass chain-init proof.
///
/// Each side batches the two operands into a single Hyrax open at
/// a shared FS point — two batched opens per pass total.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ChainInitProof {
    /// Number of MLE variables for the `chain_a` / `spec_c` matrix.
    pub a_n_vars: usize,
    /// Number of MLE variables for the `chain_b_acc` / `spec_d` vector.
    pub b_n_vars: usize,
    /// FS point for the matrix-side check.
    pub r_a: Vec<Fr>,
    /// FS point for the vector-side check.
    pub r_b: Vec<Fr>,
    /// Batched Hyrax open at `r_a` for `(chain_a[L], spec_c)`.
    pub r_a_open: <HyraxBn254 as MlPcs>::Proof,
    pub chain_a_eval: Fr,
    pub spec_c_eval: Fr,
    /// Batched Hyrax open at `r_b` for `(chain_b_acc[L], spec_d)`.
    pub r_b_open: <HyraxBn254 as MlPcs>::Proof,
    pub chain_b_eval: Fr,
    pub spec_d_eval: Fr,
}

/// Open the four tensors at the FS random points and emit the proof.
/// The honest prover commits `chain_a[L]` to the same data as
/// `spec_c` (modulo padding), so the opened evals match.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_chain_init(
    pass_com: &PassCommitments,
    pass_st: &PassProverStates,
    commitments: &TensorCommitments,
    spec_c_aux: &CommittedAux,
    spec_d_aux: &CommittedAux,
    a_n_vars: usize,
    b_n_vars: usize,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<ChainInitProof, SnarkError> {
    let _timing = crate::timing::scope("chain_init");
    let chain_a_aux = pass_st.chain_a.last().ok_or(SnarkError::ShapeMismatch {
        what: "chain_init: empty chain_a",
    })?;
    let chain_a_com = pass_com.chain_a.last().ok_or(SnarkError::ShapeMismatch {
        what: "chain_init: empty chain_a com",
    })?;
    let chain_b_aux = pass_st
        .chain_b_acc
        .last()
        .ok_or(SnarkError::ShapeMismatch {
            what: "chain_init: empty chain_b_acc",
        })?;
    let chain_b_com = pass_com
        .chain_b_acc
        .last()
        .ok_or(SnarkError::ShapeMismatch {
            what: "chain_init: empty chain_b_acc com",
        })?;

    // Bind chain commits before squeezing the FS points so the
    // prover cannot adapt them to pre-known challenges. `spec_c` and
    // `spec_d` are already in the sponge from `absorb_commitments`.
    absorb_pcs_commit(sponge, chain_a_com);
    absorb_pcs_commit(sponge, chain_b_com);
    sponge.absorb(&(a_n_vars as u64));
    sponge.absorb(&(b_n_vars as u64));
    let r_a: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(a_n_vars);
    let r_b: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(b_n_vars);

    let r_a_items = [
        BatchOpenSpec {
            aux: chain_a_aux,
            commitment: chain_a_com,
            commit_n_vars: a_n_vars,
        },
        BatchOpenSpec {
            aux: spec_c_aux,
            commitment: &commitments.spec_c,
            commit_n_vars: a_n_vars,
        },
    ];
    let (a_vals, r_a_open) =
        hyrax_open_batched_at(&params.committer_key, &r_a_items, &r_a, sponge, rng)?;
    let chain_a_eval = a_vals[0];
    let spec_c_eval = a_vals[1];

    let r_b_items = [
        BatchOpenSpec {
            aux: chain_b_aux,
            commitment: chain_b_com,
            commit_n_vars: b_n_vars,
        },
        BatchOpenSpec {
            aux: spec_d_aux,
            commitment: &commitments.spec_d,
            commit_n_vars: b_n_vars,
        },
    ];
    let (b_vals, r_b_open) =
        hyrax_open_batched_at(&params.committer_key, &r_b_items, &r_b, sponge, rng)?;
    let chain_b_eval = b_vals[0];
    let spec_d_eval = b_vals[1];

    debug_assert_eq!(
        chain_a_eval, spec_c_eval,
        "chain_init: chain_a[L](r_a) must equal spec_c(r_a)"
    );
    debug_assert_eq!(
        chain_b_eval, spec_d_eval,
        "chain_init: chain_b_acc[L](r_b) must equal spec_d(r_b)"
    );

    Ok(ChainInitProof {
        a_n_vars,
        b_n_vars,
        r_a,
        r_b,
        r_a_open,
        chain_a_eval,
        spec_c_eval,
        r_b_open,
        chain_b_eval,
        spec_d_eval,
    })
}

/// Replay the FS challenges, verify both batched opens, check the
/// two equality identities.
pub(crate) fn verify_chain_init(
    proof: &ChainInitProof,
    pass_com: &PassCommitments,
    commitments: &TensorCommitments,
    a_n_vars: usize,
    b_n_vars: usize,
    params: &SnarkParams,
    sponge: &mut impl CryptographicSponge,
) -> Result<(), SnarkError> {
    if proof.a_n_vars != a_n_vars || proof.b_n_vars != b_n_vars {
        return Err(SnarkError::ShapeMismatch {
            what: "chain_init: n_vars mismatch",
        });
    }

    let chain_a_com = pass_com.chain_a.last().ok_or(SnarkError::ShapeMismatch {
        what: "chain_init: empty chain_a (verify)",
    })?;
    let chain_b_com = pass_com
        .chain_b_acc
        .last()
        .ok_or(SnarkError::ShapeMismatch {
            what: "chain_init: empty chain_b_acc (verify)",
        })?;

    // Mirror the prover: bind chain commits before squeezing.
    absorb_pcs_commit(sponge, chain_a_com);
    absorb_pcs_commit(sponge, chain_b_com);
    sponge.absorb(&(a_n_vars as u64));
    sponge.absorb(&(b_n_vars as u64));
    let r_a: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(a_n_vars);
    let r_b: Vec<Fr> = sponge.squeeze_field_elements::<Fr>(b_n_vars);
    if r_a != proof.r_a || r_b != proof.r_b {
        return Err(SnarkError::TranscriptMismatch);
    }

    let r_a_items = [
        BatchVerifySpec {
            commitment: chain_a_com,
            value: proof.chain_a_eval,
            commit_n_vars: a_n_vars,
        },
        BatchVerifySpec {
            commitment: &commitments.spec_c,
            value: proof.spec_c_eval,
            commit_n_vars: a_n_vars,
        },
    ];
    let r_a_ok = hyrax_verify_batched_at(
        &params.verifier_key,
        &r_a_items,
        &proof.r_a,
        &proof.r_a_open,
        sponge,
    )?;
    if !r_a_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "chain_init_r_a_batch",
        });
    }

    let r_b_items = [
        BatchVerifySpec {
            commitment: chain_b_com,
            value: proof.chain_b_eval,
            commit_n_vars: b_n_vars,
        },
        BatchVerifySpec {
            commitment: &commitments.spec_d,
            value: proof.spec_d_eval,
            commit_n_vars: b_n_vars,
        },
    ];
    let r_b_ok = hyrax_verify_batched_at(
        &params.verifier_key,
        &r_b_items,
        &proof.r_b,
        &proof.r_b_open,
        sponge,
    )?;
    if !r_b_ok {
        return Err(SnarkError::PcsOpenRejected {
            which: "chain_init_r_b_batch",
        });
    }

    if proof.chain_a_eval != proof.spec_c_eval {
        return Err(SnarkError::ChainInitMismatch);
    }
    if proof.chain_b_eval != proof.spec_d_eval {
        return Err(SnarkError::ChainInitMismatch);
    }
    Ok(())
}

/// Derive `(a_n_vars, b_n_vars)` from the public architecture. These
/// are the native commit `n_vars` shared by `chain_a[L] / spec_c`
/// and `chain_b_acc[L] / spec_d`, which also doubles as the FS
/// squeeze length so the open's point dimension matches the commit.
pub(crate) fn chain_init_n_vars(
    arch: &crate::crown::network::NetworkArchitecture,
    property: &crate::crown::output_property::Property,
) -> (usize, usize) {
    let n_spec = property.c_matrix.nrows();
    let n_out = arch.output_dim();
    (
        native_matrix_n_vars(n_spec, n_out),
        native_vector_n_vars(n_spec),
    )
}
