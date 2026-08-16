//! Native Hyrax PCS over BN254 G1. Wire-compatible with
//! `ark_poly_commit::hyrax`: identical [`HyraxCommitment`] and
//! [`HyraxProof`] shapes; the Bulletproofs-style dot-product argument
//! is the same protocol arkworks implements upstream.
//!
//! The only visible difference from upstream is [`CommitState`]: the
//! per-row Pedersen blinders are `pub`. Arkworks keeps them
//! `pub(crate)`, which prevents us from forming
//! `r_batch[j] = Σ_k ρ^k · randomness_k[j]` for a batched commit. Making
//! the field public unblocks the [`open_batched_at`] /
//! [`verify_batched_at`] path, which produces one Hyrax proof attesting
//! to `K` MLE openings at a shared evaluation point. Soundness reduces
//! to single-poly Hyrax via Pedersen homomorphism plus a
//! Schwartz–Zippel argument on the FS-squeezed `ρ`.

use ark_bn254::{Fr, G1Affine};
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, UniformRand};
use ark_poly::{DenseMultilinearExtension, Polynomial};
use ark_poly_commit::hyrax::{HyraxCommitment, HyraxProof, HyraxUniversalParams};
use ark_serialize::CanonicalSerialize;
use ark_std::rand::RngCore;
use thiserror::Error;

/// Hyrax universal parameters: the Pedersen vector `com_key` plus a
/// separate hiding generator `h`. Identical to upstream.
pub type CommitterKey = HyraxUniversalParams<G1Affine>;
pub type VerifierKey = HyraxUniversalParams<G1Affine>;
/// Wire-compatible with arkworks' `HyraxCommitment<G1Affine>`.
pub type Commitment = HyraxCommitment<G1Affine>;
/// Wire-compatible with arkworks' `HyraxProof<G1Affine>`.
pub type Proof = HyraxProof<G1Affine>;

/// Per-commit prover state. `randomness` is the per-row Pedersen blinder
/// vector of length `dim = 2^(n_vars / 2)`; public so the SNARK driver
/// can build the linearly combined randomness for a batched open.
#[derive(Clone, Debug)]
pub struct CommitState {
    pub randomness: Vec<Fr>,
}

#[derive(Debug, Error)]
pub enum HyraxError {
    #[error("eval table size {got} is not a positive power of two")]
    BadEvalTable { got: usize },
    #[error("number of variables must be even (Hyrax sqrt-MLE layout); got {got}")]
    OddNumVars { got: usize },
    #[error("opening point has {got} variables but commit has {expected}")]
    BadPointDim { got: usize, expected: usize },
    #[error("committer key has {got} generators but commit needs {expected}")]
    SmallKey { got: usize, expected: usize },
    #[error("randomness vector has {got} entries but row-dim is {expected}")]
    BadRandomness { got: usize, expected: usize },
    #[error("batched open: empty item list")]
    EmptyBatch,
    #[error("batched open: item {idx} has n_vars {got} but item 0 has {expected}")]
    HeterogenousBatchVars {
        idx: usize,
        got: usize,
        expected: usize,
    },
    #[error("batched verify: item {idx} commit row count {got} != expected {expected}")]
    HeterogenousBatchRows {
        idx: usize,
        got: usize,
        expected: usize,
    },
}

fn n_vars_from_len(len: usize) -> Result<usize, HyraxError> {
    if !len.is_power_of_two() || len == 0 {
        return Err(HyraxError::BadEvalTable { got: len });
    }
    Ok(len.trailing_zeros() as usize)
}

fn pedersen_commit(key: &[G1Affine], scalars: &[Fr]) -> G1Affine {
    assert_eq!(
        key.len(),
        scalars.len(),
        "pedersen_commit: key/scalars length mismatch"
    );
    let bigints: Vec<_> = scalars.iter().map(|s| s.into_bigint()).collect();
    <<G1Affine as AffineRepr>::Group as VariableBaseMSM>::msm_bigint(key, &bigints).into_affine()
}

/// Extract row `j` of the `dim×dim` column-major matrix that `evals`
/// represents (`flat[c·dim + j]` is matrix entry `(j, c)`). We build it
/// on the fly into `scratch` to avoid a transposed copy.
fn col_major_row<'a>(evals: &'a [Fr], j: usize, dim: usize, scratch: &'a mut Vec<Fr>) -> &'a [Fr] {
    scratch.clear();
    for c in 0..dim {
        scratch.push(evals[c * dim + j]);
    }
    scratch.as_slice()
}

/// Tensor product of `(1 - x_i, x_i)` over `values`. Matches the
/// indexing of `ark_poly_commit::hyrax::utils::tensor_prime`: the FIRST
/// input element corresponds to the HIGHEST output bit. We iterate in
/// reverse so each newly-added value claims the next-higher output bit.
fn tensor_prime(values: &[Fr]) -> Vec<Fr> {
    let mut out = vec![Fr::from(1u64)];
    for &v in values.iter().rev() {
        let one_minus_v = Fr::from(1u64) - v;
        let mut next = Vec::with_capacity(out.len() * 2);
        for &t in &out {
            next.push(t * one_minus_v);
        }
        for &t in &out {
            next.push(t * v);
        }
        out = next;
    }
    out
}

fn inner_product(a: &[Fr], b: &[Fr]) -> Fr {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
}

fn serialize_compressed_vec<T: CanonicalSerialize>(v: &T) -> Vec<u8> {
    let mut buf = Vec::new();
    v.serialize_compressed(&mut buf)
        .expect("serialize compressed");
    buf
}

/// Commit to the MLE whose flat eval table is `evals` (length
/// `2^n_vars`, with `n_vars` even). Wire-compatible with arkworks Hyrax.
pub fn commit(
    ck: &CommitterKey,
    evals: &[Fr],
    rng: &mut dyn RngCore,
) -> Result<(Commitment, CommitState), HyraxError> {
    let _timing = crate::timing::counter("pcs_commit");
    let n_vars = n_vars_from_len(evals.len())?;
    if n_vars % 2 == 1 {
        return Err(HyraxError::OddNumVars { got: n_vars });
    }
    let dim = 1usize << (n_vars / 2);
    if ck.com_key.len() < dim {
        return Err(HyraxError::SmallKey {
            got: ck.com_key.len(),
            expected: dim,
        });
    }
    let com_key = &ck.com_key[..dim];

    let mut row_coms = Vec::with_capacity(dim);
    let mut randomness = Vec::with_capacity(dim);
    let mut scratch = Vec::with_capacity(dim);
    for j in 0..dim {
        let row = col_major_row(evals, j, dim, &mut scratch);
        let r = Fr::rand(rng);
        let c = (pedersen_commit(com_key, row) + ck.h * r).into_affine();
        row_coms.push(c);
        randomness.push(r);
    }
    Ok((HyraxCommitment { row_coms }, CommitState { randomness }))
}

/// Run the Bulletproofs-style dot-product opening at `point`. Used by
/// both single-poly and batched opens. Mirrors
/// `ark_poly_commit::hyrax::open`'s transcript shape:
/// `absorb(ck, row_coms, point, com_eval, com_d, com_b) → squeeze c`.
/// Callers must absorb the claimed value into the sponge BEFORE this
/// call so a lying claim flips `c`.
fn open_inner(
    ck: &CommitterKey,
    evals: &[Fr],
    randomness: &[Fr],
    com: &Commitment,
    point: &[Fr],
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<(Fr, Proof), HyraxError> {
    let n_vars = n_vars_from_len(evals.len())?;
    if n_vars % 2 == 1 {
        return Err(HyraxError::OddNumVars { got: n_vars });
    }
    if point.len() != n_vars {
        return Err(HyraxError::BadPointDim {
            got: point.len(),
            expected: n_vars,
        });
    }
    let dim = 1usize << (n_vars / 2);
    if randomness.len() != dim {
        return Err(HyraxError::BadRandomness {
            got: randomness.len(),
            expected: dim,
        });
    }
    if com.row_coms.len() != dim {
        return Err(HyraxError::BadPointDim {
            got: com.row_coms.len(),
            expected: dim,
        });
    }
    if ck.com_key.len() < dim {
        return Err(HyraxError::SmallKey {
            got: ck.com_key.len(),
            expected: dim,
        });
    }
    let com_key = &ck.com_key[..dim];

    // LE order matches the MLE indexing convention upstream uses.
    let point_rev: Vec<Fr> = point.iter().rev().copied().collect();
    let l = tensor_prime(&point_rev[n_vars / 2..]);
    let r = tensor_prime(&point_rev[..n_vars / 2]);

    // lt = l · mat. Iterate c-outer, j-inner: the inner loop reads
    // `evals[c*dim..]` sequentially (stride 1, cache-friendly). The
    // transposed traversal strides by `dim` and thrashes L1 once
    // `dim × dim` exceeds the cache.
    let mut lt: Vec<Fr> = Vec::with_capacity(dim);
    for c in 0..dim {
        let row = &evals[c * dim..(c + 1) * dim];
        lt.push(inner_product(&l, row));
    }
    let r_lt = inner_product(&l, randomness);
    let eval = inner_product(&lt, &r);

    let r_eval = Fr::rand(rng);
    let com_eval = (com_key[0] * eval + ck.h * r_eval).into_affine();

    let d: Vec<Fr> = (0..dim).map(|_| Fr::rand(rng)).collect();
    let b = inner_product(&r, &d);
    let r_d = Fr::rand(rng);
    let r_b = Fr::rand(rng);
    let com_d = (pedersen_commit(com_key, &d) + ck.h * r_d).into_affine();
    let com_b = (com_key[0] * b + ck.h * r_b).into_affine();

    sponge.absorb(&serialize_compressed_vec(ck));
    sponge.absorb(&serialize_compressed_vec(&com.row_coms));
    sponge.absorb(&point.to_vec());
    sponge.absorb(&serialize_compressed_vec(&com_eval));
    sponge.absorb(&serialize_compressed_vec(&com_d));
    sponge.absorb(&serialize_compressed_vec(&com_b));
    let c = sponge.squeeze_field_elements::<Fr>(1)[0];

    let mut z = Vec::with_capacity(dim);
    for j in 0..dim {
        z.push(d[j] + c * lt[j]);
    }
    let z_d = c * r_lt + r_d;
    let z_b = c * r_eval + r_b;

    Ok((
        eval,
        HyraxProof {
            com_eval,
            com_d,
            com_b,
            z,
            z_d,
            z_b,
        },
    ))
}

/// Verify a Bulletproofs-style dot-product opening at `point`.
fn verify_inner(
    vk: &VerifierKey,
    com: &Commitment,
    point: &[Fr],
    proof: &Proof,
    sponge: &mut impl CryptographicSponge,
) -> Result<bool, HyraxError> {
    let n_vars = point.len();
    if n_vars % 2 == 1 {
        return Err(HyraxError::OddNumVars { got: n_vars });
    }
    let dim = 1usize << (n_vars / 2);
    if com.row_coms.len() != dim {
        return Err(HyraxError::BadPointDim {
            got: com.row_coms.len(),
            expected: dim,
        });
    }
    if proof.z.len() != dim {
        return Err(HyraxError::BadPointDim {
            got: proof.z.len(),
            expected: dim,
        });
    }
    if vk.com_key.len() < dim {
        return Err(HyraxError::SmallKey {
            got: vk.com_key.len(),
            expected: dim,
        });
    }
    let com_key = &vk.com_key[..dim];

    let point_rev: Vec<Fr> = point.iter().rev().copied().collect();
    let l = tensor_prime(&point_rev[n_vars / 2..]);
    let r = tensor_prime(&point_rev[..n_vars / 2]);

    sponge.absorb(&serialize_compressed_vec(vk));
    sponge.absorb(&serialize_compressed_vec(&com.row_coms));
    sponge.absorb(&point.to_vec());
    sponge.absorb(&serialize_compressed_vec(&proof.com_eval));
    sponge.absorb(&serialize_compressed_vec(&proof.com_d));
    sponge.absorb(&serialize_compressed_vec(&proof.com_b));
    let c = sponge.squeeze_field_elements::<Fr>(1)[0];

    // (1) pedersen(com_key, z) + h · z_d == com_d + c · (l · row_coms)
    let lhs1 = (pedersen_commit(com_key, &proof.z) + vk.h * proof.z_d).into_affine();
    let l_bigints: Vec<_> = l.iter().map(|x| x.into_bigint()).collect();
    let l_dot_row = <<G1Affine as AffineRepr>::Group as VariableBaseMSM>::msm_bigint(
        &com.row_coms[..dim],
        &l_bigints,
    )
    .into_affine();
    let rhs1 = (proof.com_d + l_dot_row * c).into_affine();
    if lhs1 != rhs1 {
        return Ok(false);
    }

    // (2) com_key[0] · <z, r> + h · z_b == com_b + c · com_eval
    let zr = inner_product(&proof.z, &r);
    let lhs2 = (com_key[0] * zr + vk.h * proof.z_b).into_affine();
    let rhs2 = (proof.com_b + proof.com_eval * c).into_affine();
    if lhs2 != rhs2 {
        return Ok(false);
    }

    Ok(true)
}

// ---------------------------------------------------------------------------
// Public single-poly open / verify.
// ---------------------------------------------------------------------------

/// Open the committed MLE at `point` (length `n_vars`, even).
///
/// Sponge contract: caller MUST `sponge.absorb(&claimed_value)` before
/// calling; mirrors the [`crate::snark_primitives::polynomial_commitment::HyraxBn254`]
/// wrapper.
pub fn open_at(
    ck: &CommitterKey,
    evals: &[Fr],
    state: &CommitState,
    com: &Commitment,
    point: &[Fr],
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<(Fr, Proof), HyraxError> {
    let _timing = crate::timing::counter("pcs_open");
    open_inner(ck, evals, &state.randomness, com, point, sponge, rng)
}

/// Verify a single-poly Hyrax opening at `point`. Caller MUST
/// `sponge.absorb(&claimed_value)` before this call (mirrors [`open_at`]).
pub fn verify_at(
    vk: &VerifierKey,
    com: &Commitment,
    point: &[Fr],
    proof: &Proof,
    sponge: &mut impl CryptographicSponge,
) -> Result<bool, HyraxError> {
    verify_inner(vk, com, point, proof, sponge)
}

// ---------------------------------------------------------------------------
// Batched-open primitive: many polys, one shared point.
// ---------------------------------------------------------------------------

/// Prover-side input for one tensor in a batched open.
pub struct BatchOpenItem<'a> {
    pub com: &'a Commitment,
    pub evals: &'a [Fr],
    pub state: &'a CommitState,
}

/// Verifier-side input for one tensor in a batched open.
pub struct BatchVerifyItem<'a> {
    pub com: &'a Commitment,
    pub value: Fr,
}

// The `(C_k, y_k)`-absorb-then-squeeze-`ρ` step is inlined at each call
// site below: `CryptographicSponge` is not dyn-compatible, so we cannot
// factor it through `&mut dyn`.

fn powers_of(rho: Fr, k: usize) -> Vec<Fr> {
    let mut out = Vec::with_capacity(k);
    let mut p = Fr::from(1u64);
    for _ in 0..k {
        out.push(p);
        p *= rho;
    }
    out
}

fn eval_mle_at(evals: &[Fr], point: &[Fr]) -> Fr {
    // Arkworks' DenseMultilinearExtension fixes our LE-indexing
    // convention (point variable 0 at the lowest bit). Cold path
    // (one call per tensor per batched open).
    let mle = DenseMultilinearExtension::from_evaluations_slice(point.len(), evals);
    mle.evaluate(&point.to_vec())
}

/// Produce one Hyrax proof attesting to `K` MLE openings at `point`.
/// Returns the per-tensor claimed values `y_k` so the caller can ship
/// them in the public proof and rebind them on the verifier side.
///
/// Sponge contract: absorbs `(K, com_0, y_0, …, com_{K-1}, y_{K-1})`,
/// squeezes `ρ`, absorbs `y_batch`, then delegates to `open_inner`
/// (which absorbs `ck, row_coms_batch, point, com_eval, com_d, com_b`
/// and squeezes `c`). The verifier mirrors exactly.
pub fn open_batched_at(
    ck: &CommitterKey,
    items: &[BatchOpenItem<'_>],
    point: &[Fr],
    sponge: &mut impl CryptographicSponge,
    rng: &mut dyn RngCore,
) -> Result<(Vec<Fr>, Proof), HyraxError> {
    let _timing = crate::timing::counter("pcs_open");
    if items.is_empty() {
        return Err(HyraxError::EmptyBatch);
    }
    let n_vars_first = n_vars_from_len(items[0].evals.len())?;
    if n_vars_first % 2 == 1 {
        return Err(HyraxError::OddNumVars { got: n_vars_first });
    }
    if point.len() != n_vars_first {
        return Err(HyraxError::BadPointDim {
            got: point.len(),
            expected: n_vars_first,
        });
    }
    let dim = 1usize << (n_vars_first / 2);
    for (idx, item) in items.iter().enumerate().skip(1) {
        let n_vars_k = n_vars_from_len(item.evals.len())?;
        if n_vars_k != n_vars_first {
            return Err(HyraxError::HeterogenousBatchVars {
                idx,
                got: n_vars_k,
                expected: n_vars_first,
            });
        }
    }
    for (idx, item) in items.iter().enumerate() {
        if item.com.row_coms.len() != dim {
            return Err(HyraxError::HeterogenousBatchRows {
                idx,
                got: item.com.row_coms.len(),
                expected: dim,
            });
        }
        if item.state.randomness.len() != dim {
            return Err(HyraxError::BadRandomness {
                got: item.state.randomness.len(),
                expected: dim,
            });
        }
    }

    let values: Vec<Fr> = items
        .iter()
        .map(|item| eval_mle_at(item.evals, point))
        .collect();

    sponge.absorb(&(items.len() as u64));
    for k in 0..items.len() {
        sponge.absorb(&serialize_compressed_vec(&items[k].com.row_coms));
        sponge.absorb(&values[k]);
    }
    let rho = sponge.squeeze_field_elements::<Fr>(1)[0];
    let powers = powers_of(rho, items.len());

    // Linearly combine eval tables, randomness, and row commitments.
    let n_evals = items[0].evals.len();
    let mut evals_batch = vec![Fr::from(0u64); n_evals];
    let mut randomness_batch = vec![Fr::from(0u64); dim];
    let mut row_coms_batch_proj: Vec<<G1Affine as AffineRepr>::Group> =
        vec![<<G1Affine as AffineRepr>::Group as ark_std::Zero>::zero(); dim];
    for (k, item) in items.iter().enumerate() {
        let pk = powers[k];
        for i in 0..n_evals {
            evals_batch[i] += pk * item.evals[i];
        }
        for j in 0..dim {
            randomness_batch[j] += pk * item.state.randomness[j];
        }
        for j in 0..dim {
            row_coms_batch_proj[j] += item.com.row_coms[j] * pk;
        }
    }
    let row_coms_batch =
        <<G1Affine as AffineRepr>::Group as CurveGroup>::normalize_batch(&row_coms_batch_proj);
    let com_batch = HyraxCommitment {
        row_coms: row_coms_batch,
    };

    let value_batch: Fr = values
        .iter()
        .zip(powers.iter())
        .fold(Fr::from(0u64), |acc, (y, p)| acc + *y * *p);

    debug_assert_eq!(
        eval_mle_at(&evals_batch, point),
        value_batch,
        "batched_open_at: y_batch != Σ ρ^k · y_k (indexing bug)"
    );

    // Mirror the single-poly wrapper: bind the batched claimed value
    // into the sponge before delegating to the dot-product argument.
    sponge.absorb(&value_batch);
    let (_y_batch, proof) = open_inner(
        ck,
        &evals_batch,
        &randomness_batch,
        &com_batch,
        point,
        sponge,
        rng,
    )?;
    Ok((values, proof))
}

/// Verify a batched open. Re-derives `ρ`, builds
/// `C_batch = Σ ρ^k · C_k` and `y_batch = Σ ρ^k · y_k`, and runs the
/// single-poly verify against `(C_batch, y_batch)`.
pub fn verify_batched_at(
    vk: &VerifierKey,
    items: &[BatchVerifyItem<'_>],
    point: &[Fr],
    proof: &Proof,
    sponge: &mut impl CryptographicSponge,
) -> Result<bool, HyraxError> {
    if items.is_empty() {
        return Err(HyraxError::EmptyBatch);
    }
    let n_vars = point.len();
    if n_vars % 2 == 1 {
        return Err(HyraxError::OddNumVars { got: n_vars });
    }
    let dim = 1usize << (n_vars / 2);
    for (idx, item) in items.iter().enumerate() {
        if item.com.row_coms.len() != dim {
            return Err(HyraxError::HeterogenousBatchRows {
                idx,
                got: item.com.row_coms.len(),
                expected: dim,
            });
        }
    }

    sponge.absorb(&(items.len() as u64));
    for item in items.iter() {
        sponge.absorb(&serialize_compressed_vec(&item.com.row_coms));
        sponge.absorb(&item.value);
    }
    let rho = sponge.squeeze_field_elements::<Fr>(1)[0];
    let powers = powers_of(rho, items.len());

    let mut row_coms_batch_proj: Vec<<G1Affine as AffineRepr>::Group> =
        vec![<<G1Affine as AffineRepr>::Group as ark_std::Zero>::zero(); dim];
    let mut value_batch = Fr::from(0u64);
    for (k, item) in items.iter().enumerate() {
        let pk = powers[k];
        value_batch += pk * item.value;
        for j in 0..dim {
            row_coms_batch_proj[j] += item.com.row_coms[j] * pk;
        }
    }
    let row_coms_batch =
        <<G1Affine as AffineRepr>::Group as CurveGroup>::normalize_batch(&row_coms_batch_proj);
    let com_batch = HyraxCommitment {
        row_coms: row_coms_batch,
    };

    // Mirror the single-poly verify: bind y_batch before delegating.
    sponge.absorb(&value_batch);
    verify_inner(vk, &com_batch, point, proof, sponge)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_poly::DenseMultilinearExtension;
    use ark_poly::Polynomial;
    use ark_poly_commit::hyrax::HyraxPC;
    use ark_poly_commit::PolynomialCommitment;
    use ark_std::test_rng;

    fn rand_evals(n_vars: usize, rng: &mut impl RngCore) -> Vec<Fr> {
        (0..(1usize << n_vars)).map(|_| Fr::rand(rng)).collect()
    }

    fn fresh_sponge() -> ark_crypto_primitives::sponge::merlin::Transcript {
        let label: &[u8] = b"panda-pcs-native-test";
        <ark_crypto_primitives::sponge::merlin::Transcript as CryptographicSponge>::new(&label)
    }

    fn setup_ck(n_vars: usize, rng: &mut impl RngCore) -> CommitterKey {
        // Upstream's hash-derived setup, so commits stay verifiable
        // under upstream Hyrax too.
        type Inner = HyraxPC<G1Affine, DenseMultilinearExtension<Fr>>;
        let pp = Inner::setup(1, Some(n_vars), rng).unwrap();
        let (ck, _vk) = Inner::trim(&pp, 1, 0, None).unwrap();
        ck
    }

    fn lift_be_to_le(point_be: &[Fr], n_vars: usize) -> Vec<Fr> {
        // Matches `pcs_helpers::lift_point_to_max`: BE → LE, zero-pad.
        let mut out: Vec<Fr> = point_be.iter().rev().copied().collect();
        out.resize(n_vars, Fr::from(0u64));
        out
    }

    #[test]
    fn round_trip_commit_open_verify() {
        let mut rng = test_rng();
        for n_vars in [2usize, 4, 6, 8] {
            let ck = setup_ck(n_vars, &mut rng);
            let evals = rand_evals(n_vars, &mut rng);
            let (com, state) = commit(&ck, &evals, &mut rng).unwrap();
            // BE input mirrors the SNARK driver's convention.
            let point_be: Vec<Fr> = (0..n_vars).map(|_| Fr::rand(&mut rng)).collect();
            let point_le = lift_be_to_le(&point_be, n_vars);

            let expected_value = {
                let mle = DenseMultilinearExtension::from_evaluations_slice(n_vars, &evals);
                mle.evaluate(&point_le.to_vec())
            };
            let mut psponge = fresh_sponge();
            psponge.absorb(&expected_value);
            let (val, proof) =
                open_at(&ck, &evals, &state, &com, &point_le, &mut psponge, &mut rng).unwrap();
            assert_eq!(val, expected_value);

            let mut vsponge = fresh_sponge();
            vsponge.absorb(&val);
            let ok = verify_at(&ck, &com, &point_le, &proof, &mut vsponge).unwrap();
            assert!(ok, "honest open rejected");
        }
    }

    #[test]
    fn lying_value_rejected() {
        let mut rng = test_rng();
        let n_vars = 4;
        let ck = setup_ck(n_vars, &mut rng);
        let evals = rand_evals(n_vars, &mut rng);
        let (com, state) = commit(&ck, &evals, &mut rng).unwrap();
        let point_le: Vec<Fr> = (0..n_vars).map(|_| Fr::rand(&mut rng)).collect();

        let mle = DenseMultilinearExtension::from_evaluations_slice(n_vars, &evals);
        let true_val = mle.evaluate(&point_le.to_vec());

        let mut psponge = fresh_sponge();
        psponge.absorb(&true_val);
        let (_, proof) =
            open_at(&ck, &evals, &state, &com, &point_le, &mut psponge, &mut rng).unwrap();

        let mut vsponge = fresh_sponge();
        vsponge.absorb(&(true_val + Fr::from(1u64))); // lie
        let ok = verify_at(&ck, &com, &point_le, &proof, &mut vsponge).unwrap();
        assert!(!ok, "verifier accepted a lying claimed value");
    }

    #[test]
    fn batched_round_trip() {
        let mut rng = test_rng();
        let n_vars = 6;
        let ck = setup_ck(n_vars, &mut rng);
        let polys: Vec<Vec<Fr>> = (0..4).map(|_| rand_evals(n_vars, &mut rng)).collect();
        let mut coms = Vec::new();
        let mut states = Vec::new();
        for evals in &polys {
            let (c, s) = commit(&ck, evals, &mut rng).unwrap();
            coms.push(c);
            states.push(s);
        }
        let point_le: Vec<Fr> = (0..n_vars).map(|_| Fr::rand(&mut rng)).collect();

        let items: Vec<BatchOpenItem> = polys
            .iter()
            .enumerate()
            .map(|(k, evals)| BatchOpenItem {
                com: &coms[k],
                evals,
                state: &states[k],
            })
            .collect();

        let mut psponge = fresh_sponge();
        let (values, proof) =
            open_batched_at(&ck, &items, &point_le, &mut psponge, &mut rng).unwrap();

        // Independent reference: each value should match the MLE eval.
        for (k, evals) in polys.iter().enumerate() {
            let mle = DenseMultilinearExtension::from_evaluations_slice(n_vars, evals);
            assert_eq!(values[k], mle.evaluate(&point_le.to_vec()));
        }

        let vitems: Vec<BatchVerifyItem> = coms
            .iter()
            .zip(values.iter())
            .map(|(com, &value)| BatchVerifyItem { com, value })
            .collect();
        let mut vsponge = fresh_sponge();
        let ok = verify_batched_at(&ck, &vitems, &point_le, &proof, &mut vsponge).unwrap();
        assert!(ok, "honest batched open rejected");
    }

    #[test]
    fn batched_lying_value_rejected() {
        let mut rng = test_rng();
        let n_vars = 4;
        let ck = setup_ck(n_vars, &mut rng);
        let polys: Vec<Vec<Fr>> = (0..3).map(|_| rand_evals(n_vars, &mut rng)).collect();
        let mut coms = Vec::new();
        let mut states = Vec::new();
        for evals in &polys {
            let (c, s) = commit(&ck, evals, &mut rng).unwrap();
            coms.push(c);
            states.push(s);
        }
        let point_le: Vec<Fr> = (0..n_vars).map(|_| Fr::rand(&mut rng)).collect();

        let items: Vec<BatchOpenItem> = polys
            .iter()
            .enumerate()
            .map(|(k, evals)| BatchOpenItem {
                com: &coms[k],
                evals,
                state: &states[k],
            })
            .collect();

        let mut psponge = fresh_sponge();
        let (mut values, proof) =
            open_batched_at(&ck, &items, &point_le, &mut psponge, &mut rng).unwrap();

        values[1] += Fr::from(7u64);

        let vitems: Vec<BatchVerifyItem> = coms
            .iter()
            .zip(values.iter())
            .map(|(com, &value)| BatchVerifyItem { com, value })
            .collect();
        let mut vsponge = fresh_sponge();
        let ok = verify_batched_at(&ck, &vitems, &point_le, &proof, &mut vsponge).unwrap();
        assert!(!ok, "verifier accepted a lying y_k in the batched open");
    }

    #[test]
    fn batched_swapped_commits_rejected() {
        // Swapping (com, value) pairs on the verifier side absorbs a
        // different `(C, y)` sequence, so ρ diverges and verify fails.
        let mut rng = test_rng();
        let n_vars = 4;
        let ck = setup_ck(n_vars, &mut rng);
        let polys: Vec<Vec<Fr>> = (0..2).map(|_| rand_evals(n_vars, &mut rng)).collect();
        let (com0, st0) = commit(&ck, &polys[0], &mut rng).unwrap();
        let (com1, st1) = commit(&ck, &polys[1], &mut rng).unwrap();
        let point_le: Vec<Fr> = (0..n_vars).map(|_| Fr::rand(&mut rng)).collect();

        let items = [
            BatchOpenItem {
                com: &com0,
                evals: &polys[0],
                state: &st0,
            },
            BatchOpenItem {
                com: &com1,
                evals: &polys[1],
                state: &st1,
            },
        ];
        let mut psponge = fresh_sponge();
        let (values, proof) =
            open_batched_at(&ck, &items, &point_le, &mut psponge, &mut rng).unwrap();

        let vitems = [
            BatchVerifyItem {
                com: &com1,
                value: values[0],
            },
            BatchVerifyItem {
                com: &com0,
                value: values[1],
            },
        ];
        let mut vsponge = fresh_sponge();
        let ok = verify_batched_at(&ck, &vitems, &point_le, &proof, &mut vsponge).unwrap();
        assert!(!ok, "verifier accepted swapped (com, value) pairing");
    }
}
