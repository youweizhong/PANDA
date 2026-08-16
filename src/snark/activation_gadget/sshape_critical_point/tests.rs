//! Unit tests for the critical-point gadget.

use super::super::sshape_endpoint::SshapeLineKind;
use super::prove_sshape_critical_point;
use crate::crown::network::ActivationKind;
use crate::quantization::scale::Scale;
use crate::snark::commitment::commit::{native_vector_n_vars, CommittedAux};
use crate::snark::params::SnarkParams;
use crate::snark_primitives::finite_field::signed_lift_to_fr;
use crate::snark_primitives::polynomial_commitment::{HyraxBn254, MlPcs};
use ark_bn254::Fr;
use ark_crypto_primitives::sponge::merlin::Transcript;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_std::rand::RngCore;
use ark_std::test_rng;

fn test_params() -> SnarkParams {
    use crate::crown::network::{Layer, Network};
    use crate::crown::output_property::{Property, Side};
    use ndarray::{array, Array1, Array2};
    let net = Network::new(vec![Layer::linear(
        array![[1.0]] as Array2<f64>,
        array![0.0] as Array1<f64>,
    )
    .unwrap()])
    .unwrap();
    let prop = Property::new(Array2::eye(1), Array1::zeros(1), Side::Both).unwrap();
    let mut rng = test_rng();
    SnarkParams::setup(
        &net,
        &prop,
        14,
        crate::snark::preprocess::test_shared(19, 19, 19),
        &mut rng,
    )
    .unwrap()
}

fn commit_vec(
    params: &SnarkParams,
    codes: &[i128],
    n_vars: usize,
    rng: &mut impl RngCore,
) -> (CommittedAux, <HyraxBn254 as MlPcs>::Commitment) {
    let n_padded = 1usize << n_vars;
    let mut padded: Vec<Fr> = codes.iter().map(|&v| signed_lift_to_fr(v)).collect();
    padded.resize(n_padded, Fr::from(0u64));
    let (commit, state) = HyraxBn254::commit(&params.committer_key, &padded, Some(rng)).unwrap();
    ((padded, state), commit)
}

/// Smoke test that the prover fails closed on a degenerate input
/// (flat line, slope 0) — any of the documented rejection paths is
/// acceptable.
#[test]
fn smoke_sigmoid_phase3c_returns_scale_precondition_error() {
    let params = test_params();
    let mut rng = test_rng();
    let n: usize = 4;
    let n_vars = native_vector_n_vars(n);
    let n_padded = 1usize << n_vars;
    let s_w = Scale::from_pow2(11);
    let s_d = Scale::from_pow2(0);
    let s_b = Scale::from_pow2(0);
    let d_codes = vec![0i128; n_padded];
    let b_codes = vec![0i128; n_padded];
    let (d_aux, d_commit) = commit_vec(&params, &d_codes, n_vars, &mut rng);
    let (b_aux, b_commit) = commit_vec(&params, &b_codes, n_vars, &mut rng);
    let preact_l: Vec<i128> = vec![-2048, -2048, -2048, -2048];
    let preact_u: Vec<i128> = vec![2048, 2048, 2048, 2048];
    let (preact_l_aux, preact_l_commit) = commit_vec(&params, &preact_l, n_vars, &mut rng);
    let (preact_u_aux, preact_u_commit) = commit_vec(&params, &preact_u, n_vars, &mut rng);
    let mut sponge = <Transcript as CryptographicSponge>::new(&b"sshape3c-test".as_slice());
    let r = prove_sshape_critical_point(
        7,
        ActivationKind::Sigmoid,
        SshapeLineKind::Upper,
        &preact_l,
        &preact_u,
        &preact_l_aux,
        &preact_l_commit,
        &preact_u_aux,
        &preact_u_commit,
        &d_aux,
        &d_commit,
        &b_aux,
        &b_commit,
        s_d,
        s_b,
        s_w,
        &params,
        &mut sponge,
        &mut rng,
    );
    assert!(
        matches!(
            r,
            Err(crate::snark::errors::SnarkError::RelaxationSoundnessFinalCheckFailed { .. })
                | Err(crate::snark::errors::SnarkError::RelaxationSoundnessSshapeInvalid { .. })
                | Err(crate::snark::errors::SnarkError::Reserved { .. })
        ),
        "expected fail-closed rejection from prove_sshape_critical_point, got {r:?}"
    );
}
