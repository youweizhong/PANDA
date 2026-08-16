//! End-to-end tests for the LogUp-GKR primitives via the public surface
//! (`Fraction`, `LogUpCircuit`, `prove_circuit`, `verify_circuit_with_top`).

use super::*;
use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::{One, Zero};
use ark_std::{test_rng, UniformRand};

fn fresh_sponge() -> ark_crypto_primitives::sponge::merlin::Transcript {
    <ark_crypto_primitives::sponge::merlin::Transcript as CryptographicSponge>::new(
        &b"panda-logup-test".as_slice(),
    )
}

#[test]
fn fraction_addition_is_associative() {
    let a = Fraction::new(Fr::from(1u64), Fr::from(3u64));
    let b = Fraction::new(Fr::from(2u64), Fr::from(5u64));
    let c = Fraction::new(Fr::from(4u64), Fr::from(7u64));
    let lhs = (a + b) + c;
    let rhs = a + (b + c);
    // Cross-multiplication: the un-reduced rationals are equal.
    assert_eq!(
        lhs.numerator * rhs.denominator,
        rhs.numerator * lhs.denominator
    );
}

#[test]
fn lookup_circuit_output_matches_direct_sum() {
    let mut rng = test_rng();
    let alpha = Fr::rand(&mut rng);
    let witness: Vec<Fr> = (1..=8).map(|i| Fr::from(i as u64)).collect();
    let circuit = LogUpCircuit::lookup(&witness, alpha).unwrap();
    let direct: Fraction<Fr> = witness
        .iter()
        .map(|w| Fraction::new(-Fr::one(), *w - alpha))
        .reduce(|a, b| a + b)
        .unwrap();
    let out = circuit.output();
    assert_eq!(
        direct.numerator * out.denominator,
        out.numerator * direct.denominator
    );
}

#[test]
fn logup_identity_holds_when_witness_is_in_table() {
    let mut rng = test_rng();
    let table: Vec<Fr> = (0..8).map(Fr::from).collect();
    let witness_idx = vec![0u64, 0, 3, 5, 5, 5, 7, 7];
    let witness: Vec<Fr> = witness_idx.iter().copied().map(Fr::from).collect();
    let mut multiplicities = vec![Fr::zero(); 8];
    for &i in &witness_idx {
        multiplicities[i as usize] += Fr::one();
    }
    let alpha = Fr::rand(&mut rng);
    let lookup = LogUpCircuit::lookup(&witness, alpha).unwrap();
    let table_circ = LogUpCircuit::table(&table, &multiplicities, alpha).unwrap();
    let lo_top = lookup.output();
    let ta_top = table_circ.output();
    let combined = lo_top.numerator * ta_top.denominator + lo_top.denominator * ta_top.numerator;
    assert_eq!(combined, Fr::zero(), "LogUp identity should hold");
}

#[test]
fn logup_identity_breaks_for_out_of_table_witness() {
    let mut rng = test_rng();
    let table: Vec<Fr> = (0..8).map(Fr::from).collect();
    let witness: Vec<Fr> = vec![Fr::from(99u64); 8]; // not in table
    let multiplicities = vec![Fr::zero(); 8];
    let alpha = Fr::rand(&mut rng);
    let lookup = LogUpCircuit::lookup(&witness, alpha).unwrap();
    let table_circ = LogUpCircuit::table(&table, &multiplicities, alpha).unwrap();
    let lo_top = lookup.output();
    let ta_top = table_circ.output();
    let combined = lo_top.numerator * ta_top.denominator + lo_top.denominator * ta_top.numerator;
    assert_ne!(combined, Fr::zero());
}

#[test]
fn prove_verify_roundtrip_lookup_circuit() {
    let mut rng = test_rng();
    let witness: Vec<Fr> = (1..=8).map(|i| Fr::from(i as u64)).collect();
    let alpha = Fr::rand(&mut rng);
    let circuit = LogUpCircuit::lookup(&witness, alpha).unwrap();
    let top = circuit.output();
    let top_layer = circuit.layers.last().unwrap();
    let top_halves = match top_layer {
        LogUpLayer::Generic {
            numerator,
            denominator,
        } => [numerator[0], numerator[1], denominator[0], denominator[1]],
        _ => panic!("top layer should be Generic for n>1"),
    };
    let mut prover_sponge = fresh_sponge();
    let proof = prove_circuit(&circuit, &mut prover_sponge).unwrap();

    let top_num = top_halves[0] * top_halves[3] + top_halves[1] * top_halves[2];
    assert_eq!(top.numerator, top_num);
    let mut verifier_sponge = fresh_sponge();
    verify_circuit_with_top(
        &proof,
        circuit.num_vars(),
        top_halves,
        top_num,
        &mut verifier_sponge,
    )
    .unwrap();
}

#[test]
fn tampered_round_poly_breaks_verifier() {
    let mut rng = test_rng();
    let witness: Vec<Fr> = (1..=8).map(|i| Fr::from(i as u64)).collect();
    let alpha = Fr::rand(&mut rng);
    let circuit = LogUpCircuit::lookup(&witness, alpha).unwrap();
    let top_layer = circuit.layers.last().unwrap();
    let top_halves = match top_layer {
        LogUpLayer::Generic {
            numerator,
            denominator,
        } => [numerator[0], numerator[1], denominator[0], denominator[1]],
        _ => unreachable!(),
    };
    let top_num = top_halves[0] * top_halves[3] + top_halves[1] * top_halves[2];
    let mut prover_sponge = fresh_sponge();
    let mut proof = prove_circuit(&circuit, &mut prover_sponge).unwrap();
    proof.layers[0].rounds[0].at_zero += Fr::from(1u64);
    let mut verifier_sponge = fresh_sponge();
    let verdict = verify_circuit_with_top(
        &proof,
        circuit.num_vars(),
        top_halves,
        top_num,
        &mut verifier_sponge,
    );
    assert!(verdict.is_err());
}

/// Suppress dead-code warnings on the currently-unused `prove_lookup`
/// and `verify_circuit` helpers; both are public API for the SNARK
/// driver to come.
#[test]
fn _api_smoke() {
    let _ = prove_lookup::<Fr, ark_crypto_primitives::sponge::merlin::Transcript>;
    let _ = verify_circuit::<Fr, ark_crypto_primitives::sponge::merlin::Transcript>;
}

/// Regression: trivial-circuit (size-2 table) prove/verify round-trip.
/// The prover must β-fold the top, not ship the aggregated fraction,
/// or the verifier's `current_num = (1-β)·n0 + β·n1` diverges.
#[test]
fn prove_verify_roundtrip_trivial_table_circuit() {
    let mut rng = test_rng();
    let table: Vec<Fr> = vec![Fr::from(0u64), Fr::from(1u64)];
    let multiplicities: Vec<Fr> = vec![Fr::from(2u64), Fr::from(3u64)];
    let alpha = Fr::rand(&mut rng);
    let circuit = LogUpCircuit::table(&table, &multiplicities, alpha).unwrap();
    assert_eq!(
        circuit.layers.len(),
        1,
        "size-2 table → single-layer circuit"
    );
    assert_eq!(circuit.num_vars(), 0, "num_vars = 0 for trivial circuit");

    let top_layer = circuit.layers.last().unwrap();
    let top_halves = match top_layer {
        LogUpLayer::InitialTable {
            numerator,
            denominator,
        } => [numerator[0], numerator[1], denominator[0], denominator[1]],
        _ => unreachable!("trivial table circuit is InitialTable"),
    };
    let top_num = top_halves[0] * top_halves[3] + top_halves[1] * top_halves[2];

    let mut prover_sponge = fresh_sponge();
    let proof = prove_circuit(&circuit, &mut prover_sponge).unwrap();
    assert!(proof.layers.is_empty(), "no sumcheck transitions");

    let mut verifier_sponge = fresh_sponge();
    verify_circuit_with_top(
        &proof,
        circuit.num_vars(),
        top_halves,
        top_num,
        &mut verifier_sponge,
    )
    .expect("trivial-circuit round-trip must verify");

    // Regression sanity: bottom_point must be `[β]`, not `[]` as the
    // earlier buggy version emitted.
    assert_eq!(
        proof.bottom_point.len(),
        1,
        "trivial-circuit bottom_point must have length 1 (the squeezed β)"
    );
}
