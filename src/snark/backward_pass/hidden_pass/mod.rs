//! Hidden-layer preactivation bound proofs.
//!
//! For every Linear layer immediately followed by an Activation, the
//! float-CROWN engine derives the relaxation from preactivation
//! bounds at that layer. A hidden pass proves those bounds are
//! themselves valid CROWN bounds: it runs a smaller backward pass
//! starting from an identity `spec_c` at the target layer's output
//! and walking down to the input box.
//!
//! Structurally a hidden pass is a shorter version of the final
//! pass: same per-step gadgets (linear backward, activation
//! backward, ReLU LogUp, rescale, b_acc step, activation matrix,
//! concretize, output-bound inequality), driven on a truncated
//! chain. Shared commitments (weights, biases, relaxations, input
//! box) are reused from the final pass's `TensorCommitments`; only
//! chain tensors and per-step intermediates are committed fresh per
//! pass.
//!
//! Specific to hidden passes:
//!
//! - `pass_lower` / `pass_upper`: chain tensors + intermediates
//!   committed via `commit::commit_pass` on the shorter trace.
//! - `chain_init_from_identity`: pins `chain_a[target + 1]` to the
//!   canonical quantized identity and `chain_b_acc[target + 1]` to
//!   zero.
//! - per-direction `output_bound_inequality`: binds the committed
//!   preact bound to `b_acc_final + acc_w` via a slack range check.

mod prove;
mod verify;

pub(crate) use prove::prove_hidden_passes;
pub(crate) use verify::verify_hidden_passes;

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::CryptographicSponge;

use crate::snark::commitment::multilinear_extensions::eval_multilinear_full;

/// MLE evaluation of the `n × n` identity matrix quantized at scale
/// `one_code_scale`, at FS point `r` over the padded commit layout
/// `native_matrix_n_vars(n, n)`.
///
/// Builds the padded eval table directly (mostly zeros with
/// `one_code` on the diagonal) and runs `eval_multilinear_full`. `n`
/// is small for hidden layers, so the `O(2^n_vars)` cost is trivial
/// and avoids the padding-correction factors a closed-form
/// `eq(r_outer, r_inner)` would need.
pub(crate) fn identity_mle_eval(
    r: &[Fr],
    n: usize,
    one_code_scale: crate::quantization::scale::Scale,
) -> Fr {
    let log_n = if n <= 1 {
        0
    } else {
        n.next_power_of_two().trailing_zeros() as usize
    };
    let pow_n = 1usize << log_n;
    // Match Hyrax's even-bumped n_vars (see native_matrix_n_vars).
    let mut total_n_vars = log_n + log_n;
    if total_n_vars % 2 == 1 {
        total_n_vars += 1;
    }
    if total_n_vars < 2 {
        total_n_vars = 2;
    }
    debug_assert_eq!(r.len(), total_n_vars);

    let one_code = crate::quantization::quantized_scalar::Qf::from_real(1.0, one_code_scale).code;
    let one_code_fr = crate::snark_primitives::finite_field::signed_lift_to_fr(one_code);

    let table_len = 1usize << total_n_vars;
    let mut evals = vec![Fr::from(0u64); table_len];
    for i in 0..n {
        // Row-major layout from `commit::pad_matrix_native`.
        evals[i * pow_n + i] = one_code_fr;
    }
    eval_multilinear_full(&evals, r)
}

/// Absorb the FS tag identifying which hidden pass is being
/// processed. Prover and verifier must call this at the same
/// transcript position.
pub(crate) fn absorb_hidden_pass_tag(
    sponge: &mut impl CryptographicSponge,
    target_layer_idx: usize,
    n_spec: usize,
) {
    sponge.absorb(&(target_layer_idx as u64));
    sponge.absorb(&(n_spec as u64));
}
