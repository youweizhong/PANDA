//! Verifier rejects when a mandatory subproof field is `None`.
//!
//! The set of mandatory components is derived from the public
//! statement's `Property::Side`. Dropping any must surface as
//! `SnarkError::MissingRequiredComponent { what }` up-front, before any
//! FS work. Paired with `verify::require_mandatory_components`; adding
//! a new mandatory field there should add a test here.

use super::fixtures::{expect_reject_after_tamper, prove_small_relu};
use crate::snark::errors::SnarkError;

/// Drop one mandatory `Option<...>` field and assert the verifier
/// rejects with the matching `what` tag.
macro_rules! missing_component_test {
    ($test_name:ident, $field:ident, $what:literal) => {
        #[test]
        fn $test_name() {
            let mut p = prove_small_relu();
            p.proof.$field = None;
            let err = expect_reject_after_tamper(&p, concat!("missing ", $what));
            assert!(
                matches!(err, SnarkError::MissingRequiredComponent { what: $what }),
                "expected MissingRequiredComponent {{ what: {:?} }}, got {err:?}",
                $what,
            );
        }
    };
}

missing_component_test!(
    missing_public_binding_rejects,
    public_binding,
    "public_binding"
);
missing_component_test!(
    missing_output_bound_lower_rejects,
    output_bound_lower,
    "output_bound_lower"
);
missing_component_test!(
    missing_chain_init_lower_rejects,
    chain_init_lower,
    "chain_init_lower"
);
missing_component_test!(
    missing_activation_matrix_lower_rejects,
    activation_matrix_lower,
    "activation_matrix_lower"
);
missing_component_test!(
    missing_linear_backward_lower_rejects,
    linear_backward_lower,
    "linear_backward_lower"
);
missing_component_test!(
    missing_concretize_lower_rejects,
    concretize_lower,
    "concretize_lower"
);
// Two-sided property requires upper components too.
missing_component_test!(
    missing_output_bound_upper_rejects,
    output_bound_upper,
    "output_bound_upper"
);
