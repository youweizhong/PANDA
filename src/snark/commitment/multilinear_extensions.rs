//! Small MLE (multilinear extension) helpers used across the SNARK.
//!
//! Big-endian variable convention: the first variable in any
//! evaluation point binds the highest-order bit of the flat index.
//! The arkworks Hyrax boundary uses the opposite convention, with
//! conversion handled in `pcs_helpers::lift_point_to_max`.

use ark_bn254::Fr;

/// Evaluate the multilinear extension of `evals` at point `r`.
/// `evals.len()` must equal `2^r.len()`.
pub fn eval_multilinear_full(evals: &[Fr], r: &[Fr]) -> Fr {
    debug_assert!(evals.len().is_power_of_two());
    debug_assert_eq!(evals.len(), 1 << r.len());
    let mut tab = evals.to_vec();
    for &ri in r {
        let half = tab.len() / 2;
        for i in 0..half {
            let delta = tab[half + i] - tab[i];
            tab[i] += ri * delta;
        }
        tab.truncate(half);
    }
    tab[0]
}

/// Fix the first `r.len()` (MSB) variables to `r` and return the
/// resulting evaluation table over the remaining LSB variables.
pub fn partial_eval_msb(evals: &[Fr], r: &[Fr]) -> Vec<Fr> {
    let mut tab = evals.to_vec();
    for &ri in r {
        let half = tab.len() / 2;
        for i in 0..half {
            let delta = tab[half + i] - tab[i];
            tab[i] += ri * delta;
        }
        tab.truncate(half);
    }
    tab
}

/// Fix the last `r_lsb.len()` (LSB) variables to `r_lsb`, returning
/// the resulting evaluation table over the remaining MSB variables.
pub fn partial_eval_lsb(evals: &[Fr], r_lsb: &[Fr], r_lsb_log: usize) -> Vec<Fr> {
    debug_assert_eq!(r_lsb.len(), r_lsb_log);
    let total = evals.len();
    let stride = 1usize << r_lsb_log;
    debug_assert_eq!(total % stride, 0);
    let n_msb = total / stride;
    let mut out = Vec::with_capacity(n_msb);
    for i in 0..n_msb {
        let row = &evals[i * stride..(i + 1) * stride];
        out.push(eval_multilinear_full(row, r_lsb));
    }
    out
}

/// Build the eq-poly evaluation table against `a`, MSB-first:
/// `tab[i] = ∏_k (a_k · b_k + (1 − a_k)(1 − b_k))` where `b` is the
/// MSB-first binary representation of `i`.
pub fn build_eq_table(a: &[Fr]) -> Vec<Fr> {
    let nv = a.len();
    let mut tab = vec![Fr::from(1u64); 1 << nv];
    for (k, &ak) in a.iter().enumerate() {
        let stride = 1usize << (nv - 1 - k);
        for block in (0..(1 << nv)).step_by(stride * 2) {
            for i in 0..stride {
                let lo = block + i;
                let hi = block + stride + i;
                let v_lo = tab[lo];
                let v_hi = tab[hi];
                tab[lo] = v_lo * (Fr::from(1u64) - ak);
                tab[hi] = v_hi * ak;
            }
        }
    }
    tab
}

/// Build the eq-poly table against `r_spec` over `{0,1}^lns`, then
/// tile it across `lni` extra LSB variables. The output has size
/// `2^{lns + lni}` and is constant in the j-axis bits.
pub fn build_eq_table_tiled(r_spec: &[Fr], lns: usize, lni: usize) -> Vec<Fr> {
    debug_assert_eq!(r_spec.len(), lns);
    let pow_lns = 1usize << lns;
    let pow_lni = 1usize << lni;
    let mut eq = vec![Fr::from(1u64); pow_lns];
    for (k, &ak) in r_spec.iter().enumerate() {
        let stride = 1usize << (lns - 1 - k);
        for block in (0..pow_lns).step_by(stride * 2) {
            for i in 0..stride {
                let lo = block + i;
                let hi = block + stride + i;
                let v_lo = eq[lo];
                let v_hi = eq[hi];
                eq[lo] = v_lo * (Fr::from(1u64) - ak);
                eq[hi] = v_hi * ak;
            }
        }
    }
    let mut out = vec![Fr::from(0u64); pow_lns * pow_lni];
    for i in 0..pow_lns {
        for j in 0..pow_lni {
            out[i * pow_lni + j] = eq[i];
        }
    }
    out
}

/// Tile a length-`2^lni` j-axis vector across `2^lns` i-rows.
/// The result is constant in the i-bits.
pub fn tile_j_along_i(j_table: &[Fr], lns: usize, lni: usize) -> Vec<Fr> {
    let pow_lni = 1usize << lni;
    let pow_lns = 1usize << lns;
    debug_assert_eq!(j_table.len(), pow_lni);
    let mut out = vec![Fr::from(0u64); pow_lns * pow_lni];
    for i in 0..pow_lns {
        out[i * pow_lni..(i + 1) * pow_lni].copy_from_slice(j_table);
    }
    out
}

/// Build the row-major MLE evaluation table of a `QArray2`,
/// zero-padded to next-pow2 in each dimension. Returns the table
/// plus `(log_rows, log_cols)`.
pub fn mle_table_from_matrix(
    m: &crate::quantization::quantized_array::QArray2,
) -> (Vec<Fr>, (usize, usize)) {
    let rows = m.nrows();
    let cols = m.ncols();
    let log_rows = next_pow2_log(rows);
    let log_cols = next_pow2_log(cols);
    let pow_rows = 1usize << log_rows;
    let pow_cols = 1usize << log_cols;
    let mut tab = vec![Fr::from(0u64); pow_rows * pow_cols];
    for i in 0..rows {
        for j in 0..cols {
            tab[i * pow_cols + j] =
                crate::snark_primitives::finite_field::signed_lift_to_fr(m.codes[[i, j]]);
        }
    }
    (tab, (log_rows, log_cols))
}

/// Build the MLE evaluation table of a `QArray1`, zero-padded to
/// next-pow2 length.
pub fn mle_table_from_vector(v: &crate::quantization::quantized_array::QArray1) -> Vec<Fr> {
    let len = v.len();
    let log_len = next_pow2_log(len);
    let pow_len = 1usize << log_len;
    let mut tab = vec![Fr::from(0u64); pow_len];
    for (slot, code) in tab.iter_mut().zip(v.codes.iter()).take(len) {
        *slot = crate::snark_primitives::finite_field::signed_lift_to_fr(*code);
    }
    tab
}

/// `ceil(log2(n))`, with `0` and `1` mapped to `0`.
pub fn next_pow2_log(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let next = n.next_power_of_two();
    next.trailing_zeros() as usize
}

/// Concatenate two `Fr` slices into a fresh `Vec`.
pub fn concat(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
    let mut v = Vec::with_capacity(a.len() + b.len());
    v.extend_from_slice(a);
    v.extend_from_slice(b);
    v
}

/// Evaluate the eq-polynomial `eq(a, b) = ∏_k (a_k b_k + (1 − a_k)(1 − b_k))`
/// at two equal-length points.
pub fn eval_eq(a: &[Fr], b: &[Fr]) -> Fr {
    debug_assert_eq!(a.len(), b.len());
    let one = Fr::from(1u64);
    let mut acc = one;
    for (ak, bk) in a.iter().zip(b.iter()) {
        acc *= *ak * *bk + (one - *ak) * (one - *bk);
    }
    acc
}
