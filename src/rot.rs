//! Rotation matrices of R_q elements and the verifier's O(d) row fold.
//!
//! Multiplication by a \in R_q is the d*d matrix
//!     M_a = [cf(a), rot(cf(a)), …, rot^{d−1}(cf(a))],
//! with rot(v) = (−v_{d−1}, v_0, …, v_{d−2}), so M_a·cf(x) = cf(a·x).

use crate::ext::Fq4;
use crate::field::Fq;

/// rot(v) = (−v_{d−1}, v_0, …, v_{d−2}): multiplication by X in R_q.
pub fn rot(v: &[Fq]) -> Vec<Fq> {
    let d = v.len();
    let mut out = Vec::with_capacity(d);
    out.push(-v[d - 1]);
    out.extend_from_slice(&v[..d - 1]);
    out
}

/// Columns of M_a. O(d²) memory; reference and tests only.
pub fn rotation_matrix_columns(a: &[Fq]) -> Vec<Vec<Fq>> {
    let mut cols = Vec::with_capacity(a.len());
    let mut c = a.to_vec();
    for _ in 0..a.len() {
        let next = rot(&c);
        cols.push(c);
        c = next;
    }
    cols
}

/// u = \alpha^T M_a for \alpha = (1, \alpha, …, \alpha^{d−1}), in O(d), using
///     u_0 = a(\alpha),   u_{j+1} = \alpha u_j − (\alpha^d + 1) a_{d−1−j},
/// which holds because the top entry of rot^j(cf(a)) is a_{d−1−j}.
pub fn alpha_fold(a: &[Fq], alpha: Fq4) -> Vec<Fq4> {
    let d = a.len();
    let alpha_d_plus_1 = alpha.pow(d as u128) + Fq4::ONE;
    let mut u = Vec::with_capacity(d);
    let mut acc = Fq4::ZERO;
    for &ai in a.iter().rev() {
        acc = acc * alpha + Fq4::from_base(ai); // Horner
    }
    u.push(acc);
    for j in 0..d - 1 {
        let next = alpha * u[j] - alpha_d_plus_1.mul_base(a[d - 1 - j]);
        u.push(next);
    }
    u
}