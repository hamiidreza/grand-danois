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

/// Columns of M_a. O(d^2) memory; reference and tests only.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::RingMul;
    use crate::testutil::Rng;

    fn fq_vec(xs: &[i64]) -> Vec<Fq> {
        xs.iter().map(|&x| Fq::from_i64(x)).collect()
    }

    #[test]
    fn rot_is_multiplication_by_x() {
        // Worked example: a = 3 + X + 2X^3, so a * X = −2 + 3X + X^2.
        assert_eq!(rot(&fq_vec(&[3, 1, 0, 2])), fq_vec(&[-2, 3, 1, 0]));

        let mut rng = Rng::new(3);
        for &d in &[2, 64, 1024] {
            let rm = RingMul::new(d);
            let a = rng.vec_fq(d);
            let mut x = vec![Fq::ZERO; d];
            x[1] = Fq::ONE;
            assert_eq!(rot(&a), rm.mul(&a, &x), "d = {d}");
        }
    }

    /// \alpha^T * v = \Sum_i \alpha^i v_i: the fold of a single column, done directly.
    fn fold_column(col: &[Fq], alpha: Fq4) -> Fq4 {
        let mut acc = Fq4::ZERO;
        let mut p = Fq4::ONE;
        for &c in col {
            acc += p.mul_base(c);
            p *= alpha;
        }
        acc
    }

    #[test]
    fn alpha_fold_matches_explicit_fold() {
        let mut rng = Rng::new(4);
        for &d in &[1, 2, 4, 16, 128, 1024] {
            let (a, alpha) = (rng.vec_fq(d), rng.fq4());
            let explicit: Vec<Fq4> = rotation_matrix_columns(&a)
                .iter()
                .map(|col| fold_column(col, alpha))
                .collect();
            assert_eq!(alpha_fold(&a, alpha), explicit, "d = {d}");
        }
    }

    #[test]
    fn alpha_fold_of_monomials() {
        let d = 8;
        let alpha = Rng::new(5).fq4();
        let pows: Vec<Fq4> = (0..d).map(|i| alpha.pow(i as u128)).collect();

        // a = 1: M_a = I, so u = (1, \alpha, …, \alpha^{d−1}); the correction never fires.
        let mut one = vec![Fq::ZERO; d];
        one[0] = Fq::ONE;
        assert_eq!(alpha_fold(&one, alpha), pows);

        // a = X^{d−1}: column 0 is e_{d−1}, and every later column has wrapped
        // to −e_{j−1}, so u = (\alpha^{d−1}, −1, −\alpha, …, −\alpha^{d−2}).
        let mut top = vec![Fq::ZERO; d];
        top[d - 1] = Fq::ONE;
        let mut expected = vec![pows[d - 1]];
        expected.extend(pows[..d - 1].iter().map(|&p| -p));
        assert_eq!(alpha_fold(&top, alpha), expected);
    }
}
