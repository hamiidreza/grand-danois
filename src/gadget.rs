//! Gadget decomposition G^{-1} with balanced digits.
//!
//! g = (1, b, ..., b^{\delta-1}) with b = 2^{log_b}. Each x \in F_q becomes \delta digits
//! in [-b/2, b/2) with \sum_k x_k b^k \equiv x (mod q).

use crate::field::{Fq, Q};

pub struct Gadget {
    log_b: u32,
    delta: usize,
    lo: i64, // smallest integer with a \delta-digit balanced expansion
    hi: i64, // largest one
}

impl Gadget {
    pub fn new(log_b: u32, delta: usize) -> Self {
        assert!(
            log_b >= 1 && log_b as usize * delta <= 62,
            "unsupported gadget parameters"
        );
        let b = 1i64 << log_b;
        assert!(
            (b as i128).pow(delta as u32) >= Q as i128,
            "need b^delta >= q"
        );
        let s: i64 = (0..delta).map(|k| b.pow(k as u32)).sum(); // 1 + b + … + b^{\delta−1}
        Gadget {
            log_b,
            delta,
            lo: -(b / 2) * s,
            hi: (b / 2 - 1) * s,
        }
    }

    /// The \delta balanced digits of x, least significant first.
    pub fn decompose_scalar(&self, x: Fq) -> Vec<i64> {
        let b = 1i64 << self.log_b;
        let half = b / 2;
        let mut v = x.centered();
        if v > self.hi {
            v -= Q as i64;
        } else if v < self.lo {
            v += Q as i64;
        }
        let mut digits = Vec::with_capacity(self.delta);
        for _ in 0..self.delta {
            let r = (v + half).rem_euclid(b) - half; // balanced remainder in [−b/2, b/2)
            digits.push(r);
            v = (v - r) / b; // exact: v - r is divisible by b
        }
        debug_assert_eq!(v, 0, "carry left over");
        digits
    }

    /// s \in R_q -> (\hat{s}^(0), ..., \hat{s}^(\delta-1)), where coefficient t of \hat{s}^(k) is
    /// digit k of coefficient t of s. So s = \Sum_k b^k \hat{s}^(k), |\hat{s}^(k)|_\infty <= b/2.
    pub fn decompose_ring(&self, s: &[Fq]) -> Vec<Vec<Fq>> {
        let mut out = vec![vec![Fq::ZERO; s.len()]; self.delta];
        for (t, &x) in s.iter().enumerate() {
            for (k, r) in self.decompose_scalar(x).into_iter().enumerate() {
                out[k][t] = Fq::from_i64(r);
            }
        }
        out
    }

    /// G^{-1}: (s_0, ..., s_{\eta-1}) \in R_q^\eta -> \hat{s} \in R_q^{\eta \delta}, element-major,
    /// digit-minor: \hat{s} = (\hat{s}_0^(0), ..., \hat{s}_0^(\delta-1), \hat{s}_1^(0), ..., \hat{s}_1^(\delta-1), ...).
    pub fn decompose(&self, s: &[Vec<Fq>]) -> Vec<Vec<Fq>> {
        s.iter().flat_map(|si| self.decompose_ring(si)).collect()
    }

    /// G * \hat{s} with G = I_\eta \otimes (1, b, ..., b^{\delta-1}): each block of \delta consecutive
    /// elements of \hat{s} recombines to \Sum_k b^k \hat{s}^(k). Inverts `decompose`.
    pub fn recompose(&self, s_hat: &[Vec<Fq>]) -> Vec<Vec<Fq>> {
        assert_eq!(
            s_hat.len() % self.delta,
            0,
            "length must be a multiple of delta"
        );
        let b = Fq::from_i64(1i64 << self.log_b);
        s_hat
            .chunks(self.delta)
            .map(|block| {
                let mut acc = vec![Fq::ZERO; block[0].len()];
                let mut bk = Fq::ONE;
                for part in block {
                    for (a, &p) in acc.iter_mut().zip(part) {
                        *a += bk * p;
                    }
                    bk *= b;
                }
                acc
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Rng;

    #[test]
    fn hand_examples_base_2_16() {
        let g = Gadget::new(16, 2);
        let digits = |v: i64| g.decompose_scalar(Fq::from_i64(v));
        assert_eq!(digits(70000), vec![4464, 1]); // 4464 + 1*65536
        assert_eq!(digits(-1), vec![-1, 0]);
        assert_eq!(digits(32768), vec![-32768, 1]); // b/2 is not a digit: -b/2 + b
        // (q-1)/2 > hi, so it is decomposed as (q-1)/2 - q = -2147483599
        // = 49 - 32768 * 65536.
        assert_eq!(digits((Q as i64 - 1) / 2), vec![49, -32768]);
    }

    #[test]
    fn digits_are_short_and_recompose() {
        let mut rng = Rng::new(6);
        let h = (Q as i64 - 1) / 2;
        for &(log_b, delta) in &[(8, 4), (16, 2), (11, 3), (1, 32)] {
            let g = Gadget::new(log_b, delta);
            let b = 1i64 << log_b;
            let mut xs: Vec<Fq> = (0..1000).map(|_| rng.fq()).collect();
            for v in [0, 1, -1, h, -h, g.hi, g.hi + 1, g.lo.max(-h)] {
                xs.push(Fq::from_i64(v));
            }
            for x in xs {
                let digits = g.decompose_scalar(x);
                assert!(digits.iter().all(|&r| -b / 2 <= r && r < b / 2));
                let value: i128 = digits
                    .iter()
                    .rev()
                    .fold(0, |acc, &r| acc * b as i128 + r as i128);
                assert_eq!(
                    Fq::from_i128(value),
                    x,
                    "(log_b, delta) = ({log_b}, {delta})"
                );
            }
        }
    }

    #[test]
    fn decompose_is_element_major_and_inverts_recompose() {
        let mut rng = Rng::new(7);
        let (g, d, eta, delta) = (Gadget::new(8, 4), 64, 3, 4);
        let s: Vec<Vec<Fq>> = (0..eta).map(|_| rng.vec_fq(d)).collect();
        let s_hat = g.decompose(&s);
        assert_eq!(s_hat.len(), eta * delta);
        assert_eq!(s_hat[1], g.decompose_ring(&s[0])[1]); // digit 1 of s_0
        assert_eq!(s_hat[delta], g.decompose_ring(&s[1])[0]); // digit 0 of s_1
        assert!(s_hat.iter().flatten().all(|x| x.centered().abs() <= 128));
        assert_eq!(g.recompose(&s_hat), s);
    }
}
