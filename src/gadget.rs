//! Gadget decomposition G^{-1} with balanced digits.
//!
//! g = (1, b, …, b^{\delta−1}) with b = 2^{log_b}. Each x \in F_q becomes \delta digits
//! in [−b/2, b/2) with \sum_k x_k b^k \equiv x (mod q).

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
            v = (v - r) / b; // exact: v − r is divisible by b
        }
        debug_assert_eq!(v, 0, "carry left over");
        digits
    }

    /// s \in R_q -> (\hat{s}^(0), ..., \hat{s}^(\delta−1)), where coefficient t of \hat{s}^(k) is
    /// digit k of coefficient t of s. So s = \Sum_k b^k \delta{s}^(k), |\hat{s}^(k)|_\infty <= b/2.
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
    /// digit-minor: \hat{s} = (\hat{s}_0^(0), ..., \hat{s}_0^(\delta-1), \hat{s}_1^(0), ..., \hat{s}_1^(\delta−1), ...).
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
