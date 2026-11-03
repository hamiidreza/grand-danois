//! Multiplication in R_q = F_q[X]/(X^d + 1) and R_{q^4} = F_{q^4}[X]/(X^d + 1)
//! for the two-splitting modulus q = 2^32 − 99.
//!
//! There is no NTT mod q, so we multiply exactly over the integers via NTTs
//! modulo three auxiliary primes, recombine with CRT, and reduce mod q.

use crate::ext::{Ext4, Fq4, W};
use crate::field::{Fp, Fq, P1, P2, P3, Q};
use crate::ntt::NegacyclicNtt;

/// Lift x \in F_q to its centered integer, then reduce mod P.
#[inline]
fn lift<const P: u64>(x: Fq) -> Fp<P> {
    Fp::<P>::from_i64(x.centered())
}

/// The product a*b as an integer polynomial, reduced mod one auxiliary prime P.
fn residues<const P: u64>(ntt: &NegacyclicNtt<P>, a: &[Fq], b: &[Fq]) -> Vec<Fp<P>> {
    let la: Vec<Fp<P>> = a.iter().map(|&x| lift(x)).collect();
    let lb: Vec<Fp<P>> = b.iter().map(|&x| lift(x)).collect();
    ntt.mul(&la, &lb)
}

/// Lift each of the four F_q coefficients of x \in F_{q^4}, keeping Y^4 = W.
#[inline]
fn lift4<const P: u64>(x: Fq4) -> Ext4<P, W> {
    Ext4([lift(x.0[0]), lift(x.0[1]), lift(x.0[2]), lift(x.0[3])])
}

/// Same as `residues`, but for R_{q^4}: coefficients are Ext4 elements mod P.
fn residues4<const P: u64>(ntt: &NegacyclicNtt<P>, a: &[Fq4], b: &[Fq4]) -> Vec<Ext4<P, W>> {
    let la: Vec<Ext4<P, W>> = a.iter().map(|&x| lift4(x)).collect();
    let lb: Vec<Ext4<P, W>> = b.iter().map(|&x| lift4(x)).collect();
    ntt.mul(&la, &lb)
}

pub struct RingMul {
    d: usize,
    ntt1: NegacyclicNtt<P1>,
    ntt2: NegacyclicNtt<P2>,
    ntt3: NegacyclicNtt<P3>,
    p1_inv_mod_p2: u128,
    p1p2_inv_mod_p3: u128,
}

impl RingMul {
    pub fn new(d: usize) -> Self {
        let m = P1 as u128 * P2 as u128 * P3 as u128;
        let half_q = ((Q - 1) / 2) as u128;
        let worst = d as u128 * 4 * W as u128 * half_q * half_q;
        assert!(worst < m / 2, "CRT moduli too small for d = {d}");
        RingMul {
            d,
            ntt1: NegacyclicNtt::new(d),
            ntt2: NegacyclicNtt::new(d),
            ntt3: NegacyclicNtt::new(d),
            p1_inv_mod_p2: Fp::<P2>::new(P1).inverse().unwrap().value() as u128,
            p1p2_inv_mod_p3: Fp::<P3>::new((P1 * P2) % P3).inverse().unwrap().value() as u128,
        }
    }

    pub fn degree(&self) -> usize {
        self.d
    }

    fn crt(&self, r1: u64, r2: u64, r3: u64) -> Fq {
        let (p1, p2, p3) = (P1 as u128, P2 as u128, P3 as u128);
        let x1 = r1 as u128;
        let t2 = (r2 as u128 + p2 - x1 % p2) % p2 * self.p1_inv_mod_p2 % p2;
        let x12 = x1 + t2 * p1; // exact value mod p1*p2
        let t3 = (r3 as u128 + p3 - x12 % p3) % p3 * self.p1p2_inv_mod_p3 % p3;
        let x = x12 + t3 * p1 * p2; // exact value mod M=p1*p2*p3, in [0, M)
        let m = p1 * p2 * p3;
        let signed = if x < m / 2 {
            x as i128
        } else {
            x as i128 - m as i128
        };
        Fq::from_i128(signed)
    }

    /// Product in R_q.
    pub fn mul(&self, a: &[Fq], b: &[Fq]) -> Vec<Fq> {
        assert_eq!(a.len(), self.d);
        assert_eq!(b.len(), self.d);
        let r1 = residues(&self.ntt1, a, b);
        let r2 = residues(&self.ntt2, a, b);
        let r3 = residues(&self.ntt3, a, b);
        (0..self.d)
            .map(|i| self.crt(r1[i].value(), r2[i].value(), r3[i].value()))
            .collect()
    }

    /// Product in R_{q^4}.
    pub fn mul4(&self, a: &[Fq4], b: &[Fq4]) -> Vec<Fq4> {
        assert_eq!(a.len(), self.d);
        assert_eq!(b.len(), self.d);
        let r1 = residues4(&self.ntt1, a, b);
        let r2 = residues4(&self.ntt2, a, b);
        let r3 = residues4(&self.ntt3, a, b);
        (0..self.d)
            .map(|i| {
                let mut out = [Fq::ZERO; 4];
                for (c, o) in out.iter_mut().enumerate() {
                    *o = self.crt(r1[i].0[c].value(), r2[i].0[c].value(), r3[i].0[c].value());
                }
                Ext4(out)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rot::rotation_matrix_columns;
    use crate::testutil::Rng;

    /// Reference product in R_q: M_a * cf(x), computed directly in F_q.
    fn reference_mul(a: &[Fq], x: &[Fq]) -> Vec<Fq> {
        let mut out = vec![Fq::ZERO; a.len()];
        for (col, &xj) in rotation_matrix_columns(a).iter().zip(x) {
            for (o, &c) in out.iter_mut().zip(col) {
                *o += c * xj;
            }
        }
        out
    }

    /// Reference product in R_{q^4}: split a = \Sum_k a^(k) Y^k into four R_q
    /// components, multiply all pairs with `reference_mul`, and fold Y^4 = W.
    fn reference_mul4(a: &[Fq4], b: &[Fq4]) -> Vec<Fq4> {
        let w = Fq::new(W);
        let mut out = vec![Fq4::ZERO; a.len()];
        for k in 0..4 {
            for l in 0..4 {
                let ak: Vec<Fq> = a.iter().map(|x| x.0[k]).collect();
                let bl: Vec<Fq> = b.iter().map(|x| x.0[l]).collect();
                let prod = reference_mul(&ak, &bl);
                let (slot, factor) = if k + l < 4 {
                    (k + l, Fq::ONE)
                } else {
                    (k + l - 4, w)
                };
                for (o, &p) in out.iter_mut().zip(&prod) {
                    o.0[slot] += factor * p;
                }
            }
        }
        out
    }

    #[test]
    fn mul_matches_rotation_matrix() {
        let mut rng = Rng::new(1);
        for &d in &[2, 64, 1024] {
            let rm = RingMul::new(d);
            let (a, x) = (rng.vec_fq(d), rng.vec_fq(d));
            assert_eq!(rm.mul(&a, &x), reference_mul(&a, &x), "d = {d}");
        }
    }

    #[test]
    fn mul_is_exact_at_the_crt_bound() {
        let d = 1024;
        let rm = RingMul::new(d);
        let h = Fq::from_i64((Q as i64 - 1) / 2); // largest centered value
        for (a0, x0) in [(h, h), (h, -h)] {
            let (a, x) = (vec![a0; d], vec![x0; d]);
            assert_eq!(rm.mul(&a, &x), reference_mul(&a, &x));
        }
    }

    #[test]
    fn mul4_matches_componentwise_reference() {
        let mut rng = Rng::new(2);
        for &d in &[2, 64, 1024] {
            let rm = RingMul::new(d);
            let (a, b) = (rng.vec_fq4(d), rng.vec_fq4(d));
            assert_eq!(rm.mul4(&a, &b), reference_mul4(&a, &b), "d = {d}");
        }
    }

    #[test]
    fn mul4_is_exact_at_the_crt_bound() {
        let d = 1024;
        let rm = RingMul::new(d);
        let h = Fq::from_i64((Q as i64 - 1) / 2);
        for (a0, b0) in [(h, h), (h, -h)] {
            let (a, b) = (vec![Ext4([a0; 4]); d], vec![Ext4([b0; 4]); d]);
            assert_eq!(rm.mul4(&a, &b), reference_mul4(&a, &b));
        }
    }
}
