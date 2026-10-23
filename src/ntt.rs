//! Negacyclic NTT for T[X]/(X^d + 1) modulo an NTT-friendly prime P.

use crate::ext::Ext4;
use crate::field::Fp;
use core::ops::{Add, Mul, Sub};

/// Anything the NTT mod P can transform: an F_p-module.
pub trait Module<const P: u64>: Copy + Default + Add<Output = Self> + Sub<Output = Self> {
    fn scale(self, s: Fp<P>) -> Self;
}

/// A Module that is also a ring, so pointwise products make sense.
pub trait Algebra<const P: u64>: Module<P> + Mul<Output = Self> {}

impl<const P: u64> Module<P> for Fp<P> {
    #[inline]
    fn scale(self, s: Fp<P>) -> Self {
        self * s
    }
}
impl<const P: u64> Algebra<P> for Fp<P> {}

impl<const P: u64, const W: u64> Module<P> for Ext4<P, W> {
    #[inline]
    fn scale(self, s: Fp<P>) -> Self {
        self.mul_base(s)
    }
}
impl<const P: u64, const W: u64> Algebra<P> for Ext4<P, W> {}

/// Precomputed tables for one prime P and ring degree d.
pub struct NegacyclicNtt<const P: u64> {
    d: usize,
    psi_pows: Vec<Fp<P>>,            // psi^i for i < d
    psi_inv_pows_scaled: Vec<Fp<P>>, // psi^{-i} · d^{-1}
    twiddles: Vec<Fp<P>>,            // omega^i for i < d/2
    inv_twiddles: Vec<Fp<P>>,        // omega^{-i} for i < d/2
}

fn powers<const P: u64>(g: Fp<P>, n: usize) -> Vec<Fp<P>> {
    let mut v = Vec::with_capacity(n);
    let mut x = Fp::ONE;
    for _ in 0..n {
        v.push(x);
        x *= g;
    }
    v
}

impl<const P: u64> NegacyclicNtt<P> {
    /// Precompute tables for a negacyclic NTT of length d.
    pub fn new(d: usize) -> Self {
        assert!(
            d.is_power_of_two() && d >= 2,
            "NTT length must be a power of two"
        );
        let psi = Fp::<P>::primitive_root_of_unity(2 * d);
        let psi_inv = psi.inverse().unwrap();
        let d_inv = Fp::<P>::new(d as u64).inverse().unwrap();
        let psi_pows = powers(psi, d);
        let psi_inv_pows_scaled = powers(psi_inv, d).into_iter().map(|x| x * d_inv).collect();
        let twiddles = powers(psi * psi, d / 2);
        let inv_twiddles = powers(psi_inv * psi_inv, d / 2);
        Self {
            d,
            psi_pows,
            psi_inv_pows_scaled,
            twiddles,
            inv_twiddles,
        }
    }

    pub fn degree(&self) -> usize {
        self.d
    }

    /// Coefficients -> evaluation (NTT) form.
    pub fn forward<T: Module<P>>(&self, a: &mut [T]) {
        assert_eq!(a.len(), self.d);
        for (x, &p) in a.iter_mut().zip(&self.psi_pows) {
            *x = x.scale(p);
        }
        cyclic_ntt(a, &self.twiddles);
    }

    /// Evaluation (NTT) form -> coefficients.
    pub fn inverse<T: Module<P>>(&self, a: &mut [T]) {
        assert_eq!(a.len(), self.d);
        cyclic_ntt(a, &self.inv_twiddles);
        for (x, &p) in a.iter_mut().zip(&self.psi_inv_pows_scaled) {
            *x = x.scale(p);
        }
    }

    /// Product of two coefficient vectors in T[X]/(X^d + 1).
    pub fn mul<T: Algebra<P>>(&self, a: &[T], b: &[T]) -> Vec<T> {
        let mut fa = a.to_vec();
        let mut fb = b.to_vec();
        self.forward(&mut fa);
        self.forward(&mut fb);
        for (x, y) in fa.iter_mut().zip(&fb) {
            *x = *x * *y;
        }
        self.inverse(&mut fa);
        fa
    }
}

fn bit_reverse_permute<T>(a: &mut [T]) {
    let n = a.len();
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            a.swap(i, j);
        }
    }
}

/// Iterative radix-2 Cooley–Tukey; `tw[i] = omega^i` for a primitive n-th root omega.
fn cyclic_ntt<const P: u64, T: Module<P>>(a: &mut [T], tw: &[Fp<P>]) {
    let n = a.len();
    bit_reverse_permute(a);
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = n / len;
        for start in (0..n).step_by(len) {
            for j in 0..half {
                let u = a[start + j];
                let v = a[start + j + half].scale(tw[j * step]);
                a[start + j] = u + v;
                a[start + j + half] = u - v;
            }
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn show_tables_for_17() {
        let t = NegacyclicNtt::<17>::new(4);
        println!("psi_pows            = {:?}", t.psi_pows);
        println!("psi_inv_pows_scaled = {:?}", t.psi_inv_pows_scaled);
        println!("twiddles            = {:?}", t.twiddles);
        println!("inv_twiddles        = {:?}", t.inv_twiddles);
    }

    #[test]
    fn cyclic_ntt_evaluates_at_powers_of_omega() {
        // p = 17, n = 4, omega = 13: output j must be a(omega^j).
        let omega = Fp::<17>::new(13);
        let tw = powers(omega, 2);
        let a: Vec<Fp<17>> = [3, 1, 0, 2].iter().map(|&x| Fp::new(x)).collect();
        let mut out = a.clone();
        cyclic_ntt(&mut out, &tw);
        println!("NTT of (3, 1, 0, 2) = {:?}", out);
        for j in 0..4 {
            let x = omega.pow(j as u64);
            let mut val = Fp::ZERO;
            let mut x_pow = Fp::ONE;
            for &c in &a {
                val += c * x_pow;
                x_pow *= x;
            }
            assert_eq!(out[j], val, "wrong value at j = {j}");
        }
    }

    #[test]
    fn negacyclic_mul_small_example() {
        let t = NegacyclicNtt::<17>::new(4);
        let a: Vec<Fp<17>> = [3, 1, 0, 2].iter().map(|&x| Fp::new(x)).collect();
        let b: Vec<Fp<17>> = [1, 0, 2, 1].iter().map(|&x| Fp::new(x)).collect();

        let mut fa = a.clone();
        t.forward(&mut fa);
        let mut fb = b.clone();
        t.forward(&mut fb);
        println!("forward(a) = {:?}", fa);
        println!("forward(b) = {:?}", fb);

        let c = t.mul(&a, &b);
        println!("a * b      = {:?}", c);
        // (3 + X + 2X^3)(1 + 2X^2 + X^3) = 2 − 3X + 4X^2 + 7X^3 mod X^4 + 1
        let expected: Vec<Fp<17>> = [2, -3, 4, 7].iter().map(|&x| Fp::from_i64(x)).collect();
        assert_eq!(c, expected);

        let mut back = fa.clone();
        t.inverse(&mut back);
        assert_eq!(back, a);
    }
}
