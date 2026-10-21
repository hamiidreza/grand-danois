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
    psi_pow: Vec<Fp<P>>,             // psi^i for i < d
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
        let psi_pow = powers(psi, d);
        let psi_inv_pows_scaled = powers(psi_inv, d).into_iter().map(|x| x * d_inv).collect();
        let twiddles = powers(psi * psi, d / 2);
        let inv_twiddles = powers(psi_inv * psi_inv, d / 2);
        Self {
            d,
            psi_pow,
            psi_inv_pows_scaled,
            twiddles,
            inv_twiddles,
        }
    }

    pub fn degree(&self) -> usize {
        self.d
    }
}
