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