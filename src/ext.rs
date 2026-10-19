//! Degree-4 extensions Z_p[Y]/(Y^4 − W), as one generic type `Ext4<P, W>`.
//!
//! The protocol extension field is F_{q^4} = F_q[Y]/(Y^4 − 2), about 2^128
//! elements. Y^4 − 2 is irreducible because 2 is a non-square mod q and
//! q ≡ 1 (mod 4) (Capelli's criterion); a test checks the non-square condition.
//!
//! The same type over the auxiliary primes, Ext4<P_i, 2>, is just the ring
//! used for exact CRT multiplication in R_{q^4}; there it need not be a field.

use crate::field::{Fp, Q};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Y^4 = W in the protocol extension.
pub const W: u64 = 2;

/// The protocol extension field F_{q^4}.
pub type Fq4 = Ext4<Q, W>;

#[derive(Clone, Copy, PartialEq, Eq, Default, Hash, Debug)]
pub struct Ext4<const P: u64, const W: u64>(pub [Fp<P>; 4]);

impl<const P: u64, const W: u64> Ext4<P, W> {
    pub const ZERO: Self = Ext4([Fp::ZERO; 4]);
    pub const ONE: Self = Ext4([Fp::ONE, Fp::ZERO, Fp::ZERO, Fp::ZERO]);

    /// Embed a base-field element.
    pub fn from_base(x: Fp<P>) -> Self {
        Ext4([x, Fp::ZERO, Fp::ZERO, Fp::ZERO])
    }

    /// Multiply by a base-field scalar (4 base multiplications).
    #[inline]
    pub fn mul_base(self, s: Fp<P>) -> Self {
        let a = self.0;
        Ext4([a[0] * s, a[1] * s, a[2] * s, a[3] * s])
    }

    pub fn pow(self, mut e: u128) -> Self {
        let mut base = self;
        let mut acc = Self::ONE;
        while e > 0 {
            if e & 1 == 1 {
                acc *= base;
            }
            base *= base;
            e >>= 1;
        }
        acc
    }

    /// Inverse via x^(p^4 − 2); valid when Y^4 − W is irreducible mod p.
    /// Slow but simple; only used off the hot path.
    pub fn inverse(self) -> Option<Self> {
        if self == Self::ZERO {
            None
        } else {
            Some(self.pow((P as u128).pow(4) - 2))
        }
    }
}

impl<const P: u64, const W: u64> Add for Ext4<P, W> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let (a, b) = (self.0, rhs.0);
        Ext4([a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])
    }
}

impl<const P: u64, const W: u64> Sub for Ext4<P, W> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let (a, b) = (self.0, rhs.0);
        Ext4([a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]])
    }
}

impl<const P: u64, const W: u64> Neg for Ext4<P, W> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let a = self.0;
        Ext4([-a[0], -a[1], -a[2], -a[3]])
    }
}

impl<const P: u64, const W: u64> Mul for Ext4<P, W> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = (self.0, rhs.0);
        let mut c = [Fp::<P>::ZERO; 7];
        for i in 0..4 {
            for j in 0..4 {
                c[i + j] += a[i] * b[j];
            }
        }
        // Reduce with Y^4 = W.
        let w = Fp::<P>::new(W);
        Ext4([c[0] + w * c[4], c[1] + w * c[5], c[2] + w * c[6], c[3]])
    }
}

impl<const P: u64, const W: u64> AddAssign for Ext4<P, W> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u64, const W: u64> SubAssign for Ext4<P, W> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u64, const W: u64> MulAssign for Ext4<P, W> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Fq;
    use crate::testutil::Rng;

    #[test]
    fn modulus_polynomial_is_irreducible() {
        // q ≡ 1 mod 4 and W a non-square  ⇒  Y^4 − W irreducible.
        assert_eq!(Q % 4, 1);
        assert!(!Fq::new(W).is_square());
    }

    #[test]
    fn y_to_the_fourth_is_w() {
        let y = Fq4::from_base(Fq::ZERO) + Ext4([Fq::ZERO, Fq::ONE, Fq::ZERO, Fq::ZERO]);
        assert_eq!(y * y * y * y, Fq4::from_base(Fq::new(W)));
    }

    #[test]
    fn field_axioms_and_inverse() {
        let mut rng = Rng::new(2);
        for _ in 0..300 {
            let (a, b, c) = (rng.fq4(), rng.fq4(), rng.fq4());
            assert_eq!(a * b, b * a);
            assert_eq!((a * b) * c, a * (b * c));
            assert_eq!(a * (b + c), a * b + a * c);
            assert_eq!(a - b + b, a);
            let s = rng.fq();
            assert_eq!(a.mul_base(s), a * Fq4::from_base(s));
        }
        for _ in 0..20 {
            let a = rng.fq4();
            assert_eq!(a * a.inverse().unwrap(), Fq4::ONE);
        }
    }
}