//! Prime fields F_p for primes p < 2^32, as one generic type `Fp<P>`.
//!
//! The protocol field is F_q with q = 2^32 − 99. Since q ≡ 5 (mod 8), F_q has
//! a primitive 4th root of unity but no 8th, so X^d + 1 splits into exactly two
//! irreducible factors mod q. That keeps short elements invertible (good
//! challenge sets and extraction), at the price of no direct NTT mod q.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Protocol modulus q = 2^32 − 99, prime, q ≡ 5 (mod 8).
pub const Q: u64 = 4_294_967_197;
/// Auxiliary NTT primes for CRT multiplication.
pub const P1: u64 = 2_013_265_921; // BabyBear, 15·2^27 + 1
pub const P2: u64 = 2_130_706_433; // KoalaBear, 127·2^24 + 1
pub const P3: u64 = 998_244_353; //   119·2^23 + 1

/// The protocol field.
pub type Fq = Fp<Q>;

#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Fp<const P: u64>(u64);

impl<const P: u64> Fp<P> {
    pub const ZERO: Self = Fp(0);
    pub const ONE: Self = Fp(1);

    pub const fn new(x: u64) -> Self {
        Fp(x % P)
    }

    pub fn from_i64(x: i64) -> Self {
        Fp(x.rem_euclid(P as i64) as u64)
    }

    pub fn from_i128(x: i128) -> Self {
        Fp(x.rem_euclid(P as i128) as u64)
    }

    /// Canonical representative in [0, p).
    pub fn value(self) -> u64 {
        self.0
    }

    /// Centered representative in [−(p−1)/2, (p−1)/2], used for norms and lifting.
    pub fn centered(self) -> i64 {
        let v = self.0 as i64;
        if v > (P as i64 - 1) / 2 {
            v - P as i64
        } else {
            v
        }
    }

    pub fn pow(self, mut e: u64) -> Self {
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

    pub fn inverse(self) -> Option<Self> {
        if self.0 == 0 {
            None
        } else {
            Some(self.pow(P - 2))
        }
    }

    /// Euler's criterion. Zero counts as a square.
    pub fn is_square(self) -> bool {
        self.0 == 0 || self.pow((P - 1) / 2) == Self::ONE
    }

    /// The exponent s of the largest power of two dividing p − 1.
    pub const fn two_adicity() -> u32 {
        (P - 1).trailing_zeros()
    }

    /// Smallest integer g ≥ 2 that is a non-square mod p.
    pub fn smallest_non_square() -> Self {
        let mut g = 2;
        loop {
            let x = Self::new(g);
            if !x.is_square() {
                return x;
            }
            g += 1;
        }
    }

    /// A primitive `order`-th root of unity; `order` must be a power of two
    /// dividing p − 1. For a non-square g, g^((p−1)/order) has exact order.
    pub fn primitive_root_of_unity(order: usize) -> Self {
        assert!(order.is_power_of_two(), "order must be a power of two");
        assert!(
            order.trailing_zeros() <= Self::two_adicity(),
            "no primitive {order}-th root of unity mod {P}"
        );
        Self::smallest_non_square().pow((P - 1) / order as u64)
    }
}

impl<const P: u64> Add for Fp<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let s = self.0 + rhs.0; // < 2^33
        Fp(if s >= P { s - P } else { s })
    }
}

impl<const P: u64> Sub for Fp<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Fp(if self.0 >= rhs.0 {
            self.0 - rhs.0
        } else {
            self.0 + P - rhs.0
        })
    }
}

impl<const P: u64> Mul for Fp<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Fp((self.0 * rhs.0) % P) // < 2^64 since p < 2^32
    }
}

impl<const P: u64> Neg for Fp<P> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Fp(if self.0 == 0 { 0 } else { P - self.0 })
    }
}

impl<const P: u64> AddAssign for Fp<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u64> SubAssign for Fp<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u64> MulAssign for Fp<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u64> fmt::Debug for Fp<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<const P: u64> fmt::Display for Fp<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Rng;

    fn is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        let mut i = 2;
        while i * i <= n {
            if n % i == 0 {
                return false;
            }
            i += 1;
        }
        true
    }

    #[test]
    fn protocol_modulus_is_two_splitting() {
        assert!(is_prime(Q));
        assert_eq!(Q, (1u64 << 32) - 99);
        assert_eq!(Q % 8, 5);
        assert_eq!(Fq::two_adicity(), 2);
        // A primitive 4th root of unity i exists: i^2 = −1.
        let i = Fq::primitive_root_of_unity(4);
        assert_eq!(i * i, -Fq::ONE);
    }

    #[test]
    #[should_panic(expected = "no primitive 8-th root")]
    fn no_eighth_root_mod_q() {
        Fq::primitive_root_of_unity(8);
    }

    #[test]
    fn auxiliary_primes_support_d_up_to_2048() {
        for p in [P1, P2, P3] {
            assert!(is_prime(p), "{p} not prime");
            assert!((p - 1).trailing_zeros() >= 12, "{p}: no 4096-th root");
            assert!(p < (1 << 32));
        }
    }

    #[test]
    fn roots_of_unity_have_exact_order() {
        for log in 1..=Fp::<P1>::two_adicity() {
            let n = 1usize << log;
            let w = Fp::<P1>::primitive_root_of_unity(n);
            assert_eq!(w.pow(n as u64), Fp::ONE);
            assert_eq!(
                w.pow((n / 2) as u64),
                -Fp::<P1>::ONE,
                "order not exact for n = {n}"
            );
        }
    }

    fn check_axioms<const P: u64>(seed: u64) {
        let mut rng = Rng::new(seed);
        for _ in 0..1000 {
            let (a, b, c) = (rng.fp::<P>(), rng.fp::<P>(), rng.fp::<P>());
            assert_eq!(a + b, b + a);
            assert_eq!(a * b, b * a);
            assert_eq!((a * b) * c, a * (b * c));
            assert_eq!(a * (b + c), a * b + a * c);
            assert_eq!(a - b + b, a);
            assert_eq!(a + (-a), Fp::ZERO);
            if a != Fp::ZERO {
                assert_eq!(a * a.inverse().unwrap(), Fp::ONE);
            }
        }
        assert_eq!(Fp::<P>::ZERO.inverse(), None);
    }

    #[test]
    fn field_axioms() {
        check_axioms::<Q>(1);
        check_axioms::<P1>(2);
        check_axioms::<P3>(3);
    }

    #[test]
    fn centered_representatives() {
        assert_eq!(Fq::from_i64(-1).centered(), -1);
        assert_eq!(Fq::from_i64(5).centered(), 5);
        let half = (Q as i64 - 1) / 2;
        assert_eq!(Fq::from_i64(half).centered(), half);
        assert_eq!(Fq::from_i64(-half).centered(), -half);
    }
}
