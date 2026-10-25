//! Multiplication in R_q = F_q[X]/(X^d + 1) and R_{q^4} = F_{q^4}[X]/(X^d + 1)
//! for the two-splitting modulus q = 2^32 − 99.
//!
//! There is no NTT mod q, so we multiply exactly over the integers via NTTs
//! modulo three auxiliary primes, recombine with CRT, and reduce mod q.

use crate::ext::W;
use crate::field::{Fp, P1, P2, P3, Q};
use crate::ntt::NegacyclicNtt;

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
}