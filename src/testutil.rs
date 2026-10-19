//! Deterministic randomness for tests (SplitMix64). Not cryptographic.

use crate::ext::{Ext4, Fq4};
use crate::field::{Fp, Fq, Q};

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn fp<const P: u64>(&mut self) -> Fp<P> {
        Fp::new(self.next_u64())
    }

    pub fn fq(&mut self) -> Fq {
        self.fp::<Q>()
    }

    pub fn fq4(&mut self) -> Fq4 {
        Ext4([self.fq(), self.fq(), self.fq(), self.fq()])
    }
}