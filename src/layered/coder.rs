//! The entropy coder shared by the SWL2 chunks: an adaptive binary range coder
//! (LZMA style, 12-bit probabilities, each the mean of a fast and a slow
//! estimate) with an adaptive Elias-gamma binarization that also models the two
//! bits below the leading one.
use crate::Error;

/// Probability resolution in bits.
const PROB_BITS: u32 = 12;

/// An adaptive bit probability: the mean of a fast and a slow estimate of
/// P(bit = 0), in units of 2^-PROB_BITS.
#[derive(Clone, Copy, Debug)]
pub(super) struct Prob {
    fast: u16,
    slow: u16,
}
impl Prob {
    pub(super) const NEW: Prob = Prob {
        fast: 1 << (PROB_BITS - 1),
        slow: 1 << (PROB_BITS - 1),
    };
    fn p(self) -> u32 {
        ((u32::from(self.fast) + u32::from(self.slow)) >> 1).clamp(31, (1 << PROB_BITS) - 31)
    }
    fn update(&mut self, bit: bool) {
        const FAST: u32 = 4;
        const SLOW: u32 = 7;
        if bit {
            self.fast -= self.fast >> FAST;
            self.slow -= self.slow >> SLOW;
        } else {
            self.fast += ((1 << PROB_BITS) - self.fast) >> FAST;
            self.slow += ((1 << PROB_BITS) - self.slow) >> SLOW;
        }
    }
}

pub(super) fn corrupt() -> Error {
    Error::InvalidParameter("SWL2 stream")
}

pub(super) struct RangeEncoder {
    low: u64,
    range: u32,
    cache: u8,
    pending: u64,
    out: Vec<u8>,
}
impl RangeEncoder {
    pub(super) fn new() -> Self {
        Self {
            low: 0,
            range: u32::MAX,
            cache: 0,
            pending: 1,
            out: Vec::new(),
        }
    }
    fn shift_low(&mut self) {
        if self.low < 0xFF00_0000 || self.low >= 1 << 32 {
            let carry = (self.low >> 32) as u8;
            let mut byte = self.cache;
            loop {
                self.out.push(byte.wrapping_add(carry));
                byte = 0xFF;
                self.pending -= 1;
                if self.pending == 0 {
                    break;
                }
            }
            self.cache = (self.low >> 24) as u8;
        }
        self.pending += 1;
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }
    fn normalize(&mut self) {
        while self.range < 1 << 24 {
            self.range <<= 8;
            self.shift_low();
        }
    }
    pub(super) fn bit(&mut self, p: &mut Prob, bit: bool) {
        let bound = (self.range >> PROB_BITS) * p.p();
        if bit {
            self.low += u64::from(bound);
            self.range -= bound;
        } else {
            self.range = bound;
        }
        p.update(bit);
        self.normalize();
    }
    /// `bits` equiprobable bits of `value`, most significant first.
    pub(super) fn direct(&mut self, value: u64, bits: u32) {
        for i in (0..bits).rev() {
            self.range >>= 1;
            if (value >> i) & 1 == 1 {
                self.low += u64::from(self.range);
            }
            self.normalize();
        }
    }
    /// Bytes written so far plus the bytes a flush would add: a size estimate
    /// for comparing alternatives.
    pub(super) fn size(&self) -> usize {
        self.out.len() + self.pending as usize + 4
    }
    pub(super) fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

pub(super) struct RangeDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    range: u32,
    code: u32,
}
impl<'a> RangeDecoder<'a> {
    pub(super) fn new(data: &'a [u8]) -> Result<Self, Error> {
        if data.len() < 5 || data[0] != 0 {
            return Err(corrupt());
        }
        let mut d = Self {
            data,
            pos: 0,
            range: u32::MAX,
            code: 0,
        };
        for _ in 0..5 {
            d.code = (d.code << 8) | u32::from(d.byte()?);
        }
        Ok(d)
    }
    fn byte(&mut self) -> Result<u8, Error> {
        let b = self.data.get(self.pos).copied().ok_or_else(corrupt)?;
        self.pos += 1;
        Ok(b)
    }
    fn normalize(&mut self) -> Result<(), Error> {
        while self.range < 1 << 24 {
            self.range <<= 8;
            self.code = (self.code << 8) | u32::from(self.byte()?);
        }
        Ok(())
    }
    pub(super) fn bit(&mut self, p: &mut Prob) -> Result<bool, Error> {
        let bound = (self.range >> PROB_BITS) * p.p();
        let bit = if self.code < bound {
            self.range = bound;
            false
        } else {
            self.code -= bound;
            self.range -= bound;
            true
        };
        p.update(bit);
        self.normalize()?;
        Ok(bit)
    }
    pub(super) fn direct(&mut self, bits: u32) -> Result<u64, Error> {
        let mut value = 0;
        for _ in 0..bits {
            self.range >>= 1;
            let b = self.code >= self.range;
            if b {
                self.code -= self.range;
            }
            value = (value << 1) | u64::from(b);
            self.normalize()?;
        }
        Ok(value)
    }
}

/// Bits below the leading one coded adaptively (a binary tree per length).
const MODELLED: usize = 2;

/// Adaptive Elias-gamma code of v+1: the length in unary with one context per
/// position, the first `MODELLED` bits below the leading one with a binary tree
/// of contexts per length, and the remaining bits direct.
#[derive(Clone)]
pub(super) struct Gamma {
    prefix: [Prob; 48],
    tree: [[Prob; 1 << MODELLED]; 48],
}
impl Gamma {
    pub(super) fn new() -> Self {
        Self {
            prefix: [Prob::NEW; 48],
            tree: [[Prob::NEW; 1 << MODELLED]; 48],
        }
    }
    pub(super) fn put(&mut self, e: &mut RangeEncoder, v: u64) {
        let m = v + 1;
        let width = (64 - m.leading_zeros()) as usize;
        for i in 0..width - 1 {
            e.bit(&mut self.prefix[i], true);
        }
        e.bit(&mut self.prefix[width - 1], false);
        let below = width - 1;
        let modelled = below.min(MODELLED);
        let mut node = 1;
        for i in 0..modelled {
            let bit = (m >> (below - 1 - i)) & 1;
            e.bit(&mut self.tree[width][node], bit == 1);
            node = 2 * node + bit as usize;
        }
        e.direct(m, (below - modelled) as u32);
    }
    pub(super) fn get(&mut self, d: &mut RangeDecoder) -> Result<u64, Error> {
        let mut width = 1;
        while d.bit(&mut self.prefix[width - 1])? {
            width += 1;
            if width > 46 {
                return Err(corrupt());
            }
        }
        let below = width - 1;
        let modelled = below.min(MODELLED);
        let mut node = 1;
        for _ in 0..modelled {
            node = 2 * node + usize::from(d.bit(&mut self.tree[width][node])?);
        }
        let rest = d.direct((below - modelled) as u32)?;
        // `node` holds the leading one and the modelled bits.
        Ok(((node as u64) << (below - modelled) | rest) - 1)
    }
}

/// Bits needed for `v` as an unsigned field (0 needs 0 bits).
pub(super) fn width(v: u64) -> u32 {
    64 - v.leading_zeros()
}

/// Bits needed for `v` as a two's complement field (0 needs 0 bits).
pub(super) fn signed_width(v: i64) -> u32 {
    if v == 0 {
        0
    } else if v > 0 {
        width(v as u64) + 1
    } else {
        width(!v as u64) + 1
    }
}

/// Little-endian reader of a chunk's byte header.
pub(super) struct Bytes<'a> {
    data: &'a [u8],
    pub(super) pos: usize,
}
impl<'a> Bytes<'a> {
    pub(super) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    pub(super) fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.pos.checked_add(n).ok_or_else(corrupt)?;
        let s = self.data.get(self.pos..end).ok_or_else(corrupt)?;
        self.pos = end;
        Ok(s)
    }
    pub(super) fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }
    pub(super) fn u16(&mut self) -> Result<u16, Error> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub(super) fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(super) fn f32(&mut self) -> Result<f32, Error> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(super) fn rest(&mut self) -> &'a [u8] {
        let s = &self.data[self.pos..];
        self.pos = self.data.len();
        s
    }
}
