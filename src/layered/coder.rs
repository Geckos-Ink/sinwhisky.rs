//! Entropy and bit coders shared by the SWL2 chunks: an adaptive binary range
//! coder (LZMA style, 11-bit probabilities, shift 5) with an adaptive Elias-gamma
//! binarization, and an MSB-first bit packer for fixed-width fields.
use crate::Error;

pub(super) const PROB_INIT: u16 = 1024;

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
    pub(super) fn bit(&mut self, p: &mut u16, bit: bool) {
        let bound = (self.range >> 11) * u32::from(*p);
        if bit {
            self.low += u64::from(bound);
            self.range -= bound;
            *p -= *p >> 5;
        } else {
            self.range = bound;
            *p += (2048 - *p) >> 5;
        }
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
    pub(super) fn bit(&mut self, p: &mut u16) -> Result<bool, Error> {
        let bound = (self.range >> 11) * u32::from(*p);
        let bit = if self.code < bound {
            self.range = bound;
            *p += (2048 - *p) >> 5;
            false
        } else {
            self.code -= bound;
            self.range -= bound;
            *p -= *p >> 5;
            true
        };
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

/// Adaptive Elias-gamma code of v+1: the length in unary with one context per
/// position, the first bit below the leading one with a context per length, and
/// the remaining bits direct.
#[derive(Clone)]
pub(super) struct Gamma {
    prefix: [u16; 48],
    first: [u16; 48],
}
impl Gamma {
    pub(super) fn new() -> Self {
        Self {
            prefix: [PROB_INIT; 48],
            first: [PROB_INIT; 48],
        }
    }
    pub(super) fn put(&mut self, e: &mut RangeEncoder, v: u64) {
        let m = v + 1;
        let width = (64 - m.leading_zeros()) as usize;
        for i in 0..width - 1 {
            e.bit(&mut self.prefix[i], true);
        }
        e.bit(&mut self.prefix[width - 1], false);
        if width >= 2 {
            e.bit(&mut self.first[width], (m >> (width - 2)) & 1 == 1);
            e.direct(m, width as u32 - 2);
        }
    }
    pub(super) fn get(&mut self, d: &mut RangeDecoder) -> Result<u64, Error> {
        let mut width = 1;
        while d.bit(&mut self.prefix[width - 1])? {
            width += 1;
            if width > 46 {
                return Err(corrupt());
            }
        }
        if width == 1 {
            return Ok(0);
        }
        let first = u64::from(d.bit(&mut self.first[width])?);
        let rest = d.direct(width as u32 - 2)?;
        Ok(((1u64 << (width - 1)) | (first << (width - 2)) | rest) - 1)
    }
}

/// MSB-first bit packer for fixed-width fields.
pub(super) struct BitWriter {
    out: Vec<u8>,
    used: u32,
}
impl BitWriter {
    pub(super) fn new() -> Self {
        Self {
            out: Vec::new(),
            used: 8,
        }
    }
    pub(super) fn put(&mut self, value: u64, bits: u32) {
        for i in (0..bits).rev() {
            if self.used == 8 {
                self.out.push(0);
                self.used = 0;
            }
            if (value >> i) & 1 == 1 {
                *self.out.last_mut().unwrap() |= 0x80 >> self.used;
            }
            self.used += 1;
        }
    }
    /// Two's complement in `bits` bits; the caller guarantees the range.
    pub(super) fn put_signed(&mut self, value: i64, bits: u32) {
        if bits > 0 {
            self.put((value as u64) & mask(bits), bits);
        }
    }
    pub(super) fn finish(self) -> Vec<u8> {
        self.out
    }
}

pub(super) fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
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

pub(super) struct BitReader<'a> {
    data: &'a [u8],
    bit: usize,
}
impl<'a> BitReader<'a> {
    pub(super) fn new(data: &'a [u8]) -> Self {
        Self { data, bit: 0 }
    }
    pub(super) fn get(&mut self, bits: u32) -> Result<u64, Error> {
        if bits > 64 || self.bit + bits as usize > self.data.len() * 8 {
            return Err(corrupt());
        }
        let mut v = 0u64;
        for _ in 0..bits {
            let byte = self.data[self.bit / 8];
            v = (v << 1) | u64::from((byte >> (7 - self.bit % 8)) & 1);
            self.bit += 1;
        }
        Ok(v)
    }
    pub(super) fn get_signed(&mut self, bits: u32) -> Result<i64, Error> {
        if bits == 0 {
            return Ok(0);
        }
        let v = self.get(bits)?;
        let shift = 64 - bits;
        Ok(((v << shift) as i64) >> shift)
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
    pub(super) fn i8(&mut self) -> Result<i8, Error> {
        Ok(self.take(1)?[0] as i8)
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
