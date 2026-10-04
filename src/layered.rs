//! Layered closed-loop coding (PAPER_EQ_COMPRESSION.md, Section 4): an
//! experimental `SWL1` byte stream, not a stable or fuzz-validated format.
//!
//! Layer 1 (tonal): the fitted tones of a `ToneAnalysis` below `tonal_upper` are
//! linked into partial tracks across frames (centres k*hop). A track is a series
//! of breakpoints (frequency, amplitude, phase); frequencies are predicted by the
//! track's own gradient (linear extrapolation of the last two breakpoints) and
//! phases by integrating the frequency, so only the changes of gradient and the
//! phase corrections are coded. Quantization steps follow the decoded amplitude,
//! so that each parameter costs about `tone_precision` of waveform error. The
//! decoder renders the tracks with McAulay-Quatieri cubic phase between
//! breakpoints, linear amplitude, and a fade of one hop at births and deaths.
//!
//! Layer 2 (integer corrections): the source minus the rounded decoded tonal
//! track, predicted per block of 4096 samples by a fixed closed-loop polynomial
//! (order 0, 1 or 2, chosen by the encoder) and quantized with step `2*tolerance+1`.
//! Every decoded sample is therefore within `tolerance` of the source; 0 is
//! lossless. Samples within the tolerance quantize to zero, which the adaptive
//! coder writes for a fraction of a bit: only the larger differences are written,
//! as small integers.
//!
//! Both layers are written with an adaptive binary range coder (LZMA style,
//! 11-bit probabilities). The decoder needs no source samples and reproduces the
//! encoder's reconstruction exactly on the same platform (it evaluates the same
//! floating-point synthesis); cross-platform agreement of `sin`/`cos` is untested.
//! Byte counts include the complete header.
use crate::bands::ToneAnalysis;
use crate::{buffer, Error};
use std::f64::consts::{PI, TAU};

const MAGIC: &[u8; 4] = b"SWL1";
const HEADER: usize = 4 + 1 + 1 + 4 + 4 + 4 + 8 + 4 + 4;
const BLOCK: usize = 4096;
const MAX_SAMPLES: usize = 1 << 28;
const MAX_TRACKS: usize = 4096;
/// Largest |sample|: 24-bit PCM.
const SAMPLE_LIMIT: i32 = 1 << 23;

/// Settings of the layered coder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayeredParams {
    /// Largest error at every decoded sample, in sample units, 0..=2^20.
    pub tolerance: u32,
    /// Waveform error each tonal parameter's quantization may cause, in sample
    /// units (finite, 1e-3..=1e6). Coarser is cheaper; the error goes to layer 2.
    pub tone_precision: f64,
    /// Only tones below this frequency (radians per sample) are coded as tracks.
    pub tonal_upper: f64,
    /// Largest change of a track's frequency between frames, in resolutions 2*pi/T.
    pub max_jump_bins: f64,
}
impl Default for LayeredParams {
    fn default() -> Self {
        Self {
            tolerance: 0,
            tone_precision: 4.0,
            tonal_upper: PI,
            max_jump_bins: 1.0,
        }
    }
}
impl LayeredParams {
    fn validate(&self) -> Result<(), Error> {
        if self.tolerance > 1 << 20
            || !(self.tone_precision >= 1e-3 && self.tone_precision <= 1e6)
            || !(self.tonal_upper > 0.0 && self.tonal_upper <= PI)
            || !(self.max_jump_bins > 0.0 && self.max_jump_bins <= 64.0)
        {
            return Err(Error::InvalidParameter("layered coder parameters"));
        }
        Ok(())
    }
}

/// What an encoding contains and costs. Bytes are those of the complete stream.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LayeredStats {
    pub samples: usize,
    pub frames: usize,
    pub tracks: usize,
    pub breakpoints: usize,
    pub header_bytes: usize,
    pub tonal_bytes: usize,
    pub correction_bytes: usize,
    pub total_bytes: usize,
    /// Samples whose correction is nonzero (the "written" differences).
    pub written: usize,
    pub max_error: u32,
    pub rms_error: f64,
    /// RMS of the source, of the decoded tonal track, and of the source minus
    /// the rounded tonal track (what layer 2 has to code).
    pub source_rms: f64,
    pub tonal_rms: f64,
    pub residual_rms: f64,
}

#[derive(Debug, Clone)]
pub struct Encoded {
    pub bytes: Vec<u8>,
    /// The decoded samples, as the decoder will reproduce them.
    pub decoded: Vec<i32>,
    pub stats: LayeredStats,
}

// ---------------------------------------------------------------- range coder

const PROB_INIT: u16 = 1024;

struct RangeEncoder {
    low: u64,
    range: u32,
    cache: u8,
    pending: u64,
    out: Vec<u8>,
}
impl RangeEncoder {
    fn new() -> Self {
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
    fn bit(&mut self, p: &mut u16, bit: bool) {
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
    fn direct(&mut self, value: u64, bits: u32) {
        for i in (0..bits).rev() {
            self.range >>= 1;
            if (value >> i) & 1 == 1 {
                self.low += u64::from(self.range);
            }
            self.normalize();
        }
    }
    fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

struct RangeDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    range: u32,
    code: u32,
}
impl<'a> RangeDecoder<'a> {
    fn new(data: &'a [u8]) -> Result<Self, Error> {
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
    fn bit(&mut self, p: &mut u16) -> Result<bool, Error> {
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
    fn direct(&mut self, bits: u32) -> Result<u64, Error> {
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

fn corrupt() -> Error {
    Error::InvalidParameter("SWL1 stream")
}

/// Adaptive Elias-gamma code of v+1: unary length with one context per position,
/// then the bits below the leading one, written directly.
#[derive(Clone)]
struct Gamma {
    prefix: [u16; 48],
}
impl Gamma {
    fn new() -> Self {
        Self {
            prefix: [PROB_INIT; 48],
        }
    }
    fn put(&mut self, e: &mut RangeEncoder, v: u64) {
        let m = v + 1;
        let width = 64 - m.leading_zeros();
        for i in 0..width - 1 {
            e.bit(&mut self.prefix[i as usize], true);
        }
        e.bit(&mut self.prefix[(width - 1) as usize], false);
        e.direct(m, width - 1);
    }
    fn get(&mut self, d: &mut RangeDecoder) -> Result<u64, Error> {
        let mut width = 1;
        while d.bit(&mut self.prefix[width - 1])? {
            width += 1;
            if width > 47 {
                return Err(corrupt());
            }
        }
        let rest = d.direct(width as u32 - 1)?;
        Ok(((1u64 << (width - 1)) | rest) - 1)
    }
    fn put_signed(&mut self, e: &mut RangeEncoder, v: i64) {
        self.put(e, ((v << 1) ^ (v >> 63)) as u64);
    }
    fn get_signed(&mut self, d: &mut RangeDecoder) -> Result<i64, Error> {
        let z = self.get(d)?;
        Ok((z >> 1) as i64 ^ -((z & 1) as i64))
    }
}

// --------------------------------------------------------------- tonal layer

/// A decoded breakpoint: frame index, frequency, amplitude, phase at the centre.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Breakpoint {
    frame: usize,
    omega: f64,
    amplitude: f64,
    phase: f64,
}

#[derive(Debug, Clone)]
struct LiveTrack {
    index: usize,
    level: u64,
    omega: f64,
    previous: Option<f64>,
    phase: f64,
}

/// Quantization of one track parameter set, shared by encoder and decoder.
struct Steps {
    level: f64,
    hop: f64,
}
impl Steps {
    fn amplitude(&self, level: u64) -> f64 {
        level as f64 * 2.0 * self.level
    }
    /// Phase step 2*pi/2^b with amplitude*step/2 <= precision.
    fn phase(&self, amplitude: f64) -> f64 {
        let bits = (TAU * amplitude / (2.0 * self.level))
            .log2()
            .ceil()
            .clamp(3.0, 30.0);
        TAU / bits.exp2()
    }
    /// Frequency step 2^-e with amplitude*step*hop/4 <= precision.
    fn frequency(&self, amplitude: f64) -> f64 {
        let e = (amplitude * self.hop / (4.0 * self.level))
            .log2()
            .ceil()
            .clamp(0.0, 50.0);
        (-e).exp2()
    }
}

fn wrap(angle: f64) -> f64 {
    let a = angle.rem_euclid(TAU);
    if a > PI {
        a - TAU
    } else {
        a
    }
}

struct TonalModel {
    continues: [u16; 2],
    amplitude: Gamma,
    frequency: Gamma,
    phase: Gamma,
    births: Gamma,
    new_amplitude: Gamma,
    new_frequency: Gamma,
}
impl TonalModel {
    fn new() -> Self {
        Self {
            continues: [PROB_INIT; 2],
            amplitude: Gamma::new(),
            frequency: Gamma::new(),
            phase: Gamma::new(),
            births: Gamma::new(),
            new_amplitude: Gamma::new(),
            new_frequency: Gamma::new(),
        }
    }
}

fn predicted_frequency(t: &LiveTrack) -> f64 {
    // The track's own gradient: linear extrapolation of the last two breakpoints.
    t.omega + t.previous.map_or(0.0, |p| t.omega - p)
}

/// Encode the tracks, returning the bytes and the decoded breakpoints per track.
fn encode_tonal(
    analysis: &ToneAnalysis,
    params: &LayeredParams,
) -> Result<(Vec<u8>, Vec<Vec<Breakpoint>>), Error> {
    let steps = Steps {
        level: params.tone_precision,
        hop: analysis.hop() as f64,
    };
    let jump = params.max_jump_bins * TAU / analysis.window() as f64;
    let mut e = RangeEncoder::new();
    let mut model = TonalModel::new();
    let mut tracks: Vec<Vec<Breakpoint>> = Vec::new();
    let mut live: Vec<LiveTrack> = Vec::new();
    for (k, frame) in analysis.frames().iter().enumerate() {
        let mut tones: Vec<_> = frame
            .tones
            .iter()
            .filter(|t| t.omega < params.tonal_upper)
            .map(|t| (t, (t.amplitude / (2.0 * steps.level)).round()))
            .filter(|(_, level)| *level >= 1.0)
            .collect();
        // Strongest first, each takes the nearest unmatched predicted track.
        tones.sort_by(|a, b| b.0.amplitude.total_cmp(&a.0.amplitude));
        let mut matched: Vec<Option<usize>> = vec![None; live.len()];
        let mut births = Vec::new();
        for (index, &(tone, _)) in tones.iter().enumerate() {
            let nearest = live
                .iter()
                .enumerate()
                .filter(|(i, t)| {
                    matched[*i].is_none() && (predicted_frequency(t) - tone.omega).abs() <= jump
                })
                .min_by(|a, b| {
                    let da = (predicted_frequency(a.1) - tone.omega).abs();
                    let db = (predicted_frequency(b.1) - tone.omega).abs();
                    da.total_cmp(&db)
                });
            match nearest {
                Some((i, _)) => matched[i] = Some(index),
                None if live.len() + births.len() < MAX_TRACKS => births.push(index),
                None => {}
            }
        }
        let mut next = Vec::with_capacity(live.len() + births.len());
        for (t, m) in live.iter().zip(&matched) {
            let context = usize::from(t.previous.is_some());
            e.bit(&mut model.continues[context], m.is_some());
            let Some(index) = *m else { continue };
            let (tone, level) = tones[index];
            let level = level as u64;
            model
                .amplitude
                .put_signed(&mut e, level as i64 - t.level as i64);
            let amplitude = steps.amplitude(level);
            let predicted = predicted_frequency(t);
            let q = steps.frequency(amplitude);
            let d = ((tone.omega - predicted) / q).round() as i64;
            model.frequency.put_signed(&mut e, d);
            let omega = predicted + d as f64 * q;
            let guess = t.phase + 0.5 * (t.omega + omega) * steps.hop;
            let qp = steps.phase(amplitude);
            let dp = (wrap(tone.phase - guess) / qp).round() as i64;
            model.phase.put_signed(&mut e, dp);
            let phase = wrap(guess + dp as f64 * qp);
            tracks[t.index].push(Breakpoint {
                frame: k,
                omega,
                amplitude,
                phase,
            });
            next.push(LiveTrack {
                index: t.index,
                level,
                omega,
                previous: Some(t.omega),
                phase,
            });
        }
        births.sort_by(|&a, &b| tones[a].0.omega.total_cmp(&tones[b].0.omega));
        model.births.put(&mut e, births.len() as u64);
        let mut anchor = 0.0;
        for &index in &births {
            let (tone, level) = tones[index];
            let level = level as u64;
            model.new_amplitude.put(&mut e, level - 1);
            let amplitude = steps.amplitude(level);
            let q = steps.frequency(amplitude);
            let d = ((tone.omega - anchor) / q).round() as i64;
            model.new_frequency.put_signed(&mut e, d);
            let omega = anchor + d as f64 * q;
            anchor = omega;
            let qp = steps.phase(amplitude);
            let bits = (TAU / qp).log2().round() as u32;
            let p = ((tone.phase.rem_euclid(TAU)) / qp).round() as u64 % (1u64 << bits);
            e.direct(p, bits);
            let phase = wrap(p as f64 * qp);
            tracks.push(vec![Breakpoint {
                frame: k,
                omega,
                amplitude,
                phase,
            }]);
            next.push(LiveTrack {
                index: tracks.len() - 1,
                level,
                omega,
                previous: None,
                phase,
            });
        }
        next.sort_by(|a, b| a.omega.total_cmp(&b.omega).then(a.index.cmp(&b.index)));
        live = next;
    }
    Ok((e.finish(), tracks))
}

fn decode_tonal(
    data: &[u8],
    frames: usize,
    hop: usize,
    precision: f64,
) -> Result<Vec<Vec<Breakpoint>>, Error> {
    let steps = Steps {
        level: precision,
        hop: hop as f64,
    };
    let mut d = RangeDecoder::new(data)?;
    let mut model = TonalModel::new();
    let mut tracks: Vec<Vec<Breakpoint>> = Vec::new();
    let mut live: Vec<LiveTrack> = Vec::new();
    for k in 0..frames {
        let mut next = Vec::with_capacity(live.len());
        for t in &live {
            let context = usize::from(t.previous.is_some());
            if !d.bit(&mut model.continues[context])? {
                continue;
            }
            let level = t.level as i64 + model.amplitude.get_signed(&mut d)?;
            if !(1..=1 << 40).contains(&level) {
                return Err(corrupt());
            }
            let level = level as u64;
            let amplitude = steps.amplitude(level);
            let predicted = predicted_frequency(t);
            let q = steps.frequency(amplitude);
            let omega = predicted + model.frequency.get_signed(&mut d)? as f64 * q;
            if !(omega.is_finite() && omega.abs() < 16.0) {
                return Err(corrupt());
            }
            let guess = t.phase + 0.5 * (t.omega + omega) * steps.hop;
            let qp = steps.phase(amplitude);
            let phase = wrap(guess + model.phase.get_signed(&mut d)? as f64 * qp);
            tracks[t.index].push(Breakpoint {
                frame: k,
                omega,
                amplitude,
                phase,
            });
            next.push(LiveTrack {
                index: t.index,
                level,
                omega,
                previous: Some(t.omega),
                phase,
            });
        }
        let births = model.births.get(&mut d)?;
        if births as usize + next.len() > MAX_TRACKS {
            return Err(corrupt());
        }
        let mut anchor = 0.0;
        for _ in 0..births {
            let level = model.new_amplitude.get(&mut d)? + 1;
            if level > 1 << 40 {
                return Err(corrupt());
            }
            let amplitude = steps.amplitude(level);
            let q = steps.frequency(amplitude);
            let omega = anchor + model.new_frequency.get_signed(&mut d)? as f64 * q;
            if !(omega.is_finite() && omega.abs() < 16.0) {
                return Err(corrupt());
            }
            anchor = omega;
            let qp = steps.phase(amplitude);
            let bits = (TAU / qp).log2().round() as u32;
            let phase = wrap(d.direct(bits)? as f64 * qp);
            tracks.push(vec![Breakpoint {
                frame: k,
                omega,
                amplitude,
                phase,
            }]);
            next.push(LiveTrack {
                index: tracks.len() - 1,
                level,
                omega,
                previous: None,
                phase,
            });
        }
        next.sort_by(|a, b| a.omega.total_cmp(&b.omega).then(a.index.cmp(&b.index)));
        live = next;
    }
    Ok(tracks)
}

/// Render tracks at the samples `0..len`: cubic phase (McAulay-Quatieri) and
/// linear amplitude between breakpoints one hop apart, and a fade of one hop
/// at constant frequency before the first and after the last breakpoint.
fn render(tracks: &[Vec<Breakpoint>], len: usize, hop: usize) -> Result<Vec<f64>, Error> {
    let mut out = buffer(len)?;
    out.resize(len, 0.0);
    let h = hop as f64;
    let mut add = |from: i64, theta: &dyn Fn(f64) -> (f64, f64)| {
        // Samples centre+0 .. centre+hop-1 of a segment starting at `from`.
        for j in 0..hop as i64 {
            let n = from + j;
            if n < 0 || n >= len as i64 {
                continue;
            }
            let (amplitude, angle) = theta(j as f64);
            out[n as usize] += amplitude * angle.cos();
        }
    };
    for track in tracks {
        let first = track[0];
        let start = (first.frame * hop) as i64;
        add(start - hop as i64, &|t| {
            let a = first.amplitude * t / h;
            (a, first.phase + first.omega * (t - h))
        });
        for pair in track.windows(2) {
            let (b0, b1) = (pair[0], pair[1]);
            let lag = b1.phase;
            let advance = b0.phase + b0.omega * h;
            let m = ((advance - lag + 0.5 * (b1.omega - b0.omega) * h) / TAU).round();
            let target = lag + TAU * m - advance;
            let alpha = 3.0 * target / (h * h) - (b1.omega - b0.omega) / h;
            let beta = -2.0 * target / (h * h * h) + (b1.omega - b0.omega) / (h * h);
            add((b0.frame * hop) as i64, &|t| {
                let a = b0.amplitude + (b1.amplitude - b0.amplitude) * t / h;
                (
                    a,
                    b0.phase + b0.omega * t + alpha * t * t + beta * t * t * t,
                )
            });
        }
        let last = track[track.len() - 1];
        add((last.frame * hop) as i64, &|t| {
            let a = last.amplitude * (1.0 - t / h);
            (a, last.phase + last.omega * t)
        });
    }
    if out.iter().all(|v| v.is_finite()) {
        Ok(out)
    } else {
        Err(Error::InvalidParameter("tonal rendering overflow"))
    }
}

// ----------------------------------------------------------- correction layer

struct CorrectionModel {
    order: [u16; 3],
    zero: [u16; 4],
    sign: [u16; 4],
    magnitude: Vec<Gamma>,
}
impl CorrectionModel {
    fn new() -> Self {
        Self {
            order: [PROB_INIT; 3],
            zero: [PROB_INIT; 4],
            sign: [PROB_INIT; 4],
            magnitude: vec![Gamma::new(); 4],
        }
    }
}

fn bucket(q: i64) -> usize {
    match q.unsigned_abs() {
        0 => 0,
        1..=2 => 1,
        3..=8 => 2,
        _ => 3,
    }
}

fn predict(order: usize, history: &[i64]) -> i64 {
    let n = history.len();
    match (order, n) {
        (1, 1..) => history[n - 1],
        (2, 2..) => 2 * history[n - 1] - history[n - 2],
        (2, 1) => history[0],
        _ => 0,
    }
}

/// Quantize a block with a closed-loop predictor whose history (reconstructed
/// residual) is extended in place; returns the quantization indices.
fn quantize_block(residual: &[i64], order: usize, step: i64, history: &mut Vec<i64>) -> Vec<i64> {
    let half = (step - 1) / 2;
    residual
        .iter()
        .map(|&e| {
            let p = predict(order, history);
            let q = (e - p + half).div_euclid(step);
            history.push(p + q * step);
            q
        })
        .collect()
}

fn encode_corrections(residual: &[i64], tolerance: u32) -> (Vec<u8>, Vec<i64>, usize) {
    let step = 2 * i64::from(tolerance) + 1;
    let mut e = RangeEncoder::new();
    let mut model = CorrectionModel::new();
    let mut history: Vec<i64> = Vec::with_capacity(residual.len());
    let mut written = 0;
    let mut previous = 0;
    for block in residual.chunks(BLOCK) {
        // Choose the order with the smallest sum of |index| (closed loop).
        let seed = history[history.len().saturating_sub(2)..].to_vec();
        let mut best = (u64::MAX, 0);
        for order in 0..3 {
            let mut trial = seed.clone();
            let q = quantize_block(block, order, step, &mut trial);
            let cost: u64 = q.iter().map(|v| v.unsigned_abs()).sum();
            if cost < best.0 {
                best = (cost, order);
            }
        }
        let order = best.1;
        e.bit(&mut model.order[0], order > 0);
        if order > 0 {
            e.bit(&mut model.order[1], order > 1);
        }
        let indices = quantize_block(block, order, step, &mut history);
        for q in indices {
            let b = bucket(previous);
            e.bit(&mut model.zero[b], q != 0);
            if q != 0 {
                written += 1;
                e.bit(&mut model.sign[b], q < 0);
                model.magnitude[b].put(&mut e, q.unsigned_abs() - 1);
            }
            previous = q;
        }
    }
    (e.finish(), history, written)
}

fn decode_corrections(data: &[u8], len: usize, tolerance: u32) -> Result<Vec<i64>, Error> {
    let step = 2 * i64::from(tolerance) + 1;
    let mut d = RangeDecoder::new(data)?;
    let mut model = CorrectionModel::new();
    let mut history: Vec<i64> = buffer(len)?;
    let mut previous = 0;
    let mut remaining = len;
    while remaining > 0 {
        let count = remaining.min(BLOCK);
        let mut order = usize::from(d.bit(&mut model.order[0])?);
        if order > 0 && d.bit(&mut model.order[1])? {
            order = 2;
        }
        for _ in 0..count {
            let b = bucket(previous);
            let q = if d.bit(&mut model.zero[b])? {
                let negative = d.bit(&mut model.sign[b])?;
                let m = model.magnitude[b].get(&mut d)? + 1;
                if m > 1 << 40 {
                    return Err(corrupt());
                }
                if negative {
                    -(m as i64)
                } else {
                    m as i64
                }
            } else {
                0
            };
            let p = predict(order, &history);
            history.push(p + q * step);
            previous = q;
        }
        remaining -= count;
    }
    Ok(history)
}

// ------------------------------------------------------------------ stream

fn frame_count(len: usize, hop: usize) -> usize {
    (len + 2 * hop).div_ceil(hop) - 1
}

fn rounded(tonal: &[f64]) -> Vec<i64> {
    tonal
        .iter()
        .map(|v| v.round().clamp(-(1i64 << 40) as f64, (1i64 << 40) as f64) as i64)
        .collect()
}

fn rms<T: Copy + Into<f64>>(values: impl Iterator<Item = T>, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    (values.map(|v| v.into().powi(2)).sum::<f64>() / n as f64).sqrt()
}

/// Encode integer samples (|x| < 2^23). With `analysis` (of the same samples)
/// the tonal layer codes its tracks below `params.tonal_upper`; without it the
/// stream has the correction layer only (the control).
pub fn encode(
    samples: &[i32],
    analysis: Option<&ToneAnalysis>,
    params: LayeredParams,
) -> Result<Encoded, Error> {
    params.validate()?;
    if samples.is_empty() || samples.len() > MAX_SAMPLES {
        return Err(Error::InvalidParameter("layered coder samples"));
    }
    if let Some(i) = samples
        .iter()
        .position(|v| v.unsigned_abs() >= SAMPLE_LIMIT as u32)
    {
        return Err(Error::NonFiniteSample(i));
    }
    let hop = match analysis {
        Some(a) if a.len() != samples.len() => return Err(Error::LengthMismatch),
        Some(a) => a.hop(),
        None => 1024,
    };
    let (tonal_bytes, tracks) = match analysis {
        Some(a) => encode_tonal(a, &params)?,
        None => (Vec::new(), Vec::new()),
    };
    let tonal = render(&tracks, samples.len(), hop)?;
    let prediction = rounded(&tonal);
    let residual: Vec<i64> = samples
        .iter()
        .zip(&prediction)
        .map(|(&x, &p)| i64::from(x) - p)
        .collect();
    let (correction_bytes, reconstructed, written) =
        encode_corrections(&residual, params.tolerance);
    let mut bytes = buffer(HEADER + tonal_bytes.len() + correction_bytes.len())?;
    bytes.extend_from_slice(MAGIC);
    bytes.push(1);
    bytes.push(u8::from(analysis.is_some()));
    bytes.extend_from_slice(&(samples.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(hop as u32).to_le_bytes());
    bytes.extend_from_slice(&params.tolerance.to_le_bytes());
    bytes.extend_from_slice(&params.tone_precision.to_le_bytes());
    bytes.extend_from_slice(&(tonal_bytes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(correction_bytes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&tonal_bytes);
    bytes.extend_from_slice(&correction_bytes);
    let decoded: Vec<i32> = prediction
        .iter()
        .zip(&reconstructed)
        .map(|(&p, &r)| (p + r).clamp(-i64::from(SAMPLE_LIMIT), i64::from(SAMPLE_LIMIT)) as i32)
        .collect();
    let errors = samples
        .iter()
        .zip(&decoded)
        .map(|(&x, &y)| (i64::from(x) - i64::from(y)).unsigned_abs());
    let max_error = errors.clone().max().unwrap_or(0) as u32;
    let n = samples.len();
    let stats = LayeredStats {
        samples: n,
        frames: if analysis.is_some() {
            frame_count(n, hop)
        } else {
            0
        },
        tracks: tracks.len(),
        breakpoints: tracks.iter().map(Vec::len).sum(),
        header_bytes: HEADER,
        tonal_bytes: tonal_bytes.len(),
        correction_bytes: correction_bytes.len(),
        total_bytes: bytes.len(),
        written,
        max_error,
        rms_error: rms(errors.map(|e| e as f64), n),
        source_rms: rms(samples.iter().map(|&x| x as f64), n),
        tonal_rms: rms(tonal.iter().copied(), n),
        residual_rms: rms(residual.iter().map(|&e| e as f64), n),
    };
    Ok(Encoded {
        bytes,
        decoded,
        stats,
    })
}

/// Decode a stream from `encode`. Needs no source samples. Rejects malformed or
/// truncated streams and bounds every allocation by the header's sample count
/// (at most 2^28) before reading.
pub fn decode(bytes: &[u8]) -> Result<Vec<i32>, Error> {
    if bytes.len() < HEADER || &bytes[..4] != MAGIC || bytes[4] != 1 || bytes[5] > 1 {
        return Err(corrupt());
    }
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    let len = word(6);
    let hop = word(10);
    let tolerance = word(14) as u32;
    let precision = f64::from_le_bytes(bytes[18..26].try_into().unwrap());
    let (tonal_len, correction_len) = (word(26), word(30));
    if len == 0
        || len > MAX_SAMPLES
        || !(8..=32768).contains(&hop)
        || tolerance > 1 << 20
        || !(1e-3..=1e6).contains(&precision)
        || HEADER
            .checked_add(tonal_len)
            .and_then(|v| v.checked_add(correction_len))
            != Some(bytes.len())
    {
        return Err(corrupt());
    }
    let tonal_data = &bytes[HEADER..HEADER + tonal_len];
    let correction_data = &bytes[HEADER + tonal_len..];
    let tracks = if bytes[5] == 1 {
        decode_tonal(tonal_data, frame_count(len, hop), hop, precision)?
    } else if tonal_len != 0 {
        return Err(corrupt());
    } else {
        Vec::new()
    };
    let prediction = rounded(&render(&tracks, len, hop)?);
    let reconstructed = decode_corrections(correction_data, len, tolerance)?;
    Ok(prediction
        .iter()
        .zip(&reconstructed)
        .map(|(&p, &r)| (p + r).clamp(-i64::from(SAMPLE_LIMIT), i64::from(SAMPLE_LIMIT)) as i32)
        .collect())
}
