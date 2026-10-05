//! SWL2: layered closed-loop coding of integer audio (PAPER_EQ_COMPRESSION.md,
//! Section 4). Experimental: not a stable or fuzz-validated format. The byte
//! layout is specified in docs/SWL2.md.
//!
//! A stream is a sequence of segments (one, or 2^S frames each). A key segment
//! restarts every adaptive state, so decoding can start there (`decode_from`);
//! a continuation inherits the correction layer's state. Each segment has its
//! own stereo mode and, per channel, a list of chunks, each predicting part of
//! the signal from what the previous chunks left:
//!
//! 1. `gradients` (optional): partial tracks of a frequency band as groups of
//!    frequency-gradient segments that share start, duration and volume function;
//!    range coded, continuing tracks predicted from their predecessors.
//! 2. `anchors` (optional): SinWhisky anchors at dynamic positions whose circles
//!    rebuild a low band (or all) of what the gradients left.
//! 3. `correction` (always, last): closed-loop LPC of the source with the rounded
//!    layer prediction as side information (a per-block gain decides how much of
//!    it to subtract), optionally followed by a cascade of sign-sign LMS stages
//!    (1024, 256 and 16 taps; each block chooses how many to use), and
//!    quantization step `2*tolerance + 1`.
//!
//! Every decoded sample is within `tolerance` of the source (0 is lossless) and
//! the decoder needs no source samples. Lossless stereo streams may decorrelate
//! the two channels (left/side, side/right, mid/side) as FLAC does.
//!
//! `encode_auto` chooses everything by exact coded size, segment by segment: the
//! stereo mode, whether a gradient layer pays and for which band and precision,
//! whether anchors pay, and the block size; a layer is kept only if the segment
//! gets smaller.
//! Gradients and anchors are rendered in floating point and rounded, so
//! decoding is exact only on the platform that encoded (the correction layer is
//! integer); cross-platform agreement of `sin`/`cos` is untested.
mod anchors;
mod coder;
mod correction;
mod gradients;

pub use anchors::AnchorConfig;
pub use gradients::GradientConfig;

use crate::analog::{AnalogFilter, SpiralCurve};
use crate::bands::{ToneAnalysis, ToneParams};
use crate::Error;
use coder::{corrupt, Bytes};

const MAGIC: &[u8; 4] = b"SWL2";
const VERSION: u8 = 4;
const HEADER: usize = 16;
const MAX_SAMPLES: usize = 1 << 28;
/// Largest |sample| of the source: 24-bit PCM.
const SAMPLE_LIMIT: i64 = 1 << 23;
/// Tolerance value of a parametric stream: no correction layer, so the decoded
/// signal is the rounded gradient and anchor prediction alone. Its error is
/// unbounded and measured (`StreamStats::snr_db`), not guaranteed.
pub const PARAMETRIC: u32 = u32::MAX;
const GRADIENTS: u8 = 1;
const ANCHORS: u8 = 2;
const CORRECTION: u8 = 3;
/// Bytes of a chunk's type and length.
const CHUNK_HEADER: usize = 5;

/// How the two channels of a lossless stereo stream are coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stereo {
    Independent,
    /// Left and side (left - right).
    LeftSide,
    /// Side and right.
    SideRight,
    /// Mid ((left + right) >> 1) and side.
    MidSide,
}
impl Stereo {
    fn code(self) -> u8 {
        match self {
            Stereo::Independent => 0,
            Stereo::LeftSide => 1,
            Stereo::SideRight => 2,
            Stereo::MidSide => 3,
        }
    }
    fn from_code(c: u8) -> Option<Self> {
        Some(match c {
            0 => Stereo::Independent,
            1 => Stereo::LeftSide,
            2 => Stereo::SideRight,
            3 => Stereo::MidSide,
            _ => return None,
        })
    }
}

/// Layers of one coded channel in one segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelConfig {
    pub gradients: Option<GradientConfig>,
    pub anchors: Option<AnchorConfig>,
    /// Correction blocks of 2^block_exponent samples, 8..=15.
    pub block_exponent: u32,
    /// Follow the LPC with the adaptive LMS cascade (false: LPC only).
    pub cascade: bool,
}
impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            gradients: None,
            anchors: None,
            block_exponent: 12,
            cascade: true,
        }
    }
}

/// One segment: whether it is a key (every adaptive state starts afresh, so
/// decoding can start there), its stereo mode and its channels' layers.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentConfig {
    pub key: bool,
    /// Must be `Independent` unless the stream has two channels and tolerance 0.
    pub stereo: Stereo,
    /// One per coded channel.
    pub channels: Vec<ChannelConfig>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamConfig {
    /// Largest error at every decoded sample, 0..=2^20 (0 is lossless), or
    /// `PARAMETRIC` for no correction layer.
    pub tolerance: u32,
    /// Segments of 2^segment_exponent frames (12..=24; the last may be
    /// shorter), or 0 for a single segment.
    pub segment_exponent: u32,
    /// One per segment, in order; the first must be a key.
    pub segments: Vec<SegmentConfig>,
}
impl StreamConfig {
    /// A single key segment with these settings.
    pub fn single(tolerance: u32, stereo: Stereo, channels: Vec<ChannelConfig>) -> Self {
        Self {
            tolerance,
            segment_exponent: 0,
            segments: vec![SegmentConfig {
                key: true,
                stereo,
                channels,
            }],
        }
    }
}

/// Ranges of the segments of a stream of `len` frames.
fn segment_ranges(len: usize, exponent: u32) -> Vec<std::ops::Range<usize>> {
    if exponent == 0 {
        return std::iter::once(0..len).collect();
    }
    let size = 1usize << exponent;
    (0..len.div_ceil(size))
        .map(|s| s * size..((s + 1) * size).min(len))
        .collect()
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ChannelStats {
    pub gradient_bytes: usize,
    pub anchor_bytes: usize,
    pub correction_bytes: usize,
    pub groups: usize,
    pub members: usize,
    pub anchors: usize,
    /// Samples whose correction index is nonzero (difference above tolerance).
    pub written: usize,
    pub mean_order: f64,
    /// Mean gain the correction blocks gave the layers' prediction (0..=1).
    pub mean_gain: f64,
    /// Mean number of LMS cascade stages the correction blocks used (0..=3).
    pub mean_stages: f64,
    /// RMS of the coded channel and of what the correction layer had to code.
    pub source_rms: f64,
    pub residual_rms: f64,
}
impl ChannelStats {
    /// Sums of counts and frame-weighted means over segments of `frames`.
    fn combine(parts: &[(usize, ChannelStats)]) -> Self {
        let total: usize = parts.iter().map(|p| p.0).sum::<usize>().max(1);
        let mut out = ChannelStats::default();
        let (mut source, mut residual) = (0.0, 0.0);
        for &(frames, s) in parts {
            let w = frames as f64 / total as f64;
            out.gradient_bytes += s.gradient_bytes;
            out.anchor_bytes += s.anchor_bytes;
            out.correction_bytes += s.correction_bytes;
            out.groups += s.groups;
            out.members += s.members;
            out.anchors += s.anchors;
            out.written += s.written;
            out.mean_order += w * s.mean_order;
            out.mean_gain += w * s.mean_gain;
            out.mean_stages += w * s.mean_stages;
            source += w * s.source_rms * s.source_rms;
            residual += w * s.residual_rms * s.residual_rms;
        }
        out.source_rms = source.sqrt();
        out.residual_rms = residual.sqrt();
        out
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamStats {
    pub frames: usize,
    pub header_bytes: usize,
    pub total_bytes: usize,
    pub segments: usize,
    pub keys: usize,
    pub max_error: u32,
    pub rms_error: f64,
    /// 10*log10(source energy / error energy); infinite when lossless.
    pub snr_db: f64,
    /// Per coded channel, over all segments.
    pub channels: Vec<ChannelStats>,
}

#[derive(Debug, Clone)]
pub struct Encoded {
    pub bytes: Vec<u8>,
    /// The samples `decode` will return.
    pub decoded: Vec<Vec<i32>>,
    pub stats: StreamStats,
    pub config: StreamConfig,
}

fn transform(channels: &[Vec<i32>], stereo: Stereo) -> Vec<Vec<i64>> {
    let wide = |c: &Vec<i32>| c.iter().map(|&v| i64::from(v)).collect::<Vec<i64>>();
    if stereo == Stereo::Independent {
        return channels.iter().map(wide).collect();
    }
    let (l, r) = (wide(&channels[0]), wide(&channels[1]));
    let side: Vec<i64> = l.iter().zip(&r).map(|(a, b)| a - b).collect();
    match stereo {
        Stereo::LeftSide => vec![l, side],
        Stereo::SideRight => vec![side, r],
        _ => {
            let mid = l.iter().zip(&r).map(|(a, b)| (a + b) >> 1).collect();
            vec![mid, side]
        }
    }
}

fn inverse(coded: Vec<Vec<i64>>, stereo: Stereo) -> Vec<Vec<i64>> {
    if stereo == Stereo::Independent || coded.len() != 2 {
        return coded;
    }
    let (a, b) = (&coded[0], &coded[1]);
    let (l, r): (Vec<i64>, Vec<i64>) = match stereo {
        Stereo::LeftSide => (a.clone(), a.iter().zip(b).map(|(l, s)| l - s).collect()),
        Stereo::SideRight => (a.iter().zip(b).map(|(s, r)| s + r).collect(), b.clone()),
        _ => a
            .iter()
            .zip(b)
            .map(|(&m, &s)| {
                let mid = (m << 1) | (s & 1);
                ((mid + s) >> 1, (mid - s) >> 1)
            })
            .unzip(),
    };
    vec![l, r]
}

fn round_all(values: &[f64]) -> Vec<i64> {
    values
        .iter()
        .map(|v| v.round().clamp(-(1i64 << 40) as f64, (1i64 << 40) as f64) as i64)
        .collect()
}

fn rms(values: impl Iterator<Item = f64>, n: usize) -> f64 {
    (values.map(|v| v * v).sum::<f64>() / n.max(1) as f64).sqrt()
}

struct Channel {
    chunks: Vec<(u8, Vec<u8>)>,
    reconstructed: Vec<i64>,
    stats: ChannelStats,
    /// Correction state after the segment (fresh in a parametric stream).
    state: correction::State,
}
impl Channel {
    fn bytes(&self) -> usize {
        1 + self
            .chunks
            .iter()
            .map(|c| CHUNK_HEADER + c.1.len())
            .sum::<usize>()
    }
}

/// The anchor target: what the gradients left, or its band below `cutoff`.
fn anchor_target(remainder: &[i64], cutoff: Option<f64>) -> Result<Vec<f64>, Error> {
    let r: Vec<f64> = remainder.iter().map(|&v| v as f64).collect();
    match cutoff {
        None => Ok(r),
        Some(c) => {
            let filter = AnalogFilter::butterworth_lowpass(4, c)?;
            SpiralCurve::from_samples(&r)?
                .filter_zero_phase(&filter)?
                .samples()
        }
    }
}

fn encode_channel(
    x: &[i64],
    analysis: Option<&ToneAnalysis>,
    config: &ChannelConfig,
    tolerance: u32,
    state: correction::State,
) -> Result<Channel, Error> {
    if !(8..=15).contains(&config.block_exponent) {
        return Err(Error::InvalidParameter("correction block exponent"));
    }
    let len = x.len();
    let mut prediction = vec![0i64; len];
    let mut chunks = Vec::new();
    let mut stats = ChannelStats::default();
    if let Some(g) = &config.gradients {
        let analysis = analysis.ok_or(Error::InvalidParameter("gradients need a tone analysis"))?;
        if analysis.len() != len {
            return Err(Error::LengthMismatch);
        }
        let payload = gradients::encode(analysis, g)?;
        let parsed = gradients::parse(&payload, len)?;
        let mut tonal = vec![0.0; len];
        gradients::render(&parsed, &mut tonal);
        for (p, t) in prediction.iter_mut().zip(round_all(&tonal)) {
            *p += t;
        }
        stats.gradient_bytes = payload.len();
        stats.groups = parsed.groups();
        stats.members = parsed.members();
        chunks.push((GRADIENTS, payload));
    }
    if let Some(a) = &config.anchors {
        let remainder: Vec<i64> = x.iter().zip(&prediction).map(|(v, p)| v - p).collect();
        let target = anchor_target(&remainder, a.cutoff)?;
        let payload = anchors::encode(&target, a)?;
        let parsed = anchors::parse(&payload, len)?;
        for (p, v) in prediction
            .iter_mut()
            .zip(round_all(&anchors::render(&parsed, len)))
        {
            *p += v;
        }
        stats.anchor_bytes = payload.len();
        stats.anchors = parsed.len();
        chunks.push((ANCHORS, payload));
    }
    stats.source_rms = rms(x.iter().map(|&v| v as f64), len);
    stats.residual_rms = rms(x.iter().zip(&prediction).map(|(v, p)| (v - p) as f64), len);
    if tolerance == PARAMETRIC {
        return Ok(Channel {
            chunks,
            reconstructed: prediction,
            stats,
            state,
        });
    }
    let layered = !chunks.is_empty();
    let correction = correction::encode(
        x,
        layered.then_some(&prediction[..]),
        tolerance,
        config.block_exponent,
        config.cascade,
        state,
    );
    stats.correction_bytes = correction.payload.len();
    stats.written = correction.written;
    stats.mean_order = correction.mean_order;
    stats.mean_gain = correction.mean_gain;
    stats.mean_stages = correction.mean_stages;
    let reconstructed = correction.reconstructed;
    chunks.push((CORRECTION, correction.payload));
    Ok(Channel {
        chunks,
        reconstructed,
        stats,
        state: correction.state,
    })
}

fn validate_input(channels: &[Vec<i32>]) -> Result<usize, Error> {
    let len = channels.first().map_or(0, Vec::len);
    if channels.is_empty() || channels.len() > 8 || len == 0 || len > MAX_SAMPLES {
        return Err(Error::InvalidParameter("SWL2 channels"));
    }
    if channels.iter().any(|c| c.len() != len) {
        return Err(Error::LengthMismatch);
    }
    for c in channels {
        if let Some(i) = c.iter().position(|&v| i64::from(v).abs() >= SAMPLE_LIMIT) {
            return Err(Error::NonFiniteSample(i));
        }
    }
    Ok(len)
}

fn validate_config(config: &StreamConfig, c: usize, len: usize) -> Result<(), Error> {
    let ranges = segment_ranges(len, config.segment_exponent);
    let ok = (config.tolerance <= 1 << 20 || config.tolerance == PARAMETRIC)
        && (config.segment_exponent == 0 || (12..=24).contains(&config.segment_exponent))
        && config.segments.len() == ranges.len()
        && config.segments.first().is_some_and(|s| s.key)
        && config.segments.iter().all(|s| {
            s.channels.len() == c
                && (s.stereo == Stereo::Independent || (c == 2 && config.tolerance == 0))
        });
    if ok {
        Ok(())
    } else {
        Err(Error::InvalidParameter("SWL2 stream configuration"))
    }
}

/// A coded segment: its configuration and channels.
struct Segment {
    config: SegmentConfig,
    channels: Vec<Channel>,
}

fn assemble(
    channels: &[Vec<i32>],
    config: StreamConfig,
    coded: Vec<Segment>,
) -> Result<Encoded, Error> {
    let len = channels[0].len();
    let mut bytes = Vec::with_capacity(HEADER);
    bytes.extend_from_slice(MAGIC);
    bytes.push(VERSION);
    bytes.push(channels.len() as u8);
    bytes.push(config.segment_exponent as u8);
    bytes.push(0);
    bytes.extend_from_slice(&(len as u32).to_le_bytes());
    bytes.extend_from_slice(&config.tolerance.to_le_bytes());
    let mut decoded: Vec<Vec<i32>> = vec![Vec::with_capacity(len); channels.len()];
    let mut parts: Vec<Vec<(usize, ChannelStats)>> = vec![Vec::new(); channels.len()];
    let mut keys = 0;
    let segments = coded.len();
    for segment in coded {
        let mut body = vec![u8::from(segment.config.key) | segment.config.stereo.code() << 1];
        keys += usize::from(segment.config.key);
        for c in &segment.channels {
            body.push(c.chunks.len() as u8);
            for (kind, payload) in &c.chunks {
                body.push(*kind);
                body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
                body.extend_from_slice(payload);
            }
        }
        bytes.extend_from_slice(&(body.len() as u32).to_le_bytes());
        bytes.extend(body);
        let frames = segment.channels[0].reconstructed.len();
        for (p, c) in parts.iter_mut().zip(&segment.channels) {
            p.push((frames, c.stats));
        }
        let restored = inverse(
            segment
                .channels
                .into_iter()
                .map(|c| c.reconstructed)
                .collect(),
            segment.config.stereo,
        );
        for (out, c) in decoded.iter_mut().zip(restored) {
            out.extend(
                c.iter()
                    .map(|&v| v.clamp(-SAMPLE_LIMIT, SAMPLE_LIMIT) as i32),
            );
        }
    }
    let mut max_error = 0u64;
    let mut square = 0.0;
    let mut energy = 0.0;
    for (source, back) in channels.iter().zip(&decoded) {
        for (&a, &b) in source.iter().zip(back) {
            let e = (i64::from(a) - i64::from(b)).unsigned_abs();
            max_error = max_error.max(e);
            square += (e as f64) * (e as f64);
            energy += f64::from(a) * f64::from(a);
        }
    }
    let total = len * channels.len();
    Ok(Encoded {
        stats: StreamStats {
            frames: len,
            header_bytes: HEADER,
            total_bytes: bytes.len(),
            segments,
            keys,
            max_error: max_error.min(u64::from(u32::MAX)) as u32,
            rms_error: (square / total as f64).sqrt(),
            snr_db: 10.0 * (energy / square).log10(),
            channels: parts.iter().map(|p| ChannelStats::combine(p)).collect(),
        },
        bytes,
        decoded,
        config,
    })
}

/// The frames `range` of every channel.
fn slice(channels: &[Vec<i32>], range: &std::ops::Range<usize>) -> Vec<Vec<i32>> {
    channels.iter().map(|c| c[range.clone()].to_vec()).collect()
}

/// Encode with a fixed configuration. `analyses[s][c]` is the tone analysis of
/// coded channel c (after the stereo transform) in segment s, needed where its
/// config has gradients.
pub fn encode(
    channels: &[Vec<i32>],
    analyses: &[Vec<Option<&ToneAnalysis>>],
    config: &StreamConfig,
) -> Result<Encoded, Error> {
    let len = validate_input(channels)?;
    let c = channels.len();
    validate_config(config, c, len)?;
    let ranges = segment_ranges(len, config.segment_exponent);
    if analyses.len() != ranges.len() || analyses.iter().any(|a| a.len() != c) {
        return Err(Error::InvalidParameter("SWL2 tone analyses"));
    }
    let mut states: Vec<correction::State> = vec![correction::State::new(); c];
    let mut coded = Vec::with_capacity(ranges.len());
    for ((range, segment), analyses) in ranges.iter().zip(&config.segments).zip(analyses) {
        if segment.key {
            states = vec![correction::State::new(); c];
        }
        let signals = transform(&slice(channels, range), segment.stereo);
        let mut out = Vec::with_capacity(c);
        for (((x, cfg), analysis), state) in signals
            .iter()
            .zip(&segment.channels)
            .zip(analyses)
            .zip(states.drain(..))
        {
            out.push(encode_channel(x, *analysis, cfg, config.tolerance, state)?);
        }
        states = out.iter().map(|ch| ch.state.clone()).collect();
        coded.push(Segment {
            config: segment.clone(),
            channels: out,
        });
    }
    assemble(channels, config.clone(), coded)
}

/// Decode a stream from `encode` or `encode_auto`. Needs no source samples; every
/// count is bounded by the header's frame count (at most 2^28) before reading.
pub fn decode(bytes: &[u8]) -> Result<Vec<Vec<i32>>, Error> {
    decode_from(bytes, 0)
}

/// Decode from segment `first` on, which must be a key segment: the earlier
/// segments are skipped by their lengths, not decoded. Returns the frames from
/// that segment's start (`first << segment_exponent`) to the end.
pub fn decode_from(bytes: &[u8], first: usize) -> Result<Vec<Vec<i32>>, Error> {
    let mut b = Bytes::new(bytes);
    if b.take(4)? != MAGIC || b.u8()? != VERSION {
        return Err(corrupt());
    }
    let channels = b.u8()? as usize;
    let exponent = u32::from(b.u8()?);
    let _reserved = b.u8()?;
    let len = b.u32()? as usize;
    let tolerance = b.u32()?;
    if !(1..=8).contains(&channels)
        || len == 0
        || len > MAX_SAMPLES
        || (tolerance > 1 << 20 && tolerance != PARAMETRIC)
        || (exponent != 0 && !(12..=24).contains(&exponent))
    {
        return Err(corrupt());
    }
    let ranges = segment_ranges(len, exponent);
    if first >= ranges.len() {
        return Err(Error::InvalidParameter("SWL2 segment"));
    }
    for _ in 0..first {
        let size = b.u32()? as usize;
        b.take(size)?;
    }
    let mut out: Vec<Vec<i32>> = Vec::with_capacity(channels);
    for _ in 0..channels {
        out.push(crate::buffer(len - ranges[first].start)?);
    }
    let mut states: Vec<correction::State> = Vec::new();
    for (index, range) in ranges.into_iter().enumerate().skip(first) {
        let size = b.u32()? as usize;
        let mut body = Bytes::new(b.take(size)?);
        let flags = body.u8()?;
        let key = flags & 1 == 1;
        let stereo = Stereo::from_code((flags >> 1) & 3).ok_or_else(corrupt)?;
        if flags >> 3 != 0 || (index == 0 && !key) {
            return Err(corrupt());
        }
        if index == first && !key {
            return Err(Error::InvalidParameter("SWL2 segment is not a key"));
        }
        if stereo != Stereo::Independent && (channels != 2 || tolerance != 0) {
            return Err(corrupt());
        }
        if key {
            states = vec![correction::State::new(); channels];
        }
        let frames = range.len();
        let mut coded = Vec::with_capacity(channels);
        for state in states.iter_mut() {
            coded.push(decode_channel(&mut body, frames, tolerance, state)?);
        }
        if body.pos != size {
            return Err(corrupt());
        }
        for (o, c) in out.iter_mut().zip(inverse(coded, stereo)) {
            o.extend(
                c.iter()
                    .map(|&v| v.clamp(-SAMPLE_LIMIT, SAMPLE_LIMIT) as i32),
            );
        }
    }
    if b.pos != bytes.len() {
        return Err(corrupt());
    }
    Ok(out)
}

/// One channel of a segment: its chunks, rendered and corrected; `state` is
/// the correction state before the segment and becomes the one after it.
fn decode_channel(
    b: &mut Bytes,
    len: usize,
    tolerance: u32,
    state: &mut correction::State,
) -> Result<Vec<i64>, Error> {
    let count = b.u8()?;
    let mut prediction = vec![0i64; len];
    let mut corrected = None;
    let mut seen = [false; 4];
    for _ in 0..count {
        let kind = b.u8()?;
        let size = b.u32()? as usize;
        let payload = b.take(size)?;
        if !(1..=3).contains(&kind)
            || seen[kind as usize]
            || corrected.is_some()
            || (kind == CORRECTION && tolerance == PARAMETRIC)
        {
            return Err(corrupt());
        }
        seen[kind as usize] = true;
        match kind {
            GRADIENTS => {
                let parsed = gradients::parse(payload, len)?;
                let mut tonal = vec![0.0; len];
                gradients::render(&parsed, &mut tonal);
                for (p, t) in prediction.iter_mut().zip(round_all(&tonal)) {
                    *p += t;
                }
            }
            ANCHORS => {
                let parsed = anchors::parse(payload, len)?;
                for (p, v) in prediction
                    .iter_mut()
                    .zip(round_all(&anchors::render(&parsed, len)))
                {
                    *p += v;
                }
            }
            _ => {
                let inherited = std::mem::replace(state, correction::State::new());
                let (c, next) = correction::decode(payload, &prediction, tolerance, inherited)?;
                *state = next;
                corrected = Some(c);
            }
        }
    }
    match (corrected, tolerance == PARAMETRIC) {
        (Some(c), false) => Ok(c),
        (None, true) => Ok(prediction),
        _ => Err(corrupt()),
    }
}

/// What `encode_auto` may try.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoParams {
    pub tolerance: u32,
    /// Tone analysis of each coded channel; `upper` is the widest gradient band.
    pub tones: ToneParams,
    /// Try gradient layers (needs the tone analysis, the slow part).
    pub gradients: bool,
    /// Try SinWhisky anchor layers.
    pub anchors: bool,
    /// Give the correction layer the LMS cascade (false: LPC only).
    pub cascade: bool,
    /// Segments of 2^segment_exponent frames (12..=24), each choosing its own
    /// stereo mode and layers, or 0 for one segment.
    pub segment_exponent: u32,
    /// Every `key_interval`-th segment is a key (random access, adaptive state
    /// restarts); 0: only the first.
    pub key_interval: usize,
}

/// One configuration the search measured.
#[derive(Debug, Clone, PartialEq)]
pub struct Trial {
    pub segment: usize,
    pub channel: usize,
    pub config: ChannelConfig,
    pub bytes: usize,
}

/// Choose, segment by segment, the stereo mode and, per channel, the layers and
/// their parameters that give the smallest segment, by exact coded size from
/// the state the previous segment left (greedy: a choice is never revisited).
/// Without any helpful layer a channel is the correction layer alone, so it is
/// never larger than that. Returns the stream and every trial.
pub fn encode_auto(
    channels: &[Vec<i32>],
    params: &AutoParams,
) -> Result<(Encoded, Vec<Trial>), Error> {
    let len = validate_input(channels)?;
    let c = channels.len();
    // Long blocks win most often with the cascade (fewer predictor headers).
    let plain = ChannelConfig {
        cascade: params.cascade,
        block_exponent: if params.cascade { 13 } else { 12 },
        ..ChannelConfig::default()
    };
    let tolerance = params.tolerance;
    if tolerance > 1 << 20 {
        // Parametric streams have no exact size/quality trade-off to search.
        return Err(Error::InvalidParameter("SWL2 tolerance"));
    }
    if params.segment_exponent != 0 && !(12..=24).contains(&params.segment_exponent) {
        return Err(Error::InvalidParameter("SWL2 segment exponent"));
    }
    let ranges = segment_ranges(len, params.segment_exponent);
    let mut trials = Vec::new();
    let mut coded: Vec<Segment> = Vec::with_capacity(ranges.len());
    let mut states: Vec<correction::State> = vec![correction::State::new(); c];
    let upper = params.tones.upper;
    let base = (tolerance as f64).max(2.0);
    for (index, range) in ranges.iter().enumerate() {
        let key = index == 0 || (params.key_interval > 0 && index % params.key_interval == 0);
        if key {
            states = vec![correction::State::new(); c];
        }
        let source = slice(channels, range);
        let plainly = |x: &[i64], state: &correction::State| {
            encode_channel(x, None, &plain, tolerance, state.clone())
        };
        // Stereo: the cheapest pair among the four signals, coded plainly,
        // each channel from its own inherited state.
        let (stereo, mut firsts) = if c == 2 && tolerance == 0 {
            let wide = |c: &Vec<i32>| c.iter().map(|&v| i64::from(v)).collect::<Vec<i64>>();
            let (l, r) = (wide(&source[0]), wide(&source[1]));
            let s: Vec<i64> = l.iter().zip(&r).map(|(a, b)| a - b).collect();
            let m: Vec<i64> = l.iter().zip(&r).map(|(a, b)| (a + b) >> 1).collect();
            let (l0, r1) = (plainly(&l, &states[0])?, plainly(&r, &states[1])?);
            let (s0, s1) = (plainly(&s, &states[0])?, plainly(&s, &states[1])?);
            let m0 = plainly(&m, &states[0])?;
            let options = [
                (l0.bytes() + r1.bytes(), Stereo::Independent),
                (l0.bytes() + s1.bytes(), Stereo::LeftSide),
                (s0.bytes() + r1.bytes(), Stereo::SideRight),
                (m0.bytes() + s1.bytes(), Stereo::MidSide),
            ];
            let stereo = options.iter().min_by_key(|p| p.0).unwrap().1;
            let pair = match stereo {
                Stereo::Independent => [l0, r1],
                Stereo::LeftSide => [l0, s1],
                Stereo::SideRight => [s0, r1],
                Stereo::MidSide => [m0, s1],
            };
            (stereo, pair.into_iter().map(Some).collect::<Vec<_>>())
        } else {
            (Stereo::Independent, Vec::new())
        };
        firsts.resize_with(c, || None);
        let signals = transform(&source, stereo);
        let mut configs = Vec::with_capacity(c);
        let mut out = Vec::with_capacity(c);
        for (channel, (x, first)) in signals.iter().zip(firsts).enumerate() {
            let state = &states[channel];
            let analysis = if params.gradients {
                let samples: Vec<f64> = x.iter().map(|&v| v as f64).collect();
                Some(ToneAnalysis::new(&samples, params.tones)?)
            } else {
                None
            };
            let first = match first {
                Some(f) => f,
                None => plainly(x, state)?,
            };
            trials.push(Trial {
                segment: index,
                channel,
                config: plain,
                bytes: first.bytes(),
            });
            let mut best = (first, plain);
            let mut consider = |cfg: ChannelConfig,
                                best: &mut (Channel, ChannelConfig)|
             -> Result<(), Error> {
                let coded = encode_channel(x, analysis.as_ref(), &cfg, tolerance, state.clone())?;
                trials.push(Trial {
                    segment: index,
                    channel,
                    config: cfg,
                    bytes: coded.bytes(),
                });
                if coded.bytes() < best.0.bytes() {
                    *best = (coded, cfg);
                }
                Ok(())
            };
            if analysis.is_some() {
                let gradient =
                    |lower: f64, top: f64, precision: f64, segment: f64| GradientConfig {
                        lower,
                        upper: top,
                        precision,
                        segment_tolerance: segment,
                        min_amplitude: 0.0,
                        max_jump_bins: 1.0,
                    };
                for (lo, hi) in [
                    (0.0, 0.25),
                    (0.0, 0.5),
                    (0.0, 1.0),
                    (0.0625, 0.5),
                    (0.0625, 1.0),
                    (0.25, 1.0),
                ] {
                    for precision in [base, 4.0 * base] {
                        let cfg = ChannelConfig {
                            gradients: Some(gradient(
                                lo * upper,
                                hi * upper,
                                precision,
                                2.0 * precision,
                            )),
                            ..plain
                        };
                        consider(cfg, &mut best)?;
                    }
                }
                // Refine precision and segmentation around the best band.
                if let Some(g) = best.1.gradients {
                    for (p, s) in [(0.5, 1.0), (2.0, 1.0), (1.0, 0.5), (1.0, 4.0)] {
                        let cfg = ChannelConfig {
                            gradients: Some(gradient(
                                g.lower,
                                g.upper,
                                (g.precision * p).max(1e-3),
                                g.segment_tolerance * p * s,
                            )),
                            ..best.1
                        };
                        consider(cfg, &mut best)?;
                    }
                }
            }
            if params.anchors {
                let start = best.1;
                for cutoff in [Some(0.02), Some(0.06), None] {
                    for scale in [1.0, 4.0] {
                        let tol = base * scale;
                        let cfg = ChannelConfig {
                            anchors: Some(AnchorConfig {
                                cutoff,
                                tolerance: tol,
                                step: (0.5 * tol).max(1e-3),
                                max_gap: 1024,
                            }),
                            ..start
                        };
                        consider(cfg, &mut best)?;
                    }
                }
            }
            // Block size: one step either way, then onwards while it pays.
            let centre = best.1.block_exponent;
            for direction in [-1i32, 1] {
                let mut exponent = centre as i32 + direction;
                while (8..=15).contains(&exponent) {
                    let before = best.1.block_exponent;
                    let cfg = ChannelConfig {
                        block_exponent: exponent as u32,
                        ..best.1
                    };
                    consider(cfg, &mut best)?;
                    if best.1.block_exponent == before {
                        break;
                    }
                    exponent += direction;
                }
            }
            configs.push(best.1);
            out.push(best.0);
        }
        states = out.iter().map(|ch| ch.state.clone()).collect();
        coded.push(Segment {
            config: SegmentConfig {
                key,
                stereo,
                channels: configs,
            },
            channels: out,
        });
    }
    let config = StreamConfig {
        tolerance,
        segment_exponent: params.segment_exponent,
        segments: coded.iter().map(|s| s.config.clone()).collect(),
    };
    let encoded = assemble(channels, config, coded)?;
    Ok((encoded, trials))
}
