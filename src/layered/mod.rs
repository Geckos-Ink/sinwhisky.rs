//! SWL2: layered closed-loop coding of integer audio (PAPER_EQ_COMPRESSION.md,
//! Section 4). Experimental: not a stable or fuzz-validated format. The byte
//! layout is specified in docs/SWL2.md.
//!
//! Each channel is a list of chunks, each predicting part of the signal from what
//! the previous chunks left:
//!
//! 1. `gradients` (optional): partial tracks of a frequency band as groups of
//!    frequency-gradient segments that share start, duration and volume function;
//!    fields are bit-packed with widths declared in the chunk header.
//! 2. `anchors` (optional): SinWhisky anchors at dynamic positions whose circles
//!    rebuild a low band (or all) of what the gradients left.
//! 3. `correction` (always, last): closed-loop LPC of the source with the rounded
//!    layer prediction as side information (a per-block gain decides how much of
//!    it to subtract) and quantization step `2*tolerance + 1`.
//!
//! Every decoded sample is within `tolerance` of the source (0 is lossless) and
//! the decoder needs no source samples. Lossless stereo streams may decorrelate
//! the two channels (left/side, side/right, mid/side) as FLAC does.
//!
//! `encode_auto` chooses everything by exact coded size: the stereo mode, whether
//! a gradient layer pays and for which band and precision, whether anchors pay,
//! and the block size; a layer is kept only if the complete stream gets smaller.
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
const VERSION: u8 = 2;
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

/// Layers of one coded channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelConfig {
    pub gradients: Option<GradientConfig>,
    pub anchors: Option<AnchorConfig>,
    /// Correction blocks of 2^block_exponent samples, 8..=15.
    pub block_exponent: u32,
}
impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            gradients: None,
            anchors: None,
            block_exponent: 12,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamConfig {
    /// Largest error at every decoded sample, 0..=2^20 (0 is lossless), or
    /// `PARAMETRIC` for no correction layer.
    pub tolerance: u32,
    /// Must be `Independent` unless the stream has two channels and tolerance 0.
    pub stereo: Stereo,
    /// One per coded channel.
    pub channels: Vec<ChannelConfig>,
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
    /// RMS of the coded channel and of what the correction layer had to code.
    pub source_rms: f64,
    pub residual_rms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamStats {
    pub frames: usize,
    pub header_bytes: usize,
    pub total_bytes: usize,
    pub max_error: u32,
    pub rms_error: f64,
    /// 10*log10(source energy / error energy); infinite when lossless.
    pub snr_db: f64,
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
        });
    }
    let layered = !chunks.is_empty();
    let correction = correction::encode(
        x,
        layered.then_some(&prediction[..]),
        tolerance,
        config.block_exponent,
    );
    stats.correction_bytes = correction.payload.len();
    stats.written = correction.written;
    stats.mean_order = correction.mean_order;
    stats.mean_gain = correction.mean_gain;
    let reconstructed = correction.reconstructed;
    chunks.push((CORRECTION, correction.payload));
    Ok(Channel {
        chunks,
        reconstructed,
        stats,
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

fn assemble(
    channels: &[Vec<i32>],
    config: &StreamConfig,
    coded: Vec<Channel>,
) -> Result<Encoded, Error> {
    let len = channels[0].len();
    let mut bytes = Vec::with_capacity(HEADER + coded.iter().map(Channel::bytes).sum::<usize>());
    bytes.extend_from_slice(MAGIC);
    bytes.push(VERSION);
    bytes.push(channels.len() as u8);
    bytes.push(config.stereo.code());
    bytes.push(0);
    bytes.extend_from_slice(&(len as u32).to_le_bytes());
    bytes.extend_from_slice(&config.tolerance.to_le_bytes());
    for c in &coded {
        bytes.push(c.chunks.len() as u8);
        for (kind, payload) in &c.chunks {
            bytes.push(*kind);
            bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            bytes.extend_from_slice(payload);
        }
    }
    let stats_channels: Vec<ChannelStats> = coded.iter().map(|c| c.stats).collect();
    let restored = inverse(
        coded.into_iter().map(|c| c.reconstructed).collect(),
        config.stereo,
    );
    let decoded: Vec<Vec<i32>> = restored
        .iter()
        .map(|c| {
            c.iter()
                .map(|&v| v.clamp(-SAMPLE_LIMIT, SAMPLE_LIMIT) as i32)
                .collect()
        })
        .collect();
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
            max_error: max_error.min(u64::from(u32::MAX)) as u32,
            rms_error: (square / total as f64).sqrt(),
            snr_db: 10.0 * (energy / square).log10(),
            channels: stats_channels,
        },
        bytes,
        decoded,
        config: config.clone(),
    })
}

/// Encode with a fixed configuration. `analyses[c]` is the tone analysis of coded
/// channel c (after the stereo transform), needed where its config has gradients.
pub fn encode(
    channels: &[Vec<i32>],
    analyses: &[Option<&ToneAnalysis>],
    config: &StreamConfig,
) -> Result<Encoded, Error> {
    validate_input(channels)?;
    let c = channels.len();
    if config.channels.len() != c
        || analyses.len() != c
        || (config.tolerance > 1 << 20 && config.tolerance != PARAMETRIC)
        || (config.stereo != Stereo::Independent && (c != 2 || config.tolerance != 0))
    {
        return Err(Error::InvalidParameter("SWL2 stream configuration"));
    }
    let coded = transform(channels, config.stereo);
    let mut out = Vec::with_capacity(c);
    for ((x, cfg), analysis) in coded.iter().zip(&config.channels).zip(analyses) {
        out.push(encode_channel(x, *analysis, cfg, config.tolerance)?);
    }
    assemble(channels, config, out)
}

/// Decode a stream from `encode` or `encode_auto`. Needs no source samples; every
/// count is bounded by the header's frame count (at most 2^28) before reading.
pub fn decode(bytes: &[u8]) -> Result<Vec<Vec<i32>>, Error> {
    let mut b = Bytes::new(bytes);
    if b.take(4)? != MAGIC || b.u8()? != VERSION {
        return Err(corrupt());
    }
    let channels = b.u8()? as usize;
    let stereo = Stereo::from_code(b.u8()?).ok_or_else(corrupt)?;
    let _reserved = b.u8()?;
    let len = b.u32()? as usize;
    let tolerance = b.u32()?;
    if !(1..=8).contains(&channels)
        || len == 0
        || len > MAX_SAMPLES
        || (tolerance > 1 << 20 && tolerance != PARAMETRIC)
        || (stereo != Stereo::Independent && (channels != 2 || tolerance != 0))
    {
        return Err(corrupt());
    }
    let mut coded = Vec::with_capacity(channels);
    for _ in 0..channels {
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
                _ => corrected = Some(correction::decode(payload, &prediction, tolerance)?),
            }
        }
        let channel = match (corrected, tolerance == PARAMETRIC) {
            (Some(c), false) => c,
            (None, true) => prediction,
            _ => return Err(corrupt()),
        };
        coded.push(channel);
    }
    if b.pos != bytes.len() {
        return Err(corrupt());
    }
    Ok(inverse(coded, stereo)
        .iter()
        .map(|c| {
            c.iter()
                .map(|&v| v.clamp(-SAMPLE_LIMIT, SAMPLE_LIMIT) as i32)
                .collect()
        })
        .collect())
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
}

/// One configuration the search measured.
#[derive(Debug, Clone, PartialEq)]
pub struct Trial {
    pub channel: usize,
    pub config: ChannelConfig,
    pub bytes: usize,
}

/// Choose the stereo mode and, per channel, the layers and their parameters that
/// give the smallest complete stream, by exact coded size. Without any helpful
/// layer the result is the correction layer alone, so it is never larger than
/// that. Returns the stream and every trial.
pub fn encode_auto(
    channels: &[Vec<i32>],
    params: &AutoParams,
) -> Result<(Encoded, Vec<Trial>), Error> {
    let len = validate_input(channels)?;
    let plain = ChannelConfig::default();
    let tolerance = params.tolerance;
    if tolerance > 1 << 20 {
        // Parametric streams have no exact size/quality trade-off to search.
        return Err(Error::InvalidParameter("SWL2 tolerance"));
    }
    let size = |x: &[i64], a: Option<&ToneAnalysis>, cfg: &ChannelConfig| {
        encode_channel(x, a, cfg, tolerance).map(|c| c.bytes())
    };
    // Stereo: the cheapest pair among the four signals, coded plainly.
    let stereo = if channels.len() == 2 && tolerance == 0 {
        let wide = |c: &Vec<i32>| c.iter().map(|&v| i64::from(v)).collect::<Vec<i64>>();
        let (l, r) = (wide(&channels[0]), wide(&channels[1]));
        let s: Vec<i64> = l.iter().zip(&r).map(|(a, b)| a - b).collect();
        let m: Vec<i64> = l.iter().zip(&r).map(|(a, b)| (a + b) >> 1).collect();
        let [cl, cr, cs, cm] = [&l, &r, &s, &m].map(|x| size(x, None, &plain));
        let (cl, cr, cs, cm) = (cl?, cr?, cs?, cm?);
        [
            (cl + cr, Stereo::Independent),
            (cl + cs, Stereo::LeftSide),
            (cs + cr, Stereo::SideRight),
            (cm + cs, Stereo::MidSide),
        ]
        .into_iter()
        .min_by_key(|p| p.0)
        .map(|p| p.1)
        .unwrap()
    } else {
        Stereo::Independent
    };
    let coded = transform(channels, stereo);
    let mut trials = Vec::new();
    let mut analyses = Vec::with_capacity(coded.len());
    let mut configs = Vec::with_capacity(coded.len());
    let upper = params.tones.upper;
    let base = (tolerance as f64).max(2.0);
    for (index, x) in coded.iter().enumerate() {
        let analysis = if params.gradients {
            let samples: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            Some(ToneAnalysis::new(&samples, params.tones)?)
        } else {
            None
        };
        let mut best = (size(x, None, &plain)?, plain);
        trials.push(Trial {
            channel: index,
            config: plain,
            bytes: best.0,
        });
        let mut consider =
            |cfg: ChannelConfig, best: &mut (usize, ChannelConfig)| -> Result<(), Error> {
                let bytes = size(x, analysis.as_ref(), &cfg)?;
                trials.push(Trial {
                    channel: index,
                    config: cfg,
                    bytes,
                });
                if bytes < best.0 {
                    *best = (bytes, cfg);
                }
                Ok(())
            };
        if analysis.is_some() {
            let gradient = |lower: f64, top: f64, precision: f64, segment: f64| GradientConfig {
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
        for exponent in [11, 13] {
            let cfg = ChannelConfig {
                block_exponent: exponent,
                ..best.1
            };
            consider(cfg, &mut best)?;
        }
        analyses.push(analysis);
        configs.push(best.1);
    }
    let config = StreamConfig {
        tolerance,
        stereo,
        channels: configs,
    };
    let refs: Vec<Option<&ToneAnalysis>> = analyses.iter().map(Option::as_ref).collect();
    let encoded = encode(channels, &refs, &config)?;
    debug_assert_eq!(encoded.decoded.len(), channels.len());
    let _ = len;
    Ok((encoded, trials))
}
