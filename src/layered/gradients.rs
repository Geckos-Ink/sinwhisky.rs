//! Frequency-gradient groups: the tonal chunk of SWL2.
//!
//! Fitted tones (`ToneAnalysis`) in a frequency band are linked into partial
//! tracks across window centres (k*hop). Each track is cut into segments whose
//! phase is quadratic in time, `phi + omega*t + gradient*t^2/2` (a constant
//! frequency gradient), fitted by least squares to the unwrapped phases and the
//! frequencies at its breakpoints; a segment grows while every breakpoint stays
//! within `segment_tolerance` in waveform units. Segments that start at the same
//! centre and last the same number of hops form a group, which shares its start,
//! its duration and its volume-change function (constant, linear per member, or
//! one exponential ratio for all members). Members carry frequency, gradient,
//! amplitude and phase.
//!
//! The chunk is byte-wise: a header declares the bit width of every field, just
//! enough for the largest value in the chunk, and the records are bit-packed
//! MSB-first. A member's precision class c (5 bits) scales its phase, frequency and
//! gradient quanta so each parameter costs about `precision` of waveform error:
//! phase 2*pi/2^(c+3), frequency 2^-(c+t+1), gradient 2^-(c+2t), with
//! t = bits of the segment's span in samples. See docs/SWL2.md.
use super::coder::{corrupt, signed_width, width, BitReader, BitWriter, Bytes};
use crate::bands::ToneAnalysis;
use crate::Error;
use std::f64::consts::{PI, TAU};

const MAX_DURATION: usize = 4096;
const MAX_MEMBERS: usize = 4096;
const MAX_WIDTH: u32 = 62;

/// Which tones become gradient groups, and how precisely.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientConfig {
    /// Band of coded tones, radians per sample: `lower <= omega < upper`.
    pub lower: f64,
    pub upper: f64,
    /// Waveform error each quantized parameter may cause (sample units, >0).
    pub precision: f64,
    /// Largest phase-and-amplitude misfit at a breakpoint before a segment is
    /// cut (sample units, >0).
    pub segment_tolerance: f64,
    /// Tones below this amplitude (and below `precision`) are not coded.
    pub min_amplitude: f64,
    /// Largest frequency change per hop for linking, in resolutions 2*pi/T.
    pub max_jump_bins: f64,
}
impl GradientConfig {
    pub(super) fn validate(&self) -> Result<(), Error> {
        let ok = self.lower >= 0.0
            && self.lower < self.upper
            && self.upper <= PI
            && (1e-3..=1e6).contains(&self.precision)
            && self.segment_tolerance.is_finite()
            && self.segment_tolerance > 0.0
            && self.min_amplitude.is_finite()
            && self.min_amplitude >= 0.0
            && self.max_jump_bins > 0.0
            && self.max_jump_bins <= 64.0;
        if ok {
            Ok(())
        } else {
            Err(Error::InvalidParameter("gradient configuration"))
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Breakpoint {
    frame: usize,
    omega: f64,
    amplitude: f64,
    phase: f64,
}

/// Strongest-first nearest-neighbour linking of each frame's tones to the
/// tracks of the previous frame, predicted by their own gradient.
fn link(analysis: &ToneAnalysis, config: &GradientConfig) -> Vec<Vec<Breakpoint>> {
    let jump = config.max_jump_bins * TAU / analysis.window() as f64;
    let floor = config.min_amplitude.max(config.precision);
    let mut tracks: Vec<Vec<Breakpoint>> = Vec::new();
    let mut live: Vec<usize> = Vec::new();
    let predict = |t: &[Breakpoint]| {
        let last = t[t.len() - 1];
        match t.len() {
            1 => last.omega,
            n => 2.0 * last.omega - t[n - 2].omega,
        }
    };
    for (k, frame) in analysis.frames().iter().enumerate() {
        let mut tones: Vec<_> = frame
            .tones
            .iter()
            .filter(|t| t.omega >= config.lower && t.omega < config.upper && t.amplitude >= floor)
            .collect();
        tones.sort_by(|a, b| b.amplitude.total_cmp(&a.amplitude));
        let mut taken = vec![false; live.len()];
        let mut next = Vec::with_capacity(live.len());
        for tone in tones {
            let point = Breakpoint {
                frame: k,
                omega: tone.omega,
                amplitude: tone.amplitude,
                phase: tone.phase,
            };
            let nearest = live
                .iter()
                .enumerate()
                .filter(|(i, &t)| !taken[*i] && (predict(&tracks[t]) - tone.omega).abs() <= jump)
                .min_by(|a, b| {
                    let da = (predict(&tracks[*a.1]) - tone.omega).abs();
                    let db = (predict(&tracks[*b.1]) - tone.omega).abs();
                    da.total_cmp(&db)
                });
            match nearest {
                Some((i, &t)) => {
                    taken[i] = true;
                    tracks[t].push(point);
                    next.push(t);
                }
                None => {
                    tracks.push(vec![point]);
                    next.push(tracks.len() - 1);
                }
            }
        }
        live = next;
    }
    tracks
}

/// A segment before quantization; time origin at its first centre.
#[derive(Debug, Clone, Copy)]
struct Segment {
    start: usize,
    duration: usize,
    phase: f64,
    omega: f64,
    gradient: f64,
    a0: f64,
    a1: f64,
    fade_in: bool,
    fade_out: bool,
}

/// Least squares of `phase + omega*t + gradient*t^2/2` to unwrapped phases and
/// (weighted by hop/2) frequencies at the breakpoints.
fn fit_phase(points: &[(f64, f64, f64)], hop: f64) -> [f64; 3] {
    let mut n = [[0.0f64; 3]; 3];
    let mut b = [0.0f64; 3];
    let h = 0.5 * hop;
    for &(t, theta, omega) in points {
        let rows = [([1.0, t, 0.5 * t * t], theta), ([0.0, h, h * t], h * omega)];
        for (row, value) in rows {
            for i in 0..3 {
                b[i] += row[i] * value;
                for j in 0..3 {
                    n[i][j] += row[i] * row[j];
                }
            }
        }
    }
    // 3x3 Gaussian elimination with partial pivoting.
    let mut m = [[0.0f64; 4]; 3];
    for i in 0..3 {
        m[i][..3].copy_from_slice(&n[i]);
        m[i][3] = b[i];
    }
    for col in 0..3 {
        let pivot = (col..3)
            .max_by(|&a, &c| m[a][col].abs().total_cmp(&m[c][col].abs()))
            .unwrap();
        m.swap(col, pivot);
        if m[col][col].abs() < 1e-300 {
            return [points[0].1, points[0].2, 0.0];
        }
        let pivot_row = m[col];
        for (row, line) in m.iter_mut().enumerate() {
            if row != col {
                let f = line[col] / pivot_row[col];
                for (value, &p) in line.iter_mut().zip(&pivot_row).skip(col) {
                    *value -= f * p;
                }
            }
        }
    }
    [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]]
}

fn segment_track(track: &[Breakpoint], hop: f64, tolerance: f64) -> Vec<Segment> {
    // Unwrap phases along the track with the mean frequency of each step.
    let mut theta = Vec::with_capacity(track.len());
    theta.push(track[0].phase);
    for i in 1..track.len() {
        let guess = theta[i - 1] + 0.5 * (track[i - 1].omega + track[i].omega) * hop;
        let turns = ((guess - track[i].phase) / TAU).round();
        theta.push(track[i].phase + TAU * turns);
    }
    if track.len() == 1 {
        let p = track[0];
        return vec![Segment {
            start: p.frame,
            duration: 0,
            phase: p.phase,
            omega: p.omega,
            gradient: 0.0,
            a0: p.amplitude,
            a1: p.amplitude,
            fade_in: true,
            fade_out: true,
        }];
    }
    let fit = |s: usize, e: usize| {
        let points: Vec<(f64, f64, f64)> = (s..=e)
            .map(|i| ((i - s) as f64 * hop, theta[i] - theta[s], track[i].omega))
            .collect();
        let [p, w, g] = fit_phase(&points, hop);
        let (a0, a1) = (track[s].amplitude, track[e].amplitude);
        let misfit = (s..=e)
            .map(|i| {
                let t = (i - s) as f64 * hop;
                let model = p + w * t + 0.5 * g * t * t;
                let share = (i - s) as f64 / (e - s) as f64;
                let amplitude = a0 + (a1 - a0) * share;
                track[i].amplitude * (theta[i] - theta[s] - model).abs()
                    + (track[i].amplitude - amplitude).abs()
            })
            .fold(0.0, f64::max);
        ([p + theta[s], w, g], misfit)
    };
    let mut out = Vec::new();
    let mut s = 0;
    while s + 1 < track.len() {
        let mut e = s + 1;
        let mut best = fit(s, e).0;
        while e + 1 < track.len() && e + 1 - s <= MAX_DURATION {
            let (candidate, misfit) = fit(s, e + 1);
            if misfit > tolerance {
                break;
            }
            best = candidate;
            e += 1;
        }
        out.push(Segment {
            start: track[s].frame,
            duration: e - s,
            phase: best[0],
            omega: best[1],
            gradient: best[2],
            a0: track[s].amplitude,
            a1: track[e].amplitude,
            fade_in: s == 0,
            fade_out: e + 1 == track.len(),
        });
        s = e;
    }
    out
}

/// A decoded (or quantized) member: exact values the renderer uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Member {
    class: u32,
    fade_in: bool,
    fade_out: bool,
    omega_index: u64,
    gradient_index: i64,
    a0_index: u64,
    a1_index: u64,
    phase_index: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Envelope {
    Constant,
    Linear,
    Exponential(i64),
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Group {
    start: usize,
    duration: usize,
    envelope: Envelope,
    members: Vec<Member>,
}

/// Bits of the span a member's parameters must stay accurate over.
fn span_bits(duration: usize, hop: usize) -> u32 {
    width(((duration.max(1) + 1) * hop) as u64)
}
fn phase_bits(class: u32) -> u32 {
    class + 3
}
fn omega_exponent(class: u32, duration: usize, hop: usize) -> u32 {
    class + span_bits(duration, hop) + 1
}
fn gradient_exponent(class: u32, duration: usize, hop: usize) -> u32 {
    class + 2 * span_bits(duration, hop)
}

/// Decoded chunk: what the renderer needs.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Gradients {
    hop: usize,
    precision: f64,
    groups: Vec<Group>,
}
impl Gradients {
    pub(super) fn groups(&self) -> usize {
        self.groups.len()
    }
    pub(super) fn members(&self) -> usize {
        self.groups.iter().map(|g| g.members.len()).sum()
    }
}

/// Link, segment, quantize and pack. Returns the chunk payload.
pub(super) fn encode(analysis: &ToneAnalysis, config: &GradientConfig) -> Result<Vec<u8>, Error> {
    config.validate()?;
    let hop = analysis.hop();
    let h = hop as f64;
    let p = config.precision;
    let mut segments: Vec<Segment> = Vec::new();
    for track in link(analysis, config) {
        segments.extend(segment_track(&track, h, config.segment_tolerance));
    }
    segments.sort_by(|a, b| {
        (a.start, a.duration)
            .cmp(&(b.start, b.duration))
            .then(a.omega.total_cmp(&b.omega))
    });
    let mut groups: Vec<Group> = Vec::new();
    let mut i = 0;
    while i < segments.len() {
        let mut j = i;
        while j < segments.len()
            && j - i < MAX_MEMBERS
            && segments[j].start == segments[i].start
            && segments[j].duration == segments[i].duration
        {
            j += 1;
        }
        let (start, duration) = (segments[i].start, segments[i].duration);
        let mut members = Vec::with_capacity(j - i);
        for s in &segments[i..j] {
            let largest = s.a0.max(s.a1);
            let class = (largest / p).log2().ceil().clamp(0.0, 24.0) as u32;
            let eo = omega_exponent(class, duration, hop);
            let eg = gradient_exponent(class, duration, hop);
            let bp = phase_bits(class);
            let a1 = if duration == 0 { s.a0 } else { s.a1 };
            members.push(Member {
                class,
                fade_in: s.fade_in,
                fade_out: s.fade_out,
                omega_index: (s.omega.max(0.0) * (eo as f64).exp2()).round() as u64,
                gradient_index: if duration == 0 {
                    0
                } else {
                    (s.gradient * (eg as f64).exp2()).round() as i64
                },
                a0_index: (s.a0 / p).round() as u64,
                a1_index: (a1 / p).round() as u64,
                phase_index: ((s.phase.rem_euclid(TAU) / TAU * (bp as f64).exp2()).round() as u64)
                    & ((1u64 << bp) - 1),
            });
        }
        // Volume change function shared by the group.
        let envelope = if duration == 0 || members.iter().all(|m| m.a0_index == m.a1_index) {
            Envelope::Constant
        } else {
            let mut ratios: Vec<i64> = members
                .iter()
                .filter(|m| m.a0_index > 0 && m.a1_index > 0)
                .map(|m| (8.0 * (m.a1_index as f64 / m.a0_index as f64).log2()).round() as i64)
                .collect();
            ratios.sort_unstable();
            let shared = ratios.get(ratios.len() / 2).copied();
            let fits = |r: i64| {
                members.iter().all(|m| {
                    let end = (m.a0_index as f64 * (r as f64 / 8.0).exp2()).round() as i64;
                    (end - m.a1_index as i64).abs() <= 1
                })
            };
            match shared {
                Some(r) if ratios.len() == members.len() && fits(r) => Envelope::Exponential(r),
                _ => Envelope::Linear,
            }
        };
        groups.push(Group {
            start,
            duration,
            envelope,
            members,
        });
        i = j;
    }
    pack(hop, p, &groups)
}

fn pack(hop: usize, precision: f64, groups: &[Group]) -> Result<Vec<u8>, Error> {
    let mut w_start = 0;
    let mut w_duration = 0;
    let mut w_count = 0;
    let mut w_amplitude = 0;
    let mut w_ratio = 0;
    let mut o_omega: i64 = -64;
    let mut o_gradient: i64 = -64;
    let mut previous = 0;
    for g in groups {
        w_start = w_start.max(width((g.start - previous) as u64));
        previous = g.start;
        w_duration = w_duration.max(width(g.duration as u64));
        w_count = w_count.max(width(g.members.len() as u64 - 1));
        if let Envelope::Exponential(r) = g.envelope {
            w_ratio = w_ratio.max(signed_width(r));
        }
        for m in &g.members {
            w_amplitude = w_amplitude.max(width(m.a0_index)).max(width(m.a1_index));
            let eo = omega_exponent(m.class, g.duration, hop) as i64;
            o_omega = o_omega.max(width(m.omega_index) as i64 - eo);
            if g.duration > 0 {
                let eg = gradient_exponent(m.class, g.duration, hop) as i64;
                o_gradient = o_gradient.max(signed_width(m.gradient_index) as i64 - eg);
            }
        }
    }
    let o_omega = o_omega.clamp(-127, 127);
    let o_gradient = o_gradient.clamp(-127, 127);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(hop as u16).to_le_bytes());
    bytes.extend_from_slice(&(precision as f32).to_le_bytes());
    for w in [w_start, w_duration, w_count, w_amplitude, w_ratio] {
        bytes.push(w as u8);
    }
    bytes.push(o_omega as i8 as u8);
    bytes.push(o_gradient as i8 as u8);
    bytes.extend_from_slice(&(groups.len() as u32).to_le_bytes());
    let mut bits = BitWriter::new();
    let mut previous = 0;
    for g in groups {
        bits.put((g.start - previous) as u64, w_start);
        previous = g.start;
        bits.put(g.duration as u64, w_duration);
        let (code, ratio) = match g.envelope {
            Envelope::Constant => (0, None),
            Envelope::Linear => (1, None),
            Envelope::Exponential(r) => (2, Some(r)),
        };
        bits.put(code, 2);
        if let Some(r) = ratio {
            bits.put_signed(r, w_ratio);
        }
        bits.put(g.members.len() as u64 - 1, w_count);
        for m in &g.members {
            let (wo, wg) = member_widths(m.class, g.duration, hop, o_omega, o_gradient);
            if wo > MAX_WIDTH || wg > MAX_WIDTH {
                return Err(Error::InvalidParameter("gradient field too wide"));
            }
            bits.put(u64::from(m.class), 5);
            bits.put(u64::from(m.fade_in), 1);
            bits.put(u64::from(m.fade_out), 1);
            bits.put(m.omega_index, wo);
            if g.duration > 0 {
                bits.put_signed(m.gradient_index, wg);
            }
            bits.put(m.a0_index, w_amplitude);
            if g.envelope == Envelope::Linear {
                bits.put(m.a1_index, w_amplitude);
            }
            bits.put(m.phase_index, phase_bits(m.class));
        }
    }
    bytes.extend(bits.finish());
    Ok(bytes)
}

fn member_widths(
    class: u32,
    duration: usize,
    hop: usize,
    o_omega: i64,
    o_gradient: i64,
) -> (u32, u32) {
    let wo = (o_omega + omega_exponent(class, duration, hop) as i64).max(0) as u32;
    let wg = (o_gradient + gradient_exponent(class, duration, hop) as i64).max(0) as u32;
    (wo, wg)
}

/// Parse a chunk payload; `len` bounds every count.
pub(super) fn parse(payload: &[u8], len: usize) -> Result<Gradients, Error> {
    let mut b = Bytes::new(payload);
    let hop = b.u16()? as usize;
    let precision = f64::from(b.f32()?);
    let w: Vec<u32> = (0..5)
        .map(|_| b.u8().map(u32::from))
        .collect::<Result<_, _>>()?;
    let (w_start, w_duration, w_count, w_amplitude, w_ratio) = (w[0], w[1], w[2], w[3], w[4]);
    let o_omega = i64::from(b.i8()?);
    let o_gradient = i64::from(b.i8()?);
    let count = b.u32()? as usize;
    if !(8..=32768).contains(&hop)
        || !(1e-3..=1e6).contains(&precision)
        || w.iter().any(|&v| v > MAX_WIDTH)
        || count > len
    {
        return Err(corrupt());
    }
    let mut bits = BitReader::new(b.rest());
    let mut groups = Vec::with_capacity(count);
    let mut start = 0usize;
    for _ in 0..count {
        start = start
            .checked_add(bits.get(w_start)? as usize)
            .ok_or_else(corrupt)?;
        let duration = bits.get(w_duration)? as usize;
        if start.saturating_mul(hop) > len + hop || duration > MAX_DURATION {
            return Err(corrupt());
        }
        let envelope = match bits.get(2)? {
            0 => Envelope::Constant,
            1 => Envelope::Linear,
            2 => Envelope::Exponential(bits.get_signed(w_ratio)?),
            _ => return Err(corrupt()),
        };
        if duration == 0 && envelope != Envelope::Constant {
            return Err(corrupt());
        }
        let members = bits.get(w_count)? as usize + 1;
        if members > MAX_MEMBERS {
            return Err(corrupt());
        }
        let mut list = Vec::with_capacity(members);
        for _ in 0..members {
            let class = bits.get(5)? as u32;
            if class > 24 {
                return Err(corrupt());
            }
            let fade_in = bits.get(1)? == 1;
            let fade_out = bits.get(1)? == 1;
            let (wo, wg) = member_widths(class, duration, hop, o_omega, o_gradient);
            if wo > MAX_WIDTH || wg > MAX_WIDTH {
                return Err(corrupt());
            }
            let omega_index = bits.get(wo)?;
            let gradient_index = if duration > 0 {
                bits.get_signed(wg)?
            } else {
                0
            };
            let a0_index = bits.get(w_amplitude)?;
            let a1_index = match envelope {
                Envelope::Linear => bits.get(w_amplitude)?,
                _ => a0_index,
            };
            let phase_index = bits.get(phase_bits(class))?;
            list.push(Member {
                class,
                fade_in,
                fade_out,
                omega_index,
                gradient_index,
                a0_index,
                a1_index,
                phase_index,
            });
        }
        groups.push(Group {
            start,
            duration,
            envelope,
            members: list,
        });
    }
    Ok(Gradients {
        hop,
        precision,
        groups,
    })
}

/// Add every member to `out` (sample n = start*hop + t): the quadratic phase over
/// the group's span with its volume function. A track's first segment fades in
/// over one hop and its last fades out over one hop, at constant frequency.
/// Where a track continues from one segment into the next, both are extended by
/// half a hop across the shared centre and cross-faded linearly, so the joint
/// has no jump.
pub(super) fn render(gradients: &Gradients, out: &mut [f64]) {
    let hop = gradients.hop;
    let h = hop as f64;
    let x = 0.5 * h;
    let len = out.len() as i64;
    for g in &gradients.groups {
        let span = (g.duration * hop) as f64;
        let origin = (g.start * hop) as i64;
        for m in &g.members {
            let omega =
                m.omega_index as f64 / (omega_exponent(m.class, g.duration, hop) as f64).exp2();
            let gradient = m.gradient_index as f64
                / (gradient_exponent(m.class, g.duration, hop) as f64).exp2();
            let phase = m.phase_index as f64 * TAU / (phase_bits(m.class) as f64).exp2();
            let a0 = m.a0_index as f64 * gradients.precision;
            let a1 = m.a1_index as f64 * gradients.precision;
            let amplitude = |t: f64| -> f64 {
                let v = match g.envelope {
                    Envelope::Constant => a0,
                    Envelope::Linear => a0 + (a1 - a0) * t / span,
                    Envelope::Exponential(r) => a0 * (r as f64 / 8.0 * t / span).exp2(),
                };
                v.max(0.0)
            };
            let end_amplitude = if g.duration == 0 { a0 } else { amplitude(span) };
            let end_omega = omega + gradient * span;
            let end_phase = phase + omega * span + 0.5 * gradient * span * span;
            let first = if m.fade_in { -h } else { -x };
            let last = span + if m.fade_out { h } else { x };
            let lo = (origin + first as i64).max(0);
            let hi = (origin + last as i64).min(len);
            for n in lo..hi {
                let t = (n - origin) as f64;
                let (a, angle) = if m.fade_in && t < 0.0 {
                    (a0 * (t + h) / h, phase + omega * t)
                } else if m.fade_out && t >= span {
                    let u = t - span;
                    (end_amplitude * (1.0 - u / h), end_phase + end_omega * u)
                } else {
                    let mut weight = 1.0;
                    if !m.fade_in && t < x {
                        weight = (t + x) / (2.0 * x);
                    }
                    if !m.fade_out && t >= span - x {
                        weight = weight.min((span + x - t) / (2.0 * x));
                    }
                    (
                        weight * amplitude(t),
                        phase + omega * t + 0.5 * gradient * t * t,
                    )
                };
                out[n as usize] += a * angle.cos();
            }
        }
    }
}
