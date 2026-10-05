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
//! A member's precision class c scales its phase, frequency and gradient quanta
//! so each parameter costs about `precision` of waveform error: phase
//! 2*pi/2^(c+3), frequency 2^-(c+t+1), gradient 2^-(c+2t), with t = bits of the
//! segment's span in samples. The chunk is range coded with an adaptive model
//! per field. A member that continues a track names its predecessor among the
//! tracks still open at its start (by rank in end frequency) and codes only the
//! residuals of its class, frequency, amplitude and phase from the
//! predecessor's end values, which encoder and decoder compute alike; a new
//! member codes its frequency as a step from the previous new member of its
//! group. See docs/SWL2.md.
use super::coder::{corrupt, width, Bytes, Gamma, Prob, RangeDecoder, RangeEncoder};
use crate::bands::ToneAnalysis;
use crate::Error;
use std::f64::consts::{PI, TAU};

const MAX_DURATION: usize = 4096;
const MAX_MEMBERS: usize = 4096;

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
    /// Index of the track the segment belongs to.
    track: usize,
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
            track: 0,
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
            track: 0,
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
    for (id, track) in link(analysis, config).into_iter().enumerate() {
        segments.extend(
            segment_track(&track, h, config.segment_tolerance)
                .into_iter()
                .map(|s| Segment { track: id, ..s }),
        );
    }
    // The last member (group, index) of each track so far.
    let mut last: std::collections::HashMap<usize, (usize, usize)> =
        std::collections::HashMap::new();
    let mut links: Vec<Vec<Option<(usize, usize)>>> = Vec::new();
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
        let mut group_links = Vec::with_capacity(j - i);
        for s in &segments[i..j] {
            group_links.push(if s.fade_in {
                None
            } else {
                last.get(&s.track).copied()
            });
            last.insert(s.track, (groups.len(), members.len()));
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
        links.push(group_links);
        i = j;
    }
    pack(hop, p, &groups, &links)
}

/// Where a track's next segment should start: the end of an open member.
#[derive(Debug, Clone, Copy)]
struct OpenEnd {
    omega: f64,
    phase: f64,
    amplitude_index: u64,
    class: u32,
    /// (group, member) of the open member.
    key: (usize, usize),
}

/// Members whose track continues (no fade-out), by the frame where they end,
/// each list in increasing end frequency. Encoder and decoder keep the same
/// list, so a continuing member names its predecessor by rank.
#[derive(Default)]
struct OpenTracks {
    by_frame: std::collections::BTreeMap<usize, Vec<OpenEnd>>,
}
impl OpenTracks {
    fn insert(&mut self, frame: usize, end: OpenEnd) {
        let list = self.by_frame.entry(frame).or_default();
        let at = list.partition_point(|e| {
            e.omega
                .total_cmp(&end.omega)
                .then(e.key.cmp(&end.key))
                .is_lt()
        });
        list.insert(at, end);
    }
    fn rank(&self, frame: usize, key: (usize, usize)) -> Option<usize> {
        self.by_frame.get(&frame)?.iter().position(|e| e.key == key)
    }
    fn take(&mut self, frame: usize, rank: usize) -> Option<OpenEnd> {
        let list = self.by_frame.get_mut(&frame)?;
        (rank < list.len()).then(|| list.remove(rank))
    }
    /// Forget tracks that ended before `frame`: nothing can continue them.
    fn drop_before(&mut self, frame: usize) {
        self.by_frame = self.by_frame.split_off(&frame);
    }
}

/// The end of a decoded member: frequency and phase at the end of its span (in
/// binary64 additions and multiplications only, so encoder and decoder agree on
/// every platform) and its end amplitude index.
fn open_end(m: &Member, g: &Group, hop: usize, key: (usize, usize)) -> OpenEnd {
    let span = (g.duration * hop) as f64;
    let omega = m.omega_index as f64 / (omega_exponent(m.class, g.duration, hop) as f64).exp2();
    let gradient =
        m.gradient_index as f64 / (gradient_exponent(m.class, g.duration, hop) as f64).exp2();
    let phase = m.phase_index as f64 * TAU / (phase_bits(m.class) as f64).exp2();
    OpenEnd {
        omega: omega + gradient * span,
        phase: phase + omega * span + 0.5 * gradient * span * span,
        amplitude_index: if g.envelope == Envelope::Linear {
            m.a1_index
        } else {
            m.a0_index
        },
        class: m.class,
        key,
    }
}

/// Predicted frequency and phase indices of a continuing member of class
/// `class` in a group of `duration` hops.
fn predicted(end: &OpenEnd, class: u32, duration: usize, hop: usize) -> (u64, u64) {
    let eo = omega_exponent(class, duration, hop);
    let bp = phase_bits(class);
    let omega = (end.omega.max(0.0) * (eo as f64).exp2()).round();
    let phase = (end.phase.rem_euclid(TAU) / TAU * (bp as f64).exp2()).round() as u64;
    (
        omega.min(MAX_INDEX as f64) as u64,
        phase & ((1u64 << bp) - 1),
    )
}

/// Index of frequency `f` on the grid of exponent `e`.
fn on_grid(f: f64, e: u32) -> u64 {
    (f * (e as f64).exp2()).round().min(MAX_INDEX as f64) as u64
}

/// Largest frequency or gradient index magnitude.
const MAX_INDEX: u64 = 1 << 46;

fn signed(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}
fn unsigned(v: u64) -> i64 {
    (v >> 1) as i64 ^ -((v & 1) as i64)
}

/// Phase difference as the signed residue of the shortest way round.
fn wrap(d: i64, bits: u32) -> i64 {
    let full = 1i64 << bits;
    let r = d.rem_euclid(full);
    if r >= full / 2 {
        r - full
    } else {
        r
    }
}

/// Adaptive models of the range-coded chunk body.
struct Models {
    start: Gamma,
    duration: Gamma,
    envelope: [Prob; 2],
    ratio: Gamma,
    count: Gamma,
    continued: Prob,
    fade_out: [Prob; 2],
    link: Gamma,
    class: Gamma,
    class_change: Gamma,
    omega: Gamma,
    omega_step: Gamma,
    omega_residual: Gamma,
    gradient: [Gamma; 2],
    amplitude: Gamma,
    amplitude_residual: Gamma,
    amplitude_end: Gamma,
    phase_residual: Gamma,
}
impl Models {
    fn new() -> Self {
        Self {
            start: Gamma::new(),
            duration: Gamma::new(),
            envelope: [Prob::NEW; 2],
            ratio: Gamma::new(),
            count: Gamma::new(),
            continued: Prob::NEW,
            fade_out: [Prob::NEW; 2],
            link: Gamma::new(),
            class: Gamma::new(),
            class_change: Gamma::new(),
            omega: Gamma::new(),
            omega_step: Gamma::new(),
            omega_residual: Gamma::new(),
            gradient: [Gamma::new(), Gamma::new()],
            amplitude: Gamma::new(),
            amplitude_residual: Gamma::new(),
            amplitude_end: Gamma::new(),
            phase_residual: Gamma::new(),
        }
    }
}

/// Header: u16 hop, f32 precision, u32 group count; then the range-coded groups.
fn pack(
    hop: usize,
    precision: f64,
    groups: &[Group],
    links: &[Vec<Option<(usize, usize)>>],
) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(hop as u16).to_le_bytes());
    bytes.extend_from_slice(&(precision as f32).to_le_bytes());
    bytes.extend_from_slice(&(groups.len() as u32).to_le_bytes());
    let mut e = RangeEncoder::new();
    let mut m = Models::new();
    let mut open = OpenTracks::default();
    let mut previous = 0;
    // Rank of the last claimed predecessor at the current start frame.
    let mut last_rank = 0usize;
    for (gi, g) in groups.iter().enumerate() {
        if g.start != previous || gi == 0 {
            open.drop_before(g.start);
            last_rank = 0;
        }
        m.start.put(&mut e, (g.start - previous) as u64);
        previous = g.start;
        m.duration.put(&mut e, g.duration as u64);
        if g.duration > 0 {
            let (code, ratio) = match g.envelope {
                Envelope::Constant => (0, None),
                Envelope::Linear => (1, None),
                Envelope::Exponential(r) => (2, Some(r)),
            };
            e.bit(&mut m.envelope[0], code != 0);
            if code != 0 {
                e.bit(&mut m.envelope[1], code == 2);
            }
            if let Some(r) = ratio {
                m.ratio.put(&mut e, signed(r));
            }
        }
        m.count.put(&mut e, g.members.len() as u64 - 1);
        // Frequency of the previous new member of the group.
        let mut below: Option<f64> = None;
        for (mi, member) in g.members.iter().enumerate() {
            let link = links[gi][mi];
            e.bit(&mut m.continued, link.is_some());
            let end = match link {
                Some(key) => {
                    let rank = open
                        .rank(g.start, key)
                        .ok_or(Error::InvalidParameter("gradient track link"))?;
                    m.link.put(&mut e, signed(rank as i64 - last_rank as i64));
                    last_rank = rank;
                    open.take(g.start, rank)
                }
                None => None,
            };
            e.bit(&mut m.fade_out[usize::from(end.is_some())], member.fade_out);
            let bp = phase_bits(member.class);
            match end {
                Some(end) => {
                    m.class_change.put(
                        &mut e,
                        signed(i64::from(member.class) - i64::from(end.class)),
                    );
                    let (omega, phase) = predicted(&end, member.class, g.duration, hop);
                    m.omega_residual
                        .put(&mut e, signed(member.omega_index as i64 - omega as i64));
                    m.amplitude_residual.put(
                        &mut e,
                        signed(member.a0_index as i64 - end.amplitude_index as i64),
                    );
                    let d = wrap(member.phase_index as i64 - phase as i64, bp);
                    m.phase_residual.put(&mut e, signed(d));
                }
                None => {
                    m.class.put(&mut e, u64::from(member.class));
                    let eo = omega_exponent(member.class, g.duration, hop);
                    match below {
                        Some(f) => {
                            let guess = on_grid(f, eo);
                            m.omega_step
                                .put(&mut e, signed(member.omega_index as i64 - guess as i64));
                        }
                        None => m.omega.put(&mut e, member.omega_index),
                    }
                    below = Some(member.omega_index as f64 / (eo as f64).exp2());
                    m.amplitude.put(&mut e, member.a0_index);
                    e.direct(member.phase_index, bp);
                }
            }
            if g.duration > 0 {
                m.gradient[usize::from(end.is_some())].put(&mut e, signed(member.gradient_index));
            }
            if g.envelope == Envelope::Linear {
                m.amplitude_end.put(
                    &mut e,
                    signed(member.a1_index as i64 - member.a0_index as i64),
                );
            }
            if member.omega_index >= MAX_INDEX || member.gradient_index.unsigned_abs() >= MAX_INDEX
            {
                return Err(Error::InvalidParameter("gradient field too wide"));
            }
        }
        for (mi, member) in g.members.iter().enumerate() {
            if !member.fade_out {
                open.insert(g.start + g.duration, open_end(member, g, hop, (gi, mi)));
            }
        }
    }
    bytes.extend(e.finish());
    Ok(bytes)
}

/// Parse a chunk payload; `len` bounds every count.
pub(super) fn parse(payload: &[u8], len: usize) -> Result<Gradients, Error> {
    let mut b = Bytes::new(payload);
    let hop = b.u16()? as usize;
    let precision = f64::from(b.f32()?);
    let count = b.u32()? as usize;
    if !(8..=32768).contains(&hop) || !(1e-3..=1e6).contains(&precision) || count > len {
        return Err(corrupt());
    }
    let mut d = RangeDecoder::new(b.rest())?;
    let mut m = Models::new();
    let mut open = OpenTracks::default();
    let mut groups: Vec<Group> = Vec::with_capacity(count);
    let mut start = 0usize;
    let mut last_rank = 0usize;
    for gi in 0..count {
        let delta = d_get(&mut m.start, &mut d, len as u64)? as usize;
        if delta > 0 || gi == 0 {
            open.drop_before(start + delta);
            last_rank = 0;
        }
        start += delta;
        let duration = d_get(&mut m.duration, &mut d, MAX_DURATION as u64)? as usize;
        if start.saturating_mul(hop) > len + hop {
            return Err(corrupt());
        }
        let envelope = if duration > 0 && d.bit(&mut m.envelope[0])? {
            if d.bit(&mut m.envelope[1])? {
                Envelope::Exponential(unsigned(d_get(&mut m.ratio, &mut d, 1 << 40)?))
            } else {
                Envelope::Linear
            }
        } else {
            Envelope::Constant
        };
        let members = d_get(&mut m.count, &mut d, MAX_MEMBERS as u64 - 1)? as usize + 1;
        let mut group = Group {
            start,
            duration,
            envelope,
            members: Vec::with_capacity(members),
        };
        let mut below: Option<f64> = None;
        for _ in 0..members {
            let continued = d.bit(&mut m.continued)?;
            let end = if continued {
                let step = unsigned(d_get(&mut m.link, &mut d, 1 << 32)?);
                let rank = usize::try_from(last_rank as i64 + step).map_err(|_| corrupt())?;
                last_rank = rank;
                Some(open.take(start, rank).ok_or_else(corrupt)?)
            } else {
                None
            };
            let fade_out = d.bit(&mut m.fade_out[usize::from(continued)])?;
            let (class, omega_index, a0_index, phase_index) = match end {
                Some(end) => {
                    let change = unsigned(d_get(&mut m.class_change, &mut d, 64)?);
                    let class = i64::from(end.class) + change;
                    if !(0..=24).contains(&class) {
                        return Err(corrupt());
                    }
                    let class = class as u32;
                    let (omega, phase) = predicted(&end, class, duration, hop);
                    let omega_index = omega as i64
                        + unsigned(d_get(&mut m.omega_residual, &mut d, 2 * MAX_INDEX)?);
                    let a0 = end.amplitude_index as i64
                        + unsigned(d_get(&mut m.amplitude_residual, &mut d, 1 << 40)?);
                    let bp = phase_bits(class);
                    let p = (phase as i64
                        + unsigned(d_get(&mut m.phase_residual, &mut d, 1 << 40)?))
                    .rem_euclid(1 << bp);
                    if !(0..MAX_INDEX as i64).contains(&omega_index) || a0 < 0 {
                        return Err(corrupt());
                    }
                    (class, omega_index as u64, a0 as u64, p as u64)
                }
                None => {
                    let class = d_get(&mut m.class, &mut d, 24)? as u32;
                    let eo = omega_exponent(class, duration, hop);
                    let omega_index = match below {
                        Some(f) => {
                            let v = on_grid(f, eo) as i64
                                + unsigned(d_get(&mut m.omega_step, &mut d, 2 * MAX_INDEX)?);
                            if !(0..MAX_INDEX as i64).contains(&v) {
                                return Err(corrupt());
                            }
                            v as u64
                        }
                        None => d_get(&mut m.omega, &mut d, MAX_INDEX - 1)?,
                    };
                    below = Some(omega_index as f64 / (eo as f64).exp2());
                    let a0 = d_get(&mut m.amplitude, &mut d, 1 << 40)?;
                    let phase = d.direct(phase_bits(class))?;
                    (class, omega_index, a0, phase)
                }
            };
            let gradient_index = if duration > 0 {
                let g = unsigned(d_get(
                    &mut m.gradient[usize::from(continued)],
                    &mut d,
                    2 * MAX_INDEX,
                )?);
                if g.unsigned_abs() >= MAX_INDEX {
                    return Err(corrupt());
                }
                g
            } else {
                0
            };
            let a1_index = if envelope == Envelope::Linear {
                let a1 = a0_index as i64 + unsigned(d_get(&mut m.amplitude_end, &mut d, 1 << 41)?);
                u64::try_from(a1).map_err(|_| corrupt())?
            } else {
                a0_index
            };
            group.members.push(Member {
                class,
                fade_in: !continued,
                fade_out,
                omega_index,
                gradient_index,
                a0_index,
                a1_index,
                phase_index,
            });
        }
        for (mi, member) in group.members.iter().enumerate() {
            if !member.fade_out {
                open.insert(start + duration, open_end(member, &group, hop, (gi, mi)));
            }
        }
        groups.push(group);
    }
    Ok(Gradients {
        hop,
        precision,
        groups,
    })
}

/// An adaptive gamma value no larger than `limit`.
fn d_get(model: &mut Gamma, d: &mut RangeDecoder, limit: u64) -> Result<u64, Error> {
    let v = model.get(d)?;
    if v > limit {
        return Err(corrupt());
    }
    Ok(v)
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

#[cfg(test)]
mod tests {
    use super::*;

    type Links = Vec<Vec<Option<(usize, usize)>>>;

    /// Three tracks of four segments each, every continuation starting exactly
    /// where its predecessor ends (rounded to its own grid).
    fn tracks(hop: usize) -> (Vec<Group>, Links) {
        let mut groups: Vec<Group> = Vec::new();
        let mut links = Vec::new();
        for k in 0..4 {
            let mut members = Vec::new();
            let mut group_links = Vec::new();
            for (t, &f) in [0.05f64, 0.2, 0.7].iter().enumerate() {
                let class = 6 + t as u32;
                let duration = 3;
                let (omega_index, phase_index, a0_index, link) = if k == 0 {
                    let eo = omega_exponent(class, duration, hop);
                    ((f * (eo as f64).exp2()) as u64, 17 + t as u64, 90, None)
                } else {
                    let g = &groups[k - 1];
                    let key = (k - 1, t);
                    let end = open_end(&g.members[t], g, hop, key);
                    let (omega, phase) = predicted(&end, class, duration, hop);
                    (omega, phase, end.amplitude_index, Some(key))
                };
                members.push(Member {
                    class,
                    fade_in: k == 0,
                    fade_out: k == 3,
                    omega_index,
                    gradient_index: 5 - 3 * t as i64,
                    a0_index,
                    a1_index: a0_index,
                    phase_index,
                });
                group_links.push(link);
            }
            groups.push(Group {
                start: 3 * k,
                duration: 3,
                envelope: Envelope::Constant,
                members,
            });
            links.push(group_links);
        }
        (groups, links)
    }

    #[test]
    fn linked_members_round_trip_and_cost_less_than_new_ones() {
        let hop = 256;
        let (groups, links) = tracks(hop);
        let bytes = pack(hop, 2.0, &groups, &links).unwrap();
        let parsed = parse(&bytes, 1 << 16).unwrap();
        assert_eq!(parsed.groups, groups);
        // The same parameters with every member coded as a new track.
        let unlinked: Vec<Group> = groups
            .iter()
            .map(|g| Group {
                members: g
                    .members
                    .iter()
                    .map(|m| Member {
                        fade_in: true,
                        fade_out: true,
                        ..*m
                    })
                    .collect(),
                ..g.clone()
            })
            .collect();
        let none: Links = links.iter().map(|l| vec![None; l.len()]).collect();
        let fresh = pack(hop, 2.0, &unlinked, &none).unwrap();
        assert_eq!(parse(&fresh, 1 << 16).unwrap().groups, unlinked);
        assert!(
            bytes.len() + 12 < fresh.len(),
            "{} vs {}",
            bytes.len(),
            fresh.len()
        );
    }
}
