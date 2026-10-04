//! Band division, equalization and extrusion by fitted components
//! (PAPER_EQ_COMPRESSION.md, Sections 2 and 3).
//!
//! `ToneAnalysis` determines the circle curve of a signal and fits tones in Hann
//! windows of duration T advancing by T/2: the exact maxima of |FAFT| below an
//! upper frequency, then one two-way banded joint fit (`Faft::fit_banded`), so each
//! tone carries its own height with its neighbours' leakage removed. The tonal
//! track is the overlap-add of the windowed tones; the periodic Hann windows sum to
//! one, so a stationary tone is reproduced exactly.
//!
//! `divide` assigns each fitted tone to bands through crossover weights that sum
//! to one (brick wall or raised cosine): a tone goes to its band whole, with its
//! kernel. The residual (source minus tonal track) is divided by zero-phase
//! Butterworth low-passes applied exactly to its line curve (`analog`), taken as
//! complementary differences. The bands therefore add back to the source samples
//! exactly (to rounding), whatever the fit. Equalization is a weighted sum of the
//! bands; extrusion isolates one band and keeps the exact remainder.
//!
//! The divider is as good as the fit: content that the tones do not explain is
//! divided by the residual filters, which have transition bands. Tones above
//! `ToneParams::upper` are not fitted and stay in the residual.
use crate::analog::{AnalogFilter, SpiralCurve};
use crate::faft::{CircleCurve, PeakParams, Weighting};
use crate::geometric_faft::{GeometryFrame, GeometryParams, HierarchyParams};
use crate::spectral::Complex;
use crate::{buffer, Error};
use std::f64::consts::{PI, TAU};

/// Tone analysis settings. Frequencies are radians per source sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneParams {
    /// Window duration T in samples, even, 16..=65536; the hop is T/2.
    pub window: usize,
    /// Highest fitted frequency, in (0, pi].
    pub upper: f64,
    /// Peaks below this fraction of the strongest |FAFT| maximum are ignored,
    /// and fitted tones whose significance falls below the same level are pruned.
    pub relative_floor: f64,
    /// Most tones per window, 1..=4096.
    pub max_tones: usize,
    /// Reach of the banded joint fit, in resolutions 2*pi/T; infinite is dense.
    pub reach_bins: f64,
    /// Rounds of per-component Newton refinement of the frequencies after the
    /// joint fit (`Faft::fit_refined`), 0..=64; 0 keeps the |FAFT| maxima.
    pub refine: usize,
    /// Circle determination: largest error at each source sample, amplitude units.
    pub max_error: f64,
    /// Circle determination: largest support, 2..=256.
    pub max_span: usize,
}
impl Default for ToneParams {
    fn default() -> Self {
        Self {
            window: 2048,
            upper: PI,
            relative_floor: 0.01,
            max_tones: 256,
            reach_bins: 16.0,
            refine: 3,
            max_error: 4.0,
            max_span: 128,
        }
    }
}
impl ToneParams {
    fn validate(&self) -> Result<(), Error> {
        if !self.window.is_multiple_of(2)
            || !(16..=65536).contains(&self.window)
            || !(self.upper > 0.0 && self.upper <= PI)
            || !(self.relative_floor > 0.0 && self.relative_floor < 1.0)
            || !(1..=4096).contains(&self.max_tones)
            || self.reach_bins.is_nan()
            || self.reach_bins <= 0.0
            || self.refine > 64
            || !self.max_error.is_finite()
            || self.max_error < 0.0
            || !(2..=256).contains(&self.max_span)
        {
            return Err(Error::InvalidParameter("tone analysis parameters"));
        }
        Ok(())
    }
}

/// One fitted tone around its window centre: `amplitude*cos(omega*u + phase)`
/// with `u = n - centre` in samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tone {
    pub omega: f64,
    pub amplitude: f64,
    pub phase: f64,
}

/// The tones of one window. `centre` is in source samples (frame k is centred at
/// k*hop and covers `[centre - hop, centre + hop)`); energies are those of the
/// joint fit of the circle curve in this window (`SpectralFit`).
#[derive(Debug, Clone, PartialEq)]
pub struct ToneFrame {
    pub centre: usize,
    pub tones: Vec<Tone>,
    pub energy: f64,
    pub residual_energy: f64,
    pub bound: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ToneStats {
    pub frames: usize,
    pub tones: usize,
    pub rotations: usize,
    pub pruned: usize,
    /// Largest error of the determined circle curve at the (padded) source samples.
    pub curve_max_error: f64,
}

/// Tones fitted in every window of one mono signal. No source samples are kept.
#[derive(Debug, Clone, PartialEq)]
pub struct ToneAnalysis {
    len: usize,
    window: usize,
    frames: Vec<ToneFrame>,
    stats: ToneStats,
}
impl ToneAnalysis {
    /// Pad with T/2 zeros before the signal and at least T/2 after it, determine
    /// the circle curve, and fit tones in every Hann window. Every source sample
    /// is covered by exactly two windows.
    pub fn new(samples: &[f64], params: ToneParams) -> Result<Self, Error> {
        params.validate()?;
        if samples.is_empty() {
            return Err(Error::InvalidParameter("tone analysis samples"));
        }
        if let Some(i) = samples.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFiniteSample(i));
        }
        let hop = params.window / 2;
        let padded_len = (samples.len() + 2 * hop).div_ceil(hop) * hop;
        let mut padded = buffer(padded_len)?;
        padded.resize(hop, 0.0);
        padded.extend_from_slice(samples);
        padded.resize(padded_len, 0.0);
        let geometry = GeometryParams {
            max_error: params.max_error,
            max_span: params.max_span,
            ..GeometryParams::default()
        };
        let (frame, _) =
            GeometryFrame::hierarchical(&padded, geometry, HierarchyParams::default())?;
        let curve = CircleCurve::from_geometry(&frame)?;
        let mut stats = ToneStats {
            rotations: curve.rotations().len(),
            curve_max_error: frame.stats().max_error,
            ..ToneStats::default()
        };
        let count = padded_len / hop - 1;
        let mut frames = buffer(count)?;
        let resolution = TAU / params.window as f64;
        let search = PeakParams {
            upper: params.upper,
            relative_floor: params.relative_floor,
            max_peaks: params.max_tones,
            ..PeakParams::default()
        };
        for k in 0..count {
            let faft = curve.transform(k * hop..k * hop + params.window, Weighting::Hann)?;
            let peaks = faft.peaks(search)?.peaks;
            let strongest = peaks.iter().map(|p| p.magnitude()).fold(0.0, f64::max);
            let mut omegas: Vec<f64> = peaks.iter().map(|p| p.omega).collect();
            let reach = params.reach_bins * resolution;
            let mut fit = faft.fit_banded(&omegas, reach)?;
            // Sidelobes and duplicates explain little energy of their own.
            let floor = params.relative_floor * strongest;
            let before = omegas.len();
            omegas = fit
                .components
                .iter()
                .filter(|c| c.significance >= floor)
                .map(|c| c.omega)
                .collect();
            stats.pruned += before - omegas.len();
            if params.refine > 0 {
                fit = faft.fit_refined(&omegas, reach, params.refine, 1e-4 * resolution)?;
            } else if omegas.len() < before {
                fit = faft.fit_banded(&omegas, reach)?;
            }
            // Below half a resolution a tone is DC: its sine atom nearly vanishes
            // and its amplitude is ill-conditioned. Keep DC as a cosine at 0.
            if fit
                .components
                .iter()
                .any(|c| c.omega > 0.0 && c.omega < 0.5 * resolution)
            {
                let mut snapped: Vec<f64> = fit
                    .components
                    .iter()
                    .map(|c| {
                        if c.omega < 0.5 * resolution {
                            0.0
                        } else {
                            c.omega
                        }
                    })
                    .collect();
                snapped.dedup();
                fit = faft.fit_banded(&snapped, reach)?;
            }
            let half = 0.5 * params.window as f64;
            let tones: Vec<Tone> = fit
                .components
                .iter()
                .map(|c| {
                    // Component values are normalized c with phase origin at the
                    // window start; the tone is 2|c| cos(omega*u + arg c) around
                    // the centre.
                    let centred = rotate(c.value, c.omega * half);
                    Tone {
                        omega: c.omega,
                        amplitude: 2.0 * centred.re.hypot(centred.im),
                        phase: centred.im.atan2(centred.re),
                    }
                })
                .collect();
            stats.tones += tones.len();
            frames.push(ToneFrame {
                centre: k * hop,
                tones,
                energy: fit.energy,
                residual_energy: fit.residual_energy,
                bound: fit.bound,
            });
        }
        stats.frames = frames.len();
        Ok(Self {
            len: samples.len(),
            window: params.window,
            frames,
            stats,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn window(&self) -> usize {
        self.window
    }
    pub fn hop(&self) -> usize {
        self.window / 2
    }
    pub fn frames(&self) -> &[ToneFrame] {
        &self.frames
    }
    pub fn stats(&self) -> ToneStats {
        self.stats
    }

    /// Weighted fraction of the curve's energy the tones explain, over all frames.
    pub fn explained(&self) -> f64 {
        let energy: f64 = self.frames.iter().map(|f| f.energy).sum();
        let residual: f64 = self.frames.iter().map(|f| f.residual_energy).sum();
        if energy > 0.0 {
            1.0 - residual / energy
        } else {
            1.0
        }
    }

    /// The tonal track at the source samples, each tone scaled by `gain(omega)`:
    /// the overlap-add of `w_k * sum gain*tone` over windows (PAPER (7)).
    pub fn synthesize(&self, gain: impl Fn(f64) -> f64) -> Result<Vec<f64>, Error> {
        let hop = self.hop();
        let mut out = buffer(self.len)?;
        out.resize(self.len, 0.0);
        let shift = TAU / self.window as f64;
        for frame in &self.frames {
            let first = frame.centre.saturating_sub(hop);
            let last = (frame.centre + hop).min(self.len);
            if first >= last {
                continue;
            }
            for tone in &frame.tones {
                let g = gain(tone.omega);
                if g == 0.0 {
                    continue;
                }
                // Rotate a phasor sample by sample: A e^{i(omega*u + phase)}.
                let u0 = first as f64 - frame.centre as f64;
                let mut z = polar(g * tone.amplitude, tone.omega * u0 + tone.phase);
                let step = polar(1.0, tone.omega);
                for (n, slot) in out[first..last].iter_mut().enumerate() {
                    let u = u0 + n as f64;
                    let w = 0.5 + 0.5 * (shift * u).cos();
                    *slot += w * z.re;
                    z = mul(z, step);
                }
            }
        }
        if out.iter().all(|v| v.is_finite()) {
            Ok(out)
        } else {
            Err(Error::InvalidParameter("tonal synthesis overflow"))
        }
    }
}

fn polar(r: f64, angle: f64) -> Complex {
    let (s, c) = angle.sin_cos();
    Complex {
        re: r * c,
        im: r * s,
    }
}
fn mul(a: Complex, b: Complex) -> Complex {
    Complex {
        re: a.re * b.re - a.im * b.im,
        im: a.re * b.im + a.im * b.re,
    }
}
fn rotate(a: Complex, angle: f64) -> Complex {
    mul(a, polar(1.0, angle))
}

/// How a tone near an edge is shared between the bands on either side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Crossover {
    /// Each tone goes whole to the band that contains its frequency.
    BrickWall,
    /// A raised-cosine ramp of the given half-width (radians per sample) around
    /// each edge; the weights still sum to one.
    RaisedCosine { half_width: f64 },
}

/// Band edges (radians per sample, strictly increasing, in (0, pi]), the tone
/// crossover, and the order of the residual's zero-phase Butterworth filters,
/// whose combined response is `1/(1 + (omega/edge)^(2*order))`.
#[derive(Debug, Clone, PartialEq)]
pub struct BandSplit {
    edges: Vec<f64>,
    crossover: Crossover,
    residual_order: usize,
}
impl BandSplit {
    pub fn new(
        edges: Vec<f64>,
        crossover: Crossover,
        residual_order: usize,
    ) -> Result<Self, Error> {
        let increasing = edges.windows(2).all(|p| p[0] < p[1]);
        let in_range = edges.iter().all(|e| e.is_finite() && *e > 0.0 && *e <= PI);
        let ramp = match crossover {
            Crossover::BrickWall => true,
            Crossover::RaisedCosine { half_width } => half_width.is_finite() && half_width > 0.0,
        };
        if edges.is_empty() || edges.len() > 64 || !increasing || !in_range || !ramp {
            return Err(Error::InvalidParameter("band edges"));
        }
        if !(1..=12).contains(&residual_order) {
            return Err(Error::InvalidParameter("residual filter order"));
        }
        Ok(Self {
            edges,
            crossover,
            residual_order,
        })
    }
    pub fn edges(&self) -> &[f64] {
        &self.edges
    }
    pub fn bands(&self) -> usize {
        self.edges.len() + 1
    }

    /// Fraction of a tone at `omega` that lies below `edge`.
    fn below(&self, edge: f64, omega: f64) -> f64 {
        match self.crossover {
            Crossover::BrickWall => {
                if omega < edge {
                    1.0
                } else {
                    0.0
                }
            }
            Crossover::RaisedCosine { half_width } => {
                let x = (omega - edge + half_width) / (2.0 * half_width);
                if x <= 0.0 {
                    1.0
                } else if x >= 1.0 {
                    0.0
                } else {
                    0.5 * (1.0 + (PI * x).cos())
                }
            }
        }
    }

    /// Crossover weight of `band` (0-based) for a tone at `omega`. The weights are
    /// nonnegative and sum to one over the bands (PAPER (3)).
    pub fn weight(&self, band: usize, omega: f64) -> f64 {
        let upper = self.edges.get(band).map_or(1.0, |&e| self.below(e, omega));
        let lower = match band {
            0 => 0.0,
            b => self.below(self.edges[b - 1], omega),
        };
        (upper - lower).max(0.0)
    }
}

/// The tracks of a division: band b is `tonal[b] + residual[b]`, at the source
/// samples. They sum to the source exactly up to rounding.
#[derive(Debug, Clone, PartialEq)]
pub struct Division {
    pub tonal: Vec<Vec<f64>>,
    pub residual: Vec<Vec<f64>>,
}
impl Division {
    pub fn bands(&self) -> usize {
        self.tonal.len()
    }
    /// Band `b` (tonal plus residual part).
    pub fn band(&self, b: usize) -> Vec<f64> {
        self.tonal[b]
            .iter()
            .zip(&self.residual[b])
            .map(|(t, r)| t + r)
            .collect()
    }
}

/// Divide `samples` (the signal `analysis` was computed from) into the bands of
/// `split`: tones by crossover weight, the residual by complementary zero-phase
/// Butterworth low-passes applied exactly to its line curve. The last band takes
/// the exact remainder, so the bands sum to `samples`.
pub fn divide(
    samples: &[f64],
    analysis: &ToneAnalysis,
    split: &BandSplit,
) -> Result<Division, Error> {
    if samples.len() != analysis.len() {
        return Err(Error::LengthMismatch);
    }
    let bands = split.bands();
    let mut tonal = Vec::with_capacity(bands);
    for b in 0..bands {
        tonal.push(analysis.synthesize(|omega| split.weight(b, omega))?);
    }
    // Residual of the whole tonal track (sum of the band weights is one).
    let mut residual_signal = samples.to_vec();
    for t in &tonal {
        for (r, v) in residual_signal.iter_mut().zip(t) {
            *r -= v;
        }
    }
    let curve = SpiralCurve::from_samples(&residual_signal)?;
    let mut lows = Vec::with_capacity(split.edges.len());
    for &edge in &split.edges {
        let filter = AnalogFilter::butterworth_lowpass(split.residual_order, edge)?;
        lows.push(curve.filter_zero_phase(&filter)?.samples()?);
    }
    let mut residual = Vec::with_capacity(bands);
    for b in 0..bands {
        let upper = lows.get(b).unwrap_or(&residual_signal);
        let part: Vec<f64> = match b {
            0 => upper.clone(),
            _ => upper.iter().zip(&lows[b - 1]).map(|(u, l)| u - l).collect(),
        };
        residual.push(part);
    }
    Ok(Division { tonal, residual })
}

/// Graphic equalizer: `sum_b gains[b] * band_b`. Unit gains return the source.
pub fn equalize(division: &Division, gains: &[f64]) -> Result<Vec<f64>, Error> {
    if gains.len() != division.bands() || gains.iter().any(|g| !g.is_finite()) {
        return Err(Error::InvalidParameter("equalizer gains"));
    }
    let len = division.tonal.first().map_or(0, Vec::len);
    let mut out = vec![0.0; len];
    for (b, &g) in gains.iter().enumerate() {
        for ((o, t), r) in out
            .iter_mut()
            .zip(&division.tonal[b])
            .zip(&division.residual[b])
        {
            *o += g * (t + r);
        }
    }
    Ok(out)
}

/// Isolate the band `[lower, upper)` (radians per sample; `lower` may be 0 and
/// `upper` may be pi for an open band): returns `(isolated, remainder)` with
/// `isolated + remainder == samples` exactly.
pub fn extrude(
    samples: &[f64],
    analysis: &ToneAnalysis,
    lower: f64,
    upper: f64,
    crossover: Crossover,
    residual_order: usize,
) -> Result<(Vec<f64>, Vec<f64>), Error> {
    if !(lower >= 0.0 && lower < upper && upper <= PI) {
        return Err(Error::InvalidParameter("extrusion band"));
    }
    let mut edges = Vec::new();
    if lower > 0.0 {
        edges.push(lower);
    }
    if upper < PI {
        edges.push(upper);
    }
    if edges.is_empty() {
        return Ok((samples.to_vec(), vec![0.0; samples.len()]));
    }
    let band = usize::from(lower > 0.0);
    let division = divide(
        samples,
        analysis,
        &BandSplit::new(edges, crossover, residual_order)?,
    )?;
    let isolated = division.band(band);
    let remainder = samples.iter().zip(&isolated).map(|(x, i)| x - i).collect();
    Ok((isolated, remainder))
}
