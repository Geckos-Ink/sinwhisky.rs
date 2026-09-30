//! FAFT experiment: fast rotations, projected circles, and a dyadic model search.
//!
//! The transform kernel is an ordinary radix-2 FFT, not a new Fourier algorithm.
//! Optimization chooses disjoint parent OR child blocks, not additive residual
//! children. Error is measured at source samples; no intersample bound is claimed.
use crate::spectral::Complex;
use crate::{buffer, Error};
use std::f64::consts::TAU;
use std::ops::Range;

fn finite(z: Complex) -> bool {
    z.re.is_finite() && z.im.is_finite()
}

/// Unnormalized forward DFT, or 1/N-normalized inverse, using radix-2 rotations.
/// Power-of-two lengths (including one) only. O(N log N), no integration.
/// Allocations and butterfly results are checked; the input is not modified.
pub fn transform(input: &[Complex], inverse: bool) -> Result<Vec<Complex>, Error> {
    let n = input.len();
    if !n.is_power_of_two() {
        return Err(Error::InvalidParameter("FAFT power-of-two length"));
    }
    if let Some(i) = input.iter().position(|&z| !finite(z)) {
        return Err(Error::NonFiniteSample(i));
    }
    let mut out = buffer(n)?;
    out.extend_from_slice(input);
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            out.swap(i, j);
        }
    }
    let mut width = 2;
    while width <= n {
        let half = width / 2;
        // Direct sin/cos avoids long twiddle recurrences and their accumulated drift.
        for k in 0..half {
            let (s, c) = ((if inverse { TAU } else { -TAU }) * k as f64 / width as f64).sin_cos();
            for base in (0..n).step_by(width) {
                let a = out[base + k];
                let z = out[base + k + half];
                let b = Complex {
                    re: z.re * c - z.im * s,
                    im: z.re * s + z.im * c,
                };
                out[base + k] = Complex {
                    re: a.re + b.re,
                    im: a.im + b.im,
                };
                out[base + k + half] = Complex {
                    re: a.re - b.re,
                    im: a.im - b.im,
                };
            }
        }
        if width == n {
            break;
        }
        width *= 2;
    }
    if inverse {
        for z in &mut out {
            z.re /= n as f64;
            z.im /= n as f64;
        }
    }
    if out.iter().any(|&z| !finite(z)) {
        return Err(Error::InvalidParameter("FAFT arithmetic overflow"));
    }
    Ok(out)
}

/// A projection coefficient: either signed mantissas with a shared binary
/// exponent for this circle, or unquantized f64 values for very tight budgets.
/// `bits` is precision per Cartesian component, not an encoded bitrate.
#[derive(Debug, Clone, Copy)]
pub enum Precision {
    Quantized {
        bits: u8,
        exponent: i16,
        re: i32,
        im: i32,
    },
    Float64(Complex),
}
impl Precision {
    pub fn bits(self) -> u8 {
        match self {
            Self::Quantized { bits, .. } => bits,
            Self::Float64(_) => 64,
        }
    }
    pub fn coefficient(self) -> Complex {
        match self {
            Self::Float64(z) => z,
            Self::Quantized {
                exponent, re, im, ..
            } => {
                let step = 2.0_f64.powi(exponent as i32);
                Complex {
                    re: re as f64 * step,
                    im: im as f64 * step,
                }
            }
        }
    }
}

/// `Re(c * exp(i*TAU*bin*t/N))`, with local t. Its radius is |c| and
/// phase is atan2(c.im,c.re). DC/Nyquist have real coefficients only.
#[derive(Debug, Clone, Copy)]
pub struct RotatingCircle {
    pub bin: usize,
    pub precision: Precision,
}
impl RotatingCircle {
    pub fn project(self, local_position: f64, period: usize) -> f64 {
        let z = self.precision.coefficient();
        let (s, c) = (TAU * self.bin as f64 * local_position / period as f64).sin_cos();
        z.re * c - z.im * s
    }
}

/// Stable zero-curvature limit of a geometric circle, not an oscillator at DC.
/// Endpoints avoid infinity-minus-infinity and reproduce both anchors exactly.
#[derive(Debug, Clone, Copy)]
pub struct LineLimit {
    pub first: f64,
    pub last: f64,
}
impl LineLimit {
    pub fn evaluate(self, fraction: f64) -> f64 {
        (1.0 - fraction) * self.first + fraction * self.last
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Baseline {
    Constant(f64),
    Line(LineLimit),
}
impl Baseline {
    fn value(self, t: f64, n: usize) -> f64 {
        match self {
            Self::Constant(v) => v,
            Self::Line(line) => line.evaluate(t / (n - 1) as f64),
        }
    }
    fn bits(self) -> usize {
        if matches!(self, Self::Constant(_)) {
            64
        } else {
            128
        }
    }
}

/// One half-open support. Values are normalized by the enclosing frame's scale.
#[derive(Debug, Clone)]
pub struct CircleBlock {
    range: Range<usize>,
    baseline: Baseline,
    circles: Vec<RotatingCircle>,
    rmse: f64,
}
impl CircleBlock {
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
    pub fn baseline(&self) -> Baseline {
        self.baseline
    }
    pub fn circles(&self) -> &[RotatingCircle] {
        &self.circles
    }
    pub fn normalized_rmse(&self) -> f64 {
        self.rmse
    }
    fn value(&self, t: f64) -> f64 {
        self.baseline.value(t, self.range.len())
            + self
                .circles
                .iter()
                .map(|c| c.project(t, self.range.len()))
                .sum::<f64>()
    }
    fn decode(&self) -> Result<Vec<f64>, Error> {
        let n = self.range.len();
        let mut bins = buffer(n)?;
        bins.resize(n, Complex::default());
        for circle in &self.circles {
            let mut z = circle.precision.coefficient();
            let edge = circle.bin == 0 || circle.bin == n / 2;
            let scale = n as f64 * if edge { 1.0 } else { 0.5 };
            z.re *= scale;
            z.im *= scale;
            bins[circle.bin] = z;
            if !edge {
                bins[n - circle.bin] = Complex {
                    re: z.re,
                    im: -z.im,
                };
            }
        }
        let values = transform(&bins, true)?;
        let mut out = buffer(n)?;
        out.extend(
            values
                .iter()
                .enumerate()
                .map(|(i, z)| z.re + self.baseline.value(i as f64, n)),
        );
        Ok(out)
    }
    fn model_bits(&self) -> usize {
        // Explicit hypothetical schema: leaf flag + baseline tag/value + count,
        // then local bin + precision tag (7 bits) + exponent (16) + mantissas.
        // This is a cost model, NOT serialized bytes or a compression ratio.
        let index_bits = self.range.len().ilog2() as usize;
        1 + 1
            + self.baseline.bits()
            + 32
            + self
                .circles
                .iter()
                .map(|c| {
                    let components = if c.bin == 0 || c.bin == self.range.len() / 2 {
                        1
                    } else {
                        2
                    };
                    index_bits
                        + 7
                        + if c.precision.bits() == 64 {
                            components * 64
                        } else {
                            16 + components * c.precision.bits() as usize
                        }
                })
                .sum::<usize>()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FaftParams {
    /// Absolute RMS error in the input amplitude units; finite and positive.
    pub max_rmse: f64,
    /// Smallest child length; power of two, at least two. Set N for no splits.
    pub min_block: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct FaftStats {
    pub blocks: usize,
    pub circles: usize,
    pub descriptors: usize,
    pub line_blocks: usize,
    pub estimated_model_bits: usize,
    pub rmse: f64,
    pub max_error: f64,
}

/// Source-independent, approximate reconstruction from selected rotating circles.
/// Tree selection minimizes (baseline+circle count, estimated bits) lexicographically
/// among generated candidates with the SAME per-leaf RMS limit. This is not the
/// global fewest-circle solution over arbitrary frequencies, trends, or budgets.
#[derive(Debug, Clone)]
pub struct FaftFrame {
    len: usize,
    scale: f64,
    blocks: Vec<CircleBlock>,
    stats: FaftStats,
}
impl FaftFrame {
    pub fn new(samples: &[f64], params: FaftParams) -> Result<Self, Error> {
        let n = samples.len();
        if n < 2
            || !n.is_power_of_two()
            || params.min_block < 2
            || !params.min_block.is_power_of_two()
            || params.min_block > n
            || !params.max_rmse.is_finite()
            || params.max_rmse <= 0.0
        {
            return Err(Error::InvalidParameter("FAFT frame parameters"));
        }
        if let Some(i) = samples.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFiniteSample(i));
        }
        let scale = samples
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()))
            .max(f64::MIN_POSITIVE);
        let mut normalized = buffer(n)?;
        normalized.extend(samples.iter().map(|v| v / scale));
        let tolerance = (params.max_rmse / scale).min(4.0);
        if tolerance == 0.0 {
            return Err(Error::InvalidParameter("FAFT unresolvable tolerance"));
        }
        let selected = search(&normalized, 0, params.min_block, tolerance)?;
        let mut result = Self {
            len: n,
            scale,
            blocks: selected.blocks,
            stats: FaftStats {
                blocks: 0,
                circles: 0,
                descriptors: selected.descriptors,
                line_blocks: 0,
                estimated_model_bits: 96 + selected.bits,
                rmse: 0.0,
                max_error: 0.0,
            },
        };
        let decoded = result.decode()?;
        // Scaled sum avoids squared PCM overflow. Validate the ACTUAL decoded result.
        let mut rms = 0.0_f64;
        let mut maximum = 0.0_f64;
        for (&a, &b) in samples.iter().zip(&decoded) {
            let error = (a / scale - b / scale).abs();
            rms = rms.hypot(error / (n as f64).sqrt());
            maximum = maximum.max(error);
        }
        result.stats.rmse = rms * scale;
        result.stats.max_error = maximum * scale;
        if !result.stats.rmse.is_finite()
            || !result.stats.max_error.is_finite()
            || result.stats.rmse > params.max_rmse
        {
            return Err(Error::InvalidParameter(
                "FAFT precision cannot meet RMS budget",
            ));
        }
        result.stats.blocks = result.blocks.len();
        result.stats.circles = result.blocks.iter().map(|b| b.circles.len()).sum();
        result.stats.line_blocks = result
            .blocks
            .iter()
            .filter(|b| matches!(b.baseline, Baseline::Line(_)))
            .count();
        Ok(result)
    }
    pub fn sample_count(&self) -> usize {
        self.len
    }
    pub fn scale(&self) -> f64 {
        self.scale
    }
    pub fn blocks(&self) -> &[CircleBlock] {
        &self.blocks
    }
    pub fn stats(&self) -> FaftStats {
        self.stats
    }
    /// O(sum L log L) inverse rotations, with no source PCM or numerical integration.
    pub fn decode(&self) -> Result<Vec<f64>, Error> {
        let mut output = buffer(self.len)?;
        for block in &self.blocks {
            output.extend(block.decode()?.iter().map(|v| v * self.scale));
        }
        if output.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidParameter("FAFT decode overflow"));
        }
        Ok(output)
    }
    /// Direct projected-circle function on [0,N). Joins may be discontinuous.
    /// This is O(K) for the selected block; decode() is preferred for all samples.
    pub fn evaluate(&self, position: f64) -> Result<f64, Error> {
        if !position.is_finite() || position < 0.0 || position >= self.len as f64 {
            return Err(Error::InvalidParameter("FAFT position"));
        }
        let index = self
            .blocks
            .partition_point(|b| b.range.end as f64 <= position);
        let block = &self.blocks[index];
        let v = block.value(position - block.range.start as f64) * self.scale;
        if !v.is_finite() {
            return Err(Error::InvalidParameter("FAFT evaluation overflow"));
        }
        Ok(v)
    }
}

struct Choice {
    blocks: Vec<CircleBlock>,
    descriptors: usize,
    bits: usize,
}
impl Choice {
    fn key(&self) -> (usize, usize) {
        (self.descriptors, self.bits)
    }
}
fn search(samples: &[f64], start: usize, minimum: usize, tolerance: f64) -> Result<Choice, Error> {
    let n = samples.len();
    let baselines = [
        Baseline::Constant(samples.iter().sum::<f64>() / n as f64),
        Baseline::Line(LineLimit {
            first: samples[0],
            last: samples[n - 1],
        }),
    ];
    let mut best: Option<Choice> = None;
    for baseline in baselines {
        if let Some(block) = fit(samples, start, baseline, tolerance)? {
            let choice = Choice {
                descriptors: block.circles.len() + 1,
                bits: block.model_bits(),
                blocks: vec![block],
            };
            if best.as_ref().is_none_or(|b| choice.key() < b.key()) {
                best = Some(choice);
            }
        }
    }
    // At least one baseline descriptor per child: splitting cannot improve this.
    if best.as_ref().is_some_and(|b| b.descriptors == 1) {
        return Ok(best.unwrap());
    }
    if n > minimum {
        let left = search(&samples[..n / 2], start, minimum, tolerance);
        let right = search(&samples[n / 2..], start + n / 2, minimum, tolerance);
        match (left, right) {
            (Ok(mut a), Ok(b)) => {
                a.descriptors += b.descriptors;
                a.bits += b.bits + 1;
                a.blocks.extend(b.blocks);
                if best.as_ref().is_none_or(|old| a.key() < old.key()) {
                    best = Some(a);
                }
            }
            (Err(Error::OutputTooLarge), _) | (_, Err(Error::OutputTooLarge)) => {
                return Err(Error::OutputTooLarge)
            }
            _ => {}
        }
    }
    best.ok_or(Error::InvalidParameter(
        "FAFT precision cannot meet leaf budget",
    ))
}

fn weight(bin: usize, n: usize) -> f64 {
    if bin == 0 || bin == n / 2 {
        1.0
    } else {
        0.5
    }
}
fn energy(z: Complex) -> f64 {
    z.re * z.re + z.im * z.im
}
fn quantize(z: Complex, allowance: f64, w: f64) -> Precision {
    let peak = z.re.abs().max(z.im.abs());
    if peak > 0.0 {
        for bits in 2..=30 {
            // Keep symmetric signed mantissas representable even when rounding
            // crosses the largest positive integer at this precision.
            let mut exponent = peak.log2().floor() as i32 - (bits as i32 - 2);
            let maximum = ((1_u32 << (bits - 1)) - 1) as f64;
            if (peak / 2.0_f64.powi(exponent)).round() > maximum {
                exponent += 1;
            }
            if !(-1022..=1023).contains(&exponent) {
                continue;
            }
            let step = 2.0_f64.powi(exponent);
            let re = (z.re / step).round() as i32;
            let im = (z.im / step).round() as i32;
            let p = Precision::Quantized {
                bits,
                exponent: exponent as i16,
                re,
                im,
            };
            let q = p.coefficient();
            if w * ((q.re - z.re).powi(2) + (q.im - z.im).powi(2)) <= allowance {
                return p;
            }
        }
    }
    Precision::Float64(z)
}
fn fit(
    samples: &[f64],
    start: usize,
    baseline: Baseline,
    tolerance: f64,
) -> Result<Option<CircleBlock>, Error> {
    let n = samples.len();
    let mut residual = buffer(n)?;
    residual.extend(samples.iter().enumerate().map(|(i, v)| Complex {
        re: v - baseline.value(i as f64, n),
        im: 0.0,
    }));
    let spectrum = transform(&residual, false)?;
    let mut candidates = buffer(n / 2 + 1)?;
    for (bin, z) in spectrum.iter().enumerate().take(n / 2 + 1) {
        let edge = bin == 0 || bin == n / 2;
        let scale = (if edge { 1.0 } else { 2.0 }) / n as f64;
        let c = Complex {
            re: z.re * scale,
            im: if edge { 0.0 } else { z.im * scale },
        };
        candidates.push((bin, c, weight(bin, n) * energy(c)));
    }
    candidates.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
    // Suffix sums avoid catastrophic subtraction when removing a strong circle.
    let mut tail = buffer(candidates.len() + 1)?;
    tail.resize(candidates.len() + 1, 0.0);
    for i in (0..candidates.len()).rev() {
        tail[i] = tail[i + 1] + candidates[i].2;
    }
    let budget = (tolerance * 0.98).powi(2);
    let k = tail
        .iter()
        .position(|&e| e <= budget)
        .unwrap_or(candidates.len());
    let allowance = (budget - tail[k]).max(0.0) / k.max(1) as f64;
    // At most two inverse transforms per candidate, preserving bounded search cost.
    for quantized in [true, false] {
        let mut circles = buffer(k)?;
        for &(bin, z, _) in &candidates[..k] {
            circles.push(RotatingCircle {
                bin,
                precision: if quantized {
                    quantize(z, allowance, weight(bin, n))
                } else {
                    Precision::Float64(z)
                },
            });
        }
        let mut block = CircleBlock {
            range: start..start + n,
            baseline,
            circles,
            rmse: 0.0,
        };
        let decoded = block.decode()?;
        block.rmse = samples.iter().zip(decoded).fold(0.0_f64, |sum, (a, b)| {
            sum.hypot((a - b) / (n as f64).sqrt())
        });
        if block.rmse <= tolerance {
            return Ok(Some(block));
        }
    }
    // Roundoff can defeat Parseval at extremely tight budgets. Reject the
    // candidate rather than relaxing the requested error or doing quadratic retries.
    Ok(None)
}
