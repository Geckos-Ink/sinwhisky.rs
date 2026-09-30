//! Geometry-first FAFT experiment. Fits circumcircles from waveform points before
//! any spectral calculation. Finite geometric sums query the predicted samples;
//! no FFT or numerical integration is used by fitting, decode, or spectrum.
use crate::geometry::{arc_angles, Circle, Point};
use crate::spectral::Complex;
use crate::{buffer, Error};
use std::f64::consts::{PI, TAU};
use std::ops::Range;

fn add(a: Complex, b: Complex) -> Complex {
    Complex {
        re: a.re + b.re,
        im: a.im + b.im,
    }
}
fn sub(a: Complex, b: Complex) -> Complex {
    Complex {
        re: a.re - b.re,
        im: a.im - b.im,
    }
}
fn mul(a: Complex, b: Complex) -> Complex {
    Complex {
        re: a.re * b.re - a.im * b.im,
        im: a.re * b.im + a.im * b.re,
    }
}
fn scale(a: Complex, b: f64) -> Complex {
    Complex {
        re: a.re * b,
        im: a.im * b,
    }
}
fn phase(a: f64) -> Complex {
    let (s, c) = a.sin_cos();
    Complex { re: c, im: s }
}
fn checked(a: Complex) -> Result<Complex, Error> {
    if a.re.is_finite() && a.im.is_finite() {
        Ok(a)
    } else {
        Err(Error::InvalidParameter("geometric spectrum overflow"))
    }
}

/// Relative significand bits for each finite geometry field. This is precision,
/// not bitrate. 53 keeps f64 unchanged. Rounded fields are validated on ALL samples.
fn rounded(value: f64, bits: u8) -> f64 {
    if bits == 53 || value == 0.0 {
        return value;
    }
    let exponent =
        (value.abs().log2().floor() as i32 - (bits as i32 - 1)).max(-1022 - (bits as i32 - 1));
    let step = if exponent < -1022 {
        // powi can underflow via an overflowing reciprocal on some targets.
        f64::from_bits(1_u64 << (exponent + 1074))
    } else {
        2.0_f64.powi(exponent)
    };
    (value / step).round() * step
}

#[derive(Debug, Clone, Copy)]
pub struct GeometryParams {
    /// Maximum absolute error at each original sample, in input amplitude units.
    pub max_error: f64,
    /// Maximum dyadic source interval per candidate, 2..=256.
    pub max_span: usize,
    /// Try quarter/half/three-quarter fitting points and aperture gains .5/1/2.
    /// false tries the midpoint and gain 1 only, useful as a controlled baseline.
    pub search_phases: bool,
    /// Search increasing per-field significand widths from this value to 53.
    pub min_precision: u8,
    /// Reject a nominal geometry that already fails source checks before searching
    /// quantized versions. Faster, but omits chance improvements from quantization.
    pub prevalidate_geometry: bool,
}
impl Default for GeometryParams {
    fn default() -> Self {
        Self {
            max_error: 0.001,
            max_span: 32,
            search_phases: true,
            min_precision: 12,
            prevalidate_geometry: true,
        }
    }
}

/// A circumcircle through three source points BEFORE field quantization. Coordinates
/// are local: x=(source_position-start)*x_scale; y is normalized by frame.scale().
/// Angular interpolation passes through the three points on two equal-angle halves.
#[derive(Debug, Clone, Copy)]
pub struct WaveCircle {
    pub center: Point,
    pub radius: f64,
    pub x_scale: f64,
    pub angles: [f64; 3],
    pub middle: usize,
    pub precision: u8,
    pub aperture_gain: f64,
}
#[derive(Debug, Clone, Copy)]
pub enum WavePrimitive {
    Circle(WaveCircle),
    /// Explicit zero-curvature limit: finite endpoints, no infinite radius stored.
    Line {
        first: f64,
        last: f64,
        precision: u8,
    },
}
#[derive(Debug, Clone)]
pub struct WaveSegment {
    range: Range<usize>,
    primitive: WavePrimitive,
}
impl WaveSegment {
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
    pub fn primitive(&self) -> WavePrimitive {
        self.primitive
    }
    fn value(&self, t: f64) -> f64 {
        match self.primitive {
            WavePrimitive::Line { first, last, .. } => {
                first + (last - first) * t / self.range.len() as f64
            }
            WavePrimitive::Circle(c) => {
                let (origin, length, theta, end) = if t < c.middle as f64 {
                    (0.0, c.middle as f64, c.angles[0], c.angles[1])
                } else {
                    (
                        c.middle as f64,
                        (self.range.len() - c.middle) as f64,
                        c.angles[1],
                        c.angles[2],
                    )
                };
                let beta = (end - theta) / length;
                let delta = beta * (t - origin);
                // Anchored sine difference avoids large-radius center cancellation
                // on every prediction. Reject ill-conditioned anchor values during fit.
                let (s, cos) = theta.sin_cos();
                let anchor = c.center.y + c.radius * s;
                anchor + c.radius * (-2.0 * s * (delta * 0.5).sin().powi(2) + cos * delta.sin())
            }
        }
    }
    fn model_bits(&self) -> usize {
        // Hypothetical cost only. Geometry fields carry an 11-bit exponent each.
        match self.primitive {
            WavePrimitive::Line { precision, .. } => 1 + 8 + 6 + 2 * (11 + precision as usize),
            WavePrimitive::Circle(c) => 1 + 8 + 8 + 2 + 6 + 7 * (11 + c.precision as usize),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GeometryStats {
    pub circles: usize,
    pub lines: usize,
    pub angular_pieces: usize,
    pub candidates_tested: usize,
    pub max_error: f64,
    pub rmse: f64,
    pub estimated_model_bits: usize,
}

/// Immutable geometric predictor. No source samples, FFT coefficients, or borrowed
/// Analysis are retained. Segments cover [0,N); the terminal fitting anchor is the
/// first source sample, explicitly defining a periodic closing interval.
#[derive(Debug, Clone)]
pub struct GeometryFrame {
    len: usize,
    scale: f64,
    segments: Vec<WaveSegment>,
    pieces: Vec<Piece>,
    stats: GeometryStats,
}
impl GeometryFrame {
    pub fn new(samples: &[f64], params: GeometryParams) -> Result<Self, Error> {
        let n = samples.len();
        if n == 0
            || !(2..=256).contains(&params.max_span)
            || !params.max_span.is_power_of_two()
            || !(4..=53).contains(&params.min_precision)
            || !params.max_error.is_finite()
            || params.max_error < 0.0
        {
            return Err(Error::InvalidParameter("geometric FAFT parameters"));
        }
        if let Some(i) = samples.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFiniteSample(i));
        }
        let amplitude = samples
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()))
            .max(f64::MIN_POSITIVE);
        let tolerance = params.max_error / amplitude;
        let size = n.checked_add(1).ok_or(Error::OutputTooLarge)?;
        let mut source = buffer(size)?;
        source.extend(samples.iter().map(|v| v / amplitude));
        source.push(source[0]);
        let mut widths = buffer(10)?;
        for b in [params.min_precision, 12, 16, 20, 24, 32, 40, 53] {
            if b >= params.min_precision && !widths.contains(&b) {
                widths.push(b);
            }
        }
        widths.sort_unstable();
        let mut costs = buffer(n + 1)?;
        costs.resize(n + 1, (usize::MAX, usize::MAX));
        costs[n] = (0, 0);
        let mut choices = buffer(n)?;
        choices.resize_with(n, || None);
        let mut tested = 0;
        for start in (0..n).rev() {
            let mut length = 1;
            while length <= params.max_span && length <= n - start {
                let end = start + length;
                let mut best: Option<WaveSegment> = None;
                for &bits in &widths {
                    let line = WaveSegment {
                        range: start..end,
                        primitive: WavePrimitive::Line {
                            first: rounded(source[start], bits),
                            last: rounded(source[end], bits),
                            precision: bits,
                        },
                    };
                    tested += 1;
                    if valid(&line, &source[start..=end], tolerance) {
                        best = Some(line);
                        break;
                    }
                }
                let cheapest_circle_bits = 25 + 7 * (11 + params.min_precision as usize);
                if length >= 2
                    && best
                        .as_ref()
                        .is_none_or(|b| b.model_bits() > cheapest_circle_bits)
                {
                    let mids = if params.search_phases {
                        [length / 4, length / 2, 3 * length / 4]
                    } else {
                        [length / 2; 3]
                    };
                    let gains = if params.search_phases {
                        [0.5, 1.0, 2.0]
                    } else {
                        [1.0; 3]
                    };
                    for (mi, &middle) in mids.iter().enumerate() {
                        if middle == 0 || middle >= length || mids[..mi].contains(&middle) {
                            continue;
                        }
                        for (gi, &gain) in gains.iter().enumerate() {
                            if gains[..gi].contains(&gain) {
                                continue;
                            }
                            let y = [source[start], source[start + middle], source[end]];
                            let range = (y[0] - y[1])
                                .abs()
                                .max((y[1] - y[2]).abs())
                                .max((y[0] - y[2]).abs());
                            let x_scale = 2.0 * range.max(1e-12) * gain / length as f64;
                            let points = [
                                Point { x: 0.0, y: y[0] },
                                Point {
                                    x: middle as f64 * x_scale,
                                    y: y[1],
                                },
                                Point {
                                    x: length as f64 * x_scale,
                                    y: y[2],
                                },
                            ];
                            let Some(circle) = Circle::through(points, 1e-18) else {
                                continue;
                            };
                            // Extremely flat circles are better represented by a line.
                            if circle.radius > 1e6 {
                                continue;
                            }
                            let angles = arc_angles(circle, points);
                            if params.prevalidate_geometry {
                                let nominal = WaveSegment {
                                    range: start..end,
                                    primitive: WavePrimitive::Circle(WaveCircle {
                                        center: circle.center,
                                        radius: circle.radius,
                                        x_scale,
                                        angles,
                                        middle,
                                        precision: 53,
                                        aperture_gain: gain,
                                    }),
                                };
                                tested += 1;
                                if !valid(&nominal, &source[start..=end], tolerance) {
                                    continue;
                                }
                            }
                            for &bits in &widths {
                                let c = WaveCircle {
                                    center: Point {
                                        x: rounded(circle.center.x, bits),
                                        y: rounded(circle.center.y, bits),
                                    },
                                    radius: rounded(circle.radius, bits),
                                    x_scale: rounded(x_scale, bits),
                                    angles: angles.map(|a| rounded(a, bits)),
                                    middle,
                                    precision: bits,
                                    aperture_gain: gain,
                                };
                                let segment = WaveSegment {
                                    range: start..end,
                                    primitive: WavePrimitive::Circle(c),
                                };
                                tested += 1;
                                if valid(&segment, &source[start..=end], tolerance) {
                                    if best
                                        .as_ref()
                                        .is_none_or(|b| segment.model_bits() < b.model_bits())
                                    {
                                        best = Some(segment);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
                if let Some(segment) = best {
                    let next = (
                        costs[end].0.saturating_add(1),
                        costs[end].1.saturating_add(segment.model_bits()),
                    );
                    if next < costs[start] {
                        costs[start] = next;
                        choices[start] = Some(segment);
                    }
                }
                length *= 2;
            }
        }
        let mut segments = buffer(n)?;
        let mut at = 0;
        while at < n {
            let segment = choices[at].take().ok_or(Error::InvalidParameter(
                "geometric precision cannot meet error",
            ))?;
            at = segment.range.end;
            segments.push(segment);
        }
        let mut pieces = buffer(segments.len().checked_mul(2).ok_or(Error::OutputTooLarge)?)?;
        let mut stats = GeometryStats {
            candidates_tested: tested,
            estimated_model_bits: 96 + costs[0].1,
            ..GeometryStats::default()
        };
        for segment in &segments {
            match segment.primitive {
                WavePrimitive::Line { first, last, .. } => {
                    stats.lines += 1;
                    pieces.push(Piece {
                        start: segment.range.start,
                        len: segment.range.len(),
                        kind: PieceKind::Line {
                            first,
                            slope: (last - first) / segment.range.len() as f64,
                        },
                    });
                }
                WavePrimitive::Circle(c) => {
                    stats.circles += 1;
                    for side in 0..2 {
                        let offset = if side == 0 { 0 } else { c.middle };
                        let len = if side == 0 {
                            c.middle
                        } else {
                            segment.range.len() - c.middle
                        };
                        let (s, cos) = c.angles[side].sin_cos();
                        pieces.push(Piece {
                            start: segment.range.start + offset,
                            len,
                            kind: PieceKind::Arc {
                                first: c.center.y + c.radius * s,
                                sine: c.radius * s,
                                cosine: c.radius * cos,
                                beta: (c.angles[side + 1] - c.angles[side]) / len as f64,
                            },
                        });
                    }
                }
            }
        }
        stats.angular_pieces = pieces.len();
        let mut frame = Self {
            len: n,
            scale: amplitude,
            segments,
            pieces,
            stats,
        };
        let decoded = frame.decode()?;
        let mut rms = 0.0_f64;
        let mut max = 0.0_f64;
        for (&a, &b) in samples.iter().zip(&decoded) {
            let e = (a - b).abs();
            max = max.max(e);
            rms = rms.hypot(e / (n as f64).sqrt());
        }
        frame.stats.max_error = max;
        frame.stats.rmse = rms;
        if !frame.stats.max_error.is_finite()
            || !frame.stats.rmse.is_finite()
            || frame.stats.max_error > params.max_error
        {
            return Err(Error::InvalidParameter(
                "geometric rescaled error exceeds budget",
            ));
        }
        Ok(frame)
    }
    pub fn sample_count(&self) -> usize {
        self.len
    }
    pub fn scale(&self) -> f64 {
        self.scale
    }
    pub fn segments(&self) -> &[WaveSegment] {
        &self.segments
    }
    pub fn stats(&self) -> GeometryStats {
        self.stats
    }
    /// Periodic equal-angle prediction. Error bound applies only to original samples.
    pub fn evaluate(&self, position: f64) -> Result<f64, Error> {
        if !position.is_finite() {
            return Err(Error::InvalidParameter("geometric position"));
        }
        let t = position.rem_euclid(self.len as f64);
        let index = self.segments.partition_point(|s| s.range.end as f64 <= t);
        let segment = &self.segments[index];
        let value = segment.value(t - segment.range.start as f64) * self.scale;
        if !value.is_finite() {
            return Err(Error::InvalidParameter("geometric prediction overflow"));
        }
        Ok(value)
    }
    pub fn decode(&self) -> Result<Vec<f64>, Error> {
        let mut output = buffer(self.len)?;
        for piece in &self.pieces {
            for j in 0..piece.len {
                output.push(piece.value(j as f64) * self.scale);
            }
        }
        if output.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidParameter("geometric decode overflow"));
        }
        Ok(output)
    }
    /// Finite sampled transform over a cropped time window, with negative exponential
    /// and phase origin at window.start. O(pieces), no integration or PCM allocation.
    pub fn dtft(&self, omega: f64, window: Range<usize>) -> Result<Complex, Error> {
        if !omega.is_finite() || window.start >= window.end || window.end > self.len {
            return Err(Error::InvalidParameter("geometric spectral window"));
        }
        let mut sum = Complex::default();
        let mut correction = Complex::default();
        for piece in &self.pieces {
            let start = piece.start.max(window.start);
            let end = (piece.start + piece.len).min(window.end);
            if start >= end {
                continue;
            }
            let local = piece.shifted(start - piece.start, end - start);
            let value = mul(
                local.sum(omega),
                phase(-omega * (start - window.start) as f64),
            );
            compensated(&mut sum, &mut correction, value);
        }
        checked(scale(sum, self.scale))
    }
    /// Exact periodic-Hann weighting of a cropped predictor, using three shifted
    /// finite sums. Geometry is reused; windowing does not trigger another fit.
    pub fn hann_dtft(&self, omega: f64, window: Range<usize>) -> Result<Complex, Error> {
        if window.start >= window.end || window.end > self.len {
            return Err(Error::InvalidParameter("geometric Hann window"));
        }
        let shift = TAU / window.len() as f64;
        let center = self.dtft(omega, window.clone())?;
        let lower = self.dtft(omega - shift, window.clone())?;
        let upper = self.dtft(omega + shift, window)?;
        checked(sub(scale(center, 0.5), scale(add(lower, upper), 0.25)))
    }
    /// Stable closed sums at standard nonnegative bins. zoom samples each original
    /// interval at 1/zoom spacing, then divides the DFT by zoom to retain units.
    /// Higher zoom is model sampling density, not additional measured information.
    pub fn spectrum_closed(&self, zoom: usize) -> Result<Vec<Complex>, Error> {
        self.validate_zoom(zoom)?;
        let mut result = buffer(self.len / 2 + 1)?;
        for k in 0..=self.len / 2 {
            let omega = TAU * k as f64 / (self.len * zoom) as f64;
            let mut sum = Complex::default();
            let mut correction = Complex::default();
            for piece in &self.pieces {
                let p = piece.zoomed(zoom);
                let value = mul(p.sum(omega), phase(-omega * p.start as f64));
                compensated(&mut sum, &mut correction, value);
            }
            result.push(checked(scale(sum, self.scale / zoom as f64))?);
        }
        Ok(result)
    }
    /// Batch rotations: precompute grid roots once, then update geometric-series
    /// ratios and support phases by multiplication. O(pieces*N), no hidden FFT.
    pub fn spectrum_rotations(&self, zoom: usize) -> Result<Vec<Complex>, Error> {
        self.validate_zoom(zoom)?;
        let n = self.len * zoom;
        let bins = self.len / 2 + 1;
        let mut out = buffer(bins)?;
        out.resize(bins, Complex::default());
        let mut correction = buffer(bins)?;
        correction.resize(bins, Complex::default());
        for piece in &self.pieces {
            let p = piece.zoomed(zoom);
            if p.len == 1 {
                let root = phase(-TAU * p.start as f64 / n as f64);
                let mut support = phase(0.0);
                for k in 0..bins {
                    if k % 64 == 0 {
                        support = phase(-TAU * p.start as f64 * k as f64 / n as f64);
                    }
                    compensated(
                        &mut out[k],
                        &mut correction[k],
                        scale(support, p.value(0.0)),
                    );
                    support = mul(support, root);
                }
                continue;
            }
            let root = phase(-TAU / n as f64);
            let root_len = phase(-TAU * p.len as f64 / n as f64);
            let root_start = phase(-TAU * p.start as f64 / n as f64);
            let beta = match p.kind {
                PieceKind::Arc { beta, .. } => beta,
                _ => 0.0,
            };
            let offsets = [0.0, beta, -beta];
            let mut q = offsets.map(phase);
            let mut power = offsets.map(|a| phase(a * p.len as f64));
            let mut support = phase(0.0);
            for k in 0..bins {
                let omega = TAU * k as f64 / n as f64;
                // Reanchor every 64 bins to bound multiplicative drift.
                if k % 64 == 0 {
                    q = offsets.map(|a| phase(a - omega));
                    power = offsets.map(|a| phase((a - omega) * p.len as f64));
                    support = phase(-omega * p.start as f64);
                }
                let g = std::array::from_fn::<_, 3, _>(|side| {
                    let den = Complex {
                        re: 1.0 - q[side].re,
                        im: -q[side].im,
                    };
                    let norm = den.re * den.re + den.im * den.im;
                    if norm < 1e-8 {
                        geometric_sum(offsets[side] - omega, p.len)
                    } else {
                        let num = Complex {
                            re: 1.0 - power[side].re,
                            im: -power[side].im,
                        };
                        Complex {
                            re: (num.re * den.re + num.im * den.im) / norm,
                            im: (num.im * den.re - num.re * den.im) / norm,
                        }
                    }
                });
                let value = match p.kind {
                    PieceKind::Line { first, slope } => {
                        let den = Complex {
                            re: 1.0 - q[0].re,
                            im: -q[0].im,
                        };
                        let norm = den.re * den.re + den.im * den.im;
                        let moment = if norm < 1e-6 || omega * (p.len as f64) < 0.02 {
                            first_moment(-omega, p.len)
                        } else {
                            let numerator = sub(mul(q[0], g[0]), scale(power[0], p.len as f64));
                            Complex {
                                re: (numerator.re * den.re + numerator.im * den.im) / norm,
                                im: (numerator.im * den.re - numerator.re * den.im) / norm,
                            }
                        };
                        add(scale(g[0], first), scale(moment, slope))
                    }
                    PieceKind::Arc {
                        first,
                        sine,
                        cosine,
                        ..
                    } => arc_sum(first, sine, cosine, g),
                };
                compensated(&mut out[k], &mut correction[k], mul(value, support));
                for side in 0..3 {
                    q[side] = mul(q[side], root);
                    power[side] = mul(power[side], root_len);
                }
                support = mul(support, root_start);
            }
        }
        for z in &mut out {
            *z = checked(scale(*z, self.scale / zoom as f64))?;
        }
        Ok(out)
    }
    fn validate_zoom(&self, zoom: usize) -> Result<(), Error> {
        if !(1..=256).contains(&zoom) {
            return Err(Error::InvalidParameter("geometric zoom"));
        }
        self.len.checked_mul(zoom).ok_or(Error::OutputTooLarge)?;
        Ok(())
    }
}

fn valid(segment: &WaveSegment, source: &[f64], tolerance: f64) -> bool {
    if let WavePrimitive::Circle(c) = segment.primitive {
        if ![0, c.middle, segment.range.len()]
            .iter()
            .zip(c.angles)
            .all(|(&index, angle)| {
                let x = c.center.x + c.radius * angle.cos();
                (x - index as f64 * c.x_scale).abs() <= tolerance * 0.999
            })
        {
            return false;
        }
    }

    match segment.primitive {
        WavePrimitive::Line { first, last, .. } => {
            let slope = (last - first) / segment.range.len() as f64;
            source
                .iter()
                .enumerate()
                .all(|(j, &v)| (first + slope * j as f64 - v).abs() <= tolerance * 0.999)
        }
        WavePrimitive::Circle(c) => {
            let halves = [0, 1].map(|side| {
                let (s, cos) = c.angles[side].sin_cos();
                let origin = if side == 0 { 0 } else { c.middle };
                let length = if side == 0 {
                    c.middle
                } else {
                    segment.range.len() - c.middle
                };
                (
                    origin,
                    Piece {
                        start: 0,
                        len: length,
                        kind: PieceKind::Arc {
                            first: c.center.y + c.radius * s,
                            sine: c.radius * s,
                            cosine: c.radius * cos,
                            beta: (c.angles[side + 1] - c.angles[side]) / length as f64,
                        },
                    },
                )
            });
            source.iter().enumerate().all(|(j, &v)| {
                let (origin, piece) = halves[usize::from(j >= c.middle)];
                let y = piece.value((j - origin) as f64);
                y.is_finite() && (y - v).abs() <= tolerance * 0.999
            })
        }
    }
}
fn compensated(total: &mut Complex, correction: &mut Complex, value: Complex) {
    let y = sub(value, *correction);
    let next = add(*total, y);
    *correction = sub(sub(next, *total), y);
    *total = next;
}
#[derive(Debug, Clone, Copy)]
enum PieceKind {
    Line {
        first: f64,
        slope: f64,
    },
    Arc {
        first: f64,
        sine: f64,
        cosine: f64,
        beta: f64,
    },
}
#[derive(Debug, Clone, Copy)]
struct Piece {
    start: usize,
    len: usize,
    kind: PieceKind,
}
impl Piece {
    fn value(self, t: f64) -> f64 {
        match self.kind {
            PieceKind::Line { first, slope } => first + slope * t,
            PieceKind::Arc {
                first,
                sine,
                cosine,
                beta,
            } => first - 2.0 * sine * (beta * t * 0.5).sin().powi(2) + cosine * (beta * t).sin(),
        }
    }
    fn shifted(self, offset: usize, len: usize) -> Self {
        let kind = match self.kind {
            PieceKind::Line { first, slope } => PieceKind::Line {
                first: first + slope * offset as f64,
                slope,
            },
            PieceKind::Arc {
                first: _,
                sine,
                cosine,
                beta,
            } => {
                let (s, c) = (beta * offset as f64).sin_cos();
                PieceKind::Arc {
                    first: self.value(offset as f64),
                    sine: sine * c + cosine * s,
                    cosine: cosine * c - sine * s,
                    beta,
                }
            }
        };
        Self {
            start: self.start + offset,
            len,
            kind,
        }
    }
    fn zoomed(self, q: usize) -> Self {
        let kind = match self.kind {
            PieceKind::Line { first, slope } => PieceKind::Line {
                first,
                slope: slope / q as f64,
            },
            PieceKind::Arc {
                first,
                sine,
                cosine,
                beta,
            } => PieceKind::Arc {
                first,
                sine,
                cosine,
                beta: beta / q as f64,
            },
        };
        Self {
            start: self.start * q,
            len: self.len * q,
            kind,
        }
    }
    fn sum(self, omega: f64) -> Complex {
        if self.len == 1 {
            return Complex {
                re: self.value(0.0),
                im: 0.0,
            };
        }
        let g0 = geometric_sum(-omega, self.len);
        match self.kind {
            PieceKind::Line { first, slope } => add(
                scale(g0, first),
                scale(first_moment(-omega, self.len), slope),
            ),
            PieceKind::Arc {
                first,
                sine,
                cosine,
                beta,
            } => arc_sum(
                first,
                sine,
                cosine,
                [
                    g0,
                    geometric_sum(beta - omega, self.len),
                    geometric_sum(-beta - omega, self.len),
                ],
            ),
        }
    }
}
fn arc_sum(first: f64, sine: f64, cosine: f64, g: [Complex; 3]) -> Complex {
    let even = sub(scale(add(g[1], g[2]), 0.5), g[0]);
    let odd = sub(g[1], g[2]);
    add(
        add(scale(g[0], first), scale(even, sine)),
        Complex {
            re: odd.im * cosine * 0.5,
            im: -odd.re * cosine * 0.5,
        },
    )
}
fn wrapped(delta: f64) -> f64 {
    (delta + PI).rem_euclid(TAU) - PI
}
fn dirichlet(delta: f64, n: usize) -> (f64, f64) {
    let m = n as f64;
    let x = wrapped(delta);
    if (x * m).abs() < 0.02 {
        let m2 = m * m;
        let a = (m2 - 1.0) / 24.0;
        let b = (3.0 * m2 * m2 - 10.0 * m2 + 7.0) / 5760.0;
        (
            m * (1.0 - a * x * x + b * x.powi(4)),
            m * (-2.0 * a * x + 4.0 * b * x.powi(3)),
        )
    } else {
        let (s, c) = (x * 0.5).sin_cos();
        let (sm, cm) = (m * x * 0.5).sin_cos();
        (sm / s, 0.5 * (m * cm * s - sm * c) / (s * s))
    }
}
/// sum exp(i*delta*j), j=0..n-1. Stable DC/resonance limit; no sample loop.
pub fn geometric_sum(delta: f64, n: usize) -> Complex {
    if n == 0 {
        return Complex::default();
    }
    let x = wrapped(delta);
    let (d, _) = dirichlet(x, n);
    scale(phase(x * (n - 1) as f64 * 0.5), d)
}
fn first_moment(delta: f64, n: usize) -> Complex {
    if n == 0 {
        return Complex::default();
    }
    let x = wrapped(delta);
    let (d, derivative) = dirichlet(x, n);
    mul(
        phase(x * (n - 1) as f64 * 0.5),
        Complex {
            re: d * (n - 1) as f64 * 0.5,
            im: -derivative,
        },
    )
}
