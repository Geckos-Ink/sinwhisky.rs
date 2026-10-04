//! FAFT: Fast Angular Fourier Transform of a circle-determined waveform.
//!
//! Three stages are kept separate:
//!
//! 1. **Circle determination** fits circles through source points (for example
//!    `GeometryFrame` or the triplet `AngularFrame`). Each equal-angle half is a
//!    finite rotation `y(u) = center + radius*sin(phase + rate*u)` on a support.
//! 2. **Zoom** (SinWhisky synthesis) samples that curve at fractional positions.
//!    It produces samples, not a transform, and is never used here.
//! 3. **FAFT** integrates the same curve against `exp(-i*omega*t)` in closed form,
//!    for any real `omega` in radians per source sample. There is no frequency
//!    grid tied to a sample count, no sampling, no FFT and no quadrature.
//!
//! The fast rule is "rotate and add": each rotation's transform is multiplied by
//! its support phase and added to the previous result, as in a single-frequency
//! (Goertzel-like) recurrence, not an FFT factorization. Support phases are chained
//! by complex multiplication. Each arc is evaluated in endpoint form: its deviation
//! from the start value obeys f'' = -rate^2*(f + a), so its transform needs only
//! the end values, end slopes and rate^2*a, prepared once, and one real division per
//! query. Near omega = ±rate, where that form is 0/0, the moment form with the
//! prepared `exp(i*rate*length)` and series is used. A query needs one sin/cos pair
//! and costs O(rotations); a dense spectrum is not faster than an FFT of the source
//! samples, which computes a different (sampled) target.
//!
//! ```
//! use sinwhisky::faft::{CircleCurve, PeakParams, Weighting};
//! use sinwhisky::geometric_faft::{GeometryFrame, GeometryParams, HierarchyParams};
//! let samples: Vec<f64> = (0..512).map(|i| 1000.0 * (0.3137 * i as f64).sin()).collect();
//! let params = GeometryParams { max_error: 0.5, max_span: 64, ..GeometryParams::default() };
//! let (frame, _) = GeometryFrame::hierarchical(&samples, params, HierarchyParams::default())?;
//! let faft = CircleCurve::from_geometry(&frame)?.transform(0..512, Weighting::Hann)?;
//! let off_grid = faft.value(0.31)?; // any real omega, also above pi; no zoom
//! let peaks = faft.peaks(PeakParams::default())?.peaks;
//! let strongest = peaks.iter().max_by(|a, b| a.magnitude().total_cmp(&b.magnitude())).unwrap();
//! assert!((strongest.omega - 0.3137).abs() < 1e-3 && off_grid.re.is_finite());
//! # Ok::<(), sinwhisky::Error>(())
//! ```
use crate::aft::{AngularFrame, AngularHalf};
use crate::geometric_faft::{GeometryFrame, WavePrimitive};
use crate::spectral::Complex;
use crate::{buffer, Error};
use std::f64::consts::{PI, TAU};
use std::ops::Range;

const ONE: Complex = Complex { re: 1.0, im: 0.0 };
fn c(re: f64, im: f64) -> Complex {
    Complex { re, im }
}
fn add(a: Complex, b: Complex) -> Complex {
    c(a.re + b.re, a.im + b.im)
}
fn sub(a: Complex, b: Complex) -> Complex {
    c(a.re - b.re, a.im - b.im)
}
fn mul(a: Complex, b: Complex) -> Complex {
    c(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}
fn scale(a: Complex, s: f64) -> Complex {
    c(a.re * s, a.im * s)
}
fn conj(a: Complex) -> Complex {
    c(a.re, -a.im)
}
/// Multiply by -i.
fn minus_i(a: Complex) -> Complex {
    c(a.im, -a.re)
}
fn norm(a: Complex) -> f64 {
    a.re.hypot(a.im)
}
fn phase(angle: f64) -> Complex {
    let (s, cos) = angle.sin_cos();
    c(cos, s)
}
fn power(mut base: Complex, mut exponent: usize) -> Complex {
    let mut out = ONE;
    while exponent > 0 {
        if exponent & 1 == 1 {
            out = mul(out, base);
        }
        exponent >>= 1;
        if exponent > 0 {
            base = mul(base, base);
        }
    }
    out
}
fn checked(a: Complex) -> Result<Complex, Error> {
    if a.re.is_finite() && a.im.is_finite() {
        Ok(a)
    } else {
        Err(Error::InvalidParameter("FAFT arithmetic overflow"))
    }
}

/// One finite rotation of a circle-determined curve, in amplitude units:
/// `y(u) = center + radius*sin(phase + rate*u) + slope*u` for `0 <= u < length`,
/// with `u` measured from `start` in source samples. A line is `radius = 0`, the
/// infinite-radius limit kept explicit. This is the circle itself (center height,
/// radius, starting angle, angular rate), not an FFT coefficient. For a fitted
/// circle, `rate*length` is the side's sweep, twice an inscribed angle of the three
/// fitting points, so `rate` is geometric and need not be a physical frequency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rotation {
    pub start: usize,
    pub length: usize,
    pub center: f64,
    pub radius: f64,
    pub phase: f64,
    pub rate: f64,
    pub slope: f64,
}
impl Rotation {
    /// Anchored form: the value at u=0 plus the arc's deviation, as GeometryFrame
    /// evaluates it, so flat large-radius arcs do not subtract center and radius.
    fn value(&self, u: f64) -> f64 {
        let (s, cos) = self.phase.sin_cos();
        let delta = self.rate * u;
        self.center
            + self.radius * s
            + self.radius * (-2.0 * s * (delta * 0.5).sin().powi(2) + cos * delta.sin())
            + self.slope * u
    }
}

/// A waveform defined by contiguous (optionally stacked) rotations on `[0, len)`.
/// Stacked rotations share a support and add, as in the AFT's averaged halves.
/// The curve is what zoom samples and what FAFT integrates. No source PCM is kept.
#[derive(Debug, Clone)]
pub struct CircleCurve {
    rotations: Vec<Rotation>,
    len: usize,
}
impl CircleCurve {
    /// Validate finite fields and contiguous coverage of `[0, len)`: each rotation
    /// either continues at the previous support's end or repeats its exact support.
    pub fn new(rotations: Vec<Rotation>) -> Result<Self, Error> {
        let mut end = 0;
        let mut previous: Option<Range<usize>> = None;
        for r in &rotations {
            if r.length == 0
                || ![r.center, r.radius, r.phase, r.rate, r.slope]
                    .iter()
                    .all(|v| v.is_finite())
            {
                return Err(Error::InvalidParameter("FAFT rotation"));
            }
            let support = r.start..r.start.checked_add(r.length).ok_or(Error::OutputTooLarge)?;
            let stacked = previous.as_ref() == Some(&support);
            if !stacked && r.start != end {
                return Err(Error::InvalidParameter("FAFT rotation coverage"));
            }
            end = support.end;
            previous = Some(support);
        }
        if end == 0 {
            return Err(Error::InvalidParameter("FAFT empty curve"));
        }
        Ok(Self {
            rotations,
            len: end,
        })
    }

    /// Both equal-angle halves of every retained circle, and every finite line,
    /// exactly as `GeometryFrame` predicts them, rescaled to input amplitude units.
    pub fn from_geometry(frame: &GeometryFrame) -> Result<Self, Error> {
        let s = frame.scale();
        let mut rotations = buffer(frame.segments().len().saturating_mul(2))?;
        for segment in frame.segments() {
            let range = segment.range();
            match segment.primitive() {
                WavePrimitive::Line { first, last, .. } => rotations.push(Rotation {
                    start: range.start,
                    length: range.len(),
                    center: first * s,
                    radius: 0.0,
                    phase: 0.0,
                    rate: 0.0,
                    slope: (last - first) / range.len() as f64 * s,
                }),
                WavePrimitive::Circle(circle) => {
                    for side in 0..2 {
                        let (offset, length) = if side == 0 {
                            (0, circle.middle)
                        } else {
                            (circle.middle, range.len() - circle.middle)
                        };
                        rotations.push(Rotation {
                            start: range.start + offset,
                            length,
                            center: circle.center.y * s,
                            radius: circle.radius * s,
                            phase: circle.angles[side],
                            rate: (circle.angles[side + 1] - circle.angles[side]) / length as f64,
                            slope: 0.0,
                        });
                    }
                }
            }
        }
        Self::new(rotations)
    }

    /// The integral AFT reference model: two averaged halves per unit interval,
    /// stacked with weight 1/2. Used to check FAFT against `AngularFrame::fourier`.
    pub fn from_angular(frame: &AngularFrame) -> Result<Self, Error> {
        let mut rotations = buffer(frame.sample_count().saturating_mul(2))?;
        for (start, halves) in frame.halves().enumerate() {
            for half in halves {
                // AFT halves use u in [-1/2, 1/2]; FAFT supports start at u=0.
                rotations.push(match half {
                    AngularHalf::Arc {
                        center,
                        radius,
                        theta,
                        beta,
                    } => Rotation {
                        start,
                        length: 1,
                        center: center * 0.5,
                        radius: radius * 0.5,
                        phase: theta - beta * 0.5,
                        rate: beta,
                        slope: 0.0,
                    },
                    AngularHalf::Line { middle, slope } => Rotation {
                        start,
                        length: 1,
                        center: (middle - slope * 0.5) * 0.5,
                        radius: 0.0,
                        phase: 0.0,
                        rate: 0.0,
                        slope: slope * 0.5,
                    },
                });
            }
        }
        Self::new(rotations)
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn rotations(&self) -> &[Rotation] {
        &self.rotations
    }

    /// The curve at a real position in `[0, len)`. Zoom samples this function;
    /// it is an interpolant, not additional information about the source.
    pub fn evaluate(&self, position: f64) -> Result<f64, Error> {
        if !position.is_finite() || position < 0.0 || position >= self.len as f64 {
            return Err(Error::InvalidParameter("FAFT curve position"));
        }
        let first = self
            .rotations
            .partition_point(|r| ((r.start + r.length) as f64) <= position);
        let mut value = 0.0;
        for r in &self.rotations[first..] {
            if r.start as f64 > position {
                break;
            }
            value += r.value(position - r.start as f64);
        }
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::InvalidParameter("FAFT curve overflow"))
        }
    }

    /// Prepare the continuous transform of `window` (source samples, half-open).
    /// Cropping and `exp(i*rate*length)` are computed once, independent of omega.
    pub fn transform(&self, window: Range<usize>, weighting: Weighting) -> Result<Faft, Error> {
        if window.start >= window.end || window.end > self.len {
            return Err(Error::InvalidParameter("FAFT window"));
        }
        let mut atoms: Vec<Atom> = buffer(self.rotations.len())?;
        for r in &self.rotations {
            let start = r.start.max(window.start);
            let end = (r.start + r.length).min(window.end);
            if start >= end {
                continue;
            }
            let offset = (start - r.start) as f64;
            let length = end - start;
            // Anchored form y = value + a*(cos(rate*u)-1) + b*sin(rate*u) + slope*u,
            // so the centre height of a flat arc is never formed again.
            let (s, cos) = (r.phase + r.rate * offset).sin_cos();
            let (a, b) = (r.radius * s, r.radius * cos);
            // Endpoint data of the arc deviation f = a*(cos-1) + b*sin, which obeys
            // f'' = -rate^2*(f + a): f(0)=0, f'(0), f(L), f'(L) and rate^2*a.
            // cos(turn)-1 = -2 sin^2(turn/2) keeps flat arcs relatively accurate.
            let sweep = r.rate * length as f64;
            let (sin_sweep, cos_sweep) = sweep.sin_cos();
            let cos_minus_one = -2.0 * (0.5 * sweep).sin().powi(2);
            atoms.push(Atom {
                start: start - window.start,
                length,
                length_index: 0,
                value: r.value(offset),
                a,
                b,
                slope: r.slope,
                rate: r.rate,
                turn: c(cos_sweep, sin_sweep),
                start_slope: r.rate * b,
                end_slope: r.rate * (b * (1.0 + cos_minus_one) - a * sin_sweep),
                end_change: a * cos_minus_one + b * sin_sweep,
                curvature: r.rate * r.rate * a,
                arc: r.radius != 0.0 && r.rate != 0.0,
                advances: true,
            });
        }
        for i in 1..atoms.len() {
            if atoms[i].start == atoms[i - 1].start {
                atoms[i - 1].advances = false;
            }
        }
        let mut lengths: Vec<usize> = atoms.iter().map(|a| a.length).collect();
        lengths.sort_unstable();
        lengths.dedup();
        for atom in &mut atoms {
            atom.length_index = lengths.binary_search(&atom.length).unwrap_or(0);
            if ![
                atom.value,
                atom.a,
                atom.b,
                atom.turn.re,
                atom.turn.im,
                atom.start_slope,
                atom.end_slope,
                atom.end_change,
                atom.curvature,
            ]
            .iter()
            .all(|v| v.is_finite())
            {
                return Err(Error::InvalidParameter("FAFT prepared overflow"));
            }
        }
        let duration = window.len();
        let result = Faft {
            atoms,
            lengths,
            window,
            weighting,
            weight_sum: match weighting {
                Weighting::Rectangular => duration as f64,
                Weighting::Hann => duration as f64 * 0.5,
            },
        };
        Ok(result)
    }
}

/// Continuous-time weighting of the transform window. `Hann` is the periodic Hann
/// of the continuous window, applied analytically: no refit or sampling occurs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weighting {
    Rectangular,
    Hann,
}

#[derive(Debug, Clone, Copy)]
struct Atom {
    start: usize,
    length: usize,
    length_index: usize,
    value: f64,
    a: f64,
    b: f64,
    slope: f64,
    rate: f64,
    /// exp(i*rate*length): frequency independent, prepared once.
    turn: Complex,
    /// Endpoint data of the arc's deviation from `value`, prepared once:
    /// slopes at both ends, change over the support, and rate^2*a.
    start_slope: f64,
    end_slope: f64,
    end_change: f64,
    curvature: f64,
    arc: bool,
    /// False while the next rotation is stacked on the same support.
    advances: bool,
}

/// Series thresholds on |x| = |delta*length|. Below them, E_k uses Horner series
/// in x^2 with constant coefficients; above, the closed ratio forms lose at most
/// ~eps/|x|^(k+1) to cancellation: <1e-13 for values, <1e-11 for derivatives.
const VALUE_SERIES: f64 = 0.05;
const DERIVATIVE_SERIES: f64 = 0.5;
/// Arcs use the endpoint form unless |(omega -+ rate)*length| is below this guard.
/// Near the removable singularity at omega = ±rate the endpoint numerator cancels
/// (relative error ~eps/|x|, and ~6*eps/x^2 for its curvature term as x -> 0), so the
/// moment form with its series takes over there.
const ENDPOINT_GUARD: f64 = 0.5;
const TERMS: usize = 18;

/// 1/(n! * (n+k+1)): coefficients of E_k(delta)/L^(k+1) in powers of i*x.
const fn series_coefficients() -> [[f64; TERMS]; 4] {
    let mut out = [[0.0; TERMS]; 4];
    let mut k = 0;
    while k < 4 {
        let mut n = 0;
        let mut factorial = 1.0;
        while n < TERMS {
            if n > 0 {
                factorial *= n as f64;
            }
            out[k][n] = 1.0 / (factorial * (n + k + 1) as f64);
            n += 1;
        }
        k += 1;
    }
    out
}
const COEFFICIENTS: [[f64; TERMS]; 4] = series_coefficients();

/// `E_k(delta) / L^(k+1)` for small x: real part from even powers of i*x,
/// imaginary part from odd powers, each as a polynomial in -x^2.
fn series(k: usize, x: f64, terms: usize) -> Complex {
    let coefficients = &COEFFICIENTS[k];
    let y = -x * x;
    let (mut even, mut odd) = (0.0, 0.0);
    let mut m = terms / 2;
    while m > 0 {
        m -= 1;
        even = even * y + coefficients[2 * m];
        odd = odd * y + coefficients[2 * m + 1];
    }
    c(even, odd * x)
}

/// E_0(delta) = integral_0^L exp(i*delta*u) du, given e = exp(i*delta*L), 1/delta.
#[inline]
fn e0(delta: f64, inverse: f64, length: f64, e: Complex) -> Complex {
    let x = delta * length;
    if x.abs() < VALUE_SERIES {
        scale(series(0, x, 8), length)
    } else {
        // (e - 1)/(i*delta)
        c(e.im * inverse, (1.0 - e.re) * inverse)
    }
}

/// E_1(delta) = integral_0^L u*exp(i*delta*u) du, from E_0; for finite lines.
#[inline]
fn e1(delta: f64, inverse: f64, length: f64, e: Complex, e_0: Complex) -> Complex {
    let x = delta * length;
    if x.abs() < VALUE_SERIES {
        scale(series(1, x, 8), length * length)
    } else {
        scale(minus_i(sub(scale(e, length), e_0)), inverse)
    }
}

/// `integral_0^L u^k exp(i*delta*u) du` for k in 0..K, given `e = exp(i*delta*L)`.
/// Series near DC/resonance, closed recurrences elsewhere. No trigonometry.
fn moments<const K: usize>(delta: f64, length: f64, e: Complex) -> [Complex; K] {
    let x = delta * length;
    let mut out = [Complex::default(); K];
    if x.abs() < DERIVATIVE_SERIES {
        // |x|^n/n! < 1e-18 for n >= terms at the stated bounds.
        let terms = if x.abs() < 0.05 {
            10
        } else if x.abs() < 0.2 {
            14
        } else {
            TERMS
        };
        let mut power = length;
        for (k, slot) in out.iter_mut().enumerate() {
            *slot = scale(series(k, x, terms), power);
            power *= length;
        }
    } else {
        let inverse = 1.0 / delta;
        let mut previous = c(e.im * inverse, (1.0 - e.re) * inverse);
        out[0] = previous;
        let mut power = 1.0;
        for (k, slot) in out.iter_mut().enumerate().skip(1) {
            power *= length;
            previous = scale(
                minus_i(sub(scale(e, power), scale(previous, k as f64))),
                inverse,
            );
            *slot = previous;
        }
    }
    out
}

/// Prepared FAFT of one window. Values are normalized by the window's weight sum
/// (T for rectangular, T/2 for Hann), so a unit real sinusoid has magnitude ~0.5.
/// Phase origin is the window start unless stated otherwise.
#[derive(Debug, Clone)]
pub struct Faft {
    atoms: Vec<Atom>,
    lengths: Vec<usize>,
    window: Range<usize>,
    weighting: Weighting,
    weight_sum: f64,
}

impl Faft {
    pub fn window(&self) -> Range<usize> {
        self.window.clone()
    }
    pub fn weighting(&self) -> Weighting {
        self.weighting
    }
    /// Retained rotation pieces inside the window after cropping.
    pub fn rotation_count(&self) -> usize {
        self.atoms.len()
    }
    /// Frequency spacing whose inverse is the window duration: 2*pi/T. This is a
    /// resolution scale, not an output grid.
    pub fn resolution(&self) -> f64 {
        TAU / self.window.len() as f64
    }

    fn lengths_table(&self, q: Complex, table: &mut Vec<Complex>) {
        table.clear();
        let mut at = 0;
        let mut acc = ONE;
        for &length in &self.lengths {
            acc = mul(acc, power(q, length - at));
            at = length;
            table.push(acc);
        }
    }

    /// Unnormalized rectangular integral with phase origin at the window start.
    /// Rotate and add: each rotation's closed transform is multiplied by its chained
    /// support phase. Phases are reanchored with sin/cos every 64 advances.
    fn rect(&self, omega: f64, table: &mut Vec<Complex>) -> Complex {
        let q = phase(-omega);
        self.lengths_table(q, table);
        let inverse = if omega != 0.0 { -1.0 / omega } else { 0.0 };
        let mut sum = Complex::default();
        let mut support = ONE;
        let mut advances = 0_u32;
        for atom in &self.atoms {
            let q_length = table[atom.length_index];
            let length = atom.length as f64;
            let dc = e0(-omega, inverse, length, q_length);
            let mut phi = scale(dc, atom.value);
            let (up, down) = (atom.rate - omega, -atom.rate - omega);
            if atom.arc && (up * length).abs().min((down * length).abs()) >= ENDPOINT_GUARD {
                // Endpoint form: the deviation f obeys f'' = -rate^2*(f + a), so
                // F = [f'(L)q^L - f'(0) + i*omega*f(L)q^L + rate^2*a*E0(-omega)]
                //     / ((omega - rate)(omega + rate)): one real division per arc.
                let change = omega * atom.end_change;
                let numerator = c(
                    q_length.re * atom.end_slope - atom.start_slope - q_length.im * change
                        + atom.curvature * dc.re,
                    q_length.im * atom.end_slope + q_length.re * change + atom.curvature * dc.im,
                );
                phi = add(phi, scale(numerator, 1.0 / (up * down)));
            } else if atom.arc {
                // Moment form near omega = ±rate, where the endpoint form is 0/0.
                let plus = e0(up, 1.0 / up, length, mul(atom.turn, q_length));
                let minus = e0(down, 1.0 / down, length, mul(conj(atom.turn), q_length));
                // cos(rate*u)-1 and sin(rate*u) transforms.
                let cosine = sub(scale(add(plus, minus), 0.5), dc);
                let sine = scale(minus_i(sub(plus, minus)), 0.5);
                phi = add(phi, add(scale(cosine, atom.a), scale(sine, atom.b)));
            }
            if atom.slope != 0.0 {
                phi = add(
                    phi,
                    scale(e1(-omega, inverse, length, q_length, dc), atom.slope),
                );
            }
            sum = add(sum, mul(support, phi));
            if atom.advances {
                advances += 1;
                support = if advances.is_multiple_of(64) {
                    phase(-omega * (atom.start + atom.length) as f64)
                } else {
                    mul(support, q_length)
                };
            }
        }
        sum
    }

    /// Rectangular integral and its first two omega derivatives, phase origin at the
    /// window center (keeps the moment weights small for peak refinement).
    fn rect_derivatives(&self, omega: f64, table: &mut Vec<Complex>) -> [Complex; 3] {
        let q = phase(-omega);
        self.lengths_table(q, table);
        let center = self.window.len() as f64 * 0.5;
        let mut out = [Complex::default(); 3];
        let mut support = phase(omega * center);
        let mut advances = 0_u32;
        for atom in &self.atoms {
            let q_length = table[atom.length_index];
            let length = atom.length as f64;
            // psi[k] = integral u^k y(u) exp(-i*omega*u) du; only lines need u^3.
            let (dc, mut psi) = if atom.slope != 0.0 {
                let dc = moments::<4>(-omega, length, q_length);
                let psi =
                    [0, 1, 2].map(|k| add(scale(dc[k], atom.value), scale(dc[k + 1], atom.slope)));
                ([dc[0], dc[1], dc[2]], psi)
            } else {
                let dc = moments::<3>(-omega, length, q_length);
                (dc, dc.map(|m| scale(m, atom.value)))
            };
            let (up, down) = (atom.rate - omega, -atom.rate - omega);
            if atom.arc && (up * length).abs().min((down * length).abs()) >= ENDPOINT_GUARD {
                // Endpoint form F = N/D with D = omega^2 - rate^2, differentiated in
                // omega: F' = (N' - 2*omega*F)/D, F'' = (N'' - 4*omega*F' - 2F)/D, and
                // psi1 = i*F', psi2 = -F''. dE0(-omega)/domega = -i*E1(-omega), etc.
                let inverse = 1.0 / (up * down);
                let end = scale(q_length, atom.end_slope);
                let change = scale(q_length, atom.end_change);
                let n0 = c(
                    end.re - atom.start_slope - omega * change.im + atom.curvature * dc[0].re,
                    end.im + omega * change.re + atom.curvature * dc[0].im,
                );
                let n1 = add(
                    add(minus_i(scale(end, length)), c(-change.im, change.re)),
                    add(
                        scale(change, omega * length),
                        minus_i(scale(dc[1], atom.curvature)),
                    ),
                );
                let n2 = sub(
                    add(
                        scale(end, -length * length),
                        add(
                            scale(change, 2.0 * length),
                            minus_i(scale(change, omega * length * length)),
                        ),
                    ),
                    scale(dc[2], atom.curvature),
                );
                let f0 = scale(n0, inverse);
                let f1 = scale(sub(n1, scale(f0, 2.0 * omega)), inverse);
                let f2 = scale(
                    sub(sub(n2, scale(f1, 4.0 * omega)), scale(f0, 2.0)),
                    inverse,
                );
                psi[0] = add(psi[0], f0);
                psi[1] = add(psi[1], c(-f1.im, f1.re));
                psi[2] = sub(psi[2], f2);
            } else if atom.arc {
                // Moment form near omega = ±rate.
                let plus = moments::<3>(atom.rate - omega, length, mul(atom.turn, q_length));
                let minus =
                    moments::<3>(-atom.rate - omega, length, mul(conj(atom.turn), q_length));
                for k in 0..3 {
                    let cosine = sub(scale(add(plus[k], minus[k]), 0.5), dc[k]);
                    let sine = scale(minus_i(sub(plus[k], minus[k])), 0.5);
                    psi[k] = add(psi[k], add(scale(cosine, atom.a), scale(sine, atom.b)));
                }
            }
            // d/domega of exp(-i*omega*(sigma+u)) contributes -i*(sigma+u).
            let sigma = atom.start as f64 - center;
            let d0 = psi[0];
            let d1 = minus_i(add(scale(psi[0], sigma), psi[1]));
            let d2 = scale(
                add(
                    add(scale(psi[0], sigma * sigma), scale(psi[1], 2.0 * sigma)),
                    psi[2],
                ),
                -1.0,
            );
            out[0] = add(out[0], mul(support, d0));
            out[1] = add(out[1], mul(support, d1));
            out[2] = add(out[2], mul(support, d2));
            if atom.advances {
                advances += 1;
                support = if advances.is_multiple_of(64) {
                    phase(-omega * ((atom.start + atom.length) as f64 - center))
                } else {
                    mul(support, q_length)
                };
            }
        }
        out
    }

    fn weighted(&self, omega: f64, table: &mut Vec<Complex>) -> Complex {
        match self.weighting {
            Weighting::Rectangular => self.rect(omega, table),
            Weighting::Hann => {
                let shift = self.resolution();
                let center = self.rect(omega, table);
                let lower = self.rect(omega - shift, table);
                let upper = self.rect(omega + shift, table);
                sub(scale(center, 0.5), scale(add(lower, upper), 0.25))
            }
        }
    }

    /// Exact `integral_window y(t) w(t) exp(-i*omega*(t-start)) dt` in amplitude *
    /// source-sample units. Any finite real omega, including above pi.
    pub fn integral(&self, omega: f64) -> Result<Complex, Error> {
        if !omega.is_finite() {
            return Err(Error::InvalidParameter("FAFT frequency"));
        }
        checked(self.weighted(omega, &mut Vec::with_capacity(self.lengths.len())))
    }

    /// Normalized coefficient at any real omega (radians per source sample).
    pub fn value(&self, omega: f64) -> Result<Complex, Error> {
        Ok(scale(self.integral(omega)?, 1.0 / self.weight_sum))
    }

    /// Normalized coefficients at an arbitrary list of frequencies. O(rotations)
    /// per frequency; the list need not be uniform, sorted or bounded by pi.
    pub fn values(&self, omegas: &[f64]) -> Result<Vec<Complex>, Error> {
        let mut out = buffer(omegas.len())?;
        let mut table = Vec::with_capacity(self.lengths.len());
        for &omega in omegas {
            if !omega.is_finite() {
                return Err(Error::InvalidParameter("FAFT frequency"));
            }
            out.push(checked(scale(
                self.weighted(omega, &mut table),
                1.0 / self.weight_sum,
            ))?);
        }
        Ok(out)
    }

    /// Normalized value and first/second omega derivatives with phase origin at
    /// the window center. The magnitude equals `value(omega)`'s magnitude.
    pub fn derivatives(&self, omega: f64) -> Result<[Complex; 3], Error> {
        if !omega.is_finite() {
            return Err(Error::InvalidParameter("FAFT frequency"));
        }
        let mut table = Vec::with_capacity(self.lengths.len());
        self.centered(omega, &mut table)
    }

    fn centered(&self, omega: f64, table: &mut Vec<Complex>) -> Result<[Complex; 3], Error> {
        let out = match self.weighting {
            Weighting::Rectangular => self.rect_derivatives(omega, table),
            Weighting::Hann => {
                // Centered periodic Hann: 1/2 F(w) + 1/4 F(w-shift) + 1/4 F(w+shift).
                let shift = self.resolution();
                let a = self.rect_derivatives(omega, table);
                let b = self.rect_derivatives(omega - shift, table);
                let d = self.rect_derivatives(omega + shift, table);
                [0, 1, 2].map(|k| add(scale(a[k], 0.5), scale(add(b[k], d[k]), 0.25)))
            }
        };
        let inverse = 1.0 / self.weight_sum;
        let mut result = [Complex::default(); 3];
        for (slot, z) in result.iter_mut().zip(out) {
            *slot = checked(scale(z, inverse))?;
        }
        Ok(result)
    }

    /// Circle-derived frequency references: each rotation's own angular rate,
    /// weighted by its resonant lobe height `radius*weighted_length/2` (normalized),
    /// grouped so no group spans more than `group_width` (no transitive chains),
    /// then filtered by `relative_floor` of the strongest group weight.
    /// O(rotations log rotations); no transform is evaluated.
    /// Group weights ignore phase cancellation between rotations: they seed and
    /// rank candidates, they are not spectral magnitudes.
    pub fn references(
        &self,
        group_width: f64,
        relative_floor: f64,
    ) -> Result<Vec<Reference>, Error> {
        if !group_width.is_finite()
            || group_width < 0.0
            || !relative_floor.is_finite()
            || !(0.0..=1.0).contains(&relative_floor)
        {
            return Err(Error::InvalidParameter("FAFT reference parameters"));
        }
        let duration = self.window.len() as f64;
        let mut rates = buffer(self.atoms.len())?;
        for atom in self.atoms.iter().filter(|a| a.arc) {
            let radius = atom.a.hypot(atom.b);
            let (t0, t1) = (atom.start as f64, (atom.start + atom.length) as f64);
            let weighted_length = match self.weighting {
                Weighting::Rectangular => t1 - t0,
                Weighting::Hann => {
                    let k = TAU / duration;
                    0.5 * (t1 - t0) - 0.5 * ((k * t1).sin() - (k * t0).sin()) / k
                }
            };
            rates.push((
                atom.rate.abs(),
                0.5 * radius * weighted_length / self.weight_sum,
            ));
        }
        rates.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut groups: Vec<Reference> = buffer(rates.len())?;
        for (rate, weight) in rates {
            match groups.last_mut() {
                Some(g) if rate - g.lowest <= group_width => {
                    g.omega = (g.omega * g.weight + rate * weight)
                        / (g.weight + weight).max(f64::MIN_POSITIVE);
                    g.weight += weight;
                    g.highest = rate;
                    g.rotations += 1;
                }
                _ => groups.push(Reference {
                    omega: rate,
                    weight,
                    lowest: rate,
                    highest: rate,
                    rotations: 1,
                }),
            }
        }
        let strongest = groups.iter().map(|g| g.weight).fold(0.0, f64::max);
        groups.retain(|g| g.weight > 0.0 && g.weight >= relative_floor * strongest);
        Ok(groups)
    }

    /// Continuous relevant peaks of |FAFT| in `[lower, upper]`: seeds from a
    /// duration-limited scan and/or grouped circle rates, safeguarded Newton
    /// refinement with exact analytic derivatives, strongest-first merging, floors.
    /// The scan spacing follows the window DURATION (2*pi/T/density), not a sample
    /// count; its local maxima are heuristic seeds, not a completeness proof.
    pub fn peaks(&self, params: PeakParams) -> Result<PeakSearch, Error> {
        params.validate()?;
        let mut table = Vec::with_capacity(self.lengths.len());
        let mut stats = PeakStats::default();
        let resolution = self.resolution();
        let merge_width = params.merge_width.unwrap_or(0.25 * resolution);
        let mut seeds: Vec<Seed> = Vec::new();
        let mut strongest_scan = 0.0_f64;
        if let Some(density) = params.scan_density {
            let step = resolution / density as f64;
            // Every scanned point lies inside [lower, upper]; brackets are clamped.
            let count = ((params.upper - params.lower) / step).floor() as usize + 1;
            let grid = |k: isize| (params.lower + step * k as f64).min(params.upper);
            // Hann needs neighbours at +-resolution = +-density grid steps: share
            // one extended rectangular scan instead of three sums per point.
            let (pad, rect_count) = match self.weighting {
                Weighting::Rectangular => (0, count),
                Weighting::Hann => (density, count + 2 * density),
            };
            let mut rect = buffer(rect_count)?;
            for k in 0..rect_count {
                let omega = params.lower + step * (k as f64 - pad as f64);
                rect.push(self.rect(omega, &mut table));
            }
            stats.scan_points = count;
            stats.rect_evaluations += rect_count;
            let mut magnitude = buffer(count)?;
            for k in 0..count {
                let z = match self.weighting {
                    Weighting::Rectangular => rect[k],
                    Weighting::Hann => sub(
                        scale(rect[k + pad], 0.5),
                        scale(add(rect[k], rect[k + 2 * pad]), 0.25),
                    ),
                };
                let m = norm(z) / self.weight_sum;
                if !m.is_finite() {
                    return Err(Error::InvalidParameter("FAFT arithmetic overflow"));
                }
                magnitude.push(m);
            }
            strongest_scan = magnitude.iter().copied().fold(0.0, f64::max);
            let floor = params.floor(strongest_scan);
            for k in 0..count {
                let left = if k > 0 {
                    magnitude[k - 1]
                } else {
                    f64::NEG_INFINITY
                };
                let right = magnitude.get(k + 1).copied().unwrap_or(f64::NEG_INFINITY);
                let m = magnitude[k];
                if !(m >= floor && m > 0.0 && m >= left && m > right) {
                    continue;
                }
                // Parabolic start on log magnitude: Newton then needs few steps.
                let mut start = grid(k as isize);
                if k > 0 && k + 1 < count && left > 0.0 && right > 0.0 {
                    let (l, cm, r) = (left.ln(), m.ln(), right.ln());
                    let bend = l - 2.0 * cm + r;
                    if bend < 0.0 {
                        start += (0.5 * (l - r) / bend * step).clamp(-step, step);
                    }
                }
                seeds.push(Seed {
                    omega: start,
                    lo: grid(k as isize - 1).max(params.lower),
                    hi: grid(k as isize + 1),
                });
            }
        }
        if params.circle_seeds {
            let refs = self.references(resolution, params.reference_floor)?;
            stats.references = refs.len();
            // Circle rates are geometric; check relevance with one exact value each
            // before paying for derivative refinement.
            let mut candidates = buffer(refs.len() + 1)?;
            for r in refs {
                if r.omega >= params.lower && r.omega <= params.upper {
                    candidates.push(r.omega);
                }
            }
            if params.lower == 0.0 {
                candidates.push(0.0);
            }
            let mut magnitudes = buffer(candidates.len())?;
            for &omega in &candidates {
                magnitudes.push(norm(self.weighted(omega, &mut table)) / self.weight_sum);
            }
            stats.rect_evaluations += candidates.len()
                * match self.weighting {
                    Weighting::Rectangular => 1,
                    Weighting::Hann => 3,
                };
            let strongest = magnitudes.iter().copied().fold(strongest_scan, f64::max);
            for (&omega, &m) in candidates.iter().zip(&magnitudes) {
                if m > 0.0 && m >= params.floor(strongest) {
                    seeds.push(Seed {
                        omega,
                        lo: (omega - resolution).max(params.lower),
                        hi: (omega + resolution).min(params.upper),
                    });
                }
            }
        }
        stats.seeds = seeds.len();
        let mut candidates: Vec<Peak> = buffer(seeds.len())?;
        for seed in seeds {
            if let Some(peak) = self.refine(seed, params, &mut table, &mut stats)? {
                candidates.push(peak);
            }
        }
        candidates.sort_by(|a, b| b.magnitude().total_cmp(&a.magnitude()));
        let strongest = candidates
            .first()
            .map_or(0.0, |p| p.magnitude())
            .max(strongest_scan);
        let floor = params.floor(strongest);
        let mut peaks: Vec<Peak> = buffer(candidates.len())?;
        for p in candidates {
            if p.magnitude() < floor || p.magnitude() == 0.0 {
                stats.excluded += 1;
            } else if let Some(q) = peaks
                .iter_mut()
                .find(|q| (q.omega - p.omega).abs() <= merge_width)
            {
                q.merged += 1;
                stats.merged += 1;
            } else if peaks.len() < params.max_peaks {
                peaks.push(p);
            } else {
                stats.excluded += 1;
            }
        }
        peaks.sort_by(|a, b| a.omega.total_cmp(&b.omega));
        Ok(PeakSearch { peaks, stats })
    }

    /// Maximize |F|^2 inside the seed bracket using exact derivatives. Newton steps
    /// are used when the local model is concave and inside the bracket, which is
    /// narrowed by the derivative sign; otherwise the bracket is bisected. Only
    /// converged interior maxima are returned; flanks and failures are counted.
    fn refine(
        &self,
        seed: Seed,
        params: PeakParams,
        table: &mut Vec<Complex>,
        stats: &mut PeakStats,
    ) -> Result<Option<Peak>, Error> {
        let (mut lo, mut hi) = (seed.lo, seed.hi);
        let mut omega = seed.omega.clamp(lo, hi);
        let mut iterations = 0;
        let mut d = self.centered(omega, table)?;
        stats.derivative_evaluations += 1;
        let slope = |d: &[Complex; 3]| 2.0 * (d[0].re * d[1].re + d[0].im * d[1].im);
        let bend = |d: &[Complex; 3]| {
            2.0 * (d[1].re * d[1].re + d[1].im * d[1].im + d[0].re * d[2].re + d[0].im * d[2].im)
        };
        let converged = loop {
            let (g1, g2) = (slope(&d), bend(&d));
            if g1 > 0.0 {
                lo = lo.max(omega);
            } else if g1 < 0.0 {
                hi = hi.min(omega);
            }
            let newton = omega - g1 / g2;
            let next = if g2 < 0.0 && newton.is_finite() && newton >= lo && newton <= hi {
                newton
            } else {
                0.5 * (lo + hi)
            };
            if g1 == 0.0 || (next - omega).abs() <= params.tolerance {
                break g2 <= 0.0;
            }
            if iterations == params.max_iterations || hi - lo <= params.tolerance {
                break false;
            }
            omega = next;
            iterations += 1;
            d = self.centered(omega, table)?;
            stats.derivative_evaluations += 1;
        };
        stats.refined += 1;
        stats.iterations += iterations;
        // A maximum pinned to a bracket edge is a monotone flank owned by another
        // seed, unless it is omega=0 where a real curve's |F| is symmetric.
        let near = |a: f64, b: f64| (a - b).abs() <= 4.0 * params.tolerance;
        let edge = (near(omega, seed.lo) || near(omega, seed.hi)) && omega != 0.0;
        if edge {
            stats.flanks += 1;
            return Ok(None);
        }
        if !converged {
            stats.unconverged += 1;
            return Ok(None);
        }
        // Convert the centered value back to a window-start phase origin.
        let value = mul(d[0], phase(-omega * self.window.len() as f64 * 0.5));
        Ok(Some(Peak {
            omega,
            value,
            curvature: bend(&d),
            iterations,
            merged: 0,
        }))
    }
}

#[derive(Debug, Clone, Copy)]
struct Seed {
    omega: f64,
    lo: f64,
    hi: f64,
}

/// Grouped circle rates; `omega` is the weight-averaged rate of the group and
/// `lowest..=highest` spans at most the grouping width.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reference {
    pub omega: f64,
    pub weight: f64,
    pub lowest: f64,
    pub highest: f64,
    pub rotations: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct PeakParams {
    /// Search range in radians per source sample. `upper` may exceed pi: the
    /// continuous curve has model-generated content above the source Nyquist.
    pub lower: f64,
    pub upper: f64,
    /// Scan points per 2*pi/T (window duration), or None for circle seeds only.
    pub scan_density: Option<usize>,
    /// Also seed refinement from grouped circle rates (see `references`).
    pub circle_seeds: bool,
    /// Relative floor on circle-reference weights before refinement.
    pub reference_floor: f64,
    /// Output floor relative to the strongest scanned or refined magnitude.
    pub relative_floor: f64,
    pub absolute_floor: f64,
    /// Peaks closer than this (radians/sample) keep only the strongest;
    /// None uses a quarter of 2*pi/T. This is not a resolution claim.
    pub merge_width: Option<f64>,
    pub max_peaks: usize,
    /// Newton/bisection iterations per seed.
    pub max_iterations: usize,
    /// Frequency convergence tolerance, radians per source sample.
    pub tolerance: f64,
}
impl Default for PeakParams {
    fn default() -> Self {
        Self {
            lower: 0.0,
            upper: PI,
            scan_density: Some(2),
            circle_seeds: false,
            reference_floor: 0.01,
            relative_floor: 0.01,
            absolute_floor: 0.0,
            merge_width: None,
            max_peaks: 4096,
            max_iterations: 32,
            tolerance: 1e-12,
        }
    }
}
impl PeakParams {
    fn validate(&self) -> Result<(), Error> {
        let finite = [
            self.lower,
            self.upper,
            self.reference_floor,
            self.relative_floor,
            self.absolute_floor,
            self.tolerance,
            self.merge_width.unwrap_or(0.0),
        ]
        .iter()
        .all(|v| v.is_finite());
        if !finite
            || self.lower >= self.upper
            || self.scan_density.is_some_and(|d| !(1..=64).contains(&d))
            || (self.scan_density.is_none() && !self.circle_seeds)
            || !(0.0..=1.0).contains(&self.reference_floor)
            || !(0.0..=1.0).contains(&self.relative_floor)
            || self.absolute_floor < 0.0
            || self.merge_width.is_some_and(|w| w < 0.0)
            || self.max_peaks == 0
            || self.max_iterations > 256
            || self.tolerance <= 0.0
        {
            return Err(Error::InvalidParameter("FAFT peak parameters"));
        }
        Ok(())
    }
    fn floor(&self, strongest: f64) -> f64 {
        self.absolute_floor.max(self.relative_floor * strongest)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Peak {
    /// Continuous frequency, radians per source sample.
    pub omega: f64,
    /// Normalized coefficient with phase origin at the window start.
    pub value: Complex,
    /// d^2|F|^2/domega^2 at the peak (normalized units); negative at a maximum.
    pub curvature: f64,
    pub iterations: usize,
    /// Weaker candidates suppressed within the merge width.
    pub merged: usize,
}
impl Peak {
    pub fn magnitude(&self) -> f64 {
        norm(self.value)
    }
}

/// Work counters. A Hann derivative evaluation costs three rectangular derivative
/// sums; Hann scan points share one extended rectangular grid.
#[derive(Debug, Clone, Copy, Default)]
pub struct PeakStats {
    pub scan_points: usize,
    pub rect_evaluations: usize,
    pub references: usize,
    pub seeds: usize,
    pub refined: usize,
    pub derivative_evaluations: usize,
    pub iterations: usize,
    /// Seeds whose maximum sat on a bracket edge (monotone flank): dropped.
    pub flanks: usize,
    /// Seeds without convergence within the iteration limit: dropped, counted.
    pub unconverged: usize,
    pub merged: usize,
    pub excluded: usize,
}

#[derive(Debug, Clone)]
pub struct PeakSearch {
    pub peaks: Vec<Peak>,
    pub stats: PeakStats,
}
