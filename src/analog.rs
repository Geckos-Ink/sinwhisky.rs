//! Exact analog (continuous-time) filtering of circle curves
//! (PAPER_EQ_COMPRESSION.md, Section 2.7).
//!
//! A rational filter `H(s) = direct + sum_j residue_j/(s - pole_j)` (distinct,
//! stable poles closed under conjugation, so its impulse response is real) is
//! applied to a piecewise curve exactly: the state of each pole is carried from
//! piece to piece in closed form, `xi(s+L) = e^{pL} xi(s) + integral`, the
//! rotate-and-add chain of FAFT run in time with complex rates. There is no
//! sampling, step size or quadrature.
//!
//! The pieces are exponential polynomials, `Re[c0 + c1*u + sum_m a_m e^{mu_m u}]`
//! on `[start, start+length)`: a circle's rotation is one term with an imaginary
//! rate, a line is the polynomial part, and a decaying term is a spiral. The class
//! is closed under filtering (each pole adds one term per piece) and under time
//! reversal, which gives zero-phase filtering with response |H|^2 by filtering
//! forward, reversing, filtering again and reversing back. Stored complex
//! coefficients are those of a complex signal whose real part is the curve; a real
//! filter commutes with taking the real part.
//!
//! The curve is assumed zero before its start. A zero-phase pass also assumes it
//! is zero after its end: the forward output's tail beyond the curve is dropped,
//! so callers pad with silence where that tail matters.
use crate::faft::CircleCurve;
use crate::spectral::Complex;
use crate::{buffer, Error};
use std::f64::consts::PI;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Local complex arithmetic with operators; `Complex` stays the public type.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Cx {
    re: f64,
    im: f64,
}
impl Cx {
    const ZERO: Cx = Cx { re: 0.0, im: 0.0 };
    const ONE: Cx = Cx { re: 1.0, im: 0.0 };
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn exp(self) -> Self {
        let (s, c) = self.im.sin_cos();
        let m = self.re.exp();
        Self::new(m * c, m * s)
    }
    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    fn finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
}
impl Add for Cx {
    type Output = Cx;
    fn add(self, o: Cx) -> Cx {
        Cx::new(self.re + o.re, self.im + o.im)
    }
}
impl Sub for Cx {
    type Output = Cx;
    fn sub(self, o: Cx) -> Cx {
        Cx::new(self.re - o.re, self.im - o.im)
    }
}
impl Mul for Cx {
    type Output = Cx;
    fn mul(self, o: Cx) -> Cx {
        Cx::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}
impl Div for Cx {
    type Output = Cx;
    fn div(self, o: Cx) -> Cx {
        let d = o.re * o.re + o.im * o.im;
        Cx::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
}
impl Neg for Cx {
    type Output = Cx;
    fn neg(self) -> Cx {
        Cx::new(-self.re, -self.im)
    }
}
impl From<Complex> for Cx {
    fn from(c: Complex) -> Self {
        Cx::new(c.re, c.im)
    }
}
impl From<Cx> for Complex {
    fn from(c: Cx) -> Self {
        Complex { re: c.re, im: c.im }
    }
}

/// (e^z - 1)/z, accurate near z = 0.
fn phi1(z: Cx) -> Cx {
    if z.abs() < 0.5 {
        let (mut term, mut sum) = (Cx::ONE, Cx::ONE);
        for k in 2..24 {
            term = term * z.scale(1.0 / k as f64);
            sum = sum + term;
        }
        sum
    } else {
        (z.exp() - Cx::ONE) / z
    }
}

/// (e^z - 1 - z)/z^2, accurate near z = 0.
fn phi2(z: Cx) -> Cx {
    if z.abs() < 0.5 {
        let (mut term, mut sum) = (Cx::new(0.5, 0.0), Cx::new(0.5, 0.0));
        for k in 3..26 {
            term = term * z.scale(1.0 / k as f64);
            sum = sum + term;
        }
        sum
    } else {
        (z.exp() - Cx::ONE - z) / (z * z)
    }
}

/// A rational continuous-time filter in partial fractions. Time is measured in
/// source samples, so poles are in radians (or nepers) per sample.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalogFilter {
    direct: f64,
    poles: Vec<Complex>,
    residues: Vec<Complex>,
}
impl AnalogFilter {
    /// `H(s) = direct + sum_j residues[j]/(s - poles[j])`. Poles must be distinct,
    /// strictly stable (negative real part) and, with their residues, closed under
    /// complex conjugation (a real impulse response). At most 32 poles.
    pub fn new(direct: f64, poles: Vec<Complex>, residues: Vec<Complex>) -> Result<Self, Error> {
        let bad = || Error::InvalidParameter("analog filter");
        if poles.len() != residues.len() || poles.len() > 32 || !direct.is_finite() {
            return Err(bad());
        }
        let p: Vec<Cx> = poles.iter().map(|&c| c.into()).collect();
        let r: Vec<Cx> = residues.iter().map(|&c| c.into()).collect();
        for (i, (&pi, &ri)) in p.iter().zip(&r).enumerate() {
            if !pi.finite() || !ri.finite() || pi.re >= 0.0 {
                return Err(bad());
            }
            let scale = pi.abs().max(1e-300);
            if p[..i].iter().any(|&q| (q - pi).abs() <= 1e-12 * scale) {
                return Err(bad());
            }
            let partner = p.iter().zip(&r).any(|(&q, &s)| {
                (q - pi.conj()).abs() <= 1e-9 * scale
                    && (s - ri.conj()).abs() <= 1e-9 * ri.abs().max(1e-300)
            });
            if !partner {
                return Err(bad());
            }
        }
        Ok(Self {
            direct,
            poles,
            residues,
        })
    }

    /// Butterworth low-pass of `order` 1..=12 with cutoff `cutoff` radians per
    /// sample (finite, positive): `|H(omega)|^2 = 1/(1 + (omega/cutoff)^(2n))`.
    /// The cutoff may exceed pi: the filter acts on a continuous curve.
    pub fn butterworth_lowpass(order: usize, cutoff: f64) -> Result<Self, Error> {
        if !(1..=12).contains(&order) || !cutoff.is_finite() || cutoff <= 0.0 {
            return Err(Error::InvalidParameter("Butterworth parameters"));
        }
        let n = order;
        let mut poles = vec![Cx::ZERO; n];
        for k in 0..n.div_ceil(2) {
            let angle = PI * (2 * k + n + 1) as f64 / (2 * n) as f64;
            let p = Cx::new(angle.cos(), angle.sin()).scale(cutoff);
            poles[k] = p;
            poles[n - 1 - k] = p.conj();
        }
        if n % 2 == 1 {
            poles[n / 2] = Cx::new(-cutoff, 0.0);
        }
        let gain = cutoff.powi(n as i32);
        let residues: Vec<Cx> = (0..n)
            .map(|k| {
                let product = (0..n)
                    .filter(|&j| j != k)
                    .fold(Cx::ONE, |acc, j| acc * (poles[k] - poles[j]));
                Cx::new(gain, 0.0) / product
            })
            .collect();
        // Conjugate residues exactly, so validation sees an exactly real filter.
        let mut residues = residues;
        for k in 0..n / 2 {
            residues[n - 1 - k] = residues[k].conj();
        }
        if n % 2 == 1 {
            residues[n / 2].im = 0.0;
        }
        Self::new(
            0.0,
            poles.into_iter().map(Complex::from).collect(),
            residues.into_iter().map(Complex::from).collect(),
        )
    }

    pub fn direct(&self) -> f64 {
        self.direct
    }
    pub fn poles(&self) -> &[Complex] {
        &self.poles
    }
    pub fn residues(&self) -> &[Complex] {
        &self.residues
    }

    fn transfer(&self, s: Cx) -> Cx {
        self.poles
            .iter()
            .zip(&self.residues)
            .fold(Cx::new(self.direct, 0.0), |acc, (&p, &r)| {
                acc + Cx::from(r) / (s - Cx::from(p))
            })
    }

    /// Frequency response `H(i*omega)`: the steady-state gain and phase advance of
    /// a tone at `omega` radians per sample.
    pub fn response(&self, omega: f64) -> Complex {
        self.transfer(Cx::new(0.0, omega)).into()
    }
}

/// One piece of a spiral curve: `Re[constant + slope*u + sum coefficient*e^{rate*u}]`
/// for `0 <= u < length`, `u` measured from `start` in source samples.
#[derive(Debug, Clone, PartialEq)]
pub struct SpiralPiece {
    pub start: usize,
    pub length: usize,
    pub constant: Complex,
    pub slope: Complex,
    /// `(rate, coefficient)` pairs; an imaginary rate is a rotation (circle), a
    /// rate with negative real part a shrinking spiral.
    pub terms: Vec<(Complex, Complex)>,
}

/// A curve of exponential-polynomial pieces tiling `[0, len)`: circle curves,
/// their exact analog-filtered outputs, and the time reversals of both.
#[derive(Debug, Clone, PartialEq)]
pub struct SpiralCurve {
    pieces: Vec<SpiralPiece>,
    len: usize,
}
impl SpiralCurve {
    /// The circle curve with stacked rotations merged into one piece, and pieces
    /// cropped (exactly) to at most `max_piece` samples, which keeps every term
    /// bounded within its piece after reversal. 1 <= max_piece <= 4096.
    pub fn from_circles(curve: &CircleCurve, max_piece: usize) -> Result<Self, Error> {
        if !(1..=4096).contains(&max_piece) {
            return Err(Error::InvalidParameter("spiral piece length"));
        }
        let mut pieces: Vec<SpiralPiece> = buffer(curve.rotations().len())?;
        for r in curve.rotations() {
            // Anchored: y0 + a(cos(bu)-1) + b'sin(bu) + d u
            //         = (y0 - a) + d u + Re[(a - i b') e^{i rate u}].
            let (s, c) = r.phase.sin_cos();
            let (a, b) = (r.radius * s, r.radius * c);
            let y0 = r.center + a;
            let mut constant = Cx::new(y0, 0.0);
            let mut terms = Vec::new();
            if r.radius != 0.0 {
                constant = constant - Cx::new(a, 0.0);
                terms.push((Cx::new(0.0, r.rate), Cx::new(a, -b)));
            }
            match pieces.last_mut() {
                Some(last) if last.start == r.start && last.length == r.length => {
                    last.constant = (Cx::from(last.constant) + constant).into();
                    last.slope.re += r.slope;
                    last.terms
                        .extend(terms.into_iter().map(|(m, a)| (m.into(), a.into())));
                }
                _ => pieces.push(SpiralPiece {
                    start: r.start,
                    length: r.length,
                    constant: constant.into(),
                    slope: Complex {
                        re: r.slope,
                        im: 0.0,
                    },
                    terms: terms
                        .into_iter()
                        .map(|(m, a)| (m.into(), a.into()))
                        .collect(),
                }),
            }
        }
        Self::checked(crop(pieces, max_piece)?, curve.len())
    }

    /// The line curve through `samples` (the infinite-radius limit of circles):
    /// unit pieces joining consecutive samples, and a constant last piece on
    /// `[N-1, N)`, so `samples()` returns the input exactly.
    pub fn from_samples(samples: &[f64]) -> Result<Self, Error> {
        if samples.is_empty() {
            return Err(Error::InvalidParameter("spiral curve samples"));
        }
        if let Some(i) = samples.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFiniteSample(i));
        }
        let mut pieces = buffer(samples.len())?;
        for (i, &v) in samples.iter().enumerate() {
            let next = samples.get(i + 1).copied().unwrap_or(v);
            pieces.push(SpiralPiece {
                start: i,
                length: 1,
                constant: Complex { re: v, im: 0.0 },
                slope: Complex {
                    re: next - v,
                    im: 0.0,
                },
                terms: Vec::new(),
            });
        }
        Self::checked(pieces, samples.len())
    }

    fn checked(pieces: Vec<SpiralPiece>, len: usize) -> Result<Self, Error> {
        let mut end = 0;
        for p in &pieces {
            let finite = [p.constant, p.slope]
                .iter()
                .chain(p.terms.iter().flat_map(|(m, a)| [m, a]))
                .all(|c| c.re.is_finite() && c.im.is_finite());
            if p.start != end || p.length == 0 || !finite {
                return Err(Error::InvalidParameter("spiral curve pieces"));
            }
            end += p.length;
        }
        if end != len || len == 0 {
            return Err(Error::InvalidParameter("spiral curve coverage"));
        }
        Ok(Self { pieces, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn pieces(&self) -> &[SpiralPiece] {
        &self.pieces
    }

    /// The curve at a real position in `[0, len)`.
    pub fn evaluate(&self, position: f64) -> Result<f64, Error> {
        if !position.is_finite() || position < 0.0 || position >= self.len as f64 {
            return Err(Error::InvalidParameter("spiral curve position"));
        }
        let i = self
            .pieces
            .partition_point(|p| ((p.start + p.length) as f64) <= position);
        let p = &self.pieces[i];
        let u = position - p.start as f64;
        let mut v = Cx::from(p.constant) + Cx::from(p.slope).scale(u);
        for &(m, a) in &p.terms {
            v = v + Cx::from(a) * Cx::from(m).scale(u).exp();
        }
        Ok(v.re)
    }

    /// The curve at every integer position `0..len`.
    pub fn samples(&self) -> Result<Vec<f64>, Error> {
        let mut out = buffer(self.len)?;
        let mut powers: Vec<(Cx, Cx)> = Vec::new();
        for p in &self.pieces {
            powers.clear();
            powers.extend(
                p.terms
                    .iter()
                    .map(|&(m, a)| (Cx::from(a), Cx::from(m).exp())),
            );
            let (c0, c1) = (Cx::from(p.constant), Cx::from(p.slope));
            for u in 0..p.length {
                let mut v = c0.re + c1.re * u as f64;
                for (value, step) in powers.iter_mut() {
                    v += value.re;
                    *value = *value * *step;
                }
                out.push(v);
            }
        }
        if out.iter().all(|v| v.is_finite()) {
            Ok(out)
        } else {
            Err(Error::InvalidParameter("spiral curve overflow"))
        }
    }

    /// Exact output of `filter` for this curve, with zero state before it.
    /// Each piece gains one term per pole; O(pieces * terms * poles).
    pub fn filter(&self, filter: &AnalogFilter) -> Result<Self, Error> {
        let poles: Vec<Cx> = filter.poles.iter().map(|&c| c.into()).collect();
        let residues: Vec<Cx> = filter.residues.iter().map(|&c| c.into()).collect();
        let direct = filter.direct;
        let mut state = vec![Cx::ZERO; poles.len()];
        let mut pieces = buffer(self.pieces.len())?;
        for piece in &self.pieces {
            let (c0, c1) = (Cx::from(piece.constant), Cx::from(piece.slope));
            let length = piece.length as f64;
            let terms: Vec<(Cx, Cx)> = piece
                .terms
                .iter()
                .map(|&(m, a)| (m.into(), a.into()))
                .collect();
            for &(m, _) in &terms {
                if poles.iter().any(|&p| (m - p).abs() <= 1e-9 * p.abs()) {
                    return Err(Error::InvalidParameter("analog filter resonance"));
                }
            }
            let mut constant = c0.scale(direct);
            let mut slope = c1.scale(direct);
            let mut out_terms = Vec::with_capacity(terms.len() + poles.len());
            for &(m, a) in &terms {
                out_terms.push((m, a * filter.transfer(m)));
            }
            for (j, (&p, &r)) in poles.iter().zip(&residues).enumerate() {
                let inv = Cx::ONE / p;
                constant = constant - r * (c0 * inv + c1 * inv * inv);
                slope = slope - r * c1 * inv;
                let mut coefficient = state[j] + c0 * inv + c1 * inv * inv;
                for &(m, a) in &terms {
                    coefficient = coefficient - a / (m - p);
                }
                out_terms.push((p, r * coefficient));
                // State at the piece end, in forms without cancellation.
                let pl = p.scale(length);
                let decay = pl.exp();
                let mut next = decay * state[j]
                    + c0 * phi1(pl).scale(length)
                    + c1 * phi2(pl).scale(length * length);
                for &(m, a) in &terms {
                    next = next + a * decay * phi1((m - p).scale(length)).scale(length);
                }
                state[j] = next;
            }
            if !state.iter().all(|s| s.finite()) {
                return Err(Error::InvalidParameter("analog filter overflow"));
            }
            pieces.push(SpiralPiece {
                start: piece.start,
                length: piece.length,
                constant: constant.into(),
                slope: slope.into(),
                terms: out_terms
                    .into_iter()
                    .map(|(m, a)| (m.into(), a.into()))
                    .collect(),
            });
        }
        Self::checked(pieces, self.len)
    }

    /// Time reversal `t -> len - t`: each term's rate changes sign. Exact.
    pub fn reverse(&self) -> Result<Self, Error> {
        let mut pieces = buffer(self.pieces.len())?;
        for p in self.pieces.iter().rev() {
            let length = p.length as f64;
            let (c0, c1) = (Cx::from(p.constant), Cx::from(p.slope));
            pieces.push(SpiralPiece {
                start: self.len - p.start - p.length,
                length: p.length,
                constant: (c0 + c1.scale(length)).into(),
                slope: (-c1).into(),
                terms: p
                    .terms
                    .iter()
                    .map(|&(m, a)| {
                        let (m, a) = (Cx::from(m), Cx::from(a));
                        ((-m).into(), (a * m.scale(length).exp()).into())
                    })
                    .collect(),
            });
        }
        Self::checked(pieces, self.len)
    }

    /// Zero-phase filtering: forward, reverse, forward, reverse. The response is
    /// `|H(omega)|^2`, with no phase shift, for content away from the curve's ends.
    pub fn filter_zero_phase(&self, filter: &AnalogFilter) -> Result<Self, Error> {
        self.filter(filter)?.reverse()?.filter(filter)?.reverse()
    }
}

/// Split pieces longer than `max_piece`; a later part starts at offset o with
/// constant c0 + c1*o and coefficients a*e^{rate*o}.
fn crop(pieces: Vec<SpiralPiece>, max_piece: usize) -> Result<Vec<SpiralPiece>, Error> {
    let extra: usize = pieces
        .iter()
        .map(|p| p.length.div_ceil(max_piece) - 1)
        .sum();
    let mut out = buffer(pieces.len() + extra)?;
    for p in pieces {
        if p.length <= max_piece {
            out.push(p);
            continue;
        }
        let mut offset = 0;
        while offset < p.length {
            let length = max_piece.min(p.length - offset);
            let o = offset as f64;
            let (c0, c1) = (Cx::from(p.constant), Cx::from(p.slope));
            out.push(SpiralPiece {
                start: p.start + offset,
                length,
                constant: (c0 + c1.scale(o)).into(),
                slope: p.slope,
                terms: p
                    .terms
                    .iter()
                    .map(|&(m, a)| {
                        let m = Cx::from(m);
                        (m.into(), (Cx::from(a) * m.scale(o).exp()).into())
                    })
                    .collect(),
            });
            offset += length;
        }
    }
    Ok(out)
}
