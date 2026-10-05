//! SinWhisky anchors: a signal (a band, or everything a previous layer left)
//! described by few samples at dynamic positions and rebuilt by circles.
//!
//! Every inner anchor j defines the circle through anchors j-1, j, j+1, in the
//! plane `(time*alpha, value)` with the range aperture `alpha = 2*range/span`.
//! The circle is traversed with an angle that is a quadratic function of time
//! through the three anchors, so its angular speed is continuous at the anchor
//! even when the gaps differ (an angular acceleration, not a constant rate);
//! where that quadratic would turn back, each side is traversed at its own
//! constant rate. Between anchors j and j+1 the curve cross-fades the circle of
//! j into the circle of j+1 with a raised cosine, so it passes through every
//! anchor with the slope of that anchor's circle. Collinear, flat or
//! nearly-straight triplets use the line between the two anchors.
//!
//! The encoder places anchors greedily: it starts from the ends (and every
//! `max_gap` samples) and inserts the sample of largest error, re-evaluating only
//! the four intervals whose circles change, until every interval is within the
//! tolerance. The gaps between anchors are therefore dynamic. Gaps and values
//! are range coded, each value as a residual from a guess (zero, the previous
//! value, the line or the SinWhisky circle through the last anchors) chosen by
//! size; on what gradient groups leave, alternating extremes make zero best.
use super::coder::{corrupt, Bytes, Gamma, Prob, RangeDecoder, RangeEncoder};
use crate::Error;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::f64::consts::{PI, TAU};

/// What the anchors describe and how closely.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnchorConfig {
    /// Describe only the band below this frequency (radians per sample) of what
    /// the gradient layer left, extracted by a zero-phase Butterworth low-pass;
    /// `None` describes all of it.
    pub cutoff: Option<f64>,
    /// Largest error between anchors, sample units (> 0).
    pub tolerance: f64,
    /// Quantum of the anchor values, sample units (> 0).
    pub step: f64,
    /// Largest initial gap between anchors, 2..=65536 samples.
    pub max_gap: usize,
}
impl AnchorConfig {
    pub(super) fn validate(&self) -> Result<(), Error> {
        let ok = self.cutoff.is_none_or(|c| c > 0.0 && c <= PI)
            && self.tolerance.is_finite()
            && self.tolerance > 0.0
            && (1e-3..=1e6).contains(&self.step)
            && (2..=65536).contains(&self.max_gap);
        if ok {
            Ok(())
        } else {
            Err(Error::InvalidParameter("anchor configuration"))
        }
    }
}

enum Angle {
    /// theta(t) = theta1 + d1 (t - t1) + d2 (t - t1)^2.
    Quadratic {
        t1: f64,
        theta1: f64,
        d1: f64,
        d2: f64,
    },
    /// Constant rate on each side of t1.
    Sides { t: [f64; 3], theta: [f64; 3] },
}

struct Arc {
    cy: f64,
    radius: f64,
    angle: Angle,
}
impl Arc {
    fn value(&self, at: f64) -> f64 {
        let theta = match &self.angle {
            Angle::Quadratic { t1, theta1, d1, d2 } => {
                let u = at - t1;
                theta1 + d1 * u + d2 * u * u
            }
            Angle::Sides { t, theta } => {
                let (a, b) = if at <= t[1] { (0, 1) } else { (1, 2) };
                theta[a] + (theta[b] - theta[a]) * (at - t[a]) / (t[b] - t[a])
            }
        };
        self.cy + self.radius * theta.sin()
    }
}

/// Circle through three anchors (times strictly increasing), or None for a line.
fn circle(t: [f64; 3], v: [f64; 3]) -> Option<Arc> {
    let range = (v[0] - v[1])
        .abs()
        .max((v[1] - v[2]).abs())
        .max((v[0] - v[2]).abs());
    let span = t[2] - t[0];
    if range == 0.0 || span <= 0.0 {
        return None;
    }
    let alpha = 2.0 * range / span;
    let x = [(t[0] - t[1]) * alpha, 0.0, (t[2] - t[1]) * alpha];
    let (a00, a01) = (2.0 * (x[1] - x[0]), 2.0 * (v[1] - v[0]));
    let (a10, a11) = (2.0 * (x[2] - x[1]), 2.0 * (v[2] - v[1]));
    let b0 = x[1] * x[1] - x[0] * x[0] + v[1] * v[1] - v[0] * v[0];
    let b1 = x[2] * x[2] - x[1] * x[1] + v[2] * v[2] - v[1] * v[1];
    let det = a00 * a11 - a01 * a10;
    if det.abs() <= 1e-9 * range * range {
        return None;
    }
    let cx = (b0 * a11 - a01 * b1) / det;
    let cy = (a00 * b1 - b0 * a10) / det;
    let radius = (x[1] - cx).hypot(v[1] - cy);
    if !radius.is_finite() || radius > 1e6 * range {
        return None;
    }
    // Angles unwrapped in the direction that visits the middle anchor.
    let raw: Vec<f64> = (0..3).map(|i| (v[i] - cy).atan2(x[i] - cx)).collect();
    let direction = if (raw[1] - raw[0]).rem_euclid(TAU) <= (raw[2] - raw[0]).rem_euclid(TAU) {
        1.0
    } else {
        -1.0
    };
    let theta0 = raw[0];
    let theta1 = theta0 + direction * (direction * (raw[1] - raw[0])).rem_euclid(TAU);
    let theta2 = theta1 + direction * (direction * (raw[2] - raw[1])).rem_euclid(TAU);
    let theta = [theta0, theta1, theta2];
    let (h0, h2) = (t[1] - t[0], t[2] - t[1]);
    let (s0, s2) = ((theta1 - theta0) / h0, (theta2 - theta1) / h2);
    let d2 = (s2 - s0) / (h0 + h2);
    let d1 = s0 + d2 * h0;
    let start = d1 - 2.0 * d2 * h0;
    let end = d1 + 2.0 * d2 * h2;
    let angle = if start * direction > 0.0 && end * direction > 0.0 {
        Angle::Quadratic {
            t1: t[1],
            theta1,
            d1,
            d2,
        }
    } else {
        Angle::Sides { t, theta }
    };
    Some(Arc { cy, radius, angle })
}

/// Values of the interval [a.0, b.0) given its neighbouring anchors.
fn interval(
    prev: Option<(usize, f64)>,
    a: (usize, f64),
    b: (usize, f64),
    next: Option<(usize, f64)>,
) -> Vec<f64> {
    let tf = |p: usize| p as f64;
    let left = prev.and_then(|p| circle([tf(p.0), tf(a.0), tf(b.0)], [p.1, a.1, b.1]));
    let right = next.and_then(|n| circle([tf(a.0), tf(b.0), tf(n.0)], [a.1, b.1, n.1]));
    let length = (b.0 - a.0) as f64;
    (a.0..b.0)
        .map(|n| {
            let s = (n - a.0) as f64 / length;
            let line = a.1 + (b.1 - a.1) * s;
            let t = n as f64;
            let l = left.as_ref().map(|c| c.value(t));
            let r = right.as_ref().map(|c| c.value(t));
            match (l, r) {
                (Some(l), Some(r)) => {
                    let w = 0.5 + 0.5 * (PI * s).cos();
                    w * l + (1.0 - w) * r
                }
                (Some(v), None) | (None, Some(v)) => v,
                (None, None) => line,
            }
        })
        .collect()
}

/// The curve through `anchors` (first at 0, last at `len - 1`) at every sample.
pub(super) fn render(anchors: &[(usize, f64)], len: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(len);
    for j in 0..anchors.len().saturating_sub(1) {
        let prev = j.checked_sub(1).map(|i| anchors[i]);
        let next = anchors.get(j + 2).copied();
        out.extend(interval(prev, anchors[j], anchors[j + 1], next));
    }
    if let Some(&(_, v)) = anchors.last() {
        out.push(v);
    }
    out.truncate(len);
    out
}

#[derive(PartialEq)]
struct Candidate {
    error: f64,
    left: usize,
    right: usize,
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.error
            .total_cmp(&other.error)
            .then(other.left.cmp(&self.left))
    }
}

/// Greedy placement; returns anchors with quantized values.
fn place(target: &[f64], config: &AnchorConfig) -> Vec<(usize, f64)> {
    let n = target.len();
    let q = |v: f64| (v / config.step).round() * config.step;
    let mut anchors: BTreeMap<usize, f64> = BTreeMap::new();
    for p in (0..n).step_by(config.max_gap).chain(std::iter::once(n - 1)) {
        anchors.insert(p, q(target[p]));
    }
    let neighbour_before =
        |m: &BTreeMap<usize, f64>, p: usize| m.range(..p).next_back().map(|(&k, &v)| (k, v));
    let neighbour_after =
        |m: &BTreeMap<usize, f64>, p: usize| m.range(p + 1..).next().map(|(&k, &v)| (k, v));
    // Max error and its position inside the interval starting at `left`.
    let evaluate = |m: &BTreeMap<usize, f64>, left: usize| -> Option<(usize, f64, usize)> {
        let a = (left, *m.get(&left)?);
        let b = neighbour_after(m, left)?;
        if b.0 - a.0 < 2 {
            return Some((b.0, 0.0, a.0));
        }
        let values = interval(neighbour_before(m, left), a, b, neighbour_after(m, b.0));
        let (mut worst, mut at) = (0.0, a.0);
        for (i, v) in values.iter().enumerate().skip(1) {
            let e = (target[a.0 + i] - v).abs();
            if e > worst {
                worst = e;
                at = a.0 + i;
            }
        }
        Some((b.0, worst, at))
    };
    let mut heap = BinaryHeap::new();
    let mut current: HashMap<usize, (usize, f64, usize)> = HashMap::new();
    let lefts: Vec<usize> = anchors.keys().copied().collect();
    for &left in &lefts[..lefts.len().saturating_sub(1)] {
        if let Some((right, error, at)) = evaluate(&anchors, left) {
            current.insert(left, (right, error, at));
            heap.push(Candidate { error, left, right });
        }
    }
    while let Some(c) = heap.pop() {
        match current.get(&c.left) {
            Some(&(right, error, _)) if right == c.right && error == c.error => {}
            _ => continue,
        }
        if c.error <= config.tolerance {
            break;
        }
        let at = current[&c.left].2;
        anchors.insert(at, q(target[at]));
        let before = neighbour_before(&anchors, c.left).map(|p| p.0);
        let after = neighbour_after(&anchors, c.right).map(|_| c.right);
        for left in [before, Some(c.left), Some(at), after]
            .into_iter()
            .flatten()
        {
            if let Some((right, error, pos)) = evaluate(&anchors, left) {
                current.insert(left, (right, error, pos));
                heap.push(Candidate { error, left, right });
            }
        }
    }
    anchors.into_iter().collect()
}

/// How an anchor value is predicted from the anchors before it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Guess {
    Zero,
    Previous,
    /// The line through the last two anchors.
    Line,
    /// The SinWhisky circle through the last three anchors (aperture-scaled,
    /// the branch of the last one), evaluated at the new position; the line
    /// where it does not reach.
    Circle,
}
const GUESSES: [Guess; 4] = [Guess::Zero, Guess::Previous, Guess::Line, Guess::Circle];

/// Prediction of the value index at `t` from the earlier `(position, index)`
/// pairs. Only additions, multiplications, divisions and square roots, so it
/// is the same on every platform.
fn guess(mode: Guess, before: &[(usize, i64)], t: usize) -> i64 {
    let n = before.len();
    let line = |a: (usize, i64), b: (usize, i64)| {
        let slope = (b.1 - a.1) as f64 / (b.0 - a.0) as f64;
        b.1 as f64 + slope * (t - b.0) as f64
    };
    let v = match (mode, n) {
        (Guess::Zero, _) | (_, 0) => 0.0,
        (Guess::Previous, _) | (_, 1) => before[n - 1].1 as f64,
        (Guess::Line, _) | (_, 2) => line(before[n - 2], before[n - 1]),
        (Guess::Circle, _) => {
            let p = [before[n - 3], before[n - 2], before[n - 1]];
            extrapolate(p, t).unwrap_or_else(|| line(p[1], p[2]))
        }
    };
    v.round().clamp(-(1i64 << 50) as f64, (1i64 << 50) as f64) as i64
}

/// The circle through three anchors, in the plane `((t - t1)*alpha, v)`, at
/// `t` on the branch of the last anchor.
fn extrapolate(p: [(usize, i64); 3], t: usize) -> Option<f64> {
    let v = p.map(|q| q.1 as f64);
    let range = (v[0] - v[1])
        .abs()
        .max((v[1] - v[2]).abs())
        .max((v[0] - v[2]).abs());
    let span = (p[2].0 - p[0].0) as f64;
    if range == 0.0 {
        return None;
    }
    let alpha = 2.0 * range / span;
    let x = p.map(|q| (q.0 as f64 - p[1].0 as f64) * alpha);
    let (a00, a01) = (2.0 * (x[1] - x[0]), 2.0 * (v[1] - v[0]));
    let (a10, a11) = (2.0 * (x[2] - x[1]), 2.0 * (v[2] - v[1]));
    let b0 = x[1] * x[1] - x[0] * x[0] + v[1] * v[1] - v[0] * v[0];
    let b1 = x[2] * x[2] - x[1] * x[1] + v[2] * v[2] - v[1] * v[1];
    let det = a00 * a11 - a01 * a10;
    if det.abs() <= 1e-9 * range * range {
        return None;
    }
    let cx = (b0 * a11 - a01 * b1) / det;
    let cy = (a00 * b1 - b0 * a10) / det;
    let r2 = (x[2] - cx) * (x[2] - cx) + (v[2] - cy) * (v[2] - cy);
    let dx = (t as f64 - p[1].0 as f64) * alpha - cx;
    let inside = r2 - dx * dx;
    if inside.is_nan() || inside < 0.0 || !r2.is_finite() {
        return None;
    }
    let dy = inside.sqrt();
    Some(if v[2] >= cy { cy + dy } else { cy - dy })
}

/// Gamma contexts: bit length of the previous value, capped.
const CONTEXTS: usize = 20;
fn context(v: u64) -> usize {
    ((64 - v.leading_zeros()) as usize).min(CONTEXTS - 1)
}

/// Adaptive models: gaps by the previous gap's length; residual magnitudes by
/// the previous magnitude's length and their signs by the previous two signs
/// (extremes alternate).
struct Models {
    gaps: Vec<Gamma>,
    magnitudes: Vec<Gamma>,
    signs: [Prob; 9],
    last_gap: u64,
    last_magnitude: u64,
    /// The last two signs, each 0 (zero), 1 (positive) or 2 (negative).
    last_signs: usize,
}
impl Models {
    fn new() -> Self {
        Self {
            gaps: vec![Gamma::new(); CONTEXTS],
            magnitudes: vec![Gamma::new(); CONTEXTS],
            signs: [Prob::NEW; 9],
            last_gap: 0,
            last_magnitude: 0,
            last_signs: 0,
        }
    }
    fn put_gap(&mut self, e: &mut RangeEncoder, gap: u64) {
        self.gaps[context(self.last_gap)].put(e, gap);
        self.last_gap = gap;
    }
    fn get_gap(&mut self, d: &mut RangeDecoder) -> Result<u64, Error> {
        let gap = self.gaps[context(self.last_gap)].get(d)?;
        self.last_gap = gap;
        Ok(gap)
    }
    fn after(&mut self, r: i64) {
        self.last_magnitude = r.unsigned_abs();
        let sign = match r.signum() {
            0 => 0,
            1 => 1,
            _ => 2,
        };
        self.last_signs = (self.last_signs % 3) * 3 + sign;
    }
    fn put_residual(&mut self, e: &mut RangeEncoder, r: i64) {
        self.magnitudes[context(self.last_magnitude)].put(e, r.unsigned_abs());
        if r != 0 {
            e.bit(&mut self.signs[self.last_signs], r < 0);
        }
        self.after(r);
    }
    fn get_residual(&mut self, d: &mut RangeDecoder) -> Result<i64, Error> {
        let m = self.magnitudes[context(self.last_magnitude)].get(d)?;
        if m >= 1 << 50 {
            return Err(corrupt());
        }
        let r = if m != 0 && d.bit(&mut self.signs[self.last_signs])? {
            -(m as i64)
        } else {
            m as i64
        };
        self.after(r);
        Ok(r)
    }
}

/// Range-coded anchors: the first value, then per anchor its gap minus one
/// (context: the previous gap) and the residual of its value from the guess
/// (context: the previous residual).
fn code(anchors: &[(usize, i64)], mode: Guess) -> Vec<u8> {
    let mut e = RangeEncoder::new();
    let mut m = Models::new();
    for (j, &(t, v)) in anchors.iter().enumerate() {
        if j > 0 {
            m.put_gap(&mut e, (t - anchors[j - 1].0 - 1) as u64);
        }
        m.put_residual(&mut e, v - guess(mode, &anchors[..j], t));
    }
    e.finish()
}

/// Place anchors on `target` and pack them. Header: u32 count, f32 step, u8
/// guess; then the range-coded gaps and value residuals.
pub(super) fn encode(target: &[f64], config: &AnchorConfig) -> Result<Vec<u8>, Error> {
    config.validate()?;
    if target.is_empty() || target.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidParameter("anchor target"));
    }
    let anchors: Vec<(usize, i64)> = place(target, config)
        .into_iter()
        .map(|(t, v)| (t, (v / config.step).round() as i64))
        .collect();
    if std::env::var("SWL2_DEBUG").is_ok() {
        let sizes: Vec<usize> = GUESSES.iter().map(|&m| code(&anchors, m).len()).collect();
        eprintln!("ANCHORS n={} sizes={sizes:?}", anchors.len());
    }
    let (mode, body) = GUESSES
        .iter()
        .map(|&m| (m, code(&anchors, m)))
        .min_by_key(|c| c.1.len())
        .unwrap();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(anchors.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(config.step as f32).to_le_bytes());
    bytes.push(GUESSES.iter().position(|&g| g == mode).unwrap() as u8);
    bytes.extend(body);
    Ok(bytes)
}

/// Parse anchors (positions and dequantized values) for a signal of `len`.
pub(super) fn parse(payload: &[u8], len: usize) -> Result<Vec<(usize, f64)>, Error> {
    let mut b = Bytes::new(payload);
    let count = b.u32()? as usize;
    let step = f64::from(b.f32()?);
    let mode = *GUESSES.get(usize::from(b.u8()?)).ok_or_else(corrupt)?;
    if count == 0 || count > len || !(1e-3..=1e6).contains(&step) {
        return Err(corrupt());
    }
    let mut d = RangeDecoder::new(b.rest())?;
    let mut m = Models::new();
    let mut anchors: Vec<(usize, i64)> = Vec::with_capacity(count);
    for j in 0..count {
        let t = if j == 0 {
            0
        } else {
            let gap = m.get_gap(&mut d)?;
            let t = anchors[j - 1].0 as u64 + gap + 1;
            if t >= len as u64 {
                return Err(corrupt());
            }
            t as usize
        };
        let r = m.get_residual(&mut d)?;
        let v = guess(mode, &anchors, t)
            .checked_add(r)
            .filter(|v| v.abs() < 1 << 50)
            .ok_or_else(corrupt)?;
        anchors.push((t, v));
    }
    if anchors[count - 1].0 != len - 1 {
        return Err(corrupt());
    }
    Ok(anchors
        .into_iter()
        .map(|(t, v)| (t, v as f64 * step))
        .collect())
}
