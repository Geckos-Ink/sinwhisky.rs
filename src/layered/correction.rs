//! Correction layer: closed-loop linear prediction (LPC) with the earlier layers'
//! prediction as side information, and quantization with the odd step
//! `2*tolerance + 1`, so every reconstructed integer is within `tolerance` of the
//! source (0 is lossless).
//!
//! Each block of `2^block_exponent` samples chooses a gain g in {0, 1/4, .., 1}
//! for the layer prediction P (only when there is one) and an LPC order among
//! `ORDERS`, by the actual coded size. The block then codes
//! `x[n] - round(g*P[n])` with LPC on the same quantity over the reconstructed
//! history, so a layer whose model is wrong in a block costs nothing there beyond
//! its own bytes. Coefficients come from Levinson-Durbin on the Tukey-windowed
//! autocorrelation, with the quantization noise of the history added to its
//! zero lag in lossy streams, are quantized to at most 15-bit integers with a
//! per-block shift, and run in integer arithmetic, so the decoder is exact on
//! every platform. Indices are coded with adaptive contexts selected by a
//! running mean of recent magnitudes.
use super::coder::{corrupt, signed_width, Gamma, RangeDecoder, RangeEncoder, PROB_INIT};
use crate::{buffer, Error};

const ORDERS: [usize; 10] = [0, 1, 2, 4, 8, 12, 16, 20, 24, 32];
const MAX_ORDER: usize = 32;
const PRECISION: u32 = 15;
const CONTEXTS: usize = 24;
/// Gains of the layer prediction, in quarters.
const GAINS: u64 = 5;
/// Exactly coded order candidates per gain (the rest are ranked by estimate).
const EXACT: usize = 2;
/// Predictions and reconstructions are clamped to this magnitude.
const LIMIT: i64 = 1 << 40;

#[derive(Clone)]
struct Model {
    zero: [u16; CONTEXTS],
    sign: [u16; CONTEXTS],
    magnitude: Vec<Gamma>,
    /// About 16 times the running mean of |index|.
    mean: u64,
}
impl Model {
    fn new() -> Self {
        Self {
            zero: [PROB_INIT; CONTEXTS],
            sign: [PROB_INIT; CONTEXTS],
            magnitude: vec![Gamma::new(); CONTEXTS],
            mean: 16,
        }
    }
    fn context(&self) -> usize {
        ((64 - (self.mean >> 3).leading_zeros()) as usize).min(CONTEXTS - 1)
    }
    fn update(&mut self, q: i64) {
        self.mean = self.mean - (self.mean >> 4) + q.unsigned_abs().min(1 << 24);
    }
    fn put(&mut self, e: &mut RangeEncoder, q: i64) {
        let c = self.context();
        e.bit(&mut self.zero[c], q != 0);
        if q != 0 {
            e.bit(&mut self.sign[c], q < 0);
            self.magnitude[c].put(e, q.unsigned_abs() - 1);
        }
        self.update(q);
    }
    fn get(&mut self, d: &mut RangeDecoder) -> Result<i64, Error> {
        let c = self.context();
        let q = if d.bit(&mut self.zero[c])? {
            let negative = d.bit(&mut self.sign[c])?;
            let m = self.magnitude[c].get(d)? + 1;
            if m > 1 << 41 {
                return Err(corrupt());
            }
            if negative {
                -(m as i64)
            } else {
                m as i64
            }
        } else {
            0
        };
        self.update(q);
        Ok(q)
    }
}

/// A block's quantized predictor: integer coefficients and right shift.
#[derive(Clone, Debug, PartialEq)]
struct Predictor {
    coefficients: Vec<i64>,
    shift: u32,
}
impl Predictor {
    fn none() -> Self {
        Self {
            coefficients: Vec::new(),
            shift: 0,
        }
    }
    /// Prediction of the next value of `domain`, whose last `available` values
    /// are `domain(n - 1)`, `domain(n - 2)`, ...
    fn predict(&self, domain: impl Fn(usize) -> i64, available: usize) -> i64 {
        let mut sum: i64 = 0;
        for (i, &c) in self.coefficients.iter().enumerate().take(available) {
            sum = sum.saturating_add(c.saturating_mul(domain(i)));
        }
        let p = if self.shift > 0 {
            sum.saturating_add(1 << (self.shift - 1)) >> self.shift
        } else {
            sum
        };
        p.clamp(-LIMIT, LIMIT)
    }
    fn write(&self, e: &mut RangeEncoder) {
        e.direct(self.coefficients.len() as u64, 6);
        if self.coefficients.is_empty() {
            return;
        }
        let width = self
            .coefficients
            .iter()
            .map(|&c| signed_width(c))
            .max()
            .unwrap_or(1)
            .max(1);
        e.direct(u64::from(self.shift), 5);
        e.direct(u64::from(width), 5);
        for &c in &self.coefficients {
            e.direct((c as u64) & ((1u64 << width) - 1), width);
        }
    }
    fn read(d: &mut RangeDecoder) -> Result<Self, Error> {
        let order = d.direct(6)? as usize;
        if order > MAX_ORDER {
            return Err(corrupt());
        }
        if order == 0 {
            return Ok(Self::none());
        }
        let shift = d.direct(5)? as u32;
        let width = d.direct(5)? as u32;
        if !(1..=PRECISION).contains(&width) {
            return Err(corrupt());
        }
        let mut coefficients = Vec::with_capacity(order);
        for _ in 0..order {
            let v = d.direct(width)?;
            let s = 64 - width;
            coefficients.push(((v << s) as i64) >> s);
        }
        Ok(Self {
            coefficients,
            shift,
        })
    }
}

/// Levinson-Durbin predictors of every order up to `MAX_ORDER` (index = order),
/// from the Tukey(0.5)-windowed autocorrelation of `block`, with `noise` (the
/// variance the reconstructed history adds) on the zero lag.
fn analyse(block: &[i64], noise: f64) -> Vec<Vec<f64>> {
    let n = block.len();
    let taper = (n / 4).max(1);
    let window = |i: usize| {
        let edge = i.min(n - 1 - i);
        if edge >= taper {
            1.0
        } else {
            0.5 - 0.5 * (std::f64::consts::PI * edge as f64 / taper as f64).cos()
        }
    };
    let weights: Vec<f64> = (0..n).map(window).collect();
    let x: Vec<f64> = block
        .iter()
        .zip(&weights)
        .map(|(&v, w)| v as f64 * w)
        .collect();
    let max_order = MAX_ORDER.min(n.saturating_sub(1));
    let mut r = vec![0.0; max_order + 1];
    for (lag, slot) in r.iter_mut().enumerate() {
        *slot = x[lag..].iter().zip(&x).map(|(a, b)| a * b).sum();
    }
    let mut out = vec![Vec::new()];
    if r[0] <= 0.0 || !r[0].is_finite() {
        return out;
    }
    r[0] = r[0] * (1.0 + 1e-9) + noise * weights.iter().map(|w| w * w).sum::<f64>();
    let mut a: Vec<f64> = Vec::new();
    let mut error = r[0];
    for m in 1..=max_order {
        let mut k = r[m];
        for (i, &ai) in a.iter().enumerate() {
            k -= ai * r[m - 1 - i];
        }
        k /= error;
        if !k.is_finite() || k.abs() >= 1.0 {
            break;
        }
        let previous = a.clone();
        a.push(k);
        for i in 0..m - 1 {
            a[i] = previous[i] - k * previous[m - 2 - i];
        }
        error *= 1.0 - k * k;
        out.push(a.clone());
    }
    out
}

fn quantize_coefficients(a: &[f64]) -> Predictor {
    let largest = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if a.is_empty() || largest == 0.0 || !largest.is_finite() {
        return Predictor::none();
    }
    let top = largest.log2().floor() as i64 + 1;
    let shift = (i64::from(PRECISION) - 1 - top).clamp(0, 31) as u32;
    let limit = (1i64 << (PRECISION - 1)) - 1;
    let coefficients = a
        .iter()
        .map(|&v| ((v * (1u64 << shift) as f64).round() as i64).clamp(-limit, limit))
        .collect();
    Predictor {
        coefficients,
        shift,
    }
}

/// `round(g * P)` for a gain index (quarters), in integer arithmetic.
fn scaled(layer: i64, gain: u64) -> i64 {
    let v = layer.saturating_mul(gain as i64);
    (v + 2).div_euclid(4)
}

/// Rough coded size of indices, for ranking orders before exact trials.
fn estimate(q: &[i64]) -> f64 {
    q.iter()
        .map(|v| (1.0 + 2.0 * v.unsigned_abs() as f64).log2())
        .sum()
}

pub(super) struct Correction {
    pub payload: Vec<u8>,
    pub reconstructed: Vec<i64>,
    /// Nonzero indices: the samples whose difference exceeded the tolerance.
    pub written: usize,
    pub mean_order: f64,
    /// Mean layer gain over blocks (1 when there is no layer).
    pub mean_gain: f64,
}

/// Code `x` given the rounded layer prediction `layer` (None: no layer). Payload:
/// one byte with the block exponent (8..=15) and a layer flag (bit 7), then the
/// range-coded blocks: [gain, 3 bits, if layer] [predictor] [indices].
pub(super) fn encode(
    x: &[i64],
    layer: Option<&[i64]>,
    tolerance: u32,
    block_exponent: u32,
) -> Correction {
    let step = 2 * i64::from(tolerance) + 1;
    let noise = if tolerance > 0 {
        (step * step - 1) as f64 / 12.0
    } else {
        0.0
    };
    let block = 1usize << block_exponent;
    let zeros;
    let (layer, gains) = match layer {
        Some(l) => (l, 0..GAINS),
        None => {
            zeros = vec![0i64; x.len()];
            (&zeros[..], 4..GAINS)
        }
    };
    let has_layer = gains.start == 0;
    let mut e = RangeEncoder::new();
    let mut model = Model::new();
    let mut restored: Vec<i64> = Vec::with_capacity(x.len());
    let (mut written, mut orders, mut gain_sum, mut blocks) = (0, 0usize, 0u64, 0usize);
    let mut start = 0;
    while start < x.len() {
        let count = block.min(x.len() - start);
        let tail = restored.len().saturating_sub(MAX_ORDER);
        let mut best: Option<(usize, u64, Predictor)> = None;
        for gain in gains.clone() {
            let target: Vec<i64> = (start..start + count)
                .map(|n| x[n] - scaled(layer[n], gain))
                .collect();
            let predictors = analyse(&target, noise);
            // Rank orders by estimate, then code the best few exactly.
            let mut ranked: Vec<(f64, Predictor)> = ORDERS
                .iter()
                .filter(|&&o| o < predictors.len())
                .map(|&o| {
                    let predictor = quantize_coefficients(&predictors[o]);
                    let mut trial = restored[tail..].to_vec();
                    let shift = tail;
                    let q = run_shifted(
                        x, layer, start, count, gain, &predictor, step, &mut trial, shift,
                    );
                    let bits = estimate(&q) + 16.0 + 15.0 * predictor.coefficients.len() as f64;
                    (bits, predictor)
                })
                .collect();
            ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
            // Without a layer there is one gain: code every order exactly.
            let exact = if has_layer { EXACT } else { ORDERS.len() };
            for (_, predictor) in ranked.into_iter().take(exact) {
                let mut trial_restored = restored[tail..].to_vec();
                let q = run_shifted(
                    x,
                    layer,
                    start,
                    count,
                    gain,
                    &predictor,
                    step,
                    &mut trial_restored,
                    tail,
                );
                let mut trial = RangeEncoder::new();
                let mut trial_model = model.clone();
                if has_layer {
                    trial.direct(gain, 3);
                }
                predictor.write(&mut trial);
                for &v in &q {
                    trial_model.put(&mut trial, v);
                }
                let size = trial.size();
                if best.as_ref().is_none_or(|b| size < b.0) {
                    best = Some((size, gain, predictor));
                }
            }
        }
        let (_, gain, predictor) = best.expect("a candidate always exists");
        if has_layer {
            e.direct(gain, 3);
        }
        predictor.write(&mut e);
        orders += predictor.coefficients.len();
        gain_sum += gain;
        blocks += 1;
        let q = run_shifted(
            x,
            layer,
            start,
            count,
            gain,
            &predictor,
            step,
            &mut restored,
            0,
        );
        for v in q {
            written += usize::from(v != 0);
            model.put(&mut e, v);
        }
        start += count;
    }
    let mut payload = vec![block_exponent as u8 | if has_layer { 0x80 } else { 0 }];
    payload.extend(e.finish());
    Correction {
        payload,
        reconstructed: restored,
        written,
        mean_order: orders as f64 / blocks.max(1) as f64,
        mean_gain: if has_layer {
            gain_sum as f64 / 4.0 / blocks.max(1) as f64
        } else {
            1.0
        },
    }
}

/// Closed-loop coding of one block starting at `start`, on a history that holds
/// the reconstruction from sample `offset` on: returns the indices and appends
/// the reconstruction of x to `partial`.
#[allow(clippy::too_many_arguments)]
fn run_shifted(
    x: &[i64],
    layer: &[i64],
    start: usize,
    count: usize,
    gain: u64,
    predictor: &Predictor,
    step: i64,
    partial: &mut Vec<i64>,
    offset: usize,
) -> Vec<i64> {
    let half = (step - 1) / 2;
    let mut q = Vec::with_capacity(count);
    for n in start..start + count {
        let have = offset + partial.len();
        debug_assert_eq!(have, n);
        let available = partial.len().min(MAX_ORDER);
        let domain = |i: usize| partial[partial.len() - 1 - i] - scaled(layer[n - 1 - i], gain);
        let p = predictor
            .predict(domain, available)
            .saturating_add(scaled(layer[n], gain))
            .clamp(-LIMIT, LIMIT);
        let index = (x[n].clamp(-LIMIT, LIMIT) - p + half).div_euclid(step);
        partial.push((p + index * step).clamp(-LIMIT, LIMIT));
        q.push(index);
    }
    q
}

/// Decode `len` samples given the rounded layer prediction (zeros if none).
pub(super) fn decode(payload: &[u8], layer: &[i64], tolerance: u32) -> Result<Vec<i64>, Error> {
    let step = 2 * i64::from(tolerance) + 1;
    let (&head, data) = payload.split_first().ok_or_else(corrupt)?;
    let exponent = head & 0x7f;
    let has_layer = head & 0x80 != 0;
    if !(8..=15).contains(&exponent) || head & 0x70 != 0 {
        return Err(corrupt());
    }
    let len = layer.len();
    let block = 1usize << exponent;
    let mut d = RangeDecoder::new(data)?;
    let mut model = Model::new();
    let mut restored: Vec<i64> = buffer(len)?;
    while restored.len() < len {
        let gain = if has_layer { d.direct(3)? } else { 4 };
        if gain >= GAINS {
            return Err(corrupt());
        }
        let predictor = Predictor::read(&mut d)?;
        let start = restored.len();
        let count = block.min(len - start);
        for n in start..start + count {
            let q = model.get(&mut d)?;
            let available = n.min(MAX_ORDER);
            let domain = |i: usize| restored[n - 1 - i] - scaled(layer[n - 1 - i], gain);
            let p = predictor
                .predict(domain, available)
                .saturating_add(scaled(layer[n], gain))
                .clamp(-LIMIT, LIMIT);
            let v = q
                .checked_mul(step)
                .and_then(|v| v.checked_add(p))
                .ok_or_else(corrupt)?;
            restored.push(v.clamp(-LIMIT, LIMIT));
        }
    }
    Ok(restored)
}
