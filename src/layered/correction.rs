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
//! every platform. A block can repeat the previous block's predictor (one bit)
//! or send coefficients as differences from it. Indices are coded with adaptive
//! contexts selected by a running mean of recent magnitudes. Everything adaptive
//! (probabilities, the predictor, the stages and the last samples) can continue
//! into the next segment of the stream (`State`).
//!
//! Optionally (a header flag), a cascade of sign-sign LMS stages of 1024, 256 and
//! 16 taps follows the LPC, as in Monkey's Audio: each stage predicts the
//! residual left by the previous ones from its own reconstructed history and
//! adapts after every sample, in integer arithmetic. The stages always adapt;
//! each block says how many of them (0..=3) add their prediction, chosen by the
//! actual coded size among the best LPC candidates.
use super::coder::{corrupt, signed_width, Gamma, Prob, RangeDecoder, RangeEncoder};
use crate::{buffer, Error};

const ORDERS: [usize; 10] = [0, 1, 2, 4, 8, 12, 16, 20, 24, 32];
const MAX_ORDER: usize = 32;
const PRECISION: u32 = 15;
/// Coarser coefficient precisions tried for every exactly coded order.
const COARSER: [u32; 2] = [12, 9];
const CONTEXTS: usize = 24;
/// Gains of the layer prediction, in quarters.
const GAINS: u64 = 5;
/// Exactly coded order candidates per gain (the rest are ranked by estimate).
const EXACT: usize = 2;
/// Predictions and reconstructions are clamped to this magnitude.
const LIMIT: i64 = 1 << 40;
/// Cascade stages: taps and the weights' fixed-point shift.
const STAGES: [(usize, u32); 3] = [(1024, 15), (256, 13), (16, 11)];
/// LPC candidates (by exact size) tried again with the cascade.
const CASCADE_CANDIDATES: usize = 2;
/// The best candidate of at most this order is tried with the cascade too.
const SHORT_ORDER: usize = 2;
/// Stage histories are saturated to this magnitude, weights to `WEIGHT_LIMIT`.
const HISTORY_LIMIT: i64 = (1 << 24) - 1;
const WEIGHT_LIMIT: i32 = 1 << 20;
/// Samples between moves of a stage's history window.
const WINDOW: usize = 512;

#[derive(Clone)]
struct Model {
    zero: [Prob; CONTEXTS],
    sign: [Prob; CONTEXTS],
    magnitude: Vec<Gamma>,
    /// About 16 times the running mean of |index|.
    mean: u64,
    predictor: PredictorModel,
}
impl Model {
    fn new() -> Self {
        Self {
            zero: [Prob::NEW; CONTEXTS],
            sign: [Prob::NEW; CONTEXTS],
            magnitude: vec![Gamma::new(); CONTEXTS],
            mean: 16,
            predictor: PredictorModel::new(),
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
    /// Coefficients of `previous` aligned with this predictor's: zero beyond
    /// its order. Only meaningful when the shifts agree.
    fn delta_from(&self, previous: &Predictor) -> Option<Vec<i64>> {
        (self.shift == previous.shift && !self.coefficients.is_empty()).then(|| {
            self.coefficients
                .iter()
                .enumerate()
                .map(|(i, &c)| c - previous.coefficients.get(i).copied().unwrap_or(0))
                .collect()
        })
    }
    fn width(&self) -> u32 {
        self.coefficients
            .iter()
            .map(|&c| signed_width(c))
            .max()
            .unwrap_or(1)
            .max(1)
    }
}

/// Signed to unsigned: 0, -1, 1, -2, ... to 0, 1, 2, 3, ...
fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}
fn unzigzag(v: u64) -> i64 {
    (v >> 1) as i64 ^ -((v & 1) as i64)
}

/// Adaptive coding of a block's predictor relative to the previous block's: a
/// "same as before" bit; otherwise order and shift, then either deltas from the
/// previous coefficients (adaptive gamma, same shift only) or every coefficient
/// in a declared width (a key predictor).
#[derive(Clone)]
struct PredictorModel {
    reuse: Prob,
    delta: Prob,
    coefficients: Gamma,
}
impl PredictorModel {
    fn new() -> Self {
        Self {
            reuse: Prob::NEW,
            delta: Prob::NEW,
            coefficients: Gamma::new(),
        }
    }
    fn put(&mut self, e: &mut RangeEncoder, p: &Predictor, previous: &Predictor) {
        let same = p == previous;
        e.bit(&mut self.reuse, same);
        if same {
            return;
        }
        e.direct(p.coefficients.len() as u64, 6);
        if p.coefficients.is_empty() {
            return;
        }
        e.direct(u64::from(p.shift), 5);
        let width = p.width();
        let key_bits = 5 + p.coefficients.len() as u64 * u64::from(width);
        let delta = p.delta_from(previous).filter(|d| {
            let bits: u64 = d
                .iter()
                .map(|&v| 2 * u64::from(64 - (zigzag(v) + 1).leading_zeros()) - 1)
                .sum();
            bits < key_bits
        });
        if previous.shift == p.shift {
            e.bit(&mut self.delta, delta.is_some());
        }
        match delta {
            Some(d) => {
                for v in d {
                    self.coefficients.put(e, zigzag(v));
                }
            }
            None => {
                e.direct(u64::from(width), 5);
                for &c in &p.coefficients {
                    e.direct((c as u64) & ((1u64 << width) - 1), width);
                }
            }
        }
    }
    fn get(&mut self, d: &mut RangeDecoder, previous: &Predictor) -> Result<Predictor, Error> {
        if d.bit(&mut self.reuse)? {
            return Ok(previous.clone());
        }
        let order = d.direct(6)? as usize;
        if order > MAX_ORDER {
            return Err(corrupt());
        }
        if order == 0 {
            return Ok(Predictor::none());
        }
        let shift = d.direct(5)? as u32;
        let limit = 1i64 << PRECISION;
        let mut coefficients = Vec::with_capacity(order);
        if previous.shift == shift && d.bit(&mut self.delta)? {
            for i in 0..order {
                let base = previous.coefficients.get(i).copied().unwrap_or(0);
                let c = base + unzigzag(self.coefficients.get(d)?.min(1 << 40));
                if c.abs() >= limit {
                    return Err(corrupt());
                }
                coefficients.push(c);
            }
        } else {
            let width = d.direct(5)? as u32;
            if !(1..=PRECISION).contains(&width) {
                return Err(corrupt());
            }
            for _ in 0..order {
                let v = d.direct(width)?;
                let s = 64 - width;
                coefficients.push(((v << s) as i64) >> s);
            }
        }
        Ok(Predictor {
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

fn quantize_coefficients(a: &[f64], precision: u32) -> Predictor {
    let largest = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if a.is_empty() || largest == 0.0 || !largest.is_finite() {
        return Predictor::none();
    }
    let top = largest.log2().floor() as i64 + 1;
    let shift = (i64::from(precision) - 1 - top).clamp(0, 31) as u32;
    let limit = (1i64 << (precision - 1)) - 1;
    let coefficients = a
        .iter()
        .map(|&v| ((v * (1u64 << shift) as f64).round() as i64).clamp(-limit, limit))
        .collect();
    Predictor {
        coefficients,
        shift,
    }
}

/// One sign-sign LMS stage: predicts its input from the last `order` inputs.
#[derive(Clone)]
struct Stage {
    shift: u32,
    weights: Vec<i32>,
    /// Saturated inputs and their adaptation steps; the last `order` entries
    /// before `pos` are the history, oldest first.
    input: Vec<i32>,
    step: Vec<i32>,
    pos: usize,
    /// Running mean of |input|.
    mean: i64,
}
impl Stage {
    fn new(order: usize, shift: u32) -> Self {
        Self {
            shift,
            weights: vec![0; order],
            input: vec![0; order + WINDOW],
            step: vec![0; order + WINDOW],
            pos: order,
            mean: 0,
        }
    }
    fn order(&self) -> usize {
        self.weights.len()
    }
    fn predict(&self) -> i64 {
        let history = &self.input[self.pos - self.order()..self.pos];
        let dot: i64 = history
            .iter()
            .zip(&self.weights)
            .map(|(&h, &w)| i64::from(h) * i64::from(w))
            .sum();
        (dot + (1 << (self.shift - 1))) >> self.shift
    }
    /// Adapt to the sign of `error` (input minus prediction), then push `input`.
    fn update(&mut self, input: i64, error: i64) {
        let order = self.order();
        let steps = &self.step[self.pos - order..self.pos];
        if error > 0 {
            for (w, &d) in self.weights.iter_mut().zip(steps) {
                *w = (*w + d).clamp(-WEIGHT_LIMIT, WEIGHT_LIMIT);
            }
        } else if error < 0 {
            for (w, &d) in self.weights.iter_mut().zip(steps) {
                *w = (*w - d).clamp(-WEIGHT_LIMIT, WEIGHT_LIMIT);
            }
        }
        let v = input.clamp(-HISTORY_LIMIT, HISTORY_LIMIT);
        let m = v.abs();
        // Larger steps for inputs far above the running mean (Monkey's Audio).
        let size = if m > 3 * self.mean {
            32
        } else if m > 4 * self.mean / 3 {
            16
        } else if m > 0 {
            8
        } else {
            0
        };
        self.mean += (m - self.mean) / 16;
        self.input[self.pos] = v as i32;
        self.step[self.pos] = size * v.signum() as i32;
        // Older inputs adapt with halved steps after 1, 2 and 8 samples.
        for lag in [1, 2, 8] {
            self.step[self.pos - lag] >>= 1;
        }
        self.pos += 1;
        if self.pos == self.input.len() {
            self.input.copy_within(self.pos - order.., 0);
            self.step.copy_within(self.pos - order.., 0);
            self.pos = order;
        }
    }
}

/// The LMS stages after the LPC, each predicting what the previous ones left.
#[derive(Clone)]
struct Cascade {
    stages: Vec<Stage>,
}
impl Cascade {
    fn new() -> Self {
        Self {
            stages: STAGES.iter().map(|&(o, s)| Stage::new(o, s)).collect(),
        }
    }
    fn predict(&self) -> [i64; 3] {
        let mut out = [0; 3];
        for (o, stage) in out.iter_mut().zip(&self.stages) {
            *o = stage.predict();
        }
        out
    }
    /// Adapt every stage to the LPC residual `r` of the reconstruction, given
    /// the stages' predictions for this sample.
    fn update(&mut self, mut r: i64, predictions: &[i64; 3]) {
        for (stage, &p) in self.stages.iter_mut().zip(predictions) {
            let next = r.saturating_sub(p);
            stage.update(r, next);
            r = next;
        }
    }
}

/// Prediction of a sample from its LPC and layer part `base` plus the first
/// `uses` cascade predictions.
fn total(base: i64, cascade: &[i64; 3], uses: usize) -> i64 {
    cascade[..uses]
        .iter()
        .fold(base, |s, &p| s.saturating_add(p))
        .clamp(-LIMIT, LIMIT)
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

/// What a continuation segment inherits from the previous one: the adaptive
/// index model, the LMS stages, the last predictor and the last `MAX_ORDER`
/// reconstructed samples with their layer predictions. A key segment starts
/// from `State::new()`.
#[derive(Clone)]
pub(super) struct State {
    model: Model,
    stages: Option<Cascade>,
    previous: Predictor,
    restored: Vec<i64>,
    layer: Vec<i64>,
}
impl State {
    pub(super) fn new() -> Self {
        Self {
            model: Model::new(),
            stages: None,
            previous: Predictor::none(),
            restored: Vec::new(),
            layer: Vec::new(),
        }
    }
    /// The state after a segment: `restored` and `layer` include the
    /// inherited history.
    fn after(
        model: Model,
        stages: Option<Cascade>,
        previous: Predictor,
        restored: &[i64],
        layer: &[i64],
    ) -> Self {
        let tail = |v: &[i64]| v[v.len().saturating_sub(MAX_ORDER)..].to_vec();
        Self {
            model,
            stages,
            previous,
            restored: tail(restored),
            layer: tail(layer),
        }
    }
}

pub(super) struct Correction {
    pub payload: Vec<u8>,
    /// The state the next continuation segment starts from.
    pub state: State,
    pub reconstructed: Vec<i64>,
    /// Nonzero indices: the samples whose difference exceeded the tolerance.
    pub written: usize,
    pub mean_order: f64,
    /// Mean layer gain over blocks (1 when there is no layer).
    pub mean_gain: f64,
    /// Mean number of cascade stages used per block (0 without the cascade).
    pub mean_stages: f64,
}

/// Code `x` given the rounded layer prediction `layer` (None: no layer). Payload:
/// one byte with the block exponent (8..=15), a cascade flag (bit 4) and a layer
/// flag (bit 7), then the range-coded blocks: [gain, 3 bits, if layer]
/// [predictor] [stages used, 2 bits, if cascade] [indices].
pub(super) fn encode(
    x: &[i64],
    layer: Option<&[i64]>,
    tolerance: u32,
    block_exponent: u32,
    cascade: bool,
    state: State,
) -> Correction {
    let step = 2 * i64::from(tolerance) + 1;
    let noise = if tolerance > 0 {
        (step * step - 1) as f64 / 12.0
    } else {
        0.0
    };
    let block = 1usize << block_exponent;
    let has_layer = layer.is_some();
    let gains = if has_layer { 0..GAINS } else { 4..GAINS };
    // The inherited history comes first: samples before `inherited` are never
    // coded, only read as history.
    let inherited = state.restored.len();
    let mut extended = vec![0i64; inherited];
    extended.extend_from_slice(x);
    let x = &extended[..];
    let mut layer_extended = state.layer.clone();
    match layer {
        Some(l) => layer_extended.extend_from_slice(l),
        None => layer_extended.resize(x.len(), 0),
    }
    let layer = &layer_extended[..];
    let header = |e: &mut RangeEncoder,
                  model: &mut Model,
                  previous: &Predictor,
                  gain: u64,
                  predictor: &Predictor,
                  uses: usize| {
        if has_layer {
            e.direct(gain, 3);
        }
        model.predictor.put(e, predictor, previous);
        if cascade {
            e.direct(uses as u64, 2);
        }
    };
    let mut e = RangeEncoder::new();
    let mut model = state.model;
    let mut stages = if cascade {
        Some(state.stages.unwrap_or_else(Cascade::new))
    } else {
        None
    };
    let mut restored: Vec<i64> = Vec::with_capacity(x.len());
    restored.extend_from_slice(&state.restored);
    let mut previous = state.previous;
    let (mut written, mut orders, mut gain_sum, mut used, mut blocks) = (0, 0, 0, 0, 0);
    let mut start = inherited;
    while start < x.len() {
        let count = block.min(x.len() - start);
        let tail = restored.len().saturating_sub(MAX_ORDER);
        // A trial on a copy of the recent history: the outcome of coding the
        // block (indices, history from `tail` on, cascade state after it).
        let run = |gain: u64,
                   predictor: &Predictor,
                   mut stages: Option<Cascade>,
                   uses: usize,
                   alternatives: Option<&mut [Vec<i64>; 4]>| {
            let mut history = restored[tail..].to_vec();
            let q = run_shifted(
                x,
                layer,
                start,
                count,
                gain,
                predictor,
                step,
                &mut history,
                tail,
                stages.as_mut(),
                uses,
                alternatives,
            );
            Outcome { q, history, stages }
        };
        let size = |gain: u64, predictor: &Predictor, uses: usize, q: &[i64]| {
            let mut trial = RangeEncoder::new();
            let mut trial_model = model.clone();
            header(
                &mut trial,
                &mut trial_model,
                &previous,
                gain,
                predictor,
                uses,
            );
            for &v in q {
                trial_model.put(&mut trial, v);
            }
            trial.size()
        };
        // Exactly coded LPC candidates without the cascade.
        let mut candidates: Vec<(usize, u64, Predictor, Outcome)> = Vec::new();
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
                    let predictor = quantize_coefficients(&predictors[o], PRECISION);
                    let q = run(gain, &predictor, None, 0, None).q;
                    let bits = estimate(&q) + 16.0 + 15.0 * predictor.coefficients.len() as f64;
                    (bits, predictor)
                })
                .collect();
            ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
            // Without a layer there is one gain: code every order exactly.
            let exact = if has_layer { EXACT } else { ORDERS.len() };
            // The previous block's predictor costs one bit.
            let reuse = (!previous.coefficients.is_empty()).then(|| (0.0, previous.clone()));
            for (_, predictor) in ranked.into_iter().take(exact).chain(reuse) {
                let order = predictor.coefficients.len();
                let reused = predictor == previous;
                let mut versions = vec![predictor];
                // Coarser coefficients: fewer header bits for a little accuracy.
                for &precision in COARSER.iter().filter(|_| order > 0 && !reused) {
                    versions.push(quantize_coefficients(&predictors[order], precision));
                }
                for predictor in versions {
                    let outcome = run(gain, &predictor, None, 0, None);
                    let bytes = size(gain, &predictor, 0, &outcome.q);
                    candidates.push((bytes, gain, predictor, outcome));
                }
            }
        }
        candidates.sort_by_key(|c| c.0);
        // The best LPC candidates again with the stages: (bytes, candidate, uses,
        // outcome). The first candidate with no stage is among them, so the
        // chosen outcome always carries the adapted cascade.
        let mut best: Option<(usize, usize, usize, Outcome)> = None;
        if let Some(current) = &stages {
            let mut keep = |bytes: usize, index: usize, uses: usize, outcome: Outcome| {
                if best.as_ref().is_none_or(|b| bytes < b.0) {
                    best = Some((bytes, index, uses, outcome));
                }
            };
            // The best candidates, and the best short one, which the cascade
            // may complete more cheaply than a long one.
            let mut chosen: Vec<usize> = (0..candidates.len().min(CASCADE_CANDIDATES)).collect();
            for range in [0..=SHORT_ORDER, SHORT_ORDER + 1..=8] {
                if let Some(i) = candidates
                    .iter()
                    .position(|c| range.contains(&c.2.coefficients.len()))
                {
                    if !chosen.contains(&i) {
                        chosen.push(i);
                    }
                }
            }
            for index in chosen {
                let (_, gain, predictor, _) = &candidates[index];
                if step == 1 {
                    // Lossless: the reconstruction does not depend on the stages
                    // used, so one run gives every alternative.
                    let mut alternatives: [Vec<i64>; 4] = Default::default();
                    let shared = run(
                        *gain,
                        predictor,
                        Some(current.clone()),
                        0,
                        Some(&mut alternatives),
                    );
                    for (uses, q) in alternatives.into_iter().enumerate() {
                        let bytes = size(*gain, predictor, uses, &q);
                        let outcome = Outcome {
                            q,
                            history: shared.history.clone(),
                            stages: shared.stages.clone(),
                        };
                        keep(bytes, index, uses, outcome);
                    }
                } else {
                    let first = if index == 0 { 0 } else { 1 };
                    for uses in first..=STAGES.len() {
                        let outcome = run(*gain, predictor, Some(current.clone()), uses, None);
                        let bytes = size(*gain, predictor, uses, &outcome.q);
                        keep(bytes, index, uses, outcome);
                    }
                }
            }
        }
        let (gain, predictor, uses, outcome) = match best {
            Some((_, index, uses, outcome)) => {
                let (_, gain, predictor, _) = candidates.swap_remove(index);
                (gain, predictor, uses, outcome)
            }
            None => {
                let (_, gain, predictor, outcome) = candidates.swap_remove(0);
                (gain, predictor, 0, outcome)
            }
        };
        header(&mut e, &mut model, &previous, gain, &predictor, uses);
        orders += predictor.coefficients.len();
        previous = predictor.clone();
        gain_sum += gain;
        used += uses;
        blocks += 1;
        restored.extend_from_slice(&outcome.history[restored.len() - tail..]);
        if outcome.stages.is_some() {
            stages = outcome.stages;
        }
        for v in outcome.q {
            written += usize::from(v != 0);
            model.put(&mut e, v);
        }
        start += count;
    }
    let flags = if has_layer { 0x80 } else { 0 } | if cascade { 0x10 } else { 0 };
    let mut payload = vec![block_exponent as u8 | flags];
    payload.extend(e.finish());
    let blocks = blocks.max(1) as f64;
    Correction {
        payload,
        state: State::after(model, stages, previous, &restored, layer),
        reconstructed: restored.split_off(inherited),
        written,
        mean_order: orders as f64 / blocks,
        mean_gain: if has_layer {
            gain_sum as f64 / 4.0 / blocks
        } else {
            1.0
        },
        mean_stages: used as f64 / blocks,
    }
}

/// A block coded on a copy of the history: its indices, the history with the
/// block's reconstruction appended, and the cascade after it (if it ran).
struct Outcome {
    q: Vec<i64>,
    history: Vec<i64>,
    stages: Option<Cascade>,
}

/// Closed-loop coding of one block starting at `start`, on a history that holds
/// the reconstruction from sample `offset` on: returns the indices and appends
/// the reconstruction of x to `partial`. With `stages`, the first `uses` add
/// their predictions and all of them adapt; `alternatives[s]`, if given,
/// receives the index that using s stages would give with this reconstruction
/// (the actual indices when lossless, where the reconstruction is x for any s).
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
    mut stages: Option<&mut Cascade>,
    uses: usize,
    mut alternatives: Option<&mut [Vec<i64>; 4]>,
) -> Vec<i64> {
    let half = (step - 1) / 2;
    let mut q = Vec::with_capacity(count);
    for n in start..start + count {
        let have = offset + partial.len();
        debug_assert_eq!(have, n);
        let available = partial.len().min(MAX_ORDER);
        let domain = |i: usize| partial[partial.len() - 1 - i] - scaled(layer[n - 1 - i], gain);
        let base = predictor
            .predict(domain, available)
            .saturating_add(scaled(layer[n], gain))
            .clamp(-LIMIT, LIMIT);
        let extra = stages.as_deref().map_or([0; 3], Cascade::predict);
        let p = total(base, &extra, uses);
        let target = x[n].clamp(-LIMIT, LIMIT);
        let index = (target - p + half).div_euclid(step);
        let value = (p + index * step).clamp(-LIMIT, LIMIT);
        if let Some(a) = alternatives.as_deref_mut() {
            for (s, list) in a.iter_mut().enumerate() {
                list.push((target - total(base, &extra, s) + half).div_euclid(step));
            }
        }
        if let Some(c) = stages.as_deref_mut() {
            c.update(value - base, &extra);
        }
        partial.push(value);
        q.push(index);
    }
    q
}

/// Decode `len` samples given the rounded layer prediction (zeros if none).
pub(super) fn decode(
    payload: &[u8],
    layer: &[i64],
    tolerance: u32,
    state: State,
) -> Result<(Vec<i64>, State), Error> {
    let step = 2 * i64::from(tolerance) + 1;
    let (&head, data) = payload.split_first().ok_or_else(corrupt)?;
    let exponent = head & 0x0f;
    let has_layer = head & 0x80 != 0;
    if !(8..=15).contains(&exponent) || head & 0x60 != 0 {
        return Err(corrupt());
    }
    let mut stages = if head & 0x10 != 0 {
        Some(state.stages.unwrap_or_else(Cascade::new))
    } else {
        None
    };
    let inherited = state.restored.len();
    let mut layer_extended = state.layer;
    layer_extended.extend_from_slice(layer);
    let layer = &layer_extended[..];
    let len = layer.len();
    let block = 1usize << exponent;
    let mut d = RangeDecoder::new(data)?;
    let mut model = state.model;
    let mut previous = state.previous;
    let mut restored: Vec<i64> = buffer(len)?;
    restored.extend_from_slice(&state.restored);
    while restored.len() < len {
        let gain = if has_layer { d.direct(3)? } else { 4 };
        if gain >= GAINS {
            return Err(corrupt());
        }
        let predictor = model.predictor.get(&mut d, &previous)?;
        previous = predictor.clone();
        let uses = if stages.is_some() {
            d.direct(2)? as usize
        } else {
            0
        };
        let start = restored.len();
        let count = block.min(len - start);
        for n in start..start + count {
            let q = model.get(&mut d)?;
            let available = n.min(MAX_ORDER);
            let domain = |i: usize| restored[n - 1 - i] - scaled(layer[n - 1 - i], gain);
            let base = predictor
                .predict(domain, available)
                .saturating_add(scaled(layer[n], gain))
                .clamp(-LIMIT, LIMIT);
            let extra = stages.as_ref().map_or([0; 3], Cascade::predict);
            let p = total(base, &extra, uses);
            let v = q
                .checked_mul(step)
                .and_then(|v| v.checked_add(p))
                .ok_or_else(corrupt)?
                .clamp(-LIMIT, LIMIT);
            if let Some(c) = stages.as_mut() {
                c.update(v - base, &extra);
            }
            restored.push(v);
        }
    }
    let state = State::after(model, stages, previous, &restored, layer);
    Ok((restored.split_off(inherited), state))
}
