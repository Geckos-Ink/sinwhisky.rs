// SinWhisky and the AFT/FAFT for WebAssembly, with a SIMD (v128, f64x2) path: a reference port of
// sinwhisky.rs (AssemblyScript, compiled with `--enable simd`). Everything is f64, as in the crate.
//
// Ported contracts (the crate is the authority; ports/fixtures/vectors.json holds its outputs):
//   - circle fitting: `Analysis::new` with `AnalysisParams::original()` (the triplet's range as aperture,
//     1 for a constant triplet; determinant tolerance 4e-6 in the 2×2 system), `Circle::through`, and
//     `arc_angles` (angles unwrapped in the direction that visits the middle point).
//   - the resampler: `resample(samples, ResampleParams::original(zoom))` (equal angles, the finite-zoom
//     sine window, lines for degenerate triplets, linear fallback, originals copied).
//   - the AFT: `AngularFrame::new(samples, AnalysisParams::original())` (periodic halos, the two angular
//     halves of every unit interval averaged, a half whose endpoints miss the samples by more than 1e-8
//     replaced by the source pair's line), `evaluate`, and `fourier` (the integral on [0, N]).
//   - FAFT on that curve: `CircleCurve::from_angular(..).transform(0..N, Rectangular | Hann)`, `values`,
//     and `peaks` (scan, parabolic start, Newton with exact derivatives, flanks dropped, merge, floors).
//
// SIMD: the resampler evaluates both circles covering an interval in the two lanes of an f64x2 and
// advances their equal-angle steps by a complex rotation (no sin per output sample); FAFT evaluates two
// frequencies per f64x2. Both agree with the scalar path to rounding (about 1e-13 of the amplitude).
// Static buffers only (the `stub` runtime): no allocation, so wrapper views stay valid.

const MAX_INPUT: i32 = 4096;
const MAX_OUTPUT: i32 = 65536;
const MAX_ZOOM: i32 = 255;
const MAX_QUERIES: i32 = 8192;
const MAX_PEAKS: i32 = 1024;
const TAU: f64 = 2 * Math.PI;
/** `AnalysisParams::original().collinear_epsilon`. */
const COLLINEAR_EPSILON: f64 = 4e-6;
/** f64 per model: circle flag, center y, radius, three angles, line value, line slope. */
const MODEL: i32 = 8;
/** f64 per AFT interval for FAFT: value, slope, then two arcs × (a, b, rate, cos rate, sin rate). */
const STRIDE: i32 = 12;
/** f64 per AFT interval for evaluation: two halves × (arc flag, center/middle, radius/slope, theta, beta). */
const HALVES: i32 = 10;
/** FAFT's `VALUE_SERIES` and `DERIVATIVE_SERIES`. */
const SERIES: f64 = 0.05;
const MOMENT_SERIES: f64 = 0.5;
/** Support phases are recomputed with sin/cos every this many intervals (as the crate does). */
const ANCHOR: i32 = 64;

const input = memory.data(MAX_INPUT << 3, 16); // f64[MAX_INPUT]
const output = memory.data(MAX_OUTPUT << 3, 16); // f64[MAX_OUTPUT]: resampled samples
const models = memory.data((MAX_INPUT + 1) * MODEL << 3, 16); // f64[(MAX_INPUT + 1) · MODEL]
const halves = memory.data(MAX_INPUT * HALVES << 3, 16); // f64[MAX_INPUT · HALVES]
const atoms = memory.data(MAX_INPUT * STRIDE << 3, 16); // f64[MAX_INPUT · STRIDE]
const periodic = memory.data((MAX_INPUT + 3) << 3, 16); // f64[MAX_INPUT + 3]: the AFT's halo
const omegas = memory.data(MAX_QUERIES << 3, 16); // f64[MAX_QUERIES]
const values = memory.data(2 * MAX_QUERIES << 3, 16); // f64[2 · MAX_QUERIES]: (re, im) or magnitudes
const candidates = memory.data(MAX_PEAKS * 5 << 3, 16); // f64[MAX_PEAKS · 5]: scratch
const peaks = memory.data(MAX_PEAKS * 4 << 3, 16); // f64[MAX_PEAKS · 4]: omega, re, im, curvature
const weights = memory.data((MAX_ZOOM + 1) * 2 << 3, 16); // f64[(MAX_ZOOM + 1) · 2]: the window per step

@inline function get(base: usize, index: i32): f64 {
  return load<f64>(base + (<usize>index << 3));
}

@inline function set(base: usize, index: i32, value: f64): void {
  store<f64>(base + (<usize>index << 3), value);
}

/** Rust's `f64::rem_euclid`. */
@inline function remEuclid(x: f64, rhs: f64): f64 {
  const r = x % rhs;
  return r < 0 ? r + Math.abs(rhs) : r;
}

// ---- circles (Analysis::new, Circle::through, arc_angles) -------------------------------------------

/** Fits model `slot` from three consecutive samples, as `Analysis::new` with the original preset. */
function fitModel(slot: i32, y0: f64, y1: f64, y2: f64): void {
  const m = models + (<usize>(slot * MODEL) << 3);
  const range = Math.max(Math.max(Math.abs(y0 - y1), Math.abs(y1 - y2)), Math.abs(y0 - y2));
  const aperture = range > 0 ? range : 1.0;
  // Points (−aperture, y0), (0, y1), (aperture, y2); the C/Python 2×2 system.
  const a00 = 2.0 * aperture;
  const a01 = 2.0 * (y1 - y0);
  const a10 = 2.0 * aperture;
  const a11 = 2.0 * (y2 - y1);
  const b0 = -(aperture * aperture) + (y1 * y1 - y0 * y0);
  const b1 = aperture * aperture + (y2 * y2 - y1 * y1);
  const det = a00 * a11 - a01 * a10;
  store<f64>(m, 0);
  store<f64>(m, y1, 48);
  store<f64>(m, y2 * 0.5 - y0 * 0.5, 56);
  if (!isFinite(det) || Math.abs(det) < COLLINEAR_EPSILON) return;
  const cx = (b0 * a11 - a01 * b1) / det;
  const cy = (a00 * b1 - b0 * a10) / det;
  const radius = Math.hypot(cx, cy - y1);
  if (!isFinite(cx) || !isFinite(cy) || !isFinite(radius) || radius <= 0) return;
  const t0 = Math.atan2(y0 - cy, -aperture - cx);
  const t1 = Math.atan2(y1 - cy, -cx);
  const t2 = Math.atan2(y2 - cy, aperture - cx);
  const first = remEuclid(t1 - t0, TAU);
  const full = remEuclid(t2 - t0, TAU);
  const direction: f64 = first <= full ? 1 : -1;
  const middle = t0 + direction * remEuclid(direction * (t1 - t0), TAU);
  store<f64>(m, 1);
  store<f64>(m, cy, 8);
  store<f64>(m, radius, 16);
  store<f64>(m, t0, 24);
  store<f64>(m, middle, 32);
  store<f64>(m, middle + direction * remEuclid(direction * (t2 - t1), TAU), 40);
}

@inline function model(slot: i32, field: i32): f64 {
  return get(models, slot * MODEL + field);
}

export function inputPtr(): usize {
  return input;
}

export function outputPtr(): usize {
  return output;
}

export function modelsPtr(): usize {
  return models;
}

export function maxInput(): i32 {
  return MAX_INPUT;
}

export function maxOutput(): i32 {
  return MAX_OUTPUT;
}

export function maxZoom(): i32 {
  return MAX_ZOOM;
}

/** f64 per model in `modelsPtr()`: circle flag, center y, radius, angles ×3, line value, line slope. */
export function modelStride(): i32 {
  return MODEL;
}

// ---- the resampler (resample with ResampleParams::original) -----------------------------------------

/**
 * Fits the models of input[0..length) (model j belongs to sample j + 1; `length − 2` of them) and returns
 * how many, or −1 for a length beyond the buffer or a non-finite sample.
 */
export function swPrepare(length: i32): i32 {
  if (length < 0 || length > MAX_INPUT) return -1;
  for (let i = 0; i < length; i++) {
    if (!isFinite(get(input, i))) return -1;
  }
  for (let j = 0; j + 2 < length; j++) fitModel(j, get(input, j), get(input, j + 1), get(input, j + 2));
  return length < 3 ? 0 : length - 2;
}

/** The output length of a zoom: (length − 1)(zoom + 1) + 1, the input length below two samples. */
export function swOutputLength(length: i32, zoom: i32): i32 {
  return length < 2 ? length : (length - 1) * (zoom + 1) + 1;
}

/** A model's prediction at u = position − anchor ∈ (−1, 1) (`evaluate_angular`, or its line). */
@inline function predict(slot: i32, u: f64): f64 {
  if (model(slot, 0) == 0) return model(slot, 6) + u * model(slot, 7);
  const a = u < 0 ? model(slot, 3) : model(slot, 4);
  const b = u < 0 ? model(slot, 4) : model(slot, 5);
  const t = u < 0 ? u + 1.0 : u;
  return model(slot, 1) + model(slot, 2) * Math.sin(a + t * (b - a));
}

/**
 * Resamples the prepared input (call `swPrepare(length)` first) into output[0..swOutputLength): every
 * original sample copied, `zoom` samples inserted per interval read along equal angles and blended with
 * the original window. `simd` uses the rotation path. Returns the output length, or −1 for a zoom outside
 * 0..MAX_ZOOM or an output beyond the buffer.
 */
export function swResample(length: i32, zoom: i32, simd: bool): i32 {
  if (length < 0 || length > MAX_INPUT || zoom < 0 || zoom > MAX_ZOOM) return -1;
  const total = swOutputLength(length, zoom);
  if (total > MAX_OUTPUT) return -1;
  if (length < 2 || zoom == 0) {
    memory.copy(output, input, <usize>length << 3);
    return total;
  }
  const q = zoom + 1;
  const qf = <f64>q;
  const c = Math.PI * (qf / (qf + 1.0)) * 0.5;
  // The window of the anchor at i (u = t) and at i + 1 (u = t − 1), per step k.
  for (let k = 1; k < q; k++) {
    const t = <f64>k / qf;
    set(weights, 2 * k, Math.cos(c * t));
    set(weights, 2 * k + 1, Math.cos(c * (t - 1.0)));
  }
  let out = 0;
  for (let i = 0; i < length - 1; i++) {
    const left = get(input, i);
    const right = get(input, i + 1);
    set(output, out++, left);
    // Anchors i (when 1 ≤ i ≤ length − 2) and i + 1 (when i + 1 ≤ length − 2): models i − 1 and i.
    const hasLeft = i >= 1 && i <= length - 2;
    const hasRight = i + 1 <= length - 2;
    if (simd && (hasLeft || hasRight)) {
      out = intervalPair(i, q, hasLeft, hasRight, left, right, out);
      continue;
    }
    for (let k = 1; k < q; k++) {
      const t = <f64>k / qf;
      const position = <f64>i + t;
      let sumW: f64 = 0;
      let sumY: f64 = 0;
      for (let anchor = i < 1 ? 1 : i; anchor <= i + 1 && anchor <= length - 2; anchor++) {
        const u = position - <f64>anchor;
        if (Math.abs(u) >= 1.0) continue;
        const y = predict(anchor - 1, u);
        if (!isFinite(y)) continue;
        const w = Math.cos(c * u);
        if (w > 0) {
          sumW += w;
          sumY += w * y;
        }
      }
      const blended = sumY / sumW;
      set(output, out++, sumW > 1e-20 && isFinite(blended) ? blended : (1.0 - t) * left + t * right);
    }
  }
  set(output, out++, get(input, length - 1));
  return out;
}

/**
 * One interval's inserted samples with both covering models in the lanes of an f64x2: lane 0 the right
 * half of the circle at i (angles 1 → 2), lane 1 the left half of the circle at i + 1 (angles 0 → 1).
 * Both sweep their half in q equal angle steps, so the angle advances by a fixed rotation per step.
 */
function intervalPair(i: i32, q: i32, hasLeft: bool, hasRight: bool, left: f64, right: f64, out: i32): i32 {
  const qf = <f64>q;
  const l = i - 1;
  const r = i;
  // Per lane: circle flag, center, radius, start angle, step; or the line's value at t = 0 and slope.
  const leftArc = hasLeft && model(l, 0) != 0;
  const rightArc = hasRight && model(r, 0) != 0;
  const step0 = leftArc ? (model(l, 5) - model(l, 4)) / qf : 0;
  const step1 = rightArc ? (model(r, 4) - model(r, 3)) / qf : 0;
  const start0 = leftArc ? model(l, 4) + step0 : 0;
  const start1 = rightArc ? model(r, 3) + step1 : 0;
  let zre = f64x2.replace_lane(f64x2.splat(Math.cos(start0)), 1, Math.cos(start1));
  let zim = f64x2.replace_lane(f64x2.splat(Math.sin(start0)), 1, Math.sin(start1));
  const dre = f64x2.replace_lane(f64x2.splat(Math.cos(step0)), 1, Math.cos(step1));
  const dim = f64x2.replace_lane(f64x2.splat(Math.sin(step0)), 1, Math.sin(step1));
  const center = f64x2.replace_lane(f64x2.splat(leftArc ? model(l, 1) : 0), 1, rightArc ? model(r, 1) : 0);
  const radius = f64x2.replace_lane(f64x2.splat(leftArc ? model(l, 2) : 0), 1, rightArc ? model(r, 2) : 0);
  // Lines: value + u·slope with u = t (lane 0) and t − 1 (lane 1).
  const lineValue = f64x2.replace_lane(
    f64x2.splat(hasLeft && !leftArc ? model(l, 6) : 0), 1, hasRight && !rightArc ? model(r, 6) - model(r, 7) : 0);
  const lineSlope = f64x2.replace_lane(f64x2.splat(hasLeft && !leftArc ? model(l, 7) : 0), 1, hasRight && !rightArc ? model(r, 7) : 0);
  const arcs = i64x2.replace_lane(i64x2.splat(leftArc ? -1 : 0), 1, rightArc ? -1 : 0);
  const live = f64x2.replace_lane(f64x2.splat(hasLeft ? 1 : 0), 1, hasRight ? 1 : 0);
  for (let k = 1; k < q; k++) {
    const t = <f64>k / qf;
    const arcY = f64x2.add(center, f64x2.mul(radius, zim));
    const lineY = f64x2.add(lineValue, f64x2.mul(lineSlope, f64x2.splat(t)));
    const y = v128.bitselect(arcY, lineY, arcs);
    const w = f64x2.mul(v128.load(weights + (<usize>(2 * k) << 3)), live);
    const products = f64x2.mul(w, y);
    const sumW = f64x2.extract_lane(w, 0) + f64x2.extract_lane(w, 1);
    const blended = (f64x2.extract_lane(products, 0) + f64x2.extract_lane(products, 1)) / sumW;
    set(output, out++, sumW > 1e-20 && isFinite(blended) ? blended : (1.0 - t) * left + t * right);
    const next = f64x2.sub(f64x2.mul(zre, dre), f64x2.mul(zim, dim));
    zim = f64x2.add(f64x2.mul(zre, dim), f64x2.mul(zim, dre));
    zre = next;
  }
  return out;
}

// ---- the AFT (AngularFrame) ---------------------------------------------------------------------------

let intervals: i32 = 0;
let circleHalves: i32 = 0;
let lineHalves: i32 = 0;
let rejectedHalves: i32 = 0;

/** Half `side` of interval n for evaluation: an arc (center, radius, theta, beta) or a line. */
@inline function half(n: i32, side: i32, field: i32): f64 {
  return get(halves, n * HALVES + side * 5 + field);
}

@inline function atom(n: i32, field: i32): f64 {
  return get(atoms, n * STRIDE + field);
}

@inline function setAtom(n: i32, field: i32, value: f64): void {
  set(atoms, n * STRIDE + field, value);
}

@inline function halfValue(n: i32, side: i32, u: f64): f64 {
  if (half(n, side, 0) != 0) return half(n, side, 1) + half(n, side, 2) * Math.sin(half(n, side, 3) + half(n, side, 4) * u);
  return half(n, side, 1) + half(n, side, 2) * u;
}

/**
 * Prepares the AFT of input[0..length) (`AngularFrame::new` with the original analysis): the periodic
 * halo, the models, both halves of every interval (N intervals, the last closing to the first sample),
 * and their FAFT atoms (`CircleCurve::from_angular`, the window 0..N). Returns N, or −1 for an empty or
 * oversized frame or a non-finite sample.
 */
export function aftPrepare(length: i32): i32 {
  intervals = 0;
  circleHalves = 0;
  lineHalves = 0;
  rejectedHalves = 0;
  if (length < 1 || length > MAX_INPUT) return -1;
  for (let i = 0; i < length; i++) {
    if (!isFinite(get(input, i))) return -1;
  }
  set(periodic, 0, get(input, length - 1));
  memory.copy(periodic + 8, input, <usize>length << 3);
  set(periodic, length + 1, get(input, 0));
  set(periodic, length + 2, get(input, 1 % length));
  for (let j = 0; j <= length; j++) fitModel(j, get(periodic, j), get(periodic, j + 1), get(periodic, j + 2));
  memory.fill(atoms, 0, <usize>(length * STRIDE) << 3);
  for (let n = 0; n < length; n++) {
    const left = get(input, n);
    const right = get(input, (n + 1) % length);
    const middle = left * 0.5 + right * 0.5;
    const slope = right - left;
    for (let side = 0; side < 2; side++) {
      const base = n * HALVES + side * 5;
      const slot = n + side;
      let arc = false;
      if (model(slot, 0) != 0) {
        // The circle at sample n gives its right half (angles 1 → 2), the one at n + 1 its left half.
        const index = 1 - side;
        const beta = model(slot, 4 + index) - model(slot, 3 + index);
        const theta = model(slot, 3 + index) + beta * 0.5;
        const cy = model(slot, 1);
        const radius = model(slot, 2);
        const bound = 1e-8 * Math.max(Math.max(Math.abs(left), Math.abs(right)), 1.0);
        if (Math.abs(cy + radius * Math.sin(theta - beta * 0.5) - left) <= bound
          && Math.abs(cy + radius * Math.sin(theta + beta * 0.5) - right) <= bound) {
          arc = true;
          set(halves, base, 1);
          set(halves, base + 1, cy);
          set(halves, base + 2, radius);
          set(halves, base + 3, theta);
          set(halves, base + 4, beta);
          // The rotation center/2 + radius/2·sin(phase + beta·u), u ∈ [0, 1), in its anchored form.
          const phase = theta - beta * 0.5;
          const s = Math.sin(phase);
          const co = Math.cos(phase);
          setAtom(n, 0, atom(n, 0) + cy * 0.5 + radius * 0.5 * s);
          if (beta != 0) {
            const at = 2 + side * 5;
            setAtom(n, at, radius * 0.5 * s);
            setAtom(n, at + 1, radius * 0.5 * co);
            setAtom(n, at + 2, beta);
            setAtom(n, at + 3, Math.cos(beta));
            setAtom(n, at + 4, Math.sin(beta));
          }
          circleHalves++;
        } else {
          rejectedHalves++;
        }
      }
      if (!arc) {
        lineHalves++;
        set(halves, base, 0);
        set(halves, base + 1, middle);
        set(halves, base + 2, slope);
        setAtom(n, 0, atom(n, 0) + (middle - slope * 0.5) * 0.5);
        setAtom(n, 1, atom(n, 1) + slope * 0.5);
      }
    }
  }
  intervals = length;
  return intervals;
}

export function aftIntervals(): i32 {
  return intervals;
}

export function aftCircleHalves(): i32 {
  return circleHalves;
}

/** Line halves, rejected circles included (`AngularStats::line_halves`). */
export function aftLineHalves(): i32 {
  return lineHalves;
}

export function aftRejectedCircleHalves(): i32 {
  return rejectedHalves;
}

export function aftAtomsPtr(): usize {
  return atoms;
}

export function aftAtomStride(): i32 {
  return STRIDE;
}

export function aftOmegasPtr(): usize {
  return omegas;
}

export function aftValuesPtr(): usize {
  return values;
}

export function aftPeaksPtr(): usize {
  return peaks;
}

export function aftMaxQueries(): i32 {
  return MAX_QUERIES;
}

export function aftMaxPeaks(): i32 {
  return MAX_PEAKS;
}

/** The AFT's curve at any real position, periodically extended (`AngularFrame::evaluate`); NaN unprepared. */
export function aftEvaluate(position: f64): f64 {
  if (intervals == 0 || !isFinite(position)) return NaN;
  const t = remEuclid(position, <f64>intervals);
  const n = min(<i32>Math.floor(t), intervals - 1);
  const u = t - <f64>n - 0.5;
  return halfValue(n, 0, u) * 0.5 + halfValue(n, 1, u) * 0.5;
}

// ---- FAFT: the rectangular transform ----------------------------------------------------------------

let e0re: f64 = 0;
let e0im: f64 = 0;

/** E₀(δ) = ∫₀¹ e^{iδu} du given e = e^{iδ}, or its series near δ = 0: into e0re/e0im. */
@inline function e0(delta: f64, ere: f64, eim: f64): void {
  if (Math.abs(delta) < SERIES) {
    const d2 = delta * delta;
    e0re = 1 + d2 * (-1.0 / 6 + d2 * (1.0 / 120 + d2 * (-1.0 / 5040 + d2 * (1.0 / 362880))));
    e0im = delta * (0.5 + d2 * (-1.0 / 24 + d2 * (1.0 / 720 + d2 * (-1.0 / 40320 + d2 * (1.0 / 3628800)))));
  } else {
    e0re = eim / delta;
    e0im = (1 - ere) / delta;
  }
}

let lineRe: f64 = 0;
let lineIm: f64 = 0;

/** E₁(−ω) = ∫₀¹ u e^{−iωu} du (the slope's term), into lineRe/lineIm. */
function line(omega: f64): void {
  if (Math.abs(omega) < SERIES) {
    const d2 = omega * omega;
    lineRe = 0.5 + d2 * (-1.0 / 8 + d2 * (1.0 / 144 + d2 * (-1.0 / 5760 + d2 * (1.0 / 403200))));
    lineIm = -omega * (1.0 / 3 + d2 * (-1.0 / 30 + d2 * (1.0 / 840 + d2 * (-1.0 / 45360 + d2 * (1.0 / 3991680)))));
    return;
  }
  const qre = Math.cos(omega);
  const qim = -Math.sin(omega);
  e0(-omega, qre, qim);
  // E₁ = −i(e − E₀)/δ with δ = −ω.
  lineRe = (qim - e0im) / -omega;
  lineIm = -(qre - e0re) / -omega;
}

let rectRe: f64 = 0;
let rectIm: f64 = 0;

/** The unnormalized rectangular integral at ω, phase origin at the window start: into rectRe/rectIm. */
function rect(omega: f64): void {
  const qre = Math.cos(omega);
  const qim = -Math.sin(omega);
  e0(-omega, qre, qim);
  const dcRe = e0re;
  const dcIm = e0im;
  line(omega);
  let sre: f64 = 1;
  let sim: f64 = 0;
  let sumRe: f64 = 0;
  let sumIm: f64 = 0;
  for (let n = 0; n < intervals; n++) {
    const y0 = atom(n, 0);
    const slope = atom(n, 1);
    let pre = y0 * dcRe + slope * lineRe;
    let pim = y0 * dcIm + slope * lineIm;
    for (let slot = 0; slot < 2; slot++) {
      const base = 2 + slot * 5;
      const rate = atom(n, base + 2);
      if (rate == 0) continue;
      const a = atom(n, base);
      const b = atom(n, base + 1);
      const cr = atom(n, base + 3);
      const sr = atom(n, base + 4);
      const up = rate - omega;
      const down = -rate - omega;
      e0(up, cr * qre - sr * qim, cr * qim + sr * qre);
      const upRe = e0re;
      const upIm = e0im;
      e0(down, cr * qre + sr * qim, cr * qim - sr * qre);
      // cos(r·u) − 1 and sin(r·u) transforms.
      const cosRe = 0.5 * (upRe + e0re) - dcRe;
      const cosIm = 0.5 * (upIm + e0im) - dcIm;
      const sinRe = 0.5 * (upIm - e0im);
      const sinIm = -0.5 * (upRe - e0re);
      pre += a * cosRe + b * sinRe;
      pim += a * cosIm + b * sinIm;
    }
    sumRe += sre * pre - sim * pim;
    sumIm += sre * pim + sim * pre;
    if ((n + 1) % ANCHOR == 0) {
      sre = Math.cos(omega * <f64>(n + 1));
      sim = -Math.sin(omega * <f64>(n + 1));
    } else {
      const next = sre * qre - sim * qim;
      sim = sre * qim + sim * qre;
      sre = next;
    }
  }
  rectRe = sumRe;
  rectIm = sumIm;
}

let pairRe: v128 = f64x2.splat(0);
let pairIm: v128 = f64x2.splat(0);
let e0PairRe: v128 = f64x2.splat(0);
let e0PairIm: v128 = f64x2.splat(0);

/** E₀ in both lanes; the series is computed (and selected per lane) only where some |δ| < SERIES. */
@inline function e0Pair(delta: v128, ere: v128, eim: v128): void {
  const inverse = f64x2.div(f64x2.splat(1), delta);
  let re = f64x2.mul(eim, inverse);
  let im = f64x2.mul(f64x2.sub(f64x2.splat(1), ere), inverse);
  const small = f64x2.lt(f64x2.abs(delta), f64x2.splat(SERIES));
  if (v128.any_true(small)) {
    const d2 = f64x2.mul(delta, delta);
    const sRe = f64x2.add(f64x2.splat(1), f64x2.mul(d2, f64x2.add(f64x2.splat(-1.0 / 6), f64x2.mul(d2, f64x2.add(f64x2.splat(1.0 / 120),
      f64x2.mul(d2, f64x2.add(f64x2.splat(-1.0 / 5040), f64x2.mul(d2, f64x2.splat(1.0 / 362880)))))))));
    const sIm = f64x2.mul(delta, f64x2.add(f64x2.splat(0.5), f64x2.mul(d2, f64x2.add(f64x2.splat(-1.0 / 24), f64x2.mul(d2, f64x2.add(f64x2.splat(1.0 / 720),
      f64x2.mul(d2, f64x2.add(f64x2.splat(-1.0 / 40320), f64x2.mul(d2, f64x2.splat(1.0 / 3628800))))))))));
    re = v128.bitselect(sRe, re, small);
    im = v128.bitselect(sIm, im, small);
  }
  e0PairRe = re;
  e0PairIm = im;
}

@inline function pair(a: f64, b: f64): v128 {
  return f64x2.replace_lane(f64x2.splat(a), 1, b);
}

/** The unnormalized rectangular integrals at ω₀ and ω₁ (lanes 0 and 1): into pairRe/pairIm. */
function rectPair(omega0: f64, omega1: f64): void {
  const omega = pair(omega0, omega1);
  const qre = pair(Math.cos(omega0), Math.cos(omega1));
  const qim = pair(-Math.sin(omega0), -Math.sin(omega1));
  e0Pair(f64x2.neg(omega), qre, qim);
  const dcRe = e0PairRe;
  const dcIm = e0PairIm;
  line(omega0);
  const l0re = lineRe;
  const l0im = lineIm;
  line(omega1);
  const lRe = pair(l0re, lineRe);
  const lIm = pair(l0im, lineIm);
  const half = f64x2.splat(0.5);
  let sre = f64x2.splat(1);
  let sim = f64x2.splat(0);
  let sumRe = f64x2.splat(0);
  let sumIm = f64x2.splat(0);
  for (let n = 0; n < intervals; n++) {
    const y0 = f64x2.splat(atom(n, 0));
    const slope = f64x2.splat(atom(n, 1));
    let pre = f64x2.add(f64x2.mul(y0, dcRe), f64x2.mul(slope, lRe));
    let pim = f64x2.add(f64x2.mul(y0, dcIm), f64x2.mul(slope, lIm));
    for (let slot = 0; slot < 2; slot++) {
      const base = 2 + slot * 5;
      const rs = atom(n, base + 2);
      if (rs == 0) continue;
      const rate = f64x2.splat(rs);
      const cr = f64x2.splat(atom(n, base + 3));
      const sr = f64x2.splat(atom(n, base + 4));
      e0Pair(f64x2.sub(rate, omega), f64x2.sub(f64x2.mul(cr, qre), f64x2.mul(sr, qim)), f64x2.add(f64x2.mul(cr, qim), f64x2.mul(sr, qre)));
      const upRe = e0PairRe;
      const upIm = e0PairIm;
      e0Pair(f64x2.sub(f64x2.neg(rate), omega), f64x2.add(f64x2.mul(cr, qre), f64x2.mul(sr, qim)), f64x2.sub(f64x2.mul(cr, qim), f64x2.mul(sr, qre)));
      const cosRe = f64x2.sub(f64x2.mul(half, f64x2.add(upRe, e0PairRe)), dcRe);
      const cosIm = f64x2.sub(f64x2.mul(half, f64x2.add(upIm, e0PairIm)), dcIm);
      const sinRe = f64x2.mul(half, f64x2.sub(upIm, e0PairIm));
      const sinIm = f64x2.neg(f64x2.mul(half, f64x2.sub(upRe, e0PairRe)));
      const a = f64x2.splat(atom(n, base));
      const b = f64x2.splat(atom(n, base + 1));
      pre = f64x2.add(pre, f64x2.add(f64x2.mul(a, cosRe), f64x2.mul(b, sinRe)));
      pim = f64x2.add(pim, f64x2.add(f64x2.mul(a, cosIm), f64x2.mul(b, sinIm)));
    }
    sumRe = f64x2.add(sumRe, f64x2.sub(f64x2.mul(sre, pre), f64x2.mul(sim, pim)));
    sumIm = f64x2.add(sumIm, f64x2.add(f64x2.mul(sre, pim), f64x2.mul(sim, pre)));
    if ((n + 1) % ANCHOR == 0) {
      const t = <f64>(n + 1);
      sre = pair(Math.cos(omega0 * t), Math.cos(omega1 * t));
      sim = pair(-Math.sin(omega0 * t), -Math.sin(omega1 * t));
    } else {
      const next = f64x2.sub(f64x2.mul(sre, qre), f64x2.mul(sim, qim));
      sim = f64x2.add(f64x2.mul(sre, qim), f64x2.mul(sim, qre));
      sre = next;
    }
  }
  pairRe = sumRe;
  pairIm = sumIm;
}

// ---- FAFT: values and scans ------------------------------------------------------------------------

/**
 * values[2k], values[2k + 1] = the transform at omegas[k] (radians per sample, any real value), phase
 * origin at the window start. `normalized` divides by the weight sum (N, or N/2 for Hann: `Faft::values`);
 * otherwise the integral itself (`Faft::integral`; rectangular = `AngularFrame::fourier`). `simd` takes
 * two frequencies at a time. Returns false without a prepared frame, beyond the buffers, or for a
 * non-finite frequency.
 */
export function aftValues(count: i32, hann: bool, normalized: bool, simd: bool): bool {
  if (intervals == 0 || count < 0 || count > MAX_QUERIES) return false;
  for (let k = 0; k < count; k++) {
    if (!isFinite(get(omegas, k))) return false;
  }
  const duration = <f64>intervals;
  const shift = TAU / duration;
  const norm = normalized ? (hann ? 2 / duration : 1 / duration) : 1.0;
  let k = 0;
  if (simd) {
    for (; k + 1 < count; k += 2) {
      const w0 = get(omegas, k);
      const w1 = get(omegas, k + 1);
      rectPair(w0, w1);
      let re = pairRe;
      let im = pairIm;
      if (hann) {
        rectPair(w0 - shift, w1 - shift);
        const lRe = pairRe;
        const lIm = pairIm;
        rectPair(w0 + shift, w1 + shift);
        re = f64x2.sub(f64x2.mul(re, f64x2.splat(0.5)), f64x2.mul(f64x2.add(lRe, pairRe), f64x2.splat(0.25)));
        im = f64x2.sub(f64x2.mul(im, f64x2.splat(0.5)), f64x2.mul(f64x2.add(lIm, pairIm), f64x2.splat(0.25)));
      }
      set(values, 2 * k, f64x2.extract_lane(re, 0) * norm);
      set(values, 2 * k + 1, f64x2.extract_lane(im, 0) * norm);
      set(values, 2 * k + 2, f64x2.extract_lane(re, 1) * norm);
      set(values, 2 * k + 3, f64x2.extract_lane(im, 1) * norm);
    }
  }
  for (; k < count; k++) {
    const w = get(omegas, k);
    rect(w);
    let re = rectRe;
    let im = rectIm;
    if (hann) {
      rect(w - shift);
      const lRe = rectRe;
      const lIm = rectIm;
      rect(w + shift);
      re = 0.5 * re - 0.25 * (lRe + rectRe);
      im = 0.5 * im - 0.25 * (lIm + rectIm);
    }
    set(values, 2 * k, re * norm);
    set(values, 2 * k + 1, im * norm);
  }
  return true;
}

/**
 * Normalized magnitudes on the grid lower + k·step, k < count, into values[0..count). With `density` > 0
 * the Hann window applies (step must be (2π/N)/density: its ±2π/N neighbours share one extended
 * rectangular grid, as `Faft::peaks` scans). Returns false without a frame or beyond the buffers.
 */
export function aftScan(lower: f64, step: f64, count: i32, density: i32, simd: bool): bool {
  const pad = density;
  const total = count + 2 * pad;
  if (intervals == 0 || count < 1 || density < 0 || total > MAX_QUERIES || !isFinite(lower) || !(step > 0)) return false;
  let k = 0;
  if (simd) {
    for (; k + 1 < total; k += 2) {
      rectPair(lower + step * <f64>(k - pad), lower + step * <f64>(k + 1 - pad));
      set(values, 2 * k, f64x2.extract_lane(pairRe, 0));
      set(values, 2 * k + 1, f64x2.extract_lane(pairIm, 0));
      set(values, 2 * k + 2, f64x2.extract_lane(pairRe, 1));
      set(values, 2 * k + 3, f64x2.extract_lane(pairIm, 1));
    }
  }
  for (; k < total; k++) {
    rect(lower + step * <f64>(k - pad));
    set(values, 2 * k, rectRe);
    set(values, 2 * k + 1, rectIm);
  }
  const norm = pad > 0 ? 2 / <f64>intervals : 1 / <f64>intervals;
  // Front to back: magnitude j only reads complex values at indices ≥ j.
  for (let j = 0; j < count; j++) {
    let re = get(values, 2 * (j + pad));
    let im = get(values, 2 * (j + pad) + 1);
    if (pad > 0) {
      re = 0.5 * re - 0.25 * (get(values, 2 * j) + get(values, 2 * (j + 2 * pad)));
      im = 0.5 * im - 0.25 * (get(values, 2 * j + 1) + get(values, 2 * (j + 2 * pad) + 1));
    }
    set(values, j, Math.sqrt(re * re + im * im) * norm);
  }
  return true;
}

// ---- FAFT: derivatives and peaks -------------------------------------------------------------------

const moments = memory.data(64, 16); // ∫₀¹ uᵏ e^{iδu} du, k = 0..3: (re, im)
const derivative = memory.data(144, 16); // three Hann shifts × F, F′, F″ (re, im)

function momentsAt(delta: f64, ere: f64, eim: f64, orders: i32): void {
  if (Math.abs(delta) < MOMENT_SERIES) {
    // Σₙ (iδ)ⁿ / (n!·(n + k + 1)); 20 terms (|δ|ⁿ/n! < 1e-18 at |δ| < 0.5).
    for (let k = 0; k < orders; k++) {
      let re: f64 = 0;
      let im: f64 = 0;
      let termRe: f64 = 1;
      let termIm: f64 = 0;
      for (let n = 0; n < 20; n++) {
        const scale: f64 = 1 / <f64>(n + k + 1);
        re += termRe * scale;
        im += termIm * scale;
        const nextRe = -termIm * delta / <f64>(n + 1);
        termIm = termRe * delta / <f64>(n + 1);
        termRe = nextRe;
      }
      set(moments, 2 * k, re);
      set(moments, 2 * k + 1, im);
    }
    return;
  }
  // M₀ = (e − 1)/(iδ); Mₖ = (e − k·Mₖ₋₁)/(iδ).
  let mre = eim / delta;
  let mim = (1 - ere) / delta;
  set(moments, 0, mre);
  set(moments, 1, mim);
  for (let k = 1; k < orders; k++) {
    const dre = ere - <f64>k * mre;
    const dim = eim - <f64>k * mim;
    mre = dim / delta;
    mim = -dre / delta;
    set(moments, 2 * k, mre);
    set(moments, 2 * k + 1, mim);
  }
}

/** The rectangular integral and its first two ω-derivatives, phase origin at the window centre. */
function rectDerivatives(omega: f64, slot: i32): void {
  const qre = Math.cos(omega);
  const qim = -Math.sin(omega);
  const center = 0.5 * <f64>intervals;
  momentsAt(-omega, qre, qim, 4);
  const dc0re = get(moments, 0), dc0im = get(moments, 1);
  const dc1re = get(moments, 2), dc1im = get(moments, 3);
  const dc2re = get(moments, 4), dc2im = get(moments, 5);
  const dc3re = get(moments, 6), dc3im = get(moments, 7);
  let sre = Math.cos(omega * center);
  let sim = Math.sin(omega * center);
  let f0re: f64 = 0, f0im: f64 = 0, f1re: f64 = 0, f1im: f64 = 0, f2re: f64 = 0, f2im: f64 = 0;
  for (let n = 0; n < intervals; n++) {
    const y0 = atom(n, 0);
    const slope = atom(n, 1);
    let p0re = y0 * dc0re + slope * dc1re, p0im = y0 * dc0im + slope * dc1im;
    let p1re = y0 * dc1re + slope * dc2re, p1im = y0 * dc1im + slope * dc2im;
    let p2re = y0 * dc2re + slope * dc3re, p2im = y0 * dc2im + slope * dc3im;
    for (let s = 0; s < 2; s++) {
      const base = 2 + s * 5;
      const rate = atom(n, base + 2);
      if (rate == 0) continue;
      const a = atom(n, base);
      const b = atom(n, base + 1);
      const cr = atom(n, base + 3);
      const sr = atom(n, base + 4);
      momentsAt(rate - omega, cr * qre - sr * qim, cr * qim + sr * qre, 3);
      const u0re = get(moments, 0), u0im = get(moments, 1);
      const u1re = get(moments, 2), u1im = get(moments, 3);
      const u2re = get(moments, 4), u2im = get(moments, 5);
      momentsAt(-rate - omega, cr * qre + sr * qim, cr * qim - sr * qre, 3);
      const d0re = get(moments, 0), d0im = get(moments, 1);
      const d1re = get(moments, 2), d1im = get(moments, 3);
      const d2re = get(moments, 4), d2im = get(moments, 5);
      // a·((U + D)/2 − DC) + b·(−i)(U − D)/2, per order.
      p0re += a * (0.5 * (u0re + d0re) - dc0re) + b * 0.5 * (u0im - d0im);
      p0im += a * (0.5 * (u0im + d0im) - dc0im) - b * 0.5 * (u0re - d0re);
      p1re += a * (0.5 * (u1re + d1re) - dc1re) + b * 0.5 * (u1im - d1im);
      p1im += a * (0.5 * (u1im + d1im) - dc1im) - b * 0.5 * (u1re - d1re);
      p2re += a * (0.5 * (u2re + d2re) - dc2re) + b * 0.5 * (u2im - d2im);
      p2im += a * (0.5 * (u2im + d2im) - dc2im) - b * 0.5 * (u2re - d2re);
    }
    // d/dω of e^{−iω(σ + u)} contributes −i(σ + u); σ = n − centre.
    const sigma = <f64>n - center;
    const t1re = sigma * p0re + p1re, t1im = sigma * p0im + p1im;
    const v1re = t1im, v1im = -t1re;
    const v2re = -(sigma * sigma * p0re + 2 * sigma * p1re + p2re);
    const v2im = -(sigma * sigma * p0im + 2 * sigma * p1im + p2im);
    f0re += sre * p0re - sim * p0im; f0im += sre * p0im + sim * p0re;
    f1re += sre * v1re - sim * v1im; f1im += sre * v1im + sim * v1re;
    f2re += sre * v2re - sim * v2im; f2im += sre * v2im + sim * v2re;
    if ((n + 1) % ANCHOR == 0) {
      const t = <f64>(n + 1) - center;
      sre = Math.cos(omega * t);
      sim = -Math.sin(omega * t);
    } else {
      const next = sre * qre - sim * qim;
      sim = sre * qim + sim * qre;
      sre = next;
    }
  }
  const out = derivative + (<usize>(slot * 6) << 3);
  store<f64>(out, f0re); store<f64>(out, f0im, 8);
  store<f64>(out, f1re, 16); store<f64>(out, f1im, 24);
  store<f64>(out, f2re, 32); store<f64>(out, f2im, 40);
}

@inline function at(slot: i32, field: i32): f64 {
  return get(derivative, slot * 6 + field);
}

let c0re: f64 = 0, c0im: f64 = 0, c1re: f64 = 0, c1im: f64 = 0, c2re: f64 = 0, c2im: f64 = 0;

/** Normalized centred F, F′, F″ (`Faft::derivatives`) into c0…c2. */
function centred(omega: f64, hann: bool): void {
  const duration = <f64>intervals;
  rectDerivatives(omega, 0);
  let w0: f64 = 1, w1: f64 = 0;
  if (hann) {
    // Centred periodic Hann: ½F(ω) + ¼F(ω − Δ) + ¼F(ω + Δ).
    const shift = TAU / duration;
    rectDerivatives(omega - shift, 1);
    rectDerivatives(omega + shift, 2);
    w0 = 0.5;
    w1 = 0.25;
  }
  const norm = hann ? 2 / duration : 1 / duration;
  c0re = (w0 * at(0, 0) + w1 * (at(1, 0) + at(2, 0))) * norm;
  c0im = (w0 * at(0, 1) + w1 * (at(1, 1) + at(2, 1))) * norm;
  c1re = (w0 * at(0, 2) + w1 * (at(1, 2) + at(2, 2))) * norm;
  c1im = (w0 * at(0, 3) + w1 * (at(1, 3) + at(2, 3))) * norm;
  c2re = (w0 * at(0, 4) + w1 * (at(1, 4) + at(2, 4))) * norm;
  c2im = (w0 * at(0, 5) + w1 * (at(1, 5) + at(2, 5))) * norm;
}

/** Normalized centred F, F′, F″ at ω into values[0..6) (re, im each). False without a frame. */
export function aftDerivatives(omega: f64, hann: bool): bool {
  if (intervals == 0 || !isFinite(omega)) return false;
  centred(omega, hann);
  set(values, 0, c0re); set(values, 1, c0im);
  set(values, 2, c1re); set(values, 3, c1im);
  set(values, 4, c2re); set(values, 5, c2im);
  return true;
}

let refinedOmega: f64 = 0;

/** `Faft::refine`: 1 for a converged interior maximum, 0 for a flank, −1 without convergence. */
function refine(start: f64, lo: f64, hi: f64, hann: bool, tolerance: f64, maxIterations: i32): i32 {
  let low = lo;
  let high = hi;
  let omega = Math.min(high, Math.max(low, start));
  centred(omega, hann);
  let iterations = 0;
  let converged = false;
  while (true) {
    const g1 = 2 * (c0re * c1re + c0im * c1im);
    const g2 = 2 * (c1re * c1re + c1im * c1im + c0re * c2re + c0im * c2im);
    if (g1 > 0) low = Math.max(low, omega);
    else if (g1 < 0) high = Math.min(high, omega);
    const newton = omega - g1 / g2;
    const next = g2 < 0 && isFinite(newton) && newton >= low && newton <= high ? newton : 0.5 * (low + high);
    if (g1 == 0 || Math.abs(next - omega) <= tolerance) {
      converged = g2 <= 0;
      break;
    }
    if (iterations == maxIterations || high - low <= tolerance) break;
    omega = next;
    iterations++;
    centred(omega, hann);
  }
  refinedOmega = omega;
  const near = 4 * tolerance;
  if ((Math.abs(omega - lo) <= near || Math.abs(omega - hi) <= near) && omega != 0) return 0;
  return converged ? 1 : -1;
}

/**
 * `Faft::peaks` with scan seeds (no circle seeds): a scan of `density` points per 2π/N over [lower, upper]
 * (grid clamped to upper), its local maxima at or over the floor (max(absoluteFloor, relativeFloor ×
 * strongest)) as seeds started by a parabola on log magnitude, each refined with exact derivatives to
 * `tolerance` within `maxIterations`; flanks and unconverged seeds dropped; strongest first, peaks within
 * `mergeWidth` (a quarter of 2π/N when negative) of a kept one merged; at most `maxPeaks`; sorted by ω.
 * peaks[4k ..] = ω, value (re, im; phase origin at the window start), d²|F|²/dω². Returns the count, or −1.
 */
export function aftPeaks(lower: f64, upper: f64, density: i32, hann: bool, relativeFloor: f64, absoluteFloor: f64,
  mergeWidth: f64, maxPeaks: i32, maxIterations: i32, tolerance: f64): i32 {
  if (intervals == 0 || !isFinite(lower) || !isFinite(upper) || !(lower < upper) || density < 1 || density > 64
    || !(relativeFloor >= 0 && relativeFloor <= 1) || !(absoluteFloor >= 0) || !isFinite(absoluteFloor) || !isFinite(mergeWidth)
    || maxPeaks < 1 || maxIterations < 0 || maxIterations > 256 || !(tolerance > 0) || !isFinite(tolerance)) {
    return -1;
  }
  const duration = <f64>intervals;
  const resolution = TAU / duration;
  const merge = mergeWidth < 0 ? 0.25 * resolution : mergeWidth;
  const step = resolution / <f64>density;
  const count = <i32>Math.floor((upper - lower) / step) + 1;
  const pad = hann ? density : 0;
  if (count + 2 * pad > MAX_QUERIES) return -1;
  if (!aftScan(lower, step, count, pad, true)) return -1;
  let strongestScan: f64 = 0;
  for (let k = 0; k < count; k++) strongestScan = Math.max(strongestScan, get(values, k));
  const scanFloor = Math.max(absoluteFloor, relativeFloor * strongestScan);
  let found = 0;
  for (let k = 0; k < count; k++) {
    const m = get(values, k);
    const left = k > 0 ? get(values, k - 1) : -Infinity;
    const right = k + 1 < count ? get(values, k + 1) : -Infinity;
    if (!(m >= scanFloor && m > 0 && m >= left && m > right)) continue;
    if (found == MAX_PEAKS) return -1;
    let start = Math.min(lower + step * <f64>k, upper);
    if (k > 0 && k + 1 < count && left > 0 && right > 0) {
      const l = Math.log(left);
      const c = Math.log(m);
      const r = Math.log(right);
      const bend = l - 2 * c + r;
      if (bend < 0) start += Math.min(step, Math.max(-step, (0.5 * (l - r) / bend) * step));
    }
    const lo = Math.max(Math.min(lower + step * <f64>(k - 1), upper), lower);
    const hi = Math.min(lower + step * <f64>(k + 1), upper);
    if (refine(start, lo, hi, hann, tolerance, maxIterations) != 1) continue;
    // Back to a window-start phase origin: multiply by e^{−iωN/2}.
    const turn = -refinedOmega * duration * 0.5;
    const base = found * 5;
    set(candidates, base, refinedOmega);
    set(candidates, base + 1, Math.sqrt(c0re * c0re + c0im * c0im));
    set(candidates, base + 2, 2 * (c1re * c1re + c1im * c1im + c0re * c2re + c0im * c2im));
    set(candidates, base + 3, c0re * Math.cos(turn) - c0im * Math.sin(turn));
    set(candidates, base + 4, c0re * Math.sin(turn) + c0im * Math.cos(turn));
    found++;
  }
  // Strongest first: a candidate is kept unless under the floor or within the merge width of a kept one.
  let strongest = strongestScan;
  for (let i = 0; i < found; i++) strongest = Math.max(strongest, get(candidates, i * 5 + 1));
  const floor = Math.max(absoluteFloor, relativeFloor * strongest);
  let kept = 0;
  for (let round = 0; round < found; round++) {
    let pick = -1;
    for (let i = 0; i < found; i++) {
      const m = get(candidates, i * 5 + 1);
      if (m >= 0 && (pick < 0 || m > get(candidates, pick * 5 + 1))) pick = i;
    }
    const m = get(candidates, pick * 5 + 1);
    const omega = get(candidates, pick * 5);
    set(candidates, pick * 5 + 1, -1);
    if (m < floor || m == 0 || kept == maxPeaks || kept == MAX_PEAKS) continue;
    let near = false;
    for (let j = 0; j < kept; j++) {
      if (Math.abs(get(peaks, j * 4) - omega) <= merge) near = true;
    }
    if (near) continue;
    set(peaks, kept * 4, omega);
    set(peaks, kept * 4 + 1, get(candidates, pick * 5 + 3));
    set(peaks, kept * 4 + 2, get(candidates, pick * 5 + 4));
    set(peaks, kept * 4 + 3, get(candidates, pick * 5 + 2));
    kept++;
  }
  for (let i = 1; i < kept; i++) {
    for (let j = i; j > 0 && get(peaks, (j - 1) * 4) > get(peaks, j * 4); j--) {
      for (let f = 0; f < 4; f++) {
        const t = get(peaks, (j - 1) * 4 + f);
        set(peaks, (j - 1) * 4 + f, get(peaks, j * 4 + f));
        set(peaks, j * 4 + f, t);
      }
    }
  }
  return kept;
}
