// Transposed from appx/dsp at ec84edcd331af8810be9df42cf5d16f3a348d6d5 (2026-10-05).
// FAFT, the Fast Angular Fourier Transform (R. Cecchini, sinwhisky.rs): the Fourier transform, in closed
// form at any real frequency, of the curve SinWhisky's circles draw through the samples. There is no bin
// grid tied to the sample count, no sampling, no FFT and no quadrature.
//
// Through every three consecutive samples passes one circle (x scaled by the triplet's range, as
// SinWhisky does); read at equal angles, its arc over one sample interval is a rotation
// y(u) = y_start + a·(cos(r·u) − 1) + b·sin(r·u), 0 ≤ u < 1, where r is the angle the arc sweeps. Each
// interval averages the two circles that cover it (the angular frame of sinwhisky.rs's AFT; one at the
// ends, the samples' line where a triplet is straight or flat), so the curve passes through every
// sample. Its transform over the window [0, n − 1) is a sum of closed-form rotation integrals,
// E(δ) = ∫₀¹ e^{iδu} du = (e^{iδ} − 1)/(iδ), each multiplied by the interval's support phase e^{−iωk}:
// "rotate and add", with the support phase chained by one complex product per interval (re-anchored
// every 64). Each arc is evaluated in endpoint form (sinwhisky.rs, commit 2e1940f): its deviation f from
// the start value obeys f'' = −r²(f + a), so its transform is
//   [f'(1)·e^{−iω} − f'(0) + iω·f(1)·e^{−iω} + r²a·E₀(−ω)] / ((ω − r)(ω + r)),
// a few real products and one division from endpoint data prepared once; every term has the size of the
// deviation, so flat arcs keep relative accuracy. Within |ω ∓ r| < 0.5, where that quotient is 0/0, the
// moment form (E₀ at ±r − ω, e^{±ir} prepared once) takes over; near DC or resonance (|δ| < 0.05) E uses
// its series.
//
// A query costs O(intervals); the SIMD path evaluates two frequencies per v128 lane pair. A dense,
// uniform spectrum is not faster than an FFT of the samples (which transforms a different, sampled
// target); FAFT's use is exact values between bins and Newton-refined continuous peaks, with exact
// derivatives (frequency, amplitude and phase of a tone from a short window: a pilot's Doppler).
// Static buffers only, like the other kernels.

const MAX_INPUT: i32 = 4096;
const MAX_QUERIES: i32 = 8192;
const MAX_PEAKS: i32 = 256;
/**
 * f64 per interval: y0, slope, then two arcs × ARC: a, b, rate, cos rate, sin rate (the moment form), and
 * the endpoint data f'(0), f'(1), f(1) and r²·a of the arc's deviation f = a·(cos(r·u) − 1) + b·sin(r·u).
 */
const ARC: i32 = 9;
const STRIDE: i32 = 2 + 2 * ARC;
const TAU: f64 = 2 * Math.PI;
/** Below this |δ| (one sample interval long), E₀ and E₁ use their series. */
const SERIES: f64 = 0.05;
/** Below this |δ|, the derivative moments use their series. */
const MOMENT_SERIES: f64 = 0.5;
/**
 * Arcs use the endpoint form unless |ω ∓ rate| (one interval long) is below this guard, values and
 * derivatives alike: near the removable singularity at ω = ±rate the endpoint numerator cancels, with a
 * relative error of about ε/|x| (ε/x² for its curvature term) for values and one and two more powers of
 * 1/|x| for the first and second derivatives: under 1e-12 at the guard. sinwhisky.rs guards |δ·L| < 0.5
 * on its longer supports; on unit intervals that band holds most arcs at low frequencies, and 0.05 gave
 * the same values (within 1.4e-14 of the strongest, against quadrature) and peaks (within 1e-15 rad/sample)
 * 1.2–3.4 times faster (dsp/AGENTS.md).
 */
const ENDPOINT_GUARD: f64 = 0.05;
/** Support phases are recomputed with sin/cos every this many intervals. */
const ANCHOR: i32 = 64;

const input = memory.data(16384, 16); // f32[MAX_INPUT]
const atoms = memory.data(655360, 16); // f64[MAX_INPUT · STRIDE]
const omegas = memory.data(65536, 16); // f64[MAX_QUERIES]
const output = memory.data(131072, 16); // f64[2 · MAX_QUERIES]: values (re, im), or scan magnitudes
const peaks = memory.data(8192, 16); // f64[MAX_PEAKS · 4]: omega, re, im, curvature

let intervals: i32 = 0;

@inline function f32At(index: i32): f64 {
  return <f64>load<f32>(input + (<usize>index << 2));
}

@inline function atom(n: i32, field: i32): f64 {
  return load<f64>(atoms + (<usize>(n * STRIDE + field) << 3));
}

@inline function setAtom(n: i32, field: i32, value: f64): void {
  store<f64>(atoms + (<usize>(n * STRIDE + field) << 3), value);
}

@inline function f64At(base: usize, index: i32): f64 {
  return load<f64>(base + (<usize>index << 3));
}

@inline function setF64(base: usize, index: i32, value: f64): void {
  store<f64>(base + (<usize>index << 3), value);
}

// ---- circles ---------------------------------------------------------------------------------------

// The last fitted circle: whether it is one, its centre height and radius in sample units, and the
// angles of its three points, unwrapped in the direction that visits the middle one.
let circle = false;
let centerY: f64 = 0;
let radius: f64 = 0;
let angle0: f64 = 0;
let angle1: f64 = 0;
let angle2: f64 = 0;

@inline function wrap(value: f64): f64 {
  const wrapped = value - TAU * Math.floor(value / TAU);
  return wrapped >= TAU ? 0 : wrapped;
}

/**
 * The circle through (0, y0), (1, y1), (2, y2) with x scaled by the triplet's range (SinWhisky's
 * aperture), as `swFit` fits it, but with its angles unwrapped in the direction that visits the middle
 * point (sinwhisky.rs `arc_angles`), so each arc goes the short way between its samples. A flat or
 * straight triplet is no circle.
 */
function fitCircle(y0: f64, y1: f64, y2: f64): void {
  circle = false;
  let apr = Math.abs(y0 - y1);
  if (Math.abs(y1 - y2) > apr) apr = Math.abs(y1 - y2);
  if (Math.abs(y0 - y2) > apr) apr = Math.abs(y0 - y2);
  if (!(apr > 0)) return;
  const a = y0 / apr;
  const b = y1 / apr;
  const c = y2 / apr;
  const det = 2 * ((b - c) * -1 + (a - b));
  if (Math.abs(det) < 1e-9) return;
  const bc = a * a - 1 - b * b;
  const cd = 1 + b * b - 4 - c * c;
  const ox = (bc * (b - c) - cd * (a - b)) / det;
  const oy = (-cd + bc) / det;
  const r = Math.sqrt(ox * ox + (a - oy) * (a - oy));
  const a0 = Math.atan2(a - oy, 0 - ox);
  const a1 = Math.atan2(b - oy, 1 - ox);
  const a2 = Math.atan2(c - oy, 2 - ox);
  const first = wrap(a1 - a0);
  const full = wrap(a2 - a0);
  const direction: f64 = first <= full ? 1 : -1;
  const middle = a0 + direction * wrap(direction * (a1 - a0));
  const last = middle + direction * wrap(direction * (a2 - a1));
  centerY = oy * apr;
  radius = r * apr;
  angle0 = a0;
  angle1 = middle;
  angle2 = last;
  // Rounding can leave an endpoint off its sample for nearly straight triplets: those are lines.
  const bound = 1e-8 * Math.max(1, Math.max(Math.abs(y0), Math.max(Math.abs(y1), Math.abs(y2))));
  circle = Math.abs(centerY + radius * Math.sin(a0) - y0) <= bound
    && Math.abs(centerY + radius * Math.sin(middle) - y1) <= bound
    && Math.abs(centerY + radius * Math.sin(last) - y2) <= bound;
}

/** Adds one half (weight w) to interval n: an arc from `from` sweeping `rate`, or the samples' line. */
function addHalf(n: i32, slot: i32, w: f64, isArc: bool, from: f64, rate: f64): void {
  if (!isArc) {
    setAtom(n, 1, atom(n, 1) + w * (f32At(n + 1) - f32At(n)));
    return;
  }
  const base = 2 + slot * ARC;
  const a = w * radius * Math.sin(from);
  const b = w * radius * Math.cos(from);
  const sine = Math.sin(rate);
  // cos(r) − 1 = −2 sin²(r/2) keeps flat arcs relatively accurate.
  const half = Math.sin(0.5 * rate);
  const cosMinusOne = -2 * half * half;
  setAtom(n, base, a);
  setAtom(n, base + 1, b);
  setAtom(n, base + 2, rate);
  setAtom(n, base + 3, Math.cos(rate));
  setAtom(n, base + 4, sine);
  setAtom(n, base + 5, rate * b);
  setAtom(n, base + 6, rate * (b * (1 + cosMinusOne) - a * sine));
  setAtom(n, base + 7, a * cosMinusOne + b * sine);
  setAtom(n, base + 8, rate * rate * a);
}

export function faftInputPtr(): usize {
  return input;
}

export function faftMaxInput(): i32 {
  return MAX_INPUT;
}

export function faftAtomsPtr(): usize {
  return atoms;
}

export function faftAtomStride(): i32 {
  return STRIDE;
}

export function faftIntervals(): i32 {
  return intervals;
}

export function faftOmegasPtr(): usize {
  return omegas;
}

export function faftOutputPtr(): usize {
  return output;
}

export function faftMaxQueries(): i32 {
  return MAX_QUERIES;
}

export function faftPeaksPtr(): usize {
  return peaks;
}

export function faftMaxPeaks(): i32 {
  return MAX_PEAKS;
}

/**
 * Determines the circles of input[0..length) and prepares the curve's rotations: one interval per pair
 * of samples, the window [0, length − 1). Returns the interval count, or −1 for fewer than 2 samples, more
 * than the buffer, or a non-finite sample.
 */
export function faftPrepare(length: i32): i32 {
  intervals = 0;
  atoms32Ready = false;
  if (length < 2 || length > MAX_INPUT) return -1;
  for (let k = 0; k < length; k++) {
    if (!isFinite(f32At(k))) return -1;
  }
  memory.fill(atoms, 0, <usize>((length - 1) * STRIDE) << 3);
  // Circle k passes through samples k − 1, k, k + 1 (1 ≤ k ≤ length − 2): its left half covers interval
  // k − 1, its right half interval k.
  let rightOk = false;
  let rightFrom: f64 = 0;
  let rightRate: f64 = 0;
  let rightY: f64 = 0;
  let rightR: f64 = 0;
  for (let n = 0; n < length - 1; n++) {
    setAtom(n, 0, f32At(n));
    const hasRight = n >= 1 && n <= length - 2; // circle n
    const hasLeft = n + 1 <= length - 2; // circle n + 1
    const halves: i32 = (hasRight ? 1 : 0) + (hasLeft ? 1 : 0);
    if (halves == 0) {
      addHalf(n, 0, 1, false, 0, 0);
      continue;
    }
    const w: f64 = 1 / <f64>halves;
    if (hasRight) {
      // The circle fitted for the previous interval's left half.
      centerY = rightY;
      radius = rightR;
      addHalf(n, 0, w, rightOk, rightFrom, rightRate);
    }
    if (hasLeft) {
      fitCircle(f32At(n), f32At(n + 1), f32At(n + 2));
      addHalf(n, 1, w, circle, angle0, angle1 - angle0);
      rightOk = circle;
      rightFrom = angle1;
      rightRate = angle2 - angle1;
      rightY = centerY;
      rightR = radius;
    }
  }
  intervals = length - 1;
  return intervals;
}

/** The curve at a real position in [0, length − 1] (what SinWhisky's zoom would sample); NaN outside. */
export function faftEvaluate(position: f64): f64 {
  if (!(position >= 0) || !(position <= <f64>intervals) || intervals == 0) return NaN;
  let n = <i32>Math.floor(position);
  if (n >= intervals) n = intervals - 1;
  const u = position - <f64>n;
  let value = atom(n, 0) + atom(n, 1) * u;
  for (let slot = 0; slot < 2; slot++) {
    const base = 2 + slot * ARC;
    const rate = atom(n, base + 2);
    value += atom(n, base) * (Math.cos(rate * u) - 1) + atom(n, base + 1) * Math.sin(rate * u);
  }
  return value;
}

// ---- the rectangular transform ---------------------------------------------------------------------

// E₀(δ) and E₁(δ) over one interval, given e = e^{iδ}: results in these globals.
let e0re: f64 = 0;
let e0im: f64 = 0;

@inline function e0(delta: f64, ere: f64, eim: f64): void {
  if (Math.abs(delta) < SERIES) {
    const d2 = delta * delta;
    e0re = 1 + d2 * (-1.0 / 6 + d2 * (1.0 / 120 + d2 * (-1.0 / 5040)));
    e0im = delta * (0.5 + d2 * (-1.0 / 24 + d2 * (1.0 / 720 + d2 * (-1.0 / 40320))));
  } else {
    e0re = eim / delta;
    e0im = (1 - ere) / delta;
  }
}

/** E₀ from a precomputed 1/δ (the ratio form), or its series near δ = 0. */
@inline function e0With(delta: f64, inverse: f64, ere: f64, eim: f64): void {
  if (Math.abs(delta) < SERIES) {
    e0(delta, ere, eim);
  } else {
    e0re = eim * inverse;
    e0im = (1 - ere) * inverse;
  }
}

let rectRe: f64 = 0;
let rectIm: f64 = 0;

/** The unnormalized rectangular integral at ω, phase origin at the window start: into rectRe/rectIm. */
function rect(omega: f64): void {
  const qre = Math.cos(omega);
  const qim = -Math.sin(omega);
  // The DC and line terms are the same for every interval.
  e0(-omega, qre, qim);
  const dcRe = e0re;
  const dcIm = e0im;
  let lineRe: f64;
  let lineIm: f64;
  if (Math.abs(omega) < SERIES) {
    const d2 = omega * omega;
    lineRe = 0.5 + d2 * (-1.0 / 8 + d2 * (1.0 / 144 + d2 * (-1.0 / 5760)));
    lineIm = -omega * (1.0 / 3 + d2 * (-1.0 / 30 + d2 * (1.0 / 840 + d2 * (-1.0 / 45360))));
  } else {
    // E₁ = −i(e − E₀)/δ with δ = −ω.
    const dre = qre - dcRe;
    const dim = qim - dcIm;
    lineRe = dim / -omega;
    lineIm = -dre / -omega;
  }
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
      const base = 2 + slot * ARC;
      const a = atom(n, base);
      const b = atom(n, base + 1);
      if (a == 0 && b == 0) continue;
      const rate = atom(n, base + 2);
      const up = rate - omega;
      const down = -rate - omega;
      if (Math.min(Math.abs(up), Math.abs(down)) >= ENDPOINT_GUARD) {
        // Endpoint form: [f'(1)q − f'(0) + iω f(1) q + r²a E₀(−ω)] / ((ω − r)(ω + r)), q = e^{−iω}.
        const inverse = 1 / (up * down);
        const endSlope = atom(n, base + 6);
        const change = omega * atom(n, base + 7);
        const curvature = atom(n, base + 8);
        pre += (qre * endSlope - atom(n, base + 5) - qim * change + curvature * dcRe) * inverse;
        pim += (qim * endSlope + qre * change + curvature * dcIm) * inverse;
        continue;
      }
      // Moment form near ω = ±rate. e^{i(±r − ω)}: the arc's prepared turn times e^{−iω}. One division
      // gives both 1/δ: 1/(δ₊δ₋) times the other δ.
      const cr = atom(n, base + 3);
      const sr = atom(n, base + 4);
      const inverse = 1 / (up * down);
      e0With(up, down * inverse, cr * qre - sr * qim, cr * qim + sr * qre);
      const upRe = e0re;
      const upIm = e0im;
      e0With(down, up * inverse, cr * qre + sr * qim, cr * qim - sr * qre);
      const downRe = e0re;
      const downIm = e0im;
      // cos(r·u) − 1 and sin(r·u) transforms.
      const cosRe = 0.5 * (upRe + downRe) - dcRe;
      const cosIm = 0.5 * (upIm + downIm) - dcIm;
      const sinRe = 0.5 * (upIm - downIm);
      const sinIm = -0.5 * (upRe - downRe);
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

// Two frequencies at once: the same sums with each lane its own ω. Results in these globals.
let pairRe: v128 = f64x2.splat(0);
let pairIm: v128 = f64x2.splat(0);
let e0PairRe: v128 = f64x2.splat(0);
let e0PairIm: v128 = f64x2.splat(0);

/** E₀ for two lanes from their 1/δ, the series in a lane only where |δ| < SERIES (rare: computed only then). */
@inline function e0Pair(delta: v128, inverse: v128, ere: v128, eim: v128): void {
  let re = f64x2.mul(eim, inverse);
  let im = f64x2.mul(f64x2.sub(f64x2.splat(1), ere), inverse);
  const small = f64x2.lt(f64x2.abs(delta), f64x2.splat(SERIES));
  if (v128.any_true(small)) {
    const d2 = f64x2.mul(delta, delta);
    const seriesRe = f64x2.add(f64x2.splat(1), f64x2.mul(d2, f64x2.add(f64x2.splat(-1.0 / 6), f64x2.mul(d2, f64x2.add(f64x2.splat(1.0 / 120), f64x2.mul(d2, f64x2.splat(-1.0 / 5040)))))));
    const seriesIm = f64x2.mul(delta, f64x2.add(f64x2.splat(0.5), f64x2.mul(d2, f64x2.add(f64x2.splat(-1.0 / 24), f64x2.mul(d2, f64x2.add(f64x2.splat(1.0 / 720), f64x2.mul(d2, f64x2.splat(-1.0 / 40320))))))));
    re = v128.bitselect(seriesRe, re, small);
    im = v128.bitselect(seriesIm, im, small);
  }
  e0PairRe = re;
  e0PairIm = im;
}

/** The unnormalized rectangular integrals at ω₀ and ω₁ (lanes 0 and 1): into pairRe/pairIm. */
function rectPair(omega0: f64, omega1: f64): void {
  const omega = f64x2.replace_lane(f64x2.splat(omega0), 1, omega1);
  const qre = f64x2.replace_lane(f64x2.splat(Math.cos(omega0)), 1, Math.cos(omega1));
  const qim = f64x2.replace_lane(f64x2.splat(-Math.sin(omega0)), 1, -Math.sin(omega1));
  const minus = f64x2.neg(omega);
  e0Pair(minus, f64x2.div(f64x2.splat(1), minus), qre, qim);
  const dcRe = e0PairRe;
  const dcIm = e0PairIm;
  // E₁(−ω): the same for every interval, so its series/ratio choice is made per lane in scalar code.
  rect1Line(omega0);
  const line0Re = lineRe1;
  const line0Im = lineIm1;
  rect1Line(omega1);
  const lineRe = f64x2.replace_lane(f64x2.splat(line0Re), 1, lineRe1);
  const lineIm = f64x2.replace_lane(f64x2.splat(line0Im), 1, lineIm1);
  let sre = f64x2.splat(1);
  let sim = f64x2.splat(0);
  let sumRe = f64x2.splat(0);
  let sumIm = f64x2.splat(0);
  const half = f64x2.splat(0.5);
  for (let n = 0; n < intervals; n++) {
    const y0 = f64x2.splat(atom(n, 0));
    const slope = f64x2.splat(atom(n, 1));
    let pre = f64x2.add(f64x2.mul(y0, dcRe), f64x2.mul(slope, lineRe));
    let pim = f64x2.add(f64x2.mul(y0, dcIm), f64x2.mul(slope, lineIm));
    for (let slot = 0; slot < 2; slot++) {
      const base = 2 + slot * ARC;
      const as = atom(n, base);
      const bs = atom(n, base + 1);
      if (as == 0 && bs == 0) continue;
      const rate = f64x2.splat(atom(n, base + 2));
      const up = f64x2.sub(rate, omega);
      const down = f64x2.sub(f64x2.neg(rate), omega);
      const inverse = f64x2.div(f64x2.splat(1), f64x2.mul(up, down));
      // Endpoint form in both lanes (one division for both), as in `rect`.
      const endSlope = f64x2.splat(atom(n, base + 6));
      const change = f64x2.mul(omega, f64x2.splat(atom(n, base + 7)));
      const curvature = f64x2.splat(atom(n, base + 8));
      let devRe = f64x2.mul(f64x2.add(f64x2.sub(f64x2.sub(f64x2.mul(qre, endSlope), f64x2.splat(atom(n, base + 5))), f64x2.mul(qim, change)), f64x2.mul(curvature, dcRe)), inverse);
      let devIm = f64x2.mul(f64x2.add(f64x2.add(f64x2.mul(qim, endSlope), f64x2.mul(qre, change)), f64x2.mul(curvature, dcIm)), inverse);
      // The moment form only where a lane is within the guard (computed for both, then selected).
      const near = f64x2.lt(f64x2.min(f64x2.abs(up), f64x2.abs(down)), f64x2.splat(ENDPOINT_GUARD));
      if (v128.any_true(near)) {
        const cr = f64x2.splat(atom(n, base + 3));
        const sr = f64x2.splat(atom(n, base + 4));
        e0Pair(up, f64x2.mul(down, inverse), f64x2.sub(f64x2.mul(cr, qre), f64x2.mul(sr, qim)), f64x2.add(f64x2.mul(cr, qim), f64x2.mul(sr, qre)));
        const upRe = e0PairRe;
        const upIm = e0PairIm;
        e0Pair(down, f64x2.mul(up, inverse), f64x2.add(f64x2.mul(cr, qre), f64x2.mul(sr, qim)), f64x2.sub(f64x2.mul(cr, qim), f64x2.mul(sr, qre)));
        const downRe = e0PairRe;
        const downIm = e0PairIm;
        const cosRe = f64x2.sub(f64x2.mul(half, f64x2.add(upRe, downRe)), dcRe);
        const cosIm = f64x2.sub(f64x2.mul(half, f64x2.add(upIm, downIm)), dcIm);
        const sinRe = f64x2.mul(half, f64x2.sub(upIm, downIm));
        const sinIm = f64x2.neg(f64x2.mul(half, f64x2.sub(upRe, downRe)));
        const a = f64x2.splat(as);
        const b = f64x2.splat(bs);
        devRe = v128.bitselect(f64x2.add(f64x2.mul(a, cosRe), f64x2.mul(b, sinRe)), devRe, near);
        devIm = v128.bitselect(f64x2.add(f64x2.mul(a, cosIm), f64x2.mul(b, sinIm)), devIm, near);
      }
      pre = f64x2.add(pre, devRe);
      pim = f64x2.add(pim, devIm);
    }
    sumRe = f64x2.add(sumRe, f64x2.sub(f64x2.mul(sre, pre), f64x2.mul(sim, pim)));
    sumIm = f64x2.add(sumIm, f64x2.add(f64x2.mul(sre, pim), f64x2.mul(sim, pre)));
    if ((n + 1) % ANCHOR == 0) {
      const t = <f64>(n + 1);
      sre = f64x2.replace_lane(f64x2.splat(Math.cos(omega0 * t)), 1, Math.cos(omega1 * t));
      sim = f64x2.replace_lane(f64x2.splat(-Math.sin(omega0 * t)), 1, -Math.sin(omega1 * t));
    } else {
      const next = f64x2.sub(f64x2.mul(sre, qre), f64x2.mul(sim, qim));
      sim = f64x2.add(f64x2.mul(sre, qim), f64x2.mul(sim, qre));
      sre = next;
    }
  }
  pairRe = sumRe;
  pairIm = sumIm;
}

let lineRe1: f64 = 0;
let lineIm1: f64 = 0;

/** E₁(−ω) over one interval (the line term), into lineRe1/lineIm1. */
@inline function rect1Line(omega: f64): void {
  if (Math.abs(omega) < SERIES) {
    const d2 = omega * omega;
    lineRe1 = 0.5 + d2 * (-1.0 / 8 + d2 * (1.0 / 144 + d2 * (-1.0 / 5760)));
    lineIm1 = -omega * (1.0 / 3 + d2 * (-1.0 / 30 + d2 * (1.0 / 840 + d2 * (-1.0 / 45360))));
  } else {
    e0(-omega, Math.cos(omega), -Math.sin(omega));
    const dre = Math.cos(omega) - e0re;
    const dim = -Math.sin(omega) - e0im;
    lineRe1 = dim / -omega;
    lineIm1 = -dre / -omega;
  }
}

// ---- values ----------------------------------------------------------------------------------------

/**
 * Normalized values at omegas[0..count) (radians per sample, any real value): output[2k], output[2k + 1]
 * = the transform over the window [0, intervals) divided by its weight sum (T, or T/2 with the periodic
 * Hann window applied analytically), phase origin at the window start. A unit sinusoid reads about 0.5.
 * `simd` evaluates two frequencies at a time (the same sums, rounding apart). Returns false without a
 * prepared curve, for too many queries, or a non-finite frequency.
 */
export function faftValues(count: i32, hann: bool, simd: bool): bool {
  if (intervals == 0 || count < 0 || count > MAX_QUERIES) return false;
  for (let k = 0; k < count; k++) {
    if (!isFinite(f64At(omegas, k))) return false;
  }
  const duration = <f64>intervals;
  const shift = TAU / duration;
  const norm = hann ? 2 / duration : 1 / duration;
  let k = 0;
  if (simd) {
    for (; k + 1 < count; k += 2) {
      const w0 = f64At(omegas, k);
      const w1 = f64At(omegas, k + 1);
      if (!hann) {
        rectPair(w0, w1);
        setF64(output, 2 * k, f64x2.extract_lane(pairRe, 0) * norm);
        setF64(output, 2 * k + 1, f64x2.extract_lane(pairIm, 0) * norm);
        setF64(output, 2 * k + 2, f64x2.extract_lane(pairRe, 1) * norm);
        setF64(output, 2 * k + 3, f64x2.extract_lane(pairIm, 1) * norm);
        continue;
      }
      rectPair(w0, w1);
      const cRe = pairRe;
      const cIm = pairIm;
      rectPair(w0 - shift, w1 - shift);
      const lRe = pairRe;
      const lIm = pairIm;
      rectPair(w0 + shift, w1 + shift);
      const re = f64x2.sub(f64x2.mul(cRe, f64x2.splat(0.5)), f64x2.mul(f64x2.add(lRe, pairRe), f64x2.splat(0.25)));
      const im = f64x2.sub(f64x2.mul(cIm, f64x2.splat(0.5)), f64x2.mul(f64x2.add(lIm, pairIm), f64x2.splat(0.25)));
      setF64(output, 2 * k, f64x2.extract_lane(re, 0) * norm);
      setF64(output, 2 * k + 1, f64x2.extract_lane(im, 0) * norm);
      setF64(output, 2 * k + 2, f64x2.extract_lane(re, 1) * norm);
      setF64(output, 2 * k + 3, f64x2.extract_lane(im, 1) * norm);
    }
  }
  for (; k < count; k++) {
    const w = f64At(omegas, k);
    weighted(w, hann, shift);
    setF64(output, 2 * k, weightedRe * norm);
    setF64(output, 2 * k + 1, weightedIm * norm);
  }
  return true;
}

let weightedRe: f64 = 0;
let weightedIm: f64 = 0;

/** Unnormalized rectangular or Hann value at ω into weightedRe/weightedIm. */
function weighted(omega: f64, hann: bool, shift: f64): void {
  rect(omega);
  if (!hann) {
    weightedRe = rectRe;
    weightedIm = rectIm;
    return;
  }
  const cRe = rectRe;
  const cIm = rectIm;
  rect(omega - shift);
  const lRe = rectRe;
  const lIm = rectIm;
  rect(omega + shift);
  weightedRe = 0.5 * cRe - 0.25 * (lRe + rectRe);
  weightedIm = 0.5 * cIm - 0.25 * (lIm + rectIm);
}

/**
 * Magnitudes on a uniform grid: output[k] = |F(lower + k·step)| normalized as in `faftValues`, for
 * k < count. With `density` > 0 the Hann window applies, its ±2π/T neighbours `density` grid steps away
 * (step must be (2π/T)/density): one extended rectangular scan serves all three. Returns false without a
 * prepared curve or beyond the buffers. `mode`: 0 scalar f64, 1 two f64 lanes, 2 four f32 lanes.
 */
export function faftScan(lower: f64, step: f64, count: i32, density: i32, mode: i32): bool {
  const pad = density;
  const total = count + 2 * pad;
  if (intervals == 0 || count < 1 || density < 0 || total > MAX_QUERIES || !isFinite(lower) || !(step > 0) || mode < 0 || mode > 2) return false;
  const duration = <f64>intervals;
  // Rectangular values on the extended grid into the output buffer (re, im), then combined in place.
  let k = 0;
  if (mode == 2) {
    // Four frequencies per f32x4 instruction (about 1e-6 of the largest value).
    const last = <f64>(total - 1 - pad);
    for (; k < total; k += 4) {
      const base = <f64>(k - pad);
      rectQuad(lower + step * base, lower + step * Math.min(base + 1, last), lower + step * Math.min(base + 2, last), lower + step * Math.min(base + 3, last));
      for (let l = 0; l < 4 && k + l < total; l++) {
        setF64(output, 2 * (k + l), load<f64>(quadOut + (<usize>l << 3)));
        setF64(output, 2 * (k + l) + 1, load<f64>(quadOut + (<usize>(4 + l) << 3)));
      }
    }
  }
  if (mode == 1) {
    for (; k + 1 < total; k += 2) {
      rectPair(lower + step * <f64>(k - pad), lower + step * <f64>(k + 1 - pad));
      setF64(output, 2 * k, f64x2.extract_lane(pairRe, 0));
      setF64(output, 2 * k + 1, f64x2.extract_lane(pairIm, 0));
      setF64(output, 2 * k + 2, f64x2.extract_lane(pairRe, 1));
      setF64(output, 2 * k + 3, f64x2.extract_lane(pairIm, 1));
    }
  }
  for (; k < total; k++) {
    rect(lower + step * <f64>(k - pad));
    setF64(output, 2 * k, rectRe);
    setF64(output, 2 * k + 1, rectIm);
  }
  const norm = pad > 0 ? 2 / duration : 1 / duration;
  // Magnitudes front to back: output[j] only reads complex values at indices ≥ j.
  for (let j = 0; j < count; j++) {
    let re = f64At(output, 2 * (j + pad));
    let im = f64At(output, 2 * (j + pad) + 1);
    if (pad > 0) {
      re = 0.5 * re - 0.25 * (f64At(output, 2 * j) + f64At(output, 2 * (j + 2 * pad)));
      im = 0.5 * im - 0.25 * (f64At(output, 2 * j + 1) + f64At(output, 2 * (j + 2 * pad) + 1));
    }
    setF64(output, j, Math.sqrt(re * re + im * im) * norm);
  }
  return true;
}

// ---- derivatives and peaks -------------------------------------------------------------------------

// ∫₀¹ uᵏ e^{iδu} du for k = 0..3, given e = e^{iδ}: moments[2k], moments[2k + 1].
const moments = memory.data(64, 16);
// The three Hann shifts' value and derivatives, centred (re, im) × 3 orders: scratch.
const derivative = memory.data(144, 16);

function momentsAt(delta: f64, ere: f64, eim: f64, orders: i32): void {
  if (Math.abs(delta) < MOMENT_SERIES) {
    // Σ (iδ)ⁿ / (n!·(n + k + 1)), 20 terms (|δ|ⁿ/n! < 1e-18 at |δ| < 0.5).
    for (let k = 0; k < orders; k++) {
      let re: f64 = 0;
      let im: f64 = 0;
      let termRe: f64 = 1; // (iδ)ⁿ / n!
      let termIm: f64 = 0;
      for (let n = 0; n < 20; n++) {
        const scale: f64 = 1 / <f64>(n + k + 1);
        re += termRe * scale;
        im += termIm * scale;
        const nextRe = -termIm * delta / <f64>(n + 1);
        termIm = termRe * delta / <f64>(n + 1);
        termRe = nextRe;
      }
      setF64(moments, 2 * k, re);
      setF64(moments, 2 * k + 1, im);
    }
    return;
  }
  // M₀ = (e − 1)/(iδ); Mₖ = (e − k·Mₖ₋₁)/(iδ).
  let mre = eim / delta;
  let mim = (1 - ere) / delta;
  setF64(moments, 0, mre);
  setF64(moments, 1, mim);
  for (let k = 1; k < orders; k++) {
    const dre = ere - <f64>k * mre;
    const dim = eim - <f64>k * mim;
    mre = dim / delta;
    mim = -dre / delta;
    setF64(moments, 2 * k, mre);
    setF64(moments, 2 * k + 1, mim);
  }
}

// One arc's moment-form contribution to an interval's ψ₀, ψ₁, ψ₂ (∫ uᵏ f(u) e^{−iωu} du of its deviation),
// near resonance, where the endpoint form is 0/0: into am0re … am2im.
let am0re: f64 = 0, am0im: f64 = 0, am1re: f64 = 0, am1im: f64 = 0, am2re: f64 = 0, am2im: f64 = 0;

function arcMoments(n: i32, base: i32, omega: f64, qre: f64, qim: f64, dc0re: f64, dc0im: f64, dc1re: f64, dc1im: f64, dc2re: f64, dc2im: f64): void {
  const a = atom(n, base);
  const b = atom(n, base + 1);
  const rate = atom(n, base + 2);
  const cr = atom(n, base + 3);
  const sr = atom(n, base + 4);
  momentsAt(rate - omega, cr * qre - sr * qim, cr * qim + sr * qre, 3);
  const u0re = f64At(moments, 0), u0im = f64At(moments, 1);
  const u1re = f64At(moments, 2), u1im = f64At(moments, 3);
  const u2re = f64At(moments, 4), u2im = f64At(moments, 5);
  momentsAt(-rate - omega, cr * qre + sr * qim, cr * qim - sr * qre, 3);
  const d0re = f64At(moments, 0), d0im = f64At(moments, 1);
  const d1re = f64At(moments, 2), d1im = f64At(moments, 3);
  const d2re = f64At(moments, 4), d2im = f64At(moments, 5);
  // a·((U + D)/2 − DC) + b·(−i)(U − D)/2, per order.
  am0re = a * (0.5 * (u0re + d0re) - dc0re) + b * 0.5 * (u0im - d0im);
  am0im = a * (0.5 * (u0im + d0im) - dc0im) - b * 0.5 * (u0re - d0re);
  am1re = a * (0.5 * (u1re + d1re) - dc1re) + b * 0.5 * (u1im - d1im);
  am1im = a * (0.5 * (u1im + d1im) - dc1im) - b * 0.5 * (u1re - d1re);
  am2re = a * (0.5 * (u2re + d2re) - dc2re) + b * 0.5 * (u2im - d2im);
  am2im = a * (0.5 * (u2im + d2im) - dc2im) - b * 0.5 * (u2re - d2re);
}

/**
 * The rectangular integral and its first two ω-derivatives with phase origin at the window centre, into
 * derivative[slot·6 ..]: (re, im) of F, F′, F″.
 */
function rectDerivatives(omega: f64, slot: i32): void {
  const qre = Math.cos(omega);
  const qim = -Math.sin(omega);
  const center = 0.5 * <f64>intervals;
  // ψ_k per interval from the DC/line moments (shared) and each arc's.
  momentsAt(-omega, qre, qim, 4);
  const dc0re = f64At(moments, 0); const dc0im = f64At(moments, 1);
  const dc1re = f64At(moments, 2); const dc1im = f64At(moments, 3);
  const dc2re = f64At(moments, 4); const dc2im = f64At(moments, 5);
  const dc3re = f64At(moments, 6); const dc3im = f64At(moments, 7);
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
      const base = 2 + s * ARC;
      const a = atom(n, base);
      const b = atom(n, base + 1);
      if (a == 0 && b == 0) continue;
      const rate = atom(n, base + 2);
      const up = rate - omega;
      const down = -rate - omega;
      if (Math.min(Math.abs(up), Math.abs(down)) >= ENDPOINT_GUARD) {
        // Endpoint form F = N/D, D = ω² − r², differentiated in ω: F′ = (N′ − 2ωF)/D and
        // F″ = (N″ − 4ωF′ − 2F)/D; then ψ₁ = iF′ and ψ₂ = −F″. dE₀(−ω)/dω = −iE₁(−ω), and so on.
        const inverse = 1 / (up * down);
        const startSlope = atom(n, base + 5);
        const endSlope = atom(n, base + 6);
        const endChange = atom(n, base + 7);
        const curvature = atom(n, base + 8);
        const endRe = endSlope * qre, endIm = endSlope * qim;
        const chRe = endChange * qre, chIm = endChange * qim;
        const n0re = endRe - startSlope - omega * chIm + curvature * dc0re;
        const n0im = endIm + omega * chRe + curvature * dc0im;
        const n1re = endIm - chIm + omega * chRe + curvature * dc1im;
        const n1im = -endRe + chRe + omega * chIm - curvature * dc1re;
        const n2re = -endRe + 2 * chRe + omega * chIm - curvature * dc2re;
        const n2im = -endIm + 2 * chIm - omega * chRe - curvature * dc2im;
        const h0re = n0re * inverse, h0im = n0im * inverse;
        const h1re = (n1re - 2 * omega * h0re) * inverse, h1im = (n1im - 2 * omega * h0im) * inverse;
        const h2re = (n2re - 4 * omega * h1re - 2 * h0re) * inverse;
        const h2im = (n2im - 4 * omega * h1im - 2 * h0im) * inverse;
        p0re += h0re; p0im += h0im;
        p1re -= h1im; p1im += h1re;
        p2re -= h2re; p2im -= h2im;
        continue;
      }
      arcMoments(n, base, omega, qre, qim, dc0re, dc0im, dc1re, dc1im, dc2re, dc2im);
      p0re += am0re; p0im += am0im;
      p1re += am1re; p1im += am1im;
      p2re += am2re; p2im += am2im;
    }
    // d/dω of e^{−iω(σ + u)}: −i(σ + u). σ = n − centre.
    const sigma = <f64>n - center;
    const v0re = p0re, v0im = p0im;
    // −i(σψ₀ + ψ₁)
    const t1re = sigma * p0re + p1re, t1im = sigma * p0im + p1im;
    const v1re = t1im, v1im = -t1re;
    // −(σ²ψ₀ + 2σψ₁ + ψ₂)
    const v2re = -(sigma * sigma * p0re + 2 * sigma * p1re + p2re);
    const v2im = -(sigma * sigma * p0im + 2 * sigma * p1im + p2im);
    f0re += sre * v0re - sim * v0im; f0im += sre * v0im + sim * v0re;
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

// ---- batches: many frequencies at once, for the joint tone fit (assembly/tones.ts) ------------------

/** Most frequencies per batch call. */
export const MAX_BATCH: i32 = 2048;
export const batchOmegas = memory.data(16384, 16); // f64[MAX_BATCH]
export const batchOut = memory.data(98304, 16); // f64[MAX_BATCH · 6]
// The two lanes' DC moments, (re, im) × 4 orders each.
const laneMoments = memory.data(128, 16);

// Centred F, F′, F″ (re, im) of two frequencies, from rectDerivativesPair.
let pd0re: v128 = f64x2.splat(0), pd0im: v128 = f64x2.splat(0);
let pd1re: v128 = f64x2.splat(0), pd1im: v128 = f64x2.splat(0);
let pd2re: v128 = f64x2.splat(0), pd2im: v128 = f64x2.splat(0);

@inline function pair(a: f64, b: f64): v128 {
  return f64x2.replace_lane(f64x2.splat(a), 1, b);
}

@inline function lane(base: usize, field: i32, which: i32): f64 {
  return load<f64>(base + (<usize>(which * 8 + field) << 3));
}

/**
 * `rectDerivatives` at two frequencies per f64x2 lane pair: the endpoint form in both lanes (one
 * division for both), and the moment form only for a lane inside the guard (scalar, then inserted).
 */
function rectDerivativesPair(omega0: f64, omega1: f64): void {
  const omega = pair(omega0, omega1);
  const c0 = Math.cos(omega0), c1 = Math.cos(omega1);
  const s0 = Math.sin(omega0), s1 = Math.sin(omega1);
  const qre = pair(c0, c1);
  const qim = pair(-s0, -s1);
  const center = 0.5 * <f64>intervals;
  momentsAt(-omega0, c0, -s0, 4);
  memory.copy(laneMoments, moments, 64);
  momentsAt(-omega1, c1, -s1, 4);
  memory.copy(laneMoments + 64, moments, 64);
  const dc0re = pair(lane(laneMoments, 0, 0), lane(laneMoments, 0, 1)), dc0im = pair(lane(laneMoments, 1, 0), lane(laneMoments, 1, 1));
  const dc1re = pair(lane(laneMoments, 2, 0), lane(laneMoments, 2, 1)), dc1im = pair(lane(laneMoments, 3, 0), lane(laneMoments, 3, 1));
  const dc2re = pair(lane(laneMoments, 4, 0), lane(laneMoments, 4, 1)), dc2im = pair(lane(laneMoments, 5, 0), lane(laneMoments, 5, 1));
  const dc3re = pair(lane(laneMoments, 6, 0), lane(laneMoments, 6, 1)), dc3im = pair(lane(laneMoments, 7, 0), lane(laneMoments, 7, 1));
  let sre = pair(Math.cos(omega0 * center), Math.cos(omega1 * center));
  let sim = pair(Math.sin(omega0 * center), Math.sin(omega1 * center));
  const zero = f64x2.splat(0);
  const two = f64x2.splat(2);
  const four = f64x2.splat(4);
  const guard = f64x2.splat(ENDPOINT_GUARD);
  let f0re = zero, f0im = zero, f1re = zero, f1im = zero, f2re = zero, f2im = zero;
  for (let n = 0; n < intervals; n++) {
    const y0 = f64x2.splat(atom(n, 0));
    const slope = f64x2.splat(atom(n, 1));
    let p0re = f64x2.add(f64x2.mul(y0, dc0re), f64x2.mul(slope, dc1re));
    let p0im = f64x2.add(f64x2.mul(y0, dc0im), f64x2.mul(slope, dc1im));
    let p1re = f64x2.add(f64x2.mul(y0, dc1re), f64x2.mul(slope, dc2re));
    let p1im = f64x2.add(f64x2.mul(y0, dc1im), f64x2.mul(slope, dc2im));
    let p2re = f64x2.add(f64x2.mul(y0, dc2re), f64x2.mul(slope, dc3re));
    let p2im = f64x2.add(f64x2.mul(y0, dc2im), f64x2.mul(slope, dc3im));
    for (let slot = 0; slot < 2; slot++) {
      const base = 2 + slot * ARC;
      if (atom(n, base) == 0 && atom(n, base + 1) == 0) continue;
      const rate = f64x2.splat(atom(n, base + 2));
      const up = f64x2.sub(rate, omega);
      const down = f64x2.sub(f64x2.neg(rate), omega);
      const inverse = f64x2.div(f64x2.splat(1), f64x2.mul(up, down));
      const endSlope = f64x2.splat(atom(n, base + 6));
      const endChange = f64x2.splat(atom(n, base + 7));
      const curvature = f64x2.splat(atom(n, base + 8));
      const endRe = f64x2.mul(endSlope, qre), endIm = f64x2.mul(endSlope, qim);
      const chRe = f64x2.mul(endChange, qre), chIm = f64x2.mul(endChange, qim);
      const wChRe = f64x2.mul(omega, chRe), wChIm = f64x2.mul(omega, chIm);
      const n0re = f64x2.add(f64x2.sub(f64x2.sub(endRe, f64x2.splat(atom(n, base + 5))), wChIm), f64x2.mul(curvature, dc0re));
      const n0im = f64x2.add(f64x2.add(endIm, wChRe), f64x2.mul(curvature, dc0im));
      const n1re = f64x2.add(f64x2.add(f64x2.sub(endIm, chIm), wChRe), f64x2.mul(curvature, dc1im));
      const n1im = f64x2.sub(f64x2.add(f64x2.sub(chRe, endRe), wChIm), f64x2.mul(curvature, dc1re));
      const n2re = f64x2.sub(f64x2.add(f64x2.sub(f64x2.mul(two, chRe), endRe), wChIm), f64x2.mul(curvature, dc2re));
      const n2im = f64x2.sub(f64x2.sub(f64x2.sub(f64x2.mul(two, chIm), endIm), wChRe), f64x2.mul(curvature, dc2im));
      const h0re = f64x2.mul(n0re, inverse), h0im = f64x2.mul(n0im, inverse);
      const twoOmega = f64x2.mul(two, omega);
      const h1re = f64x2.mul(f64x2.sub(n1re, f64x2.mul(twoOmega, h0re)), inverse);
      const h1im = f64x2.mul(f64x2.sub(n1im, f64x2.mul(twoOmega, h0im)), inverse);
      const fourOmega = f64x2.mul(four, omega);
      const h2re = f64x2.mul(f64x2.sub(f64x2.sub(n2re, f64x2.mul(fourOmega, h1re)), f64x2.mul(two, h0re)), inverse);
      const h2im = f64x2.mul(f64x2.sub(f64x2.sub(n2im, f64x2.mul(fourOmega, h1im)), f64x2.mul(two, h0im)), inverse);
      // Contributions to ψ₀ (F), ψ₁ (iF′) and ψ₂ (−F″).
      let a0re = h0re, a0im = h0im;
      let a1re = f64x2.neg(h1im), a1im = h1re;
      let a2re = f64x2.neg(h2re), a2im = f64x2.neg(h2im);
      const near = f64x2.lt(f64x2.min(f64x2.abs(up), f64x2.abs(down)), guard);
      if (v128.any_true(near)) {
        if (i64x2.extract_lane(near, 0) != 0) {
          arcMoments(n, base, omega0, c0, -s0, lane(laneMoments, 0, 0), lane(laneMoments, 1, 0), lane(laneMoments, 2, 0), lane(laneMoments, 3, 0), lane(laneMoments, 4, 0), lane(laneMoments, 5, 0));
          a0re = f64x2.replace_lane(a0re, 0, am0re); a0im = f64x2.replace_lane(a0im, 0, am0im);
          a1re = f64x2.replace_lane(a1re, 0, am1re); a1im = f64x2.replace_lane(a1im, 0, am1im);
          a2re = f64x2.replace_lane(a2re, 0, am2re); a2im = f64x2.replace_lane(a2im, 0, am2im);
        }
        if (i64x2.extract_lane(near, 1) != 0) {
          arcMoments(n, base, omega1, c1, -s1, lane(laneMoments, 0, 1), lane(laneMoments, 1, 1), lane(laneMoments, 2, 1), lane(laneMoments, 3, 1), lane(laneMoments, 4, 1), lane(laneMoments, 5, 1));
          a0re = f64x2.replace_lane(a0re, 1, am0re); a0im = f64x2.replace_lane(a0im, 1, am0im);
          a1re = f64x2.replace_lane(a1re, 1, am1re); a1im = f64x2.replace_lane(a1im, 1, am1im);
          a2re = f64x2.replace_lane(a2re, 1, am2re); a2im = f64x2.replace_lane(a2im, 1, am2im);
        }
      }
      p0re = f64x2.add(p0re, a0re); p0im = f64x2.add(p0im, a0im);
      p1re = f64x2.add(p1re, a1re); p1im = f64x2.add(p1im, a1im);
      p2re = f64x2.add(p2re, a2re); p2im = f64x2.add(p2im, a2im);
    }
    const sigma = f64x2.splat(<f64>n - center);
    // v₀ = ψ₀, v₁ = −i(σψ₀ + ψ₁), v₂ = −(σ²ψ₀ + 2σψ₁ + ψ₂), each times the support phase.
    const t1re = f64x2.add(f64x2.mul(sigma, p0re), p1re);
    const t1im = f64x2.add(f64x2.mul(sigma, p0im), p1im);
    const v1re = t1im, v1im = f64x2.neg(t1re);
    const v2re = f64x2.neg(f64x2.add(f64x2.mul(sigma, f64x2.add(t1re, p1re)), p2re));
    const v2im = f64x2.neg(f64x2.add(f64x2.mul(sigma, f64x2.add(t1im, p1im)), p2im));
    f0re = f64x2.add(f0re, f64x2.sub(f64x2.mul(sre, p0re), f64x2.mul(sim, p0im)));
    f0im = f64x2.add(f0im, f64x2.add(f64x2.mul(sre, p0im), f64x2.mul(sim, p0re)));
    f1re = f64x2.add(f1re, f64x2.sub(f64x2.mul(sre, v1re), f64x2.mul(sim, v1im)));
    f1im = f64x2.add(f1im, f64x2.add(f64x2.mul(sre, v1im), f64x2.mul(sim, v1re)));
    f2re = f64x2.add(f2re, f64x2.sub(f64x2.mul(sre, v2re), f64x2.mul(sim, v2im)));
    f2im = f64x2.add(f2im, f64x2.add(f64x2.mul(sre, v2im), f64x2.mul(sim, v2re)));
    if ((n + 1) % ANCHOR == 0) {
      const t = <f64>(n + 1) - center;
      sre = pair(Math.cos(omega0 * t), Math.cos(omega1 * t));
      sim = pair(-Math.sin(omega0 * t), -Math.sin(omega1 * t));
    } else {
      const next = f64x2.sub(f64x2.mul(sre, qre), f64x2.mul(sim, qim));
      sim = f64x2.add(f64x2.mul(sre, qim), f64x2.mul(sim, qre));
      sre = next;
    }
  }
  pd0re = f0re; pd0im = f0im; pd1re = f1re; pd1im = f1im; pd2re = f2re; pd2im = f2im;
}

// ---- the fast path: four frequencies per f32x4 instruction --------------------------------------------
//
// The real-time equalizer (assembly/eq.ts) needs many values per window and tolerates about 1e-6 of
// relative error (its divider is complete whatever the analysis: the residual is the exact difference).
// `rectQuad` evaluates `rect` at four frequencies in f32 lanes: the endpoint form (the moment form inside
// the guard), blocks of BLOCK intervals summed in f32 and accumulated in f64, and the support phase
// re-anchored at every block from an f64 chain, so neither the f32 phase nor the f32 sums drift.

const atoms32 = memory.data(327680, 16); // f32[MAX_INPUT · STRIDE], converted from `atoms` on first use
let atoms32Ready = false;
const BLOCK: i32 = 64;
// Per lane (f64): q = e^{−iω}, q^BLOCK, E₀(−ω), E₁(−ω), and the f64 support phase; then the results.
const quadLanes = memory.data(512, 16); // f64[4 · 16]
const quadOut = memory.data(64, 16); // f64 re × 4, im × 4

function prepareAtoms32(): void {
  const total = intervals * STRIDE;
  for (let i = 0; i < total; i++) store<f32>(atoms32 + (<usize>i << 2), <f32>load<f64>(atoms + (<usize>i << 3)));
  atoms32Ready = true;
}

/** The prepared rotations as the fast path reads them (f32, `faftAtomStride` per interval), for the GPU. */
export function faftAtoms32Ptr(): usize {
  if (!atoms32Ready) prepareAtoms32();
  return atoms32;
}

@inline function a32(n: i32, field: i32): f32 {
  return load<f32>(atoms32 + (<usize>(n * STRIDE + field) << 2));
}

@inline function laneSlot(which: i32, field: i32): usize {
  return quadLanes + (<usize>(which * 16 + field) << 3);
}

@inline function quad(field: i32): v128 {
  let v = f32x4.splat(<f32>load<f64>(laneSlot(0, field)));
  v = f32x4.replace_lane(v, 1, <f32>load<f64>(laneSlot(1, field)));
  v = f32x4.replace_lane(v, 2, <f32>load<f64>(laneSlot(2, field)));
  return f32x4.replace_lane(v, 3, <f32>load<f64>(laneSlot(3, field)));
}

/** E₀ for four f32 lanes: the ratio form, the series where |δ| < SERIES. */
@inline function e0Quad(delta: v128, ere: v128, eim: v128, re: bool): v128 {
  const ratio = re ? f32x4.div(eim, delta) : f32x4.div(f32x4.sub(f32x4.splat(1), ere), delta);
  const small = f32x4.lt(f32x4.abs(delta), f32x4.splat(<f32>SERIES));
  if (!v128.any_true(small)) return ratio;
  const d2 = f32x4.mul(delta, delta);
  const series = re
    ? f32x4.add(f32x4.splat(1), f32x4.mul(d2, f32x4.add(f32x4.splat(<f32>(-1.0 / 6)), f32x4.mul(d2, f32x4.splat(<f32>(1.0 / 120))))))
    : f32x4.mul(delta, f32x4.add(f32x4.splat(0.5), f32x4.mul(d2, f32x4.add(f32x4.splat(<f32>(-1.0 / 24)), f32x4.mul(d2, f32x4.splat(<f32>(1.0 / 720)))))));
  return v128.bitselect(series, ratio, small);
}

/** `rect` at w0 … w3 in f32x4 lanes, about 1e-6 of the largest value: (re, im) into quadOut. */
function rectQuad(w0: f64, w1: f64, w2: f64, w3: f64): void {
  if (!atoms32Ready) prepareAtoms32();
  for (let which = 0; which < 4; which++) {
    const omega = which == 0 ? w0 : which == 1 ? w1 : which == 2 ? w2 : w3;
    const qre = Math.cos(omega);
    const qim = -Math.sin(omega);
    e0(-omega, qre, qim);
    rect1Line(omega);
    store<f64>(laneSlot(which, 0), omega);
    store<f64>(laneSlot(which, 1), qre);
    store<f64>(laneSlot(which, 2), qim);
    store<f64>(laneSlot(which, 3), Math.cos(omega * <f64>BLOCK));
    store<f64>(laneSlot(which, 4), -Math.sin(omega * <f64>BLOCK));
    store<f64>(laneSlot(which, 5), e0re);
    store<f64>(laneSlot(which, 6), e0im);
    store<f64>(laneSlot(which, 7), lineRe1);
    store<f64>(laneSlot(which, 8), lineIm1);
    store<f64>(laneSlot(which, 9), 1); // support phase (f64), re
    store<f64>(laneSlot(which, 10), 0); // im
    store<f64>(laneSlot(which, 11), 0); // sum re
    store<f64>(laneSlot(which, 12), 0); // sum im
  }
  const omega = quad(0);
  const qre = quad(1), qim = quad(2);
  const dcRe = quad(5), dcIm = quad(6);
  const lineRe = quad(7), lineIm = quad(8);
  const guard = f32x4.splat(<f32>ENDPOINT_GUARD);
  const half = f32x4.splat(0.5);
  for (let start = 0; start < intervals; start += BLOCK) {
    let sre = quad(9), sim = quad(10);
    let sumRe = f32x4.splat(0), sumIm = f32x4.splat(0);
    const end = start + BLOCK < intervals ? start + BLOCK : intervals;
    for (let n = start; n < end; n++) {
      const y0 = f32x4.splat(a32(n, 0));
      const slope = f32x4.splat(a32(n, 1));
      let pre = f32x4.add(f32x4.mul(y0, dcRe), f32x4.mul(slope, lineRe));
      let pim = f32x4.add(f32x4.mul(y0, dcIm), f32x4.mul(slope, lineIm));
      for (let slot = 0; slot < 2; slot++) {
        const base = 2 + slot * ARC;
        const as = a32(n, base);
        const bs = a32(n, base + 1);
        if (as == 0 && bs == 0) continue;
        const rate = f32x4.splat(a32(n, base + 2));
        const up = f32x4.sub(rate, omega);
        const down = f32x4.sub(f32x4.neg(rate), omega);
        const inverse = f32x4.div(f32x4.splat(1), f32x4.mul(up, down));
        const endSlope = f32x4.splat(a32(n, base + 6));
        const change = f32x4.mul(omega, f32x4.splat(a32(n, base + 7)));
        const curvature = f32x4.splat(a32(n, base + 8));
        let devRe = f32x4.mul(f32x4.add(f32x4.sub(f32x4.sub(f32x4.mul(qre, endSlope), f32x4.splat(a32(n, base + 5))), f32x4.mul(qim, change)), f32x4.mul(curvature, dcRe)), inverse);
        let devIm = f32x4.mul(f32x4.add(f32x4.add(f32x4.mul(qim, endSlope), f32x4.mul(qre, change)), f32x4.mul(curvature, dcIm)), inverse);
        const near = f32x4.lt(f32x4.min(f32x4.abs(up), f32x4.abs(down)), guard);
        if (v128.any_true(near)) {
          const cr = f32x4.splat(a32(n, base + 3));
          const sr = f32x4.splat(a32(n, base + 4));
          const uRe = f32x4.sub(f32x4.mul(cr, qre), f32x4.mul(sr, qim));
          const uIm = f32x4.add(f32x4.mul(cr, qim), f32x4.mul(sr, qre));
          const dRe = f32x4.add(f32x4.mul(cr, qre), f32x4.mul(sr, qim));
          const dIm = f32x4.sub(f32x4.mul(cr, qim), f32x4.mul(sr, qre));
          const upRe = e0Quad(up, uRe, uIm, true), upIm = e0Quad(up, uRe, uIm, false);
          const downRe = e0Quad(down, dRe, dIm, true), downIm = e0Quad(down, dRe, dIm, false);
          const cosRe = f32x4.sub(f32x4.mul(half, f32x4.add(upRe, downRe)), dcRe);
          const cosIm = f32x4.sub(f32x4.mul(half, f32x4.add(upIm, downIm)), dcIm);
          const sinRe = f32x4.mul(half, f32x4.sub(upIm, downIm));
          const sinIm = f32x4.neg(f32x4.mul(half, f32x4.sub(upRe, downRe)));
          const a = f32x4.splat(as);
          const b = f32x4.splat(bs);
          devRe = v128.bitselect(f32x4.add(f32x4.mul(a, cosRe), f32x4.mul(b, sinRe)), devRe, near);
          devIm = v128.bitselect(f32x4.add(f32x4.mul(a, cosIm), f32x4.mul(b, sinIm)), devIm, near);
        }
        pre = f32x4.add(pre, devRe);
        pim = f32x4.add(pim, devIm);
      }
      sumRe = f32x4.add(sumRe, f32x4.sub(f32x4.mul(sre, pre), f32x4.mul(sim, pim)));
      sumIm = f32x4.add(sumIm, f32x4.add(f32x4.mul(sre, pim), f32x4.mul(sim, pre)));
      const next = f32x4.sub(f32x4.mul(sre, qre), f32x4.mul(sim, qim));
      sim = f32x4.add(f32x4.mul(sre, qim), f32x4.mul(sim, qre));
      sre = next;
    }
    // Block sums into f64, and the f64 support phase advanced by q^BLOCK.
    for (let which = 0; which < 4; which++) {
      const re = which == 0 ? f32x4.extract_lane(sumRe, 0) : which == 1 ? f32x4.extract_lane(sumRe, 1) : which == 2 ? f32x4.extract_lane(sumRe, 2) : f32x4.extract_lane(sumRe, 3);
      const im = which == 0 ? f32x4.extract_lane(sumIm, 0) : which == 1 ? f32x4.extract_lane(sumIm, 1) : which == 2 ? f32x4.extract_lane(sumIm, 2) : f32x4.extract_lane(sumIm, 3);
      store<f64>(laneSlot(which, 11), load<f64>(laneSlot(which, 11)) + <f64>re);
      store<f64>(laneSlot(which, 12), load<f64>(laneSlot(which, 12)) + <f64>im);
      const pre = load<f64>(laneSlot(which, 9));
      const pim = load<f64>(laneSlot(which, 10));
      const bre = load<f64>(laneSlot(which, 3));
      const bim = load<f64>(laneSlot(which, 4));
      store<f64>(laneSlot(which, 9), pre * bre - pim * bim);
      store<f64>(laneSlot(which, 10), pre * bim + pim * bre);
    }
  }
  for (let which = 0; which < 4; which++) {
    store<f64>(quadOut + (<usize>which << 3), load<f64>(laneSlot(which, 11)));
    store<f64>(quadOut + (<usize>(4 + which) << 3), load<f64>(laneSlot(which, 12)));
  }
}

/**
 * The batch buffers, for a host that plans a fit's values (tones.ts PLAN) and evaluates them itself (on
 * the GPU, or here with `faftRectBatchFast`) before the fit reads them (EXTERNAL).
 */
export function faftBatchOmegasPtr(): usize {
  return batchOmegas;
}

export function faftBatchOutPtr(): usize {
  return batchOut;
}

export function faftBatchMax(): i32 {
  return MAX_BATCH;
}

/** `rectBatchFast` on batchOmegas[0..count) of the prepared window; false without one or for a bad count. */
export function faftRectBatchFast(count: i32): bool {
  if (intervals == 0 || count < 0 || count > MAX_BATCH) return false;
  rectBatchFast(count);
  return true;
}

/** `rectBatch` through `rectQuad`: four frequencies at a time, about 1e-6 of the largest value. */
export function rectBatchFast(count: i32): void {
  for (let k = 0; k < count; k += 4) {
    const last = count - 1;
    rectQuad(f64At(batchOmegas, k), f64At(batchOmegas, k + 1 <= last ? k + 1 : last), f64At(batchOmegas, k + 2 <= last ? k + 2 : last), f64At(batchOmegas, k + 3 <= last ? k + 3 : last));
    for (let l = 0; l < 4 && k + l < count; l++) {
      setF64(batchOut, 2 * (k + l), load<f64>(quadOut + (<usize>l << 3)));
      setF64(batchOut, 2 * (k + l) + 1, load<f64>(quadOut + (<usize>(4 + l) << 3)));
    }
  }
}

/** Unnormalized rectangular values (window-start origin) at batchOmegas[0..count) into batchOut[2k ..]. */
export function rectBatch(count: i32, simd: bool): void {
  let k = 0;
  if (simd) {
    for (; k + 1 < count; k += 2) {
      rectPair(f64At(batchOmegas, k), f64At(batchOmegas, k + 1));
      setF64(batchOut, 2 * k, f64x2.extract_lane(pairRe, 0));
      setF64(batchOut, 2 * k + 1, f64x2.extract_lane(pairIm, 0));
      setF64(batchOut, 2 * k + 2, f64x2.extract_lane(pairRe, 1));
      setF64(batchOut, 2 * k + 3, f64x2.extract_lane(pairIm, 1));
    }
  }
  for (; k < count; k++) {
    rect(f64At(batchOmegas, k));
    setF64(batchOut, 2 * k, rectRe);
    setF64(batchOut, 2 * k + 1, rectIm);
  }
}

/** Unnormalized centred rectangular F, F′, F″ (re, im) at batchOmegas[0..count) into batchOut[6k ..]. */
export function derivativesBatch(count: i32, simd: bool): void {
  let k = 0;
  if (simd) {
    for (; k + 1 < count; k += 2) {
      rectDerivativesPair(f64At(batchOmegas, k), f64At(batchOmegas, k + 1));
      for (let l = 0; l < 2; l++) {
        const out = batchOut + (<usize>((k + l) * 6) << 3);
        store<f64>(out, l == 0 ? f64x2.extract_lane(pd0re, 0) : f64x2.extract_lane(pd0re, 1));
        store<f64>(out, l == 0 ? f64x2.extract_lane(pd0im, 0) : f64x2.extract_lane(pd0im, 1), 8);
        store<f64>(out, l == 0 ? f64x2.extract_lane(pd1re, 0) : f64x2.extract_lane(pd1re, 1), 16);
        store<f64>(out, l == 0 ? f64x2.extract_lane(pd1im, 0) : f64x2.extract_lane(pd1im, 1), 24);
        store<f64>(out, l == 0 ? f64x2.extract_lane(pd2re, 0) : f64x2.extract_lane(pd2re, 1), 32);
        store<f64>(out, l == 0 ? f64x2.extract_lane(pd2im, 0) : f64x2.extract_lane(pd2im, 1), 40);
      }
    }
  }
  for (; k < count; k++) {
    rectDerivatives(f64At(batchOmegas, k), 0);
    memory.copy(batchOut + (<usize>(k * 6) << 3), derivative, 48);
  }
}

@inline function at(slot: i32, field: i32): f64 {
  return load<f64>(derivative + (<usize>(slot * 6 + field) << 3));
}

// Normalized centred F, F′, F″ at the last `centred` call.
let c0re: f64 = 0, c0im: f64 = 0, c1re: f64 = 0, c1im: f64 = 0, c2re: f64 = 0, c2im: f64 = 0;

function centred(omega: f64, hann: bool): void {
  const duration = <f64>intervals;
  rectDerivatives(omega, 0);
  let w0: f64 = 1, w1: f64 = 0;
  if (hann) {
    // Centred periodic Hann: ½F(ω) + ¼F(ω − Δ) + ¼F(ω + Δ) (the shifts' phases flip at the centre).
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

let refinedOmega: f64 = 0;

/**
 * Maximizes |F|² inside [lo, hi] from `start` with exact derivatives: Newton where the local model is
 * concave and inside the bracket (narrowed by the derivative's sign), bisection otherwise. Returns 1 for
 * a converged interior maximum (refinedOmega, and c0/c2 hold its value and second derivative), 0 for a
 * maximum on a bracket edge (a monotone flank another seed owns), −1 without convergence.
 */
function refine(start: f64, lo: f64, hi: f64, hann: bool, tolerance: f64, maxIterations: i32): i32 {
  let omega = Math.min(hi, Math.max(lo, start));
  let low = lo;
  let high = hi;
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
 * Continuous relevant peaks of |F| in [lower, upper] (sinwhisky.rs `Faft::peaks`): a scan spaced by the
 * window's duration ((2π/T)/density), its local maxima at or over `relativeFloor` of the strongest as
 * seeds (started by a parabola on log magnitude), each refined by Newton with exact derivatives to
 * `tolerance`, maxima on a bracket edge and unconverged seeds dropped, peaks within a quarter of 2π/T
 * merged (the strongest stays), at most `maxPeaks`, sorted by frequency. peaks[4k ..] = ω, value (re, im;
 * phase origin at the window start), and d²|F|²/dω² there. Returns the count, or −1 for invalid input.
 */
export function faftPeaks(lower: f64, upper: f64, density: i32, relativeFloor: f64, maxPeaks: i32, hann: bool, tolerance: f64): i32 {
  if (intervals == 0 || !(lower < upper) || !isFinite(lower) || !isFinite(upper) || density < 1 || density > 64
    || !(relativeFloor >= 0 && relativeFloor <= 1) || maxPeaks < 1 || maxPeaks > MAX_PEAKS || !(tolerance > 0)) {
    return -1;
  }
  const duration = <f64>intervals;
  const resolution = TAU / duration;
  const step = resolution / <f64>density;
  const count = <i32>Math.floor((upper - lower) / step) + 1;
  if (count + 2 * density > MAX_QUERIES) return -1;
  if (!faftScan(lower, step, count, hann ? density : 0, 1)) return -1;
  let strongest: f64 = 0;
  for (let k = 0; k < count; k++) strongest = Math.max(strongest, f64At(output, k));
  const floor = relativeFloor * strongest;
  // Candidates (ω, |F|, curvature, re, im) after the scan's magnitudes, behind them in the output buffer.
  const candidates = output + (<usize>MAX_QUERIES << 3);
  let found = 0;
  for (let k = 0; k < count && found < MAX_PEAKS; k++) {
    const m = f64At(output, k);
    const left = k > 0 ? f64At(output, k - 1) : -Infinity;
    const right = k + 1 < count ? f64At(output, k + 1) : -Infinity;
    if (!(m >= floor && m > 0 && m >= left && m > right)) continue;
    let start = lower + step * <f64>k;
    if (k > 0 && k + 1 < count && left > 0 && right > 0) {
      const l = Math.log(left);
      const c = Math.log(m);
      const r = Math.log(right);
      const bend = l - 2 * c + r;
      if (bend < 0) start += Math.min(step, Math.max(-step, (0.5 * (l - r) / bend) * step));
    }
    const lo = Math.max(lower, lower + step * <f64>(k - 1));
    const hi = Math.min(upper, lower + step * <f64>(k + 1));
    if (refine(start, lo, hi, hann, tolerance, 32) != 1) continue;
    // Back to a window-start phase origin: multiply by e^{−iωT/2}.
    const turn = -refinedOmega * 0.5 * duration;
    const re = c0re * Math.cos(turn) - c0im * Math.sin(turn);
    const im = c0re * Math.sin(turn) + c0im * Math.cos(turn);
    const base = found * 5;
    setF64(candidates, base, refinedOmega);
    setF64(candidates, base + 1, Math.sqrt(c0re * c0re + c0im * c0im));
    setF64(candidates, base + 2, 2 * (c1re * c1re + c1im * c1im + c0re * c2re + c0im * c2im));
    setF64(candidates, base + 3, re);
    setF64(candidates, base + 4, im);
    found++;
  }
  // Strongest first: keep each unless a stronger kept one is within the merge width.
  const merge = 0.25 * resolution;
  let best: f64 = strongest;
  for (let i = 0; i < found; i++) best = Math.max(best, f64At(candidates, i * 5 + 1));
  let kept = 0;
  while (kept < maxPeaks) {
    let pick = -1;
    for (let i = 0; i < found; i++) {
      const m = f64At(candidates, i * 5 + 1);
      if (m < 0) continue;
      if (pick < 0 || m > f64At(candidates, pick * 5 + 1)) pick = i;
    }
    if (pick < 0) break;
    const m = f64At(candidates, pick * 5 + 1);
    const omega = f64At(candidates, pick * 5);
    setF64(candidates, pick * 5 + 1, -1);
    if (m < relativeFloor * best || m == 0) continue;
    let near = false;
    for (let j = 0; j < kept; j++) {
      if (Math.abs(f64At(peaks, j * 4) - omega) <= merge) near = true;
    }
    if (near) continue;
    setF64(peaks, kept * 4, omega);
    setF64(peaks, kept * 4 + 1, f64At(candidates, pick * 5 + 3));
    setF64(peaks, kept * 4 + 2, f64At(candidates, pick * 5 + 4));
    setF64(peaks, kept * 4 + 3, f64At(candidates, pick * 5 + 2));
    kept++;
  }
  // By frequency (insertion sort: at most MAX_PEAKS).
  for (let i = 1; i < kept; i++) {
    for (let j = i; j > 0 && f64At(peaks, (j - 1) * 4) > f64At(peaks, j * 4); j--) {
      for (let f = 0; f < 4; f++) {
        const t = f64At(peaks, (j - 1) * 4 + f);
        setF64(peaks, (j - 1) * 4 + f, f64At(peaks, j * 4 + f));
        setF64(peaks, j * 4 + f, t);
      }
    }
  }
  return kept;
}

/**
 * `faftValues` in f32x4 lanes (`rectQuad`: four frequencies per instruction, about 1e-6 of the largest
 * value): at most MAX_BATCH/3 Hann or MAX_BATCH rectangular frequencies, into the output buffer as (re, im). The equalizer's fast path,
 * exported for its parity tests. False without a window, for too many or a non-finite frequency.
 */
export function faftValuesFast(count: i32, hann: bool): bool {
  if (intervals == 0 || count < 0 || (hann ? 3 : 1) * count > MAX_BATCH) return false;
  for (let k = 0; k < count; k++) {
    if (!isFinite(f64At(omegas, k))) return false;
  }
  const duration = <f64>intervals;
  const shift = TAU / duration;
  const per = hann ? 3 : 1;
  for (let k = 0; k < count; k++) {
    const w = f64At(omegas, k);
    setF64(batchOmegas, per * k, w);
    if (hann) {
      setF64(batchOmegas, 3 * k + 1, w - shift);
      setF64(batchOmegas, 3 * k + 2, w + shift);
    }
  }
  rectBatchFast(per * count);
  const norm = hann ? 2 / duration : 1 / duration;
  for (let k = 0; k < count; k++) {
    let re = f64At(batchOut, 2 * per * k);
    let im = f64At(batchOut, 2 * per * k + 1);
    if (hann) {
      re = 0.5 * re - 0.25 * (f64At(batchOut, 6 * k + 2) + f64At(batchOut, 6 * k + 4));
      im = 0.5 * im - 0.25 * (f64At(batchOut, 6 * k + 3) + f64At(batchOut, 6 * k + 5));
    }
    setF64(output, 2 * k, re * norm);
    setF64(output, 2 * k + 1, im * norm);
  }
  return true;
}

/**
 * Keeps intervals [first, first + count) of the prepared curve: a window whose end intervals average two
 * circles, as the interior ones do, when its samples were prepared with context on both sides (the
 * equalizer prepares T + 3 samples and keeps the middle T intervals). False outside the curve.
 */
export function faftCrop(first: i32, count: i32): bool {
  if (first < 0 || count < 1 || first + count > intervals) return false;
  if (first > 0) memory.copy(atoms, atoms + (<usize>(first * STRIDE) << 3), <usize>(count * STRIDE) << 3);
  intervals = count;
  atoms32Ready = false;
  return true;
}
