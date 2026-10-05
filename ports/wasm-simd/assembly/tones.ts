// Transposed from appx/dsp at ec84edcd331af8810be9df42cf5d16f3a348d6d5 (2026-10-05).
// The jointly fitted tone model of a FAFT window (R. Cecchini, sinwhisky.rs: src/faft/pursuit.rs `fit`
// and `kernel`, src/faft/banded.rs `fit_banded` and `fit_refined`; PAPER_EQ_COMPRESSION.md §2.4).
//
// A maximum of |FAFT| is not a component: through the window kernel K (the exact transform of the
// weight) every tone shares its height with its neighbours. Real tones α·cos(ω(t − tc)) + β·sin(ω(t − tc))
// around the window centre tc are therefore fitted jointly by weighted least squares, every quantity in
// closed form: the Gram matrix from K, the projections from FAFT values (assembly/faft.ts, prepared by
// faftPrepare). In ascending frequency the Gram matrix is banded (the coupling of two tones decays with
// their distance), so the fit is one ascending pass (band Cholesky and forward substitution, which gives
// the explained energy) and one descending pass (back substitution for the amplitudes, Takahashi's
// selected inversion for each tone's significance: the energy it alone explains). `faftFitRefined` then
// moves each frequency by safeguarded Newton steps on its own partial residual, the other tones removed
// through K, K′, K″, and refits. The weighted curve energy (Gauss–Legendre, the only quadrature) gives
// the residual and its bound sqrt(E_r/W), which holds at every real frequency.
//
// Static buffers only. At most MAX_TONES tones; the band keeps at most MAX_WIDTH neighbours above each
// tone (a denser cluster truncates the Gram matrix; the residual is then evaluated from the full form).

import { batchOmegas, batchOut, derivativesBatch, faftAtomStride, faftAtomsPtr, faftIntervals, faftOmegasPtr, MAX_BATCH, rectBatch, rectBatchFast } from './faft';

/** How transform values are evaluated: scalar f64, two f64 lanes, or four f32 lanes (about 1e-6). */
export const SCALAR: i32 = 0;
export const SIMD: i32 = 1;
export const FAST: i32 = 2;
/**
 * Two halves of a FAST evaluation, for a host that computes the values elsewhere (many windows on the
 * GPU at once, ports/webgpu/faft.mjs): PLAN only lists the rectangular frequencies a step needs in batchOmegas
 * (`tonePlanned` of them) and stops; EXTERNAL then reads their values from batchOut, where the host wrote
 * them, instead of computing them. Everything else is FAST's.
 */
export const PLAN: i32 = 3;
export const EXTERNAL: i32 = 4;
/** Rectangular frequencies the last PLAN step listed in batchOmegas. */
let planned: i32 = 0;
/** The largest frequency move of the last Newton step (`refineStep`, `refineTones`). */
let largestStep: f64 = 0;
/**
 * In FAST mode refinement takes a(ω)'s derivatives from central differences of values ±FAST_STEP
 * resolutions away (f32 noise and truncation both under 1e-3 resolution on a Newton step), instead of
 * the exact derivative kernel, which costs about eight values in f64.
 */
const FAST_STEP: f64 = 0.02;

const MAX_TONES: i32 = 256;
const MAX_WIDTH: i32 = 64;
const BAND: i32 = MAX_WIDTH + 1;
/** f64 per fitted component: ω, value (re, im; window-start origin), observed (re, im), significance. */
const FIT_STRIDE: i32 = 6;
const TAU: f64 = 2 * Math.PI;

const sorted = memory.data(2048, 16); // f64[MAX_TONES]
const observed = memory.data(4096, 16); // centred normalized (re, im) [MAX_TONES]
const kernelMinus = memory.data(133120, 16); // k(ω_i − ω_j), band storage [MAX_TONES · BAND]
const kernelPlus = memory.data(133120, 16); // k(ω_i + ω_j)
const lowerFactor = memory.data(133120, 16); // L(i, j), band storage
const takahashi = memory.data(133120, 16); // Z(i, j) = (G⁻¹)(i, j) within the band
const forward = memory.data(2048, 16); // z
const amplitudes = memory.data(4096, 16); // [cos, sin][MAX_TONES]
const unique = memory.data(2048, 16);
const active = memory.data(256, 16); // u8
const components = memory.data(12288, 16); // f64[MAX_TONES · FIT_STRIDE]
const derivs = memory.data(12288, 16); // centred normalized F, F′, F″ (re, im) [MAX_TONES · 6]
const moved = memory.data(2048, 16); // refined frequencies

let fitCount: i32 = 0;
let fitWidth: i32 = 0;
let fitTruncated = false;
let fitDegenerate: i32 = 0;
let fitExplained: f64 = 0;
let fitHann = true;

@inline function f64At(base: usize, index: i32): f64 {
  return load<f64>(base + (<usize>index << 3));
}

@inline function setF64(base: usize, index: i32, value: f64): void {
  store<f64>(base + (<usize>index << 3), value);
}

@inline function band(i: i32, j: i32): i32 {
  return i * BAND + (i - j);
}

// ---- the window kernel -----------------------------------------------------------------------------

// sinc(x) and its first two derivatives: Taylor to x¹⁶ below 1/2, closed forms above (pursuit.rs `sinc`).
let sinc0: f64 = 0, sinc1: f64 = 0, sinc2: f64 = 0;

function sinc(x: f64): void {
  if (Math.abs(x) < 0.5) {
    const y = x * x;
    let s: f64 = 0, d: f64 = 0, dd: f64 = 0;
    // (−1)ⁿ/(2n + 1)!, n = 8 … 0.
    let factorial: f64 = 1;
    for (let n = 1; n <= 8; n++) factorial *= <f64>(2 * n) * <f64>(2 * n + 1);
    for (let n = 8; n >= 0; n--) {
      const c = ((n & 1) == 0 ? 1.0 : -1.0) / factorial;
      s = s * y + c;
      if (n > 0) {
        d = d * y + <f64>(2 * n) * c;
        dd = dd * y + <f64>(2 * n * (2 * n - 1)) * c;
        factorial /= <f64>(2 * n) * <f64>(2 * n + 1);
      }
    }
    sinc0 = s;
    sinc1 = d * x;
    sinc2 = dd;
    return;
  }
  const sine = Math.sin(x);
  const cosine = Math.cos(x);
  const s = sine / x;
  const d = (cosine - s) / x;
  sinc0 = s;
  sinc1 = d;
  sinc2 = -s - (2 * d) / x;
}

// The normalized centred kernel K(ν)/W and its first two ν-derivatives: kd0, kd1, kd2.
let kd0: f64 = 0, kd1: f64 = 0, kd2: f64 = 0;

/** K(ν) = ∫_{−T/2}^{T/2} w e^{−iνu} du / W for the rectangular weight, or Hann = ½K(ν) + ¼K(ν − Δ) + ¼K(ν + Δ). */
function kernel(nu: f64, hann: bool): void {
  const duration = <f64>faftIntervals();
  const half = 0.5 * duration;
  sinc(nu * half);
  let k0 = duration * sinc0, k1 = duration * half * sinc1, k2 = duration * half * half * sinc2;
  if (!hann) {
    kd0 = k0 / duration;
    kd1 = k1 / duration;
    kd2 = k2 / duration;
    return;
  }
  const shift = TAU / duration;
  k0 *= 0.5; k1 *= 0.5; k2 *= 0.5;
  sinc((nu - shift) * half);
  k0 += 0.25 * duration * sinc0; k1 += 0.25 * duration * half * sinc1; k2 += 0.25 * duration * half * half * sinc2;
  sinc((nu + shift) * half);
  k0 += 0.25 * duration * sinc0; k1 += 0.25 * duration * half * sinc1; k2 += 0.25 * duration * half * half * sinc2;
  // W = T/2 for the Hann window.
  kd0 = k0 / half;
  kd1 = k1 / half;
  kd2 = k2 / half;
}

/** The normalized window kernel K(ν)/W (1 at ν = 0) or its `order`-th ν-derivative (0..2); NaN for a bad order or no window. */
export function faftKernel(nu: f64, hann: bool, order: i32): f64 {
  if (faftIntervals() == 0 || !isFinite(nu) || order < 0 || order > 2) return NaN;
  kernel(nu, hann);
  return order == 0 ? kd0 : order == 1 ? kd1 : kd2;
}

// ---- transform values at many frequencies ----------------------------------------------------------

/**
 * Centred normalized values a(ω_k) = F(ω_k)·e^{iωT/2}/W at src[0..count) into dst (re, im). The Hann
 * window's three rectangular values per frequency go through one batch (two or four frequencies per
 * SIMD instruction, by `mode`).
 */
function centredValuesAt(src: usize, count: i32, dst: usize, hann: bool, mode: i32): void {
  const duration = <f64>faftIntervals();
  const shift = TAU / duration;
  const per = hann ? 3 : 1;
  for (let k = 0; k < count; k++) {
    const omega = f64At(src, k);
    setF64(batchOmegas, per * k, omega);
    if (hann) {
      setF64(batchOmegas, 3 * k + 1, omega - shift);
      setF64(batchOmegas, 3 * k + 2, omega + shift);
    }
  }
  if (mode == PLAN) {
    planned = per * count;
    return;
  }
  if (mode == FAST) rectBatchFast(per * count);
  else if (mode != EXTERNAL) rectBatch(per * count, mode == SIMD);
  const norm = hann ? 2 / duration : 1 / duration;
  for (let k = 0; k < count; k++) {
    let re = f64At(batchOut, 2 * per * k);
    let im = f64At(batchOut, 2 * per * k + 1);
    if (hann) {
      re = 0.5 * re - 0.25 * (f64At(batchOut, 6 * k + 2) + f64At(batchOut, 6 * k + 4));
      im = 0.5 * im - 0.25 * (f64At(batchOut, 6 * k + 3) + f64At(batchOut, 6 * k + 5));
    }
    // Window-start origin to centred: e^{iωT/2}.
    const turn = f64At(src, k) * 0.5 * duration;
    const c = Math.cos(turn);
    const s = Math.sin(turn);
    setF64(dst, 2 * k, (re * c - im * s) * norm);
    setF64(dst, 2 * k + 1, (re * s + im * c) * norm);
  }
}

/** Centred normalized F, F′, F″ at src[0..count) into derivs[6k ..] (centred Hann: ½D(ω) + ¼D(ω ∓ Δ)). */
function centredDerivatives(src: usize, count: i32, hann: bool, simd: bool): void {
  // (Exact derivatives; FAST refinement uses `differenced` instead.)
  const duration = <f64>faftIntervals();
  const shift = TAU / duration;
  const per = hann ? 3 : 1;
  for (let k = 0; k < count; k++) {
    const omega = f64At(src, k);
    setF64(batchOmegas, per * k, omega);
    if (hann) {
      setF64(batchOmegas, 3 * k + 1, omega - shift);
      setF64(batchOmegas, 3 * k + 2, omega + shift);
    }
  }
  derivativesBatch(per * count, simd);
  const norm = hann ? 2 / duration : 1 / duration;
  for (let k = 0; k < count; k++) {
    for (let f = 0; f < 6; f++) {
      let v = f64At(batchOut, 6 * per * k + f);
      if (hann) v = 0.5 * v + 0.25 * (f64At(batchOut, 6 * (3 * k + 1) + f) + f64At(batchOut, 6 * (3 * k + 2) + f));
      setF64(derivs, 6 * k + f, v * norm);
    }
  }
}

const shifted = memory.data(4096, 16); // ω ∓ h [2 · MAX_TONES]
const shiftedValues = memory.data(8192, 16); // their centred values (re, im)

/**
 * FAST mode: centred normalized F, F′, F″ at sorted[0..count) into derivs from central differences of
 * values at ω ± h (h = FAST_STEP resolutions), F itself being the fit's own `observed`.
 */
function differenced(count: i32, hann: bool, mode: i32): void {
  const h = FAST_STEP * (TAU / <f64>faftIntervals());
  for (let k = 0; k < count; k++) {
    setF64(shifted, 2 * k, f64At(sorted, k) - h);
    setF64(shifted, 2 * k + 1, f64At(sorted, k) + h);
  }
  centredValuesAt(shifted, 2 * count, shiftedValues, hann, mode);
  if (mode == PLAN) return;
  for (let k = 0; k < count; k++) {
    for (let part = 0; part < 2; part++) {
      const below = f64At(shiftedValues, 4 * k + part);
      const above = f64At(shiftedValues, 4 * k + 2 + part);
      const centre = f64At(observed, 2 * k + part);
      setF64(derivs, 6 * k + part, centre);
      setF64(derivs, 6 * k + 2 + part, (above - below) / (2 * h));
      setF64(derivs, 6 * k + 4 + part, (above - 2 * centre + below) / (h * h));
    }
  }
}

// ---- the two-way banded fit ------------------------------------------------------------------------

/** Gram entry of the cosine (sine = false) or sine system for i − width ≤ j ≤ i. */
@inline function gram(sine: bool, i: i32, j: i32): f64 {
  const minus = f64At(kernelMinus, band(i, j));
  const plus = f64At(kernelPlus, band(i, j));
  return 0.5 * (sine ? minus - plus : minus + plus);
}

/**
 * The joint fit at sorted[0..n) (ascending, distinct): band Cholesky per system, the ascending pass
 * (explained energy), the descending pass (amplitudes) and the Takahashi inverse diagonal
 * (significance). Writes the components; false on overflow.
 */
function solve(n: i32, reach: f64, hann: bool, mode: i32): bool {
  centredValuesAt(sorted, n, observed, hann, mode);
  // Bandwidth: most components within `reach` above any component, at most MAX_WIDTH.
  let width = 0;
  let top = 0;
  for (let p = 0; p < n; p++) {
    if (top < p) top = p;
    while (top + 1 < n && f64At(sorted, top + 1) - f64At(sorted, p) <= reach) top++;
    if (top - p > width) width = top - p;
  }
  let truncated = width + 1 < n && isFinite(reach);
  if (width > MAX_WIDTH) {
    width = MAX_WIDTH;
    truncated = true;
  }
  fitWidth = width;
  fitTruncated = truncated;
  // Kernel values within the band, shared by both systems.
  for (let i = 0; i < n; i++) {
    const first = i - width > 0 ? i - width : 0;
    for (let j = first; j <= i; j++) {
      const wi = f64At(sorted, i);
      const wj = f64At(sorted, j);
      let minus: f64 = 0;
      let plus: f64 = 0;
      if (Math.abs(wi - wj) <= reach) {
        kernel(wi - wj, hann);
        minus = kd0;
        kernel(wi + wj, hann);
        plus = kd0;
      }
      setF64(kernelMinus, band(i, j), minus);
      setF64(kernelPlus, band(i, j), plus);
    }
  }
  memory.fill(unique, 0, <usize>n << 3);
  let explained: f64 = 0;
  let degenerate = 0;
  for (let system = 0; system < 2; system++) {
    const sine = system == 1;
    // Factor: L Lᵀ = G within the band; numerically dependent atoms dropped (row of zeros, unit diagonal).
    memory.fill(lowerFactor, 0, <usize>(n * BAND) << 3);
    for (let i = 0; i < n; i++) {
      store<u8>(active + <usize>i, 1);
      const first = i - width > 0 ? i - width : 0;
      for (let j = first; j <= i; j++) {
        if (j < i && load<u8>(active + <usize>j) == 0) continue;
        let v = gram(sine, i, j);
        const from = first > j - width ? first : j - width;
        for (let k = from; k < j; k++) v -= f64At(lowerFactor, band(i, k)) * f64At(lowerFactor, band(j, k));
        if (j < i) {
          setF64(lowerFactor, band(i, j), v / f64At(lowerFactor, band(j, j)));
        } else {
          const diagonal = gram(sine, i, i);
          if (isNaN(v) || diagonal <= 1e-9 || v <= 1e-10 * diagonal) {
            store<u8>(active + <usize>i, 0);
            degenerate++;
            for (let k = first; k < i; k++) setF64(lowerFactor, band(i, k), 0);
            setF64(lowerFactor, band(i, i), 1);
          } else {
            setF64(lowerFactor, band(i, i), Math.sqrt(v));
          }
        }
      }
    }
    // Ascending pass: z = L⁻¹b, the coefficients in the Gram–Schmidt basis in ascending frequency.
    for (let i = 0; i < n; i++) {
      if (load<u8>(active + <usize>i) == 0) {
        setF64(forward, i, 0);
        continue;
      }
      const first = i - width > 0 ? i - width : 0;
      let v = sine ? -f64At(observed, 2 * i + 1) : f64At(observed, 2 * i);
      for (let k = first; k < i; k++) v -= f64At(lowerFactor, band(i, k)) * f64At(forward, k);
      const z = v / f64At(lowerFactor, band(i, i));
      setF64(forward, i, z);
      explained += z * z;
    }
    // Descending pass: back substitution, highest component first.
    const x = amplitudes + (<usize>(system * MAX_TONES) << 3);
    for (let i = n - 1; i >= 0; i--) {
      if (load<u8>(active + <usize>i) == 0) {
        setF64(x, i, 0);
        continue;
      }
      let v = f64At(forward, i);
      const last = i + width < n - 1 ? i + width : n - 1;
      for (let k = i + 1; k <= last; k++) v -= f64At(lowerFactor, band(k, i)) * f64At(x, k);
      setF64(x, i, v / f64At(lowerFactor, band(i, i)));
    }
    // Selected inversion (Takahashi), also descending: Z(j, i) = (δ_ji/L_ii − Σ_{k>i} Z(j, k) L(k, i))/L_ii.
    memory.fill(takahashi, 0, <usize>(n * BAND) << 3);
    for (let i = n - 1; i >= 0; i--) {
      if (load<u8>(active + <usize>i) == 0) continue;
      const last = i + width < n - 1 ? i + width : n - 1;
      const diagonal = f64At(lowerFactor, band(i, i));
      for (let j = last; j >= i; j--) {
        let v: f64 = j == i ? 1 / diagonal : 0;
        for (let k = i + 1; k <= last; k++) {
          const hi = j >= k ? j : k;
          const lo = j >= k ? k : j;
          v -= f64At(takahashi, band(hi, lo)) * f64At(lowerFactor, band(k, i));
        }
        setF64(takahashi, band(j, i), v / diagonal);
      }
      // Removing atom i raises the residual by x_i²/(G⁻¹)_ii.
      const inverse = f64At(takahashi, band(i, i));
      if (inverse > 0) setF64(unique, i, f64At(unique, i) + f64At(x, i) * f64At(x, i) / inverse);
    }
  }
  fitDegenerate = degenerate;
  fitExplained = explained;
  const duration = <f64>faftIntervals();
  for (let p = 0; p < n; p++) {
    const omega = f64At(sorted, p);
    const re = 0.5 * f64At(amplitudes, p);
    const im = -0.5 * f64At(amplitudes, MAX_TONES + p);
    if (!isFinite(re) || !isFinite(im)) return false;
    // Centred to window-start origin: e^{−iωT/2}.
    const turn = -omega * 0.5 * duration;
    const c = Math.cos(turn);
    const s = Math.sin(turn);
    const ore = f64At(observed, 2 * p);
    const oim = f64At(observed, 2 * p + 1);
    const out = components + (<usize>(p * FIT_STRIDE) << 3);
    store<f64>(out, omega);
    store<f64>(out, re * c - im * s, 8);
    store<f64>(out, re * s + im * c, 16);
    store<f64>(out, ore * c - oim * s, 24);
    store<f64>(out, ore * s + oim * c, 32);
    store<f64>(out, Math.sqrt(0.5 * f64At(unique, p)), 40);
  }
  fitCount = n;
  fitHann = hann;
  return true;
}

/** Copies src[0..count) into `sorted`, ascending; false for a bad count, a negative, non-finite or repeated frequency. */
function loadSorted(src: usize, count: i32): bool {
  if (count < 0 || count > MAX_TONES || 6 * count > MAX_BATCH) return false;
  for (let k = 0; k < count; k++) {
    const omega = f64At(src, k);
    if (!isFinite(omega) || omega < 0) return false;
    let j = k;
    while (j > 0 && f64At(sorted, j - 1) > omega) {
      setF64(sorted, j, f64At(sorted, j - 1));
      j--;
    }
    setF64(sorted, j, omega);
  }
  for (let k = 1; k < count; k++) if (f64At(sorted, k) == f64At(sorted, k - 1)) return false;
  return true;
}

/**
 * Joint least-squares fit of real tones at src[0..count) on the prepared FAFT window, with Gram entries
 * kept for tones at most `reach` rad/sample apart (Infinity: the dense fit, up to MAX_WIDTH neighbours).
 * Components, ascending, in faftFitPtr. Returns the count, or −1 for invalid input or overflow.
 */
export function fitTones(src: usize, count: i32, reach: f64, hann: bool, mode: i32): i32 {
  fitCount = 0;
  if (faftIntervals() == 0 || isNaN(reach) || !(reach > 0) || !loadSorted(src, count)) return -1;
  if (count == 0) {
    fitExplained = 0;
    fitDegenerate = 0;
    fitTruncated = false;
    planned = 0;
    return 0;
  }
  if (mode == PLAN) {
    centredValuesAt(sorted, count, observed, hann, PLAN);
    return count;
  }
  return solve(count, reach, hann, mode) ? count : -1;
}

/**
 * `fitTones`, then up to `iterations` rounds moving each frequency by one safeguarded Newton step (at
 * most a quarter resolution) towards the maximum of its own partial residual |a(ω) − Σ_{q≠p} tone_q(ω)|²
 * (the others removed through K, K′, K″), each followed by a refit; stops once no step exceeds
 * `tolerance`. Tones that meet keep one. Returns the count, or −1.
 */
export function refineTones(src: usize, count: i32, reach: f64, iterations: i32, tolerance: f64, hann: bool, mode: i32): i32 {
  if (iterations < 0 || iterations > 64 || !(tolerance > 0)) return -1;
  let n = fitTones(src, count, reach, hann, mode);
  if (n <= 0) return n;
  const duration = <f64>faftIntervals();
  const limit = 0.25 * (TAU / duration);
  for (let round = 0; round < iterations; round++) {
    if (mode == FAST) differenced(n, hann, FAST);
    else centredDerivatives(sorted, n, hann, mode == SIMD);
    const distinct = newtonStep(n, reach, hann, limit);
    n = fitTones(moved, distinct, reach, hann, mode);
    if (n <= 0 || largestStep <= tolerance) break;
  }
  return n;
}

/**
 * One safeguarded Newton step per tone of the current fit (n tones, F, F′, F″ in derivs) towards the
 * maximum of its own partial residual; the moved frequencies, sorted and distinct, into `moved`. Returns
 * their count; the largest move in `largestStep`.
 */
function newtonStep(n: i32, reach: f64, hann: bool, limit: f64): i32 {
  let largest: f64 = 0;
  for (let p = 0; p < n; p++) {
    const omega = f64At(sorted, p);
    let r0re = f64At(derivs, 6 * p), r0im = f64At(derivs, 6 * p + 1);
    let r1re = f64At(derivs, 6 * p + 2), r1im = f64At(derivs, 6 * p + 3);
    let r2re = f64At(derivs, 6 * p + 4), r2im = f64At(derivs, 6 * p + 5);
    for (let q = 0; q < n; q++) {
      const wq = f64At(sorted, q);
      if (q == p || Math.abs(omega - wq) > reach) continue;
      // Tone q's centred value c = (α/2, −β/2) and its image c̄ at −ω_q.
      const cre = 0.5 * f64At(amplitudes, q);
      const cim = -0.5 * f64At(amplitudes, MAX_TONES + q);
      kernel(omega - wq, hann);
      const b0 = kd0, b1 = kd1, b2 = kd2;
      kernel(omega + wq, hann);
      r0re -= cre * (b0 + kd0); r0im -= cim * (b0 - kd0);
      r1re -= cre * (b1 + kd1); r1im -= cim * (b1 - kd1);
      r2re -= cre * (b2 + kd2); r2im -= cim * (b2 - kd2);
    }
    // G = |r|²: G′ = 2 Re(r̄₀r₁), G″ = 2(|r₁|² + Re(r̄₀r₂)).
    const g1 = 2 * (r0re * r1re + r0im * r1im);
    const g2 = 2 * (r1re * r1re + r1im * r1im + r0re * r2re + r0im * r2im);
    let step: f64 = 0;
    if (g2 < 0) step = Math.min(limit, Math.max(-limit, -g1 / g2));
    const next = Math.max(0, omega + step);
    largest = Math.max(largest, Math.abs(next - omega));
    setF64(moved, p, next);
  }
  // Sorted again; tones that converge onto one frequency keep one.
  let kept = 0;
  for (let p = 0; p < n; p++) {
    const omega = f64At(moved, p);
    let j = kept;
    while (j > 0 && f64At(moved, j - 1) > omega) {
      setF64(moved, j, f64At(moved, j - 1));
      j--;
    }
    setF64(moved, j, omega);
    kept++;
  }
  let distinct = 0;
  for (let p = 0; p < kept; p++) {
    if (distinct > 0 && Math.abs(f64At(moved, p) - f64At(moved, distinct - 1)) < 1e-12) continue;
    setF64(moved, distinct, f64At(moved, p));
    distinct++;
  }
  largestStep = largest;
  return distinct;
}

// ---- energy and the residual bound -----------------------------------------------------------------

// 16-point Gauss–Legendre nodes and weights on [0, 1], built on first use.
const gauss = memory.data(256, 16); // (node, weight) × 16
let gaussReady = false;

function gaussLegendre(): void {
  const n = 16;
  for (let i = 0; i < n; i++) {
    let x = Math.cos((Math.PI * (<f64>i + 0.75)) / (<f64>n + 0.5));
    let derivative: f64 = 1;
    for (let iteration = 0; iteration < 100; iteration++) {
      let p0: f64 = 1;
      let p1 = x;
      for (let k = 2; k <= n; k++) {
        const p2 = (<f64>(2 * k - 1) * x * p1 - <f64>(k - 1) * p0) / <f64>k;
        p0 = p1;
        p1 = p2;
      }
      derivative = (<f64>n * (x * p1 - p0)) / (x * x - 1);
      const dx = p1 / derivative;
      x -= dx;
      if (Math.abs(dx) < 1e-16) break;
    }
    setF64(gauss, 2 * i, 0.5 * (1 + x));
    setF64(gauss, 2 * i + 1, 1 / ((1 - x * x) * derivative * derivative));
  }
  gaussReady = true;
}

/**
 * ∫ y(t)² w(t) dt over the prepared window (amplitude² · samples): 16-point Gauss–Legendre panels on each
 * interval with at most π of phase each (2·|rate| plus the Hann rate), exact to rounding. NaN without a window.
 */
export function faftEnergy(hann: bool): f64 {
  const count = faftIntervals();
  if (count == 0) return NaN;
  if (!gaussReady) gaussLegendre();
  const atoms = faftAtomsPtr();
  const stride = faftAtomStride();
  const shift = TAU / <f64>count;
  let total: f64 = 0;
  for (let n = 0; n < count; n++) {
    const row = atoms + (<usize>(n * stride) << 3);
    const y0 = load<f64>(row);
    const slope = load<f64>(row, 8);
    // Arcs at fields 2 and 2 + ARC (= 11): a, b, rate.
    const a0 = load<f64>(row, 16), b0 = load<f64>(row, 24), r0 = load<f64>(row, 32);
    const a1 = load<f64>(row, 88), b1 = load<f64>(row, 96), r1 = load<f64>(row, 104);
    const rate = Math.max(Math.abs(r0), Math.abs(r1));
    const panels = <i32>Math.max(1, Math.ceil((2 * rate + (hann ? shift : 0)) / Math.PI));
    const width = 1 / <f64>panels;
    let sum: f64 = 0;
    for (let p = 0; p < panels; p++) {
      for (let g = 0; g < 16; g++) {
        const u = width * (<f64>p + f64At(gauss, 2 * g));
        let y = y0 + slope * u;
        if (a0 != 0 || b0 != 0) {
          const half = Math.sin(0.5 * r0 * u);
          y += -2 * a0 * half * half + b0 * Math.sin(r0 * u);
        }
        if (a1 != 0 || b1 != 0) {
          const half = Math.sin(0.5 * r1 * u);
          y += -2 * a1 * half * half + b1 * Math.sin(r1 * u);
        }
        const w: f64 = hann ? 0.5 - 0.5 * Math.cos(shift * (<f64>n + u)) : 1;
        sum += f64At(gauss, 2 * g + 1) * y * y * w;
      }
    }
    total += sum * width;
  }
  return total;
}

/**
 * The residual energy of the last fit given the window's `energy` (faftEnergy with the same weighting):
 * E − W·Σz² for an exact (untruncated) fit, or the full quadratic form E − 2W·xᵀb + W·xᵀGx of both systems
 * when the band truncated the Gram matrix (O(n²) kernel values). Never below 0.
 */
export function faftFitResidual(energy: f64): f64 {
  const duration = <f64>faftIntervals();
  const w = fitHann ? 0.5 * duration : duration;
  if (!fitTruncated) return Math.max(0, energy - w * fitExplained);
  let quadratic: f64 = 0;
  for (let system = 0; system < 2; system++) {
    const sine = system == 1;
    const x = amplitudes + (<usize>(system * MAX_TONES) << 3);
    for (let p = 0; p < fitCount; p++) {
      const b = sine ? -f64At(observed, 2 * p + 1) : f64At(observed, 2 * p);
      quadratic -= 2 * f64At(x, p) * b;
      for (let q = 0; q < fitCount; q++) {
        const wp = f64At(sorted, p);
        const wq = f64At(sorted, q);
        kernel(wp - wq, fitHann);
        const minus = kd0;
        kernel(wp + wq, fitHann);
        const g = 0.5 * (sine ? minus - kd0 : minus + kd0);
        quadratic += f64At(x, p) * g * f64At(x, q);
      }
    }
  }
  return Math.max(0, energy + w * quadratic);
}

// ---- exports ---------------------------------------------------------------------------------------

/**
 * `fitTones` on the frequencies in faftOmegasPtr; `mode` SCALAR (0), SIMD (1), FAST (2), or a host's
 * PLAN (3: lists the values it needs, returns the count) and EXTERNAL (4: fits with the values in batchOut).
 */
export function faftFit(count: i32, reach: f64, hann: bool, mode: i32): i32 {
  if (mode < SCALAR || mode > EXTERNAL) return -1;
  return fitTones(faftOmegasPtr(), count, reach, hann, mode);
}

/** `refineTones` on the frequencies in faftOmegasPtr; `mode` as for faftFit. */
export function faftFitRefined(count: i32, reach: f64, iterations: i32, tolerance: f64, hann: bool, mode: i32): i32 {
  if (mode < SCALAR || mode > FAST) return -1;
  return refineTones(faftOmegasPtr(), count, reach, iterations, tolerance, hann, mode);
}

/**
 * One refinement round of FAST `refineTones` on the current fit, for a host evaluating the values
 * elsewhere: with PLAN, lists the values the round needs (batchOmegas, `tonePlanned`); with EXTERNAL, takes
 * them from batchOut and writes the moved frequencies to `toneMovedPtr`, returning their count (the
 * largest move in `toneLargestStep`). The host then fits them (`fitTones`), as `refineTones` does. −1
 * without a fit or for another mode.
 */
export function refineStep(reach: f64, hann: bool, mode: i32): i32 {
  if (fitCount <= 0 || (mode != PLAN && mode != EXTERNAL)) return -1;
  differenced(fitCount, hann, mode);
  if (mode == PLAN) return planned;
  return newtonStep(fitCount, reach, hann, 0.25 * (TAU / <f64>faftIntervals()));
}

/** Rectangular frequencies the last PLAN step listed in the FAFT batch (for the host to evaluate). */
export function tonePlanned(): i32 {
  return planned;
}

export function toneMovedPtr(): usize {
  return moved;
}

export function toneLargestStep(): f64 {
  return largestStep;
}

/**
 * Centred normalized F, F′, F″ (re, im; sinwhisky.rs `Faft::derivatives`) at the frequencies in
 * faftOmegasPtr into faftDerivativesPtr, 6 f64 each. False without a window, for a non-finite frequency or
 * more than MAX_TONES.
 */
export function faftDerivatives(count: i32, hann: bool, simd: bool): bool {
  if (faftIntervals() == 0 || count < 0 || count > MAX_TONES) return false;
  const src = faftOmegasPtr();
  for (let k = 0; k < count; k++) if (!isFinite(f64At(src, k))) return false;
  centredDerivatives(src, count, hann, simd);
  return true;
}

export function faftDerivativesPtr(): usize {
  return derivs;
}

export function faftFitPtr(): usize {
  return components;
}

export function faftFitStride(): i32 {
  return FIT_STRIDE;
}

export function faftFitCount(): i32 {
  return fitCount;
}

export function faftFitDegenerate(): i32 {
  return fitDegenerate;
}

/** Σz² of the last fit (normalized; the explained energy is W times it when the fit was exact). */
export function faftFitExplained(): f64 {
  return fitExplained;
}

/** Whether the last fit's band truncated the Gram matrix (reach, or MAX_WIDTH). */
export function faftFitTruncated(): bool {
  return fitTruncated;
}

export function faftMaxTones(): i32 {
  return MAX_TONES;
}

// ---- the last fit, for the equalizer (assembly/eq.ts) ----------------------------------------------

/** Frequency of fitted tone p (ascending). */
export function toneOmega(p: i32): f64 {
  return f64At(sorted, p);
}

/** Cosine and sine amplitudes of tone p around the window centre: α·cos(ωu) + β·sin(ωu). */
export function toneAlpha(p: i32): f64 {
  return f64At(amplitudes, p);
}

export function toneBeta(p: i32): f64 {
  return f64At(amplitudes, MAX_TONES + p);
}

export function toneSignificance(p: i32): f64 {
  return f64At(components, p * FIT_STRIDE + 5);
}

/** |a(ω_p)|: the normalized transform's magnitude at tone p. */
export function toneObserved(p: i32): f64 {
  return Math.sqrt(f64At(observed, 2 * p) * f64At(observed, 2 * p) + f64At(observed, 2 * p + 1) * f64At(observed, 2 * p + 1));
}
