// Transposed from appx/dsp at ec84edcd331af8810be9df42cf5d16f3a348d6d5 (2026-10-05).
// The component equalizer of R. Cecchini's sinwhisky.rs (src/bands.rs, src/analog.rs;
// PAPER_EQ_COMPRESSION.md §§2–3), as a real-time stream with a fixed latency.
//
// A band divider that assigns fitted components, not spectrum bins: in Hann windows of T samples
// advancing by H = T/2, tones are fitted jointly on SinWhisky's circle curve (assembly/tones.ts, FAST
// mode: four f32 lanes, about 1e-6), each assigned whole to its band by crossover weights that sum to one
// (brick wall or raised cosine), and overlap-added (the periodic Hann windows sum to one). The residual,
// the source minus the tonal track, is divided by Butterworth low-passes applied exactly to its line
// curve (the per-pole state carried from sample to sample in closed form, analog.rs), as complementary
// differences: the last band takes the exact remainder, so the bands add back to the source whatever the
// fit. Equalization is a weighted sum of the bands; extrusion keeps one band and its remainder.
//
// Streaming. Window k (centred at kH) is fitted once its last sample and one sample of context on each
// side have arrived (its curve is then the whole signal's curve, cropped). The residual filters are
// zero-phase (|H|², as sinwhisky.rs's forward–backward pass) with a backward pass started `lookahead`
// hops ahead from a zero state (its truncation error decays at the slowest pole's rate), or causal
// (lookahead 0: response H, same completeness). Forward states are kept every CHECK samples and rebuilt in
// between for the backward pass. Latency: (lookahead + 2)·H + 2 samples. Gains ramp linearly over each
// output block. Up to MAX_CHANNELS channels, each with its own state; static buffers only.
//
// Departures from sinwhisky.rs (bands.rs), each measured in dsp/AGENTS.md: the curve is SinWhisky's
// triplet circles (the FAFT kernel's), not the hierarchical fit; seeds are the Hann FFT's local maxima,
// moved onto the tones by the joint refinement, instead of the exact |FAFT| maxima (the maxima carry
// their neighbours' leakage anyway); refinement differentiates values instead of the exact derivative
// kernel; the zero-phase pass is truncated at the lookahead.

import { faftCrop, faftInputPtr, faftIntervals, faftPrepare } from './faft';
import { fftInit, fftInputPtr, fftSize, fftSpectrumDb, fftSpectrumPtr } from './fft';
import { FAST, fitTones, refineTones, toneAlpha, toneBeta, toneObserved, toneOmega, toneSignificance } from './tones';

const MAX_CHANNELS: i32 = 2;
const MAX_WINDOW: i32 = 2048;
const MAX_HOP: i32 = 1024;
const MAX_BANDS: i32 = 6;
const MAX_EDGES: i32 = MAX_BANDS - 1;
const MAX_ORDER: i32 = 8;
const MAX_LOOKAHEAD: i32 = 3;
const MAX_BLOCK: i32 = 4096;
const MAX_SEEDS: i32 = 256;
/** Rings indexed by absolute sample time modulo their size (powers of two). */
const RING: i32 = 8192;
const MASK: i32 = RING - 1;
const X_RING: i32 = 4096;
const X_MASK: i32 = X_RING - 1;
const OUT_RING: i32 = 4096;
const OUT_MASK: i32 = OUT_RING - 1;
/** Forward residual-filter states are kept every CHECK samples, NCHECK of them. */
const CHECK: i32 = 64;
const NCHECK: i32 = 128;
const TAU: f64 = 2 * Math.PI;

// ---- configuration ---------------------------------------------------------------------------------

const edges = memory.data(64, 16); // f64[MAX_EDGES], written by the caller before eqInit
const gainsIn = memory.data(64, 16); // f64[MAX_BANDS], written by the caller before eqSetGains
const target = memory.data(64, 16); // f64[MAX_BANDS]

let configured = false;
let channels: i32 = 0;
let windowSize: i32 = 0;
let hop: i32 = 0;
let bands: i32 = 1;
let order: i32 = 4;
let lookahead: i32 = 2;
let latency: i32 = 0;
let upper: f64 = 0;
let relativeFloor: f64 = 0.01;
let maxTones: i32 = 64;
let reach: f64 = 0;
let refine: i32 = 1;
let halfWidth: f64 = 0;
let truncationDb: f64 = 0;

// Per edge and pole: the pole p, residue ρ, the forward step (e^p, φ₁(p), φ₂(p)) and the backward
// constants (α, β, and γ against every pole), complex, f64 (re, im).
const POLE_FIELDS: i32 = 12; // p, ρ, e^p, φ₁, φ₂, (α, β) as 7 complex, padded
const poles = memory.data(7680, 16); // [MAX_EDGES][MAX_ORDER][POLE_FIELDS · 2] f64 (7 complex used of 12)
const gammas = memory.data(5120, 16); // [MAX_EDGES][MAX_ORDER][MAX_ORDER] complex

@inline function f64At(base: usize, index: i32): f64 {
  return load<f64>(base + (<usize>index << 3));
}

@inline function setF64(base: usize, index: i32, value: f64): void {
  store<f64>(base + (<usize>index << 3), value);
}

@inline function poleSlot(edge: i32, pole: i32, field: i32): usize {
  return poles + (<usize>(((edge * MAX_ORDER + pole) * POLE_FIELDS + field) * 2) << 3);
}

@inline function gammaSlot(edge: i32, k: i32, j: i32): usize {
  return gammas + (<usize>(((edge * MAX_ORDER + k) * MAX_ORDER + j) * 2) << 3);
}

// ---- per-channel state -----------------------------------------------------------------------------

const xRing = memory.data(65536, 16); // [MAX_CHANNELS][X_RING] f64: the source
const rRing = memory.data(131072, 16); // [MAX_CHANNELS][RING] f64: the residual
const tonalRing = memory.data(393216, 16); // [MAX_CHANNELS][MAX_BANDS][RING] f32: tonal tracks
const outRing = memory.data(32768, 16); // [MAX_CHANNELS][OUT_RING] f32: the equalized output
const bandOut = memory.data(196608, 16); // [MAX_CHANNELS][MAX_BANDS][OUT_RING] f32: the bands
const state = memory.data(1280, 16); // [MAX_CHANNELS][MAX_EDGES][MAX_ORDER] complex: forward ξ
const checks = memory.data(163840, 16); // [MAX_CHANNELS][MAX_EDGES][NCHECK][MAX_ORDER] complex
const current = memory.data(96, 16); // [MAX_CHANNELS][MAX_BANDS] f64: ramped gains
const received = memory.data(16, 16); // i64[MAX_CHANNELS]: samples received
const hops = memory.data(16, 16); // i64[MAX_CHANNELS]: next hop index
const windowsRun = memory.data(16, 16); // i64[MAX_CHANNELS]
const tonesRun = memory.data(16, 16); // i64[MAX_CHANNELS]
const lastTones = memory.data(8, 16); // i32[MAX_CHANNELS]

// ---- scratch shared by the channels ----------------------------------------------------------------

const hann = memory.data(16384, 16); // f64[MAX_WINDOW]: 0.5 + 0.5·cos(2π(i − H)/T)
const windowTonal = memory.data(98304, 16); // [MAX_BANDS][MAX_WINDOW] f64: this window's tones
const seeds = memory.data(2048, 16); // f64[MAX_SEEDS]
const seedLevel = memory.data(2048, 16); // f64[MAX_SEEDS] (dB)
const kept = memory.data(2048, 16); // f64[MAX_SEEDS]
const lows = memory.data(40960, 16); // [MAX_EDGES][MAX_HOP] f64: the block's low-passed residual
const segment = memory.data(8320, 16); // [CHECK + 1][MAX_ORDER] complex: ξ rebuilt for one segment
const eta = memory.data(128, 16); // [MAX_ORDER] complex: the backward state

// ---- I/O -------------------------------------------------------------------------------------------

const input = memory.data(16384, 16); // f32[MAX_BLOCK]
const output = memory.data(16384, 16); // f32[MAX_BLOCK]
const bandsOutput = memory.data(98304, 16); // f32[MAX_BANDS][MAX_BLOCK]

export function eqInputPtr(): usize {
  return input;
}

export function eqOutputPtr(): usize {
  return output;
}

export function eqBandsPtr(): usize {
  return bandsOutput;
}

export function eqEdgesPtr(): usize {
  return edges;
}

export function eqGainsPtr(): usize {
  return gainsIn;
}

export function eqMaxBlock(): i32 {
  return MAX_BLOCK;
}

export function eqMaxBands(): i32 {
  return MAX_BANDS;
}

export function eqMaxChannels(): i32 {
  return MAX_CHANNELS;
}

export function eqLatency(): i32 {
  return configured ? latency : -1;
}

export function eqBands(): i32 {
  return bands;
}

/** 20·log10 of the zero-phase pass's truncation, e^{Re(p)·L} at the slowest pole (0 when causal). */
export function eqTruncationDb(): f64 {
  return truncationDb;
}

export function eqLastTones(channel: i32): i32 {
  return channel >= 0 && channel < channels ? load<i32>(lastTones + (<usize>channel << 2)) : -1;
}

export function eqWindows(channel: i32): f64 {
  return channel >= 0 && channel < channels ? <f64>load<i64>(windowsRun + (<usize>channel << 3)) : -1;
}

export function eqTones(channel: i32): f64 {
  return channel >= 0 && channel < channels ? <f64>load<i64>(tonesRun + (<usize>channel << 3)) : -1;
}

// ---- the batch path ---------------------------------------------------------------------------------

// A host that fits many windows at once (values on the GPU, ports/wasm-simd/signal-processor.mjs `equalizeBatch`) hands each
// window's tones to the stream through this slot: count, then (ω, α, β) per tone.
const toneSlot = memory.data(6152, 16); // f64[1 + 3 · MAX_SEEDS]
let useToneSlot = false;

/** With `on`, each window synthesizes the tones in eqToneSlotPtr instead of fitting its own. */
export function eqUseToneSlot(on: bool): void {
  useToneSlot = on;
}

export function eqToneSlotPtr(): usize {
  return toneSlot;
}

/**
 * The batch path's first step for one window: its curve (T + 3 samples, faftInputPtr) prepared as the
 * FAFT window, and its seeds in eqSeedsPtr, as `analyse` does. Returns the seed count (0 for none), or −1
 * when not configured.
 */
export function eqWindowSeeds(): i32 {
  return configured ? windowSeeds() : -1;
}

export function eqSeedsPtr(): usize {
  return seeds;
}

/** The fit's parameters as configured (rad/sample, rounds, the relative floor). */
export function eqReach(): f64 {
  return reach;
}

export function eqRefine(): i32 {
  return refine;
}

export function eqRelativeFloor(): f64 {
  return relativeFloor;
}

// ---- complex helpers -------------------------------------------------------------------------------

let zr: f64 = 0;
let zi: f64 = 0;

/** e^z into zr, zi. */
@inline function cexp(re: f64, im: f64): void {
  const m = Math.exp(re);
  zr = m * Math.cos(im);
  zi = m * Math.sin(im);
}

/** (e^z − 1)/z (k = 1) or (e^z − 1 − z)/z² (k = 2), series under |z| < 0.5 (analog.rs `phi1`, `phi2`). */
function phi(re: f64, im: f64, k: i32): void {
  if (Math.sqrt(re * re + im * im) < 0.5) {
    // Σ zⁿ/(n + k)!, n ≥ 0.
    let termRe: f64 = k == 1 ? 1 : 0.5;
    let termIm: f64 = 0;
    let sumRe = termRe;
    let sumIm: f64 = 0;
    for (let n = 1; n < 24; n++) {
      const scale: f64 = 1 / <f64>(n + k);
      const nextRe = (termRe * re - termIm * im) * scale;
      termIm = (termRe * im + termIm * re) * scale;
      termRe = nextRe;
      sumRe += termRe;
      sumIm += termIm;
    }
    zr = sumRe;
    zi = sumIm;
    return;
  }
  cexp(re, im);
  let nre = zr - 1;
  let nim = zi;
  if (k == 2) nre -= re;
  // Divide by z^k.
  let dre = re;
  let dim = im;
  if (k == 2) {
    const t = re * re - im * im;
    dim = 2 * re * im;
    dre = t;
  }
  const d = dre * dre + dim * dim;
  zr = (nre * dre + nim * dim) / d;
  zi = (nim * dre - nre * dim) / d;
}

// 32-point Gauss–Legendre on [0, 1] for the backward constants (built in eqInit).
const gauss = memory.data(512, 16);

function gaussLegendre(): void {
  const n = 32;
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
}

// ---- configuration ---------------------------------------------------------------------------------

/** The Butterworth low-pass of `order` with cutoff ωc: poles, residues and every constant of both passes. */
function prepareEdge(edge: i32, cutoff: f64): void {
  const n = order;
  // Poles ωc·e^{iπ(2k + n + 1)/(2n)}, conjugates paired (analog.rs `butterworth_lowpass`).
  for (let k = 0; k < (n + 1) / 2; k++) {
    const angle = (Math.PI * <f64>(2 * k + n + 1)) / <f64>(2 * n);
    const re = cutoff * Math.cos(angle);
    const im = cutoff * Math.sin(angle);
    setF64(poleSlot(edge, k, 0), 0, re);
    setF64(poleSlot(edge, k, 0), 1, im);
    setF64(poleSlot(edge, n - 1 - k, 0), 0, re);
    setF64(poleSlot(edge, n - 1 - k, 0), 1, -im);
  }
  if ((n & 1) == 1) {
    setF64(poleSlot(edge, n / 2, 0), 0, -cutoff);
    setF64(poleSlot(edge, n / 2, 0), 1, 0);
  }
  // Residues ωcⁿ / Π_{j≠k}(p_k − p_j), then made exactly conjugate.
  const gain = Math.pow(cutoff, <f64>n);
  for (let k = 0; k < n; k++) {
    let pr: f64 = 1;
    let pi: f64 = 0;
    const kr = f64At(poleSlot(edge, k, 0), 0);
    const ki = f64At(poleSlot(edge, k, 0), 1);
    for (let j = 0; j < n; j++) {
      if (j == k) continue;
      const dr = kr - f64At(poleSlot(edge, j, 0), 0);
      const di = ki - f64At(poleSlot(edge, j, 0), 1);
      const t = pr * dr - pi * di;
      pi = pr * di + pi * dr;
      pr = t;
    }
    const d = pr * pr + pi * pi;
    setF64(poleSlot(edge, k, 1), 0, (gain * pr) / d);
    setF64(poleSlot(edge, k, 1), 1, (-gain * pi) / d);
  }
  for (let k = 0; k < n / 2; k++) {
    setF64(poleSlot(edge, n - 1 - k, 1), 0, f64At(poleSlot(edge, k, 1), 0));
    setF64(poleSlot(edge, n - 1 - k, 1), 1, -f64At(poleSlot(edge, k, 1), 1));
  }
  if ((n & 1) == 1) setF64(poleSlot(edge, n / 2, 1), 1, 0);
  for (let k = 0; k < n; k++) {
    const pr = f64At(poleSlot(edge, k, 0), 0);
    const pi = f64At(poleSlot(edge, k, 0), 1);
    cexp(pr, pi);
    setF64(poleSlot(edge, k, 2), 0, zr);
    setF64(poleSlot(edge, k, 2), 1, zi);
    phi(pr, pi, 1);
    setF64(poleSlot(edge, k, 3), 0, zr);
    setF64(poleSlot(edge, k, 3), 1, zi);
    phi(pr, pi, 2);
    setF64(poleSlot(edge, k, 4), 0, zr);
    setF64(poleSlot(edge, k, 4), 1, zi);
  }
  // Backward constants for pole k (q = p_k): over one piece, ∫₀¹ e^{qu} f(n + u) du with
  // f(n + u) = Σ_j ρ_j ξ_j(n + u) and ξ_j(n + u) = e^{p_j u}ξ_j(n) + c₀·u·φ₁(p_j u) + c₁·u²·φ₂(p_j u):
  // α_k c₀ + β_k c₁ + Σ_j γ_kj ξ_j(n), with α_k = Σ_j ρ_j ∫ e^{qu} u φ₁(p_j u) du,
  // β_k = Σ_j ρ_j ∫ e^{qu} u² φ₂(p_j u) du and γ_kj = ρ_j φ₁(q + p_j). Integrals by 32-point
  // Gauss–Legendre (the integrands are entire and slowly varying: exact to rounding).
  for (let k = 0; k < n; k++) {
    const qr = f64At(poleSlot(edge, k, 0), 0);
    const qi = f64At(poleSlot(edge, k, 0), 1);
    let ar: f64 = 0, ai: f64 = 0, br: f64 = 0, bi: f64 = 0;
    for (let j = 0; j < n; j++) {
      const pr = f64At(poleSlot(edge, j, 0), 0);
      const pi = f64At(poleSlot(edge, j, 0), 1);
      const rr = f64At(poleSlot(edge, j, 1), 0);
      const ri = f64At(poleSlot(edge, j, 1), 1);
      let j1r: f64 = 0, j1i: f64 = 0, j2r: f64 = 0, j2i: f64 = 0;
      for (let g = 0; g < 32; g++) {
        const u = f64At(gauss, 2 * g);
        const w = f64At(gauss, 2 * g + 1);
        cexp(qr * u, qi * u);
        const er = zr, ei = zi;
        phi(pr * u, pi * u, 1);
        const f1r = zr * u, f1i = zi * u;
        phi(pr * u, pi * u, 2);
        const f2r = zr * u * u, f2i = zi * u * u;
        j1r += w * (er * f1r - ei * f1i);
        j1i += w * (er * f1i + ei * f1r);
        j2r += w * (er * f2r - ei * f2i);
        j2i += w * (er * f2i + ei * f2r);
      }
      ar += rr * j1r - ri * j1i;
      ai += rr * j1i + ri * j1r;
      br += rr * j2r - ri * j2i;
      bi += rr * j2i + ri * j2r;
      phi(qr + pr, qi + pi, 1);
      setF64(gammaSlot(edge, k, j), 0, rr * zr - ri * zi);
      setF64(gammaSlot(edge, k, j), 1, rr * zi + ri * zr);
    }
    setF64(poleSlot(edge, k, 5), 0, ar);
    setF64(poleSlot(edge, k, 5), 1, ai);
    setF64(poleSlot(edge, k, 6), 0, br);
    setF64(poleSlot(edge, k, 6), 1, bi);
  }
}

/**
 * Configures the equalizer and clears every channel. `window` T is a power of two, 256..2048 (hop T/2);
 * `upper` the highest fitted frequency (rad/sample, (0, π]); `relativeFloor` the share of the strongest
 * seed below which seeds and fitted tones are dropped; `maxTones` 0..256 (0: no tones, the residual
 * filters alone, an exact Butterworth crossover); `reachBins` the banded fit's
 * reach in resolutions 2π/T; `refine` Newton rounds 0..8; the band edges are eqEdgesPtr[0..edgeCount)
 * (strictly increasing, in (0, π], at most MAX_BANDS − 1); `halfWidth` 0 for brick-wall crossovers or the
 * raised cosine's half width (rad/sample); `order` the residual Butterworth order 1..8; `lookaheadHops`
 * 0 (causal residual filters) or 1..3 (zero-phase, the backward pass that many hops ahead). Gains start
 * at 1. Returns the latency in samples, or −1 for invalid settings.
 */
export function eqInit(
  channelCount: i32, window: i32, upperOmega: f64, floor: f64, tones: i32, reachBins: f64, rounds: i32,
  edgeCount: i32, crossoverHalfWidth: f64, residualOrder: i32, lookaheadHops: i32,
): i32 {
  configured = false;
  if (channelCount < 1 || channelCount > MAX_CHANNELS) return -1;
  if (window < 256 || window > MAX_WINDOW || (window & (window - 1)) != 0) return -1;
  if (!(upperOmega > 0 && upperOmega <= Math.PI) || !(floor > 0 && floor < 1)) return -1;
  if (tones < 0 || tones > MAX_SEEDS || !(reachBins > 0) || rounds < 0 || rounds > 8) return -1;
  if (edgeCount < 0 || edgeCount > MAX_EDGES || !(crossoverHalfWidth >= 0) || !isFinite(crossoverHalfWidth)) return -1;
  if (residualOrder < 1 || residualOrder > MAX_ORDER || lookaheadHops < 0 || lookaheadHops > MAX_LOOKAHEAD) return -1;
  for (let e = 0; e < edgeCount; e++) {
    const edge = f64At(edges, e);
    if (!(edge > 0 && edge <= Math.PI)) return -1;
    if (e > 0 && !(edge > f64At(edges, e - 1))) return -1;
  }
  channels = channelCount;
  windowSize = window;
  hop = window / 2;
  bands = edgeCount + 1;
  order = residualOrder;
  lookahead = lookaheadHops;
  latency = (lookahead + 2) * hop + 2;
  upper = upperOmega;
  relativeFloor = floor;
  maxTones = tones;
  reach = reachBins * (TAU / <f64>window);
  refine = rounds;
  halfWidth = crossoverHalfWidth;
  gaussLegendre();
  let slowest: f64 = -Infinity;
  for (let e = 0; e < edgeCount; e++) {
    prepareEdge(e, f64At(edges, e));
    for (let k = 0; k < order; k++) slowest = Math.max(slowest, f64At(poleSlot(e, k, 0), 0));
  }
  truncationDb = lookahead > 0 && edgeCount > 0 ? (20 / Math.LN10) * slowest * <f64>(lookahead * hop) : 0;
  for (let i = 0; i < window; i++) setF64(hann, i, 0.5 + 0.5 * Math.cos((TAU * <f64>(i - hop)) / <f64>window));
  // Clear every channel.
  memory.fill(xRing, 0, 65536);
  memory.fill(rRing, 0, 131072);
  memory.fill(tonalRing, 0, 393216);
  memory.fill(outRing, 0, 32768);
  memory.fill(bandOut, 0, 196608);
  memory.fill(state, 0, 1280);
  memory.fill(checks, 0, 163840);
  for (let c = 0; c < MAX_CHANNELS; c++) {
    store<i64>(received + (<usize>c << 3), 0);
    store<i64>(hops + (<usize>c << 3), 0);
    store<i64>(windowsRun + (<usize>c << 3), 0);
    store<i64>(tonesRun + (<usize>c << 3), 0);
    store<i32>(lastTones + (<usize>c << 2), 0);
    for (let b = 0; b < MAX_BANDS; b++) setF64(current, c * MAX_BANDS + b, 1);
  }
  for (let b = 0; b < MAX_BANDS; b++) setF64(target, b, 1);
  configured = true;
  return latency;
}

/** Sets the band gains (linear, finite) from eqGainsPtr[0..bands); they ramp over the next output block. */
export function eqSetGains(): bool {
  if (!configured) return false;
  for (let b = 0; b < bands; b++) if (!isFinite(f64At(gainsIn, b))) return false;
  for (let b = 0; b < bands; b++) setF64(target, b, f64At(gainsIn, b));
  return true;
}

/** Crossover weight of band b for a tone at ω (bands.rs `BandSplit::weight`): nonnegative, summing to one. */
function below(edge: f64, omega: f64): f64 {
  if (halfWidth == 0) return omega < edge ? 1 : 0;
  const x = (omega - edge + halfWidth) / (2 * halfWidth);
  if (x <= 0) return 1;
  if (x >= 1) return 0;
  return 0.5 * (1 + Math.cos(Math.PI * x));
}

function weight(band: i32, omega: f64): f64 {
  const top: f64 = band < bands - 1 ? below(f64At(edges, band), omega) : 1;
  const bottom: f64 = band == 0 ? 0 : below(f64At(edges, band - 1), omega);
  return Math.max(0, top - bottom);
}

// ---- ring access -----------------------------------------------------------------------------------

@inline function xAt(channel: i32, t: i64): f64 {
  return t < 0 ? 0 : load<f64>(xRing + (<usize>(channel * X_RING + (<i32>t & X_MASK)) << 3));
}

@inline function rSlot(channel: i32, t: i64): usize {
  return rRing + (<usize>(channel * RING + (<i32>t & MASK)) << 3);
}

@inline function rAt(channel: i32, t: i64): f64 {
  return t < 0 ? 0 : load<f64>(rSlot(channel, t));
}

@inline function tonalSlot(channel: i32, band: i32, t: i64): usize {
  return tonalRing + (<usize>((channel * MAX_BANDS + band) * RING + (<i32>t & MASK)) << 2);
}

@inline function stateSlot(channel: i32, edge: i32, pole: i32): usize {
  return state + (<usize>(((channel * MAX_EDGES + edge) * MAX_ORDER + pole) * 2) << 3);
}

@inline function checkSlot(channel: i32, edge: i32, t: i64, pole: i32): usize {
  const index = <i32>(t / CHECK) & (NCHECK - 1);
  return checks + (<usize>((((channel * MAX_EDGES + edge) * NCHECK + index) * MAX_ORDER + pole) * 2) << 3);
}

// ---- one hop ---------------------------------------------------------------------------------------

/** Fits window k (centred at kH) and adds its tones, by band, into the tonal rings. */
/**
 * The FAFT window and its seeds, from the curve in the FAFT input (T + 3 samples, one of context on each
 * side): prepares the window cropped to its T intervals, and writes the strongest local maxima of its
 * Hann FFT (at most maxTones, distinct) to `seeds`. Returns their count; 0 for a silent window, with no
 * tones asked for, or nothing above the floor.
 */
function windowSeeds(): i32 {
  const T = windowSize;
  const curve = faftInputPtr();
  let loud: f64 = 0;
  for (let i = 0; i < T + 3; i++) loud = Math.max(loud, Math.abs(<f64>load<f32>(curve + (<usize>i << 2))));
  if (!(loud > 1e-9) || maxTones == 0) return 0;
  if (faftPrepare(T + 3) < 0 || !faftCrop(1, T)) return 0;
  // Seeds: local maxima of the Hann FFT of the window's samples, by parabola on dB.
  const log2 = <i32>(31 - clz<i32>(T));
  if (fftSize() != T) fftInit(log2);
  const fin = fftInputPtr();
  for (let i = 0; i < T; i++) store<f32>(fin + (<usize>i << 2), load<f32>(curve + (<usize>(i + 1) << 2)));
  fftSpectrumDb();
  const spectrum = fftSpectrumPtr();
  const resolution = TAU / <f64>T;
  let last = <i32>Math.floor(upper / resolution);
  if (last > T / 2 - 1) last = T / 2 - 1;
  let strongest: f64 = -Infinity;
  for (let i = 0; i <= last; i++) strongest = Math.max(strongest, <f64>load<f32>(spectrum + (<usize>i << 2)));
  const floorDb = strongest + 20 * Math.log10(relativeFloor);
  let count = 0;
  for (let i = 0; i <= last; i++) {
    const c = <f64>load<f32>(spectrum + (<usize>i << 2));
    const l: f64 = i > 0 ? <f64>load<f32>(spectrum + (<usize>(i - 1) << 2)) : -Infinity;
    const r = <f64>load<f32>(spectrum + (<usize>(i + 1) << 2));
    if (!(c >= floorDb && c >= l && c > r)) continue;
    let offset: f64 = 0;
    if (i > 0) {
      const bend = l - 2 * c + r;
      if (bend < 0) offset = Math.max(-0.5, Math.min(0.5, (0.5 * (l - r)) / bend));
    }
    const omega = Math.max(0, (<f64>i + offset) * resolution);
    // Keep the strongest maxTones, by insertion into a list sorted by level.
    let slot = count;
    if (count == maxTones) {
      if (c <= f64At(seedLevel, count - 1)) continue;
      slot = count - 1;
    } else {
      count++;
    }
    while (slot > 0 && f64At(seedLevel, slot - 1) < c) {
      setF64(seeds, slot, f64At(seeds, slot - 1));
      setF64(seedLevel, slot, f64At(seedLevel, slot - 1));
      slot--;
    }
    setF64(seeds, slot, omega);
    setF64(seedLevel, slot, c);
  }
  // Distinct seeds (two maxima can round onto one frequency).
  for (let i = 0; i < count; i++) {
    for (let j = i + 1; j < count; j++) {
      if (f64At(seeds, j) == f64At(seeds, i)) setF64(seeds, j, f64At(seeds, j) + 1e-9 * resolution);
    }
  }
  return count;
}

function analyse(channel: i32, k: i64): void {
  const T = windowSize;
  const H = hop;
  const start = (k - 1) * <i64>H; // first sample of the window
  // The window's tonal contribution starts empty; its second half is new in the rings.
  for (let b = 0; b < bands; b++) {
    for (let i = H; i < T; i++) store<f32>(tonalSlot(channel, b, start + <i64>i), 0);
  }
  store<i32>(lastTones + (<usize>channel << 2), 0);
  // Tones fitted elsewhere (the batch path, ports/wasm-simd/signal-processor.mjs `equalizeBatch`): the host's slot.
  if (useToneSlot) {
    const supplied = <i32>f64At(toneSlot, 0);
    if (supplied > 0) synthesize(channel, start, supplied, true);
    return;
  }
  // The curve: T + 3 samples, one of context on each side, cropped to the window's T intervals.
  const curve = faftInputPtr();
  for (let i = 0; i < T + 3; i++) store<f32>(curve + (<usize>i << 2), <f32>xAt(channel, start - 1 + <i64>i));
  const count = windowSeeds();
  if (count <= 0) return;
  const resolution = TAU / <f64>T;
  // The joint fit, the significance prune, refinement, and DC snapped to 0 (bands.rs `ToneAnalysis`).
  let n = fitTones(seeds, count, reach, true, FAST);
  if (n <= 0) return;
  let loudest: f64 = 0;
  for (let p = 0; p < n; p++) loudest = Math.max(loudest, toneObserved(p));
  const floor = relativeFloor * loudest;
  let m = 0;
  for (let p = 0; p < n; p++) {
    if (toneSignificance(p) >= floor) setF64(kept, m++, toneOmega(p));
  }
  if (m == 0) return;
  n = refine > 0 ? refineTones(kept, m, reach, refine, 1e-4 * resolution, true, FAST) : m < n ? fitTones(kept, m, reach, true, FAST) : n;
  if (n <= 0) return;
  let snap = false;
  for (let p = 0; p < n; p++) {
    const w = toneOmega(p);
    if (w > 0 && w < 0.5 * resolution) snap = true;
  }
  if (snap) {
    let distinct = 0;
    for (let p = 0; p < n; p++) {
      const w = toneOmega(p) < 0.5 * resolution ? 0 : toneOmega(p);
      if (distinct > 0 && f64At(kept, distinct - 1) == w) continue;
      setF64(kept, distinct++, w);
    }
    n = fitTones(kept, distinct, reach, true, FAST);
    if (n <= 0) return;
  }
  synthesize(channel, start, n, false);
}

/**
 * Synthesis of window `start`'s n tones (the last fit's, or the host's slot): w(u)·(α cos ωu + β sin ωu)
 * for u = i − H, by a rotating phasor, into each band it weighs in, added to the tonal rings.
 */
function synthesize(channel: i32, start: i64, n: i32, fromSlot: bool): void {
  const T = windowSize;
  const H = hop;
  for (let b = 0; b < bands; b++) memory.fill(windowTonal + (<usize>(b * MAX_WINDOW) << 3), 0, <usize>T << 3);
  for (let p = 0; p < n; p++) {
    const omega = fromSlot ? f64At(toneSlot, 1 + 3 * p) : toneOmega(p);
    const alpha = fromSlot ? f64At(toneSlot, 2 + 3 * p) : toneAlpha(p);
    const beta = fromSlot ? f64At(toneSlot, 3 + 3 * p) : toneBeta(p);
    if (!isFinite(alpha) || !isFinite(beta)) continue;
    let re = Math.cos(omega * <f64>-H);
    let im = Math.sin(omega * <f64>-H);
    const sr = Math.cos(omega);
    const si = Math.sin(omega);
    for (let b = 0; b < bands; b++) {
      const g = weight(b, omega);
      if (g == 0) continue;
      const out = windowTonal + (<usize>(b * MAX_WINDOW) << 3);
      let zre = re;
      let zim = im;
      const ga = g * alpha;
      const gb = g * beta;
      for (let i = 0; i < T; i++) {
        setF64(out, i, f64At(out, i) + f64At(hann, i) * (ga * zre + gb * zim));
        const next = zre * sr - zim * si;
        zim = zre * si + zim * sr;
        zre = next;
      }
    }
  }
  for (let b = 0; b < bands; b++) {
    const src = windowTonal + (<usize>(b * MAX_WINDOW) << 3);
    for (let i = 0; i < T; i++) {
      const slot = tonalSlot(channel, b, start + <i64>i);
      store<f32>(slot, <f32>(<f64>load<f32>(slot) + f64At(src, i)));
    }
  }
  store<i32>(lastTones + (<usize>channel << 2), n);
  store<i64>(tonesRun + (<usize>channel << 3), load<i64>(tonesRun + (<usize>channel << 3)) + <i64>n);
}

/** Forward residual filters over pieces [from, to] (ξ(n + 1) from ξ(n)), keeping every CHECK-th state. */
function forward(channel: i32, from: i64, to: i64): void {
  for (let e = 0; e < bands - 1; e++) {
    for (let n = from; n <= to; n++) {
      if (n < 0) continue;
      const c0 = rAt(channel, n);
      const c1 = rAt(channel, n + 1) - c0;
      for (let j = 0; j < order; j++) {
        const s = stateSlot(channel, e, j);
        const xr = load<f64>(s);
        const xi = load<f64>(s, 8);
        const er = f64At(poleSlot(e, j, 2), 0), ei = f64At(poleSlot(e, j, 2), 1);
        const nr = er * xr - ei * xi + c0 * f64At(poleSlot(e, j, 3), 0) + c1 * f64At(poleSlot(e, j, 4), 0);
        const ni = er * xi + ei * xr + c0 * f64At(poleSlot(e, j, 3), 1) + c1 * f64At(poleSlot(e, j, 4), 1);
        store<f64>(s, nr);
        store<f64>(s, ni, 8);
        if ((n + 1) % CHECK == 0) {
          const c = checkSlot(channel, e, n + 1, j);
          store<f64>(c, nr);
          store<f64>(c, ni, 8);
        }
      }
    }
  }
}

/** ξ(t) for t in [s, s + count) of one edge into `segment`, rebuilt from the checkpoint at s (a multiple of CHECK). */
function rebuild(channel: i32, edge: i32, s: i64, count: i32): void {
  for (let j = 0; j < order; j++) {
    const c = checkSlot(channel, edge, s, j);
    setF64(segment, 2 * j, load<f64>(c));
    setF64(segment, 2 * j + 1, load<f64>(c, 8));
  }
  for (let i = 1; i < count; i++) {
    const n = s + <i64>(i - 1);
    const c0 = rAt(channel, n);
    const c1 = rAt(channel, n + 1) - c0;
    for (let j = 0; j < order; j++) {
      const xr = f64At(segment, 2 * ((i - 1) * MAX_ORDER + j));
      const xi = f64At(segment, 2 * ((i - 1) * MAX_ORDER + j) + 1);
      const er = f64At(poleSlot(edge, j, 2), 0), ei = f64At(poleSlot(edge, j, 2), 1);
      setF64(segment, 2 * (i * MAX_ORDER + j), er * xr - ei * xi + c0 * f64At(poleSlot(edge, j, 3), 0) + c1 * f64At(poleSlot(edge, j, 4), 0));
      setF64(segment, 2 * (i * MAX_ORDER + j) + 1, er * xi + ei * xr + c0 * f64At(poleSlot(edge, j, 3), 1) + c1 * f64At(poleSlot(edge, j, 4), 1));
    }
  }
}

/** Low-passed residual of every edge for the block [b0, b0 + H) into `lows`: causal, or zero-phase from `end`. */
function lowPass(channel: i32, b0: i64, end: i64): void {
  const H = hop;
  for (let e = 0; e < bands - 1; e++) {
    const row = lows + (<usize>(e * MAX_HOP) << 3);
    if (lookahead == 0) {
      // Causal: y(t) = Re Σ ρ_j ξ_j(t).
      let t = b0;
      while (t < b0 + <i64>H) {
        if (t < 0) {
          setF64(row, <i32>(t - b0), 0);
          t++;
          continue;
        }
        const s = (t / CHECK) * CHECK;
        let stop = s + <i64>CHECK;
        if (stop > b0 + <i64>H) stop = b0 + <i64>H;
        rebuild(channel, e, s, <i32>(stop - s));
        for (; t < stop; t++) {
          const i = <i32>(t - s);
          let v: f64 = 0;
          for (let j = 0; j < order; j++) {
            v += f64At(poleSlot(e, j, 1), 0) * f64At(segment, 2 * (i * MAX_ORDER + j)) - f64At(poleSlot(e, j, 1), 1) * f64At(segment, 2 * (i * MAX_ORDER + j) + 1);
          }
          setF64(row, <i32>(t - b0), v);
        }
      }
      continue;
    }
    // Zero-phase: η_k(n) = e^{p_k}η_k(n + 1) + α_k c₀ + β_k c₁ + Σ_j γ_kj ξ_j(n), from η = 0 at end + 1.
    memory.fill(eta, 0, 128);
    let n = end;
    const first = b0 > 0 ? b0 : 0;
    while (n >= first) {
      const s = (n / CHECK) * CHECK;
      rebuild(channel, e, s, <i32>(n - s) + 1);
      for (; n >= s && n >= first; n--) {
        const i = <i32>(n - s);
        const c0 = rAt(channel, n);
        const c1 = rAt(channel, n + 1) - c0;
        let v: f64 = 0;
        for (let k = 0; k < order; k++) {
          const hr = f64At(eta, 2 * k), hi = f64At(eta, 2 * k + 1);
          const er = f64At(poleSlot(e, k, 2), 0), ei = f64At(poleSlot(e, k, 2), 1);
          let nr = er * hr - ei * hi + c0 * f64At(poleSlot(e, k, 5), 0) + c1 * f64At(poleSlot(e, k, 6), 0);
          let ni = er * hi + ei * hr + c0 * f64At(poleSlot(e, k, 5), 1) + c1 * f64At(poleSlot(e, k, 6), 1);
          for (let j = 0; j < order; j++) {
            const gr = f64At(gammaSlot(e, k, j), 0), gi = f64At(gammaSlot(e, k, j), 1);
            const xr = f64At(segment, 2 * (i * MAX_ORDER + j)), xi = f64At(segment, 2 * (i * MAX_ORDER + j) + 1);
            nr += gr * xr - gi * xi;
            ni += gr * xi + gi * xr;
          }
          setF64(eta, 2 * k, nr);
          setF64(eta, 2 * k + 1, ni);
          v += f64At(poleSlot(e, k, 1), 0) * nr - f64At(poleSlot(e, k, 1), 1) * ni;
        }
        if (n < b0 + <i64>H) setF64(row, <i32>(n - b0), v);
      }
    }
    for (let t = b0; t < first; t++) setF64(row, <i32>(t - b0), 0);
  }
}

/** Hop k: fit window k, complete the residual of [(k − 1)H, kH), filter it, and emit the output block. */
function runHop(channel: i32, k: i64): void {
  const H = <i64>hop;
  analyse(channel, k);
  store<i64>(windowsRun + (<usize>channel << 3), load<i64>(windowsRun + (<usize>channel << 3)) + 1);
  // The residual of the samples both their windows have now covered.
  for (let t = (k - 1) * H; t < k * H; t++) {
    if (t < 0) continue;
    let tonal: f64 = 0;
    for (let b = 0; b < bands; b++) tonal += <f64>load<f32>(tonalSlot(channel, b, t));
    store<f64>(rSlot(channel, t), xAt(channel, t) - tonal);
  }
  forward(channel, (k - 1) * H - 1, k * H - 2);
  const b0 = (k - 1 - <i64>lookahead) * H - 1;
  lowPass(channel, b0, k * H - 2);
  // Bands and the equalized output, gains ramped across the block.
  for (let i = 0; i < hop; i++) {
    const t = b0 + <i64>i;
    if (t < 0) continue;
    const ramp = <f64>(i + 1) / <f64>hop;
    let y: f64 = 0;
    let previous: f64 = 0;
    const r = rAt(channel, t);
    for (let b = 0; b < bands; b++) {
      const low = b < bands - 1 ? f64At(lows, b * MAX_HOP + i) : r;
      const band = <f64>load<f32>(tonalSlot(channel, b, t)) + low - previous;
      previous = low;
      const g0 = f64At(current, channel * MAX_BANDS + b);
      y += (g0 + (f64At(target, b) - g0) * ramp) * band;
      store<f32>(bandOut + (<usize>((channel * MAX_BANDS + b) * OUT_RING + (<i32>t & OUT_MASK)) << 2), <f32>band);
    }
    store<f32>(outRing + (<usize>(channel * OUT_RING + (<i32>t & OUT_MASK)) << 2), <f32>y);
  }
  for (let b = 0; b < bands; b++) setF64(current, channel * MAX_BANDS + b, f64At(target, b));
}

/**
 * Processes `count` samples (at most MAX_BLOCK) of `channel` from eqInputPtr: writes as many to
 * eqOutputPtr, the equalized signal `latency` samples late (zeros before), and with `writeBands` each
 * band's track to eqBandsPtr[b · MAX_BLOCK ..] (they sum to the delayed input). Returns count, or −1.
 */
export function eqProcess(channel: i32, count: i32, writeBands: bool): i32 {
  if (!configured || channel < 0 || channel >= channels || count < 0 || count > MAX_BLOCK) return -1;
  const H = <i64>hop;
  for (let i = 0; i < count; i++) {
    const v = <f64>load<f32>(input + (<usize>i << 2));
    const at = load<i64>(received + (<usize>channel << 3));
    store<f64>(xRing + (<usize>(channel * X_RING + (<i32>at & X_MASK)) << 3), isFinite(v) ? v : 0);
    const now = at + 1;
    store<i64>(received + (<usize>channel << 3), now);
    // Hop k needs every sample up to (k + 1)H + 1.
    const k = load<i64>(hops + (<usize>channel << 3));
    if (now == (k + 1) * H + 2) {
      runHop(channel, k);
      store<i64>(hops + (<usize>channel << 3), k + 1);
    }
    const t = now - 1 - <i64>latency;
    let y: f32 = 0;
    if (t >= 0) y = load<f32>(outRing + (<usize>(channel * OUT_RING + (<i32>t & OUT_MASK)) << 2));
    store<f32>(output + (<usize>i << 2), y);
    if (writeBands) {
      for (let b = 0; b < bands; b++) {
        const v: f32 = t >= 0 ? load<f32>(bandOut + (<usize>((channel * MAX_BANDS + b) * OUT_RING + (<i32>t & OUT_MASK)) << 2)) : 0;
        store<f32>(bandsOutput + (<usize>(b * MAX_BLOCK + i) << 2), v);
      }
    }
  }
  return count;
}
