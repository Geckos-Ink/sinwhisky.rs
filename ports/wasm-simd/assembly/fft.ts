// Transposed from appx/dsp at ec84edcd331af8810be9df42cf5d16f3a348d6d5 (2026-10-05).
// Radix-2 FFT producing a Hann-windowed magnitude spectrum in dBFS.
// All buffers are static data segments: no allocation and no memory growth, so JS views over
// `memory.buffer` stay valid for the lifetime of the instance.

import { linearToDb } from './db';

const MIN_LOG2: i32 = 4;
const MAX_LOG2: i32 = 13; // 8192 points

// Byte sizes are literals because memory.data needs compile-time constants (8192 × 4 bytes).
const input = memory.data(32768, 16); // f32[n]    time-domain samples written by the caller
const window = memory.data(32768, 16); // f32[n]   periodic Hann window
const bitReversed = memory.data(32768, 16); // i32[n]
const re = memory.data(32768, 16); // f32[n]       working buffers
const im = memory.data(32768, 16); // f32[n]
const cosTable = memory.data(16384, 16); // f32[n/2]
const sinTable = memory.data(16384, 16); // f32[n/2]
const spectrum = memory.data(16388, 16); // f32[n/2 + 1] dBFS per bin
const kernelRe = memory.data(32768, 16);
const kernelIm = memory.data(32768, 16);
const convolution = memory.data(32768, 16);
let kernelReady: bool = false;

let size: i32 = 0;

@inline function f32At(base: usize, index: i32): f32 {
  return load<f32>(base + (<usize>index << 2));
}

@inline function setF32(base: usize, index: i32, value: f32): void {
  store<f32>(base + (<usize>index << 2), value);
}

export function fftMinLog2(): i32 {
  return MIN_LOG2;
}

export function fftMaxLog2(): i32 {
  return MAX_LOG2;
}

/** Current transform size in points, or 0 before a successful fftInit. */
export function fftSize(): i32 {
  return size;
}

export function fftInputPtr(): usize {
  return input;
}

export function fftSpectrumPtr(): usize {
  return spectrum;
}

/** Prepares window, twiddle and bit-reversal tables for 2^log2 points. Returns false if out of range. */
export function fftInit(log2: i32): bool {
  if (log2 < MIN_LOG2 || log2 > MAX_LOG2) {
    return false;
  }
  const n = 1 << log2;
  for (let i = 0; i < n; i++) {
    setF32(window, i, <f32>(0.5 - 0.5 * Math.cos((2.0 * Math.PI * <f64>i) / <f64>n)));
    let reversed = 0;
    let bits = i;
    for (let b = 0; b < log2; b++) {
      reversed = (reversed << 1) | (bits & 1);
      bits >>= 1;
    }
    store<i32>(bitReversed + (<usize>i << 2), reversed);
  }
  for (let k = 0; k < n >> 1; k++) {
    const angle = (2.0 * Math.PI * <f64>k) / <f64>n;
    setF32(cosTable, k, <f32>Math.cos(angle));
    setF32(sinTable, k, <f32>Math.sin(angle));
  }
  size = n;
  kernelReady = false;
  return true;
}

/**
 * Windows the input, transforms it, and writes n/2 + 1 bins of dBFS to the spectrum buffer.
 * A full-scale sine centred on a bin reads 0 dBFS (Hann coherent gain is compensated).
 */
export function fftSpectrumDb(): void {
  const n = size;
  if (n == 0) {
    return;
  }

  for (let i = 0; i < n; i++) {
    const target = load<i32>(bitReversed + (<usize>i << 2));
    setF32(re, target, f32At(input, i) * f32At(window, i));
    setF32(im, target, 0);
  }

  transform(false);

  // Amplitude of a bin-centred sine: |X[k]| · 4 / n (2/n for one-sided, ×2 for Hann gain 0.5).
  const scale: f64 = 4.0 / <f64>n;
  const nyquist = n >> 1;
  for (let k = 0; k <= nyquist; k++) {
    const real = <f64>f32At(re, k);
    const imag = <f64>f32At(im, k);
    let amplitude = Math.sqrt(real * real + imag * imag) * scale;
    if (k == 0 || k == nyquist) amplitude *= 0.5;
    setF32(spectrum, k, <f32>linearToDb(amplitude));
  }
}

function transform(inverse: bool): void {
  const n = size;
  for (let length = 2; length <= n; length <<= 1) {
    const half = length >> 1;
    const stride = n / length;
    for (let start = 0; start < n; start += length) {
      for (let j = 0; j < half; j++) {
        const c = f32At(cosTable, j * stride);
        const s = f32At(sinTable, j * stride) * (inverse ? -1 : 1);
        const a = start + j;
        const b = a + half;
        const br = f32At(re, b);
        const bi = f32At(im, b);
        // (br + i·bi) · e^(−iθ) = (br·c + bi·s) + i·(bi·c − br·s)
        const tr = br * c + bi * s;
        const ti = bi * c - br * s;
        const ar = f32At(re, a);
        const ai = f32At(im, a);
        setF32(re, b, ar - tr);
        setF32(im, b, ai - ti);
        setF32(re, a, ar + tr);
        setF32(im, a, ai + ti);
      }
    }
  }

}

export function fftConvolutionPtr(): usize { return convolution; }

/** Saves the unwindowed input as the zero-padded convolution kernel. */
export function fftSetKernel(length: i32): bool {
  if (size == 0 || length < 1 || length > size) return false;
  for (let i = 0; i < size; i++) {
    const target = load<i32>(bitReversed + (<usize>i << 2));
    setF32(re, target, i < length ? f32At(input, i) : 0);
    setF32(im, target, 0);
  }
  transform(false);
  memory.copy(kernelRe, re, <usize>size << 2);
  memory.copy(kernelIm, im, <usize>size << 2);
  kernelReady = true;
  return true;
}

/** Circular convolution; callers zero-pad and overlap-add for linear convolution. */
export function fftConvolve(): bool {
  if (!kernelReady) return false;
  for (let i = 0; i < size; i++) {
    const target = load<i32>(bitReversed + (<usize>i << 2));
    setF32(re, target, f32At(input, i));
    setF32(im, target, 0);
  }
  transform(false);
  for (let i = 0; i < size; i++) {
    const ar = f32At(re, i), ai = f32At(im, i);
    const br = f32At(kernelRe, i), bi = f32At(kernelIm, i);
    setF32(re, i, ar * br - ai * bi);
    setF32(im, i, ar * bi + ai * br);
  }
  for (let i = 0; i < size; i++) {
    const j = load<i32>(bitReversed + (<usize>i << 2));
    if (j <= i) continue;
    const ar = f32At(re, i), ai = f32At(im, i);
    setF32(re, i, f32At(re, j)); setF32(im, i, f32At(im, j));
    setF32(re, j, ar); setF32(im, j, ai);
  }
  transform(true);
  for (let i = 0; i < size; i++) setF32(convolution, i, f32At(re, i) / <f32>size);
  return true;
}
