// Zero-copy JavaScript wrapper over build/sinwhisky.wasm (assembly/index.ts): SinWhisky's original-angular
// resampler, the AFT, and FAFT on the AFT's curve, each with an optional WebAssembly SIMD path.
// Works in Node (tested with 24) and browsers. Results are copied out of the module's static buffers.

const TAU = 2 * Math.PI;

/** `PeakParams::default()` of sinwhisky.rs (scan seeds only; circle seeds are not ported). */
export const DEFAULT_PEAKS = Object.freeze({
  lower: 0,
  upper: Math.PI,
  density: 2,
  hann: true,
  relativeFloor: 0.01,
  absoluteFloor: 0,
  mergeWidth: null,
  maxPeaks: 4096,
  maxIterations: 32,
  tolerance: 1e-12,
});

export class SinWhisky {
  #x;
  #memory;

  constructor(instance) {
    this.#x = instance.exports;
    this.#memory = this.#x.memory;
  }

  /** Instantiates the module from its bytes (a BufferSource) or an already compiled WebAssembly.Module. */
  static async instantiate(source) {
    const module = source instanceof WebAssembly.Module ? source : await WebAssembly.compile(source);
    return new SinWhisky(await WebAssembly.instantiate(module, {}));
  }

  /** Node only: the module next to this file. */
  static async load() {
    const { readFile } = await import('node:fs/promises');
    return SinWhisky.instantiate(await readFile(new URL('./build/sinwhisky.wasm', import.meta.url)));
  }

  get maxInput() {
    return this.#x.maxInput();
  }

  #f64(pointer, length) {
    return new Float64Array(this.#memory.buffer, pointer, length);
  }

  #input(samples) {
    if (samples.length > this.maxInput) throw new RangeError(`at most ${this.maxInput} samples`);
    this.#f64(this.#x.inputPtr(), samples.length).set(samples);
  }

  /**
   * `resample(samples, ResampleParams::original(zoom))`: originals copied, `zoom` samples inserted per
   * interval along equal angles. `simd` advances both covering circles in one f64x2 by rotation.
   */
  resample(samples, zoom, { simd = false } = {}) {
    this.#input(samples);
    if (this.#x.swPrepare(samples.length) < 0) throw new RangeError('non-finite sample');
    const count = this.#x.swResample(samples.length, zoom, simd);
    if (count < 0) throw new RangeError(`zoom 0..${this.#x.maxZoom()} and at most ${this.#x.maxOutput()} outputs`);
    return this.#f64(this.#x.outputPtr(), count).slice();
  }

  /**
   * The resampler's fitted models for `samples` (model j belongs to sample j + 1), `modelStride` f64 each:
   * circle flag, center y, radius, the three unwrapped angles, line value, line slope. Feeds the GPU port.
   */
  models(samples) {
    this.#input(samples);
    const count = this.#x.swPrepare(samples.length);
    if (count < 0) throw new RangeError('non-finite sample');
    return this.#f64(this.#x.modelsPtr(), count * this.#x.modelStride()).slice();
  }

  get modelStride() {
    return this.#x.modelStride();
  }

  /** `AngularFrame::new(samples, AnalysisParams::original())`, prepared for evaluation and FAFT. */
  prepareAft(samples) {
    this.#input(samples);
    const intervals = this.#x.aftPrepare(samples.length);
    if (intervals < 0) throw new RangeError(`1 to ${this.maxInput} finite samples`);
    return {
      intervals,
      circleHalves: this.#x.aftCircleHalves(),
      lineHalves: this.#x.aftLineHalves(),
      rejectedCircleHalves: this.#x.aftRejectedCircleHalves(),
    };
  }

  /** The prepared curve at a real position, periodic (`AngularFrame::evaluate`). */
  evaluate(position) {
    return this.#x.aftEvaluate(position);
  }

  /** The prepared FAFT atoms, `atomStride` f64 per interval: value, slope, two arcs × (a, b, rate, cos, sin). */
  atoms() {
    return this.#f64(this.#x.aftAtomsPtr(), this.#x.aftIntervals() * this.#x.aftAtomStride()).slice();
  }

  get atomStride() {
    return this.#x.aftAtomStride();
  }

  #values(omegas, hann, normalized, simd) {
    const max = this.#x.aftMaxQueries();
    const out = new Float64Array(2 * omegas.length);
    for (let start = 0; start < omegas.length; start += max) {
      const chunk = Math.min(max, omegas.length - start);
      this.#f64(this.#x.aftOmegasPtr(), chunk).set(Array.prototype.slice.call(omegas, start, start + chunk));
      if (!this.#x.aftValues(chunk, hann, normalized, simd)) throw new RangeError('prepare the AFT first; finite frequencies only');
      out.set(this.#f64(this.#x.aftValuesPtr(), 2 * chunk), 2 * start);
    }
    return out;
  }

  /** `AngularFrame::fourier` at each ω: the integral on [0, N], as (re, im) pairs. */
  integral(omegas, { simd = false } = {}) {
    return this.#values(omegas, false, false, simd);
  }

  /** `Faft::values` on the prepared curve, rectangular or Hann (normalized; a unit sinusoid reads ~0.5). */
  values(omegas, { hann = false, simd = false } = {}) {
    return this.#values(omegas, hann, true, simd);
  }

  /** Normalized magnitudes on lower + k·step; Hann with `density` steps per 2π/N (step = 2π/N/density). */
  scan(lower, step, count, { density = 0, simd = true } = {}) {
    if (!this.#x.aftScan(lower, step, count, density, simd)) throw new RangeError('prepare the AFT first; scan beyond the buffer');
    return this.#f64(this.#x.aftValuesPtr(), count).slice();
  }

  /** `Faft::derivatives`: normalized centred F, F′, F″ at ω as [re, im] pairs. */
  derivatives(omega, { hann = true } = {}) {
    if (!this.#x.aftDerivatives(omega, hann)) throw new RangeError('prepare the AFT first');
    const v = this.#f64(this.#x.aftValuesPtr(), 6);
    return [[v[0], v[1]], [v[2], v[3]], [v[4], v[5]]];
  }

  /** `Faft::peaks` with scan seeds: [{ omega, re, im, magnitude, curvature }], by frequency. */
  peaks(params = {}) {
    const p = { ...DEFAULT_PEAKS, ...params };
    const count = this.#x.aftPeaks(p.lower, p.upper, p.density, p.hann, p.relativeFloor, p.absoluteFloor,
      p.mergeWidth ?? -1, p.maxPeaks, p.maxIterations, p.tolerance);
    if (count < 0) throw new RangeError('invalid peak parameters, no prepared AFT, or too many seeds');
    const v = this.#f64(this.#x.aftPeaksPtr(), 4 * count);
    return Array.from({ length: count }, (_, k) => ({
      omega: v[4 * k],
      re: v[4 * k + 1],
      im: v[4 * k + 2],
      magnitude: Math.hypot(v[4 * k + 1], v[4 * k + 2]),
      curvature: v[4 * k + 3],
    }));
  }

  /** 2π/N of the prepared frame. */
  get resolution() {
    return TAU / this.#x.aftIntervals();
  }
}
