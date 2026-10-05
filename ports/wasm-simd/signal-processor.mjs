// Transposed from appx/dsp at ec84edcd331af8810be9df42cf5d16f3a348d6d5 (2026-10-05).
export const wasmImports = { env: { abort() { throw new Error('sinwhisky: WebAssembly abort'); } } };
const clockMs = () => globalThis.performance?.now() ?? 0;
export const SWL2_STEREO = ['independent', 'left/side', 'side/right', 'mid/side'];
export class SignalProcessor {
  #exports;
  #faftInput;
  #faftOmegas;
  #faftOutput;
  #faftPeaks;
  #faftAtoms;
  #faftGain;


  constructor(instance) {
    this.#exports = instance.exports;
    const buffer = this.#exports.memory.buffer;
    const queries = this.#exports.faftMaxQueries();
    this.#faftInput = new Float32Array(buffer, this.#exports.faftInputPtr(), this.#exports.faftMaxInput());
    this.#faftOmegas = new Float64Array(buffer, this.#exports.faftOmegasPtr(), queries);
    this.#faftOutput = new Float64Array(buffer, this.#exports.faftOutputPtr(), 2 * queries);
    this.#faftPeaks = new Float64Array(buffer, this.#exports.faftPeaksPtr(), 4 * this.#exports.faftMaxPeaks());
    this.#faftAtoms = new Float64Array(buffer, this.#exports.faftAtomsPtr(), this.#exports.faftMaxInput() * this.#exports.faftAtomStride());
  }

  static async instantiate(source) {
    const module = source instanceof WebAssembly.Module ? source : await WebAssembly.compile(source);
    return new this(new WebAssembly.Instance(module, wasmImports));
  }
  static fromModule(module) { return new this(new WebAssembly.Instance(module, wasmImports)); }
  get maxFaftInput() {
    return this.#exports.faftMaxInput();
  }

  get maxFaftQueries() {
    return this.#exports.faftMaxQueries();
  }

  /**
   * Fits SinWhisky's circles through `samples` (2 to maxFaftInput) and prepares their curve's rotations
   * over the window [0, samples.length − 1). Returns the window's duration in samples; throws RangeError
   * for a size beyond the buffer or a non-finite sample. Does not allocate.
   */
  prepareFaft(samples) {
    if (samples.length < 2 || samples.length > this.#faftInput.length) throw new RangeError(`FAFT takes 2 to ${this.#faftInput.length} samples`);
    this.#faftInput.set(samples);
    const intervals = this.#exports.faftPrepare(samples.length);
    if (intervals < 0) throw new RangeError('FAFT input must be finite');
    return intervals;
  }

  /** The prepared window's duration in samples (0 before prepareFaft). */
  get faftDuration() {
    return this.#exports.faftIntervals();
  }

  /** Prepared rotations: y0, slope, then two arcs × (a, b, rate, cos, sin, start slope, end slope, end change, rate²a). Stride 20, for FaftGpu. */
  get faftAtoms() {
    return this.#faftAtoms.subarray(0, this.#exports.faftIntervals() * this.#exports.faftAtomStride());
  }

  get faftAtomStride() {
    return this.#exports.faftAtomStride();
  }

  /** The curve at a real position in [0, duration] (what SinWhisky's zoom samples); NaN outside. */
  faftEvaluate(position) {
    return this.#exports.faftEvaluate(position);
  }

  /**
   * The normalized transform at each of `omegas` (radians per sample, any real values, at most
   * maxFaftQueries): a view of (re, im) pairs, valid until the next FAFT call. Rectangular or Hann
   * (applied analytically); `simd` evaluates two frequencies per instruction, and `fast` four, in f32
   * lanes (about 1e-6 of the largest value; at most 682 Hann or 2048 rectangular frequencies: the equalizer's path). Does not
   * allocate beyond the returned subarray.
   */
  faftValues(omegas, options = {}) {
    if (omegas.length > this.#faftOmegas.length) throw new RangeError(`at most ${this.#faftOmegas.length} frequencies`);
    this.#faftOmegas.set(omegas);
    const ok = options.fast
      ? this.#exports.faftValuesFast(omegas.length, options.hann ? 1 : 0)
      : this.#exports.faftValues(omegas.length, options.hann ? 1 : 0, options.simd ? 1 : 0);
    if (!ok) throw new RangeError('FAFT not prepared, too many frequencies, or a frequency is not finite');
    return this.#faftOutput.subarray(0, 2 * omegas.length);
  }

  /**
   * Magnitudes on the grid lower + k·step, k < count: a view valid until the next FAFT call. With
   * `density` > 0 the Hann window applies and `step` must be (2π/T)/density (its neighbours are density
   * steps away); 0 is rectangular. `simd`: two f64 lanes (true), scalar (false), or four f32 lanes
   * ('fast', about 1e-6 of the largest value).
   */
  faftScan(lower, step, count, density = 0, simd = true) {
    if (!this.#exports.faftScan(lower, step, count, density, simd === 'fast' ? 2 : simd ? 1 : 0)) throw new RangeError('FAFT not prepared, or the scan exceeds the buffer');
    return this.#faftOutput.subarray(0, count);
  }

  /**
   * Continuous relevant peaks (sinwhisky.rs `Faft::peaks`): a duration-spaced scan (SIMD), each local
   * maximum refined by Newton with exact derivatives, flanks and unconverged seeds dropped, close peaks
   * merged, sorted by frequency. Allocates its result.
   */
  faftPeaks(options = {}) {
    const count = this.#exports.faftPeaks(
      options.lower ?? 0, options.upper ?? Math.PI, options.density ?? 2, options.relativeFloor ?? 0.01,
      options.maxPeaks ?? this.#exports.faftMaxPeaks(), options.hann === false ? 0 : 1, options.tolerance ?? 1e-12,
);
    if (count < 0) throw new RangeError('FAFT not prepared, or invalid peak options');
    const peaks = [];
    for (let k = 0; k < count; k++) {
      const [omega, re, im, curvature] = this.#faftPeaks.subarray(4 * k, 4 * k + 4);
      peaks.push({ omega, re, im, magnitude: Math.hypot(re, im), amplitude: 0, curvature });
    }
    for (const peak of peaks) peak.amplitude = (2 * peak.magnitude) / this.faftToneGain(peak.omega);
    return peaks;
  }

  /**
   * The curve's response to a tone at `omega` (radians per sample): the amplitude of its fundamental
   * over the sampled tone's. SinWhisky's circles follow a tone closely at low frequencies and lose some of
   * it near Nyquist; the loss depends on the frequency alone (not the phase or the amplitude), so it is
   * measured once, on unit tones at 257 frequencies from 0 to π, and interpolated. The first call
   * measures it (about 20 ms) and then prepares the current curve again.
   */
  faftToneGain(omega) {
    if (!this.#faftGain) {
      const length = this.#exports.faftIntervals() + 1;
      const saved = length > 1 ? Float32Array.from(this.#faftInput.subarray(0, length)) : undefined;
      const table = new Float64Array(257);
      const tone = new Float32Array(2048);
      for (let k = 0; k <= 256; k++) {
        const w = (Math.PI * Math.min(Math.max(k, 2), 254)) / 256;
        for (let n = 0; n < tone.length; n++) tone[n] = Math.sin(w * n + 0.3);
        this.prepareFaft(tone);
        const [re, im] = this.faftValues([w], { hann: true });
        table[k] = 2 * Math.hypot(re, im);
      }
      this.#faftGain = table;
      if (saved) this.prepareFaft(saved);
    }
    const x = (Math.min(Math.PI, Math.abs(omega)) / Math.PI) * 256;
    const k = Math.min(255, Math.floor(x));
    return this.#faftGain[k] + (this.#faftGain[k + 1] - this.#faftGain[k]) * (x - k);
  }

  /**
   * Centred normalized F, F′, F″ at each of `omegas` (at most maxFaftTones; sinwhisky.rs
   * `Faft::derivatives`): a view of 6 f64 per frequency (re, im of each), valid until the next FAFT call.
   */
  faftDerivatives(omegas, options = {}) {
    if (omegas.length > this.#exports.faftMaxTones()) throw new RangeError(`at most ${this.#exports.faftMaxTones()} frequencies`);
    this.#faftOmegas.set(omegas);
    if (!this.#exports.faftDerivatives(omegas.length, options.hann === false ? 0 : 1, options.simd === false ? 0 : 1)) throw new RangeError('FAFT not prepared, or a frequency is not finite');
    return new Float64Array(this.#exports.memory.buffer, this.#exports.faftDerivativesPtr(), 6 * omegas.length);
  }

  /**
   * The normalized window kernel K(ν)/W of the prepared window (1 at ν = 0) or its first or second
   * ν-derivative: the shape every tone takes on the continuous axis.
   */
  faftKernel(nu, options = {}) {
    const value = this.#exports.faftKernel(nu, options.hann === false ? 0 : 1, options.order ?? 0);
    if (Number.isNaN(value)) throw new RangeError('FAFT not prepared, or an invalid kernel query');
    return value;
  }

  /**
   * Joint least-squares fit of real tones at `omegas` (finite, nonnegative, distinct; at most
   * maxFaftTones) on the prepared window: the two-way banded solve (assembly/tones.ts), optionally
   * refined by Newton steps on each tone's partial residual. Allocates its result; not for the audio thread.
   */
  faftFit(omegas, options = {}) {
    if (omegas.length > this.#exports.faftMaxTones()) throw new RangeError(`at most ${this.#exports.faftMaxTones()} tones`);
    const duration = this.#exports.faftIntervals();
    const hann = options.hann === false ? 0 : 1;
    const simd = options.fast ? 2 : options.simd === false ? 0 : 1;
    const reach = (options.reachBins ?? 16) * ((2 * Math.PI) / duration);
    this.#faftOmegas.set(omegas);
    const refine = options.refine ?? 0;
    const count = refine > 0
      ? this.#exports.faftFitRefined(omegas.length, reach, refine, options.tolerance ?? 1e-4 * ((2 * Math.PI) / duration), hann, simd)
      : this.#exports.faftFit(omegas.length, reach, hann, simd);
    if (count < 0) throw new RangeError('FAFT not prepared, or invalid fit frequencies or options');
    const stride = this.#exports.faftFitStride();
    const raw = new Float64Array(this.#exports.memory.buffer, this.#exports.faftFitPtr(), count * stride);
    const components = [];
    for (let k = 0; k < count; k++) {
      const [omega, re, im, observedRe, observedIm, significance] = raw.subarray(k * stride, k * stride + 6);
      // Centred: c·e^{iωT/2}; the tone is 2|c|·cos(ωu + arg c).
      const turn = omega * 0.5 * duration;
      const cre = re * Math.cos(turn) - im * Math.sin(turn);
      const cim = re * Math.sin(turn) + im * Math.cos(turn);
      components.push({ omega, re, im, observedRe, observedIm, significance, amplitude: 2 * Math.hypot(cre, cim), phase: Math.atan2(cim, cre) });
    }
    const fit = { components, degenerate: this.#exports.faftFitDegenerate(), truncated: this.#exports.faftFitTruncated() !== 0 };
    if (options.energy) {
      fit.energy = this.#exports.faftEnergy(hann);
      fit.residualEnergy = this.#exports.faftFitResidual(fit.energy);
      fit.bound = Math.sqrt(fit.residualEnergy / (hann ? duration / 2 : duration));
    }
    return fit;
  }

  get maxFaftTones() {
    return this.#exports.faftMaxTones();
  }

  // ---- the component equalizer (assembly/eq.ts) ----------------------------------------------------

  /** Configures the equalizer, clears its channels and returns its fixed views. Gains start at 1. */
  configureEqualizer(options) {
    const e = this.#exports;
    const edges = new Float64Array(e.memory.buffer, e.eqEdgesPtr(), e.eqMaxBands() - 1);
    if (options.edges.length > edges.length) throw new RangeError(`at most ${edges.length} edges`);
    edges.set(options.edges);
    const latency = e.eqInit(
      options.channels ?? 1, options.window ?? 2048, options.upper ?? (2 * Math.PI * 6000) / 44_100,
      options.relativeFloor ?? 0.01, options.maxTones ?? 64, options.reachBins ?? 16, options.refine ?? 1,
      options.edges.length, options.crossoverHalfWidth ?? 0, options.order ?? 4, options.lookahead ?? 2,
);
    if (latency < 0) throw new RangeError('invalid equalizer options');
    const block = e.eqMaxBlock();
    const bands = e.eqBands();
    return {
      latency,
      bands,
      input: new Float32Array(e.memory.buffer, e.eqInputPtr(), block),
      output: new Float32Array(e.memory.buffer, e.eqOutputPtr(), block),
      bandTracks: Array.from({ length: bands }, (_, b) => new Float32Array(e.memory.buffer, e.eqBandsPtr() + 4 * b * block, block)),
      truncationDb: e.eqTruncationDb(),
    };
  }

  /** Band gains (linear), ramped over the next output block. Does not allocate. */
  setEqualizerGains(gains) {
    const view = new Float64Array(this.#exports.memory.buffer, this.#exports.eqGainsPtr(), this.#exports.eqMaxBands());
    if (gains.length !== this.#exports.eqBands()) throw new RangeError(`${this.#exports.eqBands()} gains`);
    view.set(gains);
    if (!this.#exports.eqSetGains()) throw new RangeError('equalizer not configured, or a gain is not finite');
  }

  /**
   * Processes `count` samples of `channel` already written to the input view: the output view (and,
   * with `bands`, the band views) receive as many, `latency` late. Real time; does not allocate.
   */
  processEqualizer(channel, count, bands = false) {
    if (this.#exports.eqProcess(channel, count, bands ? 1 : 0) < 0) throw new RangeError('equalizer not configured, or an invalid channel or count');
  }

  /** Statistics of a channel: windows fitted, tones in total and in the last window. */
  equalizerStats(channel) {
    return { windows: this.#exports.eqWindows(channel), tones: this.#exports.eqTones(channel), lastTones: this.#exports.eqLastTones(channel) };
  }

  /**
   * A whole mono signal through a freshly configured equalizer (offline, allocates): the equalized signal
   * and its bands, latency removed (the input is followed by `latency` zeros so its end comes out).
   */
  equalize(signal, options, gains) {
    const views = this.configureEqualizer({ ...options, channels: 1 });
    if (gains) this.setEqualizerGains(gains);
    const total = signal.length + views.latency;
    const output = new Float32Array(total);
    const bands = Array.from({ length: views.bands }, () => new Float32Array(total));
    const block = views.input.length;
    for (let at = 0; at < total; at += block) {
      const count = Math.min(block, total - at);
      for (let i = 0; i < count; i++) views.input[i] = at + i < signal.length ? signal[at + i] : 0;
      this.processEqualizer(0, count, true);
      output.set(views.output.subarray(0, count), at);
      bands.forEach((band, b) => band.set(views.bandTracks[b] .subarray(0, count), at));
    }
    return { output: output.subarray(views.latency), bands: bands.map((b) => b.subarray(views.latency)), latency: views.latency };
  }

  /**
   * `equalize` with every window's tone fit batched (offline or over a buffer, allocates): the windows'
   * seeds first, then the fit's steps (the seeds' fit, the significance prune, refinement, the DC snap,
   * exactly as the stream's `analyse`) round by round across all windows, each round's FAFT values from
   * `evaluate` at once; then the stream's synthesis and residual filters with those tones. With the
   * default WebAssembly evaluator the result is the stream's, bit for bit; with the GPU's
   * (webgpu/faft.mjs `FaftGpu.batchEvaluator`) the values are f32 throughout (about 1e-6).
   */
  async equalizeBatch(
    signal,
    options,
    gains,
    evaluate = (requests, windows) => this.faftBatchValuesWasm(requests, windows),
) {
    const started = clockMs();
    const e = this.#exports;
    const views = this.configureEqualizer({ ...options, channels: 1 });
    if (gains) this.setEqualizerGains(gains);
    const T = options.window ?? 2048;
    const H = T / 2;
    const total = signal.length + views.latency;
    // The stream's input: f32, non-finite samples as 0, then zeros.
    const x = new Float32Array(total + T + 4);
    for (let i = 0; i < signal.length; i++) {
      const v = Math.fround(signal[i]);
      x[i] = Number.isFinite(v) ? v : 0;
    }
    // The stream runs hop k once (k + 1)H + 2 samples have arrived.
    const hops = Math.max(0, Math.floor((total - 2) / H));
    const memory = () => e.memory.buffer;
    const curveView = () => new Float32Array(memory(), e.faftInputPtr(), T + 3);
    const stride = e.faftAtomStride();
    const needsAtoms = evaluate.needsAtoms === true;
    const windows = {
      curves: [], atoms: new Float32Array(0), offsets: new Int32Array(hops).fill(-1), stride, intervals: T,
    };
    const atomChunks = [];
    let atomsUsed = 0;
    const seedsOf = [];
    for (let k = 0; k < hops; k++) {
      const start = (k - 1) * H;
      const curve = new Float32Array(T + 3);
      for (let i = 0; i < T + 3; i++) {
        const t = start - 1 + i;
        curve[i] = t >= 0 ? x[t] : 0;
      }
      curveView().set(curve);
      const count = e.eqWindowSeeds();
      if (count <= 0) {
        windows.curves.push(null);
        seedsOf.push(new Float64Array(0));
        continue;
      }
      windows.curves.push(curve);
      if (needsAtoms) {
        atomChunks.push(new Float32Array(memory(), e.faftAtoms32Ptr(), T * stride).slice());
        windows.offsets[k] = atomsUsed;
        atomsUsed += T * stride;
      }
      seedsOf.push(new Float64Array(memory(), e.eqSeedsPtr(), count).slice());
    }
    if (needsAtoms) {
      windows.atoms = new Float32Array(atomsUsed);
      let at = 0;
      for (const chunk of atomChunks) {
        windows.atoms.set(chunk, at);
        at += chunk.length;
      }
    }
    const seedsMs = clockMs() - started;
    // The fit's steps per window, as generators that yield the values each step needs.
    const PLAN = 3;
    const EXTERNAL = 4;
    const reach = e.eqReach();
    const rounds = e.eqRefine();
    const relativeFloor = e.eqRelativeFloor();
    const resolution = (2 * Math.PI) / T;
    const tolerance = 1e-4 * resolution;
    const omegasIn = () => new Float64Array(memory(), e.faftOmegasPtr(), e.faftMaxQueries());
    const batchOmegas = () => new Float64Array(memory(), e.faftBatchOmegasPtr(), e.faftBatchMax());
    const batchOut = () => new Float64Array(memory(), e.faftBatchOutPtr(), 2 * e.faftBatchMax());
    const fit = (list, mode) => {
      omegasIn().set(list);
      return e.faftFit(list.length, reach, 1, mode);
    };

    function* fitStep(window, list) {
      fit(list, PLAN);
      const values = yield { window, omegas: batchOmegas().slice(0, e.tonePlanned()) };
      batchOut().set(values);
      return { n: fit(list, EXTERNAL), values };
    }
    function* pipeline(window, seeds) {
      let step = yield* fitStep(window, seeds);
      let n = step.n;
      let current = seeds;
      let values = step.values;
      if (n <= 0) return new Float64Array(0);
      let loudest = 0;
      for (let p = 0; p < n; p++) loudest = Math.max(loudest, e.toneObserved(p));
      const floor = relativeFloor * loudest;
      const keptList = [];
      for (let p = 0; p < n; p++) if (e.toneSignificance(p) >= floor) keptList.push(e.toneOmega(p));
      if (keptList.length === 0) return new Float64Array(0);
      const kept = Float64Array.from(keptList);
      if (rounds > 0) {
        step = yield* fitStep(window, kept);
        n = step.n;
        current = kept;
        values = step.values;
        for (let round = 0; n > 0 && round < rounds; round++) {
          e.refineStep(reach, 1, PLAN);
          const differences = yield { window, omegas: batchOmegas().slice(0, e.tonePlanned()) };
          // Other windows fitted meanwhile: this one's fit again, from its own values.
          batchOut().set(values);
          fit(current, EXTERNAL);
          batchOut().set(differences);
          const distinct = e.refineStep(reach, 1, EXTERNAL);
          const largest = e.toneLargestStep();
          const moved = new Float64Array(memory(), e.toneMovedPtr(), distinct).slice();
          step = yield* fitStep(window, moved);
          n = step.n;
          current = moved;
          values = step.values;
          if (n <= 0 || largest <= tolerance) break;
        }
      } else if (kept.length < n) {
        step = yield* fitStep(window, kept);
        n = step.n;
        current = kept;
        values = step.values;
      }
      if (n <= 0) return new Float64Array(0);
      // The DC snap: a tone under half a resolution goes to 0, then a refit.
      let snap = false;
      for (let p = 0; p < n; p++) {
        const w = e.toneOmega(p);
        if (w > 0 && w < 0.5 * resolution) snap = true;
      }
      if (snap) {
        const snapped = [];
        for (let p = 0; p < n; p++) {
          const w = e.toneOmega(p) < 0.5 * resolution ? 0 : e.toneOmega(p);
          if (snapped.length > 0 && snapped[snapped.length - 1] === w) continue;
          snapped.push(w);
        }
        step = yield* fitStep(window, Float64Array.from(snapped));
        n = step.n;
        if (n <= 0) return new Float64Array(0);
      }
      void current;
      const tones = new Float64Array(1 + 3 * n);
      tones[0] = n;
      for (let p = 0; p < n; p++) {
        tones[1 + 3 * p] = e.toneOmega(p);
        tones[2 + 3 * p] = e.toneAlpha(p);
        tones[3 + 3 * p] = e.toneBeta(p);
      }
      return tones;
    }
    const toneTable = seedsOf.map(() => new Float64Array(1));
    const running = new Map ();
    seedsOf.forEach((seeds, window) => {
      if (seeds.length === 0) return;
      const generator = pipeline(window, seeds);
      const first = generator.next(new Float64Array(0));
      if (first.done) toneTable[window] = first.value;
      else running.set(window, { generator, request: first.value });
    });
    const stats = { windows: hops, fitted: running.size, rounds: 0, values: 0, seedsMs, evaluateMs: 0, fitMs: 0, streamMs: 0, totalMs: 0 };
    const fitStarted = clockMs();
    while (running.size > 0) {
      const entries = [...running];
      const requests = entries.map(([, entry]) => entry.request);
      const before = clockMs();
      const results = await evaluate(requests, windows);
      stats.evaluateMs += clockMs() - before;
      stats.rounds++;
      stats.values += requests.reduce((sum, request) => sum + request.omegas.length, 0);
      entries.forEach(([window, entry], index) => {
        const next = entry.generator.next(results[index]);
        if (next.done) {
          toneTable[window] = next.value;
          running.delete(window);
        } else {
          entry.request = next.value;
        }
      });
    }
    stats.fitMs = clockMs() - fitStarted - stats.evaluateMs;
    const streamStarted = clockMs();
    // The stream, with each window's tones from the table: blocks end where a hop starts, so the slot
    // holds that hop's tones when it runs.
    const slot = () => new Float64Array(memory(), e.eqToneSlotPtr(), 1 + 3 * e.faftMaxTones());
    e.eqUseToneSlot(1);
    try {
      const output = new Float32Array(total);
      const bands = Array.from({ length: views.bands }, () => new Float32Array(total));
      let at = 0;
      for (let k = 0; at < total; k++) {
        // Hop k runs on sample (k + 1)H + 2; a block ends there (or at the end).
        const trigger = (k + 1) * H + 2;
        const end = Math.min(total, trigger, at + views.input.length);
        if (k < hops) slot().set(toneTable[k] .length > 0 ? toneTable[k] : [0]);
        for (let from = at; from < end;) {
          const count = Math.min(views.input.length, end - from);
          views.input.set(x.subarray(from, from + count));
          this.processEqualizer(0, count, true);
          output.set(views.output.subarray(0, count), from);
          bands.forEach((band, b) => band.set(views.bandTracks[b] .subarray(0, count), from));
          from += count;
        }
        at = end;
        if (end < trigger) k--; // the block was cut by its size: the same hop is still ahead
      }
      stats.streamMs = clockMs() - streamStarted;
      stats.totalMs = clockMs() - started;
      return { output: output.subarray(views.latency), bands: bands.map((b) => b.subarray(views.latency)), latency: views.latency, stats };
    } finally {
      e.eqUseToneSlot(0);
    }
  }

  /**
   * The batch fit's values in WebAssembly, window by window: the stream's own FAST values (four f32
   * lanes), so `equalizeBatch` with it equals `equalize` exactly. Re-prepares each window from its curve.
   */
  faftBatchValuesWasm(requests, windows) {
    const e = this.#exports;
    const memory = () => e.memory.buffer;
    let prepared = -1;
    return requests.map((request) => {
      if (request.window !== prepared) {
        new Float32Array(memory(), e.faftInputPtr(), windows.intervals + 3).set(windows.curves[request.window]);
        if (e.eqWindowSeeds() <= 0) throw new Error(`window ${request.window} has nothing to fit`);
        prepared = request.window;
      }
      const count = request.omegas.length;
      new Float64Array(memory(), e.faftBatchOmegasPtr(), count).set(request.omegas);
      if (!e.faftRectBatchFast(count)) throw new RangeError('FAFT batch: bad count');
      return new Float64Array(memory(), e.faftBatchOutPtr(), 2 * count).slice();
    });
  }

  // ---- SWL2 (assembly/swl2.ts; sinwhisky.rs src/layered, docs/SWL2.md) ---------------------------

  /** One correction chunk of `x` from `slot`'s state (left in the slot): its payload and reconstruction. */
  #swl2Chunk(slot, x, tolerance, exponent, cascade, effort) {
    const e = this.#exports;
    if (!e.swl2EncodeBegin(slot, tolerance, exponent, cascade ? 1 : 0, effort)) throw new RangeError('invalid SWL2 chunk settings');
    const memory = () => e.memory.buffer;
    const block = 1 << exponent;
    const parts = [];
    let length = 0;
    const decoded = new Int32Array(x.length);
    const input = new Int32Array(memory(), e.swl2SamplesInPtr(), e.swl2MaxBlock());
    const output = new Int32Array(memory(), e.swl2SamplesOutPtr(), e.swl2MaxBlock());
    const drain = () => {
      const bytes = e.swl2EncodeBytes();
      const keep = 8; // the coder's tail may still be changed by a carry; leave a few bytes
      if (bytes > 65536) {
        const take = bytes - keep;
        parts.push(new Uint8Array(memory(), e.swl2BytesOutPtr(), take).slice());
        length += take;
        e.swl2EncodeDrained(take);
      }
    };
    for (let at = 0; at < x.length; at += block) {
      const count = Math.min(block, x.length - at);
      input.set(x.subarray(at, at + count));
      if (e.swl2EncodeBlock(count) < 0) throw new RangeError('SWL2 block failed');
      decoded.set(output.subarray(0, count), at);
      drain();
    }
    const tail = e.swl2EncodeEnd();
    parts.push(new Uint8Array(memory(), e.swl2BytesOutPtr(), tail).slice());
    length += tail;
    const payload = new Uint8Array(length);
    let offset = 0;
    for (const part of parts) {
      payload.set(part, offset);
      offset += part.length;
    }
    return { payload, decoded };
  }

  /**
   * Encodes integer channels (1–6, equal lengths, |sample| < 2²³) as an SWL2 version 4 stream that
   * sinwhisky.rs decodes: correction chunks only, chosen segment by segment as its `encode_auto` does
   * (stereo mode, then the block size) from the state the previous segment left. Allocates; not for the
   * audio thread.
   */
  encodeSwl2(channels, options = {}) {
    const e = this.#exports;
    const C = channels.length;
    const N = channels[0]?.length ?? 0;
    const tolerance = options.tolerance ?? 0;
    const exponent = options.segmentExponent ?? 0;
    const cascade = options.cascade ?? true;
    const effort = options.effort === 'fast' ? 0 : 1;
    if (C < 1 || C > e.swl2MaxSlots() - 2 || N < 1 || N > 1 << 28 || channels.some((c) => c.length !== N)) throw new RangeError('SWL2: 1 to 6 channels of equal, nonzero length');
    if (!(tolerance >= 0 && tolerance <= 1 << 20 && Number.isInteger(tolerance))) throw new RangeError('SWL2 tolerance');
    if (!Number.isInteger(exponent) || (exponent !== 0 && !(exponent >= 12 && exponent <= 24))) throw new RangeError('SWL2 segment exponent');
    if (!Number.isSafeInteger(options.keyInterval ?? 0) || (options.keyInterval ?? 0) < 0) throw new RangeError('SWL2 key interval');
    if (options.effort !== undefined && options.effort !== 'fast' && options.effort !== 'full') throw new RangeError('SWL2 effort');
    const source = channels.map((c) => {
      for (let i = 0; i < c.length; i++) {
        if (!Number.isInteger(c[i]) || Math.abs(c[i]) >= 1 << 23) throw new RangeError('SWL2 samples must be integers under 2^23');
      }
      const v = Int32Array.from(c);
      if (v.some((s) => Math.abs(s) >= 1 << 23)) throw new RangeError('SWL2 samples must be under 2^23');
      return v;
    });
    const centre = options.blockSearch === false ? (options.blockExponent ?? (cascade ? 13 : 12)) : cascade ? 13 : 12;
    if (!Number.isInteger(centre) || centre < 8 || centre > 15) throw new RangeError('SWL2 block exponent');
    const size = exponent === 0 ? N : 1 << exponent;
    const header = new Uint8Array(16);
    header.set([0x53, 0x57, 0x4c, 0x32, 4, C, exponent, 0]);
    new DataView(header.buffer).setUint32(8, N, true);
    new DataView(header.buffer).setUint32(12, tolerance, true);
    const records = [header];
    const decoded = source.map(() => new Int32Array(N));
    const segments = [];
    const chunkBytes = (payload) => 1 + 5 + payload.length;
    for (let index = 0, start = 0; start < N; index++, start += size) {
      const end = Math.min(N, start + size);
      const key = index === 0 || ((options.keyInterval ?? 0) > 0 && index % options.keyInterval === 0);
      if (key) for (let c = 0; c < C; c++) e.swl2Reset(c);
      const part = source.map((c) => c.subarray(start, end));
      // Slots: 0..C−1 hold the inherited states; trials go to the others (correction states are copied).
      const code = (slot, from, x, exp) => {
        e.swl2Copy(from, slot);
        return this.#swl2Chunk(slot, x, tolerance, exp, cascade, effort);
      };
      const walk = (c, x, bestSlot, trialSlot, first) => {
        let best = { ...first, exponent: centre };
        if (options.blockSearch === false) return best;
        // Block size: one step either way, then onwards while it pays (encode_auto).
        for (const direction of [-1, 1]) {
          for (let exp = centre + direction; exp >= 8 && exp <= 15; exp += direction) {
            const before = best.exponent;
            const trial = code(trialSlot, c, x, exp);
            if (chunkBytes(trial.payload) < chunkBytes(best.payload)) {
              best = { ...trial, exponent: exp };
              e.swl2Copy(trialSlot, bestSlot);
            }
            if (best.exponent === before) break;
          }
        }
        return best;
      };
      let stereo = 0;
      const chosen = [];
      if (C === 2 && tolerance === 0 && options.stereoSearch !== false) {
        // The cheapest pair among the four signals coded plainly, each from its channel's state.
        const [l, r] = part;
        const side = l.map((v, i) => v - r[i]);
        const mid = l.map((v, i) => (v + r[i]) >> 1);
        const trials = [code(2, 0, l, centre), code(3, 1, r, centre), code(4, 0, side, centre), code(5, 1, side, centre), code(6, 0, mid, centre)];
        const pairs = [[0, 1], [0, 3], [2, 1], [4, 3]];
        const costs = pairs.map(([a, b]) => chunkBytes(trials[a] .payload) + chunkBytes(trials[b] .payload));
        stereo = costs.indexOf(Math.min(...costs));
        const signals = [[l, r], [l, side], [side, r], [mid, side]][stereo];
        const [a, b] = pairs[stereo];
        chosen.push(walk(0, signals[0], 2 + a, 7, trials[a]), walk(1, signals[1], 2 + b, 7, trials[b]));
        e.swl2Copy(2 + a, 0);
        e.swl2Copy(2 + b, 1);
      } else {
        for (let c = 0; c < C; c++) {
          const plain = code(C, c, part[c], centre);
          chosen.push(walk(c, part[c], C, C + 1, plain));
          e.swl2Copy(C, c);
        }
      }
      const bodyLength = 1 + chosen.reduce((sum, ch) => sum + chunkBytes(ch.payload), 0);
      const record = new Uint8Array(4 + bodyLength);
      const view = new DataView(record.buffer);
      view.setUint32(0, bodyLength, true);
      record[4] = (key ? 1 : 0) | (stereo << 1);
      let at = 5;
      for (const ch of chosen) {
        record[at] = 1;
        record[at + 1] = 3;
        view.setUint32(at + 2, ch.payload.length, true);
        record.set(ch.payload, at + 6);
        at += 6 + ch.payload.length;
      }
      records.push(record);
      const restored = this.#swl2Inverse(chosen.map((ch) => ch.decoded), stereo);
      restored.forEach((ch, c) => decoded[c] .set(ch, start));
      segments.push({ key, stereo: SWL2_STEREO[stereo], blockExponents: chosen.map((ch) => ch.exponent), bytes: record.length });
    }
    const bytes = new Uint8Array(records.reduce((sum, r) => sum + r.length, 0));
    let offset = 0;
    for (const r of records) {
      bytes.set(r, offset);
      offset += r.length;
    }
    return { bytes, decoded, segments };
  }

  /** Coded channels to left and right (docs/SWL2.md §1), clamped to ±2²³. */
  #swl2Inverse(coded, stereo) {
    const clamp = (v) => Math.max(-(1 << 23), Math.min(1 << 23, v));
    if (stereo === 0 || coded.length !== 2) return coded.map((c) => c.map(clamp));
    const [a, b] = coded;
    const l = new Int32Array(a.length);
    const r = new Int32Array(a.length);
    for (let i = 0; i < a.length; i++) {
      if (stereo === 1) {
        l[i] = a[i];
        r[i] = a[i] - b[i];
      } else if (stereo === 2) {
        l[i] = a[i] + b[i];
        r[i] = b[i];
      } else {
        const mid = (a[i] * 2) | (b[i] & 1);
        l[i] = (mid + b[i]) >> 1;
        r[i] = (mid - b[i]) >> 1;
      }
      l[i] = clamp(l[i]);
      r[i] = clamp(r[i]);
    }
    return [l, r];
  }

  /**
   * Decodes an SWL2 version 4 stream of correction chunks (from encodeSwl2 or sinwhisky.rs; streams with
   * gradient or anchor chunks are rejected) from segment `first` on, which must be a key. Allocates.
   */
  decodeSwl2(bytes, first = 0) {
    const e = this.#exports;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const corrupt = () => new RangeError('SWL2 stream is corrupt or unsupported');
    if (bytes.length < 16 || String.fromCharCode(...bytes.subarray(0, 4)) !== 'SWL2' || bytes[4] !== 4 || bytes[7] !== 0) throw corrupt();
    const C = bytes[5];
    const exponent = bytes[6];
    const N = view.getUint32(8, true);
    const tolerance = view.getUint32(12, true);
    if (C < 1 || C > Math.min(8, e.swl2MaxSlots()) || N < 1 || N > 1 << 28 || tolerance > 1 << 20 || (exponent !== 0 && (exponent < 12 || exponent > 24))) throw corrupt();
    const size = exponent === 0 ? N : 2 ** exponent;
    const segmentCount = Math.ceil(N / size);
    if (!Number.isInteger(first) || first < 0 || first >= segmentCount) throw new RangeError('SWL2 segment');
    let at = 16;
    for (let s = 0; s < first; s++) {
      if (at + 4 > bytes.length) throw corrupt();
      at += 4 + view.getUint32(at, true);
    }
    const out = Array.from({ length: C }, () => new Int32Array(N - first * size));
    const window = new Uint8Array(e.memory.buffer, e.swl2BytesInPtr(), e.swl2InBytes());
    const samples = () => new Int32Array(e.memory.buffer, e.swl2SamplesOutPtr(), e.swl2MaxBlock());
    for (let index = first; index < segmentCount; index++) {
      if (at + 5 > bytes.length) throw corrupt();
      const length = view.getUint32(at, true);
      const end = at + 4 + length;
      if (end > bytes.length) throw corrupt();
      const flags = bytes[at + 4];
      const key = (flags & 1) === 1;
      const stereo = (flags >> 1) & 3;
      if (flags >> 3 !== 0 || (index === 0 && !key) || (index === first && !key) || (stereo !== 0 && (C !== 2 || tolerance !== 0))) throw corrupt();
      if (key) for (let c = 0; c < C; c++) e.swl2Reset(c);
      const frames = Math.min(size, N - index * size);
      let p = at + 5;
      const coded = [];
      for (let c = 0; c < C; c++) {
        if (p + 6 > end || bytes[p] !== 1 || bytes[p + 1] !== 3) throw corrupt();
        const payloadLength = view.getUint32(p + 2, true);
        const payload = bytes.subarray(p + 6, p + 6 + payloadLength);
        if (p + 6 + payloadLength > end) throw corrupt();
        p += 6 + payloadLength;
        // The payload through the kernel's window, refilled as the decoder consumes it.
        let windowStart = 0;
        let windowLength = Math.min(payload.length, window.length);
        window.set(payload.subarray(0, windowLength));
        if (!e.swl2DecodeBegin(c, tolerance, windowLength, windowLength === payload.length ? 1 : 0)) throw corrupt();
        const channel = new Int32Array(frames);
        let done = 0;
        while (done < frames) {
          const got = e.swl2DecodeRun(Math.min(e.swl2MaxBlock(), frames - done));
          if (got < 0) throw corrupt();
          channel.set(samples().subarray(0, got), done);
          done += got;
          if (done < frames && got < Math.min(e.swl2MaxBlock(), frames - done + got)) {
            if (windowStart + windowLength === payload.length) throw corrupt();
            windowStart += e.swl2DecodeConsumed();
            windowLength = Math.min(window.length, payload.length - windowStart);
            window.set(payload.subarray(windowStart, windowStart + windowLength));
            e.swl2DecodeRefill(windowLength, windowStart + windowLength === payload.length ? 1 : 0);
          }
        }
        e.swl2DecodeEnd();
        coded.push(channel);
      }
      if (p !== end) throw corrupt();
      this.#swl2Inverse(coded, stereo).forEach((ch, c) => out[c] .set(ch, (index - first) * size));
      at = end;
    }
    if (at !== bytes.length) throw corrupt();
    return out;
  }

}
