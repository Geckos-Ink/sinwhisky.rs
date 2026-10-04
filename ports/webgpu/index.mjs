// WebGPU host for the two kernels next to this file: sinwhisky.wgsl (the original-angular resampler, one
// invocation per output sample) and aft.wgsl (FAFT of the AFT's curve, one invocation per frequency).
// Both are f32 and take their geometry from the WebAssembly port (../wasm-simd), which fits the circles
// in f64 and stays the reference: GPU results are for dense work (zoomed signals, spectra, peak seeds).
// Runs wherever WebGPU exists (browsers, Electron, Deno); Node has no WebGPU.

/** WebGPU's buffer usage and map mode flags (the spec's values). */
const MAP_READ = 0x0001;
const COPY_SRC = 0x0004;
const COPY_DST = 0x0008;
const UNIFORM = 0x0040;
const STORAGE = 0x0080;
const MAP_MODE_READ = 0x0001;
const TAU = 2 * Math.PI;

async function source(name) {
  const response = await fetch(new URL(`./${name}`, import.meta.url));
  if (!response.ok) throw new Error(`${name}: ${response.status}`);
  return response.text();
}

/**
 * Per model, the anchored f32 form sinwhisky.wgsl reads (10 values): flag, R, sin a0, cos a0, a1 − a0,
 * sin a1, cos a1, a2 − a1, line value, line slope, from the f64 models of `SinWhisky.models` (flag, cy, R,
 * a0, a1, a2, value, slope). Trigonometry and angle differences are computed in f64 before rounding.
 */
export function anchoredModels(models, stride = 8) {
  const count = models.length / stride;
  const out = new Float32Array(10 * count);
  for (let j = 0; j < count; j++) {
    const [flag, , radius, a0, a1, a2, value, slope] = models.subarray(j * stride, j * stride + 8);
    out.set([flag, radius, Math.sin(a0), Math.cos(a0), a1 - a0, Math.sin(a1), Math.cos(a1), a2 - a1, value, slope], 10 * j);
  }
  return out;
}

/** Per frequency: ω, ω/2π split into an exactly multipliable part (11 bits) and the rest; cos ω, sin ω. */
function frequencyTable(omegas) {
  const table = new Float32Array(8 * omegas.length);
  for (let k = 0; k < omegas.length; k++) {
    const cycles = omegas[k] / TAU;
    const high = Math.round(cycles * 2048) / 2048;
    table.set([omegas[k], high, cycles - high, 0, Math.cos(omegas[k]), Math.sin(omegas[k])], 8 * k);
  }
  return table;
}

export class SinWhiskyGpu {
  #device;
  #resampler;
  #aft;

  constructor(device, resampler, aft) {
    this.#device = device;
    this.#resampler = resampler;
    this.#aft = aft;
  }

  /** A device and both compiled kernels, or null where WebGPU or an adapter is missing. */
  static async create(gpu = globalThis.navigator?.gpu) {
    const adapter = await gpu?.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    const compile = async (name) => device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: await source(name) }), entryPoint: 'main' },
    });
    const [resampler, aft] = await Promise.all([compile('sinwhisky.wgsl'), compile('aft.wgsl')]);
    return new SinWhiskyGpu(device, resampler, aft);
  }

  #storage(data, usage = STORAGE | COPY_DST) {
    const buffer = this.#device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage });
    this.#device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  /** Runs `pipeline` over `count` invocations with the given input buffers and returns `outputBytes`. */
  async #run(pipeline, inputs, outputBytes, params, count) {
    const device = this.#device;
    const result = device.createBuffer({ size: Math.max(16, outputBytes), usage: STORAGE | COPY_SRC });
    const read = device.createBuffer({ size: Math.max(16, outputBytes), usage: MAP_READ | COPY_DST });
    const uniform = this.#storage(Uint32Array.from(params), UNIFORM | COPY_DST);
    const buffers = [...inputs, result, uniform];
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    encoder.copyBufferToBuffer(result, 0, read, 0, outputBytes);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(MAP_MODE_READ);
    const bytes = read.getMappedRange().slice(0, outputBytes);
    read.unmap();
    for (const buffer of [...buffers, read]) buffer.destroy();
    return bytes;
  }

  /**
   * `resample(samples, ResampleParams::original(zoom))` in f32: `models` are `SinWhisky.models(samples)`
   * (f64, fitted by the WebAssembly port).
   */
  async resample(samples, models, zoom, modelStride = 8) {
    const length = samples.length;
    if (length < 2 || zoom === 0) return Float32Array.from(samples);
    const q = zoom + 1;
    const total = (length - 1) * q + 1;
    const bytes = await this.#run(
      this.#resampler,
      [this.#storage(Float32Array.from(samples)), this.#storage(anchoredModels(models, modelStride))],
      total * 4,
      [length, q, total, 0],
      total,
    );
    return new Float32Array(bytes);
  }

  /**
   * The unnormalized rectangular FAFT at each of `omegas` (phase origin at the window start) as (re, im)
   * pairs, from `atoms` (`SinWhisky.atoms()`, `stride` f64 per interval), up to 4096 intervals.
   */
  async rect(atoms, stride, omegas) {
    const intervals = atoms.length / stride;
    if (!Number.isInteger(intervals) || intervals < 1 || intervals > 4096) throw new RangeError('1 to 4096 intervals');
    const bytes = await this.#run(
      this.#aft,
      [this.#storage(Float32Array.from(atoms)), this.#storage(frequencyTable(omegas))],
      omegas.length * 8,
      [intervals, omegas.length, stride, 0],
      omegas.length,
    );
    return new Float32Array(bytes);
  }

  /**
   * Normalized magnitudes on lower + k·step, k < count, as `SinWhisky.scan`: with `density` > 0 the Hann
   * window, its ±2π/N neighbours `density` steps away on one extended rectangular grid.
   */
  async scan(atoms, stride, lower, step, count, density = 0) {
    const intervals = atoms.length / stride;
    const total = count + 2 * density;
    const v = await this.rect(atoms, stride, Float64Array.from({ length: total }, (_, k) => lower + step * (k - density)));
    const norm = density > 0 ? 2 / intervals : 1 / intervals;
    return Float64Array.from({ length: count }, (_, j) => {
      let re = v[2 * (j + density)];
      let im = v[2 * (j + density) + 1];
      if (density > 0) {
        re = 0.5 * re - 0.25 * (v[2 * j] + v[2 * (j + 2 * density)]);
        im = 0.5 * im - 0.25 * (v[2 * j + 1] + v[2 * (j + 2 * density) + 1]);
      }
      return Math.hypot(re, im) * norm;
    });
  }

  destroy() {
    this.#device.destroy();
  }
}
