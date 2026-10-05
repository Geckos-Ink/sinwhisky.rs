// Transposed from appx/dsp at ec84edcd331af8810be9df42cf5d16f3a348d6d5 (2026-10-05).
// FAFT on the GPU (ports/README.md): the rectangular transform of SinWhisky's circle curve at
// many frequencies at once, one WebGPU invocation per frequency, from the rotations the WebAssembly kernel
// prepared (SinWhisky.prepareFaft, SinWhisky.faftAtoms). Each frequency reads its own window's rotations, so one
// dispatch serves many windows: dense scans (peak seeds, zoomed spectra), and the component equalizer's
// tone fits batched over a buffer (`batchEvaluator` for SinWhisky.equalizeBatch). f32 on the GPU with
// compensated sums, about 1e-6 of the strongest value, against
// the kernel's f64; the least-squares solves and Newton steps stay in WebAssembly. Runs where WebGPU
// exists (Electron's renderer, browsers); Node has no WebGPU. test.html checks its parity with the
// WebAssembly reference and exercises the batched equalizer.


/**
 * The kernel. Per frequency ω: the DC and line terms once, then over every interval its two arcs in
 * endpoint form (assembly/faft.ts: [f'(1)q − f'(0) + iω f(1) q + r²a E₀(−ω)] / ((ω − r)(ω + r)), from the
 * prepared endpoint data), or within |ω ∓ rate| < 0.25 the moment form E₀(±rate − ω) (series under
 * |δ| < 0.25, the ratio form above), times the interval's support phase e^{−iωn}. The phase is not chained in f32 (its error would grow with n): ω/2π is split on the CPU into
 * a part with at most 11 significant bits, whose product with n < 2¹³ is exact in f32, and a small rest,
 * so the fraction of a turn is exact to about 1e-7.
 */
export const FAFT_WGSL = /* wgsl */ `
struct Params {
  intervals: u32,
  count: u32,
  stride: u32,
  unused: u32,
};

@group(0) @binding(0) var<storage, read> atoms: array<f32>;
@group(0) @binding(1) var<storage, read> omegas: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> values: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;
// Where each frequency's window starts in atoms (f32 index): many windows in one dispatch.
@group(0) @binding(4) var<storage, read> bases: array<u32>;

fn e0(delta: f32, e: vec2<f32>) -> vec2<f32> {
  if (abs(delta) < 0.25) {
    let d2 = delta * delta;
    let re = 1.0 + d2 * (-1.0 / 6.0 + d2 * (1.0 / 120.0 + d2 * (-1.0 / 5040.0 + d2 * (1.0 / 362880.0))));
    let im = delta * (0.5 + d2 * (-1.0 / 24.0 + d2 * (1.0 / 720.0 + d2 * (-1.0 / 40320.0 + d2 * (1.0 / 3628800.0)))));
    return vec2<f32>(re, im);
  }
  return vec2<f32>(e.y / delta, (1.0 - e.x) / delta);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = id.x;
  if (k >= params.count) {
    return;
  }
  let w = omegas[k];
  let omega = w.x;
  let q = vec2<f32>(cos(omega), -sin(omega));
  let dc = e0(-omega, q);
  var line: vec2<f32>;
  if (abs(omega) < 0.25) {
    let d2 = omega * omega;
    line = vec2<f32>(
      0.5 + d2 * (-1.0 / 8.0 + d2 * (1.0 / 144.0 + d2 * (-1.0 / 5760.0))),
      -omega * (1.0 / 3.0 + d2 * (-1.0 / 30.0 + d2 * (1.0 / 840.0 + d2 * (-1.0 / 45360.0)))));
  } else {
    let d = q - dc;
    line = vec2<f32>(d.y / -omega, -d.x / -omega);
  }
  // Kahan-compensated sum: f32 over thousands of intervals keeps about 1e-7 of the total.
  var sum = vec2<f32>(0.0, 0.0);
  var carry = vec2<f32>(0.0, 0.0);
  let origin = bases[k];
  for (var n = 0u; n < params.intervals; n = n + 1u) {
    let base = origin + n * params.stride;
    var p = atoms[base] * dc + atoms[base + 1u] * line;
    for (var s = 0u; s < 2u; s = s + 1u) {
      let b = base + 2u + s * 9u;
      let a = atoms[b];
      let bb = atoms[b + 1u];
      if (a == 0.0 && bb == 0.0) {
        continue;
      }
      let rate = atoms[b + 2u];
      let du = rate - omega;
      let dd = -rate - omega;
      if (min(abs(du), abs(dd)) >= 0.25) {
        let endSlope = atoms[b + 6u];
        let change = omega * atoms[b + 7u];
        let curvature = atoms[b + 8u];
        let numerator = vec2<f32>(
          q.x * endSlope - atoms[b + 5u] - q.y * change + curvature * dc.x,
          q.y * endSlope + q.x * change + curvature * dc.y);
        p = p + numerator / (du * dd);
        continue;
      }
      let cr = atoms[b + 3u];
      let sr = atoms[b + 4u];
      let up = e0(rate - omega, vec2<f32>(cr * q.x - sr * q.y, cr * q.y + sr * q.x));
      let down = e0(-rate - omega, vec2<f32>(cr * q.x + sr * q.y, cr * q.y - sr * q.x));
      let cosine = 0.5 * (up + down) - dc;
      let sine = vec2<f32>(0.5 * (up.y - down.y), -0.5 * (up.x - down.x));
      p = p + a * cosine + bb * sine;
    }
    let turns = fract(fract(w.y * f32(n)) + w.z * f32(n));
    let angle = -6.283185307179586 * turns;
    let support = vec2<f32>(cos(angle), sin(angle));
    let term = vec2<f32>(support.x * p.x - support.y * p.y, support.x * p.y + support.y * p.x) - carry;
    let next = sum + term;
    carry = (next - sum) - term;
    sum = next;
  }
  values[k] = sum;
}
`;

/** WebGPU's buffer usage and map mode flags (the spec's values; TypeScript's DOM types declare no constants). */
const MAP_READ = 0x0001;
const COPY_SRC = 0x0004;
const COPY_DST = 0x0008;
const UNIFORM = 0x0040;
const STORAGE = 0x0080;
const MAP_MODE_READ = 0x0001;

/** Per-frequency inputs: ω, then ω/2π split into an exactly multipliable part and the rest. */
function frequencyTable(omegas) {
  const table = new Float32Array(4 * omegas.length);
  for (let k = 0; k < omegas.length; k++) {
    const omega = omegas[k];
    const cycles = omega / (2 * Math.PI);
    const high = Math.round(cycles * 2048) / 2048;
    table[4 * k] = omega;
    table[4 * k + 1] = high;
    table[4 * k + 2] = cycles - high;
  }
  return table;
}

export class FaftGpu {
  #device;
  #pipeline;

          constructor(device, pipeline) {
    this.#device = device;
    this.#pipeline = pipeline;
  }

  /** A device and the compiled kernel, or null where WebGPU or an adapter is missing. */
  static async create(gpu = (globalThis.navigator)?.gpu) {
    const adapter = await gpu?.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    const module = device.createShaderModule({ code: FAFT_WGSL });
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    return new FaftGpu(device, pipeline);
  }

  /**
   * The unnormalized rectangular transform at each of `omegas` (phase origin at the window start), as
   * (re, im) pairs, from `atoms` (SinWhisky.faftAtoms, `stride` f64 per interval; SinWhisky.faftAtomStride). Inputs
   * up to 4096 intervals (the phase split is exact up to there for |ω| ≤ 3π).
   */
  async rect(atoms, stride, omegas) {
    const intervals = atoms.length / stride;
    if (stride !== 20 || !Number.isInteger(intervals) || intervals < 1 || intervals > 4096) throw new RangeError('FAFT GPU: stride 20, 1 to 4096 intervals');
    for (let k = 0; k < omegas.length; k++) {
      if (!Number.isFinite(omegas[k])) throw new RangeError('FAFT GPU: finite frequencies only');
    }
    const atomBuffer = this.#upload(Float32Array.from(atoms));
    try {
      return await this.#dispatch(atomBuffer, intervals, stride, frequencyTable(omegas), new Uint32Array(omegas.length));
    } finally {
      atomBuffer.destroy();
    }
  }

  /**
   * An evaluator for SinWhisky.equalizeBatch: each round's values for every window at once, in one dispatch.
   * The windows' rotations go to the GPU once per batch (f32, as the WebAssembly fast path rounds them).
   */
  batchEvaluator() {
    let uploaded;
    const evaluate = async (requests, windows) => {
      if (uploaded?.windows !== windows) {
        uploaded?.buffer.destroy();
        uploaded = { windows, buffer: this.#upload(windows.atoms.length > 0 ? windows.atoms : new Float32Array(4)) };
      }
      const counts = requests.map((request) => request.omegas.length);
      const total = counts.reduce((a, b) => a + b, 0);
      const omegas = new Float64Array(total);
      const bases = new Uint32Array(total);
      let at = 0;
      for (const request of requests) {
        const offset = windows.offsets[request.window];
        if (offset < 0) throw new Error(`window ${request.window} has no prepared rotations`);
        omegas.set(request.omegas, at);
        bases.fill(offset, at, at + request.omegas.length);
        at += request.omegas.length;
      }
      const values = await this.#dispatch(uploaded.buffer, windows.intervals, windows.stride, frequencyTable(omegas), bases);
      at = 0;
      return counts.map((count) => {
        const out = Float64Array.from(values.subarray(2 * at, 2 * (at + count)));
        at += count;
        return out;
      });
    };
    return Object.assign(evaluate, { needsAtoms: true, destroy: () => { uploaded?.buffer.destroy(); uploaded = undefined; } });
  }

  #upload(data) {
    const buffer = this.#device.createBuffer({ size: Math.max(16, data.byteLength), usage: STORAGE | COPY_DST });
    this.#device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  async #dispatch(atomBuffer, intervals, stride, table, bases) {
    const device = this.#device;
    const count = bases.length;
    if (count === 0) return new Float32Array(0);
    const omegaBuffer = this.#upload(table);
    const baseBuffer = device.createBuffer({ size: Math.max(16, count * 4), usage: STORAGE | COPY_DST });
    device.queue.writeBuffer(baseBuffer, 0, bases);
    const valueBuffer = device.createBuffer({ size: Math.max(16, count * 8), usage: STORAGE | COPY_SRC });
    const readBuffer = device.createBuffer({ size: Math.max(16, count * 8), usage: MAP_READ | COPY_DST });
    const paramBuffer = device.createBuffer({ size: 16, usage: UNIFORM | COPY_DST });
    device.queue.writeBuffer(paramBuffer, 0, Uint32Array.of(intervals, count, stride, 0));
    const group = device.createBindGroup({
      layout: this.#pipeline.getBindGroupLayout(0),
      entries: [atomBuffer, omegaBuffer, valueBuffer, paramBuffer, baseBuffer].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.#pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    encoder.copyBufferToBuffer(valueBuffer, 0, readBuffer, 0, count * 8);
    device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(MAP_MODE_READ);
    const values = new Float32Array(readBuffer.getMappedRange().slice(0, count * 8));
    readBuffer.unmap();
    for (const buffer of [omegaBuffer, baseBuffer, valueBuffer, readBuffer, paramBuffer]) buffer.destroy();
    return values;
  }

  /**
   * Magnitudes on the grid lower + k·step (k < count), normalized as SinWhisky.faftScan: with `density` > 0
   * the Hann window (½F(ω) − ¼F(ω ∓ 2π/T), its neighbours `density` steps away on one extended
   * rectangular grid), else rectangular.
   */
  async scan(atoms, stride, lower, step, count, density = 0) {
    const intervals = atoms.length / stride;
    const total = count + 2 * density;
    const values = await this.rect(atoms, stride, Float64Array.from({ length: total }, (_, k) => lower + step * (k - density)));
    const norm = density > 0 ? 2 / intervals : 1 / intervals;
    return Float64Array.from({ length: count }, (_, j) => {
      let re = values[2 * (j + density)];
      let im = values[2 * (j + density) + 1];
      if (density > 0) {
        re = 0.5 * re - 0.25 * (values[2 * j] + values[2 * (j + 2 * density)]);
        im = 0.5 * im - 0.25 * (values[2 * j + 1] + values[2 * (j + 2 * density) + 1]);
      }
      return Math.hypot(re, im) * norm;
    });
  }

  destroy() {
    this.#device.destroy();
  }
}
