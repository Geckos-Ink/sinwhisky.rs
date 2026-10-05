// Endpoint FAFT and batched EQ from appx, against the same WebAssembly curve.
import { FaftGpu } from './index.mjs';

function deviation(actual, expected) {
  if (actual.length !== expected.length) throw new Error('length mismatch');
  let scale = 0;
  let error = 0;
  for (let k = 0; k < actual.length; k++) {
    if (!Number.isFinite(actual[k])) throw new Error(`non-finite result at ${k}`);
    scale = Math.max(scale, Math.abs(expected[k]));
    error = Math.max(error, Math.abs(actual[k] - expected[k]));
  }
  return error / Math.max(scale, 1e-30);
}

export async function checkUpdatedPorts(sw) {
  const gpu = await FaftGpu.create();
  if (!gpu) throw new Error('no WebGPU adapter for endpoint FAFT');
  const report = { rect: {}, scan: {}, equalizer: {} };
  try {
    for (const [name, samples] of [
      ['tones', Float32Array.from({ length: 1025 }, (_, k) => 0.6 * Math.sin(0.1237 * k) + 0.2 * Math.sin(0.83 * k + 1))],
      ['flat-arcs', Float32Array.from({ length: 257 }, (_, k) => 1 + 1e-4 * Math.cos(0.01 * k))],
      ['line', Float32Array.of(0.3, -0.7)],
    ]) {
      sw.prepareFaft(samples);
      const atoms = sw.faftAtoms.slice();
      // Query both sides of resonance and of the GPU's moment/endpoint guard.
      const rate = atoms[4];
      const omegas = [0, 1e-9, 0.1237, 0.83, Math.PI, -0.83, 3 * Math.PI,
        rate, -rate, rate - 0.250001, rate - 0.249999, rate + 0.249999, rate + 0.250001];
      const reference = Float64Array.from(sw.faftValues(omegas), v => v * sw.faftDuration);
      report.rect[name] = deviation(await gpu.rect(atoms, sw.faftAtomStride, omegas), reference);
      const step = 2 * Math.PI / sw.faftDuration / 2;
      const count = Math.floor(Math.PI / step) + 1;
      report.scan[name] = deviation(await gpu.scan(atoms, sw.faftAtomStride, 0, step, count, 2),
        sw.faftScan(0, step, count, 2));
    }
    report.emptyQueries = (await gpu.rect(sw.faftAtoms, sw.faftAtomStride, [])).length === 0;
    let rejected = 0;
    for (const [stride, omegas] of [[12, [0.1]], [20, [NaN]]]) {
      try {
        await gpu.rect(sw.faftAtoms, stride, omegas);
      } catch (error) {
        if (!(error instanceof RangeError)) throw error;
        rejected++;
      }
    }
    report.invalidQueriesRejected = rejected === 2;
    const samples = Float32Array.from({ length: 20000 }, (_, k) =>
      0.4 * Math.sin(0.031 * k) + 0.2 * Math.sin(0.2 * k + 1) + 0.15 * Math.sin(0.203 * k + 2) + 0.1 * Math.sin(0.9 * k + 3));
    const options = { edges: [0.05, 0.15, 0.6], refine: 2 };
    const gains = [1.5, 0.5, 2, 0.25];
    const stream = sw.equalize(samples, options, gains);
    const wasm = await sw.equalizeBatch(samples, options, gains);
    const evaluator = gpu.batchEvaluator();
    let batch;
    try {
      batch = await sw.equalizeBatch(samples, options, gains, evaluator);
    } finally {
      evaluator.destroy();
    }
    report.equalizer.wasmExact = stream.output.every((v, k) => v === wasm.output[k]);
    report.equalizer.outputError = deviation(batch.output, stream.output);
    report.equalizer.bandError = Math.max(...batch.bands.map((b, k) => deviation(b, stream.bands[k])));
    let sumError = 0;
    for (let k = 0; k < samples.length; k++) {
      sumError = Math.max(sumError, Math.abs(batch.bands.reduce((s, b) => s + b[k], 0) - samples[k]));
    }
    report.equalizer.sumError = sumError;
    report.equalizer.wasmStats = wasm.stats;
    report.equalizer.gpuStats = batch.stats;
    // The codec remains integer WebAssembly, including in the browser host.
    const encoded = sw.encodeSwl2([Int32Array.from(samples, v => Math.round(v * 10000))], { tolerance: 16, effort: 'fast' });
    report.codecExact = sw.decodeSwl2(encoded.bytes)[0].every((v, k) => v === encoded.decoded[0][k]);
    report.pass = Object.values(report.rect).every(v => v < 2e-4)
      && Object.values(report.scan).every(v => v < 2e-4)
      && report.equalizer.wasmExact && report.equalizer.outputError < 1e-2
      && report.equalizer.bandError < 1e-2 && sumError < 2e-7 && report.codecExact
      && report.emptyQueries && report.invalidQueriesRejected;
    return report;
  } finally {
    gpu.destroy();
  }
}
