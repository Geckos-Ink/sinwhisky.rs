// The WebGPU kernels against the WebAssembly port (f64) and sinwhisky.rs (../fixtures/vectors.json), with
// timings. Open test.html from a local server rooted at ports/ (node serve.mjs); the verdict is in the
// page title and window.sinwhiskyGpuReport.
import { SinWhisky } from '../wasm-simd/index.mjs';
import { SinWhiskyGpu } from './index.mjs';
import { checkUpdatedPorts } from './updated-test.mjs';

const status = document.getElementById('status');
const out = document.getElementById('result');

function worst(actual, expected) {
  const scale = Array.prototype.reduce.call(expected, (m, v) => Math.max(m, Math.abs(v)), 1);
  return Array.prototype.reduce.call(expected, (m, v, i) => Math.max(m, Math.abs(actual[i] - v) / scale), 0);
}

async function median(run, repeats = 5) {
  await run();
  const times = [];
  for (let r = 0; r < repeats; r++) {
    const start = performance.now();
    await run();
    times.push(performance.now() - start);
  }
  return times.sort((a, b) => a - b)[repeats >> 1];
}

async function main() {
  const sw = await SinWhisky.instantiate(await (await fetch('../wasm-simd/build/sinwhisky.wasm')).arrayBuffer());
  const vectors = await (await fetch('../fixtures/vectors.json')).json();
  const gpu = await SinWhiskyGpu.create();
  if (!gpu) throw new Error('no WebGPU adapter');
  const report = { adapter: null, resample: {}, rect: {}, scan: {}, timings: {} };
  const adapter = await navigator.gpu.requestAdapter();
  report.adapter = adapter.info ? `${adapter.info.vendor} ${adapter.info.architecture}`.trim() : 'unknown';
  let largest = { resample: 0, rect: 0, scan: 0 };
  for (const c of vectors.cases) {
    for (const { zoom, output } of c.resample) {
      const got = await gpu.resample(c.samples, sw.models(c.samples), zoom, sw.modelStride);
      const error = worst(got, output);
      report.resample[`${c.name} zoom ${zoom}`] = error;
      largest.resample = Math.max(largest.resample, error);
    }
    sw.prepareAft(c.samples);
    const atoms = sw.atoms();
    const n = c.samples.length;
    const rect = await gpu.rect(atoms, sw.atomStride, c.omegas);
    const normalized = Array.from(rect, (v) => v / n);
    report.rect[c.name] = worst(normalized, c.faftRect);
    largest.rect = Math.max(largest.rect, report.rect[c.name]);
    const step = sw.resolution / 4;
    const count = Math.floor(Math.PI / step) + 1;
    const scan = await gpu.scan(atoms, sw.atomStride, 0, step, count, 4);
    report.scan[c.name] = worst(scan, sw.scan(0, step, count, { density: 4 }));
    largest.scan = Math.max(largest.scan, report.scan[c.name]);
  }
  // Timings: a 4096-sample signal, zoom 15, and a 4095-frequency rectangular scan.
  const signal = Array.from({ length: 4096 }, (_, i) => Math.sin(0.0731 * i) + 0.4 * Math.sin(1.37 * i + 1));
  const models = sw.models(signal);
  report.timings.resampleZoom15 = {
    wasmScalarMs: await median(() => sw.resample(signal, 15)),
    wasmSimdMs: await median(() => sw.resample(signal, 15, { simd: true })),
    gpuMs: await median(() => gpu.resample(signal, models, 15)),
  };
  sw.prepareAft(signal);
  const atoms = sw.atoms();
  const step = Math.PI / 4095;
  report.timings.scan4095 = {
    wasmScalarMs: await median(() => sw.scan(0, step, 4095, { simd: false }), 3),
    wasmSimdMs: await median(() => sw.scan(0, step, 4095, { simd: true }), 3),
    gpuMs: await median(() => gpu.scan(atoms, sw.atomStride, 0, step, 4095)),
  };
  report.scanParity4095 = worst(await gpu.scan(atoms, sw.atomStride, 0, step, 4095), sw.scan(0, step, 4095));
  report.largest = largest;
  // f32 against f64 (measured on Apple Metal 3: 2.4e-7, 7.8e-7, 1.1e-6, 5.3e-7 of each output's peak).
  report.pass = largest.resample < 1e-6 && largest.rect < 1e-5 && largest.scan < 1e-5 && report.scanParity4095 < 1e-5;
  gpu.destroy();
  report.updated = await checkUpdatedPorts(sw);
  report.pass &&= report.updated.pass;
  return report;
}

main().then((report) => {
  window.sinwhiskyGpuReport = report;
  document.title = report.pass ? 'pass' : 'fail';
  status.textContent = report.pass ? 'WebGPU matches the reference' : 'WebGPU differs from the reference';
  status.className = report.pass ? 'pass' : 'fail';
  out.textContent = JSON.stringify(report, null, 2);
}, (error) => {
  window.sinwhiskyGpuReport = { error: String(error) };
  document.title = 'error';
  status.textContent = 'Error';
  status.className = 'fail';
  out.textContent = String(error?.stack ?? error);
});
