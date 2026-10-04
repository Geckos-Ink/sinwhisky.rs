// Agreement with sinwhisky.rs (../fixtures/vectors.json) and scalar vs SIMD timings, in Node.
// Usage: node bench.mjs
import { readFile } from 'node:fs/promises';
import { SinWhisky } from './index.mjs';

const sw = await SinWhisky.load();
const vectors = JSON.parse(await readFile(new URL('../fixtures/vectors.json', import.meta.url), 'utf8'));

function worst(actual, expected) {
  const scale = expected.reduce((m, v) => Math.max(m, Math.abs(v)), 1);
  return expected.reduce((m, v, i) => Math.max(m, Math.abs(actual[i] - v) / scale), 0);
}

const agreement = { resample: 0, resampleSimd: 0, curve: 0, fourier: 0, fourierSimd: 0, hann: 0, hannSimd: 0, peakOmega: 0 };
for (const c of vectors.cases) {
  for (const { zoom, output } of c.resample) {
    agreement.resample = Math.max(agreement.resample, worst(sw.resample(c.samples, zoom), output));
    agreement.resampleSimd = Math.max(agreement.resampleSimd, worst(sw.resample(c.samples, zoom, { simd: true }), output));
  }
  sw.prepareAft(c.samples);
  agreement.curve = Math.max(agreement.curve, worst(c.curve.map((_, k) => sw.evaluate(k * 0.25 + 0.1)), c.curve));
  agreement.fourier = Math.max(agreement.fourier, worst(sw.integral(c.omegas), c.aftIntegral));
  agreement.fourierSimd = Math.max(agreement.fourierSimd, worst(sw.integral(c.omegas, { simd: true }), c.aftIntegral));
  agreement.hann = Math.max(agreement.hann, worst(sw.values(c.omegas, { hann: true }), c.faftHann));
  agreement.hannSimd = Math.max(agreement.hannSimd, worst(sw.values(c.omegas, { hann: true, simd: true }), c.faftHann));
  if (c.peaks) {
    sw.peaks().forEach((p, k) => (agreement.peakOmega = Math.max(agreement.peakOmega, Math.abs(p.omega - c.peaks[4 * k]))));
  }
}
console.log('largest deviation from sinwhisky.rs (relative to each output\'s peak; peak ω in rad/sample):');
for (const [name, value] of Object.entries(agreement)) console.log(`  ${name.padEnd(13)} ${value.toExponential(1)}`);

function time(run, repeats = 7) {
  run();
  const samples = [];
  for (let r = 0; r < repeats; r++) {
    const start = performance.now();
    run();
    samples.push(performance.now() - start);
  }
  return samples.sort((a, b) => a - b)[repeats >> 1];
}

const n = 4096;
const signal = Array.from({ length: n }, (_, i) => Math.sin(0.0731 * i) + 0.4 * Math.sin(1.37 * i + 1));
console.log(`\nmedian of 7 runs, ${n} samples:`);
for (const zoom of [3, 7, 15]) {
  const scalar = time(() => sw.resample(signal, zoom));
  const simd = time(() => sw.resample(signal, zoom, { simd: true }));
  console.log(`  resample zoom ${String(zoom).padEnd(2)}  scalar ${scalar.toFixed(2)} ms  simd ${simd.toFixed(2)} ms  (${(scalar / simd).toFixed(2)}x)`);
}
sw.prepareAft(signal);
const count = 4095;
const step = Math.PI / count;
const scalar = time(() => sw.scan(0, step, count, { simd: false }), 3);
const simd = time(() => sw.scan(0, step, count, { simd: true }), 3);
console.log(`  FAFT scan ${count} frequencies  scalar ${scalar.toFixed(0)} ms  simd ${simd.toFixed(0)} ms  (${(scalar / simd).toFixed(2)}x)`);
const peaks = time(() => sw.peaks(), 3);
console.log(`  FAFT peaks (Hann, density 2, 0..pi)  ${peaks.toFixed(0)} ms, ${sw.peaks().length} peaks`);
