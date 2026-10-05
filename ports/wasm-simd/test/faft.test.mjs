// Transposed from appx/dsp at ec84edcd331af8810be9df42cf5d16f3a348d6d5 (2026-10-05).
// FAFT (assembly/faft.ts): the Fourier transform of SinWhisky's circle curve, checked against an
// independent reference: the same curve built in double precision here and integrated by Gauss–Legendre
// quadrature (sinwhisky.rs checks its FAFT the same way), plus what detection needs from it: tones found
// between bins and above 16 kHz, close tones, noise, the SIMD path, and validation.

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { before, describe, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { SinWhisky } from '../index.mjs';
import { halves, evaluate, quadrature, signal } from './curve-reference.mjs';

let dsp;
before(async () => {
  dsp = await SinWhisky.instantiate(await readFile(fileURLToPath(new URL('../build/sinwhisky.wasm', import.meta.url))));
});

describe('FAFT', () => {
  test('the curve is SinWhisky\'s circles: through every sample, and the reference between them', () => {
    const samples = signal(300, [[0.6, 0.11, 0.3], [0.25, 0.93, 1.1], [0.1, 2.4, 0]], 0.01);
    assert.equal(dsp.prepareFaft(samples), 299);
    const curve = halves(samples);
    for (let k = 0; k < samples.length; k++) assert.ok(Math.abs(dsp.faftEvaluate(k) - samples[k]) < 1e-6, `sample ${k}`);
    for (let k = 0; k < 2000; k++) {
      const position = (k * 299) / 2000 + 0.0137;
      if (position > 299) continue;
      const expected = evaluate(samples, curve, position);
      assert.ok(Math.abs(dsp.faftEvaluate(position) - expected) < 1e-9 * Math.max(1, Math.abs(expected)), `at ${position}: ${dsp.faftEvaluate(position)} vs ${expected}`);
    }
    // Flat and straight stretches are lines, and a two-sample curve is the line between them.
    const flat = Float32Array.from([1, 1, 1, 2, 3, 4, 4]);
    dsp.prepareFaft(flat);
    assert.ok(Math.abs(dsp.faftEvaluate(0.5) - 1) < 1e-12 && Math.abs(dsp.faftEvaluate(3.5) - 2.5) < 1e-9);
    dsp.prepareFaft([2, 4]);
    assert.equal(dsp.faftEvaluate(0.25), 2.5);
    assert.ok(Number.isNaN(dsp.faftEvaluate(1.5)) && Number.isNaN(dsp.faftEvaluate(-0.1)));
  });

  test('values equal the quadrature of the curve at any real frequency, rectangular and Hann', () => {
    const samples = signal(257, [[0.7, 0.3137, 0.2], [0.2, 1.7, 0.9]], 0.02, 3);
    dsp.prepareFaft(samples);
    const curve = halves(samples);
    const rates = curve.flat().flatMap((half) => (half.kind === 'arc' ? [Math.abs(half.rate)] : []));
    // DC, tiny, between bins, at an arc's own rate (resonance), above π (the curve's own content), negative.
    const omegas = [0, 1e-9, 0.04, 0.3137, 0.3137 + 0.37 * (2 * Math.PI / 256), rates[5], rates[40], 1.7, 2.9, Math.PI, 4.1, -0.8];
    for (const hann of [false, true]) {
      const values = dsp.faftValues(omegas, { hann });
      const largest = Math.max(...omegas.map((omega) => Math.hypot(...quadrature(samples, curve, omega, hann))));
      omegas.forEach((omega, k) => {
        const [re, im] = quadrature(samples, curve, omega, hann);
        const error = Math.hypot(values[2 * k] - re, values[2 * k + 1] - im);
        assert.ok(error < 1e-10 * largest, `${hann ? 'Hann' : 'rect'} ω ${omega}: ${error}`);
      });
    }
  });

  test('arcs near resonance: the endpoint form and the moment form meet at the guard, both equal to quadrature', () => {
    const samples = signal(200, [[0.5, 0.05, 0], [0.3, 0.71, 1]], 0.05, 9);
    dsp.prepareFaft(samples);
    const curve = halves(samples);
    const rates = curve.flat().flatMap((half) => (half.kind === 'arc' ? [Math.abs(half.rate)] : []));
    // Offsets inside the guard (0.05: moment form), at it, and outside it (endpoint form), on both sides.
    const omegas = rates.filter((_, k) => k % 23 === 0).flatMap((rate) => [0.003, 0.0499, 0.0501, 0.08, 0.3, 0.6].flatMap((x) => [rate + x, rate - x]));
    for (const simd of [false, true]) {
      const values = Float64Array.from(dsp.faftValues(omegas, { simd }));
      const references = omegas.map((omega) => quadrature(samples, curve, omega, false));
      const largest = Math.max(...references.map(([re, im]) => Math.hypot(re, im)));
      omegas.forEach((omega, k) => {
        const [re, im] = references[k];
        const error = Math.hypot(values[2 * k] - re, values[2 * k + 1] - im);
        assert.ok(error < 1e-10 * largest, `${simd ? 'SIMD' : 'scalar'} ω ${omega}: ${error}`);
      });
    }
  });

  test('two frequencies per SIMD instruction give the same values, and a scan the same magnitudes; four f32 lanes within 3e-6', () => {
    const samples = signal(4096, [[0.5, 0.05, 0], [0.3, 0.71, 1], [0.1, 2.6, 2]], 0.05);
    const duration = dsp.prepareFaft(samples);
    const omegas = Array.from({ length: 333 }, (_, k) => 0.001 + k * 0.0093);
    for (const hann of [false, true]) {
      const scalar = Float64Array.from(dsp.faftValues(omegas, { hann }));
      const simd = Float64Array.from(dsp.faftValues(omegas, { hann, simd: true }));
      const largest = Math.max(...scalar.map(Math.abs));
      scalar.forEach((value, k) => assert.ok(Math.abs(value - simd[k]) < 1e-12 * largest, `${k}: ${value} vs ${simd[k]}`));
    }
    const density = 2;
    const step = (2 * Math.PI) / duration / density;
    const grid = Array.from({ length: 700 }, (_, k) => 0.2 + k * step);
    const values = Float64Array.from(dsp.faftValues(grid, { hann: true }));
    for (const simd of [false, true]) {
      const scan = dsp.faftScan(0.2, step, grid.length, density, simd);
      grid.forEach((_, k) => assert.ok(Math.abs(scan[k] - Math.hypot(values[2 * k], values[2 * k + 1])) < 1e-12, `${k}`));
    }
    // Four f32 lanes: within about 1e-6 of the strongest.
    const fast = Float64Array.from(dsp.faftScan(0.2, step, grid.length, density, 'fast'));
    const strongest = Math.max(...grid.map((_, k) => Math.hypot(values[2 * k], values[2 * k + 1])));
    grid.forEach((_, k) => assert.ok(Math.abs(fast[k] - Math.hypot(values[2 * k], values[2 * k + 1])) < 3e-6 * strongest, `fast ${k}`));
    const fastValues = Float64Array.from(dsp.faftValues(omegas, { hann: true, fast: true }));
    const exactValues = Float64Array.from(dsp.faftValues(omegas, { hann: true }));
    const largest = Math.max(...exactValues.map(Math.abs));
    exactValues.forEach((v, k) => assert.ok(Math.abs(v - fastValues[k]) < 3e-6 * largest, `fast value ${k}`));
  });

  test('a unit tone reads 0.5 at its frequency, wherever it falls between bins; the curve\'s response is known', () => {
    for (const omega of [0.3137, 0.3137 + Math.PI / 512, 1.2345, 2.52]) {
      const samples = signal(1024, [[1, omega, 0.4]]);
      dsp.prepareFaft(samples);
      const [peak] = dsp.faftPeaks({ relativeFloor: 0.5 });
      assert.ok(peak && Math.abs(peak.omega - omega) < 2e-6, `${omega}: ${peak?.omega}`);
      // The curve's fundamental: 0.5 at low frequencies, less near Nyquist, by its response alone.
      if (omega < 0.4) assert.ok(Math.abs(peak .magnitude - 0.5) < 2e-3, `${omega}: ${peak .magnitude}`);
      assert.ok(Math.abs(peak .amplitude - 1) < 2e-3, `${omega}: amplitude ${peak .amplitude}`);
      assert.ok(peak .curvature < 0);
    }
    // The response: flat at low frequencies, −0.45 dB at 1.23 rad/sample and about −4 dB at 2.52.
    assert.ok(Math.abs(dsp.faftToneGain(0.1) - 1) < 1e-3);
    assert.ok(Math.abs(20 * Math.log10(dsp.faftToneGain(1.2345)) + 0.45) < 0.02);
    assert.ok(Math.abs(20 * Math.log10(dsp.faftToneGain(2.52)) + 3.97) < 0.05);
    // Measuring it leaves the prepared curve as it was.
    const samples = signal(100, [[1, 0.2, 0]]);
    dsp.prepareFaft(samples);
    dsp.faftToneGain(1);
    assert.ok(Math.abs(dsp.faftEvaluate(50) - samples[50]) < 1e-6 && dsp.faftDuration === 99);
  });

  test('peaks: tones between FFT bins, a near-ultrasonic pilot, and two tones a few bins apart, in noise', () => {
    // 48 kHz: 1 kHz, 3.217 kHz and a 19.25 kHz pilot over 2048 samples (bins 23.4 Hz apart), noise 40 dB down.
    const hz = (f) => (2 * Math.PI * f) / 48_000;
    const tones = [[0.5, hz(1_000), 0.1], [0.3, hz(3_217), 1.3], [0.2, hz(19_250), 2.2]];
    dsp.prepareFaft(signal(2048, tones, 0.005, 11));
    const peaks = dsp.faftPeaks({ relativeFloor: 0.05 });
    for (const [amplitude, omega] of tones) {
      const found = peaks.find((peak) => Math.abs(peak.omega - omega) < 1e-3);
      assert.ok(found, `ω ${omega}`);
      // Within a twentieth of a bin, while an FFT's bin is ±half a bin (a 19.25 kHz pilot at 0.5 Hz).
      assert.ok(Math.abs(found.omega - omega) < (0.05 * 2 * Math.PI) / 2047, `ω ${omega}: off by ${(found.omega - omega) / (2 * Math.PI) * 48_000} Hz`);
      assert.ok(Math.abs(found.amplitude - amplitude) < 0.02 * amplitude, `amplitude ${found.amplitude} vs ${amplitude}`);
    }
    // Two tones 3 bins apart stay two peaks.
    dsp.prepareFaft(signal(1024, [[0.5, 0.6, 0], [0.4, 0.6 + (3 * 2 * Math.PI) / 1023, 0.7]]));
    assert.equal(dsp.faftPeaks({ lower: 0.5, upper: 0.7, relativeFloor: 0.3 }).length, 2);
    // A tone's Doppler: 300 ppm (0.1 m/s) on a 19 kHz pilot over 85 ms is 5.7 Hz, half a bin.
    const pilot = hz(19_000);
    dsp.prepareFaft(signal(4096, [[0.3, pilot * 1.0003, 0]], 0.003, 5));
    const [shifted] = dsp.faftPeaks({ lower: pilot * 0.99, upper: pilot * 1.01, relativeFloor: 0.5 });
    assert.ok(shifted && Math.abs(shifted.omega / pilot - 1.0003) < 3e-6, `rate ${shifted .omega / pilot}`);
  });

  test('validation: sizes, non-finite input and frequencies, scans and peak options', () => {
    assert.throws(() => dsp.prepareFaft([1]), RangeError);
    assert.throws(() => dsp.prepareFaft(new Float32Array(dsp.maxFaftInput + 1)), RangeError);
    assert.throws(() => dsp.prepareFaft([1, Number.NaN, 2]), RangeError);
    dsp.prepareFaft(signal(64, [[1, 0.5, 0]]));
    assert.throws(() => dsp.faftValues([0.1, Number.POSITIVE_INFINITY]), RangeError);
    assert.throws(() => dsp.faftValues(new Float64Array(dsp.maxFaftQueries + 1)), RangeError);
    assert.throws(() => dsp.faftScan(0, 0.01, dsp.maxFaftQueries, 2), RangeError);
    assert.throws(() => dsp.faftScan(0, -1, 10), RangeError);
    assert.throws(() => dsp.faftPeaks({ lower: 1, upper: 0.5 }), RangeError);
    assert.throws(() => dsp.faftPeaks({ density: 0 }), RangeError);
    assert.throws(() => dsp.faftPeaks({ relativeFloor: 2 }), RangeError);
    assert.equal(dsp.faftValues([]).length, 0);
  });
});
