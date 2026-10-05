// Transposed from appx/dsp at ec84edcd331af8810be9df42cf5d16f3a348d6d5 (2026-10-05).
// The jointly fitted tone model (assembly/tones.ts; sinwhisky.rs `fit_banded`, `fit_refined`): the window
// kernel, the two-way banded fit against an independent dense least-squares fit done by quadrature of the
// reference curve, the residual and its bound, refinement from frequencies off the tones, banded against
// dense on a long harmonic chain, DC, and validation.

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { before, describe, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { SinWhisky } from '../index.mjs';
import { GAUSS, evaluate, halves, signal } from './curve-reference.mjs';

let dsp;
before(async () => {
  dsp = await SinWhisky.instantiate(await readFile(fileURLToPath(new URL('../build/sinwhisky.wasm', import.meta.url))));
});

const hannWeight = (t, T) => 0.5 - 0.5 * Math.cos((2 * Math.PI * t) / T);

/**
 * The dense weighted least-squares fit of α_p cos(ω_p u) + β_p sin(ω_p u), u = t − T/2, to the reference
 * curve, every integral by Gauss–Legendre quadrature: [α…, β…].
 */
function referenceFit(samples, omegas) {
  const curve = halves(samples);
  const T = curve.length;
  const m = 2 * omegas.length;
  const gram = Array.from({ length: m }, () => new Float64Array(m));
  const rhs = new Float64Array(m);
  const atom = (k, u) => (k < omegas.length ? Math.cos(omegas[k] * u) : Math.sin(omegas[k - omegas.length] * u));
  for (let n = 0; n < T; n++) {
    for (const [x, weight] of GAUSS) {
      const t = n + x;
      const w = hannWeight(t, T) * weight;
      const y = evaluate(samples, curve, t);
      const values = Array.from({ length: m }, (_, k) => atom(k, t - T / 2));
      for (let i = 0; i < m; i++) {
        rhs[i] += w * y * values[i];
        for (let j = 0; j < m; j++) gram[i] [j] += w * values[i] * values[j];
      }
    }
  }
  // Gaussian elimination with partial pivoting.
  const a = gram.map((row, i) => [...row, rhs[i]]);
  for (let c = 0; c < m; c++) {
    let pivot = c;
    for (let r = c + 1; r < m; r++) if (Math.abs(a[r] [c]) > Math.abs(a[pivot] [c])) pivot = r;
    [a[c], a[pivot]] = [a[pivot], a[c]];
    for (let r = c + 1; r < m; r++) {
      const f = a[r] [c] / a[c] [c];
      for (let k = c; k <= m; k++) a[r] [k] -= f * a[c] [k];
    }
  }
  const x = new Array (m).fill(0);
  for (let r = m - 1; r >= 0; r--) {
    let v = a[r] [m];
    for (let k = r + 1; k < m; k++) v -= a[r] [k] * x[k];
    x[r] = v / a[r] [r];
  }
  return x;
}

describe('joint tone fit', () => {
  test('the window kernel: 1 at 0, the Hann main lobe ends at 2 bins, derivatives match differences', () => {
    dsp.prepareFaft(signal(1025, [[1, 0.3, 0]]));
    const bin = (2 * Math.PI) / 1024;
    assert.ok(Math.abs(dsp.faftKernel(0) - 1) < 1e-15 && Math.abs(dsp.faftKernel(0, { hann: false }) - 1) < 1e-15);
    assert.ok(Math.abs(dsp.faftKernel(2 * bin)) < 1e-12, 'Hann first null');
    assert.ok(Math.abs(dsp.faftKernel(bin, { hann: false })) < 1e-12, 'rectangular first null');
    assert.ok(Math.abs(dsp.faftKernel(bin) - 0.5) < 1e-12, 'Hann at one bin');
    for (const nu of [0, 0.37 * bin, 1.9 * bin, 5.2 * bin]) {
      for (const hann of [true, false]) {
        const h = 1e-6 * bin;
        const d1 = (dsp.faftKernel(nu + h, { hann }) - dsp.faftKernel(nu - h, { hann })) / (2 * h);
        const d2 = (dsp.faftKernel(nu + h, { hann, order: 1 }) - dsp.faftKernel(nu - h, { hann, order: 1 })) / (2 * h);
        assert.ok(Math.abs(dsp.faftKernel(nu, { hann, order: 1 }) - d1) < 1e-6 * 1024, `K′ at ${nu}`);
        assert.ok(Math.abs(dsp.faftKernel(nu, { hann, order: 2 }) - d2) < 1e-5 * 1024 * 1024, `K″ at ${nu}`);
      }
    }
  });

  test('centred derivatives: SIMD pairs equal the scalar path, and both equal differences of the values', () => {
    const T = 2048;
    dsp.prepareFaft(signal(T + 1, [[0.5, 0.05, 0], [0.3, 0.71, 1], [0.1, 2.6, 2]], 0.05));
    const rates = [0.002, 0.05, 0.0507, 0.3, 0.7101, 1.2, 2.6, 3.1];
    const omegas = rates.flatMap((w) => [w, w + 0.013, w + 0.0371]);
    for (const hann of [true, false]) {
      const scalar = Float64Array.from(dsp.faftDerivatives(omegas, { hann, simd: false }));
      const simd = Float64Array.from(dsp.faftDerivatives(omegas, { hann, simd: true }));
      for (let order = 0; order < 3; order++) {
        let largest = 0;
        for (let k = 0; k < omegas.length; k++) largest = Math.max(largest, Math.hypot(scalar[6 * k + 2 * order], scalar[6 * k + 2 * order + 1]));
        scalar.forEach((v, i) => {
          if (Math.floor((i % 6) / 2) === order) assert.ok(Math.abs(v - simd[i]) < 1e-12 * largest, `order ${order} at ${i}: ${v} vs ${simd[i]}`);
        });
      }
      // F′ against central differences of F.
      const h = 1e-5;
      const plus = Float64Array.from(dsp.faftDerivatives(omegas.map((w) => w + h), { hann }));
      const minus = Float64Array.from(dsp.faftDerivatives(omegas.map((w) => w - h), { hann }));
      for (let k = 0; k < omegas.length; k++) {
        for (const part of [0, 1]) {
          const d1 = (plus[6 * k + part] - minus[6 * k + part]) / (2 * h);
          const d2 = (plus[6 * k + 2 + part] - minus[6 * k + 2 + part]) / (2 * h);
          assert.ok(Math.abs(d1 - scalar[6 * k + 2 + part]) < 1e-6 * T, `F′ ${omegas[k]}`);
          assert.ok(Math.abs(d2 - scalar[6 * k + 4 + part]) < 1e-6 * T * T, `F″ ${omegas[k]}`);
        }
      }
    }
  });

  test('amplitudes equal a dense least-squares fit by quadrature, close tones and sidelobes included', () => {
    // Two tones 1.4 bins apart (inside one main lobe) and others, in noise, at frequencies the fit is given.
    const T = 512;
    const bin = (2 * Math.PI) / T;
    const omegas = [0.21, 0.21 + 1.4 * bin, 0.55, 1.13, 2.2];
    const samples = signal(T + 1, [[0.7, 0.21, 0.4], [0.35, 0.21 + 1.4 * bin, 1.9], [0.2, 0.55, 0], [0.12, 1.13, 2.5], [0.05, 2.2, 1]], 0.02, 5);
    dsp.prepareFaft(samples);
    const reference = referenceFit(samples, omegas);
    for (const reachBins of [16, Infinity]) {
      for (const simd of [false, true]) {
        const fit = dsp.faftFit(omegas, { reachBins, simd });
        assert.equal(fit.components.length, omegas.length);
        // Dense: the same solution to rounding. A 16-bin reach drops the Gram entries of tones 27 bins and
        // more apart (Hann sidelobes about 1e-5 there), which moves the amplitudes by about that much.
        const tolerance = (reachBins === Infinity ? 1e-9 : 1e-5) * Math.max(...reference.map(Math.abs));
        assert.equal(fit.truncated, reachBins !== Infinity);
        fit.components.forEach((c, p) => {
          const alpha = c.amplitude * Math.cos(c.phase);
          const beta = -c.amplitude * Math.sin(c.phase);
          assert.ok(Math.abs(alpha - reference[p]) < tolerance, `α ${p}: ${alpha} vs ${reference[p]}`);
          assert.ok(Math.abs(beta - reference[p + omegas.length]) < tolerance, `β ${p}: ${beta} vs ${reference[p + omegas.length]}`);
        });
      }
    }
    // The tones' own heights, not the maxima: the close pair is resolved within 2 % although its peaks merge.
    const fit = dsp.faftFit(omegas);
    assert.ok(Math.abs(fit.components[0] .amplitude - 0.7) < 0.014 && Math.abs(fit.components[1] .amplitude - 0.35) < 0.007);
    // Significance: about each tone's amplitude over two when it is alone.
    assert.ok(Math.abs(fit.components[2] .significance - 0.1) < 0.01, `${fit.components[2] .significance}`);
  });

  test('the residual bounds the residual spectrum at every real frequency; the energy is the quadrature\'s', () => {
    const T = 600;
    const samples = signal(T + 1, [[0.8, 0.3, 0.1], [0.3, 0.9, 1.2]], 0.05, 13);
    dsp.prepareFaft(samples);
    const fit = dsp.faftFit([0.3, 0.9], { energy: true });
    const curve = halves(samples);
    let energy = 0;
    for (let n = 0; n < T; n++) for (const [x, weight] of GAUSS) energy += weight * evaluate(samples, curve, n + x) ** 2 * hannWeight(n + x, T);
    assert.ok(Math.abs(fit.energy - energy) < 1e-10 * energy, `${fit.energy} vs ${energy}`);
    assert.ok(fit.bound > 0 && fit.residualEnergy < 0.05 * energy);
    const grid = Array.from({ length: 900 }, (_, k) => k * 0.0037);
    const values = Float64Array.from(dsp.faftValues(grid, { hann: true }));
    let largest = 0;
    grid.forEach((omega, k) => {
      // Centred residual: a(ω) − Σ c k(ω − ω_p) + c̄ k(ω + ω_p).
      const turn = omega * T / 2;
      let re = values[2 * k] * Math.cos(turn) - values[2 * k + 1] * Math.sin(turn);
      let im = values[2 * k] * Math.sin(turn) + values[2 * k + 1] * Math.cos(turn);
      for (const c of fit.components) {
        const cre = (c.amplitude / 2) * Math.cos(c.phase);
        const cim = (c.amplitude / 2) * Math.sin(c.phase);
        const below = dsp.faftKernel(omega - c.omega);
        const above = dsp.faftKernel(omega + c.omega);
        re -= cre * (below + above);
        im -= cim * (below - above);
      }
      largest = Math.max(largest, Math.hypot(re, im));
    });
    assert.ok(largest <= fit.bound, `${largest} > ${fit.bound}`);
    assert.ok(largest > 0.1 * fit.bound, 'the bound is not vacuous');
  });

  test('refinement moves frequencies seeded off the tones onto them', () => {
    const T = 1024;
    const bin = (2 * Math.PI) / T;
    const truth = [0.123456, 0.4, 0.4 + 2.3 * bin, 1.7];
    dsp.prepareFaft(signal(T + 1, [[0.5, truth[0], 0.3], [0.4, truth[1], 1], [0.3, truth[2], 2], [0.2, truth[3], 0.5]], 0.002, 3));
    const seeds = truth.map((w, k) => w + (k % 2 === 0 ? 0.2 : -0.15) * bin);
    for (const simd of [false, true]) {
      const fit = dsp.faftFit(seeds, { refine: 12, tolerance: 1e-7 * bin, simd });
      assert.equal(fit.components.length, truth.length);
      fit.components.forEach((c, k) => assert.ok(Math.abs(c.omega - truth[k]) < 2e-3 * bin, `${k}: ${(c.omega - truth[k]) / bin} bins`));
    }
    // Amplitudes are the curve's: near Nyquist through its known response (dsp.faftToneGain).
    const fit = dsp.faftFit(seeds, { refine: 12, tolerance: 1e-7 * bin });
    const gains = truth.map((w) => dsp.faftToneGain(w));
    [0.5, 0.4, 0.3, 0.2].forEach((a, k) => assert.ok(Math.abs(fit.components[k] .amplitude / gains[k] - a) < 0.01 * a, `amplitude ${k}: ${fit.components[k] .amplitude / gains[k] }`));
  });

  test('banded against dense on twenty harmonics of 55 Hz, 1.28 bins apart (PAPER_EQ_COMPRESSION §2.4)', () => {
    const T = 1024;
    const w0 = (2 * Math.PI * 55) / 44_100;
    let state = 3;
    const random = () => ((state = (state * 1_103_515_245 + 12_345) % 2_147_483_648) / 2_147_483_648);
    const tones = Array.from({ length: 20 }, (_, k) => [0.3 / (k + 1), (k + 1) * w0, 2 * Math.PI * random()]);
    dsp.prepareFaft(signal(T + 1, tones));
    const omegas = tones.map((t) => t[1]);
    const dense = dsp.faftFit(omegas, { reachBins: Infinity });
    assert.equal(dense.truncated, false);
    const rms = (fit) => Math.sqrt(fit.components.reduce((s, c, k) => s + (c.amplitude - dense.components[k] .amplitude) ** 2, 0) / dense.components.reduce((s, c) => s + c.amplitude ** 2, 0));
    const banded4 = dsp.faftFit(omegas, { reachBins: 4 });
    const banded16 = dsp.faftFit(omegas, { reachBins: 16 });
    assert.ok(banded4.truncated && banded16.truncated);
    assert.ok(rms(banded16) < rms(banded4) && rms(banded16) < 1e-3, `${rms(banded4)} ${rms(banded16)}`);
    // The dense fit recovers the harmonics' amplitudes.
    dense.components.forEach((c, k) => assert.ok(Math.abs(c.amplitude - tones[k] [0]) < 2e-3 * 0.3, `${k}`));
  });

  test('DC is a cosine at 0: its sine atom is dropped as degenerate', () => {
    dsp.prepareFaft(Float32Array.from({ length: 513 }, (_, k) => 0.25 + 0.5 * Math.cos(0.4 * k)));
    const fit = dsp.faftFit([0, 0.4]);
    assert.equal(fit.degenerate, 1);
    assert.ok(Math.abs(fit.components[0] .amplitude - 0.25) < 2e-3 && Math.abs(fit.components[1] .amplitude - 0.5) < 2e-3);
  });

  test('validation: repeated, negative and non-finite frequencies, too many tones, no window', () => {
    dsp.prepareFaft(signal(100, [[1, 0.5, 0]]));
    assert.throws(() => dsp.faftFit([0.1, 0.1]), RangeError);
    assert.throws(() => dsp.faftFit([-0.1]), RangeError);
    assert.throws(() => dsp.faftFit([Number.NaN]), RangeError);
    assert.throws(() => dsp.faftFit(new Float64Array(dsp.maxFaftTones + 1).map((_, k) => k * 1e-3)), RangeError);
    assert.throws(() => dsp.faftFit([0.5], { reachBins: 0 }), RangeError);
    assert.equal(dsp.faftFit([]).components.length, 0);
    assert.throws(() => dsp.faftKernel(Number.NaN), RangeError);
  });
});
