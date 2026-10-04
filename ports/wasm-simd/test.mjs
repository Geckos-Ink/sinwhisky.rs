// The port against sinwhisky.rs: ../fixtures/vectors.json is written by `cargo run --example port_vectors`.
// Scalar paths follow the crate's arithmetic (agreement to libm rounding); SIMD paths reorder it.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { before, describe, test } from 'node:test';
import { SinWhisky } from './index.mjs';

const vectors = JSON.parse(await readFile(new URL('../fixtures/vectors.json', import.meta.url), 'utf8'));
let sw;
before(async () => {
  sw = await SinWhisky.load();
});

const peak = (values) => values.reduce((m, v) => Math.max(m, Math.abs(v)), 1);

/** Largest |a − b| over the scale of b. */
function close(actual, expected, tolerance, label) {
  assert.equal(actual.length, expected.length, `${label}: length`);
  const scale = peak(expected);
  let worst = 0;
  for (let i = 0; i < expected.length; i++) worst = Math.max(worst, Math.abs(actual[i] - expected[i]) / scale);
  assert.ok(worst <= tolerance, `${label}: ${worst.toExponential(2)} > ${tolerance}`);
  return worst;
}

describe('SinWhisky resampler (ResampleParams::original)', () => {
  for (const c of vectors.cases) {
    test(c.name, () => {
      for (const { zoom, output } of c.resample) {
        close(sw.resample(c.samples, zoom), output, 1e-12, `${c.name} zoom ${zoom}`);
        close(sw.resample(c.samples, zoom, { simd: true }), output, 1e-11, `${c.name} zoom ${zoom} simd`);
        // Originals are copied exactly.
        const out = sw.resample(c.samples, zoom, { simd: true });
        c.samples.forEach((s, i) => assert.equal(out[i * (zoom + 1)], s));
      }
    });
  }

  test('the paper\'s page-8 example and the edge lengths', () => {
    const paper = vectors.cases.find((c) => c.name === 'paper');
    assert.deepEqual(Array.from(sw.resample(paper.samples, 0)), paper.samples);
    assert.deepEqual(Array.from(sw.resample([], 3)), []);
    assert.deepEqual(Array.from(sw.resample([2], 3)), [2]);
    // Two samples: no circle, the line.
    close(sw.resample([0, 1], 3), [0, 0.25, 0.5, 0.75, 1], 1e-15, 'two samples');
    assert.throws(() => sw.resample([0, NaN, 1], 1), RangeError);
    assert.throws(() => sw.resample([0, 1, 2], 256), RangeError);
  });
});

describe('AFT (AngularFrame, original analysis)', () => {
  for (const c of vectors.cases) {
    test(c.name, () => {
      const stats = sw.prepareAft(c.samples);
      assert.deepEqual(
        { circleHalves: stats.circleHalves, lineHalves: stats.lineHalves, rejectedCircleHalves: stats.rejectedCircleHalves },
        c.stats,
      );
      const positions = c.curve.map((_, k) => k * 0.25 + 0.1);
      close(positions.map((p) => sw.evaluate(p)), c.curve, 1e-12, `${c.name} curve`);
      close(sw.integral(c.omegas), c.aftIntegral, 1e-11, `${c.name} fourier`);
      close(sw.integral(c.omegas, { simd: true }), c.aftIntegral, 1e-11, `${c.name} fourier simd`);
    });
  }
});

describe('FAFT on the AFT curve (CircleCurve::from_angular)', () => {
  for (const c of vectors.cases) {
    test(c.name, () => {
      sw.prepareAft(c.samples);
      for (const simd of [false, true]) {
        close(sw.values(c.omegas, { simd }), c.faftRect, 1e-11, `${c.name} rect${simd ? ' simd' : ''}`);
        close(sw.values(c.omegas, { hann: true, simd }), c.faftHann, 1e-11, `${c.name} hann${simd ? ' simd' : ''}`);
      }
      if (!c.peaks) return;
      const found = sw.peaks();
      assert.equal(found.length, c.peaks.length / 4, `${c.name}: peak count`);
      const scale = peak(c.peaks.filter((_, i) => i % 4 === 1 || i % 4 === 2));
      found.forEach((p, k) => {
        assert.ok(Math.abs(p.omega - c.peaks[4 * k]) < 1e-9, `${c.name} peak ${k}: ω ${p.omega} vs ${c.peaks[4 * k]}`);
        assert.ok(Math.abs(p.re - c.peaks[4 * k + 1]) < 1e-8 * scale && Math.abs(p.im - c.peaks[4 * k + 2]) < 1e-8 * scale, `${c.name} peak ${k}: value`);
        assert.ok(Math.abs(p.curvature - c.peaks[4 * k + 3]) <= 1e-6 * Math.abs(c.peaks[4 * k + 3]) + 1e-9, `${c.name} peak ${k}: curvature`);
      });
    });
  }

  test('a tone between bins: its exact frequency and amplitude', () => {
    const omega = 0.4123456;
    sw.prepareAft(Array.from({ length: 512 }, (_, i) => Math.sin(omega * i + 0.3)));
    const strongest = sw.peaks().reduce((a, b) => (b.magnitude > a.magnitude ? b : a));
    assert.ok(Math.abs(strongest.omega - omega) < 1e-6, `${strongest.omega}`);
    assert.ok(Math.abs(strongest.magnitude - 0.5) < 0.01, `${strongest.magnitude}`);
  });

  test('scans: the SIMD path equals the scalar one', () => {
    const c = vectors.cases[0];
    sw.prepareAft(c.samples);
    const step = sw.resolution / 4;
    close(sw.scan(0, step, 1001, { density: 4, simd: true }), sw.scan(0, step, 1001, { density: 4, simd: false }), 1e-12, 'scan');
  });

  test('validation', () => {
    assert.throws(() => sw.prepareAft([]), RangeError);
    assert.throws(() => sw.prepareAft([1, Infinity]), RangeError);
    sw.prepareAft([1, 2, 3]);
    assert.throws(() => sw.values([NaN]), RangeError);
    assert.throws(() => sw.peaks({ lower: 1, upper: 1 }), RangeError);
  });
});
