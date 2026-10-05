// Transposed from appx/dsp at ec84edcd331af8810be9df42cf5d16f3a348d6d5 (2026-10-05).
// The component equalizer (assembly/eq.ts; sinwhisky.rs `bands`): completeness at a fixed latency,
// independence of the block size and of the other channel, tones assigned whole near band edges, gains,
// the residual filters' zero-phase Butterworth response, and validation.

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { before, describe, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { SinWhisky } from '../index.mjs';
import { signal } from './curve-reference.mjs';

let dsp;
before(async () => {
  dsp = await SinWhisky.instantiate(await readFile(fileURLToPath(new URL('../build/sinwhisky.wasm', import.meta.url))));
});

const bin = (2 * Math.PI) / 2048;
const rms = (a, from, to) => {
  let sum = 0;
  for (let k = from; k < to; k++) sum += a[k] ** 2;
  return Math.sqrt(sum / (to - from));
};

describe('component equalizer', () => {
  test('unit gains return the input exactly, latency samples late; the bands add up to it', () => {
    const x = signal(30_000, [[0.4, 0.031, 0], [0.2, 0.2, 1], [0.1, 0.9, 2]], 0.05);
    for (const lookahead of [0, 1, 3]) {
      const { output, bands, latency } = dsp.equalize(x, { edges: [0.05, 0.15, 0.6], lookahead });
      assert.equal(latency, (lookahead + 2) * 1024 + 2);
      for (let k = 0; k < x.length; k++) {
        assert.equal(output[k], x[k], `lookahead ${lookahead}, sample ${k}`);
        const sum = bands.reduce((s, band) => s + band[k], 0);
        assert.ok(Math.abs(sum - x[k]) < 1e-7, `bands at ${k}: ${sum} vs ${x[k]}`);
      }
    }
  });

  test('the batch path (every window\'s fit round by round) is the stream, bit for bit', async () => {
    // Tones near band edges, close pairs, a DC offset and noise: every step of the fit (prune, refine,
    // the DC snap) runs somewhere.
    const x = signal(40_000, [[0.4, 0.031, 0], [0.2, 0.2, 1], [0.15, 0.203, 2], [0.1, 0.9, 3], [0.05, 0.0005, 0]], 0.05, 7);
    for (const options of [
      { edges: [0.05, 0.15, 0.6], refine: 1 },
      { edges: [0.1, 0.5], refine: 3, crossoverHalfWidth: 0.02, lookahead: 0 },
      { edges: [0.2], refine: 0, maxTones: 8 },
]) {
      const gains = options.edges.map((_, b) => [1, 0.5, 2, 0.25][b]).concat([0.7]);
      const stream = dsp.equalize(x, options, gains);
      const batch = await dsp.equalizeBatch(x, options, gains);
      assert.deepEqual(batch.output, stream.output, JSON.stringify(options));
      batch.bands.forEach((band, b) => assert.deepEqual(band, stream.bands[b], `band ${b}`));
      assert.ok(batch.stats.fitted > 30 && batch.stats.rounds >= 2 + (options.refine ?? 1), JSON.stringify(batch.stats));
    }
  });

  test('any block size gives the same output, and two channels keep their own state', () => {
    const a = signal(12_000, [[0.5, 0.07, 0]], 0.1, 3);
    const b = signal(12_000, [[0.3, 0.4, 1]], 0.1, 4);
    const reference = dsp.equalize(a, { edges: [0.1] }, [0.5, 2]).output;
    for (const block of [1, 128, 777, 4096]) {
      const views = dsp.configureEqualizer({ edges: [0.1], channels: 2 });
      dsp.setEqualizerGains([0.5, 2]);
      const out = new Float32Array(a.length + views.latency);
      for (let at = 0; at < out.length; at += block) {
        const count = Math.min(block, out.length - at);
        for (const [channel, x] of [[1, b], [0, a]]) {
          for (let i = 0; i < count; i++) views.input[i] = at + i < x.length ? x[at + i] : 0;
          dsp.processEqualizer(channel, count);
          if (channel === 0) out.set(views.output.subarray(0, count), at);
        }
      }
      for (let k = 0; k < a.length; k++) assert.equal(out[k + views.latency], reference[k], `block ${block}, sample ${k}`);
    }
  });

  test('a tone near an edge goes whole to its band, where a crossover splits it', () => {
    const edge = 0.1;
    for (const offset of [-3, -1, -0.4, 0.4, 1, 3]) {
      const x = signal(44_100, [[0.5, edge + offset * bin, 0.3]], 0.0005, 9);
      const component = dsp.equalize(x, { edges: [edge] }).bands;
      const crossover = dsp.equalize(x, { edges: [edge], maxTones: 0 }).bands;
      const [own, other] = offset < 0 ? [0, 1] : [1, 0];
      const leak = (bands) => 20 * Math.log10(rms(bands[other], 8192, 36_000) / rms(bands[own], 8192, 36_000));
      // The fitted tone leaves its band only through the noise and the residual (−60 dB or less); the
      // residual crossover alone gives about half of it to the other band so close to the edge.
      assert.ok(leak(component) < -60, `${offset} bins: ${leak(component).toFixed(1)} dB`);
      assert.ok(leak(crossover) > -12, `${offset} bins, crossover: ${leak(crossover).toFixed(1)} dB`);
    }
  });

  test('gains scale each band: a tone per band comes out at its gain, a zero gain removes its band', () => {
    const tones = [[0.3, 0.03, 0], [0.3, 0.2, 1], [0.3, 0.8, 2]];
    const gains = [2, 0.5, 0];
    const alone = tones.map((tone) => dsp.equalize(signal(30_000, [tone]), { edges: [0.1, 0.4] }, gains).output);
    tones.forEach(([amplitude], k) => {
      const level = rms(alone[k], 8192, 22_000) / (amplitude / Math.SQRT2);
      if (gains[k] === 0) assert.ok(level < 1e-3, `band ${k}: ${level}`);
      else assert.ok(Math.abs(20 * Math.log10(level / gains[k])) < 0.05, `band ${k}: ${level} vs ${gains[k]}`);
    });
  });

  test('the residual crossover alone is the zero-phase Butterworth pair: symmetric, |H|² in level', () => {
    // An impulse: the low band is h(−t) * h(t), symmetric about the impulse to the lookahead's truncation.
    const edge = 0.08;
    const x = new Float32Array(20_000);
    x[10_000] = 1;
    const { bands, latency } = dsp.equalize(x, { edges: [edge], maxTones: 0, order: 4, lookahead: 3 });
    assert.ok(latency > 0);
    const low = bands[0];
    let peak = 0;
    for (let d = 0; d < 400; d++) {
      peak = Math.max(peak, Math.abs(low[10_000 + d]));
      assert.ok(Math.abs(low[10_000 + d] - low[10_000 - d]) < 1e-6, `asymmetric at ${d}: ${low[10_000 + d]} vs ${low[10_000 - d]}`);
    }
    // Its transform is 1/(1 + (ω/edge)^8) (the line curve's response, close to the samples' below 0.5).
    for (const omega of [0.02, 0.06, edge, 0.1, 0.13]) {
      let re = 0;
      let im = 0;
      for (let t = 9_000; t < 11_000; t++) {
        re += low[t] * Math.cos(omega * (t - 10_000));
        im -= low[t] * Math.sin(omega * (t - 10_000));
      }
      const expected = 1 / (1 + (omega / edge) ** 8);
      assert.ok(Math.abs(re - expected) < 0.01 && Math.abs(im) < 1e-5, `ω ${omega}: ${re} + ${im}i vs ${expected}`);
    }
    assert.ok(peak > 0.01);
  });

  test('validation: windows, edges, orders, lookahead, gains, channels', () => {
    assert.throws(() => dsp.configureEqualizer({ edges: [0.1], window: 1000 }), RangeError);
    assert.throws(() => dsp.configureEqualizer({ edges: [0.2, 0.1] }), RangeError);
    assert.throws(() => dsp.configureEqualizer({ edges: [4] }), RangeError);
    assert.throws(() => dsp.configureEqualizer({ edges: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] }), RangeError);
    assert.throws(() => dsp.configureEqualizer({ edges: [0.1], order: 9 }), RangeError);
    assert.throws(() => dsp.configureEqualizer({ edges: [0.1], lookahead: 4 }), RangeError);
    assert.throws(() => dsp.configureEqualizer({ edges: [0.1], channels: 3 }), RangeError);
    const views = dsp.configureEqualizer({ edges: [0.1] });
    assert.equal(views.bands, 2);
    assert.throws(() => dsp.setEqualizerGains([1]), RangeError);
    assert.throws(() => dsp.setEqualizerGains([1, Number.NaN]), RangeError);
    assert.throws(() => dsp.processEqualizer(1, 10), RangeError);
    assert.throws(() => dsp.processEqualizer(0, views.input.length + 1), RangeError);
    // No bands: the input, delayed.
    const x = signal(5000, [[0.5, 0.3, 0]]);
    const { output } = dsp.equalize(x, { edges: [] });
    for (let k = 0; k < x.length; k++) assert.equal(output[k], x[k]);
  });
});
