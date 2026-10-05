// Transposed from appx/dsp at ec84edcd331af8810be9df42cf5d16f3a348d6d5 (2026-10-05).
// SWL2's correction layer (assembly/swl2.ts; sinwhisky.rs src/layered, docs/SWL2.md there): streams that
// sinwhisky.rs encoded (fixtures/swl2, from simulator/src/experiments/swl2-reference) decode here, and this
// encoder writes the same bytes; every sample within the tolerance; key segments decode on their own;
// corrupt streams are rejected without trapping.

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { before, describe, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { SinWhisky } from '../index.mjs';

let dsp;
before(async () => {
  dsp = await SinWhisky.instantiate(await readFile(fileURLToPath(new URL('../build/sinwhisky.wasm', import.meta.url))));
});

const fixture = async (name) => new Uint8Array(await readFile(fileURLToPath(new URL(`../../fixtures/swl2/${name}`, import.meta.url))));

/** The fixtures' source: a deterministic synthetic stereo signal (two tones, a third on the right, noise). */
function fixtureSignal(frames) {
  let state = 12345;
  const noise = () => ((state = (state * 1_103_515_245 + 12_345) % 2_147_483_648) / 2_147_483_648 - 0.5) * 400;
  const l = new Int32Array(frames);
  const r = new Int32Array(frames);
  for (let i = 0; i < frames; i++) {
    l[i] = Math.round(8000 * Math.sin(0.031 * i) + 3000 * Math.sin(0.47 * i + 1) + noise());
    r[i] = Math.round(0.7 * l[i] + 2000 * Math.sin(0.12 * i) + noise());
  }
  return [l, r];
}

const maxError = (a, b) => Math.max(...a.map((c, k) => c.reduce((m, v, i) => Math.max(m, Math.abs(v - b[k] [i])), 0)));

describe('SWL2', () => {
  test('sinwhisky.rs streams decode here, and this encoder writes the same bytes', async () => {
    const [l, r] = fixtureSignal(10_000);
    const cases = [
      { name: 'stereo-lossless-segments.swl2', channels: [l, r], options: { segmentExponent: 12, keyInterval: 2 }, tolerance: 0 },
      { name: 'stereo-t16.swl2', channels: [l, r], options: { tolerance: 16 }, tolerance: 16 },
      { name: 'mono-t128-lpc-b9.swl2', channels: [l], options: { tolerance: 128, cascade: false, blockSearch: false, blockExponent: 9 }, tolerance: 128 },
];
    for (const { name, channels, options, tolerance } of cases) {
      const stream = await fixture(name);
      const decoded = dsp.decodeSwl2(stream);
      assert.ok(maxError(decoded, channels) <= tolerance, `${name}: error ${maxError(decoded, channels)}`);
      if (tolerance === 0) assert.equal(maxError(decoded, channels), 0);
      const encoded = dsp.encodeSwl2(channels, options);
      assert.deepEqual(encoded.bytes, stream, `${name}: the bytes differ from sinwhisky.rs's`);
      assert.equal(maxError(encoded.decoded, decoded), 0, `${name}: the encoder's reconstruction is the decoder's`);
    }
  });

  test('round trips: every tolerance within its bound, the cascade on and off, fast and full effort', () => {
    const [l, r] = fixtureSignal(8_000);
    for (const tolerance of [0, 1, 16, 300]) {
      for (const cascade of [true, false]) {
        for (const effort of ['fast', 'full']) {
          const encoded = dsp.encodeSwl2([l, r], { tolerance, cascade, effort, blockSearch: false });
          const decoded = dsp.decodeSwl2(encoded.bytes);
          assert.equal(maxError(decoded, encoded.decoded), 0, `τ ${tolerance} ${cascade} ${effort}: decoder and encoder disagree`);
          assert.ok(maxError(decoded, [l, r]) <= tolerance, `τ ${tolerance} ${cascade} ${effort}`);
        }
      }
    }
    // Lossless stereo picks the cheapest mode: the right channel is mostly 0.7 of the left.
    const lossless = dsp.encodeSwl2([l, r]);
    assert.notEqual(lossless.segments[0] .stereo, 'independent');
    // A tolerance buys bytes.
    assert.ok(dsp.encodeSwl2([l, r], { tolerance: 16 }).bytes.length < 0.6 * lossless.bytes.length);
  });

  test('a key segment decodes on its own; continuations need what came before', () => {
    const [l, r] = fixtureSignal(5 * 4096);
    const encoded = dsp.encodeSwl2([l, r], { segmentExponent: 12, keyInterval: 2 });
    assert.deepEqual(encoded.segments.map((s) => s.key), [true, false, true, false, true]);
    const from = dsp.decodeSwl2(encoded.bytes, 2);
    assert.equal(from[0] .length, 3 * 4096);
    assert.equal(maxError(from, [l.subarray(2 * 4096), r.subarray(2 * 4096)]), 0);
    assert.throws(() => dsp.decodeSwl2(encoded.bytes, 1), RangeError);
  });

  test('corrupt or unsupported streams are rejected, never trapped', async () => {
    const stream = await fixture('stereo-t16.swl2');
    for (const mutate of [
      (b) => b.subarray(0, 100),
      (b) => { b[4] = 3; return b; },
      (b) => { b[22] = 1; return b; }, // the first chunk a gradients chunk: not supported here
      (b) => Uint8Array.of(...b, 0),
]) {
      assert.throws(() => dsp.decodeSwl2(mutate(Uint8Array.from(stream))), RangeError);
    }
    // Random bytes inside the range-coded payload either decode to something or are rejected.
    let state = 7;
    for (let trial = 0; trial < 20; trial++) {
      const b = Uint8Array.from(stream);
      for (let k = 0; k < 8; k++) {
        state = (state * 1_103_515_245 + 12_345) % 2_147_483_648;
        b[40 + (state % (b.length - 60))] = state & 0xff;
      }
      try {
        dsp.decodeSwl2(b);
      } catch (error) {
        assert.ok(error instanceof RangeError);
      }
    }
    assert.throws(() => dsp.encodeSwl2([Int32Array.of(1 << 23)]), RangeError);
    assert.throws(() => dsp.encodeSwl2([Int32Array.of(1, 2), Int32Array.of(1)]), RangeError);
    assert.throws(() => dsp.encodeSwl2([Int32Array.of(1)], { tolerance: -1 }), RangeError);
  });
});
