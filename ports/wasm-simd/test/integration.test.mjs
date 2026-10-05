import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { SinWhisky } from '../index.mjs';

test('legacy and finite-window FAFT keep independent geometry in one instance', async () => {
  const sw = await SinWhisky.load();
  const samples = Float32Array.from({ length: 65 }, (_, k) => Math.sin(0.3 * k));
  sw.prepareAft(samples);
  const legacy = sw.values([0.3, 1.1]);
  assert.equal(sw.prepareFaft(samples), samples.length - 1);
  const modern = sw.faftValues([0.3, 1.1]).slice();
  sw.equalize(samples, { edges: [0.4], maxTones: 0 });
  assert.deepEqual(sw.values([0.3, 1.1]), legacy);
  sw.prepareFaft(samples);
  assert.deepEqual(sw.faftValues([0.3, 1.1]), modern);
  const module = await WebAssembly.compile(await readFile(new URL('../build/sinwhisky.wasm', import.meta.url)));
  assert.deepEqual(SinWhisky.fromModule(module).resample([0, 1, 0], 1), sw.resample([0, 1, 0], 1));
});

test('SWL2 rejects values before integer coercion and non-integer controls', async () => {
  const sw = await SinWhisky.load();
  for (const sample of [NaN, Infinity, 0.1, 2 ** 32, -(2 ** 32)]) {
    assert.throws(() => sw.encodeSwl2([[sample]]), RangeError);
  }
  for (const options of [
    { segmentExponent: 12.5 }, { keyInterval: -1 }, { keyInterval: 1.5 },
    { effort: 'unknown' }, { blockSearch: false, blockExponent: 8.5 },
]) assert.throws(() => sw.encodeSwl2([[0]], options), RangeError);
  const encoded = sw.encodeSwl2([[0, 1, 2]]);
  assert.throws(() => sw.decodeSwl2(encoded.bytes, 0.5), RangeError);
  for (const flags of [1, 2, 255]) {
    const bytes = encoded.bytes.slice();
    bytes[7] = flags;
    assert.throws(() => sw.decodeSwl2(bytes), RangeError);
  }
});
