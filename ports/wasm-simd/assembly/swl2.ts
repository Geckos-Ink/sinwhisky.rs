// Transposed from appx/dsp at ec84edcd331af8810be9df42cf5d16f3a348d6d5 (2026-10-05).
// SWL2's correction layer (R. Cecchini, sinwhisky.rs: src/layered/correction.rs and coder.rs, specified
// byte by byte in docs/SWL2.md §4): closed-loop LPC with integer coefficients, then a cascade of three
// sign-sign LMS stages (1024, 256 and 16 taps, as in Monkey's Audio), quantization with the odd step
// 2τ + 1 (every reconstructed sample within τ of the source; 0 is lossless), and an LZMA-style adaptive
// binary range coder with two-rate probabilities and an adaptive gamma code. Everything is integer
// arithmetic, so a chunk decodes identically here and in sinwhisky.rs, on any platform: this kernel
// encodes and decodes the correction chunks of SWL2 version 4 streams (signal-processor.mjs frames them).
//
// The cascade keeps sinwhisky.rs's i32 weights, inputs and steps and runs in SIMD lanes: four weights per
// i32x4 update (add or subtract the steps, clamp to ±2²⁰), and exact 64-bit products by i64x2.extmul, so
// every prediction equals the scalar i64 one. Each stage's update and its next prediction are one pass.
//
// State slots (MAX_SLOTS) hold what a continuation segment inherits from the previous one: the adaptive
// probabilities and running magnitude, the last predictor, the cascade and the last 32 samples. The
// encoder chooses per block: `full` effort is sinwhisky.rs's search (every LPC order at 15-, 12- and
// 9-bit coefficients and the previous predictor sized exactly, then the cascade on the best candidates),
// `fast` effort sizes two orders and runs the cascade once. Gradient and anchor chunks (the layer
// prediction P) are not supported: P = 0, and the layer flag is never set.

const MAX_SLOTS: i32 = 8;
const MAX_ORDER: i32 = 32;
const PRECISION: i32 = 15;
const CONTEXTS: i32 = 24;
const LIMIT: i64 = 1 << 40;
const HISTORY_LIMIT: i64 = (1 << 24) - 1;
const WEIGHT_LIMIT: i32 = 1 << 20;
const MAX_BLOCK: i32 = 32768;
const OUT_BYTES: i32 = 786432;
const IN_BYTES: i32 = 131072;
/** Bytes a decoded sample may need at most (a block header with 32 gamma-coded deltas included). */
const MARGIN: i32 = 2048;

// Stages: taps, shift, and their f64 buffers in a slot (weights, then inputs and steps with WINDOW of slack).
const WINDOW: i32 = 256;
const S0: i32 = 1024;
const S1: i32 = 256;
const S2: i32 = 16;

// ---- slot layout -----------------------------------------------------------------------------------

// Probabilities: u16 fast, u16 slow. Index map (PROBS of them):
const P_ZERO: i32 = 0; // [CONTEXTS]
const P_SIGN: i32 = 24; // [CONTEXTS]
const P_MAG: i32 = 48; // [CONTEXTS] gamma codes of GAMMA probabilities each
const GAMMA: i32 = 240; // prefix[48], then tree[48][4]
const P_REUSE: i32 = P_MAG + CONTEXTS * GAMMA; // 5808
const P_DELTA: i32 = P_REUSE + 1;
const P_COEF: i32 = P_REUSE + 2; // one gamma code
const PROBS: i32 = P_COEF + GAMMA; // 6050
const MODEL_BYTES: i32 = 24576; // PROBS · 4, padded

// Cascade buffers, i32, byte offsets inside the cascade area.
const W0: i32 = 0;
const I0: i32 = W0 + S0 * 4;
const D0: i32 = I0 + (S0 + WINDOW) * 4;
const W1: i32 = D0 + (S0 + WINDOW) * 4;
const I1: i32 = W1 + S1 * 4;
const D1: i32 = I1 + (S1 + WINDOW) * 4;
const W2: i32 = D1 + (S1 + WINDOW) * 4;
const I2: i32 = W2 + S2 * 4;
const D2: i32 = I2 + (S2 + WINDOW) * 4;
const CASCADE_BYTES: i32 = D2 + (S2 + WINDOW) * 4; // 21696

// Scalars after the model and the cascade.
const SCALARS: i32 = MODEL_BYTES + CASCADE_BYTES; // 67968
const F_MEAN: i32 = SCALARS; // u64: the index model's running magnitude
const F_CASCADE: i32 = SCALARS + 8; // i32: the slot has a cascade (it ran in the last segment)
const F_ORDER: i32 = SCALARS + 12; // i32: previous predictor's order
const F_SHIFT: i32 = SCALARS + 16; // i32: its shift
const F_HISTORY: i32 = SCALARS + 20; // i32: samples of history (0..32)
const F_COEF: i32 = SCALARS + 24; // i64[32]: previous predictor's coefficients
const F_RESTORED: i32 = F_COEF + 256; // i64[32]: the last reconstructed samples, oldest first
const F_POS: i32 = F_RESTORED + 256; // i32[3]: each stage's position
const F_SMEAN: i32 = F_POS + 16; // i64[3]: each stage's running mean of |input|
const SCALAR_BYTES: i32 = 40; // positions and means
const F_USES: i32 = F_SMEAN + 28; // i32: the stages the last block used (the fast encoder's guess; not coded)
const SLOT_BYTES: i32 = 49152;

const slots = memory.data(393216, 16); // MAX_SLOTS · SLOT_BYTES
// Trial models (sizing alternatives) and two scratch cascades for the encoder's trials.
const trialModels = memory.data(98304, 16); // 4 · MODEL_BYTES
const trialMeans = memory.data(32, 16); // u64[4]
const scratchCascade = memory.data(21696, 16);
const scratchScalars = memory.data(64, 16); // pos[3] i32, mean[3] i64

@inline function slotBase(slot: i32): usize {
  return slots + <usize>(slot * SLOT_BYTES);
}

// ---- I/O buffers -----------------------------------------------------------------------------------

const samplesIn = memory.data(131072, 16); // i32[MAX_BLOCK]: the encoder's block
const samplesOut = memory.data(131072, 16); // i32[MAX_BLOCK]: reconstructed samples
const bytesOut = memory.data(OUT_BYTES, 16); // the encoder's payload
const bytesIn = memory.data(IN_BYTES, 16); // the decoder's payload window

export function swl2SamplesInPtr(): usize {
  return samplesIn;
}

export function swl2SamplesOutPtr(): usize {
  return samplesOut;
}

export function swl2BytesOutPtr(): usize {
  return bytesOut;
}

export function swl2BytesInPtr(): usize {
  return bytesIn;
}

export function swl2MaxBlock(): i32 {
  return MAX_BLOCK;
}

export function swl2MaxSlots(): i32 {
  return MAX_SLOTS;
}

export function swl2InBytes(): i32 {
  return IN_BYTES;
}

export function swl2OutBytes(): i32 {
  return OUT_BYTES;
}

// ---- probabilities and the range coder -------------------------------------------------------------

// The model the coder updates: a slot's model or a trial model.
let model: usize = 0;

@inline function probAddr(index: i32): usize {
  return model + (<usize>index << 2);
}

@inline function probability(index: i32): u32 {
  const a = probAddr(index);
  const p = (<u32>load<u16>(a) + <u32>load<u16>(a, 2)) >> 1;
  return p < 31 ? 31 : p > 4065 ? 4065 : p;
}

@inline function adapt(index: i32, bit: bool): void {
  const a = probAddr(index);
  const fast = <u32>load<u16>(a);
  const slow = <u32>load<u16>(a, 2);
  if (bit) {
    store<u16>(a, <u16>(fast - (fast >> 4)));
    store<u16>(a, <u16>(slow - (slow >> 7)), 2);
  } else {
    store<u16>(a, <u16>(fast + ((4096 - fast) >> 4)));
    store<u16>(a, <u16>(slow + ((4096 - slow) >> 7)), 2);
  }
}

function freshModel(base: usize): void {
  for (let i = 0; i < PROBS; i++) {
    store<u16>(base + (<usize>i << 2), 2048);
    store<u16>(base + (<usize>i << 2), 2048, 2);
  }
}

// The encoder: `counting` sizes without writing (a trial), else bytes go to bytesOut.
let eLow: u64 = 0;
let eRange: u32 = 0;
let eCache: u8 = 0;
let ePending: u64 = 0;
let eBytes: i32 = 0;
let counting = false;

function encoderReset(): void {
  eLow = 0;
  eRange = 0xffffffff;
  eCache = 0;
  ePending = 1;
  eBytes = 0;
}

function shiftLow(): void {
  if (eLow < 0xff000000 || eLow >= (<u64>1 << 32)) {
    const carry = <u8>(eLow >> 32);
    let byte = eCache;
    while (true) {
      if (!counting) store<u8>(bytesOut + <usize>eBytes, byte + carry);
      eBytes++;
      byte = 0xff;
      ePending--;
      if (ePending == 0) break;
    }
    eCache = <u8>(eLow >> 24);
  }
  ePending++;
  eLow = (eLow & 0x00ffffff) << 8;
}

@inline function encodeNormalize(): void {
  while (eRange < (1 << 24)) {
    eRange <<= 8;
    shiftLow();
  }
}

function putBit(index: i32, bit: bool): void {
  const bound = (eRange >> 12) * probability(index);
  if (bit) {
    eLow += <u64>bound;
    eRange -= bound;
  } else {
    eRange = bound;
  }
  adapt(index, bit);
  encodeNormalize();
}

function putDirect(value: u64, bits: i32): void {
  for (let i = bits - 1; i >= 0; i--) {
    eRange >>= 1;
    if (((value >> <u64>i) & 1) == 1) eLow += <u64>eRange;
    encodeNormalize();
  }
}

/** Bytes written plus those a flush would add (coder.rs `size`). */
@inline function encoderSize(): i32 {
  return eBytes + <i32>ePending + 4;
}

function encoderFinish(): void {
  for (let i = 0; i < 5; i++) shiftLow();
}

@inline function bitlen(v: u64): i32 {
  return 64 - <i32>clz<u64>(v);
}

/** The adaptive gamma code of v ≥ 0 at probability offset `base` (coder.rs `Gamma::put`). */
function putGamma(base: i32, v: u64): void {
  const m = v + 1;
  const width = bitlen(m);
  for (let i = 0; i < width - 1; i++) putBit(base + i, true);
  putBit(base + width - 1, false);
  const below = width - 1;
  const modelled = below < 2 ? below : 2;
  let node = 1;
  for (let i = 0; i < modelled; i++) {
    const bit = <i32>((m >> <u64>(below - 1 - i)) & 1);
    putBit(base + 48 + width * 4 + node, bit == 1);
    node = 2 * node + bit;
  }
  putDirect(m, below - modelled);
}

// The decoder over bytesIn[dPos .. dEnd).
let dRange: u32 = 0;
let dCode: u32 = 0;
let dPos: i32 = 0;
let dEnd: i32 = 0;
let corrupt = false;

@inline function nextByte(): u32 {
  if (dPos >= dEnd) {
    corrupt = true;
    return 0;
  }
  const byte = <u32>load<u8>(bytesIn + <usize>dPos);
  dPos++;
  return byte;
}

@inline function decodeNormalize(): void {
  while (dRange < (1 << 24)) {
    dRange <<= 8;
    dCode = (dCode << 8) | nextByte();
  }
}

function getBit(index: i32): bool {
  const bound = (dRange >> 12) * probability(index);
  let bit = false;
  if (dCode < bound) {
    dRange = bound;
  } else {
    dCode -= bound;
    dRange -= bound;
    bit = true;
  }
  adapt(index, bit);
  decodeNormalize();
  return bit;
}

function getDirect(bits: i32): u64 {
  let value: u64 = 0;
  for (let i = 0; i < bits; i++) {
    dRange >>= 1;
    const b = dCode >= dRange;
    if (b) dCode -= dRange;
    value = (value << 1) | (b ? 1 : 0);
    decodeNormalize();
  }
  return value;
}

function getGamma(base: i32): u64 {
  let width = 1;
  while (getBit(base + width - 1)) {
    width++;
    if (width > 46) {
      corrupt = true;
      return 0;
    }
  }
  const below = width - 1;
  const modelled = below < 2 ? below : 2;
  let node = 1;
  for (let i = 0; i < modelled; i++) node = 2 * node + (getBit(base + 48 + width * 4 + node) ? 1 : 0);
  const rest = getDirect(below - modelled);
  return ((<u64>node << <u64>(below - modelled)) | rest) - 1;
}

// ---- the index model -------------------------------------------------------------------------------

// The running magnitude of the model in use.
let mean: u64 = 16;

@inline function context(): i32 {
  const c = bitlen(mean >> 3);
  return c < CONTEXTS - 1 ? c : CONTEXTS - 1;
}

@inline function updateMean(q: i64): void {
  let a = <u64>(q < 0 ? -q : q);
  if (a > (1 << 24)) a = 1 << 24;
  mean = mean - (mean >> 4) + a;
}

function putIndex(q: i64): void {
  const c = context();
  putBit(P_ZERO + c, q != 0);
  if (q != 0) {
    putBit(P_SIGN + c, q < 0);
    putGamma(P_MAG + c * GAMMA, <u64>(q < 0 ? -q : q) - 1);
  }
  updateMean(q);
}

function getIndex(): i64 {
  const c = context();
  let q: i64 = 0;
  if (getBit(P_ZERO + c)) {
    const negative = getBit(P_SIGN + c);
    const m = getGamma(P_MAG + c * GAMMA) + 1;
    if (m > (<u64>1 << 41)) {
      corrupt = true;
      return 0;
    }
    q = negative ? -<i64>m : <i64>m;
  }
  updateMean(q);
  return q;
}

// ---- predictors ------------------------------------------------------------------------------------

// The block's predictor and the previous block's (coefficients i64[32], order, shift).
const coef = memory.data(256, 16);
let order: i32 = 0;
let shift: i32 = 0;
const prevCoef = memory.data(256, 16);
let prevOrder: i32 = 0;
let prevShift: i32 = 0;

@inline function c64(base: usize, i: i32): i64 {
  return load<i64>(base + (<usize>i << 3));
}

function samePredictor(): bool {
  if (order != prevOrder) return false;
  if (order == 0) return true;
  if (shift != prevShift) return false;
  for (let i = 0; i < order; i++) if (c64(coef, i) != c64(prevCoef, i)) return false;
  return true;
}

@inline function zigzag(v: i64): u64 {
  return <u64>((v << 1) ^ (v >> 63));
}

@inline function unzigzag(v: u64): i64 {
  return <i64>(v >> 1) ^ -<i64>(v & 1);
}

@inline function signedWidth(v: i64): i32 {
  if (v == 0) return 0;
  return v > 0 ? bitlen(<u64>v) + 1 : bitlen(<u64>~v) + 1;
}

/** Predictor coding relative to the previous block's (correction.rs `PredictorModel::put`). */
function putPredictor(): void {
  const same = samePredictor();
  putBit(P_REUSE, same);
  if (same) return;
  putDirect(<u64>order, 6);
  if (order == 0) return;
  putDirect(<u64>shift, 5);
  let width = 1;
  for (let i = 0; i < order; i++) {
    const w = signedWidth(c64(coef, i));
    if (w > width) width = w;
  }
  const keyBits = <u64>5 + <u64>order * <u64>width;
  let delta = false;
  if (shift == prevShift) {
    let bits: u64 = 0;
    for (let i = 0; i < order; i++) {
      const d = c64(coef, i) - (i < prevOrder ? c64(prevCoef, i) : 0);
      bits += <u64>(2 * bitlen(zigzag(d) + 1) - 1);
    }
    delta = bits < keyBits;
    putBit(P_DELTA, delta);
  }
  if (delta) {
    for (let i = 0; i < order; i++) putGamma(P_COEF, zigzag(c64(coef, i) - (i < prevOrder ? c64(prevCoef, i) : 0)));
  } else {
    putDirect(<u64>width, 5);
    const mask: u64 = (<u64>1 << <u64>width) - 1;
    for (let i = 0; i < order; i++) putDirect(<u64>c64(coef, i) & mask, width);
  }
}

/** Reads the block's predictor into coef/order/shift (correction.rs `PredictorModel::get`). */
function getPredictor(): void {
  if (getBit(P_REUSE)) {
    order = prevOrder;
    shift = prevShift;
    memory.copy(coef, prevCoef, 256);
    return;
  }
  const o = <i32>getDirect(6);
  if (o > MAX_ORDER) {
    corrupt = true;
    return;
  }
  order = o;
  shift = 0;
  if (o == 0) return;
  shift = <i32>getDirect(5);
  if (prevShift == shift && getBit(P_DELTA)) {
    for (let i = 0; i < o; i++) {
      const base: i64 = i < prevOrder ? c64(prevCoef, i) : 0;
      let g = getGamma(P_COEF);
      if (g > (<u64>1 << 40)) g = <u64>1 << 40;
      const c = base + unzigzag(g);
      if (c >= (1 << PRECISION) || c <= -(1 << PRECISION)) corrupt = true;
      store<i64>(coef + (<usize>i << 3), c);
    }
  } else {
    const width = <i32>getDirect(5);
    if (width < 1 || width > PRECISION) {
      corrupt = true;
      return;
    }
    for (let i = 0; i < o; i++) {
      const v = getDirect(width);
      const s = <u64>(64 - width);
      store<i64>(coef + (<usize>i << 3), (<i64>(v << s)) >> <i64>s);
    }
  }
}

// ---- the closed loop -------------------------------------------------------------------------------

// The history the block extends: the last samples, most recent at history[count − 1].
const history = memory.data(263680, 16); // i64[MAX_BLOCK + 2·MAX_ORDER + 16]
let historyCount: i32 = 0;

@inline function floorDiv(a: i64, b: i64): i64 {
  let q = a / b;
  if (a % b < 0) q -= 1;
  return q;
}

@inline function clampLimit(v: i64): i64 {
  return v < -LIMIT ? -LIMIT : v > LIMIT ? LIMIT : v;
}

/** The LPC prediction from history[0..n) (correction.rs `Predictor::predict`). */
@inline function lpc(n: i32): i64 {
  let available = n < MAX_ORDER ? n : MAX_ORDER;
  if (available > order) available = order;
  let sum: i64 = 0;
  const top = history + (<usize>(n - 1) << 3);
  for (let i = 0; i < available; i++) sum += c64(coef, i) * load<i64>(top - (<usize>i << 3));
  const p = shift > 0 ? (sum + (<i64>1 << <i64>(shift - 1))) >> <i64>shift : sum;
  return clampLimit(p);
}

// ---- the LMS cascade -------------------------------------------------------------------------------

// The cascade in use (a slot's, or the scratch one) and its scalars; p0..p2 are its stages' predictions
// for the next sample (predictCascade at the start of a run, then each update computes the next ones).
let cascade: usize = 0;
let cascadeScalars: usize = 0; // pos[3] i32, mean[3] i64 (slot or scratch layout)
let p0: i64 = 0;
let p1: i64 = 0;
let p2: i64 = 0;

@inline function stagePos(s: i32): i32 {
  return load<i32>(cascadeScalars + (<usize>s << 2));
}

@inline function setStagePos(s: i32, v: i32): void {
  store<i32>(cascadeScalars + (<usize>s << 2), v);
}

@inline function stageMean(s: i32): i64 {
  return load<i64>(cascadeScalars + 16 + (<usize>s << 3));
}

@inline function setStageMean(s: i32, v: i64): void {
  store<i64>(cascadeScalars + 16 + (<usize>s << 3), v);
}

function freshCascade(area: usize, scalars: usize): void {
  memory.fill(area, 0, CASCADE_BYTES);
  store<i32>(scalars, S0);
  store<i32>(scalars, S1, 4);
  store<i32>(scalars, S2, 8);
  store<i64>(scalars, 0, 16);
  store<i64>(scalars, 0, 24);
  store<i64>(scalars, 0, 32);
}

/** Σ W_j·v(n − j) of one stage in exact 64-bit products (four taps per step). */
@inline function dot(weights: usize, inputs: usize, taps: i32): i64 {
  let sum = i64x2.splat(0);
  for (let j = 0; j < taps; j += 4) {
    const o = <usize>j << 2;
    const w = v128.load(weights + o);
    const v = v128.load(inputs + o);
    sum = i64x2.add(sum, i64x2.add(i64x2.extmul_low_i32x4_s(w, v), i64x2.extmul_high_i32x4_s(w, v)));
  }
  return i64x2.extract_lane(sum, 0) + i64x2.extract_lane(sum, 1);
}

@inline function rounded(sum: i64, shift: i64): i64 {
  return (sum + (<i64>1 << (shift - 1))) >> shift;
}

/** The stages' predictions from their current weights and histories (correction.rs `Cascade::predict`). */
function predictCascade(): void {
  p0 = rounded(dot(cascade + W0, cascade + I0 + (<usize>(stagePos(0) - S0) << 2), S0), 15);
  p1 = rounded(dot(cascade + W1, cascade + I1 + (<usize>(stagePos(1) - S1) << 2), S1), 13);
  p2 = rounded(dot(cascade + W2, cascade + I2 + (<usize>(stagePos(2) - S2) << 2), S2), 11);
}

/**
 * One stage adapts to the sign of `error`, takes `input` into its history (`Stage::update`) and returns
 * its prediction for the next sample: the weight update and the next dot product in one pass.
 */
function stageUpdate(s: i32, weights: usize, inputs: usize, steps: usize, taps: i32, shift: i64, input: i64, error: i64): i64 {
  const pos = stagePos(s);
  const v = input < -HISTORY_LIMIT ? -HISTORY_LIMIT : input > HISTORY_LIMIT ? HISTORY_LIMIT : input;
  const m = v < 0 ? -v : v;
  const a = stageMean(s);
  // Larger steps for inputs far above the running mean (Monkey's Audio).
  const size: i64 = m > 3 * a ? 32 : m > (4 * a) / 3 ? 16 : m > 0 ? 8 : 0;
  setStageMean(s, a + (m - a) / 16);
  // The new input joins the history the next prediction reads; the update uses the steps before it.
  store<i32>(inputs + (<usize>pos << 2), <i32>v);
  const stepsFrom = steps + (<usize>(pos - taps) << 2);
  const next = inputs + (<usize>(pos + 1 - taps) << 2);
  let sum = i64x2.splat(0);
  if (error == 0) {
    for (let j = 0; j < taps; j += 4) {
      const o = <usize>j << 2;
      const w = v128.load(weights + o);
      const x = v128.load(next + o);
      sum = i64x2.add(sum, i64x2.add(i64x2.extmul_low_i32x4_s(w, x), i64x2.extmul_high_i32x4_s(w, x)));
    }
  } else {
    const lo = i32x4.splat(-WEIGHT_LIMIT);
    const hi = i32x4.splat(WEIGHT_LIMIT);
    const up = error > 0;
    for (let j = 0; j < taps; j += 4) {
      const o = <usize>j << 2;
      const d = v128.load(stepsFrom + o);
      const w0 = v128.load(weights + o);
      const w = i32x4.min_s(hi, i32x4.max_s(lo, up ? i32x4.add(w0, d) : i32x4.sub(w0, d)));
      v128.store(weights + o, w);
      const x = v128.load(next + o);
      sum = i64x2.add(sum, i64x2.add(i64x2.extmul_low_i32x4_s(w, x), i64x2.extmul_high_i32x4_s(w, x)));
    }
  }
  store<i32>(steps + (<usize>pos << 2), <i32>(v > 0 ? size : v < 0 ? -size : 0));
  // Older inputs adapt with halved steps after 1, 2 and 8 samples.
  for (let k = 0; k < 3; k++) {
    const at = steps + (<usize>(pos - (k == 2 ? 8 : k + 1)) << 2);
    store<i32>(at, load<i32>(at) >> 1);
  }
  let after = pos + 1;
  if (after == taps + WINDOW) {
    memory.copy(inputs, inputs + (<usize>(after - taps) << 2), <usize>taps << 2);
    memory.copy(steps, steps + (<usize>(after - taps) << 2), <usize>taps << 2);
    after = taps;
  }
  setStagePos(s, after);
  return rounded(i64x2.extract_lane(sum, 0) + i64x2.extract_lane(sum, 1), shift);
}

/** Every stage adapts to the LPC residual r of the reconstruction (`Cascade::update`); p0..p2 become the next predictions. */
function updateCascade(r: i64): void {
  let input = r;
  let next = input - p0;
  const q0 = stageUpdate(0, cascade + W0, cascade + I0, cascade + D0, S0, 15, input, next);
  input = next;
  next = input - p1;
  const q1 = stageUpdate(1, cascade + W1, cascade + I1, cascade + D1, S1, 13, input, next);
  input = next;
  next = input - p2;
  p2 = stageUpdate(2, cascade + W2, cascade + I2, cascade + D2, S2, 11, input, next);
  p0 = q0;
  p1 = q1;
}

@inline function total(base: i64, uses: i32): i64 {
  let p = base;
  if (uses > 0) p += p0;
  if (uses > 1) p += p1;
  if (uses > 2) p += p2;
  return clampLimit(p);
}

// ---- slots -----------------------------------------------------------------------------------------

/** A fresh state (a key segment): probabilities at 1/2, magnitude 16, order 0, no history, no cascade. */
export function swl2Reset(slot: i32): bool {
  if (slot < 0 || slot >= MAX_SLOTS) return false;
  const base = slotBase(slot);
  freshModel(base);
  store<u64>(base + F_MEAN, 16);
  store<i32>(base + F_CASCADE, 0);
  store<i32>(base + F_ORDER, 0);
  store<i32>(base + F_SHIFT, 0);
  store<i32>(base + F_HISTORY, 0);
  store<i32>(base + F_USES, 3);
  return true;
}

/** Copies a slot's whole state (to try alternatives from the same inherited state). */
export function swl2Copy(from: i32, to: i32): bool {
  if (from < 0 || from >= MAX_SLOTS || to < 0 || to >= MAX_SLOTS) return false;
  if (from != to) memory.copy(slotBase(to), slotBase(from), SLOT_BYTES);
  return true;
}

// The chunk being coded.
let active: i32 = -1;
let step: i64 = 1;
let half: i64 = 0;
let tolerance: i64 = 0;
let blockSize: i32 = 0;
let useCascade = false;
let effort: i32 = 0;
let noise: f64 = 0;

/** Loads a slot's previous predictor and history; the cascade per the chunk's flag (fresh after none). */
function open(slot: i32, withCascade: bool): void {
  const base = slotBase(slot);
  active = slot;
  model = base;
  mean = load<u64>(base + F_MEAN);
  prevOrder = load<i32>(base + F_ORDER);
  prevShift = load<i32>(base + F_SHIFT);
  memory.copy(prevCoef, base + F_COEF, 256);
  historyCount = load<i32>(base + F_HISTORY);
  memory.copy(history, base + F_RESTORED, <usize>historyCount << 3);
  useCascade = withCascade;
  cascade = base + MODEL_BYTES;
  cascadeScalars = base + F_POS;
  if (withCascade && load<i32>(base + F_CASCADE) == 0) freshCascade(cascade, cascadeScalars);
  store<i32>(base + F_CASCADE, withCascade ? 1 : 0);
}

/** Saves the model's magnitude, the predictor and the last 32 samples of history into the active slot. */
function close(): void {
  const base = slotBase(active);
  store<u64>(base + F_MEAN, mean);
  store<i32>(base + F_ORDER, prevOrder);
  store<i32>(base + F_SHIFT, prevShift);
  memory.copy(base + F_COEF, prevCoef, 256);
  const keep = historyCount < MAX_ORDER ? historyCount : MAX_ORDER;
  memory.copy(base + F_RESTORED, history + (<usize>(historyCount - keep) << 3), <usize>keep << 3);
  store<i32>(base + F_HISTORY, keep);
}

/** Keeps the last MAX_ORDER samples of history at its start (between blocks). */
function trimHistory(): void {
  if (historyCount > MAX_ORDER) {
    memory.copy(history, history + (<usize>(historyCount - MAX_ORDER) << 3), MAX_ORDER << 3);
    historyCount = MAX_ORDER;
  }
}

// ---- decoding --------------------------------------------------------------------------------------

let decodeLeft: i32 = 0; // samples left in the current block
let decodeUses: i32 = 0;
let payloadDone = false;

/**
 * Starts decoding a correction chunk into `slot` (its inherited state): the payload's first bytes are in
 * bytesIn[0..available) and `final` says whether that is all of it. Returns false for a bad header.
 */
export function swl2DecodeBegin(slot: i32, tol: i32, available: i32, final: bool): bool {
  if (slot < 0 || slot >= MAX_SLOTS || tol < 0 || tol > (1 << 20) || available < 6 || available > IN_BYTES) return false;
  const head = <i32>load<u8>(bytesIn);
  const exponent = head & 0x0f;
  if (exponent < 8 || exponent > 15 || (head & 0x60) != 0 || (head & 0x80) != 0) return false;
  if (load<u8>(bytesIn + 1) != 0) return false;
  open(slot, (head & 0x10) != 0);
  tolerance = tol;
  step = 2 * <i64>tol + 1;
  blockSize = 1 << exponent;
  corrupt = false;
  dPos = 1;
  dEnd = available;
  payloadDone = final;
  dRange = 0xffffffff;
  dCode = 0;
  for (let i = 0; i < 5; i++) dCode = (dCode << 8) | nextByte();
  decodeLeft = 0;
  if (useCascade) predictCascade();
  return true;
}

/** Bytes of the payload window consumed so far (the caller moves the rest to the front and appends). */
export function swl2DecodeConsumed(): i32 {
  return dPos;
}

/** Restarts the window after the caller kept bytes [consumed, end) at its front and appended more. */
export function swl2DecodeRefill(available: i32, final: bool): bool {
  if (available < 0 || available > IN_BYTES) return false;
  dPos = 0;
  dEnd = available;
  payloadDone = final;
  return true;
}

/**
 * Decodes up to `count` samples (at most MAX_BLOCK) into samplesOut; stops early when the payload window
 * runs low and more of it is to come. Returns the samples decoded, or −1 for a corrupt chunk.
 */
export function swl2DecodeRun(count: i32): i32 {
  if (active < 0 || count < 0 || count > MAX_BLOCK) return -1;
  let done = 0;
  while (done < count) {
    if (!payloadDone && dEnd - dPos < MARGIN) break;
    if (decodeLeft == 0) {
      // A block header: the predictor, then the stages used.
      trimHistory();
      getPredictor();
      prevOrder = order;
      prevShift = shift;
      memory.copy(prevCoef, coef, 256);
      decodeUses = useCascade ? <i32>getDirect(2) : 0;
      decodeLeft = blockSize;
      if (corrupt) return -1;
    }
    const q = getIndex();
    if (corrupt) return -1;
    const n = historyCount;
    const base = lpc(n);
    const p = total(base, decodeUses);
    const value = clampLimit(p + q * step);
    if (useCascade) updateCascade(value - base);
    store<i64>(history + (<usize>n << 3), value);
    historyCount = n + 1;
    if (historyCount >= MAX_BLOCK + MAX_ORDER) trimHistory();
    store<i32>(samplesOut + (<usize>done << 2), <i32>value);
    decodeLeft--;
    done++;
  }
  return done;
}

/** Ends the chunk: the slot keeps the state a continuation inherits. Returns the bytes consumed in the last window. */
export function swl2DecodeEnd(): i32 {
  if (active < 0) return -1;
  close();
  active = -1;
  return dPos;
}

// ---- LPC analysis (correction.rs `analyse`, `quantize_coefficients`) -------------------------------

const windowed = memory.data(262144, 16); // f64[MAX_BLOCK]
const autocorrelation = memory.data(272, 16); // f64[33]
const levinson = memory.data(8448, 16); // f64[33][32]: the predictor of every order
let orders: i32 = 1;

function analyse(count: i32): void {
  const n = count;
  const taper = n / 4 > 1 ? n / 4 : 1;
  let weightSquares: f64 = 0;
  for (let i = 0; i < n; i++) {
    const edge = i < n - 1 - i ? i : n - 1 - i;
    const w: f64 = edge >= taper ? 1 : 0.5 - 0.5 * Math.cos((Math.PI * <f64>edge) / <f64>taper);
    weightSquares += w * w;
    store<f64>(windowed + (<usize>i << 3), <f64>load<i64>(history + (<usize>(historyCount + i) << 3)) * w);
  }
  const maxOrder = MAX_ORDER < n - 1 ? MAX_ORDER : n - 1;
  for (let lag = 0; lag <= maxOrder; lag++) {
    let sum: f64 = 0;
    for (let i = 0; i + lag < n; i++) sum += load<f64>(windowed + (<usize>(i + lag) << 3)) * load<f64>(windowed + (<usize>i << 3));
    store<f64>(autocorrelation + (<usize>lag << 3), sum);
  }
  orders = 1;
  let r0 = load<f64>(autocorrelation);
  if (!(r0 > 0) || !isFinite(r0)) return;
  r0 = r0 * (1 + 1e-9) + noise * weightSquares;
  store<f64>(autocorrelation, r0);
  // Levinson–Durbin: a[m] in levinson[m · 32 ..].
  let error = r0;
  for (let m = 1; m <= maxOrder; m++) {
    let k = load<f64>(autocorrelation + (<usize>m << 3));
    for (let i = 0; i < m - 1; i++) k -= load<f64>(levinson + (<usize>((m - 1) * 32 + i) << 3)) * load<f64>(autocorrelation + (<usize>(m - 1 - i) << 3));
    k /= error;
    if (!isFinite(k) || Math.abs(k) >= 1) break;
    for (let i = 0; i < m - 1; i++) {
      const previous = levinson + (<usize>((m - 1) * 32) << 3);
      store<f64>(levinson + (<usize>(m * 32 + i) << 3), load<f64>(previous + (<usize>i << 3)) - k * load<f64>(previous + (<usize>(m - 2 - i) << 3)));
    }
    store<f64>(levinson + (<usize>(m * 32 + m - 1) << 3), k);
    error *= 1 - k * k;
    orders = m + 1;
  }
}

/** Round half away from zero (Rust's f64::round). */
@inline function roundAway(v: f64): f64 {
  return v < 0 ? -Math.floor(-v + 0.5) : Math.floor(v + 0.5);
}

/** The predictor of order o quantized to `precision` bits into coef/order/shift. */
function quantize(o: i32, precision: i32): void {
  order = 0;
  shift = 0;
  if (o == 0) return;
  let largest: f64 = 0;
  const a = levinson + (<usize>(o * 32) << 3);
  for (let i = 0; i < o; i++) largest = Math.max(largest, Math.abs(load<f64>(a + (<usize>i << 3))));
  if (largest == 0 || !isFinite(largest)) return;
  const top = <i64>Math.floor(Math.log2(largest)) + 1;
  let s = <i64>precision - 1 - top;
  if (s < 0) s = 0;
  if (s > 31) s = 31;
  const limit = <f64>((1 << (precision - 1)) - 1);
  const scale = <f64>(<u64>1 << <u64>s);
  for (let i = 0; i < o; i++) {
    let c = roundAway(load<f64>(a + (<usize>i << 3)) * scale);
    if (c > limit) c = limit;
    if (c < -limit) c = -limit;
    store<i64>(coef + (<usize>i << 3), <i64>c);
  }
  order = o;
  shift = <i32>s;
}

// ---- encoding --------------------------------------------------------------------------------------

/**
 * Starts encoding a correction chunk from `slot` (its inherited state): tolerance τ (0..2²⁰), blocks of
 * 2^exponent (8..15), the cascade or not, and the effort (0 fast, 1 full). Writes the payload's head byte.
 */
export function swl2EncodeBegin(slot: i32, tol: i32, exponent: i32, withCascade: bool, how: i32): bool {
  if (slot < 0 || slot >= MAX_SLOTS || tol < 0 || tol > (1 << 20) || exponent < 8 || exponent > 15 || how < 0 || how > 1) return false;
  open(slot, withCascade);
  tolerance = tol;
  step = 2 * <i64>tol + 1;
  half = (step - 1) / 2;
  noise = tol > 0 ? <f64>(step * step - 1) / 12 : 0;
  blockSize = 1 << exponent;
  effort = how;
  counting = false;
  encoderReset();
  store<u8>(bytesOut, <u8>(exponent | (withCascade ? 0x10 : 0)));
  eBytes = 1;
  return true;
}

// The chosen block coding.
let bestBytes: i32 = 0;
const bestCoef = memory.data(256, 16);
let bestOrder: i32 = 0;
let bestShift: i32 = 0;
let bestUses: i32 = 0;

function remember(): void {
  memory.copy(bestCoef, coef, 256);
  bestOrder = order;
  bestShift = shift;
}

function recall(): void {
  memory.copy(coef, bestCoef, 256);
  order = bestOrder;
  shift = bestShift;
}

const altBytes = memory.data(16, 16); // i32[4]

// Sizing trials: up to four trial models and counting coders running side by side.
const trialCoders = memory.data(160, 16); // [4] low u64, pending u64, range u32, cache u32, bytes u32, pad

/**
 * Starts `ways` trial coders, each with a copy of the slot's model and magnitude, and codes the block
 * header with `uses` stages (one way) or with w stages (way w of four).
 */
function trialsBegin(ways: i32, uses: i32): void {
  const saveModel = model;
  const saveMean = mean;
  for (let w = 0; w < ways; w++) {
    const m = trialModels + <usize>(w * MODEL_BYTES);
    memory.copy(m, saveModel, PROBS << 2);
    model = m;
    mean = saveMean;
    counting = true;
    encoderReset();
    putPredictor();
    if (useCascade) putDirect(<u64>(ways == 1 ? uses : w), 2);
    trialSave(w);
  }
  model = saveModel;
  mean = saveMean;
  counting = false;
}

function trialSave(w: i32): void {
  const c = trialCoders + <usize>(w * 40);
  store<u64>(c, eLow);
  store<u64>(c, ePending, 8);
  store<u32>(c, eRange, 16);
  store<u32>(c, <u32>eCache, 20);
  store<i32>(c, eBytes, 24);
  store<u64>(trialMeans + (<usize>w << 3), mean);
}

function trialLoad(w: i32): void {
  const c = trialCoders + <usize>(w * 40);
  eLow = load<u64>(c);
  ePending = load<u64>(c, 8);
  eRange = load<u32>(c, 16);
  eCache = <u8>load<u32>(c, 20);
  eBytes = load<i32>(c, 24);
  mean = load<u64>(trialMeans + (<usize>w << 3));
  model = trialModels + <usize>(w * MODEL_BYTES);
}

// Run outcomes.
let runEstimate: f64 = 0;

const RUN_ESTIMATE: i32 = 0; // Σ log2(1 + 2|q|), no coding
const RUN_SIZE: i32 = 1; // the exact size with `uses` stages (trial coder 0)
const RUN_ALTERNATIVES: i32 = 2; // lossless: the sizes for every number of stages (trial coders 0..3)
const RUN_REAL: i32 = 3; // code into the stream and keep the state

/**
 * Closed-loop coding of the block in history[historyCount ..+count) with the current predictor:
 * reconstruction appended (RUN_REAL) or on a copy (trials). With the cascade, `scratch` runs a copy of the
 * slot's cascade so trials leave it unchanged.
 */
function run(count: i32, uses: i32, mode: i32, withStages: bool): void {
  const start = historyCount;
  const saveModel = model;
  const saveMean = mean;
  if (mode == RUN_SIZE) trialsBegin(1, uses);
  else if (mode == RUN_ALTERNATIVES) trialsBegin(4, 0);
  if (mode != RUN_REAL && withStages) {
    // The trial runs the scratch copy of the slot's cascade.
    memory.copy(scratchCascade, slotBase(active) + MODEL_BYTES, CASCADE_BYTES);
    memory.copy(scratchScalars, slotBase(active) + F_POS, SCALAR_BYTES);
    cascade = scratchCascade;
    cascadeScalars = scratchScalars;
  }
  if (withStages) predictCascade();
  // The source block is kept in samplesIn; the reconstruction goes into history after `start`.
  let estimate: f64 = 0;
  for (let i = 0; i < count; i++) {
    const n = start + i;
    const target = clampLimit(<i64>load<i32>(samplesIn + (<usize>i << 2)));
    const base = lpc(n);
    const p = withStages ? total(base, uses) : base;
    const q = floorDiv(target - p + half, step);
    const value = clampLimit(p + q * step);
    if (mode == RUN_ESTIMATE) {
      estimate += Math.log2(1 + 2 * <f64>(q < 0 ? -q : q));
    } else if (mode == RUN_SIZE) {
      trialLoad(0);
      counting = true;
      putIndex(q);
      trialSave(0);
    } else if (mode == RUN_ALTERNATIVES) {
      counting = true;
      for (let w = 0; w < 4; w++) {
        trialLoad(w);
        putIndex(floorDiv(target - total(base, w) + half, step));
        trialSave(w);
      }
    } else {
      putIndex(q);
    }
    if (withStages) updateCascade(value - base);
    store<i64>(history + (<usize>n << 3), value);
  }
  counting = false;
  model = saveModel;
  // A real run keeps the magnitude it evolved; a trial leaves the slot's as it was.
  if (mode != RUN_REAL) mean = saveMean;
  runEstimate = estimate;
  if (mode == RUN_SIZE) {
    trialLoad(0);
    bestRunBytes = encoderSize();
  } else if (mode == RUN_ALTERNATIVES) {
    for (let w = 0; w < 4; w++) {
      trialLoad(w);
      store<i32>(altBytes + (<usize>w << 2), encoderSize());
    }
  }
  model = saveModel;
  if (mode != RUN_REAL) mean = saveMean;
  cascade = slotBase(active) + MODEL_BYTES;
  cascadeScalars = slotBase(active) + F_POS;
  if (mode == RUN_REAL) {
    historyCount = start + count;
  }
}

let bestRunBytes: i32 = 0;

// The real coder's registers, saved while trials use the same globals.
let realLow: u64 = 0;
let realRange: u32 = 0;
let realCache: u8 = 0;
let realPending: u64 = 0;
let realBytes: i32 = 0;

function saveReal(): void {
  realLow = eLow;
  realRange = eRange;
  realCache = eCache;
  realPending = ePending;
  realBytes = eBytes;
}

function restoreReal(): void {
  eLow = realLow;
  eRange = realRange;
  eCache = realCache;
  ePending = realPending;
  eBytes = realBytes;
}

// Candidates: (bytes, order, shift, coefficients) in order of their creation, stably sorted by bytes.
const MAX_CANDIDATES: i32 = 40;
const candidateBytes = memory.data(160, 16); // i32[MAX_CANDIDATES]
const candidateMeta = memory.data(320, 16); // (order, shift) i32 pairs
const candidateCoef = memory.data(10240, 16); // i64[MAX_CANDIDATES][32]
let candidates: i32 = 0;

function addCandidate(bytes: i32): void {
  if (candidates == MAX_CANDIDATES) return;
  store<i32>(candidateBytes + (<usize>candidates << 2), bytes);
  store<i32>(candidateMeta + (<usize>candidates << 3), order);
  store<i32>(candidateMeta + (<usize>candidates << 3), shift, 4);
  memory.copy(candidateCoef + (<usize>(candidates * 32) << 3), coef, 256);
  candidates++;
}

function loadCandidate(i: i32): void {
  order = load<i32>(candidateMeta + (<usize>i << 3));
  shift = load<i32>(candidateMeta + (<usize>i << 3), 4);
  memory.copy(coef, candidateCoef + (<usize>(i * 32) << 3), 256);
}

/** Stable insertion sort of the candidates by bytes (Rust's `sort_by_key` is stable). */
function sortCandidates(): void {
  for (let i = 1; i < candidates; i++) {
    for (let j = i; j > 0 && load<i32>(candidateBytes + (<usize>(j - 1) << 2)) > load<i32>(candidateBytes + (<usize>j << 2)); j--) {
      const b = load<i32>(candidateBytes + (<usize>j << 2));
      store<i32>(candidateBytes + (<usize>j << 2), load<i32>(candidateBytes + (<usize>(j - 1) << 2)));
      store<i32>(candidateBytes + (<usize>(j - 1) << 2), b);
      const m = load<u64>(candidateMeta + (<usize>j << 3));
      store<u64>(candidateMeta + (<usize>j << 3), load<u64>(candidateMeta + (<usize>(j - 1) << 3)));
      store<u64>(candidateMeta + (<usize>(j - 1) << 3), m);
      memory.copy(scratchCoef, candidateCoef + (<usize>(j * 32) << 3), 256);
      memory.copy(candidateCoef + (<usize>(j * 32) << 3), candidateCoef + (<usize>((j - 1) * 32) << 3), 256);
      memory.copy(candidateCoef + (<usize>((j - 1) * 32) << 3), scratchCoef, 256);
    }
  }
}

const scratchCoef = memory.data(256, 16);

// The orders tried, and their estimates.
const ORDER_LIST = memory.data<i32>([0, 1, 2, 4, 8, 12, 16, 20, 24, 32]);
const ranked = memory.data(80, 16); // f64[10] estimates
const rankedOrder = memory.data(40, 16); // i32[10]

/** The LPC-only size of the current predictor, as a candidate. */
function sizeCandidate(count: i32): void {
  saveReal();
  run(count, 0, RUN_SIZE, false);
  restoreReal();
  addCandidate(bestRunBytes);
}

/**
 * Codes the block samplesIn[0..count) (count ≤ 2^exponent; a shorter block only at the chunk's end):
 * chooses the predictor and the stages used, writes them and the indices, and appends the reconstruction
 * to samplesOut[0..count). Returns the payload bytes so far, or −1 (drain bytesOut before it fills).
 */
export function swl2EncodeBlock(count: i32): i32 {
  if (active < 0 || count < 1 || count > blockSize) return -1;
  if (eBytes + count * 20 + 4096 > OUT_BYTES) return -1;
  trimHistory();
  // The block's source after the history: analysis reads it from history[historyCount ..].
  for (let i = 0; i < count; i++) store<i64>(history + (<usize>(historyCount + i) << 3), clampLimit(<i64>load<i32>(samplesIn + (<usize>i << 2))));
  analyse(count);
  candidates = 0;
  // Rank the orders by estimate (LPC only), stably.
  let rankedCount = 0;
  for (let k = 0; k < 10; k++) {
    const o = load<i32>(ORDER_LIST + (<usize>k << 2));
    if (o >= orders) continue;
    quantize(o, PRECISION);
    run(count, 0, RUN_ESTIMATE, false);
    const bits = runEstimate + 16 + 15 * <f64>order;
    let j = rankedCount;
    while (j > 0 && load<f64>(ranked + (<usize>(j - 1) << 3)) > bits) {
      store<f64>(ranked + (<usize>j << 3), load<f64>(ranked + (<usize>(j - 1) << 3)));
      store<i32>(rankedOrder + (<usize>j << 2), load<i32>(rankedOrder + (<usize>(j - 1) << 2)));
      j--;
    }
    store<f64>(ranked + (<usize>j << 3), bits);
    store<i32>(rankedOrder + (<usize>j << 2), o);
    rankedCount++;
  }
  // Exact sizes: every ranked order (full) or the two best (fast), each at 15, 12 and 9 bits, then the
  // previous block's predictor (one bit).
  const exact = effort == 1 ? rankedCount : (rankedCount < 2 ? rankedCount : 2);
  for (let k = 0; k < exact; k++) {
    const o = load<i32>(rankedOrder + (<usize>k << 2));
    quantize(o, PRECISION);
    const reused = samePredictor();
    // Coarser versions of the order the quantized predictor kept (0 if its coefficients vanished).
    const kept = order;
    sizeCandidate(count);
    if (kept > 0 && !reused) {
      quantize(kept, 12);
      sizeCandidate(count);
      quantize(kept, 9);
      sizeCandidate(count);
    }
  }
  if (prevOrder > 0) {
    order = prevOrder;
    shift = prevShift;
    memory.copy(coef, prevCoef, 256);
    sizeCandidate(count);
  }
  sortCandidates();
  let chosen = 0;
  bestUses = 0;
  if (useCascade) {
    // The best candidates with the stages, and the best short (order ≤ 2) and medium (3..8) ones.
    const list = memory.data(16, 16); // i32[4]
    let listed = 0;
    for (let i = 0; i < candidates && i < (effort == 1 ? 2 : 1); i++) {
      store<i32>(list + (<usize>listed << 2), i);
      listed++;
    }
    if (effort == 1) {
      for (let range = 0; range < 2; range++) {
        const lo = range == 0 ? 0 : 3;
        const hi = range == 0 ? 2 : 8;
        for (let i = 0; i < candidates; i++) {
          const o = load<i32>(candidateMeta + (<usize>i << 3));
          if (o < lo || o > hi) continue;
          let seen = false;
          for (let j = 0; j < listed; j++) if (load<i32>(list + (<usize>j << 2)) == i) seen = true;
          if (!seen) {
            store<i32>(list + (<usize>listed << 2), i);
            listed++;
          }
          break;
        }
      }
    }
    let best = -1;
    for (let l = 0; l < listed; l++) {
      const index = load<i32>(list + (<usize>l << 2));
      loadCandidate(index);
      saveReal();
      if (step == 1) {
        // Lossless: the reconstruction does not depend on the stages used; one run sizes them all.
        run(count, 0, RUN_ALTERNATIVES, true);
        restoreReal();
        for (let uses = 0; uses < 4; uses++) {
          const bytes = load<i32>(altBytes + (<usize>uses << 2));
          if (best < 0 || bytes < bestBytes) {
            best = index;
            bestBytes = bytes;
            bestUses = uses;
          }
        }
      } else if (effort == 0) {
        // Fast and lossy: one run with the previous block's stages sizes every number of stages on that
        // run's reconstruction (an estimate: the reconstruction depends on the stages used).
        run(count, load<i32>(slotBase(active) + F_USES), RUN_ALTERNATIVES, true);
        restoreReal();
        for (let uses = 0; uses < 4; uses++) {
          const bytes = load<i32>(altBytes + (<usize>uses << 2));
          if (best < 0 || bytes < bestBytes) {
            best = index;
            bestBytes = bytes;
            bestUses = uses;
          }
        }
      } else {
        const first = index == 0 ? 0 : 1;
        for (let uses = first; uses <= 3; uses++) {
          run(count, uses, RUN_SIZE, true);
          restoreReal();
          if (best < 0 || bestRunBytes < bestBytes) {
            best = index;
            bestBytes = bestRunBytes;
            bestUses = uses;
          }
        }
      }
    }
    chosen = best;
  }
  loadCandidate(chosen);
  // The block for real: header, indices, and the state after it.
  putPredictor();
  if (useCascade) putDirect(<u64>bestUses, 2);
  prevOrder = order;
  prevShift = shift;
  memory.copy(prevCoef, coef, 256);
  const start = historyCount;
  store<i32>(slotBase(active) + F_USES, bestUses);
  run(count, bestUses, RUN_REAL, useCascade);
  for (let i = 0; i < count; i++) store<i32>(samplesOut + (<usize>i << 2), <i32>load<i64>(history + (<usize>(start + i) << 3)));
  return eBytes;
}

/** Bytes written to bytesOut so far (drain them, then call swl2EncodeDrained). */
export function swl2EncodeBytes(): i32 {
  return eBytes;
}

/** The caller took bytesOut[0..taken): what follows moves to the front. */
export function swl2EncodeDrained(taken: i32): bool {
  if (taken < 0 || taken > eBytes) return false;
  memory.copy(bytesOut, bytesOut + <usize>taken, <usize>(eBytes - taken));
  eBytes -= taken;
  return true;
}

/** Flushes the coder and ends the chunk; the slot keeps the state a continuation inherits. Returns the bytes left in bytesOut. */
export function swl2EncodeEnd(): i32 {
  if (active < 0) return -1;
  encoderFinish();
  close();
  active = -1;
  return eBytes;
}
