// SinWhisky's original-angular resampler on the GPU: one invocation per output sample.
// sinwhisky.rs `resample(samples, ResampleParams::original(zoom))`, in f32.
//
// Each interval [i, i + 1] is covered by the right half of the circle at sample i (its angles 1 -> 2) and
// the left half of the circle at sample i + 1 (angles 0 -> 1). Both start at sample i, so in f32 each is
// read in the anchored form y = s[i] + R (sin(a + t d) - sin a) = s[i] + 2R (cos a sin x cos x - sin a sin^2 x),
// x = t d / 2: a flat arc's large radius never subtracts its far centre. GPU sin/cos are not used (WGSL
// allows them an absolute error of 2^-11, and Metal's reach ~5e-5): sin a and cos a come from the CPU in
// f64, and the small angles go through polynomials. Degenerate triplets are lines; the blend uses the
// original window cos(pi q/(q + 1) u / 2); without a usable model the samples' line is used.

struct Params {
  length: u32,  // input samples
  q: u32,       // zoom + 1
  total: u32,   // output samples: (length - 1) q + 1
  unused: u32,
};

// Per model j (the circle at sample j + 1), prepared on the CPU in f64 (ports/webgpu/index.mjs): flag
// (1 circle, 0 line), R, sin a0, cos a0, d01 = a1 - a0, sin a1, cos a1, d12 = a2 - a1, line value, slope.
@group(0) @binding(0) var<storage, read> samples: array<f32>;
@group(0) @binding(1) var<storage, read> models: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const PI: f32 = 3.141592653589793;
const STRIDE: u32 = 10u;

fn field(j: u32, f: u32) -> f32 {
  return models[j * STRIDE + f];
}

// sin and cos for |x| <= pi/2 (Taylor to x^13 and x^14: under 1e-8).
fn sincos_small(x: f32) -> vec2<f32> {
  let x2 = x * x;
  let s = x * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362880.0
    + x2 * (-1.0 / 39916800.0 + x2 * (1.0 / 6227020800.0)))))));
  let c = 1.0 + x2 * (-0.5 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 + x2 * (1.0 / 40320.0
    + x2 * (-1.0 / 3628800.0 + x2 * (1.0 / 479001600.0 + x2 * (-1.0 / 87178291200.0)))))));
  return vec2<f32>(s, c);
}

// sin and cos for |x| <= pi, by the double angle of x / 2.
fn sincos(x: f32) -> vec2<f32> {
  let h = sincos_small(0.5 * x);
  return vec2<f32>(2.0 * h.x * h.y, 1.0 - 2.0 * h.x * h.x);
}

// A model's value at local u = position - anchor in (-1, 1), anchored at sample i = floor(position).
fn predict(j: u32, u: f32, t: f32, start: f32) -> f32 {
  if (field(j, 0u) == 0.0) {
    return field(j, 8u) + u * field(j, 9u);
  }
  // u < 0: the left half (angles 0 -> 1); otherwise the right half (angles 1 -> 2).
  var at = 5u;
  if (u < 0.0) {
    at = 2u;
  }
  let x = sincos(0.5 * t * field(j, at + 2u));
  return start + 2.0 * field(j, 1u) * (field(j, at + 1u) * x.x * x.y - field(j, at) * x.x * x.x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let m = id.x;
  if (m >= params.total) {
    return;
  }
  let q = params.q;
  let i = m / q;
  let k = m % q;
  if (k == 0u) {
    output[m] = samples[i];
    return;
  }
  let qf = f32(q);
  let t = f32(k) / qf;
  let left = samples[i];
  let right = samples[i + 1u];
  let c = PI * (qf / (qf + 1.0)) * 0.5;
  var sumW: f32 = 0.0;
  var sumY: f32 = 0.0;
  // Anchor i (model i - 1, u = t) when 1 <= i <= length - 2.
  if (i >= 1u && i + 2u <= params.length) {
    let w = sincos_small(c * t).y;
    sumW = sumW + w;
    sumY = sumY + w * predict(i - 1u, t, t, left);
  }
  // Anchor i + 1 (model i, u = t - 1) when i + 1 <= length - 2.
  if (i + 3u <= params.length) {
    let w = sincos_small(c * (t - 1.0)).y;
    sumW = sumW + w;
    sumY = sumY + w * predict(i, t - 1.0, t, left);
  }
  if (sumW > 0.0) {
    output[m] = sumY / sumW;
  } else {
    output[m] = left + (right - left) * t;
  }
}
