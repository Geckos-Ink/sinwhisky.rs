// FAFT on the GPU: the rectangular transform of the AFT's curve at many frequencies at once, one
// invocation per frequency, from the atoms the WebAssembly port prepared (SinWhisky.atoms(): per interval
// value, slope, two arcs x (a, b, rate, cos rate, sin rate); sinwhisky.rs `CircleCurve::from_angular`
// then `transform(0..N, Rectangular)`). For dense scans and peak seeds: f32, about 1e-6 of the strongest
// value against the f64 kernel; Newton refinement stays in WebAssembly.
//
// Per frequency w: the DC and line terms once, then over every interval its two arcs' E0(+-rate - w)
// (series under |delta| < 0.25, the ratio form above), times the interval's support phase e^{-iwn}. The
// support phase is not chained in f32 (its error would grow with n): w/2pi is split on the CPU into a
// part with at most 11 significant bits, whose product with n < 2^13 is exact in f32, and a small rest,
// so the fraction of a turn is exact to about 1e-7 for up to 4096 intervals. GPU sin/cos are not used
// (WGSL allows them an absolute error of 2^-11): cos w and sin w come from the CPU in f64, the support
// phase goes through a polynomial after reducing the turn to [-1/2, 1/2].

struct Params {
  intervals: u32,
  count: u32,
  stride: u32,
  unused: u32,
};

@group(0) @binding(0) var<storage, read> atoms: array<f32>;
// Two vec4 per frequency: (w, high turns, low turns, 0), (cos w, sin w, 0, 0).
@group(0) @binding(1) var<storage, read> omegas: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> values: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

// E0(delta) = integral_0^1 e^{i delta u} du, given e = e^{i delta}.
fn e0(delta: f32, e: vec2<f32>) -> vec2<f32> {
  if (abs(delta) < 0.25) {
    let d2 = delta * delta;
    let re = 1.0 + d2 * (-1.0 / 6.0 + d2 * (1.0 / 120.0 + d2 * (-1.0 / 5040.0 + d2 * (1.0 / 362880.0))));
    let im = delta * (0.5 + d2 * (-1.0 / 24.0 + d2 * (1.0 / 720.0 + d2 * (-1.0 / 40320.0 + d2 * (1.0 / 3628800.0)))));
    return vec2<f32>(re, im);
  }
  return vec2<f32>(e.y / delta, (1.0 - e.x) / delta);
}

// sin and cos for |x| <= pi: Taylor to x^13 and x^14 at x / 2 (under 1e-8), then the double angle.
fn sincos(x: f32) -> vec2<f32> {
  let h = 0.5 * x;
  let x2 = h * h;
  let s = h * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362880.0
    + x2 * (-1.0 / 39916800.0 + x2 * (1.0 / 6227020800.0)))))));
  let c = 1.0 + x2 * (-0.5 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 + x2 * (1.0 / 40320.0
    + x2 * (-1.0 / 3628800.0 + x2 * (1.0 / 479001600.0 + x2 * (-1.0 / 87178291200.0)))))));
  return vec2<f32>(2.0 * s * c, 1.0 - 2.0 * s * s);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = id.x;
  if (k >= params.count) {
    return;
  }
  let w = omegas[2u * k];
  let trig = omegas[2u * k + 1u];
  let omega = w.x;
  let q = vec2<f32>(trig.x, -trig.y);
  let dc = e0(-omega, q);
  // E1(-w) = integral_0^1 u e^{-iwu} du: the slope's term.
  var line: vec2<f32>;
  if (abs(omega) < 0.25) {
    let d2 = omega * omega;
    line = vec2<f32>(
      0.5 + d2 * (-1.0 / 8.0 + d2 * (1.0 / 144.0 + d2 * (-1.0 / 5760.0 + d2 * (1.0 / 403200.0)))),
      -omega * (1.0 / 3.0 + d2 * (-1.0 / 30.0 + d2 * (1.0 / 840.0 + d2 * (-1.0 / 45360.0 + d2 * (1.0 / 3991680.0))))));
  } else {
    let d = q - dc;
    line = vec2<f32>(d.y / -omega, -d.x / -omega);
  }
  var sum = vec2<f32>(0.0, 0.0);
  for (var n = 0u; n < params.intervals; n = n + 1u) {
    let base = n * params.stride;
    var p = atoms[base] * dc + atoms[base + 1u] * line;
    for (var s = 0u; s < 2u; s = s + 1u) {
      let b = base + 2u + s * 5u;
      let rate = atoms[b + 2u];
      if (rate == 0.0) {
        continue;
      }
      let cr = atoms[b + 3u];
      let sr = atoms[b + 4u];
      let up = e0(rate - omega, vec2<f32>(cr * q.x - sr * q.y, cr * q.y + sr * q.x));
      let down = e0(-rate - omega, vec2<f32>(cr * q.x + sr * q.y, cr * q.y - sr * q.x));
      // cos(rate u) - 1 and sin(rate u) transforms.
      let cosine = 0.5 * (up + down) - dc;
      let sine = vec2<f32>(0.5 * (up.y - down.y), -0.5 * (up.x - down.x));
      p = p + atoms[b] * cosine + atoms[b + 1u] * sine;
    }
    var turns = fract(fract(w.y * f32(n)) + w.z * f32(n));
    if (turns > 0.5) {
      turns = turns - 1.0;
    }
    let support = sincos(-6.283185307179586 * turns).yx;
    sum = sum + vec2<f32>(support.x * p.x - support.y * p.y, support.x * p.y + support.y * p.x);
  }
  values[k] = sum;
}
