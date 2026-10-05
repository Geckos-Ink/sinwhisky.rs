// Transposed from appx/dsp at ec84edcd331af8810be9df42cf5d16f3a348d6d5 (2026-10-05).
// An independent double-precision reference for the FAFT kernels: SinWhisky's circle curve built here as
// assembly/faft.ts builds it (sinwhisky.rs `arc_angles`), integrated by Gauss–Legendre quadrature, and a
// deterministic test signal. Shared by faft.test.ts, tones.test.ts and eq.test.ts.



/** The circle through (0, y0), (1, y1), (2, y2), x scaled by the range, angles unwrapped through the middle. */
export function circle(y0, y1, y2) {
  const apr = Math.max(Math.abs(y0 - y1), Math.abs(y1 - y2), Math.abs(y0 - y2));
  if (!(apr > 0)) return undefined;
  const [a, b, c] = [y0 / apr, y1 / apr, y2 / apr];
  const det = 2 * ((b - c) * -1 + (a - b));
  if (Math.abs(det) < 1e-9) return undefined;
  const bc = a * a - 1 - b * b;
  const cd = 1 + b * b - 4 - c * c;
  const ox = (bc * (b - c) - cd * (a - b)) / det;
  const oy = (-cd + bc) / det;
  const r = Math.hypot(ox, a - oy);
  const angles = [Math.atan2(a - oy, -ox), Math.atan2(b - oy, 1 - ox), Math.atan2(c - oy, 2 - ox)];
  const tau = 2 * Math.PI;
  const wrap = (x) => ((x % tau) + tau) % tau;
  const direction = wrap(angles[1] - angles[0]) <= wrap(angles[2] - angles[0]) ? 1 : -1;
  const middle = angles[0] + direction * wrap(direction * (angles[1] - angles[0]));
  const last = middle + direction * wrap(direction * (angles[2] - angles[1]));
  const cy = oy * apr;
  const radius = r * apr;
  const bound = 1e-8 * Math.max(1, Math.abs(y0), Math.abs(y1), Math.abs(y2));
  const ok = [[angles[0], y0], [middle, y1], [last, y2]].every(([angle, y]) => Math.abs(cy + radius * Math.sin(angle) - y) <= bound);
  return ok ? { cy, radius, a0: angles[0], a1: middle, a2: last } : undefined;
}

/** Each interval's halves, as the kernel builds them: the right half of circle n and the left of circle n + 1. */
export function halves(samples) {
  const y = Array.from(samples);
  const circles = y.map((_, k) => (k >= 1 && k <= y.length - 2 ? circle(y[k - 1], y[k], y[k + 1]) : undefined));
  return y.slice(0, -1).map((_, n) => {
    const list = [];
    if (n >= 1 && n <= y.length - 2) list.push(['right', circles[n]]);
    if (n + 1 <= y.length - 2) list.push(['left', circles[n + 1]]);
    if (list.length === 0) return [{ kind: 'line', w: 1 }];
    return list.map(([side, c]) => c === undefined
      ? { kind: 'line', w: 1 / list.length }
      : { kind: 'arc', from: side === 'right' ? c.a1 : c.a0, rate: side === 'right' ? c.a2 - c.a1 : c.a1 - c.a0, cy: c.cy, r: c.radius, w: 1 / list.length });
  });
}

export function evaluate(samples, curve, position) {
  const n = Math.min(curve.length - 1, Math.floor(position));
  const u = position - n;
  return curve[n] .reduce((sum, half) => sum + half.w * (half.kind === 'line'
    ? samples[n] + (samples[n + 1] - samples[n]) * u
    : half.cy + half.r * Math.sin(half.from + half.rate * u)), 0);
}

/** 16-point Gauss–Legendre nodes and weights on [0, 1] (Newton on the Legendre polynomial). */
export const GAUSS = (() => {
  const n = 16;
  const nodes = [];
  for (let i = 1; i <= n; i++) {
    let x = Math.cos((Math.PI * (i - 0.25)) / (n + 0.5));
    let derivative = 0;
    for (let iteration = 0; iteration < 100; iteration++) {
      let p0 = 1;
      let p1 = x;
      for (let k = 2; k <= n; k++) [p0, p1] = [p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k];
      derivative = (n * (x * p1 - p0)) / (x * x - 1);
      const dx = p1 / derivative;
      x -= dx;
      if (Math.abs(dx) < 1e-16) break;
    }
    nodes.push([(1 - x) / 2, 1 / ((1 - x * x) * derivative * derivative)]);
  }
  return nodes;
})();

/** ∫ y(t) w(t) e^{−iωt} dt over [0, T), normalized by the weight sum, by quadrature of the reference curve. */
export function quadrature(samples, curve, omega, hann) {
  const duration = curve.length;
  let re = 0;
  let im = 0;
  for (let n = 0; n < duration; n++) {
    for (const [u, weight] of GAUSS) {
      const t = n + u;
      const w = hann ? 0.5 - 0.5 * Math.cos((2 * Math.PI * t) / duration) : 1;
      const value = evaluate(samples, curve, t) * w * weight;
      re += value * Math.cos(omega * t);
      im -= value * Math.sin(omega * t);
    }
  }
  const norm = hann ? duration / 2 : duration;
  return [re / norm, im / norm];
}

/** A deterministic test signal: tones (amplitude, ω, phase) plus uniform noise of `noise` amplitude. */
export function signal(length, tones, noise = 0, seed = 7) {
  let state = seed;
  const random = () => {
    state = (state * 1_103_515_245 + 12_345) % 2_147_483_648;
    return state / 2_147_483_648 - 0.5;
  };
  return Float32Array.from({ length }, (_, k) => tones.reduce((sum, [a, w, p]) => sum + a * Math.sin(w * k + p), 0) + 2 * noise * random());
}
