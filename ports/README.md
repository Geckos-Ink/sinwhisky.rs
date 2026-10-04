# Ports: WebAssembly SIMD and WebGPU

These are reference ports of three pieces of the crate, written for the web platform:

- **SinWhisky's original-angular resampler:** `resample(samples, ResampleParams::original(zoom))`.
- **The AFT:** `AngularFrame` with `AnalysisParams::original()`.
- **FAFT on the AFT's curve:** `CircleCurve::from_angular`, then `transform(0..N, …)`, `values`, `derivatives` and `peaks`.

The crate stays the authority. Every port is checked against vectors exported from it, and nothing here is a Cargo dependency or changes the library.

| Folder | What | Precision | Runs in |
| --- | --- | --- | --- |
| [`wasm-simd/`](wasm-simd) | AssemblyScript → WebAssembly. A scalar path that follows the crate's arithmetic (FAFT in the earlier moment form), and a SIMD path (`v128`, `f64x2`). | f64 | Node (tested with 24), browsers, any WebAssembly host with SIMD |
| [`webgpu/`](webgpu) | WGSL compute kernels with a JavaScript host. Geometry comes from the WebAssembly port. | f32 | Browsers and Electron with WebGPU, Deno |
| [`fixtures/vectors.json`](fixtures/vectors.json) | Reference outputs from [`examples/port_vectors.rs`](../examples/port_vectors.rs). | f64 | — |

## What each port computes

### WebAssembly (`wasm-simd/assembly/index.ts`, wrapped by `wasm-simd/index.mjs`)

**Circles.** The crate's circle fit is followed step for step:

- `Analysis::new` with the original preset: the triplet's range is the aperture (1 for a constant triplet), and the determinant tolerance is 4e-6;
- `Circle::through`, then `arc_angles`;
- a degenerate triplet becomes the line `y1 + u·(y2 − y0)/2`.

**`resample(samples, zoom, { simd })`.**

- *Scalar path:* the crate's loop. Equal-angle arcs, the window `cos(π·q/(q+1)·u/2)`, and a linear fallback; originals are copied exactly.
- *SIMD path:* both circles covering an interval sit in the two lanes of an `f64x2`. Their equal angle steps advance by one complex rotation per output sample instead of a `sin`.

**AFT.**

- `prepareAft(samples)` builds the AFT's periodic halo and the two halves of every unit interval. A half whose endpoints miss the samples by more than 1e-8 is replaced by the source pair's line. It also builds the FAFT atoms: a value, a slope, and per arc `(a, b, rate, cos rate, sin rate)`.
- `evaluate(position)` is `AngularFrame::evaluate`.
- `integral(omegas)` is `AngularFrame::fourier`.

**FAFT.**

- `values(omegas, { hann, simd })`: normalized values, rectangular or with Hann applied analytically, at any real ω.
- `scan`: a dense grid; the Hann neighbours share one extended grid.
- `derivatives`: exact first and second derivatives in ω.
- `peaks`: scan seeds, a parabolic start, Newton with a bracket, flanks and unconverged seeds dropped, floors, and the merge width.

The rotate-and-add sums are closed forms, with series near resonance. Support phases are re-anchored every 64 intervals, as the crate does. The SIMD path evaluates two frequencies per `f64x2`.

**Not ported:** the uniform-x resampler, `GeometryFrame`, circle-rate seeds (`PeakParams::circle_seeds`), the codec research frame, and the earlier FAFT controls.

**Limits:** static buffers (the `stub` runtime), so memory never grows. At most 4096 input samples, 65536 resampled outputs, zoom up to 255, and 8192 frequencies per call (`values` chunks longer lists).

### WebGPU (`webgpu/`)

- **`sinwhisky.wgsl`:** one invocation per output sample. Each half-arc is read in an anchored form, `y = s[i] + 2R(cos a · sin x cos x − sin a · sin² x)` with `x = t·d/2`, so a flat arc's large radius never subtracts its far centre in f32.
- **`aft.wgsl`:** one invocation per frequency, the rectangular transform from the WebAssembly port's atoms. The support phase is not chained in f32. Instead, ω/2π is split on the CPU into a part with 11 significant bits, which multiplies exactly by n < 2¹³, and a small rest. That keeps the phase's fraction of a turn exact to about 1e-7, for up to 4096 intervals.
- **GPU trigonometry is not used.** WGSL allows `sin` and `cos` an absolute error of 2⁻¹¹, and on Apple Metal 3 they cost the resampler 3e-5. Instead:
  - `sin a`, `cos a`, `cos ω` and `sin ω` come from the CPU in f64;
  - small angles go through Taylor polynomials;
  - the support phase is reduced to ±½ turn first.
- **`index.mjs`:** `SinWhiskyGpu.create()`, `resample(samples, models, zoom)`, `rect(atoms, stride, omegas)`, and `scan(...)`. Newton refinement and peak decisions stay in WebAssembly (f64).

## Measured (2026-10-04, Apple M-series, Node 24 and the Claude desktop app's browser on Metal 3)

These are the largest deviations from sinwhisky.rs over all six fixture cases. Each is relative to that output's peak; peak ω is in radians per sample.

The crate now evaluates arcs away from resonance with the endpoint form (`src/faft.rs`), while both ports keep the moment form with its prepared turns. The two forms are equal in exact arithmetic. Regenerating the vectors moved FAFT values by at most 1.2e-15 relative and peak entries by 5.7e-14. The WebAssembly columns were re-measured against the new vectors; the WebGPU column dates from the earlier vectors, and f32 rounding is far larger than that change.

| Output | WebAssembly scalar | WebAssembly SIMD | WebGPU (f32) |
| --- | --- | --- | --- |
| resample, zoom 1, 3, 7 | 3.8e-14 | 3.8e-14 | 2.4e-7 |
| AFT curve (`evaluate`) | 2.6e-14 | — | — |
| `AngularFrame::fourier` | 1.6e-14 | 1.6e-14 | 7.8e-7 (rect, normalized) |
| FAFT Hann values | 2.0e-15 | 2.1e-15 | 1.1e-6 (Hann scan vs WebAssembly) |
| peak frequencies | 2.0e-15 | — | — |

Timings are medians on a 4096-sample signal:

| Work | WebAssembly scalar | WebAssembly SIMD | WebGPU |
| --- | --- | --- | --- |
| resample, zoom 1 | 0.42 ms | 0.46 ms (0.9×) | — |
| resample, zoom 3 | 0.67–0.75 ms | 0.48–0.51 ms (1.4–1.5×) | — |
| resample, zoom 15 | 2.5 ms | 0.62–0.70 ms (3.6–4.0×) | 1.1–1.8 ms (including transfers) |
| FAFT, 4095 frequencies (rectangular) | 193 ms | 121 ms (1.6×) | 5.4–5.7 ms (22× SIMD) |

**Reading the numbers:**

- The GPU pays off on dense FAFT scans.
- For the resampler, SIMD in WebAssembly already beats the GPU, whose time is mostly upload and readback.
- At zoom 1 the SIMD resampler is slightly slower than the scalar one: per interval it pays for two `sincos` against a single `sin`.

## Build, test, run

The checked-in `wasm-simd/build/sinwhisky.wasm` lets the ports run without a toolchain. Rebuild it after editing `assembly/index.ts`.

Install the AssemblyScript compiler (the only dependency, a development one):

```bash
cd ports/wasm-simd && npm install
```

Compile the module to `build/sinwhisky.wasm`:

```bash
npm run build
```

Run the parity tests against `../fixtures/vectors.json`:

```bash
npm test
```

Print the agreement and the scalar and SIMD timings:

```bash
npm run bench
```

Regenerate the reference vectors from the crate (from the repository root). The output is deterministic, and the file should only change with a reviewed change of the crate's behaviour:

```bash
cargo run --offline --quiet --example port_vectors > ports/fixtures/vectors.json
```

The WebGPU check runs in a browser. Start a loopback server rooted at `ports/`:

```bash
node ports/serve.mjs
```

Then open `http://127.0.0.1:8791/webgpu/test.html`. The page compares every kernel with the WebAssembly port and the Rust vectors, times the three back ends, and puts its verdict in the page title (`pass` / `fail`) and `window.sinwhiskyGpuReport`.

## Using the ports

```js
import { SinWhisky } from './wasm-simd/index.mjs';
import { SinWhiskyGpu } from './webgpu/index.mjs';

const sw = await SinWhisky.load(); // Node; in a browser: SinWhisky.instantiate(await (await fetch(url)).arrayBuffer())
const up = sw.resample(samples, 3, { simd: true }); // 4x, originals kept

sw.prepareAft(frame); // up to 4096 samples, periodic
const spectrum = sw.values([0.31, 0.3137, 4.0], { hann: true, simd: true }); // (re, im) pairs, any real ω
const peaks = sw.peaks({ upper: Math.PI }); // [{ omega, re, im, magnitude, curvature }]

const gpu = await SinWhiskyGpu.create(); // null without WebGPU
const dense = await gpu.scan(sw.atoms(), sw.atomStride, 0, sw.resolution / 4, 2049, 4); // Hann magnitudes
const zoomed = await gpu.resample(samples, sw.models(samples), 15);
```
