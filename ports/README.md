# Ports: WebAssembly SIMD and WebGPU

These are reference ports of the crate, written for the web platform:

- **SinWhisky's original-angular resampler:** `resample(samples, ResampleParams::original(zoom))`.
- **The AFT:** `AngularFrame` with `AnalysisParams::original()`.
- **FAFT on the AFT's curve:** `CircleCurve::from_angular`, then `transform(0..N, …)`, `values`, `derivatives` and `peaks`.
- **Endpoint FAFT and joint tone fitting:** the finite-window triplet-circle preset transposed from `appx/dsp`, with scalar f64, two f64 SIMD lanes and four f32 lanes.
- **Component equalizer:** streaming band division and gains, and batched fits with WebAssembly or WebGPU values.
- **SWL2 version 4 correction codec:** LPC plus the integer SIMD LMS cascade, segments, key frames, compression and decompression. Gradient/anchor layers and parametric streams are rejected.

The crate stays the authority. The legacy APIs are checked against exported Rust vectors; the new curve and fits against independent quadrature, and SWL2 against frozen Rust streams in both runtimes. Nothing here is a Cargo dependency or changes the library.

| Folder | What | Precision | Runs in |
| --- | --- | --- | --- |
| [`wasm-simd/`](wasm-simd) | AssemblyScript → WebAssembly. Legacy AFT, endpoint FAFT and tone fit, component EQ, SWL2 correction. | Legacy f64; new FAFT f32 input/f64 geometry and queries, optional f32 lanes; EQ f32 output; codec integers | Node (tested with 24), browsers, any WebAssembly host with SIMD |
| [`webgpu/`](webgpu) | WGSL compute kernels with a JavaScript host. Geometry comes from the WebAssembly port. | f32 | Browsers and Electron with WebGPU, Deno |
| [`fixtures/vectors.json`](fixtures/vectors.json) | Reference outputs from [`examples/port_vectors.rs`](../examples/port_vectors.rs). | f64 | — |
| [`fixtures/swl2/`](fixtures/swl2) | Frozen synthetic Rust-encoded streams, rechecked by [`tests/port_swl2.rs`](../tests/port_swl2.rs). | Integer | — |

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

**Not ported:** the uniform-x resampler, `GeometryFrame`, circle-rate seeds (`PeakParams::circle_seeds`), kernel ranges, pursuit and tracker, the codec research frame, and the earlier FAFT controls. The new `faftFit` provides the joint/banded fit and refinement.

**Limits:** static buffers (the `stub` runtime), so memory never grows. At most 4096 input samples, 65536 resampled outputs, zoom up to 255, and 8192 frequencies per call (`values` chunks longer lists).

### WebGPU (`webgpu/`)

- **`sinwhisky.wgsl`:** one invocation per output sample. Each half-arc is read in an anchored form, `y = s[i] + 2R(cos a · sin x cos x − sin a · sin² x)` with `x = t·d/2`, so a flat arc's large radius never subtracts its far centre in f32.
- **`aft.wgsl`:** one invocation per frequency, the rectangular transform from the WebAssembly port's atoms. The support phase is not chained in f32. Instead, ω/2π is split on the CPU into a part with 11 significant bits, which multiplies exactly by n < 2¹³, and a small rest. That keeps the phase's fraction of a turn exact to about 1e-7, for up to 4096 intervals.
- **GPU trigonometry is not used.** WGSL allows `sin` and `cos` an absolute error of 2⁻¹¹, and on Apple Metal 3 they cost the resampler 3e-5. Instead:
  - `sin a`, `cos a`, `cos ω` and `sin ω` come from the CPU in f64;
  - small angles go through Taylor polynomials;
  - the support phase is reduced to ±½ turn first.
- **`index.mjs`:** `SinWhiskyGpu.create()`, `resample(samples, models, zoom)`, `rect(atoms, stride, omegas)`, and `scan(...)`. Newton refinement and peak decisions stay in WebAssembly (f64).

### Transposed `appx` paths (2026-10-05)

Source: `appx/dsp` at commit `ec84edcd331af8810be9df42cf5d16f3a348d6d5`.
The AssemblyScript modules `faft.ts`, `tones.ts`, `eq.ts`, `swl2.ts`, and EQ's
`fft.ts`/`db.ts` dependency are copied with provenance headers. The JavaScript
wrapper is the relevant portion of `appx/dsp/src/index.ts`, with types erased;
`webgpu/faft.mjs` transposes its GPU host and embedded endpoint WGSL. No `appx`
checkout is needed to run or rebuild these ports. SWC, metering, correlation and
the other AppX product kernels are outside this transposition.

`SinWhisky` now inherits the new methods from `SignalProcessor`; it still exposes
all existing methods. `SignalProcessor` is also exported for hosts needing only
the new paths, with `instantiate` and synchronous `fromModule` constructors.

**Two distinct curve presets.** `prepareAft` retains the periodic `[0,N]` AFT,
f64 inputs, original determinant rejection and 12-field moment atoms. Its
`values`, `scan`, `derivatives`, `peaks` and `SinWhiskyGpu` stay compatible.
`prepareFaft` rounds input to f32, fits normalized triplets with `appx`'s 1e-9
determinant rejection, and prepares the finite `[0,N-1]` curve: one circle at
each edge, two averaged inside, with source-pair line fallbacks. It has separate
buffers and methods; it is not the hierarchical Rust `GeometryFrame` preset.

**Updated FAFT.** `faftValues(omegas, { hann, simd, fast })`, `faftScan`,
`faftDerivatives`, `faftPeaks`, `faftKernel`, `faftFit` and `faftToneGain` operate
on that finite curve. Values and derivatives use the endpoint kernel outside
`|omega ± rate| < 0.5`, and the moment form inside. `fast` uses four f32 lanes;
the new GPU kernel uses a 0.25 guard and compensated sums. `faftAtoms` has stride
20: value, slope, two arcs each storing `(a,b,rate,cos,sin,start_slope,end_slope,
end_change,rate²a)`. Use these atoms with **`FaftGpu`**, not `SinWhiskyGpu`.

`faftFit(omegas, options)` returns tone components (frequency, complex fitted
and observed values, amplitude, phase about the window centre, significance),
degeneracy and truncation counts. Options: `hann` (default true), `reachBins`
(16; Infinity is dense), `refine` (0), `tolerance` (1e-4 resolutions), `simd`
(true), `fast` (false), and `energy` (false). With `energy`, the result includes
weighted curve energy, residual energy and `sqrt(residual/W)` bound. This energy
uses quadrature; transform values and Gram entries are analytic.

The finite FAFT accepts 2–4096 samples, up to 8192 f64 queries per call, and 256
tones/peaks. Fast values accept 2048 rectangular or 682 Hann queries. New result
views (`faftValues`, `faftAtoms`, scans and derivatives) must be copied before
another FAFT/EQ call; legacy result methods still return copies.

**Equalizer.** `configureEqualizer(options)` returns fixed input/output and
`bandTracks` views, band count, latency and truncation dB. Set gains with
`setEqualizerGains`; write input then `processEqualizer(channel, count, true)`
to obtain each band's track. Up to two independently maintained channels and
4096 samples per block; gains ramp over a block. `equalize` returns a whole
signal and its bands with latency removed. Options:

| Option | Default / range |
| --- | --- |
| `edges` | Required; at most 5 increasing rad/sample edges in (0, pi] |
| `channels` | 1; 1–2 for streaming (whole-signal helpers use one) |
| `window` | 2048; power of two 256–2048; hop is half a window |
| `upper` | 2*pi*6000/44100 rad/sample |
| `relativeFloor`, `maxTones` | 0.01, 64; at most 256, 0 disables tones |
| `reachBins`, `refine` | 16, 1; refinement 0–8 |
| `crossoverHalfWidth` | 0 (whole-tone assignment); otherwise raised cosine |
| `order`, `lookahead` | 4 (1–8), 2 (0–3; 0 is causal) |

It preserves the input through the final residual band; unity gains reproduce
the f32 input exactly after `(lookahead+2)*window/2+2` samples of latency. This
is AppX's streaming adaptation: triplet curves, Hann FFT seeds, differences of
values for refinement, and a truncated backward filter pass. It is not a
numerically identical replacement for Rust's buffered `bands::equalize`.

`equalizeBatch(signal, options, gains, evaluator?)` gathers every window's fit
requests round by round. The default WebAssembly evaluator reproduces the
stream bit for bit; `FaftGpu.batchEvaluator()` uploads all f32 atoms once and
evaluates each round in one dispatch. Fitting and synthesis stay in WebAssembly.
Results include windows, fitted windows, rounds, values and timing statistics.
Always destroy the evaluator before destroying the GPU. Use a separate instance
for concurrent work; an awaited batch owns that instance's FAFT/EQ state. GPU
batch length is also bounded by the device's storage-buffer limits.

**SWL2.** `encodeSwl2(channels, options)` takes 1–6 equally sized nonempty integer
channels, `|sample| < 2^23`, and returns `bytes`, `decoded`, and per-segment
key/stereo/block-exponent/byte metadata. `decodeSwl2(bytes, first=0)` returns
1–8 integer channels; `first` must select a key segment. Options: `tolerance`
(0, lossless; up to 2^20), `segmentExponent` (0, one segment; otherwise 12–24),
`keyInterval` (0, only first key), `cascade` (true), `effort` ('full' or 'fast'),
`blockSearch` (true), `blockExponent` (13, or 12 without cascade; 8–15 when
search disabled), `stereoSearch` (true, lossless only). Full effort is the Rust
correction-only search; fast effort sizes fewer LPC candidates. Layers and
parametric streams are unsupported. Invalid numeric inputs are rejected before
integer coercion; reserved header flags and fractional controls are rejected.
The wrapper allocates and is unsuitable for audio callbacks; its integer SIMD
kernel keeps static buffers and continuation state.

The three SWL2 fixtures are synthetic, copied unchanged from AppX's Rust fixture
set. Node tests encode the same bytes and test corruption/key starts; Rust tests
encode/decode them with this checkout. Never regenerate these fixtures from the
port or to resolve a failing parity test.

**Validated here, 2026-10-05:** 50 Node tests, plus the browser page on Electron
44 / Apple Metal 3. Over the browser's finite-curve tone, flat-arc and line cases,
rectangular FAFT's largest relative deviation was 4.0e-6 and Hann scans 6.2e-7.
The 20,000-sample synthetic GPU equalizer differed from the stream by 5.2e-5 of
the output peak; the band sum differed from input by 3.1e-8 absolute. The default
batch was bit-identical to the stream, and the browser codec round trip matched
the encoder reconstruction. These are numerical checks on these cases, not
listening results or general precision bounds.

## Measured (2026-10-04, Apple M-series, Node 24 and the Claude desktop app's browser on Metal 3)

These are the largest deviations from sinwhisky.rs over all six fixture cases. Each is relative to that output's peak; peak ω is in radians per sample.

This table describes the legacy periodic APIs, which retain the moment form with prepared turns. The crate evaluates arcs away from resonance with the endpoint form (`src/faft.rs`); the transposed finite-window APIs above also use it. The two forms are equal in exact arithmetic. Regenerating the vectors moved FAFT values by at most 1.2e-15 relative and peak entries by 5.7e-14. The WebAssembly columns were re-measured against the new vectors; the WebGPU column dates from the earlier vectors, and f32 rounding is far larger than that change.

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

The checked-in `wasm-simd/build/sinwhisky.wasm` lets the ports run without a toolchain. Rebuild it after editing any AssemblyScript module.

Install the AssemblyScript compiler (the only dependency, a development one):

```bash
cd ports/wasm-simd && npm install
```

Compile the module to `build/sinwhisky.wasm`:

```bash
npm run build
```

Run the legacy vector checks and the finite FAFT, tone fit, EQ and SWL2 suites:

```bash
npm test
```

From the crate root, check the SWL2 fixtures against the current Rust coder:

```bash
cargo test --offline --test port_swl2
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

```js
import { FaftGpu } from './webgpu/index.mjs';

sw.prepareFaft(frame); // finite curve, duration frame.length - 1
const fitted = sw.faftFit([0.1, 0.2], { refine: 3, energy: true });
const eq = sw.equalize(samples, { edges: [0.1, 0.4] }, [1, 0.5, 0]);
const encoded = sw.encodeSwl2([pcmLeft, pcmRight], { tolerance: 16 });
const decoded = sw.decodeSwl2(encoded.bytes); // source PCM not required

const endpointGpu = await FaftGpu.create();
if (endpointGpu) {
  const evaluate = endpointGpu.batchEvaluator();
  try {
    const batch = await sw.equalizeBatch(samples, { edges: [0.1, 0.4] }, [1, 0.5, 0], evaluate);
  } finally {
    evaluate.destroy();
    endpointGpu.destroy();
  }
}
```
