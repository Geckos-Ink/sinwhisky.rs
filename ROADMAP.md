# Research roadmap

## Implemented foundation

- Standalone Rust library and numeric CLI; standard library only.
- Reference ports (outside the crate): WebAssembly with SIMD and WebGPU for the
  original-angular resampler, the AFT, and FAFT on its curve, checked against
  Rust vectors (`ports/`).
- Reusable triplet analysis with circle/line models, local coordinates, aperture,
  branches, and unwrapped angles.
- Separate uniform-x compatibility and repaired original angular interpolation.
- Frozen-reference, analytic, error-bound, coverage, CLI, and live cross-validation.
- In-memory mono segment experiment that reuses fitted circles and reports error.
- Optional local audio comparisons with anti-alias decimation and bitrate controls.
- `test_music.sh`: configurable encoded-bitrate reduction, block-based Rust
  sample-rate upscaling, and WAV/FLAC output for full tracks or excerpts.
- Rust continuous circle transforms (analytic angular, numerical uniform) and
  descriptor-based discrete Fourier bins, with independent numerical tests.
- Optional NumPy circle/subarc/Fourier lab with versioned, bounded SWR1 experiment
  streams, residual quantization, raw fallback, byte/error measurements, and
  raw/delta/line/FLAC controls. See [method and results](docs/COMPRESSION.md).
- Prepared periodic `AngularFrame` with analytic AFT bins and fractional
  evaluation, independent of the existing resampler's blending policy.
- AFT/FFT benchmarks on 513 Beethoven windows, numeric integration checks,
  frequency/phase differences, scaling plots, spectrograms, and animated spectra.
- FFT phasor control (first "FAFT" prototype, now `phasor_fft`): radix-2 forward/inverse transforms, dyadic
  constant/line/circle model selection, zero-curvature line limits, per-circle
  precision, and verified RMS budgets. Integral AFT retained as a reference preset.
- Phasor comparison across 513 Beethoven windows and 3,078 checked reconstructions,
  root/tree controls, runtime scaling, precision graphs, and an epicycle GIF.
- Geometry-first sampled spectrum (earlier FAFT attempt): circumcircles from waveform points, optimized coverage,
  dynamic geometry precision, source peak-error bounds, direct finite spectral
  sums, batched roots, cropped Hann queries, and zoomed model spectra.
- Nine point/phase/precision policies across 63 Beethoven windows, two fitting
  attempts, independent spectrum checks, omitted-point diagnostics, 80/160 kbps
  MP3 controls, plots, and an animation of an actual waveform circle.

- Hierarchical source-informed circles on arbitrary integer supports, checked
  child merging/refitting, and finite inverse rotational sums with prepared roots.
- Adaptive relevant-frequency peaks with depth/width/query limits, uncertainty
  bounds for coarse/fine predictors, reported heuristic pruning, and peak merging.
- 21-window Beethoven hierarchy/adaptive survey with two retrieval kernels,
  off-bin and unresolved near-tone controls, precision/depth/zoom trials, plots/GIF.

- **FAFT** ([definition](docs/FAFT.md)): exact closed-form Fourier integral of the
  circle-determined curve at any real frequency, with no zoom, sampling, FFT or
  quadrature. Rotate-and-add queries with prepared rotation turns, exact
  continuous Hann, analytic derivatives, duration-spaced scan plus Newton peaks,
  merging/floors, and circle-rate references. Same-curve agreement with integral
  AFT is ~1e-13; independent quadrature ~1e-14. Measured on 63 Beethoven windows
  against the earlier sampled kernels, adaptive search and FFT controls. Zoom
  convergence and the Poisson image identity are tested.
- FAFT endpoint kernel: arcs away from `omega = ±rate` use the endpoint form of
  their deviation (prepared end slopes, end change and `rate^2*a`; one real
  division per arc), and the derivatives use its differentiated quotient. Flat
  arcs keep relative accuracy (radius 1e8 within 1e-12 of quadrature). On
  synthetic 1,024-sample frames, queries were 1.11–1.20× faster and peak lists
  1.15× faster than the moment form, with values within 5e-15.

- FAFT peak selection (`faft::pursuit`): the window kernel and its widths,
  measured peak ranges with shared-side flags, a joint closed-form tone fit
  (Gram from the kernel, FAFT values as projections), residual energy with a
  Cauchy-Schwarz certificate at every frequency, significance pruning, joint
  Gauss-Newton frequency polishing per coupled cluster, residual-seeded pursuit
  from one scan, and `PeakTracker` for consecutive windows (scan-free polish,
  novelty and interval rescans, hysteresis, stable identities). On a synthetic
  test with known partials: recall 89% to 100%, no spurious peaks, p95 height
  error 4.7 to 0.6 dB; tracking 3.5 ms median per window against 6.4 ms for the
  scan (`docs/experiments/faft-pursuit-synthetic-20261004.json`).

## Current focus: FAFT of determined circles

1. Rotate and add in time: slide windows by updating only rotations that enter
   or leave, for fixed sets of continuous frequencies and the three Hann shifts.
   Measure drift and re-anchoring against full recomputation.
2. Continuous joins in circle determination. The tolerance-bounded value jumps
   between rounded circles cause the 1/z tail of the zoom difference and
   1/omega spectral tails. Test join constraints without loosening source checks.
3. Peak selection follow-up. Measure `pursue` and `PeakTracker` on the
   Beethoven windows. The certificate `sqrt(residual/W)` is loose for broadband
   residuals; look for tighter, still rigorous, bounds. Add chirp atoms (linear
   frequency change) for glides and onsets, which the tone model currently spends
   components on. Add a first-derivative-only kernel: Gauss-Newton needs `F` and
   `F'`, and a Hann derivative evaluation (8x a value) dominates pursuit cost.
   Keep comparing matched FFT/Goertzel and padded-FFT costs.
4. Circle determination whose angular rates carry physical frequency. The current
   rates are geometric (aperture-dependent); circle-seeded search found only
   41-43% of the peaks and was slower than the scan. A rate is twice an inscribed
   angle of the fitting triangle over the side's duration. At a symmetric crest of
   a tone it matches the tone only when the aperture equals the tone's largest
   central difference (tested). Find out whether any aperture rule works away from crests.
5. Endpoint kernel follow-up. The crate now uses it (see Implemented). Port it
   to `ports/wasm-simd` (scalar and SIMD) and `ports/webgpu`, which still use the
   moment form, and re-measure the Beethoven windows, whose published timings
   predate it.
6. Dense spectra are O(rotations × frequencies). Only consider butterfly/NUFFT-style
   interpolation with a stated error bound and measured crossover size; no FFT
   speed claim exists today.
7. Keep the sampled geometry/adaptive spectra, integral AFT and FFT phasor control
   as references. Their targets differ; zooming a predictor adds no observations.

## Next: equalization and compression (draft paper)

[PAPER_EQ_COMPRESSION.md](docs/PAPER_EQ_COMPRESSION.md) derives these methods;
its identities are checked by `tools/eq_compression_checks.py`.

Implemented (experimental, paper Section 6.1): `Faft::fit_banded` (two-way banded
solve, selected inversion) and `fit_refined`; exact analog filtering of circle
curves (`analog`: spiral pieces, Butterworth, zero phase); the component divider,
equalizer and extrusion (`bands`); the SWL1 layered coder (`layered`); real-file
runs via `test_music.sh quick|divide|eq|extrude|compress`. First measurement
(`docs/experiments/eq-compression-beethoven-20261005.json`, one 5 s excerpt): the
tonal layer removes 11-14 dB of residual energy, but the correction layer's
polynomial predictor already handles the low band; tracks up to 6 kHz save 2.4%
losslessly, tracks up to 2 kHz cost 3%, and with a tolerance tracks cost bytes.
Both are far above FLAC (37.25%).

1. Stronger correction predictor (adaptive LPC, as FLAC) for both the control and
   the layered stream; only then judge the tonal layer.
2. Tonal layer where linear prediction fails: wider/mid bands, breakpoint
   thinning, Hermite gradients (paper 4.2-4.3), coarser precision against tolerance.
3. Band layer at each band's own rate (paper 4.5), and per-band windows (paper 2.5).
4. Divider selectivity on synthetic tones against spectral masking and a
   Linkwitz-Riley crossover; residual curve as circles instead of lines.
5. Banded fit against the clustered pursuit (agreement, cost vs chain length);
   a banded joint Gauss-Newton to replace the coordinate refinement.
6. Faster analysis: the fast path costs ~30 ms per 2048 window on real music;
   the tracker 100-330 ms (it rescans almost every window).
7. Rotations per second of determined curves against band edge.
8. More excerpts and recordings before any compression claim; no listening claim.

## Earlier geometry-first focus (controls)

1. Prune point/phase candidates using justified prediction bounds; fitting is now
   the dominant cost (1-3 ms per 1,024-sample window, vs microseconds per FAFT
   query). Measure reuse across neighboring time windows.
2. Improve greedy hierarchical splitting/merging with shared circle parameters,
   source-checked parent/child deltas, residual circles, and reusable model banks.
   Variable supports and checked adjacent refitting now exist; no global optimum
   is established.
3. Define an objective that also constrains spectral phase/band errors: a source
   sample-error limit alone allows changes in quiet spectral bins.
4. Test adaptive per-field precision, a bounded wire format with measured bytes,
   continuity/overlap, broader audio, and prediction from truly omitted points.

## Next: establish quality and cost

1. Use varied music, speech, transients, silence, and synthetic signals; compare
   both modes with linear and bandlimited interpolation. Add blind listening
   trials and spectral metrics. Short excerpts from one recording are not enough
   to rank quality.
2. Measure fitting/synthesis/segment-search CPU and memory before optimizing.
3. Evaluate circle candidate reuse over longer ranges, merging/refitting, and
   continuity at segment boundaries. Keep changes separate from compatibility mode.

## Further codec experiments

1. Encode the rounded geometry in an actual bounded wire format and **remeasure
   decoded error**. Geometry-first FAFT already tests parameter rounding; extend
   that study to the uniform predictor's branch/extent changes. Compare coarse
   parent fits with local-triplet extrapolation, keeping refitting explicit.
2. Improve the experimental residual coder and global byte search; measure larger
   blocks, delta-coded child parameters, and sparse Fourier residual budgets.
   Residual/raw trials exist; they do not yet establish a circle-specific advantage.
3. Design a stable format only after the tool-local SWR1 trials justify it. SWR1
   has explicit framing, limits, and malformed-input tests; add fuzzing and broader
   cross-platform round trips before treating any parser as safe for untrusted data.
4. Investigate predictive/delta parameter coding, entropy coding, stereo coupling,
   perceptual weighting, block boundaries, seekability, and streaming latency.
5. Only claim compression/bitrate after counting full headers, models, and residuals;
   only claim lossless coding after exact round-trip tests across the stated domain.

## Other planned work

- Native WAV CLI, sample-rate metadata, and explicit multichannel processing.
- Validate other platforms and add CI; choose a minimum supported Rust version.
- Address uniform-x branch mismatch with a new named policy, never silently change
  the compatibility behavior.
- Study FLT separately using a tested least-squares solver and calibration model.
- Publish/extract the folder into its own Git repository when requested; package
  publishing remains disabled until its API and validation policy are ready.
