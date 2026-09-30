# Research roadmap

## Implemented foundation

- Standalone Rust library and numeric CLI; standard library only.
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
- FAFT rotating-circle experiment: radix-2 forward/inverse transforms, dyadic
  constant/line/circle model selection, zero-curvature line limits, per-circle
  precision, and verified RMS budgets. Integral AFT retained as a reference preset.
- FAFT comparison across 513 Beethoven windows and 3,078 checked reconstructions,
  root/tree controls, runtime scaling, precision graphs, and an epicycle GIF.
- Geometry-first FAFT: circumcircles from waveform points, optimized coverage,
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

## Current focus: geometry-first FAFT

1. Prune point/phase candidates using justified prediction bounds; fitting is now
   the dominant cost. Measure reuse across neighboring time windows.
2. Improve greedy hierarchical splitting/merging with shared circle parameters,
   source-checked parent/child deltas, residual circles, and reusable model banks.
   Variable supports and checked adjacent refitting now exist; no global optimum
   is established.
3. Define an objective that also constrains spectral phase/band errors: a source
   sample-error limit alone allows changes in quiet spectral bins.
4. Reduce adaptive-search oracle cost and misses; compare selected-frequency
   queries with matched FFT/Goertzel methods, group compatible angular rates, and
   tighten pruning bounds. The prepared arithmetic kernel improves cost, but
   padded FFT remains faster in the current survey. Flatness pruning is heuristic;
   the bound-only control exhausts its budget. Coarse prediction sometimes saves
   no work when its disagreement bound exceeds the relevance floor.
5. Test adaptive per-field precision, a bounded wire format with measured bytes,
   continuity/overlap, broader audio, and prediction from truly omitted points.
6. Keep the integral AFT and FFT-based phasor control as references. Their model
   definitions differ; zooming a predictor does not recover missing observations.

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
