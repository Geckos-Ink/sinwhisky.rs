# SinWhisky.rs — AI Agent Reference

This is the operational map for the standalone Rust fork: circle-based resampling
and reusable geometry for audio compression research. The resampler and numeric
CLI are implemented; the segment representation is experimental. Do not confuse
it with a finished codec, perfect reconstruction, or Fast Linear Transform (FLT).

## Read order and source authority

Read this file, then [README.md](README.md) for usage and [docs/DESIGN.md](docs/DESIGN.md)
for coordinate systems, provenance, and mode differences. Authority order:

1. [LICENSE](LICENSE) governs redistribution and attribution.
2. Public contracts in [src/lib.rs](src/lib.rs), the typed APIs, and focused
   [contract tests](tests/contracts.rs) govern current behavior. Tests assert only
   their stated cases; they do not prove audio quality.
3. Current implementation and [Cargo.toml](Cargo.toml) govern build/ownership.
4. [README.md](README.md) and [docs/DESIGN.md](docs/DESIGN.md) explain usage/design.
5. [ROADMAP.md](ROADMAP.md) records planned work, never evidence of implementation.
6. The [2014 paper](<docs/SinWhisky e FLT.pdf>) and ancestor sources are historical
   algorithm evidence. They do not define the newer uniform-x compatibility mode.

When sources disagree, inspect the owning behavior and fix code/docs/tests within
scope. Report unresolved conflicts instead of silently choosing a contract.
While this folder is nested, also obey the ancestor repository's instruction
files. Once extracted, this file is the root handbook. There are no deeper scoped
instruction files in this crate.

## Collaboration and maintenance

- Preserve unrelated working-tree changes. Inspect status and applicable instructions
  before editing; use `codex/` for new branches unless the user specifies otherwise.
- Keep this source map, README, design contracts, and roadmap synchronized with
  changes to behavior, files, interfaces, prerequisites, and validation.
- Do not modify the ancestor C/Python algorithms just to implement a Rust-only
  experiment. If changing their shared uniform contract, update both ancestors
  and cross-validation as required by the ancestor handbook.
- Keep compatibility changes explicit. Do not regenerate fixtures merely to make
  a failing test pass. Research ideas belong in [ROADMAP.md](ROADMAP.md).
- Run focused tests plus the checks below before handoff. Report skipped/unavailable
  checks accurately. Do not claim subjective listening results from error metrics.
- `target/` and Python caches are generated and ignored. Cargo generates the tracked
  lockfile; do not hand-edit it. Keep audio data and tool installations out of source.
- The folder is extraction-ready but remains part of the ancestor working tree.
  Do not add a nested `.git`, publish a crate, or create a remote as routine cleanup.

## Essential principles

### Reuse geometry independently of rendering

`Analysis` owns source samples and models; zoom/neighbors belong to synthesis.
Preserve anchors, aperture, circle branch, and angular endpoints when reusing a
circle. Codec experiments must not secretly refit or mutate the resampler's model.

### Distinguish the two mathematical methods

Uniform x and equal angle produce different waveforms. Preserve named modes and
independent tests. Do not claim the uniform mode reproduces the paper example.

### Measure codec claims

Coverage and reconstruction error must be explicit. Count real encoded bytes
before claiming compression. A `ResearchFrame` is a prediction experiment, not a
bitstream or proven codec. Its segments must never silently fill gaps with zeros.

### Keep the core portable and inspectable

Use the standard library, f64 arithmetic, and safe Rust. `src/lib.rs` forbids unsafe
code. Optional NumPy/FFmpeg research tools must not become library dependencies.

## Critical implementation contracts

- **Output/identity:** [ResampleParams::output_len](src/resample.rs) uses checked
  arithmetic; `Analysis::resample` copies originals exactly at multiples of
  `zoom+1`. Empty/singleton inputs retain length. Protected by
  `lengths_originals_and_reuse`, `validation_and_overflow`, and CLI tests.
- **Geometry:** [Circle::through/evaluate](src/geometry.rs) reject degenerate fits
  or out-of-extent predictions. Extent uses `x-cx`. Branch selection uses the
  middle sample and does not guarantee both outer samples share the branch.
  Protected by `geometry_and_extent`, `exact_circle_without_aperture`, and
  `angular_sine_and_branch_crossing`.
- **Aperture/indexing:** [Analysis::new](src/analysis.rs) uses all three pairwise
  amplitude differences; `models()[i-1]` belongs to sample `i`. Model coordinates
  are `[-a,0,a]`, not global indices. Samples/configuration must be finite and
  epsilon positive. Model access through `Analysis` is immutable.
- **Blending:** [Analysis::resample](src/resample.rs) visits anchors ascending.
  Uniform mode skips line models, blends any valid circle, and falls back to the
  source-pair line only when no finite blend exists. One circle is enough.
- **Angular mode:** `OriginalAngular` requires neighbors=1. Original aperture and
  synthesis presets are paired by `resample`; callers using `Analysis` choose
  their own fit preset. `paper_angular_example` protects the finite-zoom window.
- **FAFT:** [faft](src/faft.rs) integrates the circle curve exactly at any real
  omega; zoom is never a transform parameter. `CircleCurve` coverage is contiguous
  (stacks share a support), conversions copy geometry without refitting, and a
  zoom-z sampled transform equals FAFT plus images (tested). Peaks are exact
  maxima of that function; the duration-spaced scan is a seed heuristic. Arcs use
  the endpoint kernel outside the resonance guard and the moment form inside it;
  values and derivatives share that split. Protected by `tests/faft.rs`.
- **Peak selection:** [pursuit](src/faft/pursuit.rs) treats relevant frequencies
  as a jointly fitted tone model; `peaks` stays the exact maxima of |FAFT| and its
  arithmetic is unchanged (port vectors reproduce byte for byte). Gram entries
  come only from the window kernel K, K', K''; projections are FAFT values/first
  derivatives. Weighted curve energy is the only numerical integration
  (Gauss-Legendre panels, at most pi of phase each). Residual bound
  `sqrt(residual/W)` holds at every real omega. Pruning is by significance
  (unique energy), never by distance; frequencies move by damped Gauss-Newton per
  coupled cluster with a stated step tolerance. The tracker's novelty test is a
  heuristic, backed by interval rescans. Protected by `tests/faft_pursuit.rs`.
- **Bands/EQ/codec:** [divide](src/bands.rs) bands sum to the source samples
  exactly (last residual band is the remainder); tones are assigned whole by
  crossover weights that sum to one. [analog](src/analog.rs) filtering is exact on
  spiral pieces (zero state before the curve; zero-phase drops the forward tail
  past its end). [layered](src/layered/mod.rs) (SWL2, [spec](docs/SWL2.md))
  decode needs no source and reproduces the encoder's samples; every sample is
  within `tolerance` (0 = lossless) unless the stream is `PARAMETRIC` (no
  correction chunk, error measured as SNR). The correction chunk is integer
  arithmetic; gradient/anchor renders are binary64 (same-platform exactness only).
  `encode_auto` keeps a layer only if the exact coded size shrinks. Stereo
  decorrelation only when lossless. Sub-half-bin tones are DC (ill-conditioned
  sine atom otherwise). Protected by `tests/bands.rs` and `tests/layered.rs`.
- **Frame coverage/error:** [ResearchFrame::from_analysis](src/codec.rs) partitions
  source samples into nonempty half-open ranges. Every predicted source sample
  is checked against tolerance. Decode has no dependency on original PCM.
  `segments_cover_every_sample_with_bounded_error` protects both contracts.

## Architecture and data flow

Numeric CLI → parse f64 samples → `Analysis::new` → reusable triplet models →
`Analysis::resample` → copied originals plus blended inserted samples → stdout.

Alternative consumer: `Analysis` → `ResearchFrame::from_analysis` → ordered
circle/line/literal segments → `decode` → `ErrorMetrics`. This path evaluates
uniform-x circle predictors and does not blend segment boundaries.

FAFT consumer: source samples → circle determination (`GeometryFrame` or
`AngularFrame`) → `CircleCurve` rotations → `Faft` prepared window → exact values
at arbitrary frequencies, analytic derivatives, and relevant peaks. Zoom only
samples the same curve for synthesis and comparison. Selection: `Faft` →
`pursue` (one scan, residual seeds, joint fit, certificate) or, for consecutive
windows, `PeakTracker::update` (scan-free polish, rescans on interval/novelty).

EQ/compression consumer: samples → `ToneAnalysis` (circle curve, Hann windows
T/hop T/2, |FAFT| maxima below `upper`, `fit_banded` + significance prune +
`fit_refined`) → `divide` (tones by crossover weight; residual line curve through
zero-phase Butterworth, complementary) → `equalize`/`extrude`; or →
`layered::encode_auto` (stereo mode; per channel optional gradient groups,
optional SinWhisky anchors, then the LPC correction chunk with the layers as
side information; every choice by exact coded size) → SWL2 bytes → `decode`.
`examples/eq_audio.rs` runs it on PCM16 for `test_music.sh quick|divide|eq|extrude|compress`.

Ports consumer (outside the crate): `examples/port_vectors` → `ports/fixtures/vectors.json`
→ `ports/wasm-simd` (f64 resampler, AFT, FAFT; scalar and SIMD) → `ports/webgpu` (f32
kernels fed by the WebAssembly port's models and atoms), each checked against the vectors.

The library is synchronous, mono, and fully buffered. There are no threads,
services, persistence adapters, callbacks, or networking. The optional Python
listening tools invoke FFmpeg and the release CLI locally, per channel.
`test_music.sh` drives a standard-library Python helper that reduces encoded
bitrate and performs two-second block synthesis with source-sample halos.
The separate optional NumPy compression lab imports Rust circle descriptors and
writes bounded experimental SWR1 streams for actual byte/error comparisons;
it does not serialize `ResearchFrame` or alter the resampler.

## Linked source reference

### [Cargo.toml](Cargo.toml)

Owns package identity, edition, lib/bin entry points, and disabled publication.
There are no Cargo dependencies. Update here for build/API packaging changes;
verify with Cargo checks. No CI or release pipeline is configured.

### [Cargo.lock](Cargo.lock)

Cargo-generated package resolution, tracked for reproducibility of the CLI.
Regenerate through Cargo after manifest changes; do not add research-tool packages.

### [.gitignore](.gitignore)

Excludes `target/`, Python caches, and macOS metadata. Add runtime exclusions when
introducing new generators; it is not permission to commit audio assets elsewhere.

### [src/lib.rs](src/lib.rs)

Owns public module/re-export boundaries, `Error`, `buffer`, and `forbid(unsafe_code)`.
`buffer` uses fallible exact reservations; it is shared by analysis, synthesis, and
frames. The doctest is the minimal public flow. It does not parse file formats.

### [src/geometry.rs](src/geometry.rs)

`Point`, `Branch`, and `Circle` represent local geometry. `Circle::through` solves
the C/Python 2x2 system; `evaluate` enforces horizontal extent; `angle` and private
`arc_angles` handle angular coordinates/direction. Called by analysis and model
prediction. Never replace extent with `abs(x)<=r` or assume one branch contains all
three points. Geometry and angular contract tests own regressions.

### [src/analysis.rs](src/analysis.rs)

`Aperture` chooses disabled, raw range, or clamped gain scaling. `AnalysisParams`
provides default/original presets and private validation. `Analysis::new` owns
sample validation/copy and triplet fitting. `LocalModel` retains an anchor/aperture
and `Primitive` circle/line; `LocalCircle` stores geometry and angular endpoints.
Its `evaluate_uniform` and `evaluate_angular` use different position semantics.
Resampling and codec research consume the immutable model slice. Do not add zoom
here; `lengths_originals_and_reuse` ensures synthesis does not modify analysis.

### [src/resample.rs](src/resample.rs)

`Interpolation`, `ResampleParams`, and `output_len` own synthesis settings/checks.
`resample` pairs analysis presets for one-shot calls; `Analysis::resample` owns
window evaluation, ordered accumulation, finite-blend fallback, and copying
originals. Tests cover formulas, neighbors, extremes, reference vectors, and the
paper. Angular neighbor extension is undefined and rejected, not silently clamped.

### [src/codec.rs](src/codec.rs)

`SegmentModel::predict` evaluates circle/line/literal predictors. `Segment` owns
private coverage/model data; public access cannot mutate a frame's validated state.
`covered_until` checks every sample before extending a span.
`ResearchFrame::from_analysis` greedily selects line/existing-circle candidates;
`decode` reconstructs without original PCM. `ErrorMetrics::between` computes
max/RMS error and checks lengths/finiteness. Segment tests protect coverage, true
circle reuse, metadata validation, and tolerance. No function here serializes bytes.

### [src/spectral.rs](src/spectral.rs)

Owns `Complex`, `angular_fourier`, `uniform_fourier`, and `frame_dft_bin`.
Continuous arc transforms use exp(-i*omega*position), sample-position units,
and no blending/copying of anchors. Angular integration is analytic; uniform
integration uses explicit bounded midpoint panels after an angular substitution.
Panel count is not an accuracy bound. The discrete frame sum reads predictors,
never original PCM, and is O(N) per bin, not an FFT. `tests/spectral.rs` checks
known areas, resonance, independent quadrature, scaling/phase, Parseval, extent,
validation, and unchanged analysis. Keep continuous and sampled spectra distinct.

### [src/aft.rs](src/aft.rs)

Owns `AngularFrame` and `AngularStats`. Prepares periodic source halos with explicit
analysis parameters, averages the adjacent angular halves on each unit interval,
and closes last-to-first coverage. Degenerate/rejected halves use the actual
source-pair line. Circle endpoint acceptance is 1e-8 times max(1, endpoint scale);
rejections and line/circle halves are counted. The prepared geometry is immutable.
`evaluate` extends periodically; `fourier` analytically integrates on `[0,N]`;
`spectrum` covers the standard nonnegative bins. Full cost is O(N²), not an FFT.
Retained as the integral reference preset; FAFT computes this same integral on
this curve (through `halves`) without numerical integration.
The AFT interpolation policy is separate from finite-zoom angular resampling.
`halves` is a read-only view of the prepared halves, used by FAFT for same-curve checks.
The continuous Nyquist coefficient need not be real. `tests/aft.rs` checks
coverage, original geometry reuse, quadrature, sinc² linear control, phase shifts,
conjugacy, constants/short inputs, and validation.

### [src/geometric_faft.rs](src/geometric_faft.rs)

Circle determination for FAFT, plus the earlier sampled-spectrum control (the
"geometry-first FAFT" experiment, a zoomed DTFT, not the transform): `GeometryFrame` fits source-point
circumcircles and finite lines BEFORE spectrum retrieval, with no FFT or integral.
Searches dyadic support lengths, interior points, aperture gains, and independent
per-circle significand widths. Normalized coordinates, periodic terminal anchor,
full coverage, sample/anchor checks and final rescaled maximum error are explicit.
Rounded circle fields also check horizontal fitting anchors. Default prevalidation
rejects failed nominal models before precision search; the exhaustive control
can admit their quantized variants. DP minimizes descriptors then hypothetical
bits within the admitted bank, not arbitrary geometry. `dtft`/`hann_dtft` crop and
weight the prepared predictor without refitting; whole-grid queries use closed
finite sums or batched rotations, O(pieces*bins). Stable resonance limits,
reanchoring, and explicit line first moments replace integration. Zoom samples
that same function, not more source data. Separate `hierarchical` fits arbitrary
integer supports, splits failures against source samples, and refits/validates
adjacent merges. Depth-limited leaves use checked source-pair lines. Prepared
`GeometricSpectrum` caches roots and retrieves arbitrary-frequency finite sums;
closed-sum and arithmetic-rotation kernels are distinct controls. Sample-grid
amplitude/moment and predictor-disagreement bounds require O(N*zoom) preparation.
Tests: `tests/geometric_faft.rs`, `tests/adaptive_faft.rs`.

### [src/adaptive_faft.rs](src/adaptive_faft.rs)

Sampled-predictor (zoom DTFT) control, superseded for the transform by `faft`.
Owns nonuniform relevant-frequency search over prepared geometry. Initial coverage,
strongest-first interval refinement, bounded ternary peak refinement, depth/width/
combined-query limits, relevance floors, and strongest-first frequency merging are
explicit. Optional coarse geometry uses a shared-grid weighted L1 disagreement
bound to skip some fine probes; retained uncertainties are inspectable. Below-floor
bands use derivative bounds; sampled-quiet/flat LowGain stops are heuristic and may
miss peaks. Depth/budget/width stops are reported, never proof of completion. Output
peak brackets are queried neighbors, not confidence intervals. Tests:
`tests/adaptive_faft.rs`. Arbitrary query precision does not imply resolution
independent of source duration or a unique physical oscillator decomposition.

### [examples/adaptive_bench.rs](examples/adaptive_bench.rs) and [tools/adaptive_compare.py](tools/adaptive_compare.py)

Measure hierarchical children/checked merges against dyadic DP; adaptive roots
against the repeated-trig first attempt; coarse/fine precision, depth and zoom;
off-bin, separated and unresolved near tones; matched Hann and padded FFT controls.
Python independently predicts from exported geometry and checks all returned peaks,
off-bin controls and spread queried points, including uncertainty bounds. Requires
NumPy/Matplotlib/Pillow, FFmpeg and Cargo. Audio/models/complex query data remain
private in a new output directory outside the repo. Only aggregate JSON/figures
are published: `docs/experiments/adaptive-faft-beethoven-20260930.json` and
`docs/figures/adaptive-faft-20260930/`. Synthetic bridge:
`tests/test_adaptive_compare.py`. Default query pruning has no completeness claim;
report misses and exhausted budgets alongside timing gains.

### [examples/geometric_bench.rs](examples/geometric_bench.rs) and [tools/geometric_compare.py](tools/geometric_compare.py)

Rust benchmark records geometry/spectrum timings and exports private models.
Python independently predicts from centers/radii/angles and checks source bounds
and spectra against the SAME predictor's FFT. Compares point/phase/precision
banks, exhaustive/prevalidated policies, first/retry timings, held-out source
points, zoom density, and actual 80/160 kbps mono MP3 controls. Requires NumPy,
Matplotlib/Pillow, FFmpeg/libmp3lame and Cargo. Output must be new/outside repo;
model JSON, dense predictions and encoded audio stay private. Only aggregate
JSON and rendered PNG/SVG/GIF enter docs. Synthetic bridge tests are
`tests/test_geometric_compare.py`. Report: `docs/experiments/geometric-faft-beethoven-20260930.json`;
figures: `docs/figures/geometric-faft-20260930/`.

### [src/faft.rs](src/faft.rs)

Owns FAFT, the Fourier transform of determined circles. `Rotation` (center, radius,
phase, rate, slope on an integer support) and `CircleCurve` (contiguous coverage,
stacked supports add) come from `GeometryFrame` or `AngularFrame` without refit.
`CircleCurve::evaluate` is the curve zoom samples. `transform` crops to a window,
anchors each rotation (start value plus cos/sin deviations) and prepares
`exp(i*rate*length)`. `Faft::integral/value/values` are exact closed forms at any
finite omega (rectangular or analytic continuous Hann), chaining support phases by
multiplication, with Horner series below the stated |delta*L| thresholds.
`derivatives` returns centered first/second omega derivatives. `references` groups
circle rates (fixed width, no chains). `peaks` = duration-spaced scan (Hann shares
an extended grid), parabolic start, bracketed Newton, flank/unconverged drops,
merge width and floors, with `PeakStats`. O(rotations) per query. No FFT, sampling
or quadrature; no dense-spectrum speed claim. Circle rates are geometric, not
physical frequencies; scan completeness is measured, not guaranteed. Kernel:
arcs use the endpoint form of their deviation from the start value (prepared
`start_slope`/`end_slope`/`end_change`/`curvature`; one real division), values
and derivatives alike. Within `ENDPOINT_GUARD` (`|(omega -+ rate)*L| < 0.5`)
they fall back to the moment form with prepared turns and series. Keep the
guard: the endpoint quotient is 0/0 at resonance. The moment form rounds flat
arcs at eps*radius*length; the endpoint form keeps relative accuracy. The
published Beethoven timings predate the endpoint kernel. Tests: `tests/faft.rs`, which also pins the
paper's geometric identities (doubled inscribed angles and the crest rate,
endpoint form, infinite-radius line limit, element × array factor); doctest in
the module. Definition: [docs/FAFT.md](docs/FAFT.md).

### [src/faft/pursuit.rs](src/faft/pursuit.rs)

Child module of `faft` (private access to `Faft` atoms, `newton`, `scan_grid`,
`grid_seeds`). Owns the window kernel (`kernel`, `kernel_range`, private
derivatives), `energy` (stacked rotations summed before squaring), `peak_range`
(measured on |FAFT|, valley-shared sides flagged), `fit` (cos/sin Gram blocks,
Cholesky with dependent-atom dropping, inverse diagonal for significance),
`pursue` (single scan; residual seeds on that grid; in-round provisional
amplitudes skip explained seeds; Newton on the single-tone least-squares
criterion; per-cluster Levenberg-Marquardt polish with variable projection;
significance pruning with seed memory; final all-cluster pass) and
`PeakTracker` (predicted polish, jump limit, significance prune, certificate or
novelty `2*m^2*W` over the best residual since the last scan, interval rescans,
birth/survival hysteresis, hold). Fitted values are `(alpha - i*beta)/2` rotated
to the window start; `observed` is FAFT there. Separation inside one main lobe
is a tone-model property, not transform resolution. Not ported.

### [src/faft/banded.rs](src/faft/banded.rs)

Child of `faft` (uses `pursuit`'s `pub(super)` `kernel_derivatives`,
`centred_value`, `to_start`). `fit_banded`: joint tone fit with Gram entries kept
within `reach`, band Cholesky in ascending frequency (forward pass = explained
energy), back substitution, Takahashi selected inversion for significance; a
truncated band evaluates the residual from the full quadratic form, so `bound`
stays valid. Atoms with normalized norm below 1e-9 are dropped. Infinite reach
equals `fit`. `fit_refined`: banded fit plus coordinate Newton steps on each
partial residual (quarter-resolution limit), refit each round. Pursuit arithmetic
and port vectors are unchanged.

### [src/analog.rs](src/analog.rs)

`AnalogFilter` (partial fractions; stable, distinct, conjugate-closed poles;
`butterworth_lowpass`, `response`) and `SpiralCurve` (pieces
`Re[c0 + c1 u + sum a e^{rate u}]`): `from_circles` (stacked rotations merged,
pieces cropped exactly), `from_samples` (line curve), `filter` (exact state chain,
one term per pole and piece), `reverse`, `filter_zero_phase` (|H|^2), `samples`,
`evaluate`. Resonant input terms are rejected. No spiral AFT yet.

### [src/bands.rs](src/bands.rs)

`ToneParams`/`ToneAnalysis` (pads T/2 zeros, hierarchical circle fit, per-window
peaks + banded fit + prune + refine + DC snap; `synthesize(gain)` overlap-adds
windowed tones), `BandSplit`/`Crossover` (brick wall or raised cosine; weights sum
to one), `divide`, `equalize`, `extrude`. The divider is as good as the fit; tones
above `upper` stay in the residual. Uses the fast path, not `pursue`: the tracker
cost 100-330 ms per window on real music in development runs.

### [src/layered/](src/layered/mod.rs) and [docs/SWL2.md](docs/SWL2.md)

Experimental SWL2 stream, specified byte by byte in `docs/SWL2.md` (keep the two
in sync; bump the version for any layout change). `mod.rs`: 16-byte header, per
channel typed chunks, stereo transforms (FLAC's four, lossless only), `encode`
(fixed `StreamConfig`), `encode_auto` (search with `Trial` log), `decode`,
`PARAMETRIC`. `coder.rs`: LZMA-style range coder, adaptive gamma, MSB-first bit
packer. `correction.rs`: per block gain for the layer prediction (quarters) and
LPC order (0..32, Levinson-Durbin on Tukey autocorrelation, quantization noise on
the zero lag when lossy, 15-bit integer coefficients), chosen by trial coding;
running-magnitude contexts. `gradients.rs`: track linking, quadratic-phase
segments (frequency gradient), groups sharing start/duration/volume function,
header-declared field widths, amplitude-class quanta, cross-faded joints.
`anchors.rs`: greedy dynamic-gap anchors, circles through anchor triplets with
quadratic angle in time, raised-cosine cross-fade; optional zero-phase low-band
target. Bounded parsing and malformed-stream rejection are tested, not fuzzed.

### [examples/eq_audio.rs](examples/eq_audio.rs)

Reads interleaved s16le PCM, runs `divide`/`eq`/`extrude`/`compress`/`all` per
channel, writes PCM16 WAVs, SWL2 streams (`encode_auto`, `--search fast|full`,
optional `--parametric P [--parametric-anchors T]`) and `report.json` into a new
directory outside the repository (checked), verifies every decode and error
bound, and prints the aggregate report. Timings are wall-clock. Published
aggregates: `docs/experiments/eq-compression-beethoven-20261005.json` (SWL1) and
`docs/experiments/swl2-beethoven-20261005.json`.

### [examples/pursuit_bench.rs](examples/pursuit_bench.rs)

Deterministic synthetic comparison (no audio, no clock in the data) of `peaks`,
`pursue` and `PeakTracker` on the same fitted windows: recall, weak recall,
spurious reports, height/frequency errors, evaluation counts, timings, tracker
paths and identities, rectangular sidelobe counts. Prints one aggregate JSON;
the published snapshot is `docs/experiments/faft-pursuit-synthetic-20261004.json`.
Timings are machine-specific; regenerate rather than edit numbers.

### [examples/faft_bench.rs](examples/faft_bench.rs), [tools/faft_compare.py](tools/faft_compare.py), [tools/faft_figures.py](tools/faft_figures.py)

Rust JSON Lines per frame: hierarchical fit, FAFT preparation/queries/peaks
(density 1/2/4, circle seeds), the earlier sampled kernels and adaptive search on
the same geometry, zoom 1..64 differences, integral AFT versus FAFT on its curve,
optional dense 0..3π spectra, and private rotations. Python needs NumPy and Cargo
(FFmpeg only for non-WAV input; `afconvert` WAV works). It independently evaluates
the rotations, checks source bounds, Gauss-Legendre values and every peak's local
maximum, and matches padded source-FFT and adaptive controls. Hand-written SVG
figures (no Matplotlib). Output must be new and outside the repo; private JSON can
reconstruct audio. Published: `docs/experiments/faft-circle-arcs-beethoven-20260930.json`,
`docs/figures/faft-circle-arcs-20260930/`. Synthetic bridge: `tests/test_faft_compare.py`.

### [tools/faft_geometry_figure.py](tools/faft_geometry_figure.py)

Standard-library diagram for the paper's Figure 1:
`python3 tools/faft_geometry_figure.py docs/figures/faft-three-points.svg`.
Deterministic exact geometry of one example support, with no audio or measured
data. It solves the same circle system and angle unwrapping as
`src/geometry.rs` and asserts the doubled-inscribed-angle sweeps before drawing.

### [src/phasor_fft.rs](src/phasor_fft.rs)

Formerly `src/faft.rs`: the first "FAFT" prototype, renamed because it never used
fitted circles. Owns the FFT-based phasor control kernel and model optimizer. `transform` is a
standard radix-2 forward/inverse DFT, O(N log N). `PhasorFrame` searches dyadic
constant/endpoint-line plus rotating-circle candidates with per-circle Cartesian
precision. Children replace parents on disjoint supports; they are not residual
predictions. `LineLimit` is an explicit finite zero-curvature representation.
The O(N log² N) search minimizes baseline+circle count, then estimated bits, within
the generated candidates and equal per-leaf RMS budgets. `decode` needs no source
PCM; both leaves and final rescaled output are checked. `evaluate` has no join
continuity or intersample error guarantee. Model fields are private; descriptor
views are immutable. Bits are hypothetical costs, not a serialized format or rate.
`tests/phasor_fft.rs` checks direct DFT/inverse agreement, phase/Nyquist, sparse tones,
precision, child selection, line limits, coverage, numerical range, and errors.

### [examples/phasor_bench.rs](examples/phasor_bench.rs) and [tools/phasor_compare.py](tools/phasor_compare.py)

Renamed from `faft_bench`/`faft_compare.py` with the module. Rust JSON Lines
records warmed kernel/search/decode timings and private models.
Python compares NumPy rFFT, root/tree optimization, and integral AFT; independently
checks direct projected-circle reconstruction and RMS for every trial. Generates
aggregate JSON, PNG/SVG plots, and a measured epicycle GIF. Requires NumPy,
Matplotlib, Pillow, FFmpeg, and Cargo; none are core dependencies. Output must be
new and outside the repository. `private-selected-model.json` is invertible audio
information and must never enter version control. Tests: `tests/test_phasor_compare.py`.
Published results: `docs/experiments/faft-beethoven-20260930.json`, figures under
`docs/figures/faft-20260930/`. Treat the unchanged integral AFT as a reference preset.

### [examples/aft_bench.rs](examples/aft_bench.rs)

Numeric complete frames to JSON Lines of AFT coefficients, circle/line counts,
and warmed in-process fit/transform timing samples. Uses original analysis
parameters on caller-supplied amplitude units, without rescaling/windowing.
Optional dense midpoint samples support independent integration checks. Text I/O,
warmup, and dense evaluation are excluded from the recorded transform timings.

### [tools/aft_compare.py](tools/aft_compare.py)

Optional NumPy/Matplotlib/Pillow/FFmpeg benchmark and graph/GIF generator. Uses
PCM16 units, a mono downmix, identical periodic-Hann input samples for Rust AFT
and NumPy rfft, and a periodic linear-interpolation FT control. Reports spectral
differences separately from dense-integration discrepancies, and fitting cost
separately from transform cost. Midpoint FFT integration requires both `1/q`
and the half-step phase factor. Tests are `tests/test_aft_compare.py`.
Output must be new and outside the repository: complex spectral arrays can
reconstruct audio and must remain private. Only aggregate JSON and rendered
figures belong in docs. Figures and the silent GIF are generated from real
measured track windows, never invented spectra.

### [examples/export_circles.rs](examples/export_circles.rs)

Numeric stdin to blockwise JSON Lines of default Rust analysis circles, used by
the compression lab. Exports block-relative anchor, aperture, center, radius,
branch, and an anchor evaluation for cross-language validation. No refitting or
core CLI changes. Block size is a power of two in 4..=4096; partial blocks survive.

### [tools/compression_lab.py](tools/compression_lab.py)

Optional NumPy/FFmpeg research harness. Compares raw/delta, recursive line/circle,
quantized residual, and sparse Fourier predictors against a lossless FLAC control.
`predict` evaluates exported uniform geometry; children retain source-derived
models. `hierarchy_diagnostics` distinguishes parent restrictions from corrected
children. `encode_stream`/`decode_stream` own the explicit little-endian SWR1
experiment envelope, bounded allocation/decompression, exact coverage, raw
fallback, and independent reconstruction. Every trial checks error after
serialization, counts full file bytes, and writes only to a new directory outside
the repository. `tests/test_compression_lab.py` covers exact/bounded round trips,
partial blocks, circles, Fourier phase/quantization, and malformed streams.

### [docs/FAFT.md](docs/FAFT.md)

Owns the FAFT definition: stage separation (determination, zoom, transform), the
rotation parameters from the three-point triangle, the phasor, anchored and
endpoint closed forms, the rotate-and-add rule (Goertzel-like, not an FFT),
numerics, peak policy, circle-reference findings, the renamed control, measured
Beethoven results and reproduction. Update it with any `faft` behaviour or
measurement change; numbers only from actual runs.

### [docs/PAPER_FAFT.md](docs/PAPER_FAFT.md)

Public manuscript by Riccardo Cecchini. It derives the transform from geometry:
the circle from the anchor triangle (determinant 8×area, law of sines), rates as
doubled inscribed angles, the line as the infinite-radius limit, and the
aperture/crest relation. Then come the phasor, anchored and endpoint forms, the
high-frequency decay from value and slope jumps, time distance as phase
(element × array factor), Hann, derivatives, Poisson/zoom, FAFT versus FFT and
Goertzel, numerics, pseudocode, peak refinement, and results copied from the
FAFT snapshot. Section 6 also owns peak selection: the window kernel and peak
ranges (Tables 1–2), the zero-sum argument, the closed-form joint fit and
certificate (Propositions 7–8), joint Gauss-Newton, Algorithms 5–6, and Table 3
from the synthetic pursuit snapshot. Keep its numbers and claims synchronized with FAFT.md. Write
paragraphs on single lines and display math as `$$` blocks without `\\`, `\,`,
`\{` or `\|`, so the web renderer does not strip escapes. Numbered equations use
`\tag`. Every formula must parse in KaTeX. Figure 1 is
`docs/figures/faft-three-points.svg`, generated by `tools/faft_geometry_figure.py`.

### [docs/PAPER_FAFT-net.md](docs/PAPER_FAFT-net.md), [PDF](docs/PAPER_FAFT.pdf) and [earlier draft](docs/PAPER_FAFT_old.md)

`PAPER_FAFT-net.md` is the web copy: the same prose and captions as
PAPER_FAFT.md, with the author's version line and absolute figure URLs under
`https://geckos.ink/stuff/sinwhisky/figures/`. Regenerate it from PAPER_FAFT.md
after every edit, and upload new figures to that location. `PAPER_FAFT.pdf` is a
browser print of the web copy and goes stale when the text changes.
`PAPER_FAFT_old.md` is a superseded draft kept for history; do not update it.

### [docs/PAPER_EQ_COMPRESSION.md](docs/PAPER_EQ_COMPRESSION.md) and [tools/eq_compression_checks.py](tools/eq_compression_checks.py)

Draft extension of PAPER_FAFT.md, same author: band division (channel divider) by
fitted components, the two-way (ascending, descending) banded solution of the
joint fit, exact analog filtering of circle curves (spiral rotations), equalization
and extrusion, and a proposed layered closed-loop coding scheme (gradient-tone
tracks, band residuals at their own rate, integer corrections within a tolerance).
Nothing in it is implemented in the crate and no compression is measured; its
Section 6 lists the required experiments. It cites PAPER_FAFT.md with an F prefix
(`(F44)`, Proposition F8): keep those references valid when that paper is renumbered.
Same formatting rules as PAPER_FAFT.md. Table 1 and the figures quoted in its
Sections 2.4 and 4.2 come from `python3 tools/eq_compression_checks.py` (standard
library, deterministic, under a second), which asserts Propositions 3, 5, 6 and 8
against independent computations; regenerate rather than edit those numbers.
There is no web copy yet.

### [docs/COMPRESSION.md](docs/COMPRESSION.md)

Opens by pointing to FAFT.md and labels its FAFT-named sections as controls.
Owns hierarchical circle merging/adaptive-frequency limits, inverse rotational
sums and measured depth/precision/zoom controls; geometry-first FAFT fitting, finite sums, point/phase/precision/zoom/bitrate
comparisons and retry measurements; earlier FAFT phasor/line limits, dyadic search, dynamic precision, RMS contracts,
and benchmarks; also Fourier conventions/derivation, subarc limits, candidate search, SWR1 format,
reproduction, and measured Beethoven compression results. The current AFT section
also owns the comparison definition, timing limits, and embedded graphs/GIF.
Its snapshot is [aft-beethoven-20260930.json](docs/experiments/aft-beethoven-20260930.json),
with rendered plots under `docs/figures/aft-20260930/`. The earlier compression snapshot
is [experiments/beethoven-20260930.json](docs/experiments/beethoven-20260930.json).
Contains no recording or machine-specific source paths. Do not infer perceptual
quality or circle-specific savings from residual compression. The format is
experimental, not a stable public codec or a fuzz-validated untrusted parser.

### [src/main.rs](src/main.rs)

Owns CLI `USAGE`, `run`, and error-to-exit-code handling. Parses whitespace f64
samples, validates mode flags, invokes the library, writes one number per line.
Failure exits with code 2 and stderr diagnostics. It buffers all stdin and is not
a bounded untrusted-stream parser. [tests/cli.rs](tests/cli.rs) drives the executable.

### [examples/codec_research.rs](examples/codec_research.rs)

Demonstrates one analysis reused for upsampling and several tolerance sweeps.
Reports segment count and actual decoded error, explicitly not a compression ratio.
Run this after changes to research-frame APIs.

### [tests/contracts.rs](tests/contracts.rs)

Owns analytic/identity/length/geometry tests, the paper's angular example, known
uniform branch mismatch, invalid input/overflow tests, and segment coverage/reuse.
`close` is the numeric assertion helper. Use these tests to distinguish intended
mode changes from numerical regressions; no audio files or external tools are needed.

### [tests/reference_vectors.rs](tests/reference_vectors.rs)

`frozen_python_reference_vectors` parses the embedded fixture and asserts f64
agreement without importing Python. It checks lengths and every output value.
A failing fixture requires reviewing the shared contract, not rewriting expected data.

### [tests/fixtures/uniform.tsv](tests/fixtures/uniform.tsv)

Frozen Python outputs, validated against Rust/C by the optional cross-validation
tool. Columns record zoom, neighbors, aperture switch, input, and expected output.
Includes ramps, constants, signed peaks, and the branch-spanning steep peak.
This is intentional generated test source; regeneration requires a reviewed diff.

### [tests/cli.rs](tests/cli.rs)

`cli` drives the built binary with piped stdin; `command_line_modes_and_failures`
checks output, empty input, mode selection, invalid tokens/flags, and exit codes.
It requires no ancestor checkout or compiler beyond Rust.

### [tools/cross_validate.py](tools/cross_validate.py)

`main` loads the explicit ancestor Python module, compiles its C CLI in a temporary
folder, and compares both with the Rust binary across signals/settings. Requires
NumPy and a C compiler; missing prerequisites fail rather than silently skipping.
`--write-fixtures` updates the TSV only after successful comparisons.
The ancestor checkout is optional validation input, never a Cargo dependency.

### [tools/audio_compare.py](tools/audio_compare.py)

`main` owns excerpt decoding, anti-alias decimation, per-channel Rust CLI calls,
linear baseline, independent bitrate controls, and JSON report. `read_wav` /
`write_wav` own PCM16 I/O; `metrics` computes sample-domain error. Requires NumPy
and FFmpeg with libmp3lame. Output must be new/empty. Input recording is read-only.
Keep its signal processing, artifact names, and [listening notes](docs/LISTENING.md)
synchronized. Do not interpret MP3 bitrate controls as a circle codec benchmark.

### [test_music.sh](test_music.sh)

Executable Bash entry point for input/bitrate fraction/upscale/output arguments.
Subcommands `quick|divide|eq|extrude|compress INPUT OUTPUT_DIR [--start S]
[--seconds D] [eq_audio options]` decode an excerpt with FFmpeg/ffprobe, run
`examples/eq_audio.rs`, and add `source.wav` (and `source.flac` as the lossless
byte control for `quick`/`compress`); `quick` runs all on 5 s from 60 s.
The named `input_music`, `downscale_bitrate`, `upscale`, and `output_music` settings
also support environment-based invocation. It selects `PYTHON`, loads the sh Rust
environment when Cargo is absent, and delegates to `tools/music_test.py` from any
working directory. Keep paths quoted and preserve the helper's exit status.

### [tools/music_test.py](tools/music_test.py)

`main` validates factors/paths, selects supported MP3 bitrate, stages outputs, and
installs the result without overwriting an existing file. `audio_info` reads FFmpeg
metadata; `rust_binary` builds/discovers Cargo's actual target directory or uses an
explicit binary. `stream_resample` owns PCM pipes, two-second blocks and source
halos, and FFmpeg lossless-file output. `interpolate_block` invokes Rust separately
per channel and trims halo predictions. Keep source PCM halos, not reconstructed
samples; dropping them changes circle fits at block boundaries. Fractional upscale
uses ceil(factor) Rust synthesis followed by FFmpeg rate conversion. Requires Python
standard library, FFmpeg/libmp3lame, and Rust; no NumPy dependency. Do not equate MP3
bitrate reduction with sample-rate decimation or claim lost detail is restored.

### [tests/test_music.py](tests/test_music.py)

Optional Python integration suite with generated stereo PCM fixtures.
`test_blocks_match_whole_signal_in_both_modes` compares block output to a direct
whole-signal Rust run; the other tests protect fractional duration/rate, encoded
bitrate reduction, factor validation, and existing output preservation. It requires
FFmpeg and a release binary; missing prerequisites skip the suite explicitly.
It writes only temporary audio and does not depend on the user's recordings.

### [examples/port_vectors.rs](examples/port_vectors.rs) and [ports/fixtures/vectors.json](ports/fixtures/vectors.json)

Deterministic reference vectors for the ports (no clock, a fixed LCG): six cases (tones,
a Hann pilot, flat/ramp/step lines, PCM16-scale noise, short, the paper's) with the
original-angular resampler at zooms 1/3/7, `AngularFrame` stats, `evaluate`, `fourier`,
FAFT rectangular and Hann values, and default Hann peaks. Regenerate with
`cargo run --offline --quiet --example port_vectors > ports/fixtures/vectors.json`; it must
reproduce byte for byte, and a change is a reviewed change of the crate's behaviour.

### [ports/](ports/README.md)

Reference ports outside the crate (no Cargo dependency, no effect on the library):
`ports/wasm-simd` (AssemblyScript to WebAssembly, f64: the original-angular resampler,
the AFT, and FAFT values/scan/derivatives/peaks on its curve, scalar paths following the
crate's arithmetic except that FAFT keeps the moment form, and `v128` SIMD paths) and `ports/webgpu` (WGSL kernels in f32 for
the resampler and the rectangular FAFT, geometry from the WebAssembly port, no GPU
trigonometry). `wasm-simd/build/sinwhisky.wasm` is a checked-in build so the ports run
without a toolchain; rebuild it (`npm run build`, AssemblyScript is the only, development,
dependency) after editing `assembly/index.ts`. Tests: `ports/wasm-simd/test.mjs` (Node,
against the vectors) and `ports/webgpu/test.html` (a browser with WebGPU, served by
`ports/serve.mjs` on loopback). Not ported: uniform-x resampling, `GeometryFrame`,
circle-rate seeds, codec frames. Measured agreement and timings live in `ports/README.md`;
update them with any port or crate change that moves them. `node_modules/` stays ignored.

### [README.md](README.md)

Owns public setup, API examples, mode comparison, codec limitations, and optional
tool commands. Check examples and local links when public surfaces change.

### [docs/DESIGN.md](docs/DESIGN.md)

Owns provenance, coordinate/precision contracts, deliberate original-mode repairs,
compatibility exceptions, and future wire-format boundaries. It records why choices
exist; source/tests enforce current implementation.

### [docs/SinWhisky e FLT.pdf](<docs/SinWhisky e FLT.pdf>)

Historical Italian whitepaper, Riccardo Cecchini, 2014. SinWhisky appears on pages
6–8 and the separate FLT idea follows. Preserve this source document; do not edit
it to match new code. The printed numerical example is tested independently.

### [docs/LISTENING.md](docs/LISTENING.md)

Records one reproducible local experiment, method, numerical results, and limits.
Contains no audio assets or machine-specific music paths. Update results only from
actual runs; do not generalize a single excerpt into a codec-quality guarantee.

### [ROADMAP.md](ROADMAP.md)

Owns future quality/performance experiments, quantization/residual/format work,
platform validation, and separate FLT plans. Move items to implemented only with
code and evidence. Do not add aspirational APIs just to match this list.

### [LICENSE](LICENSE)

Preserves the ancestor MIT attribution and redistribution terms. Include it when
extracting or packaging the folder.

## Features and recurring pitfalls

### Resampling — implemented

Library and CLI both use analysis → synthesis. Uniform mode supports generalized
neighbors; angular mode preserves the repaired historical concept. Tests are
`contracts`, `reference_vectors`, and `cli`. Non-finite input rejection differs
intentionally from legacy NaN propagation. No WAV I/O exists in the Rust CLI.

### Codec foundation — experimental

Analysis → frame segments → decode → measured error. Frames are mono and retain
sample rate/count. There is no quantization or persistence; source-sample tolerance
does not constrain fractional positions, continuity, or perceptual distortion.
The greedy candidate selection is a baseline, not an optimum search.

### Listening tools — optional research

FFmpeg → NumPy low-pass/decimation → Rust per-channel interpolation → WAV/JSON.
Bitrate controls are a separate full-rate MP3 encode/decode path. No subjective
listening test is automated. See `docs/LISTENING.md` for validated scope.

The single-file `test_music.sh` path applies encoded-bitrate reduction first,
then SinWhisky to the decoded sample rate. Factor 1 skips re-encoding; supported
MP3 bitrate selection is reported. WAV/FLAC output retains the increased sample
rate; fractional factors add FFmpeg sample-rate conversion. `tests/test_music.py`
protects block-boundary equivalence, duration, validation, and output preservation.

### Uniform branch discontinuity — deliberate compatibility limitation

A large peak with capped aperture can place its neighbors on the other semicircle;
copying originals does not fix the intervening predictions. Preserve the explicit
`angular_sine_and_branch_crossing` regression and document any new policy separately.
Codec candidates must test endpoints, not infer coverage from a fitted triplet.

### Aperture and exact-circle tests — recurring regression risk

Scaling x changes geometry in original coordinates. Exact uniform-circle tests
must disable aperture. Normalizing input amplitudes can change clamped-aperture
behavior. Keep this distinction in geometry tests and listening comparisons.

### Degenerate geometry and indexing — repaired historical failures

Zero range, missing line intercept, wrong extent origin, or confusing anchor with
cx can cause wrong offsets, zero-filled gaps, or non-finite predictions. Preserve
explicit line/literal paths and half-open coverage. Constant/ramp/two-sample,
extent, and frame-coverage tests detect these errors.

## Interface ownership map

| Surface | Owner | Contract |
|---|---|---|
| `sinwhisky` crate | [src/lib.rs](src/lib.rs), [Cargo.toml](Cargo.toml) | Public f64 analysis/synthesis and Error |
| Geometry/models | [geometry](src/geometry.rs), [analysis](src/analysis.rs) | Local coordinates and immutable analysis views |
| `resample`, `Analysis::resample` | [src/resample.rs](src/resample.rs) | Named interpolation contracts |
| `AdaptiveSpectrum` | [adaptive frequencies](src/adaptive_faft.rs) | Sampled-control peaks; explicit uncertainty and work/heuristic stops |
| `GeometryFrame`, geometric queries | [direct geometry](src/geometric_faft.rs) | Source-point circles, peak-error bound, O(pieces*bins) finite sums |
| `CircleCurve`, `Faft`, `Rotation` | [FAFT](src/faft.rs) | Exact continuous transform of circle curves; peaks at real frequencies; no zoom/FFT |
| `Pursuit`, `SpectralFit`, `PeakRange`, `PeakTracker` | [peak selection](src/faft/pursuit.rs) | Joint tone fit, residual certificate, measured ranges, tracking; not a completeness proof beyond the stated bound |
| `PhasorFrame`, `transform` | [FFT phasor control](src/phasor_fft.rs) | Rotating-circle optimization and sampled FFT; RMS bound, no wire format |
| `fit_banded`, `fit_refined` | [banded fit](src/faft/banded.rs) | Two-way banded joint fit; valid bound under truncation |
| `AnalogFilter`, `SpiralCurve` | [analog](src/analog.rs) | Exact rational filtering of circle/spiral curves, zero-phase passes |
| `ToneAnalysis`, `divide`, `equalize`, `extrude` | [bands](src/bands.rs) | Complete band division by fitted tones and complementary residual filters |
| `layered::encode`/`encode_auto`/`decode` (SWL2) | [layered](src/layered/mod.rs), [spec](docs/SWL2.md) | Experimental closed-loop coder; error <= tolerance (or parametric); automatic choices by exact size; not a stable format |
| `test_music.sh quick\|divide\|eq\|extrude\|compress` | [wrapper](test_music.sh), [eq_audio](examples/eq_audio.rs) | Real-file listening outputs and byte counts against FLAC |
| `ResearchFrame`, `SegmentModel` | [src/codec.rs](src/codec.rs) | Experimental in-memory prediction only |
| Periodic angular AFT | [src/aft.rs](src/aft.rs) | Continuous angular interpolant, explicit joins and closing interval |
| Arc transforms and frame DFT | [src/spectral.rs](src/spectral.rs) | Continuous vs. sampled transforms; no FFT speed claim |
| SWR1 compression lab | [tools/compression_lab.py](tools/compression_lab.py) | Optional NumPy; independent byte decode and PCM16 error bounds |
| `sinwhisky <zoom> [neighbors] [0\|1] [uniform\|angular]` | [src/main.rs](src/main.rs) | Numeric stdin/stdout, exit 2 on error |
| Optional Python tools | [cross validation](tools/cross_validate.py), [audio comparison](tools/audio_compare.py) | Local research, explicit external prerequisites |
| `test_music.sh INPUT DOWNSCALE_BITRATE UPSCALE OUTPUT` | [Bash wrapper](test_music.sh), [music helper](tools/music_test.py) | Encoded bitrate fraction and decoded sample-rate multiplier |

No HTTP, GUI, service, plugin, or hardware interface exists. The optional Python
lab has a tool-local SWR1 byte envelope; the Rust CLI has no encoded-wire API.

## Build, run, test, debug, and release

From the crate root, with Cargo on PATH:

```sh
cargo build --offline --release
cargo test --offline
cargo test --offline --test contracts
cargo test --offline --test reference_vectors
cargo test --offline --test cli
cargo test --offline --test faft
cargo test --offline --test faft_pursuit
cargo test --offline --test phasor_fft
cargo test --offline --test geometric_faft
cargo test --offline --test adaptive_faft
cargo test --offline --test spectral
cargo test --offline --test aft
cargo test --offline --test bands
cargo test --offline --test layered
cargo fmt --check
cargo clippy --offline --all-targets -- -D warnings
cargo doc --offline --no-deps
cargo run --offline --example codec_research
printf '0 1 0 -1 0\n' | cargo run --offline --quiet -- 3
```

These commands write only generated build output (format checking does not edit
source). Use `cargo fmt` to apply formatting. Rust 1.98.1/macOS arm64 is verified;
minimum supported Rust and other platforms remain unverified. The core needs no
network, credentials, external services, or hardware. A toolchain must already be
installed; no dependency bootstrap is needed beyond Rust.

Optional external validation, after building the release binary:

```sh
python3 tools/cross_validate.py /path/to/sinwhisky.py
python3 tools/audio_compare.py /path/to/music.mp3 /tmp/sinwhisky-listening \
  --start 60 --seconds 8 --factor 4 --ffmpeg /path/to/ffmpeg
./test_music.sh /path/to/music.mp3 0.25 2 /tmp/music-upscaled.wav --seconds 8
python3 -m unittest discover -s tests -p test_music.py -v
cargo build --offline --release --example export_circles
python3 -m unittest discover -s tests -p test_compression_lab.py -v
python3 tools/compression_lab.py /path/to/music.mp3 /tmp/compression-trial \
  --starts 60 180 300 --seconds 1 --ffmpeg /path/to/ffmpeg
```

AFT plots require NumPy/Matplotlib/Pillow/FFmpeg and the Rust `aft_bench` example:

```sh
cargo build --offline --release --example aft_bench
python3 -m unittest discover -s tests -p test_aft_compare.py -v
python3 tools/aft_compare.py /path/to/music.mp3 /tmp/aft-trial \
  --starts 60 180 300 --seconds 2 --frame 1024 --hop 512 --ffmpeg /path/to/ffmpeg
```

Direct geometry trials and their synthetic bridge checks:

```sh
cargo build --offline --release --example geometric_bench --example aft_bench
python3 -m unittest discover -s tests -p test_geometric_compare.py -v
python3 tools/geometric_compare.py /path/to/music.mp3 /tmp/geometric-trial \
  --ffmpeg /path/to/ffmpeg
```

FAFT measurements (NumPy + Cargo; WAV input avoids FFmpeg):

```sh
cargo run --offline --release --example pursuit_bench   # synthetic peak selection JSON
cargo build --offline --release --example faft_bench
python3 -m unittest discover -s tests -p test_faft_compare.py -v
python3 tools/faft_compare.py /path/to/music.wav /tmp/faft-trial
```

Hierarchical merging and adaptive frequencies:

```sh
cargo build --offline --release --example adaptive_bench
python3 -m unittest discover -s tests -p test_adaptive_compare.py -v
python3 tools/adaptive_compare.py /path/to/music.mp3 /tmp/adaptive-trial \
  --ffmpeg /path/to/ffmpeg
```

Equalization/compression paper checks (standard library only, no audio), and
the real-file run (FFmpeg/ffprobe and Cargo; output outside the repository):

```sh
python3 tools/eq_compression_checks.py
./test_music.sh quick /path/to/music.mp3 /tmp/eq-quick
./test_music.sh compress /path/to/music.mp3 /tmp/eq-full --search full --upper 6000 --tolerance 0,16,128
```

Cross-validation needs NumPy/C compiler; `audio_compare.py` needs NumPy/FFmpeg.
The compression lab needs NumPy/FFmpeg and the Rust exporter; its synthetic
tests need NumPy (bridge test additionally needs the built exporter).
The single-file wrapper and its tests need Python standard library/FFmpeg and a
Rust release binary. `FFMPEG`, `PYTHON`, and `SINWHISKY_BIN` override executable
discovery; wrapper environment settings are documented in the README. Music-script
tests skip explicitly if FFmpeg/binary are missing. Audio commands write private
artifacts to their output locations. `--write-fixtures` on the
cross tool mutates test source. Publication is disabled by `publish=false`; there
is no release/deployment workflow. `cargo package --offline --allow-dirty` builds
and verifies a local archive without publishing (missing remote metadata is expected).
`RUST_BACKTRACE=1 cargo test --offline` helps
inspect failures. Optional AFT/FAFT benchmark tools exist; no security/fuzz automation is configured.

## Test ownership map

| Contract | Focused evidence |
|---|---|
| Geometry/extent/branch | `geometry_and_extent`, `angular_sine_and_branch_crossing` |
| Identity, length, model reuse | `lengths_originals_and_reuse` |
| Lines/short input | `constant_ramp_and_two_samples` |
| Exact circle | `exact_circle_without_aperture` |
| Original concept | `paper_angular_example`, `angular_sine_and_branch_crossing` |
| Parameter/size safety | `validation_and_overflow`, `neighbors_and_extreme_finite_inputs` |
| Frame coverage/error/circle reuse | `segments_cover_every_sample_with_bounded_error`, `codec_reuses_circles_and_lines_and_validates_metadata` |
| C/Python compatibility | Frozen vectors; optional live cross-validation tool |
| Hierarchical merges / roots / adaptive peaks / budgets / bounds | `tests/adaptive_faft.rs`; optional `tests/test_adaptive_compare.py` |
| Geometry-first circles / spectral sums / cropped windows / zoom | `tests/geometric_faft.rs`; optional `tests/test_geometric_compare.py` |
| FAFT exactness/derivatives/AFT same curve/zoom images/peaks/validation | `tests/faft.rs`, doctest; optional `tests/test_faft_compare.py` |
| Paper identities: inscribed-angle rates, crest rate, endpoint form, line limit, element × array factor | `tests/faft.rs` (`rotation_rates_are_twice_the_inscribed_angles_of_the_anchor_triangle`, `arc_transform_is_fixed_by_its_endpoints`, `line_is_the_infinite_radius_limit_of_an_arc`, `identical_rotations_factor_into_element_and_array_terms`) |
| Endpoint kernel: flat-arc accuracy, derivatives as moment integrals | `flat_arcs_keep_relative_accuracy_away_from_resonance`, `derivatives_match_quadrature_of_moments` |
| Kernel = tone transform, Table 1 widths, energy, joint fit, certificate, sidelobes, hidden weak tone, in-lobe separation, peak ranges, tracker identities | `tests/faft_pursuit.rs` |
| FFT phasor kernel/circles/precision/tree/line limits | `tests/phasor_fft.rs`; optional `tests/test_phasor_compare.py` |
| CLI boundary | `command_line_modes_and_failures` |
| Music wrapper/blocks/bitrate/fractional upscale | Optional `tests/test_music.py` integration suite |
| Periodic AFT / benchmark conventions | `tests/aft.rs`, optional `tests/test_aft_compare.py` |
| Circle transforms / predictor DFT | `tests/spectral.rs` |
| SWR1 byte decode/error/limits/Fourier/subarcs | Optional `tests/test_compression_lab.py` |
| Banded fit = dense fit, exact filtering vs Runge-Kutta, zero-phase response, division completeness, tones whole, DC snap | `tests/bands.rs` |
| SWL2 round trips with every layer and tolerance, stereo modes, LPC on tones vs noise, parametric mode, auto never worse than correction alone, malformed streams | `tests/layered.rs` |
| EQ/compression paper: masked-edge leakage, two-way solve, filtered rotations, spirals, gradient tones | Optional `tools/eq_compression_checks.py` (standard library) |

Known test gaps: SWR1 fuzzing/security and cross-platform rounding, other platforms,
performance budgets, broad audio corpus, blind listening, or realtime behavior.

## Data, privacy, and compatibility boundaries

Source samples are canonical inputs; analysis is derived. Research frames retain
predictors/metadata, not source PCM. Models with public fields are inspectable and
constructible, but manually constructed descriptors are not a validated wire format.
Private frame fields prevent mutation from invalidating checked coverage.

No network or secrets are needed. Keep recordings, output audio, and local paths
outside version control. The numeric CLI buffers all input and has no configurable
memory cap; do not expose it as an untrusted network service. The tool-local SWR1
parser has explicit limits/versioning and malformed-input tests, but no fuzzing
or security validation; do not promote it to an untrusted service.
Do not serialize native Rust layout or claim compatibility with legacy Python files.
This initial unpublished API may evolve; document every behavior change.

## Current status, known gaps, and roadmap

- **Implemented:** reusable analysis, uniform/angular resampling, numeric CLI,
  standalone Cargo tests, optional live comparison and listening scripts, and
  `test_music.sh` with encoded-bitrate reduction and block-based sample-rate upscaling.
- **Experimental:** FAFT, the exact continuous transform of circle curves, with
  continuous relevant peaks (measured, not complete) and their selection as a
  jointly fitted tone model with a residual certificate and frame tracking
  (synthetic evidence only); hierarchical
  children/checked variable-support merges and bounded adaptive relevant-frequency
  queries; the geometry-first sampled-spectrum control with peak-error limits;
  the FFT phasor control's rotating-circle selection with per-circle precision and
  verified RMS limits; retained integral AFT reference; mono `ResearchFrame`,
  circle transforms/predictor DFT, and an
  optional NumPy circle/subarc/Fourier lab with full byte/error measurements.
  Three short Beethoven excerpts show no material circle-specific coding gain.
- **Known gaps:** uniform branch mismatch; no stable compression format, native
  audio I/O, streaming, stereo coupling, perceptual validation, or cross-platform CI.
- **Planned:** see [ROADMAP.md](ROADMAP.md), especially sliding FAFT windows,
  continuous circle joins, peak-list completeness, off-grid/residual circle
  search, continuity, shared distortion allocation, geometry quantization, better residual
  coding, broader byte/error measurements and a separately validated FLT implementation.
  The equalization/compression draft ([paper](docs/PAPER_EQ_COMPRESSION.md)) has
  an implementation (banded fit, exact analog filters, divider/EQ/extrusion, SWL2
  coder with LPC, gradient groups, SinWhisky anchors and automatic choices). On
  three 5 s excerpts lossless SWL2 is 7.7-11.0% smaller than FLAC; gradient and
  anchor layers paid only on a quiet passage with a tolerance (paper 6.2).
- **To do (SWL2 correction layer):** add cascaded NLMS stages after the LPC in
  [correction.rs](src/layered/correction.rs), as in OptimFROG/Monkey's Audio:
  sign-sign or normalized LMS filters of decreasing length (e.g. 256, 32, 16 taps)
  adapting sample by sample on the reconstructed history, in integer arithmetic so
  the decoder stays exact on every platform. Keep the per-block choice by coded
  size (signal the cascade in the block header, bump the SWL2 version and update
  `docs/SWL2.md`), check lossless/bounded round trips in `tests/layered.rs`, and
  remeasure the three excerpts against FLAC and the current LPC-only stream.

## Task start and handoff checklist

1. Inspect working-tree status and all applicable instructions; preserve user work.
2. Identify the owning file, coordinate/mode contract, and focused tests above.
3. Check sample/metadata boundaries and whether a change breaks compatibility.
4. Implement with tests; keep real behavior separate from experimental/planned work.
5. Update README, design, roadmap, and this map when their facts change.
6. Run relevant tests, format/lint, local-link validation, and `git diff --check`.
7. Report actual outcomes, skipped checks, known limitations, and remaining work;
   leave no machine-specific audio or temporary toolchain in the source tree.
