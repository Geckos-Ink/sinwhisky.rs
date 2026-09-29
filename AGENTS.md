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
The AFT interpolation policy is separate from finite-zoom angular resampling.
The continuous Nyquist coefficient need not be real. `tests/aft.rs` checks
coverage, original geometry reuse, quadrature, sinc² linear control, phase shifts,
conjugacy, constants/short inputs, and validation.

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

### [docs/COMPRESSION.md](docs/COMPRESSION.md)

Owns Fourier conventions/derivation, subarc limits, candidate search, SWR1 format,
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
cargo test --offline --test spectral
cargo test --offline --test aft
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
inspect failures. No benchmark/security/fuzz automation is configured yet.

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
| CLI boundary | `command_line_modes_and_failures` |
| Music wrapper/blocks/bitrate/fractional upscale | Optional `tests/test_music.py` integration suite |
| Periodic AFT / benchmark conventions | `tests/aft.rs`, optional `tests/test_aft_compare.py` |
| Circle transforms / predictor DFT | `tests/spectral.rs` |
| SWR1 byte decode/error/limits/Fourier/subarcs | Optional `tests/test_compression_lab.py` |

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
- **Experimental:** mono `ResearchFrame`, circle transforms/predictor DFT, and an
  optional NumPy circle/subarc/Fourier lab with full byte/error measurements.
  Three short Beethoven excerpts show no material circle-specific coding gain.
- **Known gaps:** uniform branch mismatch; no stable compression format, native
  audio I/O, streaming, stereo coupling, perceptual validation, or cross-platform CI.
- **Planned:** see [ROADMAP.md](ROADMAP.md), especially geometry quantization, better residual
  coding, broader byte/error measurements and a separately validated FLT implementation.

## Task start and handoff checklist

1. Inspect working-tree status and all applicable instructions; preserve user work.
2. Identify the owning file, coordinate/mode contract, and focused tests above.
3. Check sample/metadata boundaries and whether a change breaks compatibility.
4. Implement with tests; keep real behavior separate from experimental/planned work.
5. Update README, design, roadmap, and this map when their facts change.
6. Run relevant tests, format/lint, local-link validation, and `git diff --check`.
7. Report actual outcomes, skipped checks, known limitations, and remaining work;
   leave no machine-specific audio or temporary toolchain in the source tree.
