# SinWhisky.rs

A standalone, dependency-free Rust library for circle-based signal resampling,
with reusable geometry for studying compression and sound codecs. This is an
initial research implementation, not a finished audio codec or a method for
recovering lost high frequencies.

The folder is ready to become its own repository: its build, tests, license,
whitepaper, and documentation are self-contained. It currently lives inside the
Python/C repository; no nested Git repository or remote is required to build it.

## What was retained

- From the portable C/Python implementation: local circle fitting, bounded
  aperture, uniform-x evaluation, configurable neighbor blending, original-sample
  preservation, and linear fallback.
- From the recovered 2014 implementation: a separate **equal-angle** mode with
  the original range aperture and finite-zoom sine window. Degenerate lines and
  short inputs are repaired. This is not bug-for-bug reproduction.
- From the Python compressor experiments: the idea of storing circle geometry
  and covered sample ranges. Rust uses explicit typed models and checks coverage
  and reconstruction error instead of copying the old unversioned binary format.

See [design and provenance](docs/DESIGN.md), the included
[2014 whitepaper](<docs/SinWhisky e FLT.pdf>), and the
[agent reference](AGENTS.md).

The [FAFT manuscript](docs/PAPER_FAFT.md) derives the transform from circle
geometry. It covers rates as inscribed angles, the line as the infinite-radius
limit, the endpoint form, and time distance as phase. Its
[web copy](docs/PAPER_FAFT-net.md) has the same text with absolute figure URLs.
A draft [extension on equalization and compression](docs/PAPER_EQ_COMPRESSION.md)
applies the transform to band division, exact filtering of circle curves and a
proposed layered coding scheme. Its identities are checked by
`python3 tools/eq_compression_checks.py`. A first implementation (experimental)
divides a track into bands that add back exactly, equalizes, isolates a band, and
codes it in layers within an error tolerance. Try it on a real file (FFmpeg and
Cargo required; the output directory must be new and outside the repository):

```sh
./test_music.sh quick /path/to/music.mp3 /tmp/eq-quick
./test_music.sh eq /path/to/music.mp3 /tmp/eq-bass --edges 250,2000 --gains-db 6,0,-12
./test_music.sh extrude /path/to/music.mp3 /tmp/eq-bass-only --band 40,250
./test_music.sh compress /path/to/music.mp3 /tmp/eq-6k --upper 6000 --tolerance 0,16
```

`quick` decodes 5 s from 60 s and writes the bands, tonal and residual tracks, an
equalized mix, an isolated band and its remainder, decoded streams, `source.wav`,
a FLAC control and `report.json`. On one excerpt the coder is not yet better than
FLAC (paper Section 6.1).

## Build and try

Requires Rust with Cargo. Verified with Rust 1.98.1 on macOS arm64; other targets
are not yet verified. The crate uses only the Rust standard library, forbids
unsafe code, and builds offline once Rust is installed.

```sh
cargo build --offline --release
cargo test --offline
cargo fmt --check
cargo clippy --offline --all-targets -- -D warnings

# Insert 3 samples between each original pair; output is one f64 per line.
printf '0 1 0 -1 0\n' | cargo run --offline --quiet -- 3

# zoom, neighbors, aperture (0/1), mode
printf '3 7 9 4 6\n' | cargo run --offline --quiet -- 3 1 1 angular
cargo run --offline --example codec_research
```

The command is a numeric stdin/stdout CLI, not a WAV or MP3 decoder.
`zoom=3` means a nominal 4x sample rate, not a bitrate setting. Output length is
`(n-1)*(zoom+1)+1`; empty input stays empty and singleton input stays unchanged.
Original samples are copied exactly. Non-finite input, invalid parameters,
length overflow, and failed output reservations return errors.

## Library usage: calculate circles once

```rust
use sinwhisky::{Analysis, AnalysisParams, ResampleParams};
use sinwhisky::codec::ResearchFrame;

fn main() -> Result<(), sinwhisky::Error> {
    let samples = [0.0, 1.0, 0.0, -1.0, 0.0];
    let analysis = Analysis::new(&samples, AnalysisParams::default())?;

    let four_times = analysis.resample(ResampleParams::new(3))?;
    let eight_times = analysis.resample(ResampleParams::new(7))?;
    for model in analysis.models() {
        println!("{model:?}");
    }

    // Experimental mono frame: reuse those same circles as segment candidates.
    let frame = ResearchFrame::from_analysis(&analysis, 44_100, 0.01)?;
    let reconstructed = frame.decode()?;
    println!("{:?}", frame.metrics());
    assert_eq!(reconstructed.len(), samples.len());
    assert_eq!(four_times.len(), 17);
    assert_eq!(eight_times.len(), 33);
    Ok(())
}
```

`Analysis` owns source samples and immutable access to the fitted models.
`LocalCircle` exposes its sample anchor, aperture, center, radius, branch, and
three unwrapped angles. Sample positions and geometric circle centers are
separate concepts. Channels are analyzed independently; the library is mono.

## Two different interpolation modes

| | `UniformX` (default) | `OriginalAngular` |
|---|---|---|
| Evaluation | `y = cy ± sqrt(r² - (x-cx)²)` at uniform x | Equal angular increments on each adjacent arc |
| Standard analysis preset | `AnalysisParams::default()` | `AnalysisParams::original()` |
| Aperture | Pairwise sample range, clamped to 1..12 | Pairwise range, no clamp; flat triplets use 1 |
| Window | `cos(pi*u/(2*neighbors))` | `cos(pi*q*u/(2*(q+1)))`, `q=zoom+1` |
| Support | Configurable neighbors ≥ 1 | Adjacent triplets only; neighbors must equal 1 |
| Degenerate model | Skip it; use segment line if no circle contributes | Blend its centered line |

Use `resample(samples, ResampleParams::original(zoom))` to select both original
presets automatically. With a reusable `Analysis`, fitting and synthesis are
independent: choose the matching `AnalysisParams` yourself.

Uniform mode matches the newer C/Python semantics within floating-point tolerance.
It is **not** equivalent to the recovered 2014 algorithm. A fixed semicircle may
exclude the outer samples on steep triplets, causing inserted values to approach
a different value than a copied original sample. The regression tests preserve
this visible compatibility limitation. Angular mode can traverse both branches.
Neither mode guarantees perceptual improvement, and normalization can alter the
uniform mode's shape because its aperture bounds use absolute amplitude units.

## Compression research foundation

`ResearchFrame` stores sample rate, original sample count, ordered `Segment`s,
and measured maximum/RMS reconstruction error. Each segment has half-open sample
coverage and one of `Circle`, `Line`, or `Literal` predictors. A greedy experiment
extends a line and an existing circle candidate, picks the longer valid span,
and uses literal samples when needed. Every covered source sample must satisfy
the requested absolute error; there are no silent gaps or overlap overwrites.

Circle segments evaluate **uniform x**, even if the analysis used original range
aperture. They do not use angular resampling or blend across segment boundaries.
The current experiment reuses the circle centered at `start+1`; it does not search
all possible circles or refit a segment. Inspect `frame.segments()` to study its
choices. `frame.decode()` works without the original `Analysis`.

`ResearchFrame` remains an in-memory approximation model without a bitstream.
Segment counts are not compression ratios. A separate optional
[circle/Fourier compression lab](docs/COMPRESSION.md) now measures actual bytes
for recursive circle/line predictors, Fourier residuals, and raw/delta/FLAC
controls. Its experimental SWR1 streams are decoded independently and checked
against PCM16 error bounds; this is not a stable audio format or a quality claim.
The resampler's contracts and dependencies are unchanged.

**FAFT** ([definition and measurements](docs/FAFT.md)) is the Fourier transform
*of the determined circles*. Each equal-angle half of a fitted circle is a finite
rotation `center + radius*sin(phase + rate*u)` on its support. `faft::CircleCurve`
collects them, and `Faft::value(omega)` integrates the curve exactly, in closed
form, at any real frequency, including between DFT bins and above the source
Nyquist. It uses no zoom, sampling, FFT or quadrature. Queries use a
rotate-and-add rule: each rotation's closed transform is multiplied by a chained
support phase and added. There are no trigonometric calls per rotation, so a
query costs O(rotations). `Faft::peaks` returns a variable list of relevant peaks
at continuous frequencies: a scan spaced by the window's duration, Newton
refinement with exact analytic derivatives, merging, and floors. Zoom stays a
SinWhisky synthesis tool that samples the curve. A zoom-z sampled spectrum equals
FAFT plus spectral images, which the tests check. On 63 Beethoven windows, an
exact continuous query took 2.4–10.6 µs and a converged peak list 1.4–10.9 ms,
against 17–49 ms for the earlier adaptive search. An FFT of the samples remains
faster for a complete fixed-bin spectrum of the samples.

A maximum is not a component: the window kernel shares each peak's height with
its neighbours, so a list of maxima carries leakage, sidelobes and hidden weak
tones. `Faft::pursue` selects relevant frequencies as a jointly fitted tone
model. It uses closed-form Gram entries from the kernel, joint Gauss-Newton
frequencies and significance pruning, and returns each tone's own height, the
residual energy and a bound on the unexplained spectrum at every frequency.
`PeakTracker` carries the selection across consecutive windows without
rescanning. `Faft::kernel`, `kernel_range` and `peak_range` measure how far a
peak's level extends. On a synthetic test with known partials, pursuit raised
recall from 89% to 100% with no spurious peaks, and tracking cost a median 3.5 ms
per window against 6.4 ms for the scan (see
[FAFT.md](docs/FAFT.md#selecting-relevant-frequencies)).

```rust
use sinwhisky::faft::{CircleCurve, PeakParams, Weighting};
use sinwhisky::geometric_faft::{GeometryFrame, GeometryParams, HierarchyParams};
fn main() -> Result<(), sinwhisky::Error> {
    let samples: Vec<f64> = (0..512).map(|i| 1000.0 * (0.3137 * i as f64).sin()).collect();
    let params = GeometryParams { max_error: 0.5, max_span: 64, ..GeometryParams::default() };
    let (frame, _) = GeometryFrame::hierarchical(&samples, params, HierarchyParams::default())?;
    let faft = CircleCurve::from_geometry(&frame)?.transform(0..512, Weighting::Hann)?;
    let off_grid = faft.value(0.31)?; // any real omega, including above pi
    let peaks = faft.peaks(PeakParams::default())?.peaks;
    assert!(!peaks.is_empty() && off_grid.re.is_finite());
    Ok(())
}
```

The earlier **geometry-first sampled spectrum** (a control) fits circles through
waveform points, then retrieves the *sampled* (optionally zoomed) DTFT of the
predicted sound from retained centers, radii, and angular phases using finite
geometric sums. Its `GeometryFrame` is also the circle-determination stage used
by FAFT. `geometric_faft::GeometryFrame` searches point
spacing, interior phase, aperture, and per-circle parameter precision, checking a
maximum source-sample error bound. `dtft` and `hann_dtft` query time-local windows
without another fit; `spectrum_rotations` computes a whole grid using batched
rotations. No FFT or integral is used in that path. Full query cost is O(PK), with
P retained angular/line pieces and K frequencies. The [geometry experiments](docs/COMPRESSION.md#geometry-first-sampled-spectrum-earlier-faft-attempt)
compare point banks, precision, phase, zoom, actual MP3 bitrate controls, and two
implementation attempts on the supplied Beethoven track.

A separate `GeometryFrame::hierarchical` preset splits failed parent circles
into source-informed children on variable integer supports, then refits adjacent
children into one checked circle or line. `prepare_spectrum` reuses those models
at arbitrary frequencies, with optional sampling zoom and Hann weighting.
`adaptive_faft::AdaptiveSpectrum` (sampled-predictor control) returns a variable list of relevant peaks,
with separate depth, query-budget, frequency-width, relevance, and merging limits.
The arithmetic rotation kernel prepares circle roots once; it uses finite sums,
not integrals or an FFT. Flatness pruning is a reported heuristic and can miss
peaks. Finer frequency queries do not create missing sample information.
See the [hierarchy/adaptive benchmark](docs/COMPRESSION.md#hierarchical-circles-and-adaptive-relevant-frequencies)
for measured costs, off-bin estimates, precision/zoom controls, and a GIF.

```rust
use sinwhisky::geometric_faft::{GeometryFrame, GeometryParams};
fn main() -> Result<(), sinwhisky::Error> {
    let samples: Vec<f64> = (0..256).map(|i| (i as f64 * 0.2).sin()).collect();
    let geometry = GeometryFrame::new(&samples, GeometryParams {
        max_error: 0.01, ..GeometryParams::default()
    })?;
    let local = geometry.hann_dtft(0.2, 64..192)?;
    let spectrum = geometry.spectrum_rotations(1)?;
    assert!(geometry.stats().max_error <= 0.01);
    assert!(local.re.is_finite() && spectrum.len() == 129);
    Ok(())
}
```

The **FFT phasor control** (the first prototype once called FAFT; module renamed
from `faft`) uses rotating phasor circles and their real projections.
`phasor_fft::transform` is a dependency-free radix-2 FFT (O(N log N));
`phasor_fft::PhasorFrame` then chooses constant/line baselines, retained circles, per-circle
precision, and optional shorter child supports. A straight line is represented
explicitly as a zero-curvature limit, never an infinite floating-point radius.
The full tree search is O(N log² N); accepted reconstructions meet the requested
sample-domain RMS bound. This is a model optimization built on an FFT, not a
claim to beat FFT or find the global fewest circles. Estimated model bits are
not an encoded bitrate. See the [phasor measurements and rotating-circle GIF](docs/COMPRESSION.md#fft-phasor-control-earlier-faft-prototype).

```rust
use sinwhisky::phasor_fft::{PhasorFrame, PhasorParams};
fn main() -> Result<(), sinwhisky::Error> {
    let samples: Vec<f64> = (0..256).map(|i| (i as f64 * 0.2).sin()).collect();
    let frame = PhasorFrame::new(&samples, PhasorParams {
        max_rmse: 0.01, min_block: 32,
    })?;
    let decoded = frame.decode()?; // No source samples required.
    assert_eq!(decoded.len(), samples.len());
    assert!(frame.stats().rmse <= 0.01);
    Ok(())
}
```

The Rust `spectral` module also computes analytic transforms of equal-angle arcs,
numerical transforms of uniform-x arcs, and DFT bins directly from retained frame
predictors. `aft::AngularFrame` extends this to a complete periodic angular
interpolant: prepare geometry once, evaluate it at fractional positions, and
integrate its spectrum analytically. This retained **AFT reference preset** is a continuous
Fourier integral, distinct from the sampled DFT; its current full spectrum takes
O(N²). The [AFT/FFT benchmark](docs/COMPRESSION.md#angular-fourier-transform-aft-reference-preset-benchmark)
includes measured Beethoven spectra, spectrograms, runtime plots, and a GIF.
Subdividing a parent circle alone cannot predict unknown error; source-informed child models
or residual information are required.

```sh
cargo test --offline --test faft --test faft_pursuit --test spectral --test aft --test phasor_fft --test geometric_faft --test adaptive_faft
# Optional: Python + NumPy + Cargo; a PCM16 WAV input needs no FFmpeg.
python3 tools/faft_compare.py /path/to/music.wav /tmp/faft-trial
# Optional: Python + NumPy + Matplotlib + Pillow + FFmpeg.
python3 tools/aft_compare.py /path/to/music.mp3 /tmp/aft-trial
python3 tools/phasor_compare.py /path/to/music.mp3 /tmp/phasor-trial
python3 tools/geometric_compare.py /path/to/music.mp3 /tmp/geometric-trial
python3 tools/adaptive_compare.py /path/to/music.mp3 /tmp/adaptive-trial
# Optional: Python + NumPy + FFmpeg. Output must be new, outside the repository.
python3 tools/compression_lab.py /path/to/music.mp3 /tmp/compression-trial \
  --starts 60 180 300 --seconds 1 --tolerances 0 16 128
```

See the [experiment and format](docs/COMPRESSION.md) for reproduction, results,
and limitations, and [the roadmap](ROADMAP.md) for further work.

## Ports: WebAssembly SIMD and WebGPU

[ports/](ports/README.md) holds reference ports of the original-angular resampler, the
AFT, and FAFT on the AFT's curve (values, scans, derivatives, Newton-refined peaks):
AssemblyScript compiled to WebAssembly with scalar and SIMD (`f64x2`) paths, and WGSL
kernels for WebGPU (f32, geometry from the WebAssembly port). They are checked against
vectors from `cargo run --offline --example port_vectors`; the WebAssembly port agrees
with the crate to about 4e-14 of each output's peak, the WebGPU kernels to about 1e-6.
They are not Cargo dependencies and do not change the library.

## Verification and listening experiments

For a single output music file, use [test_music.sh](test_music.sh). It requires
Python 3 (standard library only), FFmpeg with `libmp3lame`, and Cargo. It builds
the Rust release binary automatically and accepts paths containing spaces:

```sh
# INPUT, downscale_bitrate, upscale, OUTPUT; full track unless --seconds is set.
./test_music.sh '/path/to/music.mp3' 0.25 2 /tmp/music-upscaled.wav

# Short excerpt, original angular interpolation, fractional upscale.
./test_music.sh '/path/to/music.mp3' 0.5 1.5 /tmp/music-angular.flac \
  --start 60 --seconds 15 --mode angular

# Configuration through environment variables is also supported.
INPUT_MUSIC='/path/to/music.mp3' DOWNSCALE_BITRATE=0.25 UPSCALE=2 \
  OUTPUT_MUSIC=/tmp/music-upscaled.wav ./test_music.sh
```

`0 < downscale_bitrate <= 1` is the **encoded bitrate fraction**. For example,
320 kbps × 0.25 targets 80 kbps MP3. The script chooses the highest supported MP3
bitrate no greater than the requested rate (maximum 320 kbps), reports the choice,
and rejects requests below 8 kbps. A factor of 1 skips lossy re-encoding. Source
bitrate comes from audio metadata; use `--source-bitrate KBPS` to override it.
Re-encoding may also change the codec's sample rate, which is reported.

`upscale > 1` multiplies the **decoded sample rate**, keeping pitch and duration.
Integer factors use Rust directly; fractional factors first use the next integer
Rust factor, then FFmpeg converts to the requested rounded sample rate. This
changes sampling density; it cannot reverse information lost through compression.
Output must be a new `.wav` or `.flac` file. WAV uses PCM16/RF64 when needed; FLAC
uses 24-bit PCM. Stereo is processed channel by channel (`--channels 1` selects mono).
The default gain is 0.8 for headroom; change it with `--gain`. The script reports
Rust output peaks and samples above full scale before encoding.

Full tracks are processed in two-second blocks with overlapping source samples
so the neighboring circles remain continuous across block boundaries. Temporary
files are removed on exit and existing outputs are preserved. The helper needs no
NumPy. Set `FFMPEG=/path/to/ffmpeg` and `PYTHON=/path/to/python3` when they are not on
PATH; `SINWHISKY_BIN=/path/to/sinwhisky` skips the automatic Cargo build. Run
`./test_music.sh --help` for all options.

Core tests and frozen Python reference vectors run with Cargo alone. An optional
live comparison needs Python + NumPy, a C compiler, and the ancestor checkout:

```sh
cargo build --offline --release
python3 tools/cross_validate.py /path/to/sinwhisky.py
```

It checks Rust against both C and Python. `--write-fixtures` deliberately updates
the frozen vectors only after those checks pass; review that diff carefully.

After a release build, optional music-script integration tests check block
boundaries against whole-signal Rust output, bitrate reduction, fractional-rate
duration, invalid factors, and output preservation:

```sh
python3 -m unittest discover -s tests -p test_music.py -v
```

They require FFmpeg (or `FFMPEG`) and the release binary (or `SINWHISKY_BIN`);
missing prerequisites are reported as skipped tests.

For local music experiments, install NumPy and provide FFmpeg with `libmp3lame`:

```sh
python3 tools/audio_compare.py /path/to/music.mp3 /tmp/sinwhisky-listening \
  --start 60 --seconds 8 --factor 4 --ffmpeg /path/to/ffmpeg
```

The destination must be new or empty. The script decodes a short PCM16 excerpt at
44.1 kHz, applies shared 0.8 gain and an anti-alias filter, reduces its sample rate,
and writes linear/uniform/angular reconstructions. It also writes separate 64/128
kbps MP3 controls from the full-rate reference and a JSON metrics report. It treats
channels separately, excludes filter-edge samples from metrics, and compares with
both the source and a bandlimited reference. It never changes the input recording.

The audio tools are optional and are not library dependencies. Audio artifacts
belong outside the source tree; do not commit private recordings. See the initial
[Beethoven experiment](docs/LISTENING.md) for results and interpretation limits.

## License

[MIT](LICENSE). Original concept and ancestor code: Riccardo Cecchini.
