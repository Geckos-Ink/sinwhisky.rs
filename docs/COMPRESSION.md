# Angular Fourier and circle compression experiments

The current focus is Fourier analysis through angular circle functions. The
AFT/FFT comparison below extends the earlier single-arc integrals to complete
frames and records actual measurements from the supplied Beethoven track.
The earlier compression lab remains documented below as a separate experiment.
Neither path implements FLT or establishes recovery of missing source information.

## Angular Fourier Transform (AFT): current benchmark

The 2026-09-30 implementation now computes a complete frame spectrum from angular
circle functions. **AFT** is the project name for this continuous angular model
plus its analytic Fourier integral; it is not a new Fourier kernel or another
algorithm for the sampled DFT. Compression is outside this experiment.

### Definition and implementation

[src/aft.rs](../src/aft.rs) provides `AngularFrame::new`, `evaluate`, `fourier`,
`spectrum`, and fitting statistics. It fits periodic source halos once. On the
interval between samples i and i+1, it averages the right angular half of the
circle anchored at i and the left angular half of the circle anchored at i+1.
The last interval closes to sample 0. Degenerate halves use the source-pair line.
A circle half is rejected if its endpoint discrepancy exceeds
`1e-8 * max(1, abs(left), abs(right))`; there were no such rejections in this run.
This preserves the circle geometry rather than refitting it during integration.

One frame has period N and covers `[0,N]`. Let g(t) denote this angular
interpolant and x[n] the input samples:

```text
AFT[k] = integral from 0 to N of g(t) * exp(-i*2*pi*k*t/N) dt
FFT[k] = sum from n=0 to N-1 of x[n] * exp(-i*2*pi*k*n/N)
```

Each angular half is integrated analytically using the sinc formula below.
Line halves use sinc and its derivative. Contributions receive their global
midpoint phase and are accumulated with compensated summation. Preparing geometry
costs O(N); one frequency costs O(N); all nonnegative bins currently cost O(N²).
`fourier` also accepts arbitrary positive/negative radians-per-sample frequencies.
There is no oversampling or FFT hidden inside the AFT integration.

This equal average is an explicit new interpolation policy, separate from the
resampler's finite-zoom angular window. Joins reproduce source values within
checked fitting roundoff. The geometry fit is data-dependent: the sample-to-AFT
mapping is generally nonlinear even though integration of a fixed function is
linear. AFT coefficients are not discrete inverse-FFT inputs with an exact
reconstruction guarantee. In particular, the continuous coefficient at Nyquist
need not be real.

### Comparable inputs and controls

The supplied Beethoven track was decoded to **44.1 kHz mono PCM16**, using
FFmpeg's mono downmix, with two-second excerpts at **60, 180, and 300 seconds**.
Each excerpt supplies **171 windows**, for **513 windows total**. Frames contain
1,024 samples (23.22 ms), hop 512 (11.61 ms), and 513 nonnegative frequency bins
spaced by 43.0664 Hz. No source recording or invertible spectral arrays are stored
in this repository.

Both methods receive the identical samples after a **periodic Hann** window,
`w[n] = 0.5 - 0.5*cos(2*pi*n/N)`. Windowing happens before fitting; this is not a
claim to integrate an unwindowed angular model multiplied by a continuous Hann.
The fitting preset is `AnalysisParams::original()` (`SignalRange`, determinant
epsilon 4e-6), in **PCM16 amplitude units**, not normalized unit amplitude.
The absolute fit epsilon makes that scale relevant. Of 1,050,624 prepared
halves, 1,049,496 (99.893%) were circles; the rest were lines.

The FFT reference is NumPy's float64 `rfft`, with the same negative-exponential
convention and no forward normalization. See the [NumPy FFT definition](https://numpy.org/doc/stable/reference/routines.fft.html)
and the [Hann window reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hann.html).
A third curve shows the exact Fourier integral of the **periodic linear
interpolant**, `FFT[k] * sinc(k/N)^2`, where this sinc is `sin(pi*x)/(pi*x)`.
That control exposes ordinary interpolation effects rather than attributing all
AFT/FFT differences to numerical error.

For plotted interior positive-frequency bins, magnitude is
`20*log10(2*abs(coefficient)/(sum(w)*32768))` dBFS. DC and Nyquist are omitted from
the plots to avoid mixing their continuous/discrete one-sided conventions. The
statistical complex comparisons use the full reported nonnegative coefficient
arrays. Magnitude/phase statistics exclude DC/Nyquist and require both spectra
to exceed the corresponding FFT window's peak minus 60 dB.

### Spectra and differences

| Excerpt | Relative complex L2 difference from FFT | Median absolute magnitude difference, strong bins | Median absolute phase difference, strong bins | Dense-check discrepancy, q=128 |
|---|---:|---:|---:|---:|
| 60 s | 0.0588% | 0.0352 dB | 0.0056° | 2.11e-08 |
| 180 s | 0.0316% | 0.0238 dB | 0.0026° | 1.34e-08 |
| 300 s | 0.0876% | 0.0638 dB | 0.0109° | 2.21e-08 |

AFT and FFT selected the same dominant non-DC bin in **all 513 windows**. The
small full-spectrum differences are dominated by strong low-frequency energy;
within the 8–22.05 kHz band, relative complex differences are **10.53%, 13.97%,
and 8.49%**, respectively. The FFT remains the reference for the sampled DFT;
these differences do not show that either method is a more accurate estimate
of the unknown continuous recording.

![FFT and AFT mean spectra with linear-interpolation control](figures/aft-20260930/aft-spectra.png)

Left: RMS coefficient magnitude across each excerpt's windows. Right: ratios
of those RMS magnitudes, not an average of per-window dB ratios. The angular
model generally preserves more high-frequency magnitude than linear interpolation
in these examples. The extra very-low-level energy above the track's steep
high-frequency cutoff in the 300-second excerpt is model-generated spectral
content, **not evidence of recovered information**. No enhanced frequency
resolution or listening-quality conclusion follows. [Vector plot](figures/aft-20260930/aft-spectra.svg).

![FFT and AFT spectrograms of the 60-second excerpt](figures/aft-20260930/aft-spectrogram.png)

Both spectrograms use identical magnitude color limits. The difference panel
masks FFT bins below −90 dBFS and saturates color at ±1 dB to expose differences
in stronger bins; it is not an error map against known continuous audio.
[Vector plot](figures/aft-20260930/aft-spectrogram.svg).

![Animated FFT versus AFT spectra from the supplied Beethoven track](figures/aft-20260930/aft-evolution.gif)

The silent GIF shows **48 measured windows**, selected across 60.012–61.985 s,
at 10 frames/s. Playback is slowed; the displayed time is the actual window
center on the source track. Axes and normalization stay fixed throughout. The
difference plot zooms to −1..+0.5 dB and masks FFT bins below −90 dBFS.

### Current runtime benchmark

Measured on macOS 27.0.1 arm64, Rust 1.98.1 release mode, Python 3.12.14,
NumPy 2.3.5, and FFmpeg 7.1.1. Matplotlib 3.10.8/Pillow 12.1.1 rendered the figures.
The hardware model was not available to the harness. These are measurements of
these two implementations and **different transform targets**, not a universal
algorithm speed ranking.

The size sweep uses the same central track location of the first excerpt.
After warmup, medians use 11 AFT/fit calls and 101 FFT calls per size. Calls are
timed **inside their processes**. Output allocation is included; audio decoding,
windowing, process startup, text/JSON transfer, plotting, and dense verification
are excluded. AFT geometry preparation is separate and must be added when a frame
is new. Complete repeat timings are retained in the JSON snapshot.

| Samples N | FFT median, µs | Geometry preparation, µs | AFT on prepared geometry, ms | AFT / FFT time |
|---|---:|---:|---:|---:|
| 256 | 3.417 | 26.208 | 0.818 | 239× |
| 512 | 4.375 | 57.958 | 3.278 | 749× |
| 1,024 | 5.917 | 114.792 | 13.119 | 2,217× |
| 2,048 | 9.375 | 222.000 | 52.220 | 5,570× |
| 4,096 | 16.167 | 460.417 | 207.780 | 12,852× |

The 1,024-sample track-window medians are approximately **13 ms for AFT**,
**0.11 ms to prepare geometry**, and **6 µs for FFT**. The current AFT is therefore
much slower; the near-quadratic growth is visible. Analytic integration provides
a precise spectrum of the chosen angular curve, not an FFT speedup. No realtime
or other-platform claim is made.

![AFT and FFT runtime scaling and dense-integration convergence](figures/aft-20260930/aft-benchmark.png)

[Vector plot](figures/aft-20260930/aft-benchmark.svg).

### Independent integration check

For the central window of each excerpt, the same angular curve was evaluated
at q=8, 32, and 128 **midpoints per original interval**, then independently
integrated using NumPy's FFT. The integration step and midpoint phase are restored:

```text
numeric[k] = rfft(g((j+0.5)/q))[k] / q * exp(-i*pi*k/(N*q))
```

This check compares the same continuous waveform, unlike the original-sample
FFT comparison. Relative complex L2 discrepancies decrease by about 16× when
q increases by 4×, consistent with midpoint quadrature convergence:

| Excerpt | q=8 | q=32 | q=128 |
|---|---:|---:|---:|
| 60 s | 5.4e-06 | 3.37e-07 | 2.11e-08 |
| 180 s | 3.43e-06 | 2.14e-07 | 1.34e-08 |
| 300 s | 5.65e-06 | 3.53e-07 | 2.21e-08 |

These are convergence measurements on **three selected windows**, not a certified
error bound for every circle. Synthetic tests additionally verify known constants,
phase shifts, conjugacy, periodic coverage, original arc reuse, short inputs,
line interpolation's sinc² spectrum, and independent quadrature. Near-flat arcs
with extreme radii can still be ill-conditioned and need wider numerical study.

### Reproduce the AFT experiment

[tools/aft_compare.py](../tools/aft_compare.py) runs the Rust
[aft_bench example](../examples/aft_bench.rs) and writes summary JSON, PNG/SVG
plots, the GIF, and private complex spectral arrays. NumPy, Matplotlib, Pillow,
and FFmpeg are optional research tools, not Cargo dependencies. Install them in
a separate Python environment. The destination must be new and outside the repo.

```sh
cargo test --offline --test aft --test spectral
cargo build --offline --release --example aft_bench
python3 -m unittest discover -s tests -p test_aft_compare.py -v
python3 tools/aft_compare.py /path/to/music.mp3 /tmp/aft-trial \
  --starts 60 180 300 --seconds 2 --rate 44100 --frame 1024 --hop 512 \
  --sizes 256 512 1024 2048 4096 --ffmpeg /path/to/ffmpeg
```

`--binary` selects an existing release example; otherwise Cargo builds it and
its actual target directory is discovered. The original recording is read-only.
Only the summary and rendered figures are copied into docs; keep
`private-spectra.npz` outside version control because its coefficients can
reconstruct audio. The [complete AFT benchmark snapshot](experiments/aft-beethoven-20260930.json)
contains parameters, numerical results, model counts, and individual timing samples.

Validation passed the complete Cargo suite (22 tests including the doctest),
formatting, Clippy with warnings denied, rustdoc with warnings denied, release
examples, and all 13 optional Python tests. The existing resampler behavior is
unchanged. The older compression investigation follows for historical context.

## Fourier transforms from geometry

The new Rust [spectral module](../src/spectral.rs) has three operations:

- `angular_fourier`: analytic continuous transform of an individual equal-angle
  circle, split at the sample anchor because its two angular slopes may differ.
- `uniform_fourier`: numerical continuous transform of a uniform-x semicircle,
  integrated over its valid extent with an explicit quadrature panel count.
- `frame_dft_bin`: the discrete transform of integer predictions from retained
  frame descriptors, without allocating the decoded PCM or accessing the source.
  This takes O(N) per bin; computing every bin this way takes O(N²).

All use the negative-exponential convention. The continuous APIs integrate over
sample position, with angular frequency `omega = 2*pi*Hz/sample_rate`. Divide the
integral by sample rate to express time in seconds. They operate on individual
unblended functions: they do not include resampler windows or copied originals.

For one angular half, write `y(t) = cy + r*sin(theta + beta*(t-m))` on an interval
of length L centered at m. Define `sinc(x) = sin(x)/x`, with `sinc(0)=1`, and
`Sminus = sinc((beta-omega)*L/2)`, `Splus = sinc((beta+omega)*L/2)`. Its integral is:

```text
exp(-i*omega*m) * L * (
  cy*sinc(omega*L/2)
  + r/2*sin(theta)*(Sminus+Splus)
  - i*r/2*cos(theta)*(Sminus-Splus)
)
```

The implementation uses a small-argument sinc expansion at DC and resonance.
For uniform x, substitute `x_local = cx + r*sin(theta)`. Then
`y = cy ± r*cos(theta)` and `dt = r/aperture*cos(theta)*dtheta`. Midpoint quadrature
in theta avoids the infinite endpoint slope of the semicircle in x. More panels
are still necessary for oscillatory integrands; the API does not certify a
quadrature error bound. Very ill-conditioned large circles remain a limitation.

The continuous integral is not the DFT of the sampled signal. For example, the
uniform semicircle through `[0,1,0]` has area pi/2, the angular sine arc has area
4/pi, and the sampled sequence has DC sum 1. Their differences are intentional.
See Julius O. Smith's [continuous/discrete Fourier definitions](https://www.dsprelated.com/freebooks/sasp/Fourier_Transforms_Continuous_Discrete_Time_Frequency.html)
for the transform conventions. The formulas above are implemented and checked
against independently sampled quadrature, phase shifts, scaling, and known areas.

## What a predicted subcircle can tell us

Restricting a parent circle to smaller time intervals produces **subarcs of the
same circle**. It neither reduces its radius nor supplies new observations.
Predictions at those positions remain identical. Refitting three exact points
from that parent would recover the same circle in exact arithmetic; doing so
numerically is unnecessary and can add rounding error.

To improve prediction, a child must introduce source information. The experiment
bisects sample coverage and selects another already-fitted Rust circle or a line
for each child. It measures parent error and corrected-child error against every
source sample. A parent cannot discover its own error from generated subarcs
alone. A transmitted residual or independently measured child model is needed.

Every circle comes from `Analysis::new` with the default uniform preset on a
normalized PCM16 block. The Python lab reuses its anchor, aperture, center, radius,
and branch. It does not refit circles or modify `Analysis`. At each span it tests
circles anchored at its first interior sample, middle, and last interior sample,
alongside the endpoint line. These are local-triplet circles **extrapolated** over
a larger span, not circles fitted through widely spaced parent endpoints.
Invalid extent or excessive prediction is rejected. Branch mismatch is never
silently accepted as coverage. Child descriptors retain block-relative anchors.

The diagnostics compare parent restrictions with source-informed child choices
at four depths, using squared error. They are diagnostic trees, not necessarily
the exact trees selected for minimum encoded size. Child improvement is measured,
not assumed. The byte search evaluates model-only leaves, split children, raw
PCM, and (where enabled) residual leaves. It chooses the smallest independently
zlib-compressed candidate at each node. This is a local heuristic: different
child bytes can change compression at their parent, so adding candidates does
not guarantee a globally smaller file.

## Methods and error control

[compression_lab.py](../tools/compression_lab.py) compares:

| Method | Predictor/payload |
|---|---|
| `pcm` | PCM16, zlib per block |
| `delta` | First sample followed by first differences as int32, zlib |
| `line_tree` | Recursive lines or raw PCM |
| `circle_tree` | Recursive lines, reused circles, or raw PCM |
| `line_residual` | Line tree plus quantized residual leaves |
| `circle_residual` | Same search with reused circle candidates |
| `fourier_residual` | Largest K real-DFT bins plus residual, or raw PCM |
| `line_fourier_residual` | Endpoint line + K residual bins + remaining residual |
| `circle_fourier_residual` | Best of eligible line/circle bases + K residual bins |
| FLAC | FFmpeg lossless reference encode, independently decoded and checked |

Fourier trials use NumPy's FFT of integer predictor samples and of the source
residual; they do not substitute a continuous arc integral for a DFT. Linearity
lets the two spectra add, but geometry alone does not give the original spectrum
unless prediction error is zero. The K bins are ranked by conjugate-pair energy,
with proper DC/Nyquist handling; coefficients normalized by block length are
serialized as float32. There is no window or overlap-add. Block boundaries and
maximum boundary-jump error are explicitly reported.

For a requested integer maximum error E in PCM16 units, let `q=2*E+1`. Predictors
are decoded, rounded to nearest even, and clipped to PCM16 before residuals are
computed. Encode `round((source-prediction)/q)` as int32. Decode prediction plus
q times residual, then clip to PCM16. This bounds integer error by E, including
at E=0. Geometry is serialized as float64; Fourier coefficients are evaluated
**after** float32 serialization. The full final stream is decoded and compared
with the source. At E=0 the assertion is exact equality of decoded PCM16, not
losslessness relative to the MP3's earlier uncompressed recording.

Raw fallback limits payload growth, but headers and zlib overhead can still make
a stream larger than PCM. All reported sizes include the whole serialized file.
The FLAC reference uses its normal metadata and stereo coding; the research
methods encode channels independently. The PCM denominator excludes WAV headers.
These ratios are not comparisons against the source MP3's encoded size.

## Experimental SWR1 envelope

This tool-local format is not a stable crate API or a general-purpose codec.
There is no compatibility commitment, seeking, streaming, or cross-platform
rounding guarantee. The decoder accepts bounded local experiments; it has
malformed-input tests, but no fuzz/security validation.

All integers/floats use explicit little endian encodings, never native layouts.
The 16-byte file header is `4s magic=SWR1, u32 rate, u16 channels, u16 block,
u32 frames`. Limits are 384000 Hz, 1–2 channels, 2 million frames, and power-of-two
blocks of 4–4096 frames. Blocks appear in channel-major order. Each has a u32
compressed-byte length then one zlib member. Partial final blocks retain their
actual length. Decompression is bounded by `128*frames_in_block+1024` bytes;
truncation, trailing data, excessive sizes, invalid tags/values, and bad coverage
are rejected before acceptance.

Coverage and split positions are implicit in block length and midpoint bisection.
A node begins with one ASCII byte:

- `R`: covered PCM16 values.
- `D`: covered int32 deltas, first sample relative to zero.
- `L`: float64 value at span start and float64 slope, in PCM16 amplitude units.
- `C`: u16 block-relative anchor, four float64 values (aperture, cx, cy, radius),
  i8 branch (+1/-1). Circle amplitudes are normalized by 32768.
- `S`: left then right child. Splits below five samples are rejected.
- `Q`: positive odd u16 q, predictor, then one int32 residual per covered sample.

A Q predictor can be L/C, `F`, or `B`. F is u16 bin count followed by sorted,
unique `(u16 index, f32 real, f32 imaginary)` records; conjugates are implicit.
DC and even-length Nyquist bins must be real. B is L/C followed by F and means
add the Fourier correction to the already-rounded/clipped base, then round/clip
again. F/B are not recursive. Trees are bounded to depth 12. Node tags, parameters,
Fourier bins, residuals, zlib checksums, per-block lengths, and the file header
are all included in the measured size.

## Reproduce

Requires Rust, Python with NumPy, and FFmpeg. Neither NumPy nor FFmpeg is a Cargo
dependency. The script builds the `export_circles` example offline and queries
Cargo's target directory; `--exporter` can select an existing executable.

```sh
cargo test --offline --test spectral
cargo build --offline --release --example export_circles
python3 -m unittest discover -s tests -p test_compression_lab.py -v
python3 tools/compression_lab.py /path/to/music.mp3 /tmp/compression-trial \
  --starts 60 180 300 --seconds 1 --rate 44100 --channels 2 \
  --block 256 --keep 8 --tolerances 0 16 128 --ffmpeg /path/to/ffmpeg
```

The output directory must be new and outside the repository. It contains a
reference WAV/FLAC, each complete `.swr` stream, its decoded WAV, and a JSON report
with sample errors, boundary errors, SNR, node counts, sizes, and wall times.
Reports omit source paths. PCM16 uses no extra gain; no resampling is requested
when the source already matches the specified rate. The input recording is
read-only. Keep all recordings and generated audio outside version control.

## Measured results

Tested on 2026-09-30 using three one-second excerpts of the user-provided
Beethoven MP3, starting at 60, 180, and 300 seconds. Each was decoded to stereo
PCM16 at 44.1 kHz with no gain. Blocks contain 256 samples per channel; Fourier
trials retain eight conjugate-pair bins. Rust 1.98.1, NumPy 2.3.5, zlib 1.2.12,
and FFmpeg 7.1.1 ran on macOS arm64. No subjective listening result is claimed.
The [complete numerical report](experiments/beethoven-20260930.json) includes
per-excerpt times, errors, boundary jumps, and selected node counts.

Across all three excerpts, the reference contains **529,200 PCM bytes**. The
table sums three independent complete files for each setting:

| Method | E=0 bytes | E=16 bytes | E=128 bytes |
|---|---:|---:|---:|
| PCM + zlib | 502,613 | — | — |
| Delta + zlib | 463,310 | — | — |
| Line tree | 502,609 | 497,368 | 346,861 |
| Circle/line tree | 502,609 | 497,355 | 347,066 |
| Line + residual tree | 502,463 | 345,514 | **162,310** |
| Circle/line + residual tree | 502,463 | **345,513** | **162,310** |
| Fourier + residual | 502,613 | 346,690 | 185,486 |
| Line + Fourier + residual | 502,613 | 345,899 | 191,088 |
| Circle/line + Fourier + residual | 502,613 | 345,899 | 191,088 |
| FLAC reference, exact | **190,956** | — | — |

Every stream was independently decoded. All E=0 trials reproduced the decoded
PCM16 exactly; all lossy trials satisfied their maximum error bound. At E=16,
circle/line residual coding uses 65.29% of PCM, with aggregate RMS error 9.46
PCM16 units. It saves just **one byte** over line-only residual coding across
three files. At E=128 it uses 30.67%, ties line-only coding exactly, and has RMS
error 73.40 units. Its per-excerpt SNR ranges from 16.58 to 32.84 dB; maximum
sample error alone should not be read as transparency.

FLAC uses 36.08% of PCM with **zero** error, substantially better than these
research methods at E=0 or E=16. Its usual stereo prediction and larger blocks
are advantages that this lab intentionally does not reproduce. The E=128
size reduction below FLAC trades away fidelity. None of this establishes a
competitive circle audio codec.

The circle/line+Fourier methods selected **zero circle bases** in the saved files
at every tolerance. At E=16 the recursive circle residual files retained four
circle descriptors total; at E=128 they retained none. This is evidence against
paying for the current full-precision local-circle descriptors in these settings,
not evidence that every circle representation must fail. Model-only trees can
get slightly larger when circles are offered because node-local zlib costs do
not optimize the final parent's compressed size.

Source-informed subdivision does improve prediction in the diagnostic search:

| Excerpt offset | Whole-block predictor RMSE | After four subdivision levels |
|---|---:|---:|
| 60 s | 769.99 | 165.07 |
| 180 s | 514.35 | 37.98 |
| 300 s | 4,078.05 | 812.95 |

These errors are PCM16 units, before transmitting residuals. The table describes
line/circle candidate selection on smaller spans; a restricted parent alone has
unchanged predictions. The improvement requires child/source information, whose
parameter cost is then counted in the encoded trials.

The next useful experiments are wider-span parent fits as an explicitly separate
policy, quantized/delta-coded child geometry, and residual coding with a stronger
entropy model. Sweep block size and Fourier budget and compare matched measured
RMS/maximum error before interpreting small byte differences. Broader music,
speech, transients, silence, longer excerpts, other machines, and blind listening
remain untested. The synthetic suite covers ideal circles, tones, ramps, silence,
noise, short blocks, clipping endpoints, and malformed envelopes.

Validation recorded for the earlier compression experiment passed the complete
Cargo suite (including its spectral tests and doctest), formatting, Clippy with warnings denied, rustdoc,
release build, and the existing codec example. All 10 optional Python tests ran
with NumPy/FFmpeg available. Live validation passed 144 Rust/C/Python comparisons
without modifying frozen fixtures. All 69 saved SWR1 files were reopened and
their decoded lengths, node counts, and full file sizes checked against the report.
