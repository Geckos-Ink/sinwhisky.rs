# Design, contracts, and provenance

## Source selection

This Rust fork was derived from the `sinwhisky.py` checkout at revision
`77cf2e2` (2026-09-27 inspection). The original MIT notice is retained.
The relevant ancestor files are `original/whisky.c`, `original/flt.c`,
`sinwhisky_c/sinwhisky.c`, `sinwhisky_c/sinwhisky.h`, `v2/whisky.py`,
`tests/test_sinwhisky.py`, and `docs/original-vs-generated.md`.
These ancestor paths are provenance, not build-time dependencies.

The [whitepaper](<SinWhisky e FLT.pdf>) explains the 2014 concept. It is historical
algorithm evidence, not a complete specification for the newer uniform-x method.
The recovered source interpolates angles; the portable C/Python code interpolates
x on a selected semicircle. Both are retained as named modes instead of conflated.

The newer C code uses **double** geometry/accumulation and **float** input/output.
Rust uses f64 throughout, like the Python reference. Comparisons use `1e-9` absolute
and relative tolerance for Python, `2e-4` for the float-I/O C CLI. These comparisons
are numerical tests on the selected inputs, not proof of equality for every signal.

## Analysis and synthesis are separate

`Analysis::new` validates samples/configuration, owns a copy of the samples, and
fits each interior triplet once. Endpoints have no model. `models()[i-1]` is anchored
at source sample `i`. The aperture uses the maximum of all three pairwise y
separations, including the first-to-third difference. The uniform preset uses
`clamp(range*gain, min, max)`, defaults `1, 1, 12`, epsilon `1e-10`.

For each model fit the circle through `(-a,y[i-1]), (0,y[i]), (a,y[i+1])`.
The 2x2 determinant rejection is shared with the C/Python implementation. The
original preset uses `4e-6` because that determinant is four times the original
circumcircle determinant (`1e-6` tolerance). Unstable or degenerate fits retain
an explicit centered line model rather than an invalid circle with radius zero.

The descriptor stores two coordinate systems:

- `anchor`: original sample position; it is not the circle's geometric center.
- `circle.center`, radius, aperture: local scaled geometry, evaluated at
  `x_local=(global_position-anchor)*aperture`.

The selected y(x) branch contains the middle point. It need not contain both
outer points. Extent means `abs(x_local-cx) <= radius`, not `abs(x_local) <= radius`.
Angles are unwrapped in the direction passing through the middle point, repairing
wrap ambiguity in the recovered source. They support crossing the upper/lower
semicircle boundary without forcing a single y(x) branch.

`Analysis::resample` allocates an output after checked length arithmetic, copies
all source samples exactly, evaluates local models, and accumulates in ascending
anchor order. If no finite blend exists, it uses the source pair's linear
interpolation. No clipping or normalization occurs inside the library.
Empty/singleton signals are handled directly. Unlike the ancestor API, Rust
rejects non-finite samples and invalid parameters instead of clamping parameters
or propagating NaNs. Very large finite geometry can fail to fit and fall back.

## Original angular behavior

Use range aperture, equal angular increments separately over the left and right
halves of a triplet, and the finite-zoom window
`cos(pi*q*u/(2*(q+1)))`. Only the two adjacent triplets contribute at inserted
positions. The paper's `[3,7,9,4,6]`, zoom 3 example is a regression fixture.
`[0,1,0]` produces an angular sine arc, rather than a semicircle sampled at uniform x.

Intentional repairs: flat-triplet zero aperture, missing line intercept,
two-sample zero filling, invalid-radius use, and angular wrap ambiguity.
An original-mode line uses the middle value and the outer-point slope. These
repairs make this a concept-preserving mode, not binary reproduction of old bugs.
The angular mode has no defined generalized-neighbor extension; reject it rather
than silently assigning a novel meaning to the original contract.

## Reusing circles for a codec

`ResearchFrame::from_analysis` is a concrete baseline for investigating the user's
circle compression idea. It owns metadata and segment descriptors, never a borrow
of analysis/source PCM. Segment ranges partition the source exactly. For each
start position, test a line and the already computed circle anchored at start+1;
extend until the first sample exceeds the requested absolute error or leaves the
circle's extent. Prefer longer coverage and a line on ties. Fallback is literal.

Every candidate is checked at **all** covered sample positions, including the
triplet's endpoints. This prevents branch mismatch from falsely claiming an error
bound. Tolerance applies to sampled amplitudes in the input's units; it does not
bound intersample error or perception. Segments do not blend and may have slope
changes at boundaries. Metrics are computed from the actual decoded frame.

The analysis descriptors are not a wire schema. They include redundant research
information (angles, branch, aperture, anchors) and may be bigger than PCM. A future
format must explicitly encode version, mode, sample rate, channel layout, sample
count, coverage, parameter precision, residual strategy, and resource limits.
Do not serialize Rust memory layouts or adopt the Python prototype format without
a specification and malformed-input tests.

## Spectral and hierarchical research

The separate [compression experiment](COMPRESSION.md) adds finite-interval
circle transforms and a tool-local, versioned PCM16 test envelope. `spectral`
uses the retained geometry without altering fitting, blending, or source-anchor
copying. Equal-angle integrals are analytic; uniform-x integrals use bounded-work
quadrature. `frame_dft_bin` sums discrete segment predictions and is deliberately
not labeled an FFT or the original source's spectrum.

The NumPy lab imports the default Rust analysis descriptors through
`examples/export_circles.rs`. Parent spans extrapolate existing local circles;
children restrict the parent's function for diagnostics, then may select another
source-informed circle or line. Predicted subarcs alone cannot reveal source
error. All integer positions are checked after serialization and reconstruction.
The optional SWR1 envelope has explicit endian fields, bounded decompression,
coverage, raw fallback, quantized residuals, and malformed-input tests. It is not
a serialization of `ResearchFrame` and carries no stable compatibility promise.
Full framing, models, and residuals count toward byte measurements. The Rust core
remains dependency-free; no audio data or Python packages enter Cargo.

## Periodic angular Fourier model

`aft::AngularFrame` defines a separate continuous interpolation policy for AFT.
It fits `Analysis` once with periodic source halos and the caller's explicit
parameters. On each unit sample interval it averages the right angular half of
the left circle and the left angular half of the right circle. Missing or
numerically inconsistent circle halves become the source-pair line. The closing
interval from last sample to first is included, so one frame spans `[0,N]`.
This uses neither the original resampler's finite-zoom window nor its centered
line fallback. It does not change either established resampling mode.

Each prepared half has a closed sine integral, with sinc limits at resonance;
line halves use the derivative of sinc. Global phases are applied at interval
midpoints and accumulated with compensation. Geometry is independent of requested
frequencies. `fourier` accepts radians per sample; `spectrum` evaluates `N/2+1`
bins at `2*pi*k/N`, with no FFT approximation. The same frame can be evaluated at
fractional times for independent numerical integration. Direct cost is O(N) per
frequency and O(N²) for a full spectrum.

The integral is linear in an already-fixed waveform. The process of fitting that
waveform from samples is generally nonlinear, so AFT is not an alternative DFT
algorithm or an established inverse transform. The continuous coefficient at
Nyquist need not be real. Do not feed these coefficients to a discrete inverse
FFT and assume exact reconstruction. The [current comparison](COMPRESSION.md#angular-fourier-transform-aft-current-benchmark)
uses identical periodic-Hann-windowed samples, a linear-interpolation control,
and a dense midpoint-FFT check with the required integration weight/phase.

## Scope boundaries

Fast Linear Transform (FLT) is a separate calibration/equalization idea.
`original/flt.c` in the ancestor contains a recovered equation solver, but no full
FFT calibration/audio equalizer pipeline. This Rust crate does not port that
unverified solver. A future robust linear-system implementation belongs in a
separate module, not inside the resampler or codec geometry.

No network, external service, persistence, or realtime thread exists in the Rust
core. Heap allocation and full-frame buffering make it unsuitable for claiming
allocation-free audio callback operation. There is no stable public API or
bitstream compatibility guarantee in this initial `0.1.0`, unpublished crate.
