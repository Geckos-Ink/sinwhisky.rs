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

## Periodic angular Fourier model (reference preset)

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
FFT and assume exact reconstruction. The [current comparison](COMPRESSION.md#angular-fourier-transform-aft-reference-preset-benchmark)
uses identical periodic-Hann-windowed samples, a linear-interpolation control,
and a dense midpoint-FFT check with the required integration weight/phase.
`AngularFrame::halves` exposes the prepared halves read-only. FAFT reuses them
to compute the same integral without integration; the values agree to
about 1e-13.

## FAFT: the transform of determined circles

`faft` keeps three stages apart, which earlier experiments had mixed. Circle
determination (`GeometryFrame`, triplet `Analysis`/`AngularFrame`) fits geometry
from source points. Zoom (SinWhisky synthesis) samples the resulting curve.
FAFT integrates that curve against `exp(-i*omega*t)`. Each equal-angle half is
represented as a `Rotation`: `center + radius*sin(phase + rate*u) + slope*u` on
an integer support. `CircleCurve` validates contiguous coverage; stacked
rotations on one support add, which represents the AFT's averaged halves.
Conversions from `GeometryFrame` and `AngularFrame` copy geometry without
refitting and reproduce each model's own evaluation (tested pointwise).

`CircleCurve::transform` crops rotations to an integer window once, converts
them to an anchored form (start value plus cosine/sine deviations) to avoid
center/radius cancellation, and prepares each `exp(i*rate*length)`. A query then
chains support phases by multiplication. It uses one sin/cos pair per frequency,
periodic re-anchoring and no per-rotation transcendental calls. Near DC or
resonance, stated thresholds switch to Horner series. The continuous Hann is
exact as three shifted rectangular transforms. Values are normalized by the
window weight sum with phase origin at the window start. Derivative queries use
the window centre. Frequencies are arbitrary finite reals, including above pi.
The result is the exact transform of the curve, not of the source samples, and
the source error bound does not extend to intersample values.

Zoom is not a transform parameter. A zoom-z sampled transform equals FAFT plus
its spectral images and half-jump terms at tolerance-bounded joins (Poisson
summation, tested). `peaks` scans at a spacing set by the window duration, not
by a sample count. It starts Newton from parabolic estimates, refines with exact
analytic derivatives inside a derivative-sign bracket, and drops flanks and
unconverged seeds, counting them. It merges peaks within a stated width and
applies relative/absolute floors. The scan is a heuristic seed source with a
measured miss rate, not a completeness bound. `references` groups rotation rates
in fixed-width groups (no transitive chains). Rates are geometric and were
measured to be poor spectral seeds, so circle seeding is an option, not the
default. Costs are O(rotations) per frequency and O(rotations × scan points) per
peak list. No FFT speed claim is made. [FAFT.md](FAFT.md) holds the definition,
measurements and limits.

A list of maxima is not a list of components: the window kernel `K` (the exact
transform of the weight) shares every peak's height with its neighbours. The
`faft::pursuit` submodule therefore treats relevant frequencies as a jointly
fitted model of real tones around the window centre. All Gram entries come from
`K`, `K'` and `K''`. The data side is FAFT values and first derivatives at the
selected frequencies. The residual energy needs one scalar, the weighted curve
energy, which is the only quantity computed by numerical integration
(Gauss-Legendre panels with bounded phase). Cauchy-Schwarz then bounds the
residual spectrum at every real frequency. Components are pruned by
significance (unique energy), not by a distance. Frequencies are polished by
damped Gauss-Newton per coupled cluster, with a stated step tolerance. Seeds
come from the residual on the single scan grid. `PeakTracker` replaces the scan
by a polish from predicted frequencies and rescans on an interval or when the
residual energy grows (a heuristic novelty test). Separation inside one main
lobe comes from the tone model, not from added transform resolution. `peaks`
is unchanged and remains the maxima of the transform.

## FFT phasor control (formerly `faft`)

`phasor_fft::transform` computes the sampled DFT through radix-2 rotations. This is a
conventional FFT kernel. `PhasorFrame` (formerly `faft::FaftFrame`) adds a separate dyadic model search: local
constant or endpoint-line baseline, energy-ranked Fourier circles, and dynamic
Cartesian precision per circle. Parent and child supports compete; chosen leaves
partition the input. Child circles are source-informed replacements, not residual
predictions obtained from the parent. Fitted triplet `Analysis` geometry is neither
mutated nor used as a surrogate for phasor coefficients.

`LineLimit` stores affine endpoints as the stable zero-curvature limit of a
geometric circle. It does not encode an infinite radius. A zero-frequency phasor
is a constant and cannot replace a nonconstant line. Tests check collinearity,
near-line acceptance, and convergence of the rationalized large-radius sagitta.

Frames require a power-of-two length ≥2 and positive finite RMS tolerance.
Internal normalization keeps coefficient arithmetic away from amplitude overflow.
The optimizer counts each baseline plus each rotating circle, breaking ties using
an explicit hypothetical model-bit cost. It minimizes that objective within the
generated candidates and equal per-leaf budgets, not across arbitrary frequencies
or all quantizations. Quantization uses per-circle binary exponents and signed
2–30-bit components, with an f64 escape. Every accepted leaf and the final rescaled
decode are checked; impossible precision budgets return errors. No source PCM is
retained. Fast decode uses inverse rotations; fractional `evaluate` sums projections
on the selected support and may be discontinuous at child joins. The sampled RMS
bound does not imply a maximum or intersample error bound.

The core transform is O(N log N); the full dyadic search is O(N log² N).
`estimated_model_bits` is a selection metric, not measured storage. There is no
phasor serializer or new resampling mode. The integral `AngularFrame` remains
unchanged as a reference preset with a different continuous transform target.
The [phasor experiment](COMPRESSION.md#fft-phasor-control-earlier-faft-prototype)
records definitions, error accounting, runtime boundaries, and actual comparisons.

## Geometry-first sampled spectrum (control)

This was an earlier FAFT attempt. Its spectra are sampled (zoomed) DTFTs of the
fitted curve; `GeometryFrame` itself is now the circle-determination stage for
FAFT. `geometric_faft::GeometryFrame` fits independent time/amplitude circumcircles
through selected waveform points, then retains centers, radii, local x scale,
three unwrapped angles, and coverage. Its predictions advance equal angles on
each side of the fitting middle point. Fitting searches dyadic spans, midpoint
fractions, aperture gains, and significand widths. It does not obtain models from
FFT coefficients or mutate the resampler's Analysis. The terminal source anchor
is sample 0, specifying periodic coverage on `[0,N)`.

Every accepted descriptor checks all source samples plus its terminal anchor;
rounded circle fields also check the three horizontal anchors in scaled
coordinates. The final decoded maximum error is measured in original amplitude
units, so normalization cannot hide unrepresentable small components. RMS uses
scaled hypot accumulation of absolute errors. Extremely flat/ill-conditioned
circles remain finite endpoint lines or shorter candidates. There is no guaranteed
fractional-position or perceptual error bound. Model data are private and exposed
through immutable descriptor views, with no source PCM retained.

Dynamic programming minimizes descriptor count, then hypothetical model bits,
within the actual candidate bank. Default prevalidation rejects nominal geometry
that fails before quantization; the exhaustive control admits quantized variants
of those failures. Neither policy finds the global fewest arbitrary circles.
Circles have independent significand widths and per-field floating exponents;
there is no wire format or measured circle bitrate.

Prepared angular halves use finite exponential sums and line first moments to
compute the sampled predictor's spectrum. `dtft` can crop any retained support;
`hann_dtft` combines three shifted queries to weight that same predictor without
refitting. Batch rotations reuse frequency-grid roots, with stable near-resonance
limits and periodic reanchoring. No FFT or integration occurs in this module.
Whole-spectrum cost is O(PK), and can be quadratic when many pieces are required.
Zoom changes the density at which the SAME geometric function is sampled and
restores scale with a `1/zoom` normalization; it cannot add observations.

The [direct geometry experiments](COMPRESSION.md#geometry-first-sampled-spectrum-earlier-faft-attempt)
measure error against the source separately from numerical retrieval error,
compare different point/phase/precision banks and two implementation attempts,
and test omitted points, zoom, and encoded MP3 bitrate controls. Spectra at a
certain source time are finite-window queries, not an intrinsic single-sample
frequency decomposition. The FFT-based control and integral AFT reference remain
available with their earlier public contracts.

## Hierarchical geometry and adaptive frequencies

The hierarchy is the circle determination used by FAFT. Its `GeometricSpectrum`
and `AdaptiveSpectrum` compute the sampled (zoom) transform and remain controls.

`GeometryFrame::hierarchical` is a separate greedy preset. Parent supports can
have any integer length up to the configured cap (2..256). A failed fit splits
at a large source deviation from the endpoint line, clipped to the central half
of the support. Children fit original source points, never the parent's predicted
samples. At the hierarchy depth or minimum-span limit, checked one-interval lines
preserve coverage and the source error contract. Adjacent children can be refitted
into one model over their combined support; every source sample and the terminal
anchor are rechecked. Up to the requested number of merge passes are attempted.
Merging can cross earlier split/root boundaries. Statistics distinguish splitting,
fallback, attempted merges, accepted merges, and the initial descriptor count.
This search has no global minimal-circle claim and retains no source samples.

`GeometricSpectrum` crops the immutable predictor once, compiles rotations, and
computes amplitude/centered derivative bounds on the requested sampling grid.
Preparation is O(window length * zoom); queries use finite geometric series and
line first moments. `value` retains the closed-sum reference, while
`value_rotations` uses prepared angular roots, integer powers, complex products,
and ratios. It requires one new sin/cos pair per rectangular query, three for
Hann, plus stable near-resonance fallbacks. It is not a purely addition/multiply
algorithm. Values use radians per original source sample and normalization by
grid length (half the grid length for periodic Hann); a unit real sinusoid has
approximately 0.5 coefficient magnitude. Prepared `rotations()` exposes finite
supports, local centers, sine/cosine projections, rates and line slopes.

`AdaptiveSpectrum::search` uses an initial cover, strongest-first midpoint
refinement, then bounded ternary refinement of observed peak brackets. Output
frequencies are arbitrary floating values, not a fixed FFT-bin count. Interval
depth and peak refinement iterations each use `max_depth`; their combined work
also obeys a hard oracle-call budget. Width and peak-merging limits are separate.
Strongest-first merging suppresses nearby candidates without transitive merging
across an indefinitely long chain. The relative relevance floor uses an observed
peak lower bound, with an optional absolute floor. It is not a psychoacoustic
relevance metric. DC and upper-band edge candidates are included.

An optional coarse predictor is prepared on the SAME time range, zoom grid and
weights as the target. A streaming weighted L1 difference bounds their complex
spectral discrepancy at every frequency. A below-floor coarse probe can skip its
fine query; retained probes include that uncertainty. Different model tolerances
and precisions are allowed; different query grids/weights are rejected. Higher
zoom compares more points of the retained function, with no source intersample
error guarantee. Small rounding cushions are used for bounds, not an interval
arithmetic proof on every platform. The library rejects nonfinite bounds.

Bands expose their stop reason and a predictor-magnitude upper bound. `BelowFloor`
uses the centered derivative bound; `LowGain` uses sampled-quiet or flat behavior
as a deliberate heuristic and may miss a narrow peak between probes. `Depth`, `Width`, and `Budget` expose remaining
resolution/work limits. Disabling flatness pruning does not imply completion if
the budget or depth is exhausted. Peak brackets are neighboring sampled
frequencies, not confidence intervals. The finite-window spectrum is continuous
in frequency, but fine query spacing is distinct from resolving multiple source
oscillators or identifying a unique inverse rotating chain. The
[measured experiment](COMPRESSION.md#hierarchical-circles-and-adaptive-relevant-frequencies)
compares matched padded FFTs, repeated attempts, precision/depth/zoom controls,
and unresolved close tones without claiming an FFT speed advantage.

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
