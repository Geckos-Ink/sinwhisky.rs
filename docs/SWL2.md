# SWL2 stream format (experimental)

SWL2 is the byte stream of the layered coder in [src/layered/](../src/layered/mod.rs),
derived in [PAPER_EQ_COMPRESSION.md](PAPER_EQ_COMPRESSION.md), Section 4. It is a
research format: not stable, not fuzz-validated, and not safe as an untrusted
service input. This page specifies every byte; the code is the reference.

Conventions: integers are little-endian; `f32` is IEEE 754 binary32, little-endian.
Time is in source samples; frequencies in radians per sample. `bitlen(v)` is the
number of bits of `v` (0 for 0). `zigzag(v)` maps 0, -1, 1, -2, ... to 0, 1, 2,
3, ...; a "signed gamma" is the adaptive gamma code (Section 4) of `zigzag(v)`.
Every chunk body after its byte header is range coded (Section 4); each chunk
starts its own range coder.

## 1. Stream

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | Magic `SWL2` |
| 4 | 1 | Version, 4 (earlier versions are not decoded) |
| 5 | 1 | Channels C, 1..=8 |
| 6 | 1 | Segment exponent S: 0 (one segment) or 12..=24 (segments of 2^S frames, the last shorter) |
| 7 | 1 | Reserved, written 0, ignored |
| 8 | 4 | Frames N per channel, 1..=2^28 |
| 12 | 4 | Tolerance τ: 0..=2^20, or `0xFFFFFFFF` for a parametric stream |

Then one record per segment, in order:

| Size | Field |
|---:|---|
| 4 | Body length B |
| 1 | Flags: bit 0 key; bits 1-2 stereo, 0 independent, 1 left/side, 2 side/right, 3 mid/side; bits 3-7 zero |
| B - 1 | C coded channels, each a chunk count K and K chunks (1 byte type, 4 bytes payload length L, L bytes payload) |

Chunk types: 1 gradients, 2 anchors, 3 correction. Each type appears at most once
per channel and segment. The correction chunk is required and last, except in a
parametric stream, which has none. The first segment must be a key. Bytes after
the last segment are an error. A stereo code other than 0 requires C = 2 and τ = 0.

**Key and continuation segments.** At a key segment every coded channel's
correction state starts afresh (Section 4); a continuation segment inherits it
from the same coded channel of the previous segment. Nothing else crosses a
segment boundary: the gradient and anchor chunks describe their own segment only
(tracks are cut at boundaries), and the stereo mode may change from one segment
to the next. A decoder can therefore start at any key segment, skipping the
earlier ones by their body lengths, and a stream can be produced and consumed
segment by segment.

**Decoding a channel of a segment.** The prediction is `P[n] = round(G[n]) + round(A[n])`,
the rounded renders of the gradient and anchor chunks (0 where absent), with
`round` to nearest, ties away from zero. The correction chunk reconstructs `x̂`
from `P` (Section 4). A parametric channel is `x̂ = P`.

**Stereo.** Coded channels `(a, b)` give `(left, right)`: left/side `(a, a - b)`;
side/right `(a + b, b)`; mid/side `m = 2a | (b & 1)`, left `(m + b) >> 1`, right
`(m - b) >> 1` (arithmetic shifts). Output samples are clamped to ±2^23.

Every decoded sample of a stream with τ ≤ 2^20 is within τ of the source. The
gradient and anchor renders use `sin`/`cos` in binary64, so the decoder reproduces
the encoder exactly on the platform that encoded; the correction chunk alone is
integer arithmetic, and the description chunks are parsed with additions,
multiplications, divisions and square roots only.

## 2. Gradients chunk (type 1): frequency-gradient groups

| Size | Field |
|---:|---|
| 2 | Hop H, 8..=32768 (frame k is centred at sample k·H) |
| 4 | Precision p (`f32`), 1e-3..=1e6: amplitude quantum |
| 4 | Group count G, ≤ N |

Then the range-coded groups, in increasing start. Every field below has its own
adaptive model (one gamma code or one adaptive bit per field; the gradient and
fade-out fields have one per value of "continued").

Per group:

1. Start delta (gamma): start = previous start + delta (first previous = 0), in frames.
2. Duration D in hops (gamma), ≤ 4096.
3. If D > 0: a bit "not constant", and if set a bit "exponential" (else linear);
   if exponential, the ratio ρ (signed gamma), shared by all members.
   D = 0 is constant.
4. Members minus one (gamma), ≤ 4095.

Per member:

1. A bit "continued": the member continues a track that ended at this group's
   start (its fade-in flag is the negation). If set, the link step (signed
   gamma) names its predecessor (below).
2. Fade out (bit): the track ends here.
3. A continued member: class change (signed gamma) from the predecessor's class
   c, then the residuals (signed gamma) of the frequency index, of the start
   amplitude index and of the phase index from their predictions (below); the
   phase residual is the difference taken modulo 2^(c+3) into
   `[-2^(c+2), 2^(c+2))`. A new member: class c (gamma), ≤ 24; the frequency
   index (gamma), or, if an earlier new member of this group exists, its
   difference (signed gamma) from that member's frequency on this member's grid,
   `round(ω_prev·2^e_ω)`; the start amplitude index (gamma); the phase index,
   c + 3 direct bits.
4. If D > 0: the gradient index (signed gamma).
5. If linear: the end amplitude index minus the start one (signed gamma).

with `s = bitlen((max(D,1) + 1)·H)`, `e_ω = c + s + 1` and `e_γ = c + 2s`.
Parameters: `ω = i_ω / 2^e_ω` (i_ω < 2^46), `γ = i_γ / 2^e_γ` (|i_γ| < 2^46),
`φ = 2π·i_φ / 2^(c+3)`, `A_0 = p·i_0`, `A_1 = p·i_1` (linear) or `A_0`. The class
sets the quanta so each parameter costs about p of waveform error for a member of
amplitude up to p·2^c.

**Open tracks.** After a group is read, each of its members without fade-out is
added to the list of tracks open at frame start + D, with its end frequency
`ω_e = ω + γ·D·H`, end phase `θ_e = φ + ω·D·H + γ·(D·H)²/2` (binary64, in this
order of operations), end amplitude index (`i_1` if linear, else `i_0`) and
class; each frame's list is ordered by `ω_e`, then by (group, member) order. When
a group's start differs from the previous group's (and at the first group),
lists of earlier frames are discarded and the last rank is reset to 0. A
continued member's predecessor is the entry at rank `r = last rank + step` of
the list at its group's start; that entry is removed and the last rank becomes r.
Its predictions are `round(max(ω_e, 0)·2^e_ω)` (at most 2^46), the end amplitude
index, and `round(rem(θ_e, 2π)/2π·2^(c+3)) mod 2^(c+3)`, with the member's own c,
D and H.

**Render.** With `t` the time from `start·H`, `S = D·H` and `h = H/2`, each member
adds, for every sample `n = start·H + t` of the segment:

- main, `0 ≤ t < S`: `A(t)·cos(φ + ωt + γt²/2)`, where `A(t)` is `A_0`
  (constant), `A_0 + (A_1 - A_0)·t/S` (linear) or `A_0·2^(ρ·t/(8S))` (exponential),
  clamped at 0;
- start: with fade in, `-H ≤ t < 0`: `A_0·(t + H)/H·cos(φ + ωt)`; without, the main
  formula over `-h ≤ t < h` weighted by `(t + h)/H`;
- end: with fade out, `S ≤ t < S + H`: `A_e·(1 - (t - S)/H)·cos(θ_e + ω_e(t - S))`
  with `A_e = A(S)` (`A_0` if D = 0), `ω_e = ω + γS`, `θ_e = φ + ωS + γS²/2`;
  without, the main formula over `S - h ≤ t < S + h` weighted by `(S + h - t)/H`
  (the smaller weight where both apply).

Consecutive segments of a track meet at a centre and cross-fade over one hop.

## 3. Anchors chunk (type 2): SinWhisky anchors

| Size | Field |
|---:|---|
| 4 | Anchor count M, 1..=N |
| 4 | Value quantum q (`f32`), 1e-3..=1e6 |
| 1 | Guess g: 0 zero, 1 previous, 2 line, 3 circle |

Then, range coded, per anchor j: if j > 0, its gap minus one (gamma; context
`min(bitlen(previous gap minus one), 19)`, the first previous being 0); then the
residual `r = v_j - guess_j` of its value index: `|r|` (gamma; context
`min(bitlen(previous |r|), 19)`) and, if nonzero, a sign bit (1 negative; context
the signs of the previous two residuals, 3 states each, initially zero).
Positions start at 0 and add each gap plus one; the last must be N - 1. Values
are `q·index`, |index| < 2^50.

**Guesses** from the anchors before j (all of them 0 for j = 0; "previous" for
j = 1 unless zero; "line" for j = 2 when the guess is the circle), rounded to the
nearest integer (ties away from zero): zero; the previous index; the line through
the last two anchors at `t_j`; the circle through the last three, in the plane
`((t - t_{j-2})·α, v)` with `α = 2·range/(t_{j-1} - t_{j-3})` (range: the largest
difference of the three indices), centre from the same 2×2 system as below,
`r² = (x_{j-1} - c_x)² + (v_{j-1} - c_y)²`, `d = (t_j - t_{j-2})·α - c_x`, and
the value `c_y ± sqrt(r² - d²)` on the side of the last anchor (`+` if
`v_{j-1} ≥ c_y`); the line where the range is 0, the determinant is at most
`1e-9·range²` or `r² < d²`. The encoder writes the guess that gives the fewest bytes.

**Render.** For anchors `(t_j, v_j)`, every inner anchor j has the circle through
anchors j-1, j, j+1 in the plane `((t - t_j)·α, v)` with
`α = 2·max(|v_{j-1}-v_j|, |v_j-v_{j+1}|, |v_{j-1}-v_{j+1}|) / (t_{j+1} - t_{j-1})`.
The circle is a line when α = 0, the determinant of its centre equations is at
most `1e-9·range²`, or its radius exceeds `1e6·range`. Angles of the three
anchors are unwrapped in the direction that visits the middle one; the angle over
time is the quadratic through the three `(t, θ)` pairs, or, if that quadratic
reverses direction within the span, linear on each side. Its value is
`c_y + r·sin θ(t)`. On `[t_j, t_{j+1})`, with `s = (t - t_j)/(t_{j+1} - t_j)`, the
render is `w·L(t) + (1 - w)·R(t)`, `w = (1 + cos πs)/2`, where L is the circle of
anchor j and R that of anchor j+1; only one of them where the other does not
exist or is a line, and the straight line between the two anchors where neither
does. The last sample is `v_{M-1}`.

## 4. Correction chunk (type 3)

| Size | Field |
|---:|---|
| 1 | Bits 0-3: block exponent b, 8..=15 (blocks of 2^b samples); bit 4: cascade flag (LMS stages follow the LPC); bit 7: layer flag (the channel has a prediction P); bits 5-6: 0 |
| rest | Range-coded blocks |

**Range coder** (LZMA style): 32-bit range starting at `0xFFFFFFFF`; the code
value is the first 5 bytes, of which the first is 0; renormalize by one byte while
the range is below 2^24. An adaptive bit holds two estimates of P(0) in units of
2^-12, `f` and `s`, both starting at 2048; its probability is
`p = clamp((f + s) >> 1, 31, 4065)` and splits the range at
`bound = (range >> 12)·p`. A 0 bit keeps the low part and sets
`f += (4096 - f) >> 4`, `s += (4096 - s) >> 7`; a 1 bit the high part and sets
`f -= f >> 4`, `s -= s >> 7`. A direct bit halves the range without a probability.
Multi-bit direct fields are most significant bit first.

**Gamma code** of `v ≥ 0` (all chunks): with `m = v + 1` and `l = bitlen(m)` ≤ 46,
`l - 1` one-bits and a zero-bit (adaptive, one probability per position); then the
first `min(2, l - 1)` bits below the leading one, adaptive, with a binary tree of
probabilities per `l` (node 1, then `2·node + bit`); then the remaining bits
direct. Each use of the code (a field, a context) has its own probabilities.

**Per block** (the last block is shorter):

1. If the layer flag is set: gain g, 3 direct bits, 0..=4 (gain g/4).
2. The predictor (below).
3. If the cascade flag is set: stages used u, 2 direct bits, 0..=3.
4. One index q per sample (below).

**Predictor.** An adaptive bit "same as the previous block's predictor" (order 0
before the first block of a key segment). Otherwise: order o, 6 direct bits,
≤ 32; if o > 0: shift k, 5 direct bits; then, only if k equals the previous
predictor's shift, an adaptive bit "delta". With delta, every coefficient is
`c_i = c'_i + d_i` with `d_i` a signed gamma (one shared code) and `c'_i` the
previous coefficient (0 beyond its order); |c_i| < 2^15. Without (a key
predictor): width w, 5 direct bits, 1..=15, then o coefficients `c_i`, w direct
bits each, signed. The encoder uses deltas when their estimated size
(`Σ 2·bitlen(zigzag(d_i) + 1) - 1` bits) is below `5 + o·w`.

**Index coding.** A running magnitude `m` starts at 16; the context is
`min(23, bitlen(m >> 3))` and after each index `m = m - (m >> 4) + min(|q|, 2^24)`.
Per context there are adaptive bits for "nonzero" and "negative", and a gamma
code for `|q| - 1`.

**Reconstruction.** With `g' = g` (4 when there is no layer flag),
`scale(v) = floor((v·g' + 2) / 4)` and `h(i) = x̂[i] - scale(P[i])`:
`pred = ((Σ_{i < min(o, n, 32)} c_i·h(n-1-i) + 2^(k-1)) >> k) + scale(P[n])`
(no rounding term when k = 0), in saturating 64-bit integers, clamped to ±2^40;
`x̂[n] = pred + q·(2τ + 1)`, clamped to ±2^40. Since the encoder chooses
`q = floor((x[n] - pred + τ) / (2τ + 1))`, every `|x̂[n] - x[n]| ≤ τ`. In a
continuation segment n counts from the inherited history (below), so the first
samples use up to 32 samples of the previous segment, with their own P.

**LMS cascade** (cascade flag set). Call the prediction above `base`. Three
sign-sign LMS stages follow, as in Monkey's Audio, with (taps o, shift k) =
(1024, 15), (256, 13), (16, 11). They start from zero at a key segment, or when
the previous segment of the channel had no cascade, and persist across blocks
and continuation segments. Stage i keeps weights `W_j` and, for its past inputs
`v`, steps `d` (j = 1..o is the lag; inputs and steps before its start are 0) and
a running mean `a`, all integers. Its prediction is
`p_i = (Σ_j W_j·v(n-j) + 2^(k-1)) >> k` (64-bit). Then
`pred = base + p_1 + … + p_u` (saturating, clamped to ±2^40) and `x̂[n]` is
reconstructed as above from this `pred`.

After `x̂[n]`, every stage adapts, whatever u is: `r_0 = x̂[n] - base`; stage i
takes input `r_{i-1}` and output `r_i = r_{i-1} - p_i` (saturating), and:

1. if `r_i > 0` each `W_j += s_j`, if `r_i < 0` each `W_j -= s_j`, clamped to
   ±2^20, where `s_j = d(n-j) >> h(j)` (arithmetic shift) with h(1) = 0,
   h(2) = 1, h(3..=8) = 2 and h(j ≥ 9) = 3;
2. the new input is `v(n) = clamp(r_{i-1}, ±(2^24 - 1))`; with `m = |v(n)|`, its
   step is `d(n) = c·sign(v(n))`, where c is 32 if `m > 3a`, else 16 if
   `m > floor(4a/3)`, else 8 if `m > 0`, else 0;
3. `a += (m - a) / 16`, the quotient truncated toward zero (a starts at 0).

**State across segments.** A continuation segment's correction chunk starts with
the previous segment's (same coded channel) adaptive probabilities and running
magnitude (indices and predictor coding), its last predictor, its cascade (if
both segments have the cascade flag) and its last 32 reconstructed samples with
their predictions P (0 where it had no layer). A key segment starts with fresh
probabilities, `m = 16`, an order-0 previous predictor, no history and zero
stages. If a channel's segment is parametric or the stream is, nothing is kept.

The cascade is integer arithmetic too, so this layer decodes identically on every
platform.

## 5. What the encoder chooses

`encode_auto` measures complete coded sizes and keeps the smallest, segment by
segment from the state the previous segment left (greedy): the stereo code
(lossless only; the four signals coded plainly, each from its coded channel's
state), and per channel whether a gradients chunk pays (band, p, segment
tolerance), whether an anchors chunk pays (low band or all, tolerance), and the
block exponent (starting at 13 with the cascade, 12 without, one step either
way, continuing while it pays). Every `key_interval`-th segment is a key (0: only
the first). Inside the correction chunk every block chooses g and the LPC predictor
by exact size among the orders `{0,1,2,4,8,12,16,20,24,32}` (all of them without
a layer, the two best by an estimate per gain with one), each at 15-, 12- and
9-bit coefficient precision, and the previous block's predictor; then, with the
cascade, u for the two smallest LPC candidates and the smallest of order ≤ 2 and
of order 3..=8 (a short LPC that the cascade completes is often cheapest, which
coding without the stages cannot see). The anchor guess is the one with the
fewest bytes. None of these choices needs to be signalled beyond the fields above.
