# SWL2 stream format (experimental)

SWL2 is the byte stream of the layered coder in [src/layered/](../src/layered/mod.rs),
derived in [PAPER_EQ_COMPRESSION.md](PAPER_EQ_COMPRESSION.md), Section 4. It is a
research format: not stable, not fuzz-validated, and not safe as an untrusted
service input. This page specifies every byte; the code is the reference.

Conventions: integers are little-endian; `f32` is IEEE 754 binary32, little-endian;
`i8` is two's complement. Bit-packed areas are written most significant bit first,
start at a byte boundary, and are padded with zero bits to the end of the chunk.
A field of width 0 is absent and its value is 0. Signed packed fields are two's
complement in their width. `bitlen(v)` is the number of bits of `v` (0 for 0).
Time is in source samples; frequencies in radians per sample.

## 1. Stream

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | Magic `SWL2` |
| 4 | 1 | Version, 3 (version 2 had no LMS cascade; it is not decoded) |
| 5 | 1 | Channels C, 1..=8 |
| 6 | 1 | Stereo: 0 independent, 1 left/side, 2 side/right, 3 mid/side |
| 7 | 1 | Reserved, written 0, ignored |
| 8 | 4 | Frames N per channel, 1..=2^28 |
| 12 | 4 | Tolerance τ: 0..=2^20, or `0xFFFFFFFF` for a parametric stream |

Then C coded channels, each:

| Size | Field |
|---:|---|
| 1 | Chunk count K |
| K × (5 + L) | Chunks: 1 byte type, 4 bytes payload length L, L bytes payload |

Chunk types: 1 gradients, 2 anchors, 3 correction. Each type appears at most once
per channel. The correction chunk is required and last, except in a parametric
stream, which has none. Bytes after the last channel are an error. A stereo code
other than 0 requires C = 2 and τ = 0.

**Decoding a channel.** The prediction is `P[n] = round(G[n]) + round(A[n])`, the
rounded renders of the gradient and anchor chunks (0 where absent), with `round`
to nearest, ties away from zero. The correction chunk reconstructs `x̂` from `P`
(Section 4). A parametric channel is `x̂ = P`.

**Stereo.** Coded channels `(a, b)` give `(left, right)`: left/side `(a, a - b)`;
side/right `(a + b, b)`; mid/side `m = 2a | (b & 1)`, left `(m + b) >> 1`, right
`(m - b) >> 1` (arithmetic shifts). Output samples are clamped to ±2^23.

Every decoded sample of a stream with τ ≤ 2^20 is within τ of the source. The
gradient and anchor renders use `sin`/`cos` in binary64, so the decoder reproduces
the encoder exactly on the platform that encoded; the correction chunk alone is
integer arithmetic.

## 2. Gradients chunk (type 1): frequency-gradient groups

| Size | Field |
|---:|---|
| 2 | Hop H, 8..=32768 (frame k is centred at sample k·H) |
| 4 | Precision p (`f32`), 1e-3..=1e6: amplitude quantum |
| 1 | W_start: width of group start deltas |
| 1 | W_dur: width of durations |
| 1 | W_count: width of member counts minus one |
| 1 | W_amp: width of amplitude indices |
| 1 | W_ratio: width of exponential ratios |
| 1 | O_ω (`i8`): frequency width offset |
| 1 | O_γ (`i8`): gradient width offset |
| 4 | Group count G, ≤ N |

All widths are at most 62. Then G bit-packed groups, in increasing start:

| Bits | Field |
|---:|---|
| W_start | Start delta: start = previous start + delta (first previous = 0), in frames |
| W_dur | Duration D in hops, ≤ 4096 |
| 2 | Volume function: 0 constant, 1 linear, 2 exponential (D = 0 requires 0) |
| W_ratio | If exponential: ratio ρ (signed), shared by all members |
| W_count | Members minus one, ≤ 4095 |

and per member:

| Bits | Field |
|---:|---|
| 5 | Precision class c, ≤ 24 |
| 1 | Fade in (the track starts here) |
| 1 | Fade out (the track ends here) |
| w_ω | Frequency index i_ω |
| w_γ | Gradient index i_γ (signed), only if D > 0 |
| W_amp | Start amplitude index i_0 |
| W_amp | End amplitude index i_1, only if the volume function is linear |
| c + 3 | Phase index i_φ |

with `s = bitlen((max(D,1) + 1)·H)`, `e_ω = c + s + 1`, `e_γ = c + 2s`,
`w_ω = max(0, O_ω + e_ω)` and `w_γ = max(0, O_γ + e_γ)`. The encoder chooses every
width and offset as the smallest that holds its largest value: just enough bits.
Parameters: `ω = i_ω / 2^e_ω`, `γ = i_γ / 2^e_γ`, `φ = 2π·i_φ / 2^(c+3)`,
`A_0 = p·i_0`, `A_1 = p·i_1` (linear) or `A_0`. The class sets the quanta so each
parameter costs about p of waveform error for a member of amplitude up to p·2^c.

**Render.** With `t` the time from `start·H`, `S = D·H` and `h = H/2`, each member
adds, for every sample `n = start·H + t` in `0..N`:

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
| 1 | W_gap ≤ 32 |
| 1 | W_val ≤ 62 |

Then, bit-packed: M - 1 gaps minus one (W_gap bits each), then M value indices
(W_val bits, signed). Positions start at 0 and add each gap; the last must be
N - 1. Values are `q·index`.

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
the range is below 2^24. An adaptive bit with probability `p` (11 bits, initial
1024) splits the range at `bound = (range >> 11)·p`; a 0 bit keeps the low part
and adds `(2048 - p) >> 5` to p, a 1 bit the high part and subtracts `p >> 5`. A
direct bit halves the range without a probability. Multi-bit direct fields are
most significant bit first.

**Per block** (the last block is shorter):

1. If the layer flag is set: gain g, 3 direct bits, 0..=4 (gain g/4).
2. Predictor order o, 6 direct bits, ≤ 32. If o > 0: shift k, 5 direct bits;
   width w, 5 direct bits, 1..=15; then o coefficients `c_i`, w direct bits each,
   signed.
3. If the cascade flag is set: stages used u, 2 direct bits, 0..=3.
4. One index q per sample (below).

**Index coding.** A running magnitude `m` starts at 16; the context is
`min(23, bitlen(m >> 3))` and after each index `m = m - (m >> 4) + min(|q|, 2^24)`.
Per context there are adaptive bits for "nonzero" and "negative", and an adaptive
gamma code for `|q| - 1`: with `v = |q|` and `l = bitlen(v)`, `l - 1` one-bits and a
zero-bit (adaptive, one probability per position), then, if `l ≥ 2`, the bit below
the leading one (adaptive, one probability per `l`), then the remaining `l - 2`
bits direct.

**Reconstruction.** With `g' = g` (4 when there is no layer flag),
`scale(v) = floor((v·g' + 2) / 4)` and `h(i) = x̂[i] - scale(P[i])`:
`pred = ((Σ_{i < min(o, n, 32)} c_i·h(n-1-i) + 2^(k-1)) >> k) + scale(P[n])`
(no rounding term when k = 0), in saturating 64-bit integers, clamped to ±2^40;
`x̂[n] = pred + q·(2τ + 1)`, clamped to ±2^40. Since the encoder chooses
`q = floor((x[n] - pred + τ) / (2τ + 1))`, every `|x̂[n] - x[n]| ≤ τ`.

**LMS cascade** (cascade flag set). Call the prediction above `base`. Three
sign-sign LMS stages follow, as in Monkey's Audio, with (taps o, shift k) =
(1024, 15), (256, 13), (16, 11). They start from zero at the start of the
channel and persist across blocks. Stage i keeps weights `W_j` and, for its
past inputs `v`, steps `d` (j = 1..o is the lag; inputs and steps before the
start of the channel are 0) and a running mean `a`, all integers. Its
prediction is `p_i = (Σ_j W_j·v(n-j) + 2^(k-1)) >> k` (64-bit). Then
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

The cascade is integer arithmetic too, so this layer decodes identically on every
platform.

## 5. What the encoder chooses

`encode_auto` measures complete coded sizes and keeps the smallest: the stereo
code (lossless only), and per channel whether a gradients chunk pays (band, p,
segment tolerance), whether an anchors chunk pays (low band or all, tolerance),
and the block exponent; inside the correction chunk every block chooses g and o
(the LPC candidates by exact size), then, with the cascade, u for the two best
candidates by exact size.
None of these choices needs to be signalled beyond the fields above.
