# Initial local listening experiment

Run on 2026-09-27 using the user-provided Beethoven MP3 (320 kbps, stereo,
44.1 kHz). Only the eight-second excerpt starting at 60 seconds was used.
The source recording and generated audio are not distributed in this repository.
The reference is decoded MP3 audio, not an uncompressed studio master.

## Procedure

[The reproducible script](../tools/audio_compare.py) decodes to PCM16, applies a
common gain of 0.8, then a 257-tap symmetric Hamming-windowed sinc filter with
cutoff `0.45/4` cycles per source sample. Reflect padding avoids zero padding at
clip boundaries. Keep every fourth sample (11,025 Hz), then reconstruct to
44,100 Hz using linear interpolation, uniform-x SinWhisky, or original angular
SinWhisky. Process stereo channels independently. All methods retain the same
low-rate originals exactly; no independent volume normalization is applied.

Because output covers the interval from the first through last retained sample,
the aligned result contains 352,797 frames. Metrics exclude 256 frames at either
edge and use floating-point output before final PCM16 quantization. No clipping
occurred in the measured region. These are waveform error measurements, not
perceptual audio scores; no subjective listening verdict has been established.

## Results

| Interpolation | RMS error vs decoded source | SNR vs source (dB) | RMS error vs bandlimited reference | SNR vs bandlimited (dB) |
|---|---:|---:|---:|---:|
| Linear | 0.00263062 | 26.31 | 0.00239867 | 27.11 |
| Uniform-x SinWhisky | 0.00147046 | 31.37 | 0.00102508 | 34.49 |
| Original angular SinWhisky | 0.00164228 | 30.41 | 0.00125395 | 32.74 |

Both circle methods reduce sample-domain error relative to linear interpolation
on this excerpt; the uniform mode has the lowest error here. Comparison with the
bandlimited reference isolates interpolation from the intentionally discarded
high frequencies. It does not establish a universal quality ranking.

## Independent bitrate controls

Full-rate reference audio was separately encoded with FFmpeg 7.1 / libmp3lame,
then decoded and aligned. These files do **not** use circle compression.

| MP3 target bitrate | File bytes (including metadata) | RMS error vs reference | SNR (dB) |
|---|---:|---:|---:|
| 64 kbps stereo | 64,617 | 0.00531987 | 20.20 |
| 128 kbps stereo | 129,192 | 0.00406145 | 22.54 |

A lower waveform error for an interpolator is not evidence that it outperforms
MP3 at those bitrates: the interpolator has no encoded representation or bitrate
in this experiment, and MP3 optimizes perceptual rather than sample-domain error.
Re-encoding an existing MP3 also introduces generation loss.

## Repeat and extend

Build the release binary and run the command in the [README](../README.md).
The output directory contains source/filtered/low-rate WAVs, each reconstruction,
bitrate-control MP3s/WAVs, and `report.json`. Listen at matched playback gain.
Keep audio and local paths outside version control. Next tests need varied
material, transient/spectral analysis, a bandlimited reconstruction baseline,
blind comparisons, and actual encoded-byte measurements for a future codec.
