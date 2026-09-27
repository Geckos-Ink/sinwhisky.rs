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

## Next: establish quality and cost

1. Use varied music, speech, transients, silence, and synthetic signals; compare
   both modes with linear and bandlimited interpolation. Add blind listening
   trials and spectral metrics. One music excerpt is not enough to rank quality.
2. Measure fitting/synthesis/segment-search CPU and memory before optimizing.
3. Evaluate circle candidate reuse over longer ranges, merging/refitting, and
   continuity at segment boundaries. Keep changes separate from compatibility mode.

## Codec experiments (not implemented)

1. Quantize geometry and **remeasure decoded error**, including branch/extent changes.
2. Add residuals and explicit raw blocks so difficult material has a bounded-cost
   fallback. Compare an actual byte representation against PCM and existing codecs.
3. Specify versioned framing, bounded parser sizes, exact coverage, endianness,
   channel layout, and sample normalization. Add malformed-stream/fuzz tests before
   accepting untrusted encoded data.
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
