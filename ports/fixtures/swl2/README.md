# Frozen SWL2 reference streams

These synthetic fixtures were copied unchanged from `appx/dsp/test/fixtures/swl2`
at `ec84edcd331af8810be9df42cf5d16f3a348d6d5` (2026-10-05). They were produced
by the Rust `swl2-reference` helper using this crate, not by the Wasm encoder.
There are no recorded audio inputs.

All use the 10,000-frame signal in `fixtureSignal` in
[`swl2.test.mjs`](../../wasm-simd/test/swl2.test.mjs), reproduced independently by
[`tests/port_swl2.rs`](../../../tests/port_swl2.rs).

| Fixture | Rust configuration |
| --- | --- |
| `stereo-lossless-segments.swl2` | `encode_auto`, tolerance 0, segment exponent 12, key interval 2, cascade on, gradients/anchors off |
| `stereo-t16.swl2` | `encode_auto`, tolerance 16, single segment, cascade on, gradients/anchors off |
| `mono-t128-lpc-b9.swl2` | `encode`, left channel only, tolerance 128, independent stereo code, block exponent 9, cascade off, no layers |

Both Rust and Node tests encode the same bytes and verify decode/error bounds;
Node additionally tests corruption and key starts. Keep these fixtures frozen.
Do not regenerate them from the port or to make a parity failure pass.
