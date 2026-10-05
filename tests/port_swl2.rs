//! The frozen synthetic SWL2 fixtures transposed from appx/dsp must also agree
//! with this checkout's Rust encoder. Never regenerate them from WebAssembly.
use sinwhisky::bands::ToneParams;
use sinwhisky::layered::{
    decode, decode_from, encode, encode_auto, AutoParams, ChannelConfig, Stereo, StreamConfig,
};

fn fixture_signal(frames: usize) -> Vec<Vec<i32>> {
    // Match JavaScript's binary64 LCG and Math.round, including negative ties.
    let mut state = 12345.0;
    let mut noise = || {
        state = (state * 1_103_515_245.0 + 12_345.0) % 2_147_483_648.0;
        (state / 2_147_483_648.0 - 0.5) * 400.0
    };
    let mut left = Vec::with_capacity(frames);
    let mut right = Vec::with_capacity(frames);
    for i in 0..frames {
        let t = i as f64;
        let l = (8000.0 * (0.031 * t).sin() + 3000.0 * (0.47 * t + 1.0).sin() + noise() + 0.5)
            .floor() as i32;
        let r = (0.7 * f64::from(l) + 2000.0 * (0.12 * t).sin() + noise() + 0.5).floor() as i32;
        left.push(l);
        right.push(r);
    }
    vec![left, right]
}

#[test]
fn frozen_port_streams_match_current_rust_bytes_and_error_bounds() {
    let source = fixture_signal(10_000);
    let lossless = include_bytes!("../ports/fixtures/swl2/stereo-lossless-segments.swl2");
    let lossy = include_bytes!("../ports/fixtures/swl2/stereo-t16.swl2");
    let lpc = include_bytes!("../ports/fixtures/swl2/mono-t128-lpc-b9.swl2");
    for (bytes, tolerance, segment_exponent, key_interval) in [
        (lossless.as_slice(), 0, 12, 2),
        (lossy.as_slice(), 16, 0, 0),
    ] {
        let params = AutoParams {
            tolerance,
            tones: ToneParams::default(),
            gradients: false,
            anchors: false,
            cascade: true,
            segment_exponent,
            key_interval,
        };
        let (encoded, _) = encode_auto(&source, &params).unwrap();
        assert_eq!(encoded.bytes, bytes);
        let decoded = decode(bytes).unwrap();
        assert_eq!(decoded, encoded.decoded);
        for (got, original) in decoded.iter().flatten().zip(source.iter().flatten()) {
            assert!(got.abs_diff(*original) <= tolerance);
        }
    }
    let tail: Vec<Vec<i32>> = source.iter().map(|c| c[8192..].to_vec()).collect();
    assert_eq!(decode_from(lossless, 2).unwrap(), tail);
    let config = StreamConfig::single(
        128,
        Stereo::Independent,
        vec![ChannelConfig {
            gradients: None,
            anchors: None,
            block_exponent: 9,
            cascade: false,
        }],
    );
    let encoded = encode(&source[..1], &[vec![None]], &config).unwrap();
    assert_eq!(encoded.bytes, lpc);
    assert_eq!(decode(lpc).unwrap(), encoded.decoded);
    for (got, original) in encoded.decoded[0].iter().zip(&source[0]) {
        assert!(got.abs_diff(*original) <= 128);
    }
}
