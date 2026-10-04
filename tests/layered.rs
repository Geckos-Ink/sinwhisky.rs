//! SWL2 layered coding (PAPER_EQ_COMPRESSION.md, Section 4; docs/SWL2.md): exact
//! and bounded round trips with every layer, stereo decorrelation, the parametric
//! mode, the automatic search and malformed streams.
use sinwhisky::bands::{ToneAnalysis, ToneParams};
use sinwhisky::layered::{
    decode, encode, encode_auto, AnchorConfig, AutoParams, ChannelConfig, GradientConfig, Stereo,
    StreamConfig, PARAMETRIC,
};

fn noise(seed: &mut u64) -> f64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    (*seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

/// PCM16-scale gliding and steady tones plus noise, faded in and out.
fn signal(n: usize, noise_level: f64, seed: u64) -> Vec<i32> {
    let mut s = seed;
    let mut phase = 0.0;
    (0..n)
        .map(|i| {
            phase += 0.031 + 0.004 * i as f64 / n as f64;
            let edge = (i.min(n - 1 - i) as f64 / 2048.0).min(1.0);
            let fade = 0.5 - 0.5 * (std::f64::consts::PI * edge).cos();
            let v = fade * (6000.0 * phase.sin() + 2500.0 * (0.117 * i as f64 + 0.4).sin())
                + noise_level * noise(&mut s);
            v.round() as i32
        })
        .collect()
}

fn analysis(x: &[i32]) -> ToneAnalysis {
    let samples: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    ToneAnalysis::new(
        &samples,
        ToneParams {
            window: 1024,
            max_error: 2.0,
            ..ToneParams::default()
        },
    )
    .unwrap()
}

fn gradients() -> GradientConfig {
    GradientConfig {
        lower: 0.0,
        upper: std::f64::consts::PI,
        precision: 2.0,
        segment_tolerance: 4.0,
        min_amplitude: 0.0,
        max_jump_bins: 1.0,
    }
}

fn anchors(cutoff: Option<f64>) -> AnchorConfig {
    AnchorConfig {
        cutoff,
        tolerance: 8.0,
        step: 2.0,
        max_gap: 512,
    }
}

fn worst(a: &[Vec<i32>], b: &[Vec<i32>]) -> u32 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).unsigned_abs()))
        .max()
        .unwrap()
}

fn plain(tolerance: u32, channels: usize) -> StreamConfig {
    StreamConfig {
        tolerance,
        stereo: Stereo::Independent,
        channels: vec![ChannelConfig::default(); channels],
    }
}

#[test]
fn every_layer_round_trips_within_its_tolerance() {
    let x = signal(12000, 40.0, 7);
    let a = analysis(&x);
    let layers = [
        ChannelConfig::default(),
        ChannelConfig {
            gradients: Some(gradients()),
            ..ChannelConfig::default()
        },
        ChannelConfig {
            anchors: Some(anchors(Some(0.05))),
            ..ChannelConfig::default()
        },
        ChannelConfig {
            gradients: Some(gradients()),
            anchors: Some(anchors(None)),
            block_exponent: 10,
        },
    ];
    for tolerance in [0, 3, 16, 128] {
        for layer in layers {
            let config = StreamConfig {
                channels: vec![layer],
                ..plain(tolerance, 1)
            };
            let encoded = encode(std::slice::from_ref(&x), &[Some(&a)], &config).unwrap();
            let decoded = decode(&encoded.bytes).unwrap();
            assert_eq!(decoded, encoded.decoded, "decoder reproduces the encoder");
            let error = worst(std::slice::from_ref(&x), &decoded);
            assert!(
                error <= tolerance,
                "tolerance {tolerance}: {error} with {layer:?}"
            );
            assert_eq!(encoded.stats.max_error, error);
            assert_eq!(encoded.stats.total_bytes, encoded.bytes.len());
            let s = encoded.stats.channels[0];
            assert_eq!(s.groups > 0, layer.gradients.is_some());
            assert_eq!(s.anchors > 0, layer.anchors.is_some());
        }
    }
}

#[test]
fn stereo_decorrelation_is_lossless_and_rejected_when_lossy() {
    let l = signal(9000, 30.0, 1);
    let r: Vec<i32> = l
        .iter()
        .enumerate()
        .map(|(i, v)| v / 2 + (i as i32 % 7) - 3)
        .collect();
    let channels = vec![l, r];
    for stereo in [
        Stereo::Independent,
        Stereo::LeftSide,
        Stereo::SideRight,
        Stereo::MidSide,
    ] {
        let config = StreamConfig {
            stereo,
            ..plain(0, 2)
        };
        let encoded = encode(&channels, &[None, None], &config).unwrap();
        assert_eq!(decode(&encoded.bytes).unwrap(), channels, "{stereo:?}");
    }
    let lossy = StreamConfig {
        stereo: Stereo::MidSide,
        ..plain(4, 2)
    };
    assert!(encode(&channels, &[None, None], &lossy).is_err());
}

#[test]
fn linear_prediction_codes_tones_far_below_noise() {
    let tonal = signal(16384, 8.0, 3);
    let mut s = 11;
    let noisy: Vec<i32> = (0..16384)
        .map(|_| (6000.0 * noise(&mut s)).round() as i32)
        .collect();
    let size = |x: &Vec<i32>| {
        encode(std::slice::from_ref(x), &[None], &plain(0, 1))
            .unwrap()
            .stats
    };
    let (t, n) = (size(&tonal), size(&noisy));
    // Two tones are predictable by a short LPC; white noise is not.
    assert!(t.total_bytes * 3 < n.total_bytes, "{t:?} vs {n:?}");
    assert!(t.channels[0].mean_order >= 2.0);
}

#[test]
fn parametric_streams_keep_only_the_layers() {
    let x = signal(12000, 8.0, 5);
    let a = analysis(&x);
    let config = StreamConfig {
        channels: vec![ChannelConfig {
            gradients: Some(gradients()),
            ..ChannelConfig::default()
        }],
        ..plain(PARAMETRIC, 1)
    };
    let encoded = encode(std::slice::from_ref(&x), &[Some(&a)], &config).unwrap();
    assert_eq!(decode(&encoded.bytes).unwrap(), encoded.decoded);
    assert_eq!(encoded.stats.channels[0].correction_bytes, 0);
    // Two tones and weak noise: the gradients alone carry most of the signal.
    assert!(encoded.stats.snr_db > 20.0, "{:?}", encoded.stats);
    assert!(encode(
        std::slice::from_ref(&x),
        &[Some(&a)],
        &plain(PARAMETRIC - 1, 1)
    )
    .is_err());
}

#[test]
fn automatic_search_never_loses_to_the_correction_layer_alone() {
    let x = signal(10000, 20.0, 9);
    let channels = vec![x.clone(), x.iter().map(|v| v / 3).collect()];
    for tolerance in [0, 16] {
        let params = AutoParams {
            tolerance,
            tones: ToneParams {
                window: 1024,
                max_error: 2.0,
                ..ToneParams::default()
            },
            gradients: true,
            anchors: true,
        };
        let (encoded, trials) = encode_auto(&channels, &params).unwrap();
        assert_eq!(decode(&encoded.bytes).unwrap(), encoded.decoded);
        assert!(worst(&channels, &encoded.decoded) <= tolerance);
        for c in 0..2 {
            let mine: Vec<_> = trials.iter().filter(|t| t.channel == c).collect();
            let alone = mine
                .iter()
                .find(|t| t.config == ChannelConfig::default())
                .unwrap();
            let chosen = mine
                .iter()
                .find(|t| t.config == encoded.config.channels[c])
                .unwrap();
            assert!(chosen.bytes <= alone.bytes);
            assert_eq!(chosen.bytes, mine.iter().map(|t| t.bytes).min().unwrap());
        }
        if tolerance > 0 {
            assert_eq!(encoded.config.stereo, Stereo::Independent);
        }
    }
}

#[test]
fn malformed_streams_are_rejected() {
    let x = signal(3000, 40.0, 2);
    let a = analysis(&x);
    let config = StreamConfig {
        channels: vec![ChannelConfig {
            gradients: Some(gradients()),
            anchors: Some(anchors(None)),
            ..ChannelConfig::default()
        }],
        ..plain(0, 1)
    };
    let bytes = encode(std::slice::from_ref(&x), &[Some(&a)], &config)
        .unwrap()
        .bytes;
    assert!(decode(&bytes[..bytes.len() - 1]).is_err());
    assert!(decode(&bytes[..10]).is_err());
    let mut extra = bytes.clone();
    extra.push(0);
    assert!(decode(&extra).is_err());
    let mut wrong = bytes.clone();
    wrong[0] = b'X';
    assert!(decode(&wrong).is_err());
    let mut huge = bytes.clone();
    huge[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(decode(&huge).is_err());
    // Corrupted bytes must not panic: either an error or samples of the right shape.
    for i in (16..bytes.len()).step_by(5) {
        let mut c = bytes.clone();
        c[i] ^= 0x5a;
        if let Ok(v) = decode(&c) {
            assert_eq!(v.len(), 1);
            assert_eq!(v[0].len(), x.len());
        }
    }
    assert!(encode(&[vec![]], &[None], &plain(0, 1)).is_err());
    assert!(encode(&[vec![1 << 23]], &[None], &plain(0, 1)).is_err());
    let needs_analysis = StreamConfig {
        channels: vec![ChannelConfig {
            gradients: Some(gradients()),
            ..ChannelConfig::default()
        }],
        ..plain(0, 1)
    };
    assert!(encode(std::slice::from_ref(&x), &[None], &needs_analysis).is_err());
}
