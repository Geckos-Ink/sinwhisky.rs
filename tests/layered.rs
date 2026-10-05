//! SWL2 layered coding (PAPER_EQ_COMPRESSION.md, Section 4; docs/SWL2.md): exact
//! and bounded round trips with every layer, with and without the LMS cascade,
//! stereo decorrelation, the parametric mode, segments with key and continuation
//! frames, random access, the automatic search and malformed streams.
use sinwhisky::bands::{ToneAnalysis, ToneParams};
use sinwhisky::layered::{
    decode, decode_from, encode, encode_auto, AnchorConfig, AutoParams, ChannelConfig,
    GradientConfig, SegmentConfig, Stereo, StreamConfig, PARAMETRIC,
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
    StreamConfig::single(
        tolerance,
        Stereo::Independent,
        vec![ChannelConfig::default(); channels],
    )
}

fn single(tolerance: u32, channel: ChannelConfig) -> StreamConfig {
    StreamConfig::single(tolerance, Stereo::Independent, vec![channel])
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
            cascade: false,
        },
    ];
    for tolerance in [0, 3, 16, 128] {
        for layer in layers
            .iter()
            .flat_map(|&l| [false, true].map(|cascade| ChannelConfig { cascade, ..l }))
        {
            let config = single(tolerance, layer);
            let encoded = encode(std::slice::from_ref(&x), &[vec![Some(&a)]], &config).unwrap();
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
        let config = StreamConfig::single(0, stereo, vec![ChannelConfig::default(); 2]);
        let encoded = encode(&channels, &[vec![None, None]], &config).unwrap();
        assert_eq!(decode(&encoded.bytes).unwrap(), channels, "{stereo:?}");
    }
    let lossy = StreamConfig::single(4, Stereo::MidSide, vec![ChannelConfig::default(); 2]);
    assert!(encode(&channels, &[vec![None, None]], &lossy).is_err());
}

#[test]
fn linear_prediction_codes_tones_far_below_noise() {
    let tonal = signal(16384, 8.0, 3);
    let mut s = 11;
    let noisy: Vec<i32> = (0..16384)
        .map(|_| (6000.0 * noise(&mut s)).round() as i32)
        .collect();
    let size = |x: &Vec<i32>| {
        encode(std::slice::from_ref(x), &[vec![None]], &plain(0, 1))
            .unwrap()
            .stats
    };
    let (t, n) = (size(&tonal), size(&noisy));
    // Two tones are predictable by a short LPC; white noise is not.
    assert!(t.total_bytes * 3 < n.total_bytes, "{t:?} vs {n:?}");
    assert!(t.channels[0].mean_order >= 2.0);
}

/// The LMS stages reach far behind the order-32 LPC: noise repeating with a
/// period of 200 samples is unpredictable for the LPC and learnt by the cascade.
#[test]
fn cascade_learns_what_the_lpc_cannot_reach() {
    let mut s = 5;
    let period: Vec<f64> = (0..200).map(|_| 4000.0 * noise(&mut s)).collect();
    let repeating: Vec<i32> = (0..32768)
        .map(|i| (period[i % 200] + 20.0 * noise(&mut s)).round() as i32)
        .collect();
    let white: Vec<i32> = (0..32768)
        .map(|_| (4000.0 * noise(&mut s)).round() as i32)
        .collect();
    for tolerance in [0, 16] {
        let size = |x: &Vec<i32>, cascade: bool| {
            let config = single(
                tolerance,
                ChannelConfig {
                    cascade,
                    ..ChannelConfig::default()
                },
            );
            let encoded = encode(std::slice::from_ref(x), &[vec![None]], &config).unwrap();
            assert_eq!(decode(&encoded.bytes).unwrap(), encoded.decoded);
            assert!(worst(std::slice::from_ref(x), &encoded.decoded) <= tolerance);
            encoded.stats
        };
        let (lpc, lms) = (size(&repeating, false), size(&repeating, true));
        assert_eq!(lpc.channels[0].mean_stages, 0.0);
        assert!(lms.channels[0].mean_stages > 0.0, "{lms:?}");
        assert!(
            lms.total_bytes * 4 < lpc.total_bytes * 3,
            "tolerance {tolerance}: {} vs {}",
            lms.total_bytes,
            lpc.total_bytes
        );
        // On white noise the blocks can decline the stages; little is lost.
        let (lpc, lms) = (size(&white, false), size(&white, true));
        assert!(
            lms.total_bytes * 100 <= lpc.total_bytes * 101,
            "tolerance {tolerance}: {} vs {}",
            lms.total_bytes,
            lpc.total_bytes
        );
    }
}

#[test]
fn parametric_streams_keep_only_the_layers() {
    let x = signal(12000, 8.0, 5);
    let a = analysis(&x);
    let config = single(
        PARAMETRIC,
        ChannelConfig {
            gradients: Some(gradients()),
            ..ChannelConfig::default()
        },
    );
    let encoded = encode(std::slice::from_ref(&x), &[vec![Some(&a)]], &config).unwrap();
    assert_eq!(decode(&encoded.bytes).unwrap(), encoded.decoded);
    assert_eq!(encoded.stats.channels[0].correction_bytes, 0);
    // Two tones and weak noise: the gradients alone carry most of the signal.
    assert!(encoded.stats.snr_db > 20.0, "{:?}", encoded.stats);
    assert!(encode(
        std::slice::from_ref(&x),
        &[vec![Some(&a)]],
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
            // Lossy cascade trials rerun the stages per choice (slow unoptimized);
            // the round-trip test covers them.
            cascade: tolerance == 0,
            segment_exponent: 0,
            key_interval: 0,
        };
        let (encoded, trials) = encode_auto(&channels, &params).unwrap();
        assert_eq!(decode(&encoded.bytes).unwrap(), encoded.decoded);
        assert!(worst(&channels, &encoded.decoded) <= tolerance);
        for c in 0..2 {
            let mine: Vec<_> = trials.iter().filter(|t| t.channel == c).collect();
            // The first trial is the correction layer alone.
            let alone = mine[0];
            assert!(alone.config.gradients.is_none() && alone.config.anchors.is_none());
            assert_eq!(alone.config.cascade, params.cascade);
            let chosen = mine
                .iter()
                .find(|t| t.config == encoded.config.segments[0].channels[c])
                .unwrap();
            assert!(chosen.bytes <= alone.bytes);
            assert_eq!(chosen.bytes, mine.iter().map(|t| t.bytes).min().unwrap());
        }
        if tolerance > 0 {
            assert_eq!(encoded.config.segments[0].stereo, Stereo::Independent);
        }
    }
}

#[test]
fn malformed_streams_are_rejected() {
    let x = signal(3000, 40.0, 2);
    let a = analysis(&x);
    let config = single(
        0,
        ChannelConfig {
            gradients: Some(gradients()),
            anchors: Some(anchors(None)),
            ..ChannelConfig::default()
        },
    );
    let bytes = encode(std::slice::from_ref(&x), &[vec![Some(&a)]], &config)
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
    assert!(encode(&[vec![]], &[vec![None]], &plain(0, 1)).is_err());
    assert!(encode(&[vec![1 << 23]], &[vec![None]], &plain(0, 1)).is_err());
    let needs_analysis = single(
        0,
        ChannelConfig {
            gradients: Some(gradients()),
            ..ChannelConfig::default()
        },
    );
    assert!(encode(std::slice::from_ref(&x), &[vec![None]], &needs_analysis).is_err());
    // The first segment must be a key.
    let mut first = plain(0, 1);
    first.segments[0].key = false;
    assert!(encode(std::slice::from_ref(&x), &[vec![None]], &first).is_err());
    let mut flags = bytes.clone();
    flags[20] &= !1;
    assert!(decode(&flags).is_err());
}

/// Segments: continuations inherit the correction state, keys restart it and
/// can be decoded alone; every segment chooses its own stereo mode and layers.
#[test]
fn segments_continue_or_restart_and_keys_decode_alone() {
    let n = 3 * 4096 + 1000;
    let l = signal(n, 30.0, 4);
    let r: Vec<i32> = l.iter().map(|v| v / 2 + 3).collect();
    let channels = vec![l, r];
    let a = |x: &[i32]| analysis(x);
    let ranges = [0..4096, 4096..8192, 8192..12288, 12288..n];
    // Gradients in segment 1 on the left channel (an independent segment).
    let tonal = a(&channels[0][4096..8192]);
    let layered = ChannelConfig {
        gradients: Some(gradients()),
        ..ChannelConfig::default()
    };
    let segment = |key: bool, stereo: Stereo, first: ChannelConfig| SegmentConfig {
        key,
        stereo,
        channels: vec![first, ChannelConfig::default()],
    };
    for tolerance in [0, 16] {
        let mid = if tolerance == 0 {
            Stereo::MidSide
        } else {
            Stereo::Independent
        };
        let config = StreamConfig {
            tolerance,
            segment_exponent: 12,
            segments: vec![
                segment(true, Stereo::Independent, ChannelConfig::default()),
                segment(false, Stereo::Independent, layered),
                segment(true, mid, ChannelConfig::default()),
                segment(false, mid, ChannelConfig::default()),
            ],
        };
        let analyses = vec![
            vec![None, None],
            vec![Some(&tonal), None],
            vec![None, None],
            vec![None, None],
        ];
        let encoded = encode(&channels, &analyses, &config).unwrap();
        assert_eq!(encoded.stats.segments, 4);
        assert_eq!(encoded.stats.keys, 2);
        assert!(encoded.stats.channels[0].groups > 0);
        let decoded = decode(&encoded.bytes).unwrap();
        assert_eq!(decoded, encoded.decoded);
        assert!(worst(&channels, &decoded) <= tolerance);
        // Random access at the second key: the same samples from there on.
        let tail = decode_from(&encoded.bytes, 2).unwrap();
        for (t, d) in tail.iter().zip(&decoded) {
            assert_eq!(t[..], d[8192..]);
        }
        assert!(decode_from(&encoded.bytes, 1).is_err(), "not a key");
        assert!(decode_from(&encoded.bytes, 4).is_err(), "no such segment");
        // A continuation is cheaper than a key over the same samples.
        let mut keys = config.clone();
        keys.segments[3].key = true;
        let restarted = encode(&channels, &analyses, &keys).unwrap();
        assert!(restarted.bytes.len() > encoded.bytes.len());
        assert!(worst(&channels, &decode(&restarted.bytes).unwrap()) <= tolerance);
        let _ = &ranges;
    }
    // The automatic search per segment, with a key every second segment.
    let params = AutoParams {
        tolerance: 0,
        tones: ToneParams::default(),
        gradients: false,
        anchors: false,
        cascade: true,
        segment_exponent: 12,
        key_interval: 2,
    };
    let (encoded, trials) = encode_auto(&channels, &params).unwrap();
    let keys: Vec<bool> = encoded.config.segments.iter().map(|s| s.key).collect();
    assert_eq!(keys, [true, false, true, false]);
    assert_eq!(decode(&encoded.bytes).unwrap(), channels);
    assert_eq!(
        decode_from(&encoded.bytes, 2).unwrap()[1][..],
        channels[1][8192..]
    );
    assert!(trials.iter().any(|t| t.segment == 3));
}
