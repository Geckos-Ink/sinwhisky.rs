//! Layered closed-loop coding (PAPER_EQ_COMPRESSION.md, Section 4): exact
//! round trips, the error contract, the tonal layer against the correction-only
//! control, and malformed streams.
use sinwhisky::bands::{ToneAnalysis, ToneParams};
use sinwhisky::layered::{decode, encode, LayeredParams};

/// Deterministic PCM16-scale test signal: a gliding and a steady tone plus noise,
/// faded in and out over 2048 samples (an abrupt start is not a tone).
fn signal(n: usize, noise: f64) -> Vec<i32> {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut phase = 0.0;
    (0..n)
        .map(|i| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let u = (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            phase += 0.031 + 0.004 * i as f64 / n as f64;
            let edge = (i.min(n - 1 - i) as f64 / 2048.0).min(1.0);
            let fade = 0.5 - 0.5 * (std::f64::consts::PI * edge).cos();
            let v =
                fade * (6000.0 * phase.sin() + 2500.0 * (0.117 * i as f64 + 0.4).sin()) + noise * u;
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

#[test]
fn lossless_and_bounded_round_trips() {
    let x = signal(12000, 40.0);
    let a = analysis(&x);
    for tolerance in [0, 3, 16, 128] {
        for tonal in [true, false] {
            let params = LayeredParams {
                tolerance,
                ..LayeredParams::default()
            };
            let encoded = encode(&x, tonal.then_some(&a), params).unwrap();
            let decoded = decode(&encoded.bytes).unwrap();
            assert_eq!(decoded, encoded.decoded, "decoder reproduces the encoder");
            let worst = x
                .iter()
                .zip(&decoded)
                .map(|(p, q)| (p - q).unsigned_abs())
                .max()
                .unwrap();
            assert!(worst <= tolerance, "tolerance {tolerance}: error {worst}");
            assert_eq!(encoded.stats.max_error, worst);
            let s = encoded.stats;
            assert_eq!(s.total_bytes, encoded.bytes.len());
            assert_eq!(
                s.header_bytes + s.tonal_bytes + s.correction_bytes,
                s.total_bytes
            );
            assert_eq!(s.tracks > 0, tonal);
        }
    }
}

#[test]
fn tonal_layer_explains_tones_and_saves_bytes_on_a_tonal_signal() {
    let x = signal(16384, 8.0);
    let a = analysis(&x);
    for tolerance in [0, 16] {
        let params = LayeredParams {
            tolerance,
            ..LayeredParams::default()
        };
        let tonal = encode(&x, Some(&a), params).unwrap().stats;
        let control = encode(&x, None, params).unwrap().stats;
        assert!(tonal.residual_rms < 0.05 * tonal.source_rms, "{tonal:?}");
        assert!(
            tonal.total_bytes < control.total_bytes,
            "{tonal:?} vs {control:?}"
        );
        // Two tones, plus short-lived tracks at the abrupt start and end.
        assert!(tonal.tracks <= 12, "{tonal:?}");
    }
}

#[test]
fn malformed_streams_are_rejected() {
    let x = signal(3000, 40.0);
    let a = analysis(&x);
    let bytes = encode(&x, Some(&a), LayeredParams::default())
        .unwrap()
        .bytes;
    assert!(decode(&bytes[..bytes.len() - 1]).is_err());
    assert!(decode(&bytes[..10]).is_err());
    let mut wrong = bytes.clone();
    wrong[0] = b'X';
    assert!(decode(&wrong).is_err());
    let mut huge = bytes.clone();
    huge[6..10].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(decode(&huge).is_err());
    // Corrupted payload bytes must not panic: either an error or some samples.
    for i in (34..bytes.len()).step_by(7) {
        let mut c = bytes.clone();
        c[i] ^= 0x5a;
        if let Ok(v) = decode(&c) {
            assert_eq!(v.len(), x.len());
        }
    }
    assert!(encode(&[], None, LayeredParams::default()).is_err());
    assert!(encode(&[1 << 23], None, LayeredParams::default()).is_err());
    let bad = LayeredParams {
        tone_precision: 0.0,
        ..LayeredParams::default()
    };
    assert!(encode(&x, None, bad).is_err());
    assert!(encode(&x[..100], Some(&a), LayeredParams::default()).is_err());
}
