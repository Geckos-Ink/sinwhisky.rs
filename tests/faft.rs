use sinwhisky::faft::{transform, Baseline, FaftFrame, FaftParams, LineLimit, Precision};
use sinwhisky::spectral::Complex;
use sinwhisky::Error;
use std::f64::consts::TAU;

fn model(x: &[f64], eps: f64, min_block: usize) -> FaftFrame {
    FaftFrame::new(
        x,
        FaftParams {
            max_rmse: eps,
            min_block,
        },
    )
    .unwrap()
}

#[test]
fn rotations_match_direct_dft_and_inverse_including_edges() {
    for n in [1, 2, 4, 8, 32, 128] {
        let x: Vec<_> = (0..n)
            .map(|i| Complex {
                re: (i as f64 * 1.234).sin() + 0.25,
                im: (i as f64 * 0.377).cos(),
            })
            .collect();
        let f = transform(&x, false).unwrap();
        for (k, z) in f.iter().enumerate() {
            let mut expected = Complex::default();
            for (j, v) in x.iter().enumerate() {
                let (s, c) = (-TAU * k as f64 * j as f64 / n as f64).sin_cos();
                expected.re += v.re * c - v.im * s;
                expected.im += v.re * s + v.im * c;
            }
            assert!((z.re - expected.re).hypot(z.im - expected.im) < 2e-11);
        }
        for (a, b) in x.iter().zip(transform(&f, true).unwrap()) {
            assert!((a.re - b.re).hypot(a.im - b.im) < 1e-13);
        }
    }
    let n = 64;
    let x: Vec<_> = (0..n)
        .map(|i| 10.0 + if i % 2 == 0 { 3.0 } else { -3.0 })
        .collect();
    let frame = model(&x, 1e-10, n);
    assert_eq!(frame.stats().circles, 1);
    assert_eq!(frame.blocks()[0].circles()[0].bin, n / 2);
    assert!(frame.stats().max_error < 1e-10);
}

#[test]
fn sparse_circles_have_phase_precision_and_source_independent_decode() {
    let n = 256;
    let x: Vec<_> = (0..n)
        .map(|i| {
            let t = TAU * i as f64 / n as f64;
            20.0 + 1000.0 * (7.0 * t + 0.3).cos() + 0.8 * (31.0 * t - 0.7).cos()
        })
        .collect();
    let frame = model(&x, 0.005, n);
    assert_eq!(frame.stats().circles, 2);
    let bits: Vec<_> = frame.blocks()[0]
        .circles()
        .iter()
        .map(|c| c.precision.bits())
        .collect();
    assert!(bits[0] > bits[1], "{bits:?}");
    for c in frame.blocks()[0].circles() {
        if let Precision::Quantized { bits, re, im, .. } = c.precision {
            let maximum = (1_i32 << (bits - 1)) - 1;
            assert!(re.abs() <= maximum && im.abs() <= maximum);
        }
    }
    let decoded = frame.decode().unwrap();
    let rms = (x
        .iter()
        .zip(&decoded)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    assert!(rms <= 0.005);
    assert!((rms - frame.stats().rmse).abs() < 1e-11);
    for (i, v) in decoded.iter().enumerate() {
        assert!((frame.evaluate(i as f64).unwrap() - v).abs() < 2e-10);
    }
    // Fractional evaluation follows the retained projected circles, not AFT arcs.
    for t in [0.25, 12.5, 100.7] {
        let phase = TAU * t / n as f64;
        let expected = 20.0 + 1000.0 * (7.0 * phase + 0.3).cos() + 0.8 * (31.0 * phase - 0.7).cos();
        assert!((frame.evaluate(t).unwrap() - expected).abs() < 0.02);
    }
    drop(x);
    assert_eq!(decoded, frame.decode().unwrap());
}

#[test]
fn children_replace_parent_when_local_rotations_need_fewer_descriptors() {
    let x: Vec<_> = (0..256)
        .map(|i| {
            let bin = if i < 128 { 5.0 } else { 17.0 };
            (TAU * bin * (i % 128) as f64 / 128.0 + 0.2).cos()
        })
        .collect();
    let root = model(&x, 1e-5, 256);
    let tree = model(&x, 1e-5, 32);
    assert!(tree.stats().descriptors < root.stats().descriptors);
    assert_eq!(tree.stats().blocks, 2);
    assert_eq!(tree.stats().circles, 2);
    assert_eq!(tree.blocks()[0].range(), 0..128);
    assert_eq!(tree.blocks()[1].range(), 128..256);
    assert!(tree.stats().rmse <= 1e-5);
}

#[test]
fn line_limit_is_explicit_and_large_radius_converges() {
    let x: Vec<_> = (0..128).map(|i| 7.0 + 2.0 * i as f64).collect();
    let frame = model(&x, 1e-10, 16);
    assert_eq!(frame.stats().circles, 0);
    assert_eq!(frame.stats().line_blocks, 1);
    assert!(matches!(frame.blocks()[0].baseline(), Baseline::Line(_)));
    assert!((frame.evaluate(10.5).unwrap() - 28.0).abs() < 1e-12);
    let line = LineLimit {
        first: -1.0,
        last: 3.0,
    };
    let mut previous = f64::INFINITY;
    for radius in [10.0_f64, 100.0, 1e4, 1e8] {
        // A circle tangent to y=2x+1. Use the rationalized sagitta;
        // r-sqrt(r*r-t*t) would cancel as curvature approaches zero.
        let t = 0.5_f64;
        let sag = t * t / (radius + (radius * radius - t * t).sqrt());
        let cos = 1.0 / 5.0_f64.sqrt();
        let sin = 2.0 * cos;
        let px = t * cos - sag * sin;
        let py = 1.0 + t * sin + sag * cos;
        let error = (py - line.evaluate((px + 1.0) / 2.0)).abs();
        assert!(error < previous);
        previous = error;
    }
    assert!(previous < 3e-9);
    let near_line: Vec<_> = (0..128).map(|i| x[i] + 1e-8 * (i as f64).sin()).collect();
    assert_eq!(model(&near_line, 1e-6, 16).stats().circles, 0);
    let flat = model(&vec![8.0; 128], 1e-12, 16);
    assert_eq!(flat.stats().descriptors, 1);
    assert_eq!(flat.stats().circles, 0);
}

#[test]
fn bounds_coverage_and_validation_for_broad_spectra() {
    let mut seed = 1_u64;
    let x: Vec<_> = (0..512)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 32) as f64 / u32::MAX as f64 * 2.0 - 1.0
        })
        .collect();
    for eps in [0.5, 0.01, 1e-10] {
        let frame = model(&x, eps, 32);
        if eps == 1e-10 {
            assert!(frame
                .blocks()
                .iter()
                .flat_map(|b| b.circles())
                .any(|c| c.precision.bits() == 64));
        }
        for circle in frame.blocks().iter().flat_map(|b| b.circles()) {
            if let Precision::Quantized { bits, re, im, .. } = circle.precision {
                let maximum = (1_i32 << (bits - 1)) - 1;
                assert!(re.abs() <= maximum && im.abs() <= maximum);
            }
        }
        let y = frame.decode().unwrap();
        assert_eq!(y.len(), x.len());
        let rms = (x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / 512.0).sqrt();
        assert!(rms <= eps);
        let mut end = 0;
        for block in frame.blocks() {
            assert_eq!(block.range().start, end);
            end = block.range().end;
        }
        assert_eq!(end, x.len());
        assert!(frame.evaluate(-1.0).is_err());
        assert!(frame.evaluate(512.0).is_err());
        assert!(frame.evaluate(f64::NAN).is_err());
    }
    for x in [vec![], vec![1.0], vec![1.0; 3]] {
        assert!(FaftFrame::new(
            &x,
            FaftParams {
                max_rmse: 1.0,
                min_block: 2
            }
        )
        .is_err());
    }
    for eps in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(FaftFrame::new(
            &[0.0; 8],
            FaftParams {
                max_rmse: eps,
                min_block: 2
            }
        )
        .is_err());
    }
    assert_eq!(
        FaftFrame::new(
            &[0.0, f64::NAN],
            FaftParams {
                max_rmse: 1.0,
                min_block: 2
            }
        )
        .unwrap_err(),
        Error::NonFiniteSample(1)
    );
    assert!(transform(&[], false).is_err());
    assert!(transform(
        &[Complex {
            re: f64::MAX,
            im: 0.0
        }; 2],
        false
    )
    .is_err());
    // Internal normalization keeps huge but finite amplitudes representable.
    assert!(FaftFrame::new(
        &[1e200, -1e200, 1e200, -1e200],
        FaftParams {
            max_rmse: 1e190,
            min_block: 2
        }
    )
    .is_ok());
}
