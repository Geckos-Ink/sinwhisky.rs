use sinwhisky::geometric_faft::{geometric_sum, GeometryFrame, GeometryParams, WavePrimitive};
use sinwhisky::spectral::Complex;
use sinwhisky::{phasor_fft::transform, Error};
use std::f64::consts::{PI, TAU};
fn distance(a: Complex, b: Complex) -> f64 {
    (a.re - b.re).hypot(a.im - b.im)
}
fn fit(x: &[f64], e: f64) -> GeometryFrame {
    GeometryFrame::new(
        x,
        GeometryParams {
            max_error: e,
            max_span: 32,
            search_phases: true,
            min_precision: 8,
            prevalidate_geometry: true,
        },
    )
    .unwrap()
}
#[test]
fn finite_sums_match_direct_rotations_at_dc_resonance_and_wrap() {
    for n in [0, 1, 2, 17, 256, 8192] {
        for delta in [0.0, 1e-14, 1e-7, 0.2, PI, TAU, TAU + 1e-7, -1.4] {
            let mut direct = Complex::default();
            for j in 0..n {
                direct.re += (delta * j as f64).cos();
                direct.im += (delta * j as f64).sin();
            }
            assert!(
                distance(geometric_sum(delta, n), direct) < 5e-8,
                "n={n} delta={delta}"
            );
        }
    }
}
#[test]
fn geometry_only_spectra_match_predicted_pcm_fft_and_zoom() {
    let x: Vec<_> = (0..64)
        .map(|i| 1000.0 * (i as f64 * 0.33).sin() + 80.0 * (i as f64 * 1.14).cos())
        .collect();
    let frame = fit(&x, 2.0);
    assert!(frame.stats().circles > 0);
    let decoded = frame.decode().unwrap();
    for (i, y) in decoded.iter().enumerate() {
        assert!((frame.evaluate(i as f64).unwrap() - y).abs() < 1e-10);
    }
    for zoom in [1, 2, 8, 64, 256] {
        let dense: Vec<_> = (0..64 * zoom)
            .map(|j| Complex {
                re: frame.evaluate(j as f64 / zoom as f64).unwrap(),
                im: 0.0,
            })
            .collect();
        let fft = transform(&dense, false).unwrap();
        let closed = frame.spectrum_closed(zoom).unwrap();
        let rotations = frame.spectrum_rotations(zoom).unwrap();
        for k in 0..=32 {
            let expected = Complex {
                re: fft[k].re / zoom as f64,
                im: fft[k].im / zoom as f64,
            };
            assert!(
                distance(closed[k], expected) < 2e-7,
                "zoom={zoom}, k={k}, closed={:?} exp={:?}",
                closed[k],
                expected
            );
            assert!(
                distance(rotations[k], expected) < 2e-6,
                "zoom={zoom}, k={k}, rot={:?} exp={:?}",
                rotations[k],
                expected
            );
        }
    }
    assert!(frame.stats().max_error <= 2.0);
}
#[test]
fn cropped_queries_preserve_time_origin_and_conjugacy() {
    let x: Vec<_> = (0..57)
        .map(|i| 40.0 * (i as f64 * 0.47).sin() + 2.0 * i as f64)
        .collect();
    let frame = fit(&x, 0.2);
    let pcm = frame.decode().unwrap();
    for window in [0..57, 7..36, 19..21, 56..57] {
        for omega in [0.0, 1e-10, 0.13, PI, -2.1] {
            let mut direct = Complex::default();
            for (j, &v) in pcm[window.clone()].iter().enumerate() {
                direct.re += v * (omega * j as f64).cos();
                direct.im -= v * (omega * j as f64).sin();
            }
            let mut windowed = Complex::default();
            for (j, &v) in pcm[window.clone()].iter().enumerate() {
                let y = v * (0.5 - 0.5 * (TAU * j as f64 / window.len() as f64).cos());
                windowed.re += y * (omega * j as f64).cos();
                windowed.im -= y * (omega * j as f64).sin();
            }
            assert!(distance(frame.hann_dtft(omega, window.clone()).unwrap(), windowed) < 2e-7);
            let actual = frame.dtft(omega, window.clone()).unwrap();
            assert!(
                distance(actual, direct) < 1e-7,
                "{window:?} {omega} {actual:?} {direct:?}"
            );
            let reverse = frame.dtft(-omega, window.clone()).unwrap();
            assert!(
                distance(
                    reverse,
                    Complex {
                        re: actual.re,
                        im: -actual.im
                    }
                ) < 1e-7
            );
        }
    }
}
#[test]
fn circumcircle_anchors_precision_coverage_and_zero_curvature_limit() {
    // Samples of a time/amplitude circle: fitting endpoints/middle gives retained
    // radius/center from points. Equal-angle intersample values are a new policy.
    let x: Vec<_> = (0..65)
        .map(|i| 200.0 + (10000.0 - (i as f64 - 32.0).powi(2)).sqrt())
        .collect();
    let frame = fit(&x, 0.001);
    let mut end = 0;
    let mut precision = Vec::new();
    for segment in frame.segments() {
        let range = segment.range();
        assert_eq!(range.start, end);
        end = range.end;
        if let WavePrimitive::Circle(c) = segment.primitive() {
            assert!(c.radius.is_finite() && c.radius > 0.0);
            precision.push(c.precision);
            for (local, angle) in [
                (0, c.angles[0]),
                (c.middle, c.angles[1]),
                (range.len(), c.angles[2]),
            ] {
                let actual = (c.center.y + c.radius * angle.sin()) * frame.scale();
                let expected = x[(range.start + local) % x.len()];
                assert!((actual - expected).abs() <= 0.001);
            }
        }
    }
    assert_eq!(end, x.len());
    assert!(!precision.is_empty());
    let ramp: Vec<_> = (0..128).map(|i| 3.0 + 2.0 * i as f64).collect();
    let line = fit(&ramp, 1e-8);
    // A ramp is affine except for the explicit periodic wrap to its first value.
    for s in line.segments() {
        if matches!(s.primitive(), WavePrimitive::Circle(_)) {
            assert_eq!(s.range().end, ramp.len());
        }
    }
    assert!(line
        .segments()
        .iter()
        .any(|s| s.range().len() == 32 && matches!(s.primitive(), WavePrimitive::Line { .. })));
    let flat = fit(&[5.0; 64], 0.0);
    assert_eq!(flat.stats().circles, 0);
    assert!(flat.decode().unwrap().iter().all(|&v| v == 5.0));
}
#[test]
fn prediction_bounds_noise_short_signals_and_validation() {
    let mut seed = 7_u64;
    let noise: Vec<_> = (0..80)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 32) as f64 / 1e7
        })
        .collect();
    for x in [vec![3.0], vec![1.0, -2.0], noise] {
        for tolerance in [0.001, 1.0, 10.0] {
            let frame = fit(&x, tolerance);
            let output = frame.decode().unwrap();
            assert!(x
                .iter()
                .zip(output)
                .all(|(a, b)| (a - b).abs() <= tolerance));
        }
    }
    // Normalization must not hide an unrepresentable tiny sample next to a huge one.
    assert!(GeometryFrame::new(
        &[1e300, 1e-20, 1e300, 1e300],
        GeometryParams {
            max_error: 1e-30,
            ..GeometryParams::default()
        }
    )
    .is_err());
    let large_range = GeometryFrame::new(
        &[1e300, 1e-20, 1e300, 1e300],
        GeometryParams {
            max_error: 1e-19,
            ..GeometryParams::default()
        },
    )
    .unwrap();
    assert!(
        (large_range.stats().rmse - 5e-21).abs() < 1e-35,
        "{:?} {:?}",
        large_range.stats(),
        large_range.decode()
    );
    let params = GeometryParams::default();
    assert!(GeometryFrame::new(&[], params).is_err());
    assert_eq!(
        GeometryFrame::new(&[f64::NAN], params).unwrap_err(),
        Error::NonFiniteSample(0)
    );
    for invalid in [
        GeometryParams {
            max_error: -1.0,
            ..params
        },
        GeometryParams {
            max_span: 3,
            ..params
        },
        GeometryParams {
            min_precision: 2,
            ..params
        },
    ] {
        assert!(GeometryFrame::new(&[1.0, 2.0], invalid).is_err());
    }
    let frame = fit(&[1.0, 2.0], 0.01);
    assert!(frame.evaluate(f64::NAN).is_err());
    assert!(frame.spectrum_closed(0).is_err());
    assert!(frame.dtft(0.0, 0..0).is_err());
    assert!(frame.dtft(0.0, 0..3).is_err());
}
