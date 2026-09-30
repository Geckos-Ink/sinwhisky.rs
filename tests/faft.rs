use sinwhisky::aft::AngularFrame;
use sinwhisky::faft::{CircleCurve, PeakParams, Rotation, Weighting};
use sinwhisky::geometric_faft::{GeometryFrame, GeometryParams, HierarchyParams};
use sinwhisky::spectral::Complex;
use sinwhisky::{AnalysisParams, Error};
use std::f64::consts::{PI, TAU};

fn distance(a: Complex, b: Complex) -> f64 {
    (a.re - b.re).hypot(a.im - b.im)
}
fn magnitude(a: Complex) -> f64 {
    a.re.hypot(a.im)
}

/// Gauss-Legendre nodes/weights on [-1,1], computed independently by Newton.
fn legendre(n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let mut x = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
            let mut derivative = 0.0;
            for _ in 0..100 {
                let (mut p0, mut p1) = (1.0, x);
                for k in 2..=n {
                    let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                    p0 = p1;
                    p1 = p2;
                }
                derivative = n as f64 * (x * p1 - p0) / (x * x - 1.0);
                let dx = p1 / derivative;
                x -= dx;
                if dx.abs() < 1e-16 {
                    break;
                }
            }
            (x, 2.0 / ((1.0 - x * x) * derivative * derivative))
        })
        .collect()
}

/// Independent quadrature of y(t)w(t)exp(-i*omega*(t-start)) using only the curve's
/// pointwise evaluation, split at integer points where rotations may join.
fn quadrature(
    curve: &CircleCurve,
    window: std::ops::Range<usize>,
    hann: bool,
    omega: f64,
) -> Complex {
    let nodes = legendre(24);
    let panels = 8;
    let duration = window.len() as f64;
    let mut sum = Complex::default();
    for unit in window.clone() {
        for p in 0..panels {
            let a = unit as f64 + p as f64 / panels as f64;
            let h = 0.5 / panels as f64;
            for &(x, w) in &nodes {
                let t = a + h * (x + 1.0);
                let local = t - window.start as f64;
                let weight = if hann {
                    0.5 - 0.5 * (TAU * local / duration).cos()
                } else {
                    1.0
                };
                let y = curve.evaluate(t).unwrap() * weight * w * h;
                sum.re += y * (omega * local).cos();
                sum.im -= y * (omega * local).sin();
            }
        }
    }
    sum
}

fn mixed_curve() -> CircleCurve {
    let arc = |start, length, center, radius, phase, rate| Rotation {
        start,
        length,
        center,
        radius,
        phase,
        rate,
        slope: 0.0,
    };
    CircleCurve::new(vec![
        arc(0, 3, 0.5, 1.7, 0.2, 0.9),
        // Negative angular direction and a flat large-radius arc.
        arc(3, 2, -0.3, 2.0, 2.4, -1.3),
        arc(5, 4, -9.9e5, 1e6, PI * 0.5 - 3e-4, 1.5e-4),
        Rotation {
            start: 9,
            length: 5,
            center: 0.8,
            radius: 0.0,
            phase: 0.0,
            rate: 0.0,
            slope: -0.21,
        },
        // Two stacked rotations on one support add together.
        arc(14, 1, 0.1, 0.4, -0.7, 2.9),
        arc(14, 1, -0.2, 0.3, 1.1, -0.4),
        arc(15, 6, 0.0, 0.9, 0.0, 0.52),
    ])
    .unwrap()
}

#[test]
fn closed_form_matches_independent_quadrature_at_continuous_frequencies() {
    let curve = mixed_curve();
    assert_eq!(curve.len(), 21);
    for window in [0..21, 2..17, 5..9, 14..15] {
        for weighting in [Weighting::Rectangular, Weighting::Hann] {
            let faft = curve.transform(window.clone(), weighting).unwrap();
            // DC, tiny, resonant with each rate, both signs, above source Nyquist.
            for omega in [
                0.0, 1e-13, 1e-6, 0.52, -0.52, 0.9, 1.3, 2.9, PI, 0.3137, 7.7, -11.0,
            ] {
                let expected =
                    quadrature(&curve, window.clone(), weighting == Weighting::Hann, omega);
                let actual = faft.integral(omega).unwrap();
                assert!(
                    distance(actual, expected) < 1e-9 * (1.0 + magnitude(expected)),
                    "{window:?} {weighting:?} omega={omega}: {actual:?} vs {expected:?}"
                );
                let normalized = faft.value(omega).unwrap();
                let sum = match weighting {
                    Weighting::Rectangular => window.len() as f64,
                    Weighting::Hann => window.len() as f64 * 0.5,
                };
                assert!(
                    distance(
                        normalized,
                        Complex {
                            re: actual.re / sum,
                            im: actual.im / sum
                        }
                    ) < 1e-12
                );
            }
        }
    }
}

#[test]
fn analytic_derivatives_match_finite_differences() {
    let curve = mixed_curve();
    for weighting in [Weighting::Rectangular, Weighting::Hann] {
        let faft = curve.transform(1..20, weighting).unwrap();
        let centered = |omega: f64| {
            let z = faft.value(omega).unwrap();
            let (s, c) = (omega * 19.0 * 0.5).sin_cos();
            Complex {
                re: z.re * c - z.im * s,
                im: z.re * s + z.im * c,
            }
        };
        for omega in [0.0, 0.2, 0.9, 1.7, 3.3] {
            let d = faft.derivatives(omega).unwrap();
            assert!(distance(d[0], centered(omega)) < 1e-12);
            let h = 1e-5;
            let (a, b) = (centered(omega - h), centered(omega + h));
            let first = Complex {
                re: (b.re - a.re) / (2.0 * h),
                im: (b.im - a.im) / (2.0 * h),
            };
            // The second difference uses a wider step: the 1e6-radius arc carries
            // ~1e-10 roundoff that 1/h^2 would amplify.
            let wide = 1e-4;
            let (a2, b2, mid) = (
                centered(omega - wide),
                centered(omega + wide),
                centered(omega),
            );
            let second = Complex {
                re: (b2.re - 2.0 * mid.re + a2.re) / (wide * wide),
                im: (b2.im - 2.0 * mid.im + a2.im) / (wide * wide),
            };
            // Finite differences: truncation ~h^2 and roundoff ~eps/h^2, relative.
            let (m1, m2) = (magnitude(first), magnitude(second));
            assert!(
                distance(d[1], first) < 1e-8 * (1.0 + m1),
                "{weighting:?} {omega}"
            );
            assert!(
                distance(d[2], second) < 1e-5 * (1.0 + m2),
                "{weighting:?} {omega}"
            );
        }
    }
}

#[test]
fn same_curve_as_integral_aft_reference_without_integration() {
    let x: Vec<_> = (0..64)
        .map(|i| 900.0 * (i as f64 * 0.41).sin() + 120.0 * (i as f64 * 1.9).cos() + 7.0)
        .collect();
    let aft = AngularFrame::new(&x, AnalysisParams::original()).unwrap();
    let curve = CircleCurve::from_angular(&aft).unwrap();
    assert_eq!(curve.len(), 64);
    assert_eq!(curve.rotations().len(), 128);
    for i in 0..64 {
        for u in [0.0, 0.25, 0.5, 0.875] {
            let t = i as f64 + u;
            assert!((curve.evaluate(t).unwrap() - aft.evaluate(t).unwrap()).abs() < 1e-9);
        }
    }
    let faft = curve.transform(0..64, Weighting::Rectangular).unwrap();
    for omega in [0.0, TAU / 64.0, 0.41, 1.0 / 3.0, 1.9, PI, 4.2, -0.8] {
        let reference = aft.fourier(omega).unwrap();
        let actual = faft.integral(omega).unwrap();
        assert!(
            distance(actual, reference) < 1e-9 * (1.0 + magnitude(reference)),
            "{omega}: {actual:?} vs {reference:?}"
        );
    }
}

fn hierarchical(x: &[f64], error: f64) -> GeometryFrame {
    GeometryFrame::hierarchical(
        x,
        GeometryParams {
            max_error: error,
            max_span: 64,
            min_precision: 12,
            ..GeometryParams::default()
        },
        HierarchyParams::default(),
    )
    .unwrap()
    .0
}

#[test]
fn geometry_curve_is_the_fitted_predictor_and_zoom_only_samples_it() {
    let n = 96;
    let x: Vec<_> = (0..n)
        .map(|i| 1000.0 * (i as f64 * 0.37 + 0.2).sin() + 90.0 * (i as f64 * 1.21).cos())
        .collect();
    let frame = hierarchical(&x, 1.0);
    let curve = CircleCurve::from_geometry(&frame).unwrap();
    for t in [0.0, 0.5, 3.25, 17.0, 50.125, 95.9] {
        assert!((curve.evaluate(t).unwrap() - frame.evaluate(t).unwrap()).abs() < 1e-8);
    }
    // Zoom z samples the curve; its sampled transform approaches the FAFT integral
    // only as z grows, and equals FAFT plus spectral images (Poisson summation).
    let faft = curve.transform(0..n, Weighting::Hann).unwrap();
    let sampled = |z: usize, omega: f64| {
        let mut sum = Complex::default();
        for j in 0..n * z {
            let t = j as f64 / z as f64;
            let y =
                curve.evaluate(t).unwrap() * (0.5 - 0.5 * (TAU * t / n as f64).cos()) / z as f64;
            sum.re += y * (omega * t).cos();
            sum.im -= y * (omega * t).sin();
        }
        sum
    };
    for omega in [0.37, 1.21, 2.5] {
        let exact = faft.integral(omega).unwrap();
        let errors: Vec<_> = [1, 4, 16]
            .iter()
            .map(|&z| distance(sampled(z, omega), exact))
            .collect();
        assert!(
            errors[1] < errors[0] * 0.25 && errors[2] < errors[1] * 0.25,
            "{errors:?}"
        );
        let z = 4;
        let mut images = exact;
        for m in 1..=4000 {
            for sign in [-1.0, 1.0] {
                let image = faft.integral(omega + sign * TAU * (z * m) as f64).unwrap();
                images.re += image.re;
                images.im += image.im;
            }
        }
        // Rounded geometry may jump by a tolerance-bounded amount at a join: the
        // sampled sum takes the right value, Poisson the midpoint of the limits.
        for t in 1..n {
            let t = t as f64;
            let jump = curve.evaluate(t).unwrap() - curve.evaluate(t - 1e-9).unwrap();
            let half = 0.5 * jump * (0.5 - 0.5 * (TAU * t / n as f64).cos()) / z as f64;
            images.re += half * (omega * t).cos();
            images.im -= half * (omega * t).sin();
        }
        let zoomed = sampled(z, omega);
        assert!(
            distance(zoomed, images) < 1e-3 * distance(zoomed, exact),
            "{omega}: {} vs {}",
            distance(zoomed, images),
            distance(zoomed, exact)
        );
    }
}

#[test]
fn pure_rotation_peak_is_off_grid_and_exact() {
    let n = 256;
    let omega0 = 0.473_829_1;
    let tone = CircleCurve::new(vec![Rotation {
        start: 0,
        length: n,
        center: 0.0,
        radius: 1.0,
        phase: 0.3,
        rate: omega0,
        slope: 0.0,
    }])
    .unwrap();
    let faft = tone.transform(0..n, Weighting::Hann).unwrap();
    let result = faft.peaks(PeakParams::default()).unwrap();
    let peak = result
        .peaks
        .iter()
        .max_by(|a, b| a.magnitude().total_cmp(&b.magnitude()))
        .unwrap();
    // Only the conjugate lobe's leakage separates the peak from the rate.
    assert!((peak.omega - omega0).abs() < 1e-6, "{peak:?}");
    assert!((peak.magnitude() - 0.5).abs() < 1e-4);
    assert!(peak.curvature < 0.0);
    let bins = peak.omega / faft.resolution();
    assert!(
        (bins - bins.round()).abs() > 0.05,
        "peak should be between DFT bins"
    );
    let d = faft.derivatives(peak.omega).unwrap();
    assert!((d[0].re * d[1].re + d[0].im * d[1].im).abs() < 1e-9);
    // Magnitude is locally maximal at the refined continuous frequency.
    for h in [1e-4, 1e-3] {
        assert!(magnitude(faft.value(peak.omega + h).unwrap()) < peak.magnitude());
        assert!(magnitude(faft.value(peak.omega - h).unwrap()) < peak.magnitude());
    }
    assert!(result.stats.unconverged == 0);
}

#[test]
fn fitted_tones_are_found_by_scan_and_by_circle_references() {
    let n = 512;
    let (w1, w2) = (0.3137, 1.1291);
    let x: Vec<_> = (0..n)
        .map(|i| 800.0 * (w1 * i as f64 + 0.4).sin() + 300.0 * (w2 * i as f64).cos())
        .collect();
    let frame = hierarchical(&x, 0.5);
    let curve = CircleCurve::from_geometry(&frame).unwrap();
    let faft = curve.transform(0..n, Weighting::Hann).unwrap();
    let scan = faft
        .peaks(PeakParams {
            relative_floor: 0.1,
            ..PeakParams::default()
        })
        .unwrap();
    for w in [w1, w2] {
        assert!(
            scan.peaks.iter().any(|p| (p.omega - w).abs() < 2e-3),
            "{w} missing from {:?}",
            scan.peaks
        );
    }
    let strongest = scan
        .peaks
        .iter()
        .max_by(|a, b| a.magnitude().total_cmp(&b.magnitude()))
        .unwrap();
    assert!((strongest.omega - w1).abs() < 2e-3);
    assert!((strongest.magnitude() - 400.0).abs() < 8.0);
    // No two reported peaks within the default merge width.
    for pair in scan.peaks.windows(2) {
        assert!(pair[1].omega - pair[0].omega > 0.25 * faft.resolution());
    }
    let references = faft.references(faft.resolution(), 0.01).unwrap();
    assert!(!references.is_empty());
    for r in &references {
        assert!(r.highest - r.lowest <= faft.resolution() + 1e-15);
        assert!(r.lowest <= r.omega && r.omega <= r.highest && r.weight > 0.0);
    }
    let circles = faft
        .peaks(PeakParams {
            scan_density: None,
            circle_seeds: true,
            relative_floor: 0.1,
            ..PeakParams::default()
        })
        .unwrap();
    assert_eq!(circles.stats.scan_points, 0);
    assert!(circles.stats.references > 0);
    // Circle rates are geometric seeds: the refined values are exact FAFT maxima
    // wherever they converge, but completeness is measured, not guaranteed.
    for p in &circles.peaks {
        let d = faft.derivatives(p.omega).unwrap();
        assert!(
            (d[0].re * d[1].re + d[0].im * d[1].im).abs() < 1e-6 * p.magnitude().powi(2) + 1e-12
        );
    }
}

#[test]
fn close_tones_remain_unresolved_in_a_short_window() {
    let n = 512;
    let rate = 8000.0;
    let (a, b) = (TAU * 401.337 / rate, TAU * 403.337 / rate);
    let x: Vec<_> = (0..n)
        .map(|i| (a * i as f64).sin() + (b * i as f64).sin())
        .collect();
    let faft = CircleCurve::from_geometry(&hierarchical(&x, 1e-4))
        .unwrap()
        .transform(0..n, Weighting::Hann)
        .unwrap();
    let peaks = faft
        .peaks(PeakParams {
            relative_floor: 0.2,
            ..PeakParams::default()
        })
        .unwrap()
        .peaks;
    // Arbitrary query precision does not separate tones closer than ~2*pi/T.
    assert_eq!(peaks.len(), 1, "{peaks:?}");
    assert!(peaks[0].omega > a && peaks[0].omega < b);
}

#[test]
fn validation_and_explicit_limits() {
    let arc = Rotation {
        start: 0,
        length: 4,
        center: 0.0,
        radius: 1.0,
        phase: 0.0,
        rate: 0.5,
        slope: 0.0,
    };
    assert!(CircleCurve::new(vec![]).is_err());
    assert!(CircleCurve::new(vec![Rotation { length: 0, ..arc }]).is_err());
    assert!(CircleCurve::new(vec![Rotation {
        rate: f64::NAN,
        ..arc
    }])
    .is_err());
    // Gaps, overlaps and stacks with different supports are rejected.
    assert!(CircleCurve::new(vec![arc, Rotation { start: 5, ..arc }]).is_err());
    assert!(CircleCurve::new(vec![arc, Rotation { start: 2, ..arc }]).is_err());
    assert!(CircleCurve::new(vec![arc, Rotation { length: 3, ..arc }]).is_err());
    assert!(CircleCurve::new(vec![Rotation { start: 1, ..arc }]).is_err());
    let curve = CircleCurve::new(vec![arc, Rotation { start: 4, ..arc }]).unwrap();
    assert_eq!(curve.len(), 8);
    assert!(curve.evaluate(8.0).is_err() && curve.evaluate(-0.1).is_err());
    assert!(curve.transform(3..3, Weighting::Hann).is_err());
    assert!(curve.transform(0..9, Weighting::Hann).is_err());
    let faft = curve.transform(0..8, Weighting::Rectangular).unwrap();
    assert_eq!(faft.rotation_count(), 2);
    assert_eq!(
        faft.value(f64::INFINITY).unwrap_err(),
        Error::InvalidParameter("FAFT frequency")
    );
    assert!(faft.values(&[0.1, f64::NAN]).is_err());
    assert!(faft.derivatives(f64::NAN).is_err());
    assert!(faft.references(-1.0, 0.1).is_err());
    let params = PeakParams::default();
    for invalid in [
        PeakParams {
            lower: 1.0,
            upper: 1.0,
            ..params
        },
        PeakParams {
            scan_density: Some(0),
            ..params
        },
        PeakParams {
            scan_density: None,
            circle_seeds: false,
            ..params
        },
        PeakParams {
            relative_floor: 1.5,
            ..params
        },
        PeakParams {
            merge_width: Some(-1.0),
            ..params
        },
        PeakParams {
            max_peaks: 0,
            ..params
        },
        PeakParams {
            tolerance: 0.0,
            ..params
        },
    ] {
        assert!(faft.peaks(invalid).is_err());
    }
    // Silence: every value is exactly zero and no peak is reported.
    let silence = CircleCurve::new(vec![Rotation { radius: 0.0, ..arc }])
        .unwrap()
        .transform(0..4, Weighting::Hann)
        .unwrap();
    assert_eq!(silence.value(0.7).unwrap(), Complex::default());
    assert!(silence.peaks(params).unwrap().peaks.is_empty());
}
