use sinwhisky::aft::AngularFrame;
use sinwhisky::faft::{CircleCurve, PeakParams, Rotation, Weighting};
use sinwhisky::geometric_faft::{GeometryFrame, GeometryParams, HierarchyParams};
use sinwhisky::geometry::Point;
use sinwhisky::spectral::Complex;
use sinwhisky::{Analysis, AnalysisParams, Aperture, Error, Primitive};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

fn distance(a: Complex, b: Complex) -> f64 {
    (a.re - b.re).hypot(a.im - b.im)
}
fn magnitude(a: Complex) -> f64 {
    a.re.hypot(a.im)
}
fn product(a: Complex, b: Complex) -> Complex {
    Complex {
        re: a.re * b.re - a.im * b.im,
        im: a.re * b.im + a.im * b.re,
    }
}
fn quotient(a: Complex, b: Complex) -> Complex {
    let d = b.re * b.re + b.im * b.im;
    Complex {
        re: (a.re * b.re + a.im * b.im) / d,
        im: (a.im * b.re - a.re * b.im) / d,
    }
}
/// exp(-i*angle).
fn delay(angle: f64) -> Complex {
    Complex {
        re: angle.cos(),
        im: -angle.sin(),
    }
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

/// Independent quadrature of (t-tc)^power * y(t)w(t)exp(-i*omega*(t-tc)), with tc
/// the window centre: the integrals behind the centred value and its derivatives.
fn centred_moment(
    curve: &CircleCurve,
    window: std::ops::Range<usize>,
    hann: bool,
    omega: f64,
    power: i32,
) -> Complex {
    let nodes = legendre(24);
    let panels = 8;
    let duration = window.len() as f64;
    let centre = window.start as f64 + 0.5 * duration;
    let mut sum = Complex::default();
    for unit in window.clone() {
        for p in 0..panels {
            let a = unit as f64 + p as f64 / panels as f64;
            let h = 0.5 / panels as f64;
            for &(x, w) in &nodes {
                let t = a + h * (x + 1.0);
                let weight = if hann {
                    0.5 - 0.5 * (TAU * (t - window.start as f64) / duration).cos()
                } else {
                    1.0
                };
                let local = t - centre;
                let y = curve.evaluate(t).unwrap() * weight * local.powi(power) * w * h;
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
fn rotation_rates_are_twice_the_inscribed_angles_of_the_anchor_triangle() {
    // PAPER_FAFT.md section 2: the sweeps are 2*gamma_2 and 2*gamma_0, signed by the
    // orientation of P0 P1 P2, and r = |P0 P2| / (2 sin gamma_1). Unit aperture.
    let unit = AnalysisParams {
        aperture: Aperture::Disabled,
        ..AnalysisParams::default()
    };
    let samples = [0.3, 1.9, -0.4, -2.2, 0.7, 3.1, 2.9, -1.0, 0.05, 0.6];
    let analysis = Analysis::new(&samples, unit).unwrap();
    let interior = |p: Point, q: Point, r: Point| {
        let (u, v) = ((q.x - p.x, q.y - p.y), (r.x - p.x, r.y - p.y));
        (u.0 * v.1 - u.1 * v.0).abs().atan2(u.0 * v.0 + u.1 * v.1)
    };
    let mut circles = 0;
    for (i, model) in analysis.models().iter().enumerate() {
        let Primitive::Circle(c) = model.primitive else {
            continue;
        };
        let p = [-1.0, 0.0, 1.0].map(|x| Point {
            x,
            y: samples[(i as f64 + 1.0 + x) as usize],
        });
        let orientation = ((p[1].x - p[0].x) * (p[2].y - p[1].y)
            - (p[1].y - p[0].y) * (p[2].x - p[1].x))
            .signum();
        let gamma = [
            interior(p[0], p[1], p[2]),
            interior(p[1], p[2], p[0]),
            interior(p[2], p[0], p[1]),
        ];
        assert!((c.angles[1] - c.angles[0] - orientation * 2.0 * gamma[2]).abs() < 1e-12);
        assert!((c.angles[2] - c.angles[1] - orientation * 2.0 * gamma[0]).abs() < 1e-12);
        let chord = (p[2].x - p[0].x).hypot(p[2].y - p[0].y);
        assert!((c.circle.radius - chord / (2.0 * gamma[1].sin())).abs() < 1e-10 * c.circle.radius);
        circles += 1;
    }
    assert_eq!(circles, samples.len() - 2);

    // Three samples on a crest of a*cos(w0*t): the rate is -2*atan(h/aperture), h the
    // drop. It equals the tone only for aperture a*sin(w0); the range aperture always
    // gives a quarter turn per sample, so circle rates are not signal frequencies.
    let (a, w0) = (1000.0, 0.05_f64);
    let crest = [a * w0.cos(), a, a * w0.cos()];
    let rate = |aperture| {
        let params = AnalysisParams {
            aperture,
            ..AnalysisParams::default()
        };
        let Primitive::Circle(c) = Analysis::new(&crest, params).unwrap().models()[0].primitive
        else {
            panic!("crest must fit a circle")
        };
        c.angles[1] - c.angles[0]
    };
    let matched = a * w0.sin();
    let fixed = |alpha| Aperture::Adaptive {
        gain: 0.0,
        min: alpha,
        max: alpha,
    };
    assert!((rate(fixed(matched)) + w0).abs() < 1e-12);
    // Apertures near the 1.25 drop square amplitudes near 1000 in the 2x2 system.
    assert!((rate(Aperture::SignalRange) + FRAC_PI_2).abs() < 1e-9);
    let h = a * (1.0 - w0.cos());
    assert!((rate(Aperture::Disabled) + 2.0 * h.atan()).abs() < 1e-9);
}

#[test]
fn arc_transform_is_fixed_by_its_endpoints() {
    // PAPER_FAFT.md Proposition 2: y'' = -rate^2*(y - center) on a rotation, so its
    // transform needs only the end values and slopes, the centre height and the rate:
    // [(y'_L q - y'_0) + i*omega*(y_L q - y_0) - rate^2*center*E0(-omega)] / (omega^2 - rate^2)
    // with q = exp(-i*omega*L). A line is the case rate = 0 (omega != 0).
    let cases = [
        (0.5, 1.7, 0.2, 0.9, 0.0, 3),
        (-0.3, 2.0, 2.4, -1.3, 0.0, 2),
        (0.1, 0.4, -0.7, 2.9, 0.0, 1),
        (7.0, 120.0, 1.1, 0.05, 0.0, 9),
        (0.8, 0.0, 0.0, 0.0, -0.21, 5),
    ];
    for (center, radius, phase, rate, slope, length) in cases {
        let rotation = Rotation {
            start: 0,
            length,
            center,
            radius,
            phase,
            rate,
            slope,
        };
        let faft = CircleCurve::new(vec![rotation])
            .unwrap()
            .transform(0..length, Weighting::Rectangular)
            .unwrap();
        let l = length as f64;
        let value = |u: f64| center + radius * (phase + rate * u).sin() + slope * u;
        let tangent = |u: f64| radius * rate * (phase + rate * u).cos() + slope;
        for omega in [0.0, 0.37, -1.1, 2.2, PI, 7.7] {
            if (omega * omega - rate * rate).abs() < 1e-2 {
                continue;
            }
            let q = delay(omega * l);
            // E0(-omega) = i*(q - 1)/omega, or L at omega = 0.
            let dc = if omega == 0.0 {
                Complex { re: l, im: 0.0 }
            } else {
                Complex {
                    re: -q.im / omega,
                    im: (q.re - 1.0) / omega,
                }
            };
            let (y0, yl, d0, dl) = (value(0.0), value(l), tangent(0.0), tangent(l));
            let curvature = rate * rate * center;
            let numerator = Complex {
                re: dl * q.re - d0 - omega * yl * q.im - curvature * dc.re,
                im: dl * q.im + omega * (yl * q.re - y0) - curvature * dc.im,
            };
            let denominator = omega * omega - rate * rate;
            let expected = Complex {
                re: numerator.re / denominator,
                im: numerator.im / denominator,
            };
            let actual = faft.integral(omega).unwrap();
            assert!(
                distance(actual, expected) < 1e-12 * (1.0 + magnitude(expected)),
                "rate={rate} omega={omega}: {actual:?} vs {expected:?}"
            );
        }
    }
}

#[test]
fn flat_arcs_keep_relative_accuracy_away_from_resonance() {
    // A radius of 1e8 around a curve of height ~50: the moment form rounds at about
    // eps*radius*length (~1e-8 here). The endpoint kernel works on the deviation
    // from the start value with prepared sin/cos of the sweep, so it matches the
    // quadrature of the evaluated curve to its own rounding level.
    let curve = CircleCurve::new(vec![Rotation {
        start: 0,
        length: 16,
        center: -1e8 + 50.0,
        radius: 1e8,
        phase: FRAC_PI_2 - 1e-6,
        rate: 2e-7,
        slope: 0.0,
    }])
    .unwrap();
    let faft = curve.transform(0..16, Weighting::Rectangular).unwrap();
    for omega in [0.7, 2.9, -1.3, 5.0] {
        let expected = quadrature(&curve, 0..16, false, omega);
        let actual = faft.integral(omega).unwrap();
        assert!(
            distance(actual, expected) < 1e-12 * (1.0 + magnitude(expected)),
            "{omega}: {:e}",
            distance(actual, expected)
        );
    }
}

#[test]
fn line_is_the_infinite_radius_limit_of_an_arc() {
    // PAPER_FAFT.md section 2.5: fix the start value y0, start slope d and start angle;
    // as the radius grows the rate d/(r*cos(phase)) vanishes and the arc's transform
    // approaches the line's, with an error that falls like 1/r.
    let (y0, d, phase, length) = (0.7, -0.21, 0.3_f64, 5);
    let transform = |rotation| {
        CircleCurve::new(vec![rotation])
            .unwrap()
            .transform(0..length, Weighting::Rectangular)
            .unwrap()
    };
    let line = transform(Rotation {
        start: 0,
        length,
        center: y0,
        radius: 0.0,
        phase: 0.0,
        rate: 0.0,
        slope: d,
    });
    for omega in [0.0, 0.9, -2.3, 4.0] {
        let target = line.integral(omega).unwrap();
        let errors: Vec<f64> = [1e1, 1e2, 1e3, 1e4]
            .iter()
            .map(|&radius| {
                let arc = transform(Rotation {
                    start: 0,
                    length,
                    center: y0 - radius * phase.sin(),
                    radius,
                    phase,
                    rate: d / (radius * phase.cos()),
                    slope: 0.0,
                });
                distance(arc.integral(omega).unwrap(), target)
            })
            .collect();
        for pair in errors.windows(2) {
            let ratio = pair[0] / pair[1];
            assert!((8.0..12.5).contains(&ratio), "{omega}: {errors:?}");
        }
        assert!(errors[3] < 1e-4 * (1.0 + magnitude(target)), "{errors:?}");
    }
}

#[test]
fn identical_rotations_factor_into_element_and_array_terms() {
    // PAPER_FAFT.md Remark 2: M copies of one rotation, back to back, transform to the
    // element's transform times the Dirichlet ratio of the time shifts. The narrow
    // peaks sit at the harmonics 2*pi*j/L of the spacing, whatever the circle's rate.
    let (length, count) = (4, 9);
    let piece = |start| Rotation {
        start,
        length,
        center: 0.2,
        radius: 1.3,
        phase: 0.4,
        rate: 0.8,
        slope: 0.0,
    };
    let element = CircleCurve::new(vec![piece(0)])
        .unwrap()
        .transform(0..length, Weighting::Rectangular)
        .unwrap();
    let train = CircleCurve::new((0..count).map(|k| piece(k * length)).collect())
        .unwrap()
        .transform(0..length * count, Weighting::Rectangular)
        .unwrap();
    let one = Complex { re: 1.0, im: 0.0 };
    let minus = |a: Complex, b: Complex| Complex {
        re: a.re - b.re,
        im: a.im - b.im,
    };
    for omega in [0.3, 1.1, 2.0, 2.9, 5.3, -0.7] {
        let array = quotient(
            minus(one, delay(omega * (length * count) as f64)),
            minus(one, delay(omega * length as f64)),
        );
        let expected = product(element.integral(omega).unwrap(), array);
        let actual = train.integral(omega).unwrap();
        assert!(
            distance(actual, expected) < 1e-12 * (1.0 + magnitude(expected)),
            "{omega}"
        );
    }
    for j in 1..4 {
        let omega = TAU * j as f64 / length as f64;
        let phi = element.integral(omega).unwrap();
        let expected = Complex {
            re: phi.re * count as f64,
            im: phi.im * count as f64,
        };
        assert!(distance(train.integral(omega).unwrap(), expected) < 1e-12 * magnitude(expected));
    }
}

#[test]
fn derivatives_match_quadrature_of_moments() {
    // d/domega brings down -i*(t-tc): A' = -i*M1/W and A'' = -M2/W, with M_p the
    // centred moments, at DC, near and far from resonance, and above pi.
    let curve = mixed_curve();
    for weighting in [Weighting::Rectangular, Weighting::Hann] {
        let hann = weighting == Weighting::Hann;
        for window in [0..21, 1..20, 5..9] {
            let faft = curve.transform(window.clone(), weighting).unwrap();
            let weight = window.len() as f64 * if hann { 0.5 } else { 1.0 };
            for omega in [0.0, 0.2, 0.52, 0.9, 1.31, 2.9, 3.3, 7.7, -1.1] {
                let d = faft.derivatives(omega).unwrap();
                let m = [0, 1, 2].map(|p| centred_moment(&curve, window.clone(), hann, omega, p));
                let expected = [
                    m[0],
                    Complex {
                        re: m[1].im,
                        im: -m[1].re,
                    },
                    Complex {
                        re: -m[2].re,
                        im: -m[2].im,
                    },
                ];
                for k in 0..3 {
                    let e = Complex {
                        re: expected[k].re / weight,
                        im: expected[k].im / weight,
                    };
                    assert!(
                        distance(d[k], e) < 1e-10 * (1.0 + magnitude(e)),
                        "{window:?} {weighting:?} omega={omega} k={k}: {:?} vs {e:?}",
                        d[k]
                    );
                }
            }
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
