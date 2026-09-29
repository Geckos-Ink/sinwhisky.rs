use sinwhisky::aft::AngularFrame;
use sinwhisky::spectral::{angular_fourier, Complex};
use sinwhisky::{Analysis, AnalysisParams, Primitive};
use std::f64::consts::TAU;

fn distance(a: Complex, b: Complex) -> f64 {
    (a.re - b.re).hypot(a.im - b.im)
}

#[test]
fn aft_periodic_coverage_reuses_adjacent_angular_halves() {
    let samples = [300.0, 700.0, 900.0, 400.0, 600.0];
    let frame = AngularFrame::new(&samples, AnalysisParams::original()).unwrap();
    assert_eq!(frame.sample_count(), 5);
    assert_eq!(frame.stats().circle_halves, 10);
    let extended = [600.0, 300.0, 700.0, 900.0, 400.0, 600.0, 300.0, 700.0];
    let analysis = Analysis::new(&extended, AnalysisParams::original()).unwrap();
    for (i, sample) in samples.iter().enumerate() {
        assert!((frame.evaluate(i as f64).unwrap() - sample).abs() < 1e-9);
        for fraction in [0.1, 0.4, 0.9] {
            let t = i as f64 + fraction;
            let value: f64 = analysis.models()[i..i + 2]
                .iter()
                .map(|model| {
                    let Primitive::Circle(c) = model.primitive else {
                        panic!("circle expected")
                    };
                    c.evaluate_angular(t + 1.0).unwrap() * 0.5
                })
                .sum();
            assert!((frame.evaluate(t).unwrap() - value).abs() < 1e-9);
            assert!((frame.evaluate(t - 5.0).unwrap() - value).abs() < 1e-9);
        }
    }
    for omega in [0.0, 0.3, -1.7, 3.0] {
        let mut expected = Complex::default();
        for i in 0..5 {
            for model in &analysis.models()[i..i + 2] {
                let Primitive::Circle(c) = model.primitive else {
                    panic!("circle expected")
                };
                let a = angular_fourier(c, (i + 1) as f64..(i + 2) as f64, omega).unwrap();
                let (s, cos) = omega.sin_cos();
                expected.re += (a.re * cos - a.im * s) * 0.5;
                expected.im += (a.re * s + a.im * cos) * 0.5;
            }
        }
        assert!(distance(frame.fourier(omega).unwrap(), expected) < 1e-9);
    }
}

#[test]
fn aft_matches_independent_midpoint_quadrature() {
    let inputs = [
        vec![100.0; 9],
        vec![0.0, 0.0, 1.0, 2.0, 2.0, 2.0],
        vec![0.0, 100.0, 0.0, -30.0],
        (0..31).map(|i| 1000.0 * (i as f64 * 0.61).sin()).collect(),
    ];
    for samples in inputs {
        let frame = AngularFrame::new(&samples, AnalysisParams::original()).unwrap();
        for omega in [0.0, 1e-10, 0.7, -2.2, std::f64::consts::PI] {
            let exact = frame.fourier(omega).unwrap();
            let numeric = |q: usize| {
                let mut v = Complex::default();
                for j in 0..samples.len() * q {
                    let t = (j as f64 + 0.5) / q as f64;
                    let y = frame.evaluate(t).unwrap() / q as f64;
                    v.re += y * (omega * t).cos();
                    v.im -= y * (omega * t).sin();
                }
                v
            };
            assert!(distance(exact, numeric(1024)) < 0.003);
            assert!(distance(exact, numeric(4096)) < 0.0002);
        }
    }
}

#[test]
fn aft_lines_have_the_known_periodic_linear_spectrum() {
    // Explicitly force degenerate-model fallback, then verify sinc^2 times DFT.
    let samples = [2.0, -3.0, 7.0, 1.0, 0.5, 8.0, 9.0];
    let frame = AngularFrame::new(
        &samples,
        AnalysisParams {
            collinear_epsilon: f64::MAX,
            ..AnalysisParams::original()
        },
    )
    .unwrap();
    assert_eq!(frame.stats().line_halves, 2 * samples.len());
    for (k, actual) in frame.spectrum().unwrap().into_iter().enumerate() {
        let omega = TAU * k as f64 / samples.len() as f64;
        let s = if k == 0 {
            1.0
        } else {
            (omega * 0.5).sin() / (omega * 0.5)
        };
        let mut expected = Complex::default();
        for (i, y) in samples.iter().enumerate() {
            expected.re += y * (omega * i as f64).cos() * s * s;
            expected.im -= y * (omega * i as f64).sin() * s * s;
        }
        assert!(distance(actual, expected) < 1e-11);
    }
}

#[test]
fn aft_period_shift_symmetry_and_short_inputs() {
    let samples = [100.0, 300.0, -50.0, 20.0, 75.0, 80.0, 200.0, 0.0];
    let frame = AngularFrame::new(&samples, AnalysisParams::original()).unwrap();
    let mut shifted = samples;
    shifted.rotate_right(2);
    let other = AngularFrame::new(&shifted, AnalysisParams::original()).unwrap();
    for (k, a) in frame.spectrum().unwrap().into_iter().enumerate() {
        let omega = TAU * k as f64 / 8.0;
        let (s, c) = (-2.0 * omega).sin_cos();
        let expected = Complex {
            re: a.re * c - a.im * s,
            im: a.re * s + a.im * c,
        };
        assert!(distance(other.fourier(omega).unwrap(), expected) < 1e-9);
        let negative = frame.fourier(-omega).unwrap();
        assert!(
            distance(
                negative,
                Complex {
                    re: a.re,
                    im: -a.im
                }
            ) < 1e-9
        );
    }
    for n in [1, 2, 3, 8] {
        let constant = AngularFrame::new(&vec![2.0; n], AnalysisParams::original()).unwrap();
        let spectrum = constant.spectrum().unwrap();
        assert!((spectrum[0].re - 2.0 * n as f64).abs() < 1e-12);
        for z in &spectrum[1..] {
            assert!(z.re.hypot(z.im) < 1e-12);
        }
    }
    assert!(AngularFrame::new(&[], AnalysisParams::default()).is_err());
    assert_eq!(
        AngularFrame::new(&[f64::NAN], AnalysisParams::default()).unwrap_err(),
        sinwhisky::Error::NonFiniteSample(0)
    );
    assert!(frame.fourier(f64::INFINITY).is_err());
    assert!(frame.evaluate(f64::NAN).is_err());
}
