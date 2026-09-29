use sinwhisky::codec::ResearchFrame;
use sinwhisky::spectral::{angular_fourier, frame_dft_bin, uniform_fourier, Complex};
use sinwhisky::{Analysis, AnalysisParams, Aperture, LocalCircle, Primitive};
use std::f64::consts::{PI, TAU};

fn circle(samples: &[f64], params: AnalysisParams) -> LocalCircle {
    match Analysis::new(samples, params).unwrap().models()[0].primitive {
        Primitive::Circle(c) => c,
        _ => panic!("expected a circle"),
    }
}

fn close(a: Complex, b: Complex, tolerance: f64) {
    assert!(
        (a.re - b.re).hypot(a.im - b.im) < tolerance,
        "{a:?} != {b:?}"
    );
}

#[test]
fn circle_transforms_have_known_dc_and_resonance() {
    let c = circle(&[0.0, 1.0, 0.0], AnalysisParams::default());
    close(
        angular_fourier(c, 0.0..2.0, 0.0).unwrap(),
        Complex {
            re: 4.0 / PI,
            im: 0.0,
        },
        1e-12,
    );
    // Integral sin(pi*t/2) exp(-i*pi*t/2), including the removable resonance.
    close(
        angular_fourier(c, 0.0..2.0, PI / 2.0).unwrap(),
        Complex { re: 0.0, im: -1.0 },
        1e-12,
    );
    close(
        uniform_fourier(c, 0.0..2.0, 0.0, 4096).unwrap(),
        Complex {
            re: PI / 2.0,
            im: 0.0,
        },
        1e-10,
    );
    // The continuous area differs from the sum of the three source samples.
    let a = Analysis::new(&[0.0, 1.0, 0.0], AnalysisParams::default()).unwrap();
    let frame = ResearchFrame::from_analysis(&a, 48000, 0.0).unwrap();
    close(
        frame_dft_bin(&frame, 0).unwrap(),
        Complex { re: 1.0, im: 0.0 },
        1e-12,
    );
}

#[test]
fn angular_formula_matches_independent_quadrature_and_subdivisions() {
    for samples in [[3.0, 7.0, 9.0], [0.0, 100.0, 0.0], [2.0, -1.0, 0.0]] {
        let c = circle(&samples, AnalysisParams::original());
        for omega in [0.0, 1e-12, -2.3, 0.8, 8.0] {
            let (lo, hi) = (0.2, 1.8);
            let exact = angular_fourier(c, lo..hi, omega).unwrap();
            let panels = 100_000;
            let step = (hi - lo) / panels as f64;
            let mut numeric = Complex::default();
            for i in 0..panels {
                let t = lo + (i as f64 + 0.5) * step;
                let y = c.evaluate_angular(t).unwrap();
                numeric.re += y * (omega * t).cos() * step;
                numeric.im -= y * (omega * t).sin() * step;
            }
            close(exact, numeric, 1e-7);
            let left = angular_fourier(c, lo..0.7, omega).unwrap();
            let right = angular_fourier(c, 0.7..hi, omega).unwrap();
            close(
                exact,
                Complex {
                    re: left.re + right.re,
                    im: left.im + right.im,
                },
                1e-10,
            );
        }
    }
}

#[test]
fn uniform_transform_converges_with_extent_branch_aperture_and_phase() {
    for sign in [-1.0, 1.0] {
        let c = circle(
            &[0.0, sign, 0.0],
            AnalysisParams {
                aperture: Aperture::Disabled,
                ..AnalysisParams::default()
            },
        );
        for omega in [0.0, -1.7, 5.0] {
            let coarse = uniform_fourier(c, 0.1..1.7, omega, 1024).unwrap();
            let fine = uniform_fourier(c, 0.1..1.7, omega, 8192).unwrap();
            let panels = 100_000;
            let step = 1.6 / panels as f64;
            let mut numeric = Complex::default();
            for i in 0..panels {
                let t = 0.1 + (i as f64 + 0.5) * step;
                let y = c.evaluate_uniform(t).unwrap();
                numeric.re += y * (omega * t).cos() * step;
                numeric.im -= y * (omega * t).sin() * step;
            }
            close(fine, numeric, 1e-7);
            close(coarse, fine, 1e-5);
            let shifted = LocalCircle {
                anchor: c.anchor + 7,
                ..c
            };
            let z = uniform_fourier(shifted, 7.1..8.7, omega, 8192).unwrap();
            let (s, cos) = (-7.0 * omega).sin_cos();
            close(
                z,
                Complex {
                    re: fine.re * cos - fine.im * s,
                    im: fine.re * s + fine.im * cos,
                },
                1e-11,
            );
            let scaled = LocalCircle { aperture: 2.0, ..c };
            let z = uniform_fourier(scaled, 0.55..1.35, omega * 2.0, 8192).unwrap();
            let (s, cos) = (-omega).sin_cos();
            close(
                z,
                Complex {
                    re: (fine.re * cos - fine.im * s) / 2.0,
                    im: (fine.re * s + fine.im * cos) / 2.0,
                },
                1e-11,
            );
        }
    }
}

#[test]
fn descriptor_dft_matches_decoded_signal_and_parseval() {
    let samples: Vec<_> = (0..64)
        .map(|i| (TAU * i as f64 * 3.0 / 64.0).sin())
        .collect();
    let analysis = Analysis::new(&samples, AnalysisParams::default()).unwrap();
    let models = analysis.models().to_vec();
    let frame = ResearchFrame::from_analysis(&analysis, 44100, 0.01).unwrap();
    let decoded = frame.decode().unwrap();
    let mut energy = 0.0;
    for k in 0..samples.len() {
        let mut expected = Complex::default();
        for (i, y) in decoded.iter().enumerate() {
            let phase = TAU * k as f64 * i as f64 / samples.len() as f64;
            expected.re += y * phase.cos();
            expected.im -= y * phase.sin();
        }
        let actual = frame_dft_bin(&frame, k).unwrap();
        close(actual, expected, 1e-10);
        energy += actual.re.powi(2) + actual.im.powi(2);
    }
    assert!(
        (energy / samples.len() as f64 - decoded.iter().map(|y| y * y).sum::<f64>()).abs() < 1e-10
    );
    assert_eq!(analysis.models(), models);
    assert!(frame_dft_bin(&frame, samples.len()).is_err());
}

#[test]
fn invalid_transform_requests_fail() {
    let c = circle(&[0.0, 1.0, 0.0], AnalysisParams::default());
    for range in [0.0..3.0, 2.0..1.0, 1.0..1.0, f64::NAN..1.0] {
        assert!(angular_fourier(c, range.clone(), 0.0).is_err());
        assert!(uniform_fourier(c, range, 0.0, 64).is_err());
    }
    assert!(angular_fourier(c, 0.0..2.0, f64::INFINITY).is_err());
    assert!(uniform_fourier(c, 0.0..2.0, 0.0, 0).is_err());
    assert!(uniform_fourier(c, 0.0..2.0, 0.0, usize::MAX).is_err());
    assert!(uniform_fourier(LocalCircle { aperture: 0.0, ..c }, 0.0..2.0, 0.0, 64).is_err());
    let analysis = Analysis::new(&[], AnalysisParams::default()).unwrap();
    assert!(frame_dft_bin(
        &ResearchFrame::from_analysis(&analysis, 44100, 0.0).unwrap(),
        0
    )
    .is_err());
}
