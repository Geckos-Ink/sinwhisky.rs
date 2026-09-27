use sinwhisky::codec::{ErrorMetrics, ResearchFrame, SegmentModel};
use sinwhisky::geometry::{Branch, Circle, Point};
use sinwhisky::{resample, Analysis, AnalysisParams, Aperture, Error, Primitive, ResampleParams};

fn close(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() <= tolerance,
        "{a} != {b} (tolerance {tolerance})"
    );
}

#[test]
fn geometry_and_extent() {
    let circle = Circle::through(
        [
            Point { x: -1.0, y: 0.0 },
            Point { x: 0.0, y: 1.0 },
            Point { x: 1.0, y: 0.0 },
        ],
        1e-10,
    )
    .unwrap();
    assert_eq!(circle.branch, Branch::Upper);
    close(circle.radius, 1.0, 1e-12);
    close(circle.evaluate(0.5).unwrap(), 0.75_f64.sqrt(), 1e-12);
    assert_eq!(circle.evaluate(1.01), None);
    let translated = Circle {
        center: Point { x: 5.0, y: 2.0 },
        ..circle
    };
    assert_eq!(translated.evaluate(0.0), None);
    assert_eq!(translated.evaluate(5.0), Some(3.0));
    assert!(Circle::through([Point { x: 0.0, y: 1.0 }; 3], 1e-10).is_none());
}

#[test]
fn lengths_originals_and_reuse() {
    for n in [0, 1, 2, 3, 20] {
        let samples: Vec<f64> = (0..n).map(|i| (i as f64 * 0.43).sin()).collect();
        for analysis_params in [AnalysisParams::default(), AnalysisParams::original()] {
            let analysis = Analysis::new(&samples, analysis_params).unwrap();
            let models = analysis.models().to_vec();
            for zoom in [0, 1, 3, 7] {
                for p in [ResampleParams::new(zoom), ResampleParams::original(zoom)] {
                    let out = analysis.resample(p).unwrap();
                    assert_eq!(out.len(), if n == 0 { 0 } else { (n - 1) * (zoom + 1) + 1 });
                    for (i, value) in samples.iter().enumerate() {
                        assert_eq!(out[i * (zoom + 1)].to_bits(), value.to_bits());
                    }
                }
            }
            assert_eq!(analysis.models(), models);
        }
    }
}

#[test]
fn constant_ramp_and_two_samples() {
    for samples in [vec![2.5; 6], vec![-1.0, 1.0, 3.0, 5.0], vec![2.0, 6.0]] {
        for params in [ResampleParams::new(3), ResampleParams::original(3)] {
            let out = resample(&samples, params).unwrap();
            for (i, &y) in out.iter().enumerate() {
                close(
                    y,
                    samples[0] + i as f64 / 4.0 * (samples[1] - samples[0]),
                    1e-12,
                );
            }
        }
    }
}

#[test]
fn exact_circle_without_aperture() {
    for sign in [-1.0, 1.0] {
        let samples: Vec<f64> = (-3..=3)
            .map(|x| sign * (25.0 - (x * x) as f64).sqrt())
            .collect();
        let a = Analysis::new(
            &samples,
            AnalysisParams {
                aperture: Aperture::Disabled,
                ..AnalysisParams::default()
            },
        )
        .unwrap();
        let out = a.resample(ResampleParams::new(4)).unwrap();
        for (i, &y) in out.iter().enumerate() {
            let x = -3.0 + i as f64 / 5.0;
            close(y, sign * (25.0 - x * x).sqrt(), 1e-9);
        }
    }
}

#[test]
fn paper_angular_example() {
    let expected = [
        3.0, 4.157316, 5.213207, 6.162361, 7.0, 7.759608, 8.408535, 8.854186, 9.0, 7.977124, 6.5,
        5.022877, 4.0, 4.014061, 4.369707, 5.045354, 6.0,
    ];
    let out = resample(&[3.0, 7.0, 9.0, 4.0, 6.0], ResampleParams::original(3)).unwrap();
    for (&a, b) in out.iter().zip(expected) {
        close(a, b, 4e-6);
    }
}

#[test]
fn angular_sine_and_branch_crossing() {
    for amplitude in [0.1, 1.0, 100.0] {
        let out = resample(&[0.0, amplitude, 0.0], ResampleParams::original(3)).unwrap();
        for (i, &y) in out.iter().enumerate() {
            close(
                y / amplitude,
                (i as f64 * std::f64::consts::PI / 8.0).sin(),
                1e-12,
            );
        }
    }
    // Explicitly keep the uniform compatibility limitation visible.
    let a = Analysis::new(&[0.0, 100.0, 0.0], AnalysisParams::default()).unwrap();
    let Primitive::Circle(c) = a.models()[0].primitive else {
        panic!("expected circle")
    };
    close(c.evaluate_uniform(0.0).unwrap(), 98.56, 1e-10);
    close(c.evaluate_angular(0.0).unwrap(), 0.0, 1e-10);
    close(c.evaluate_angular(2.0).unwrap(), 0.0, 1e-10);
}

#[test]
fn validation_and_overflow() {
    assert_eq!(
        resample(&[0.0, f64::NAN], ResampleParams::new(3)),
        Err(Error::NonFiniteSample(1))
    );
    assert!(Analysis::new(
        &[],
        AnalysisParams {
            collinear_epsilon: 0.0,
            ..AnalysisParams::default()
        }
    )
    .is_err());
    for aperture in [
        Aperture::Adaptive {
            gain: f64::NAN,
            min: 1.0,
            max: 12.0,
        },
        Aperture::Adaptive {
            gain: 1.0,
            min: 2.0,
            max: 1.0,
        },
    ] {
        assert!(Analysis::new(
            &[],
            AnalysisParams {
                aperture,
                ..AnalysisParams::default()
            }
        )
        .is_err());
    }
    assert_eq!(
        ResampleParams::new(usize::MAX).output_len(2),
        Err(Error::OutputTooLarge)
    );
    assert_eq!(
        ResampleParams::new(1).output_len(usize::MAX),
        Err(Error::OutputTooLarge)
    );
    assert!(resample(
        &[0.0, 1.0],
        ResampleParams {
            neighbors: 0,
            ..ResampleParams::new(1)
        }
    )
    .is_err());
    assert!(resample(
        &[0.0, 1.0],
        ResampleParams {
            neighbors: 2,
            ..ResampleParams::original(1)
        }
    )
    .is_err());
}

#[test]
fn neighbors_and_extreme_finite_inputs() {
    let y: Vec<f64> = (0..48).map(|i| (i as f64 * 0.4).sin()).collect();
    let a = Analysis::new(&y, AnalysisParams::default()).unwrap();
    for neighbors in [1, 2, 3, usize::MAX] {
        let out = a
            .resample(ResampleParams {
                neighbors,
                ..ResampleParams::new(4)
            })
            .unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
    }
    for p in [ResampleParams::new(3), ResampleParams::original(3)] {
        let out = resample(&[f64::MAX, -f64::MAX, f64::MAX], p).unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
    }
}

#[test]
fn segments_cover_every_sample_with_bounded_error() {
    for samples in [
        vec![],
        vec![7.0],
        vec![2.5; 12],
        vec![0.0, 100.0, 0.0],
        (0..200).map(|i| (i as f64 * 0.2).sin()).collect(),
    ] {
        let analysis = Analysis::new(&samples, AnalysisParams::default()).unwrap();
        for tolerance in [0.0, 0.001, 0.05] {
            let frame = ResearchFrame::from_analysis(&analysis, 44100, tolerance).unwrap();
            let out = frame.decode().unwrap();
            assert_eq!(out.len(), samples.len());
            assert_eq!(frame.sample_count(), samples.len());
            assert_eq!(frame.sample_rate(), 44100);
            let mut end = 0;
            for segment in frame.segments() {
                assert_eq!(segment.range().start, end);
                assert!(segment.range().end > end);
                end = segment.range().end;
            }
            assert_eq!(end, samples.len());
            assert!(frame.metrics().max_absolute <= tolerance);
            assert_eq!(
                frame.metrics(),
                ErrorMetrics::between(&samples, &out).unwrap()
            );
        }
    }
}

#[test]
fn codec_reuses_circles_and_lines_and_validates_metadata() {
    let y: Vec<f64> = (-3..=3).map(|x| (25.0 - (x * x) as f64).sqrt()).collect();
    let a = Analysis::new(
        &y,
        AnalysisParams {
            aperture: Aperture::Disabled,
            ..AnalysisParams::default()
        },
    )
    .unwrap();
    let frame = ResearchFrame::from_analysis(&a, 48000, 1e-10).unwrap();
    assert_eq!(frame.segments().len(), 1);
    let SegmentModel::Circle(circle) = frame.segments()[0].model() else {
        panic!("expected circle")
    };
    assert_eq!(a.models()[0].primitive, Primitive::Circle(circle));
    assert!(ResearchFrame::from_analysis(&a, 0, 0.0).is_err());
    assert!(ResearchFrame::from_analysis(&a, 48000, -1.0).is_err());
    assert_eq!(
        ErrorMetrics::between(&[0.0], &[]),
        Err(Error::LengthMismatch)
    );
    assert_eq!(
        ErrorMetrics::between(&[f64::NAN], &[0.0]),
        Err(Error::NonFiniteSample(0))
    );
}
