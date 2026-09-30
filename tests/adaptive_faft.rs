use sinwhisky::adaptive_faft::{AdaptiveSpectrum, FrequencyParams, StopReason};
use sinwhisky::geometric_faft::{GeometryFrame, GeometryParams, HierarchyParams};
use sinwhisky::spectral::Complex;
use std::f64::consts::{PI, TAU};
fn fit(x: &[f64], error: f64) -> GeometryFrame {
    GeometryFrame::hierarchical(
        x,
        GeometryParams {
            max_error: error,
            max_span: 117,
            min_precision: 12,
            ..GeometryParams::default()
        },
        HierarchyParams::default(),
    )
    .unwrap()
    .0
}
fn distance(a: Complex, b: Complex) -> f64 {
    (a.re - b.re).hypot(a.im - b.im)
}
#[test]
fn arbitrary_support_children_merge_only_after_source_validation() {
    let x: Vec<_> = (0..131)
        .map(|j| (j as f64 * 0.17).sin() + 0.07 * (j as f64 * 0.53).cos())
        .collect();
    let p = GeometryParams {
        max_error: 0.01,
        max_span: 63,
        min_precision: 8,
        ..GeometryParams::default()
    };
    let (unmerged, initial) = GeometryFrame::hierarchical(
        &x,
        p,
        HierarchyParams {
            merge_passes: 0,
            ..HierarchyParams::default()
        },
    )
    .unwrap();
    let (merged, stats) = GeometryFrame::hierarchical(&x, p, HierarchyParams::default()).unwrap();
    assert!(stats.splits > 0);
    assert!(stats.merges_accepted > 0, "{stats:?}");
    assert_eq!(initial.initial_segments, unmerged.segments().len());
    assert_eq!(
        stats.initial_segments - stats.merges_accepted,
        merged.segments().len()
    );
    assert!(merged.segments().len() < unmerged.segments().len());
    assert!(merged
        .segments()
        .iter()
        .any(|s| !s.range().len().is_power_of_two()));
    let mut at = 0;
    for s in merged.segments() {
        assert_eq!(s.range().start, at);
        at = s.range().end;
        assert!(s.range().len() <= 63);
    }
    assert_eq!(at, x.len());
    for (&a, b) in x.iter().zip(merged.decode().unwrap()) {
        assert!((a - b).abs() <= 0.01);
    }
    let (_, limited) = GeometryFrame::hierarchical(
        &x,
        p,
        HierarchyParams {
            max_depth: 0,
            merge_passes: 0,
            ..HierarchyParams::default()
        },
    )
    .unwrap();
    assert!(limited.depth_limited > 0 && limited.fallback_intervals > 0);
}
#[test]
fn prepared_inverse_rotation_sums_match_direct_arbitrary_frequencies() {
    let x: Vec<_> = (0..93)
        .map(|j| 100.0 * (j as f64 * 0.31).sin() + 5.0 * (j as f64 * 1.17).cos())
        .collect();
    let f = fit(&x, 0.1);
    let coarse = fit(&x, 3.0);
    for zoom in [1, 2, 8] {
        for hann in [false, true] {
            let spectrum = f.prepare_spectrum(7..88, zoom, hann).unwrap();
            let c = coarse.prepare_spectrum(7..88, zoom, hann).unwrap();
            let bound = spectrum.difference_bound(&c).unwrap();
            let n = 81 * zoom;
            let components: Vec<_> = spectrum.rotations().collect();
            for omega in [0.0, 1e-9, 0.310293, -0.741, PI] {
                let mut direct = Complex::default();
                for j in 0..n {
                    let v = f.evaluate(7.0 + j as f64 / zoom as f64).unwrap();
                    let part = components
                        .iter()
                        .find(|p| p.start <= j && j < p.start + p.length)
                        .unwrap();
                    let t = (j - part.start) as f64;
                    let inverse = (part.constant
                        + part.sine * (part.rate * t).sin()
                        + part.cosine * (part.rate * t).cos()
                        + part.slope * t)
                        * spectrum.scale();
                    assert!((v - inverse).abs() < 1e-7);
                    let weight = if hann {
                        0.5 - 0.5 * (TAU * j as f64 / n as f64).cos()
                    } else {
                        1.0
                    };
                    let y = v * weight / (n as f64 * if hann { 0.5 } else { 1.0 });
                    direct.re += y * (omega * j as f64 / zoom as f64).cos();
                    direct.im -= y * (omega * j as f64 / zoom as f64).sin();
                }
                let actual = spectrum.value(omega).unwrap();
                assert!(distance(actual, direct) < 1e-8);
                assert!(distance(spectrum.value_rotations(omega).unwrap(), direct) < 1e-8);
                assert!(distance(actual, c.value(omega).unwrap()) <= bound + 1e-8);
                let delta = 1e-5;
                let change = (actual.re.hypot(actual.im)
                    - spectrum
                        .value(omega + delta)
                        .unwrap()
                        .re
                        .hypot(spectrum.value(omega + delta).unwrap().im))
                .abs();
                assert!(change <= spectrum.lipschitz_bound() * delta + 1e-9);
            }
        }
    }
}
#[test]
fn non_bin_peak_refines_and_close_candidates_merge() {
    let n = 256;
    let omega = 0.473829;
    let x: Vec<_> = (0..n)
        .map(|j| (omega * j as f64 + 0.3).sin() + 0.4 * (1.11361 * j as f64).cos())
        .collect();
    let f = fit(&x, 0.0001);
    let s = f.prepare_spectrum(0..n, 1, true).unwrap();
    let params = FrequencyParams {
        initial_intervals: 128,
        min_width: 1e-6,
        relative_floor: 0.1,
        max_evaluations: 20000,
        ..FrequencyParams::default()
    };
    let a = AdaptiveSpectrum::search(&s, None, params).unwrap();
    let peak = a
        .peaks
        .iter()
        .max_by(|a, b| {
            a.value
                .re
                .hypot(a.value.im)
                .total_cmp(&b.value.re.hypot(b.value.im))
        })
        .unwrap();
    assert!((peak.omega - omega).abs() < 2e-5, "{peak:?} {:?}", a.stats);
    assert!((peak.omega / (TAU / n as f64)).fract().abs() > 0.01);
    assert!(a.peaks.iter().any(|p| (p.omega - 1.11361).abs() < 2e-4));
    assert!(a.bands.iter().any(|b| b.reason == StopReason::LowGain));
    assert!(!a.stats.budget_exhausted);
    let merged = AdaptiveSpectrum::search(
        &s,
        None,
        FrequencyParams {
            merge_width: 1.0,
            ..params
        },
    )
    .unwrap();
    assert!(merged.peaks.len() < a.peaks.len());
    assert!(merged.peaks.iter().any(|p| p.merged_candidates > 1));
}
#[test]
fn coarse_precision_pruning_budget_depth_and_quiet_bands_are_explicit() {
    let x: Vec<_> = (0..128).map(|j| (0.7321 * j as f64).sin()).collect();
    let f = fit(&x, 0.0001);
    let c = fit(&x, 0.003);
    let s = f.prepare_spectrum(0..128, 1, true).unwrap();
    let c = c.prepare_spectrum(0..128, 1, true).unwrap();
    let params = FrequencyParams {
        initial_intervals: 64,
        relative_floor: 0.1,
        max_evaluations: 10000,
        ..FrequencyParams::default()
    };
    let a = AdaptiveSpectrum::search(&s, Some(&c), params).unwrap();
    assert!(a.stats.skipped_fine > 0 && a.stats.disagreement_bound > 0.0);
    assert!(a
        .points
        .iter()
        .filter(|p| p.verified)
        .all(|p| distance(p.value, s.value(p.omega).unwrap()) < 1e-9));
    for p in a.points.iter().filter(|p| !p.verified) {
        assert!(distance(p.value, s.value(p.omega).unwrap()) <= p.uncertainty + 1e-9);
    }
    for budget in [3, 4, 17, 256] {
        let a = AdaptiveSpectrum::search(
            &s,
            Some(&c),
            FrequencyParams {
                max_evaluations: budget,
                ..params
            },
        )
        .unwrap();
        assert!(a.stats.coarse_queries + a.stats.fine_queries <= budget);
        assert!(a.stats.budget_exhausted);
        let mut at = 0.0;
        for b in &a.bands {
            assert!((b.lower - at).abs() < 1e-13);
            at = b.upper;
        }
        assert_eq!(at, PI);
    }
    let depth = AdaptiveSpectrum::search(
        &s,
        None,
        FrequencyParams {
            max_depth: 0,
            prune_flat: false,
            ..params
        },
    )
    .unwrap();
    assert!(depth.bands.iter().any(|b| b.reason == StopReason::Depth));
    let silence = fit(&[0.0; 37], 0.0)
        .prepare_spectrum(0..37, 1, false)
        .unwrap();
    let quiet = AdaptiveSpectrum::search(
        &silence,
        None,
        FrequencyParams {
            absolute_floor: 1e-9,
            ..params
        },
    )
    .unwrap();
    assert!(quiet.peaks.is_empty());
    assert!(quiet
        .bands
        .iter()
        .all(|b| b.reason == StopReason::BelowFloor));
    assert_eq!(quiet.stats.fine_queries, 129);
}
#[test]
fn certified_pruned_bands_bound_unsampled_magnitudes_and_invalid_inputs() {
    let f = fit(
        &(0..64).map(|i| (i as f64 * 0.63).cos()).collect::<Vec<_>>(),
        0.001,
    );
    let s = f.prepare_spectrum(0..64, 2, true).unwrap();
    let params = FrequencyParams {
        relative_floor: 0.1,
        prune_flat: false,
        max_evaluations: 30000,
        ..FrequencyParams::default()
    };
    let result = AdaptiveSpectrum::search(&s, None, params).unwrap();
    for b in result
        .bands
        .iter()
        .filter(|b| b.reason == StopReason::BelowFloor)
    {
        for t in [0.13, 0.31, 0.72] {
            let z = s.value(b.lower + (b.upper - b.lower) * t).unwrap();
            assert!(z.re.hypot(z.im) <= b.magnitude_upper + 1e-10);
        }
    }
    assert!(AdaptiveSpectrum::search(
        &s,
        None,
        FrequencyParams {
            max_depth: 33,
            ..params
        }
    )
    .is_err());
    assert!(AdaptiveSpectrum::search(
        &s,
        None,
        FrequencyParams {
            lower: -0.1,
            ..params
        }
    )
    .is_err());
    assert!(s
        .difference_bound(&f.prepare_spectrum(0..64, 1, true).unwrap())
        .is_err());
    assert!(s.value(f64::NAN).is_err());
    assert!(f.prepare_spectrum(3..3, 1, true).is_err());
    assert!(GeometryFrame::hierarchical(
        &[f64::NAN],
        GeometryParams::default(),
        HierarchyParams::default()
    )
    .is_err());
    assert!(GeometryFrame::hierarchical(
        &[1.0],
        GeometryParams::default(),
        HierarchyParams {
            max_depth: 33,
            ..HierarchyParams::default()
        }
    )
    .is_err());
}
