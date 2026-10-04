use sinwhisky::faft::{
    CircleCurve, Faft, FramePath, PeakParams, PeakTracker, PursuitParams, PursuitStop, Rotation,
    TrackerParams, Weighting,
};
use sinwhisky::geometric_faft::{GeometryFrame, GeometryParams, HierarchyParams};
use sinwhisky::spectral::Complex;
use sinwhisky::Error;
use std::f64::consts::{PI, TAU};

fn magnitude(a: Complex) -> f64 {
    a.re.hypot(a.im)
}

/// A curve that is exactly a sum of sinusoids: one stacked rotation per tone on
/// the whole support. `tones` are (amplitude, omega, phase).
fn tones(n: usize, tones: &[(f64, f64, f64)]) -> CircleCurve {
    CircleCurve::new(
        tones
            .iter()
            .map(|&(radius, rate, phase)| Rotation {
                start: 0,
                length: n,
                center: 0.0,
                radius,
                phase,
                rate,
                slope: 0.0,
            })
            .collect(),
    )
    .unwrap()
}

fn sum_of_tones(t: f64, tones: &[(f64, f64, f64)]) -> f64 {
    tones.iter().map(|&(r, w, p)| r * (p + w * t).sin()).sum()
}

fn hierarchical(x: &[f64], error: f64) -> GeometryFrame {
    GeometryFrame::hierarchical(
        x,
        GeometryParams {
            max_error: error,
            max_span: 64,
            ..GeometryParams::default()
        },
        HierarchyParams::default(),
    )
    .unwrap()
    .0
}

/// Dense midpoint rule, an independent check of closed forms (error O(h^2)).
fn midpoint(n: usize, steps: usize, weighting: Weighting, f: impl Fn(f64) -> f64) -> f64 {
    let h = n as f64 / steps as f64;
    (0..steps)
        .map(|k| {
            let t = (k as f64 + 0.5) * h;
            let w = match weighting {
                Weighting::Rectangular => 1.0,
                Weighting::Hann => 0.5 - 0.5 * (TAU * t / n as f64).cos(),
            };
            f(t) * w
        })
        .sum::<f64>()
        * h
}

#[test]
fn kernel_is_the_exact_transform_of_a_tone() {
    let n = 300;
    let (radius, rate, phase0) = (1.7, 0.4321, 0.9);
    for weighting in [Weighting::Rectangular, Weighting::Hann] {
        let faft = tones(n, &[(radius, rate, phase0)])
            .transform(0..n, weighting)
            .unwrap();
        assert!((faft.kernel(0.0).unwrap() - 1.0).abs() < 1e-15);
        // y = r sin(phi + w t) = alpha cos(w(t-tc)) + beta sin(w(t-tc)), tc = n/2.
        let theta = phase0 + rate * n as f64 * 0.5;
        let (alpha, beta) = (radius * theta.sin(), radius * theta.cos());
        let c = Complex {
            re: 0.5 * alpha,
            im: -0.5 * beta,
        };
        for omega in [0.0, 0.1, 0.43, 0.4321, 0.45, 1.3, 4.0] {
            let below = faft.kernel(omega - rate).unwrap();
            let above = faft.kernel(omega + rate).unwrap();
            let model = Complex {
                re: c.re * (below + above),
                im: c.im * (below - above),
            };
            let centred = faft.derivatives(omega).unwrap()[0];
            let error = (centred.re - model.re).hypot(centred.im - model.im);
            assert!(error < 1e-12, "{weighting:?} {omega}: {error}");
        }
        let fit = faft.fit(&[rate]).unwrap();
        assert!(fit.residual_energy < 1e-10 * fit.energy, "{fit:?}");
        assert!((fit.components[0].magnitude() - 0.5 * radius).abs() < 1e-10);
    }
}

#[test]
fn kernel_ranges_match_the_window_tables() {
    let n = 1024;
    let curve = tones(n, &[(1.0, 0.3, 0.0)]);
    let bins = |faft: &Faft, db: f64| faft.kernel_range(db).unwrap() / faft.resolution();
    let rect = curve.transform(0..n, Weighting::Rectangular).unwrap();
    let hann = curve.transform(0..n, Weighting::Hann).unwrap();
    // Half-widths in units of 2*pi/T (classical window figures [Harris 1978]).
    assert!(
        (bins(&rect, 3.0) - 0.4422).abs() < 1e-3,
        "{}",
        bins(&rect, 3.0)
    );
    assert!((bins(&rect, 6.0) - 0.6025).abs() < 1e-3);
    assert!(
        (bins(&hann, 3.0) - 0.7191).abs() < 1e-3,
        "{}",
        bins(&hann, 3.0)
    );
    assert!((bins(&hann, 6.0) - 0.9984).abs() < 1e-3);
    // First nulls end the main lobes.
    assert!(rect.kernel(rect.resolution()).unwrap().abs() < 1e-12);
    assert!(hann.kernel(2.0 * hann.resolution()).unwrap().abs() < 1e-12);
    // Highest sidelobes: -13.26 dB (rectangular) and -31.47 dB (Hann).
    let sidelobe = |faft: &Faft, from: f64| {
        (0..4000)
            .map(|k| {
                faft.kernel((from + k as f64 * 1e-3) * faft.resolution())
                    .unwrap()
                    .abs()
            })
            .fold(0.0, f64::max)
    };
    assert!((20.0 * sidelobe(&rect, 1.0).log10() + 13.26).abs() < 0.01);
    assert!((20.0 * sidelobe(&hann, 2.0).log10() + 31.47).abs() < 0.01);
    assert!(rect.kernel_range(0.0).is_err() && rect.kernel(f64::NAN).is_err());
}

#[test]
fn energy_matches_independent_integration() {
    let n = 256;
    let spec = [(1.3, 0.21, 0.4), (0.4, 1.9, -1.0)];
    let curve = tones(n, &spec);
    for weighting in [Weighting::Rectangular, Weighting::Hann] {
        let faft = curve.transform(0..n, weighting).unwrap();
        let exact = midpoint(n, 400_000, weighting, |t| sum_of_tones(t, &spec).powi(2));
        let energy = faft.energy().unwrap();
        assert!((energy - exact).abs() < 1e-8 * exact, "{energy} {exact}");
    }
    // A fitted curve with lines and many short arcs, against the curve itself.
    let x: Vec<f64> = (0..200)
        .map(|i| 500.0 * (0.37 * i as f64).sin() + 120.0 * (1.7 * i as f64).cos())
        .collect();
    let curve = CircleCurve::from_geometry(&hierarchical(&x, 0.5)).unwrap();
    let faft = curve.transform(10..190, Weighting::Hann).unwrap();
    let exact = midpoint(180, 360_000, Weighting::Hann, |t| {
        curve.evaluate(10.0 + t).unwrap().powi(2)
    });
    let energy = faft.energy().unwrap();
    assert!((energy - exact).abs() < 1e-7 * exact, "{energy} {exact}");
}

#[test]
fn joint_fit_removes_neighbour_leakage_and_certifies_the_rest() {
    let n = 512;
    let res = TAU / n as f64;
    // Equal tones 1.5 bins apart: each peak height is biased by the other.
    let spec = [
        (1.0, 0.6, 0.2),
        (1.0, 0.6 + 1.5 * res, 1.4),
        (0.05, 2.2, 0.0),
    ];
    let faft = tones(n, &spec).transform(0..n, Weighting::Hann).unwrap();
    let fit = faft.fit(&[spec[0].1, spec[1].1]).unwrap();
    for c in &fit.components {
        assert!((c.magnitude() - 0.5).abs() < 1e-3, "{c:?}");
        assert!(c.leakage_db().abs() > 0.1, "{}", c.leakage_db());
    }
    // The third tone is not modelled: its energy is the residual (~ r^2 W/2).
    let w = n as f64 * 0.5;
    assert!((fit.residual_energy / (0.05f64.powi(2) * w * 0.5) - 1.0).abs() < 0.05);
    // Certificate: no residual value exceeds the bound at any frequency.
    let comps = &fit.components;
    for k in 0..4000 {
        let omega = 4.0 * k as f64 / 4000.0;
        let centred = faft.derivatives(omega).unwrap()[0];
        let mut residual = centred;
        for c in comps {
            let start = Complex {
                re: c.value.re,
                im: c.value.im,
            };
            // Back to the centred origin.
            let (s, cs) = (c.omega * n as f64 * 0.5).sin_cos();
            let v = Complex {
                re: start.re * cs - start.im * s,
                im: start.re * s + start.im * cs,
            };
            let below = faft.kernel(omega - c.omega).unwrap();
            let above = faft.kernel(omega + c.omega).unwrap();
            residual.re -= v.re * (below + above);
            residual.im -= v.im * (below - above);
        }
        assert!(magnitude(residual) <= fit.bound * (1.0 + 1e-9), "{omega}");
    }
    // Adding a frequency never increases the residual energy (nested least squares).
    let more = faft.fit(&[spec[0].1, spec[1].1, 2.2]).unwrap();
    assert!(more.residual_energy <= fit.residual_energy);
    assert!(more.residual_energy < 1e-9 * more.energy && more.explained() > 1.0 - 1e-9);
    assert!(faft.fit(&[0.1, 0.1]).is_err() && faft.fit(&[-0.1]).is_err());
    // DC has no sine atom: it is dropped, not an error.
    assert_eq!(faft.fit(&[0.0, 0.6]).unwrap().degenerate, 1);
}

#[test]
fn pursuit_explains_rectangular_sidelobes_by_their_tone() {
    let n = 512;
    let spec = [(1.0, 0.7311, 0.3)];
    let faft = tones(n, &spec)
        .transform(0..n, Weighting::Rectangular)
        .unwrap();
    let scan = faft.peaks(PeakParams::default()).unwrap();
    // Every -13 dB ... -40 dB sidelobe is a local maximum above the 1% floor.
    assert!(scan.peaks.len() > 10, "{}", scan.peaks.len());
    let exact = PursuitParams {
        polish_tolerance: 1e-9,
        ..PursuitParams::default()
    };
    let pursuit = faft.pursue(exact).unwrap();
    assert_eq!(pursuit.components.len(), 1, "{pursuit:?}");
    assert!((pursuit.components[0].omega - 0.7311).abs() < 1e-12);
    // Sidelobes were explained by the provisional tone before any Newton work.
    assert!(pursuit.stats.search.excluded >= 10 && pursuit.stats.search.refined <= 4);
    assert!((pursuit.components[0].magnitude() - 0.5).abs() < 1e-9);
    assert_eq!(pursuit.stop, PursuitStop::Certified);
    assert!(pursuit.bound < 1e-6 && pursuit.explained() > 1.0 - 1e-9);
}

#[test]
fn pursuit_finds_a_weak_tone_hidden_on_a_strong_flank() {
    let n = 1024;
    let res = TAU / n as f64;
    let (strong, weak) = (0.5, 0.5 + 2.6 * res);
    // -30 dB at 2.6 bins: on the strong tone's -31.5 dB Hann sidelobe.
    let spec = [(1.0, strong, 0.0), (0.0316, weak, 2.0), (0.3, 1.7, 0.5)];
    let faft = tones(n, &spec).transform(0..n, Weighting::Hann).unwrap();
    let params = PeakParams {
        relative_floor: 0.005,
        ..PeakParams::default()
    };
    let scan = faft.peaks(params).unwrap();
    // The scan reports both sidelobes of each strong tone as peaks, and the weak
    // tone only as a maximum biased in frequency and by more than 3 dB in height.
    assert_eq!(scan.peaks.len(), 7, "{:?}", scan.peaks);
    let biased = scan
        .peaks
        .iter()
        .find(|p| (p.omega - weak).abs() < 0.5 * res)
        .unwrap();
    assert!((biased.omega - weak).abs() > 0.1 * res);
    assert!(20.0 * (biased.magnitude() / (0.5 * 0.0316)).log10() > 3.0);
    let pursuit = faft
        .pursue(PursuitParams {
            search: PeakParams {
                max_peaks: 64,
                ..params
            },
            polish_tolerance: 1e-9,
            ..PursuitParams::default()
        })
        .unwrap();
    assert_eq!(pursuit.components.len(), 3, "{pursuit:?}");
    for (c, &(r, w, _)) in pursuit.components.iter().zip(&spec_sorted(&spec)) {
        assert!((c.omega - w).abs() < 1e-9, "{c:?}");
        assert!((c.magnitude() / (0.5 * r) - 1.0).abs() < 1e-9, "{c:?}");
    }
    assert!(pursuit.explained() > 1.0 - 1e-9);
}

fn spec_sorted(spec: &[(f64, f64, f64)]) -> Vec<(f64, f64, f64)> {
    let mut s = spec.to_vec();
    s.sort_by(|a, b| a.1.total_cmp(&b.1));
    s
}

#[test]
fn pursuit_on_a_fitted_curve_reports_measured_completeness() {
    let n = 1024;
    let spec = [
        (800.0, 0.3137, 0.4),
        (300.0, 1.1291, 0.0),
        (40.0, 0.36, 1.0),
    ];
    let x: Vec<f64> = (0..n).map(|i| sum_of_tones(i as f64, &spec)).collect();
    let curve = CircleCurve::from_geometry(&hierarchical(&x, 0.5)).unwrap();
    let faft = curve.transform(0..n, Weighting::Hann).unwrap();
    let pursuit = faft.pursue(PursuitParams::default()).unwrap();
    for &(r, w, _) in &spec {
        let c = pursuit
            .components
            .iter()
            .find(|c| (c.omega - w).abs() < 1e-3)
            .unwrap_or_else(|| panic!("{w} missing: {pursuit:?}"));
        // The AFT is the transform of the fitted curve: at 5.6 samples per period
        // its circles carry less of the tone than the samples (the rest is in the
        // model's images), so only the slow tones are compared with the source.
        if w < 1.0 {
            assert!((c.magnitude() / (0.5 * r) - 1.0).abs() < 0.01, "{c:?}");
        } else {
            assert!(c.leakage_db().abs() < 0.01, "{c:?}");
        }
    }
    // The weak tone 7.5 bins from the strong one is observed ~0.09 dB high; the
    // joint fit removes that leakage.
    let weak = pursuit
        .components
        .iter()
        .find(|c| (c.omega - 0.36).abs() < 1e-3)
        .unwrap();
    assert!(weak.leakage_db() > 0.05, "{weak:?}");
    // Every reported component is above the floor; the residual is bounded.
    assert!(pursuit
        .components
        .iter()
        .all(|c| c.magnitude() >= pursuit.floor));
    assert!(pursuit.explained() > 0.999 && pursuit.bound.is_finite());
    assert!(pursuit.stats.fits > 0 && pursuit.stats.rounds >= 1);
}

#[test]
fn peak_range_measures_isolated_and_shared_peaks() {
    let n = 1024;
    let res = TAU / n as f64;
    let isolated = tones(n, &[(1.0, 1.2345, 0.0)])
        .transform(0..n, Weighting::Hann)
        .unwrap();
    let range = isolated.peak_range(1.2345, 6.0).unwrap();
    let ideal = isolated.kernel_range(6.0).unwrap();
    assert!(!range.lower_valley && !range.upper_valley);
    assert!(
        (range.upper - 1.2345 - ideal).abs() < 1e-3 * res,
        "{range:?}"
    );
    assert!((1.2345 - range.lower - ideal).abs() < 1e-3 * res);
    // Two equal tones three bins apart, in phase at the window centre, share
    // their -20 dB range: the valley between them stays near -9 dB.
    let pair = tones(n, &[(1.0, 1.0, 0.0), (1.0, 1.0 + 3.0 * res, PI)])
        .transform(0..n, Weighting::Hann)
        .unwrap();
    let peak = pair
        .peaks(PeakParams::default())
        .unwrap()
        .peaks
        .into_iter()
        .max_by(|a, b| a.magnitude().total_cmp(&b.magnitude()))
        .unwrap();
    let range = pair.peak_range(peak.omega, 20.0).unwrap();
    assert!(range.lower_valley != range.upper_valley, "{range:?}");
    assert!(range.width() > 0.0 && range.level < range.magnitude);
    assert!(isolated.peak_range(1.2345, -1.0).is_err());
}

#[test]
fn tracker_keeps_identities_and_skips_scans() {
    let rate = 8000.0;
    let n = 512;
    let hop = 256;
    let frames = 24;
    let total = n + hop * (frames - 1);
    // A gliding tone, a steady tone, and a tone that starts at frame 12.
    let glide =
        |t: f64| TAU * (440.0 * t / rate + 0.5 * 30.0 * (t / rate).powi(2) / (total as f64 / rate));
    let onset = 12 * hop + n / 2;
    let x: Vec<f64> = (0..total)
        .map(|i| {
            let t = i as f64;
            1000.0 * glide(t).sin()
                + 500.0 * (TAU * 1250.0 * t / rate).sin()
                + if i >= onset {
                    300.0 * (TAU * 2100.0 * t / rate).sin()
                } else {
                    0.0
                }
        })
        .collect();
    let curve = CircleCurve::from_geometry(&hierarchical(&x, 0.5)).unwrap();
    let mut tracker = PeakTracker::new(TrackerParams {
        pursuit: PursuitParams {
            search: PeakParams {
                relative_floor: 0.05,
                max_peaks: 64,
                ..PeakParams::default()
            },
            ..PursuitParams::default()
        },
        birth_floor: 0.1,
        ..TrackerParams::default()
    })
    .unwrap();
    let mut scans = 0;
    let mut steady_id = None;
    let mut glide_id = None;
    let mut born_late = None;
    for f in 0..frames {
        let window = f * hop..f * hop + n;
        let faft = curve.transform(window, Weighting::Hann).unwrap();
        let frame = tracker.update(&faft).unwrap();
        if !matches!(frame.path, FramePath::Tracked { .. }) {
            scans += 1;
        }
        let near = |hz: f64| {
            frame
                .peaks
                .iter()
                .find(|p| (p.omega - TAU * hz / rate).abs() < 2.0 * faft.resolution())
                .copied()
        };
        let steady = near(1250.0).expect("steady tone tracked");
        assert_eq!(*steady_id.get_or_insert(steady.id), steady.id);
        let centre = (f * hop + n / 2) as f64;
        let glide_hz = 440.0 + 30.0 * centre / total as f64;
        let g = near(glide_hz).expect("glide tracked");
        assert_eq!(*glide_id.get_or_insert(g.id), g.id);
        assert!((g.omega - TAU * glide_hz / rate).abs() < 0.05 * faft.resolution());
        if f * hop >= onset {
            // Fully inside the window: the new tone has its own identity.
            let late = near(2100.0).expect("onset tracked");
            assert_eq!(*born_late.get_or_insert(late.id), late.id);
            assert!(late.id != steady.id && late.id != g.id);
        }
    }
    assert!(born_late.is_some());
    assert!(scans < frames / 2, "{scans} scans in {frames} frames");
    assert!(PeakTracker::new(TrackerParams {
        birth_floor: 0.001,
        ..TrackerParams::default()
    })
    .is_err());
    assert!(PeakTracker::new(TrackerParams {
        rescan_every: 0,
        ..TrackerParams::default()
    })
    .is_err());
}

#[test]
fn pursuit_validation_and_silence() {
    let n = 64;
    let silence = tones(n, &[(0.0, 0.5, 0.0)])
        .transform(0..n, Weighting::Hann)
        .unwrap();
    let pursuit = silence.pursue(PursuitParams::default()).unwrap();
    assert!(pursuit.components.is_empty() && pursuit.stop == PursuitStop::Silent);
    let invalid = [
        PursuitParams {
            max_rounds: 0,
            ..PursuitParams::default()
        },
        PursuitParams {
            search: PeakParams {
                scan_density: None,
                circle_seeds: true,
                ..PeakParams::default()
            },
            ..PursuitParams::default()
        },
        PursuitParams {
            search: PeakParams::default(), // max_peaks 4096 > 1024
            ..PursuitParams::default()
        },
    ];
    for p in invalid {
        assert_eq!(
            silence.pursue(p).unwrap_err(),
            Error::InvalidParameter("FAFT pursuit parameters")
        );
    }
}

#[test]
fn joint_polish_separates_tones_inside_one_main_lobe() {
    // Two tones 1.0 and 1.5 resolutions apart, at several relative phases. The
    // Hann transform has one maximum for the pair (no local maximum per tone),
    // but the tone model fitted by joint Gauss-Newton recovers both exactly.
    let n = 1024;
    let res = TAU / n as f64;
    for separation in [1.0, 1.5] {
        for phase in [0.0, 1.0, 2.0, 3.0] {
            let spec = [
                (1000.0, 0.2, 0.0),
                (300.0, 0.34194, 0.0),
                (100.0, 0.34194 + separation * res, phase),
            ];
            let faft = tones(n, &spec).transform(0..n, Weighting::Hann).unwrap();
            let pursuit = faft
                .pursue(PursuitParams {
                    search: PeakParams {
                        max_peaks: 64,
                        ..PeakParams::default()
                    },
                    polish_tolerance: 1e-9,
                    ..PursuitParams::default()
                })
                .unwrap();
            assert_eq!(
                pursuit.components.len(),
                3,
                "{separation} {phase}: {pursuit:?}"
            );
            for (c, &(r, w, _)) in pursuit.components.iter().zip(&spec_sorted(&spec)) {
                assert!((c.omega - w).abs() < 1e-6 * res, "{c:?}");
                assert!((c.magnitude() / (0.5 * r) - 1.0).abs() < 1e-6, "{c:?}");
                // Shared lobes: the pair's unique energies are below their heights.
                if w > 0.3 {
                    assert!(c.significance < c.magnitude(), "{c:?}");
                }
            }
            assert_eq!(pursuit.stop, PursuitStop::Certified);
        }
    }
}
