//! Band division, exact analog filtering of circle curves and the two-way banded
//! fit (PAPER_EQ_COMPRESSION.md, Sections 2 and 3).
use sinwhisky::analog::{AnalogFilter, SpiralCurve};
use sinwhisky::bands::{divide, equalize, extrude, BandSplit, Crossover, ToneAnalysis, ToneParams};
use sinwhisky::faft::{CircleCurve, Rotation, Weighting};
use sinwhisky::spectral::Complex;
use std::f64::consts::{PI, TAU};

fn tones(n: usize, parts: &[(f64, f64, f64)]) -> Vec<f64> {
    (0..n)
        .map(|i| {
            parts
                .iter()
                .map(|&(w, a, p)| a * (w * i as f64 + p).sin())
                .sum()
        })
        .collect()
}

/// One exact sinusoid as a single rotation on [0, n).
fn sinusoid_curve(n: usize, omega: f64, amplitude: f64, phase: f64) -> CircleCurve {
    CircleCurve::new(vec![Rotation {
        start: 0,
        length: n,
        center: 0.0,
        radius: amplitude,
        phase,
        rate: omega,
        slope: 0.0,
    }])
    .unwrap()
}

#[test]
fn banded_fit_with_full_reach_equals_the_dense_fit() {
    let n = 1024;
    let x = tones(
        n,
        &[
            (0.05, 900.0, 0.3),
            (0.062, 300.0, 1.1),
            (0.4, 50.0, -0.4),
            (0.41, 40.0, 2.0),
        ],
    );
    let curve = CircleCurve::new(
        x.windows(2)
            .enumerate()
            .map(|(i, w)| Rotation {
                start: i,
                length: 1,
                center: w[0],
                radius: 0.0,
                phase: 0.0,
                rate: 0.0,
                slope: w[1] - w[0],
            })
            .collect(),
    )
    .unwrap();
    let faft = curve.transform(0..n - 1, Weighting::Hann).unwrap();
    let omegas = [0.05, 0.062, 0.4, 0.41, 0.9];
    let dense = faft.fit(&omegas).unwrap();
    let banded = faft.fit_banded(&omegas, f64::INFINITY).unwrap();
    let tight = faft.fit_banded(&omegas, 0.1).unwrap();
    for ((d, b), t) in dense
        .components
        .iter()
        .zip(&banded.components)
        .zip(&tight.components)
    {
        assert_eq!(d.omega, b.omega);
        let scale = d.value.re.hypot(d.value.im).max(1.0);
        assert!((d.value.re - b.value.re).abs() < 1e-9 * scale);
        assert!((d.value.im - b.value.im).abs() < 1e-9 * scale);
        assert!((d.significance - b.significance).abs() < 1e-7 * scale);
        // Components 0.34 rad (55 bins) apart barely couple: truncating changes little.
        assert!((d.value.re - t.value.re).abs() < 1e-3 * scale);
    }
    assert!((dense.residual_energy - banded.residual_energy).abs() < 1e-9 * dense.energy);
    // The truncated fit evaluates its residual from the full quadratic form, so it
    // cannot be below the exact least-squares residual.
    assert!(tight.residual_energy >= dense.residual_energy - 1e-9 * dense.energy);
    assert!(faft.fit_banded(&[0.1, 0.1], 1.0).is_err());
    assert!(faft.fit_banded(&[0.1], 0.0).is_err());
}

#[test]
fn butterworth_response_and_validation() {
    for order in 1..=8 {
        let f = AnalogFilter::butterworth_lowpass(order, 0.2).unwrap();
        for &w in &[0.0, 0.1, 0.2, 0.37, 1.5] {
            let h = f.response(w);
            let expected = 1.0 / (1.0 + (w / 0.2f64).powi(2 * order as i32));
            assert!(
                (h.re * h.re + h.im * h.im - expected).abs() < 1e-12,
                "order {order} w {w}"
            );
        }
    }
    let p = |re, im| Complex { re, im };
    assert!(AnalogFilter::new(0.0, vec![p(0.1, 0.0)], vec![p(1.0, 0.0)]).is_err());
    assert!(AnalogFilter::new(0.0, vec![p(-0.1, 0.3)], vec![p(1.0, 0.0)]).is_err());
    assert!(AnalogFilter::butterworth_lowpass(0, 0.1).is_err());
}

/// Runge-Kutta reference for the states of a filter driven by a continuous input.
fn reference_output(filter: &AnalogFilter, input: impl Fn(f64) -> f64, n: usize) -> Vec<f64> {
    let poles: Vec<(f64, f64)> = filter.poles().iter().map(|p| (p.re, p.im)).collect();
    let mut state = vec![(0.0, 0.0); poles.len()];
    let steps = 400;
    let h = 1.0 / steps as f64;
    let mut out = Vec::with_capacity(n);
    let output = |state: &[(f64, f64)], t: f64| {
        let mut v = filter.direct() * input(t);
        for (s, r) in state.iter().zip(filter.residues()) {
            v += r.re * s.0 - r.im * s.1;
        }
        v
    };
    for i in 0..n {
        out.push(output(&state, i as f64));
        for k in 0..steps {
            let t = i as f64 + k as f64 * h;
            // Stay on the left limit of [i, i+1): the pieces join at integers.
            let input = |t: f64| input(t.min(i as f64 + 1.0 - 1e-9));
            for (s, &(pr, pi)) in state.iter_mut().zip(&poles) {
                let f = |t: f64, (a, b): (f64, f64)| (pr * a - pi * b + input(t), pr * b + pi * a);
                let k1 = f(t, *s);
                let k2 = f(t + h / 2.0, (s.0 + h / 2.0 * k1.0, s.1 + h / 2.0 * k1.1));
                let k3 = f(t + h / 2.0, (s.0 + h / 2.0 * k2.0, s.1 + h / 2.0 * k2.1));
                let k4 = f(t + h, (s.0 + h * k3.0, s.1 + h * k3.1));
                s.0 += h / 6.0 * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0);
                s.1 += h / 6.0 * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1);
            }
        }
    }
    out
}

#[test]
fn filtering_a_circle_curve_is_exact() {
    // Two arcs, a stacked pair and a line: every kind of piece.
    let rot = |start, length, center, radius, phase, rate, slope| Rotation {
        start,
        length,
        center,
        radius,
        phase,
        rate,
        slope,
    };
    let curve = CircleCurve::new(vec![
        rot(0, 7, 0.2, 1.3, 0.4, -0.7, 0.0),
        rot(7, 5, 0.0, 0.0, 0.0, 0.0, 0.15),
        rot(12, 9, 0.5, 0.6, -1.0, 0.45, 0.0),
        rot(12, 9, -0.2, 0.3, 0.2, 1.2, 0.0),
        rot(21, 19, -0.1, 2.0, 2.5, 0.2, 0.01),
    ])
    .unwrap();
    let filter = AnalogFilter::butterworth_lowpass(3, 0.6).unwrap();
    let exact = SpiralCurve::from_circles(&curve, 4)
        .unwrap()
        .filter(&filter)
        .unwrap()
        .samples()
        .unwrap();
    let reference = reference_output(&filter, |t| curve.evaluate(t.min(39.999_999)).unwrap(), 40);
    for (i, (a, b)) in exact.iter().zip(&reference).enumerate() {
        assert!((a - b).abs() < 1e-8, "sample {i}: {a} vs {b}");
    }
    // Cropping pieces does not change the curve or its filtered output.
    let coarse = SpiralCurve::from_circles(&curve, 4096).unwrap();
    let fine = SpiralCurve::from_circles(&curve, 3).unwrap();
    for i in 0..40 {
        let t = i as f64 + 0.37;
        assert!((coarse.evaluate(t).unwrap() - curve.evaluate(t).unwrap()).abs() < 1e-12);
        assert!((fine.evaluate(t).unwrap() - curve.evaluate(t).unwrap()).abs() < 1e-12);
    }
}

#[test]
fn steady_state_and_zero_phase_responses() {
    let (n, omega) = (4000, 0.05);
    let curve = SpiralCurve::from_circles(&sinusoid_curve(n, omega, 1.0, 0.3), 32).unwrap();
    let filter = AnalogFilter::butterworth_lowpass(4, 0.06).unwrap();
    let h = filter.response(omega);
    let (gain, shift) = (h.re.hypot(h.im), h.im.atan2(h.re));
    let forward = curve.filter(&filter).unwrap().samples().unwrap();
    let zero = curve.filter_zero_phase(&filter).unwrap().samples().unwrap();
    for i in 1500..2500 {
        let t = i as f64;
        assert!((forward[i] - gain * (omega * t + 0.3 + shift).sin()).abs() < 1e-9);
        assert!((zero[i] - gain * gain * (omega * t + 0.3).sin()).abs() < 1e-9);
    }
    // Reversal is exact and an involution; a line curve returns its samples.
    let x = tones(64, &[(0.3, 2.0, 0.1)]);
    let lines = SpiralCurve::from_samples(&x).unwrap();
    assert_eq!(lines.samples().unwrap(), x);
    let back = lines
        .reverse()
        .unwrap()
        .reverse()
        .unwrap()
        .samples()
        .unwrap();
    assert!(back.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-12));
}

fn analysis(x: &[f64]) -> ToneAnalysis {
    ToneAnalysis::new(
        x,
        ToneParams {
            window: 1024,
            max_error: 0.5,
            ..ToneParams::default()
        },
    )
    .unwrap()
}

#[test]
fn tones_are_fitted_and_resynthesized() {
    let n = 6000;
    let x = tones(n, &[(0.09, 1000.0, 0.2), (0.31, 300.0, 1.0)]);
    let a = analysis(&x);
    assert_eq!(a.hop(), 512);
    let middle = &a.frames()[5];
    let strongest = middle
        .tones
        .iter()
        .max_by(|p, q| p.amplitude.total_cmp(&q.amplitude))
        .unwrap();
    assert!((strongest.omega - 0.09).abs() < 1e-5 && (strongest.amplitude - 1000.0).abs() < 1.0);
    // A stationary tone overlap-adds back to itself.
    let tonal = a.synthesize(|_| 1.0).unwrap();
    // Away from the abrupt start and end, which windowed tones cannot represent.
    let worst = x[1024..n - 1024]
        .iter()
        .zip(&tonal[1024..n - 1024])
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max);
    assert!(worst < 2.0, "tonal synthesis error {worst}");
    assert!(middle.residual_energy < 1e-6 * middle.energy && a.explained() > 0.9);
}

#[test]
fn division_is_complete_and_assigns_tones_whole() {
    let n = 8192;
    let (low, high) = (0.10, 0.13); // 16.3 and 21.2 bins of a 1024 window
    let edge = 0.115; // 2.4 bins from each tone
    let x = tones(n, &[(low, 1000.0, 0.0), (high, 500.0, 0.7)]);
    let a = analysis(&x);
    for crossover in [
        Crossover::BrickWall,
        Crossover::RaisedCosine { half_width: 0.005 },
    ] {
        let split = BandSplit::new(vec![edge], crossover, 4).unwrap();
        for w in [0.0, 0.112, 0.115, 0.118, 3.0] {
            assert!(((0..2).map(|b| split.weight(b, w)).sum::<f64>() - 1.0).abs() < 1e-15);
        }
        let d = divide(&x, &a, &split).unwrap();
        let sum: Vec<f64> = (0..n).map(|i| d.band(0)[i] + d.band(1)[i]).collect();
        assert!(sum.iter().zip(&x).all(|(s, v)| (s - v).abs() < 1e-9));
        // Each band holds its own tone; measure away from the track ends.
        let lower = d.band(0);
        let upper = d.band(1);
        let reference_low = tones(n, &[(low, 1000.0, 0.0)]);
        let reference_high = tones(n, &[(high, 500.0, 0.7)]);
        for i in 1024..n - 1024 {
            assert!((lower[i] - reference_low[i]).abs() < 3.0, "low band at {i}");
            assert!(
                (upper[i] - reference_high[i]).abs() < 3.0,
                "high band at {i}"
            );
        }
    }
    // Unit gains return the source; a muted band removes its tone.
    let split = BandSplit::new(vec![edge], Crossover::BrickWall, 4).unwrap();
    let d = divide(&x, &a, &split).unwrap();
    let same = equalize(&d, &[1.0, 1.0]).unwrap();
    assert!(same.iter().zip(&x).all(|(s, v)| (s - v).abs() < 1e-9));
    let (isolated, remainder) = extrude(&x, &a, 0.0, edge, Crossover::BrickWall, 4).unwrap();
    assert!(isolated
        .iter()
        .zip(&remainder)
        .zip(&x)
        .all(|((i, r), v)| (i + r - v).abs() < 1e-9));
    assert!(BandSplit::new(vec![0.2, 0.1], Crossover::BrickWall, 4).is_err());
    assert!(BandSplit::new(vec![0.2], Crossover::BrickWall, 0).is_err());
    assert!(extrude(&x, &a, 0.3, 0.2, Crossover::BrickWall, 4).is_err());
    let _ = (PI, TAU);
}

#[test]
fn near_dc_content_is_fitted_as_dc_not_as_an_ill_conditioned_tone() {
    // A DC offset with a slow drift: peaks within a hair of omega = 0 would give
    // the sine atom a vanishing norm and a huge amplitude.
    let n = 8192;
    let x: Vec<f64> = (0..n)
        .map(|i| 300.0 + 40.0 * (i as f64 / n as f64) + 800.0 * (0.2 * i as f64).sin())
        .collect();
    let a = analysis(&x);
    let resolution = TAU / 1024.0;
    for frame in a.frames() {
        for tone in &frame.tones {
            assert!(tone.amplitude < 2000.0, "{tone:?}");
            assert!(
                tone.omega == 0.0 || tone.omega >= 0.5 * resolution,
                "{tone:?}"
            );
        }
    }
    let _ = PI;
}
