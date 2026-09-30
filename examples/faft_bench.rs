//! Private JSON Lines comparing the circle-arc FAFT with earlier kernels on the SAME
//! fitted geometry. Usage: faft_bench N ERROR SPAN REPEATS QUERIES [DENSE] < frames
//!
//! Per frame: hierarchical circle fit; FAFT preparation, arbitrary-frequency
//! queries (continuous Hann), peaks by duration scan + Newton and by circle
//! references; the earlier sampled (zoom) DTFT kernels and adaptive search on the
//! same geometry; the integral AFT reference against FAFT on the AFT's own curve.
//! DENSE>0 adds |FAFT| and zoom-1/zoom-4 sampled magnitudes on DENSE uniform
//! frequencies in [0, 3*pi), exposing the images that zoom sampling creates.
//! Rotations, query values and peaks are invertible model data: keep them private.
use sinwhisky::adaptive_faft::{AdaptiveSpectrum, FrequencyParams};
use sinwhisky::aft::AngularFrame;
use sinwhisky::faft::{CircleCurve, Faft, PeakParams, PeakSearch, Weighting};
use sinwhisky::geometric_faft::{GeometryFrame, GeometryParams, HierarchyParams};
use sinwhisky::spectral::Complex;
use sinwhisky::AnalysisParams;
use std::f64::consts::{PI, TAU};
use std::hint::black_box;
use std::io::{self, Read, Write};
use std::time::Instant;

fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}
fn timed<T>(repeats: usize, mut f: impl FnMut() -> T) -> (T, Vec<u128>) {
    let mut out = f();
    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let begin = Instant::now();
        out = black_box(f());
        times.push(begin.elapsed().as_nanos());
    }
    (out, times)
}
fn norm(z: Complex) -> f64 {
    z.re.hypot(z.im)
}
fn relative_l2(a: &[Complex], reference: &[Complex]) -> f64 {
    let mut error = 0.0_f64;
    let mut total = 0.0_f64;
    for (x, y) in a.iter().zip(reference) {
        error = error.hypot((x.re - y.re).hypot(x.im - y.im));
        total = total.hypot(norm(*y));
    }
    error / total.max(f64::MIN_POSITIVE)
}
fn peaks_json(search: &PeakSearch) -> String {
    let list: Vec<_> = search
        .peaks
        .iter()
        .map(|p| {
            format!(
                "[{},{},{},{}]",
                p.omega, p.value.re, p.value.im, p.iterations
            )
        })
        .collect();
    let s = search.stats;
    format!(
        "{{\"peaks\":[{}],\"scan_points\":{},\"rect_evaluations\":{},\"references\":{},\"seeds\":{},\"refined\":{},\"derivative_evaluations\":{},\"iterations\":{},\"flanks\":{},\"unconverged\":{},\"merged\":{},\"excluded\":{}}}",
        list.join(","),
        s.scan_points,
        s.rect_evaluations,
        s.references,
        s.seeds,
        s.refined,
        s.derivative_evaluations,
        s.iterations,
        s.flanks,
        s.unconverged,
        s.merged,
        s.excluded
    )
}
fn complex_json(values: &[Complex]) -> String {
    let list: Vec<_> = values
        .iter()
        .map(|z| format!("[{},{}]", z.re, z.im))
        .collect();
    format!("[{}]", list.join(","))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if !(5..=6).contains(&a.len()) {
        return Err("usage: faft_bench N ERROR SPAN REPEATS QUERIES [DENSE] < frames".into());
    }
    let n: usize = a[0].parse()?;
    let error: f64 = a[1].parse()?;
    let span: usize = a[2].parse()?;
    let repeats: usize = a[3].parse()?;
    let queries: usize = a[4].parse()?;
    let dense: usize = a.get(5).map(|v| v.parse()).transpose()?.unwrap_or(0);
    if !(8..=8192).contains(&n)
        || !(1..=31).contains(&repeats)
        || !(1..=4096).contains(&queries)
        || dense > 65536
    {
        return Err("invalid benchmark sizes".into());
    }
    let geometry = GeometryParams {
        max_error: error,
        max_span: span,
        min_precision: 12,
        ..GeometryParams::default()
    };
    // Settings of the earlier adaptive survey, for a like-for-like search control.
    let adaptive = FrequencyParams {
        max_depth: 12,
        max_evaluations: 8192,
        min_width: TAU / n as f64 / 1024.0,
        merge_width: TAU / n as f64 * 0.5,
        relative_floor: 0.02,
        initial_intervals: n / 4,
        ..FrequencyParams::default()
    };
    let scan = PeakParams {
        relative_floor: 0.02,
        ..PeakParams::default()
    };
    let circles = PeakParams {
        scan_density: None,
        circle_seeds: true,
        ..scan
    };
    // Deterministic arbitrary (off-grid) query frequencies in [0, pi).
    let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
    let omegas: Vec<f64> = (0..queries)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            PI * (seed >> 11) as f64 / (1_u64 << 53) as f64
        })
        .collect();
    let mut text = String::new();
    io::stdin().read_to_string(&mut text)?;
    let source: Vec<f64> = text
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if source.is_empty() || !source.len().is_multiple_of(n) {
        return Err("expected complete frames".into());
    }
    let mut out = io::BufWriter::new(io::stdout().lock());
    for x in source.chunks_exact(n) {
        let ((frame, _), fit_ns) = timed(repeats, || {
            GeometryFrame::hierarchical(x, geometry, HierarchyParams::default()).unwrap()
        });
        let (faft, prepare_ns) = timed(repeats, || {
            CircleCurve::from_geometry(&frame)
                .unwrap()
                .transform(0..n, Weighting::Hann)
                .unwrap()
        });
        let curve = CircleCurve::from_geometry(&frame)?;
        let (values, faft_ns) = timed(repeats, || faft.values(&omegas).unwrap());

        // Earlier sampled-predictor kernels on the same geometry (zoom 1, Hann).
        let (sampled, sampled_prepare_ns) =
            timed(repeats, || frame.prepare_spectrum(0..n, 1, true).unwrap());
        let (closed, closed_ns) = timed(repeats, || {
            omegas
                .iter()
                .map(|&w| sampled.value(w).unwrap())
                .collect::<Vec<_>>()
        });
        let (_, rotations_ns) = timed(repeats, || {
            omegas
                .iter()
                .map(|&w| sampled.value_rotations(w).unwrap())
                .collect::<Vec<_>>()
        });
        // Zoom samples the SAME curve more densely; its DTFT tends to the FAFT.
        let mut zoom = Vec::new();
        for z in [1_usize, 2, 4, 8, 16, 64] {
            let spectrum = frame.prepare_spectrum(0..n, z, true)?;
            let v: Vec<_> = omegas
                .iter()
                .map(|&w| spectrum.value_rotations(w))
                .collect::<Result<_, _>>()?;
            zoom.push(format!("[{z},{}]", relative_l2(&v, &values)));
        }

        // Peaks: continuous FAFT (scan + Newton, circle references) vs adaptive DTFT.
        let (by_scan, scan_ns) = timed(repeats, || faft.peaks(scan).unwrap());
        let (by_circles, circles_ns) = timed(repeats, || faft.peaks(circles).unwrap());
        let (by_adaptive, adaptive_ns) = timed(repeats, || {
            AdaptiveSpectrum::search(&sampled, None, adaptive).unwrap()
        });
        let references = faft.references(faft.resolution(), 0.01)?;
        // Scan density is a seed heuristic: count peaks at 1/2/4 points per 2*pi/T.
        let mut densities = Vec::new();
        for d in [1_usize, 2, 4] {
            let p = PeakParams {
                scan_density: Some(d),
                ..scan
            };
            let (r, ns) = timed(repeats, || faft.peaks(p).unwrap());
            let shared = r
                .peaks
                .iter()
                .filter(|q| {
                    by_scan
                        .peaks
                        .iter()
                        .any(|b| (b.omega - q.omega).abs() < 1e-6)
                })
                .count();
            densities.push(format!("[{d},{},{shared},{}]", r.peaks.len(), median(ns)));
        }

        // Integral AFT reference and FAFT on the AFT's own triplet curve.
        let (aft, aft_fit_ns) = timed(repeats, || {
            AngularFrame::new(x, AnalysisParams::original()).unwrap()
        });
        let (aft_values, aft_ns) = timed(repeats, || {
            omegas
                .iter()
                .map(|&w| aft.fourier(w).unwrap())
                .collect::<Vec<_>>()
        });
        let (aft_faft, aft_prepare_ns): (Faft, _) = timed(repeats, || {
            CircleCurve::from_angular(&aft)
                .unwrap()
                .transform(0..n, Weighting::Rectangular)
                .unwrap()
        });
        let (aft_faft_values, aft_faft_ns) = timed(repeats, || {
            omegas
                .iter()
                .map(|&w| aft_faft.integral(w).unwrap())
                .collect::<Vec<_>>()
        });

        let mut spectra = String::new();
        if dense > 0 {
            let grid: Vec<f64> = (0..dense)
                .map(|k| 3.0 * PI * k as f64 / dense as f64)
                .collect();
            let exact: Vec<f64> = faft.values(&grid)?.into_iter().map(norm).collect();
            let mut sampled_dense = Vec::new();
            for z in [1, 4] {
                let s = frame.prepare_spectrum(0..n, z, true)?;
                let v: Vec<f64> = grid
                    .iter()
                    .map(|&w| s.value_rotations(w).map(norm))
                    .collect::<Result<_, _>>()?;
                sampled_dense.push(v);
            }
            spectra = format!(
                ",\"dense_omega\":{grid:?},\"dense_faft\":{exact:?},\"dense_zoom1\":{:?},\"dense_zoom4\":{:?}",
                sampled_dense[0], sampled_dense[1]
            );
        }
        let stats = frame.stats();
        let per_query = |t: Vec<u128>| median(t) as f64 / queries as f64;
        write!(
            out,
            "{{\"n\":{n},\"circles\":{},\"lines\":{},\"rotations\":{},\"aft_rotations\":{},\"max_error\":{},\"rmse\":{},\
\"fit_ns\":{},\"prepare_ns\":{},\"faft_query_ns\":{},\"sampled_prepare_ns\":{},\"closed_query_ns\":{},\"rotation_query_ns\":{},\
\"aft_fit_ns\":{},\"aft_query_ns\":{},\"aft_faft_prepare_ns\":{},\"aft_faft_query_ns\":{},\"aft_faft_relative_l2\":{},\
\"sampled_vs_faft_relative_l2\":{},\"zoom_relative_l2\":[{}],\
\"scan_ns\":{},\"circles_ns\":{},\"adaptive_ns\":{},\"adaptive_queries\":{},\"adaptive_budget_exhausted\":{},\
\"densities\":[{}],\"scan\":{},\"circle_seeded\":{},\"adaptive_peaks\":{},\"references\":[{}],\
\"omegas\":{:?},\"values\":{},\"model\":[{}]{spectra}}}",
            stats.circles,
            stats.lines,
            faft.rotation_count(),
            aft_faft.rotation_count(),
            stats.max_error,
            stats.rmse,
            median(fit_ns),
            median(prepare_ns),
            per_query(faft_ns),
            median(sampled_prepare_ns),
            per_query(closed_ns),
            per_query(rotations_ns),
            median(aft_fit_ns),
            per_query(aft_ns),
            median(aft_prepare_ns),
            per_query(aft_faft_ns),
            relative_l2(&aft_faft_values, &aft_values),
            relative_l2(&closed, &values),
            zoom.join(","),
            median(scan_ns),
            median(circles_ns),
            median(adaptive_ns),
            by_adaptive.stats.coarse_queries + by_adaptive.stats.fine_queries,
            by_adaptive.stats.budget_exhausted,
            densities.join(","),
            peaks_json(&by_scan),
            peaks_json(&by_circles),
            complex_json(
                &by_adaptive
                    .peaks
                    .iter()
                    .map(|p| Complex {
                        re: p.omega,
                        im: norm(p.value),
                    })
                    .collect::<Vec<_>>()
            ),
            references
                .iter()
                .map(|r| format!("[{},{},{}]", r.omega, r.weight, r.rotations))
                .collect::<Vec<_>>()
                .join(","),
            omegas,
            complex_json(&values),
            curve
                .rotations()
                .iter()
                .map(|r| format!(
                    "[{},{},{},{},{},{},{}]",
                    r.start, r.length, r.center, r.radius, r.phase, r.rate, r.slope
                ))
                .collect::<Vec<_>>()
                .join(",")
        )?;
        writeln!(out)?;
    }
    Ok(())
}
