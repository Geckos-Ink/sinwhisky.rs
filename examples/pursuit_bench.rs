//! Synthetic, deterministic comparison of FAFT peak selection policies on the SAME
//! fitted curve: the duration scan + Newton (`peaks`), the joint-fit residual
//! pursuit (`pursue`) and the frame-to-frame tracker (`PeakTracker`).
//! Usage: pursuit_bench [REPEATS]   (prints one JSON object)
//!
//! The signal is generated here (no audio): two harmonic notes (one gliding, one
//! entering halfway), a strong/weak pair 1.5 resolutions apart, a -40 dB tone and
//! low-level noise, at 44.1 kHz with 1024-sample Hann windows and a 512-sample hop.
//! Truth is the instantaneous frequency at each window centre. A truth partial is
//! recovered when a reported peak lies within half a resolution; a report farther
//! than one resolution from every truth partial is spurious. Heights compare the
//! reported magnitude (observed for `peaks`, fitted for the others) with A/2.
use sinwhisky::faft::{
    CircleCurve, Faft, FramePath, PeakParams, PeakTracker, PursuitParams, TrackerParams, Weighting,
};
use sinwhisky::geometric_faft::{GeometryFrame, GeometryParams, HierarchyParams};
use std::f64::consts::TAU;
use std::hint::black_box;
use std::time::Instant;

const RATE: f64 = 44_100.0;
const N: usize = 1024;
const HOP: usize = 512;
const FRAMES: usize = 48;

/// (frequency Hz at time t seconds, amplitude, active?) for every partial.
fn partials(t: f64, total: f64) -> Vec<(f64, f64, bool)> {
    let mut out = Vec::new();
    let f0 = 220.0 + 6.0 * t / total; // slow glide
    for k in 1..=8 {
        out.push((f0 * k as f64, 1000.0 / k as f64, true));
    }
    let entered = t >= 0.5 * total;
    for k in 1..=5 {
        out.push((330.0 * k as f64 * 1.0013, 400.0 / k as f64, entered));
    }
    let bin = RATE / N as f64;
    out.push((2400.0, 300.0, true));
    out.push((2400.0 + 1.5 * bin, 100.0, true));
    out.push((3150.0, 20.0, true)); // -40 dB re 2000 peak-to-peak
    out
}

fn signal(len: usize) -> Vec<f64> {
    let total = len as f64 / RATE;
    let mut phases = [0.0_f64; 16];
    let mut seed = 0x2545_F491_4F6C_DD1D_u64;
    (0..len)
        .map(|i| {
            let t = i as f64 / RATE;
            let mut y = 0.0;
            for (p, (f, a, on)) in partials(t, total).into_iter().enumerate() {
                phases[p] += TAU * f / RATE;
                if on {
                    y += a * (phases[p] + 0.37 * p as f64).sin();
                }
            }
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            y + 6.0 * ((seed >> 11) as f64 / (1_u64 << 53) as f64 - 0.5)
        })
        .collect()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[derive(Default)]
struct Score {
    truth: usize,
    recovered: usize,
    weak_recovered: usize,
    weak_truth: usize,
    spurious: usize,
    reported: usize,
    height_db: Vec<f64>,
    frequency_bins: Vec<f64>,
    seconds: Vec<f64>,
    derivative_evaluations: usize,
    rect_evaluations: usize,
}
impl Score {
    fn add(&mut self, faft: &Faft, truth: &[(f64, f64)], reported: &[(f64, f64)]) {
        let res = faft.resolution();
        self.reported += reported.len();
        for &(w, m) in reported {
            if !truth.iter().any(|&(tw, _)| (tw - w).abs() <= res) {
                self.spurious += 1;
            }
            let _ = m;
        }
        for &(tw, a) in truth {
            self.truth += 1;
            let weak =
                a <= 25.0 || (tw - TAU * (2400.0 + 1.5 * RATE / N as f64) / RATE).abs() < 1e-9;
            if weak {
                self.weak_truth += 1;
            }
            if let Some(&(w, m)) = reported
                .iter()
                .filter(|&&(w, _)| (w - tw).abs() <= 0.5 * res)
                .min_by(|x, y| (x.0 - tw).abs().total_cmp(&(y.0 - tw).abs()))
            {
                self.recovered += 1;
                if weak {
                    self.weak_recovered += 1;
                }
                self.height_db.push((20.0 * (m / (0.5 * a)).log10()).abs());
                self.frequency_bins.push((w - tw).abs() / res);
            }
        }
    }
    fn json(&self, name: &str, frames: usize) -> String {
        let pct = |v: &[f64], q: f64| {
            let mut v = v.to_vec();
            v.sort_by(f64::total_cmp);
            v[((v.len() - 1) as f64 * q).round() as usize]
        };
        format!(
            "\"{name}\":{{\"recall\":{:.4},\"weak_recall\":{:.4},\"reported_per_frame\":{:.2},\"spurious_per_frame\":{:.2},\"height_error_db_median\":{:.4},\"height_error_db_p95\":{:.4},\"frequency_error_bins_median\":{:.2e},\"frequency_error_bins_p95\":{:.2e},\"ms_per_frame_median\":{:.4},\"derivative_evaluations_per_frame\":{:.1},\"rect_evaluations_per_frame\":{:.1}}}",
            self.recovered as f64 / self.truth as f64,
            self.weak_recovered as f64 / self.weak_truth.max(1) as f64,
            self.reported as f64 / frames as f64,
            self.spurious as f64 / frames as f64,
            pct(&self.height_db, 0.5),
            pct(&self.height_db, 0.95),
            pct(&self.frequency_bins, 0.5),
            pct(&self.frequency_bins, 0.95),
            1e3 * median(self.seconds.clone()),
            self.derivative_evaluations as f64 / frames as f64,
            self.rect_evaluations as f64 / frames as f64,
        )
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repeats: usize = std::env::args()
        .nth(1)
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(5)
        .clamp(1, 31);
    let len = N + HOP * (FRAMES - 1);
    let total = len as f64 / RATE;
    let samples = signal(len);
    let (frame, _) = GeometryFrame::hierarchical(
        &samples,
        GeometryParams {
            max_error: 4.0,
            max_span: 128,
            ..GeometryParams::default()
        },
        HierarchyParams::default(),
    )?;
    let curve = CircleCurve::from_geometry(&frame)?;
    let floor = 0.01;
    let scan = PeakParams {
        relative_floor: floor,
        ..PeakParams::default()
    };
    let pursuit = PursuitParams {
        search: PeakParams {
            max_peaks: 256,
            ..scan
        },
        ..PursuitParams::default()
    };
    let tracker_params = TrackerParams {
        pursuit,
        ..TrackerParams::default()
    };
    // Truth for windows not straddling the onset of the second note.
    let onset = 0.5 * total;
    let mut truths = Vec::new();
    let mut fafts = Vec::new();
    let mut rotations = 0;
    for f in 0..FRAMES {
        let window = f * HOP..f * HOP + N;
        let (start, end) = (window.start as f64 / RATE, window.end as f64 / RATE);
        let centre = 0.5 * (start + end);
        let straddles = start < onset && end > onset;
        let truth: Vec<(f64, f64)> = partials(centre, total)
            .into_iter()
            .filter(|p| p.2)
            .map(|(hz, a, _)| (TAU * hz / RATE, a))
            .collect();
        let faft = curve.transform(window, Weighting::Hann)?;
        rotations += faft.rotation_count();
        truths.push((!straddles).then_some(truth));
        fafts.push(faft);
    }
    let scored = truths.iter().filter(|t| t.is_some()).count();
    let mut by_scan = Score::default();
    let mut by_pursuit = Score::default();
    let mut by_tracker = Score::default();
    let mut paths = [0_usize; 4];
    let mut certified = 0;
    let mut explained = Vec::new();
    let mut bound_ratio = Vec::new();
    let mut energy_us = Vec::new();
    // Rectangular weighting: sidelobes against joint fit, on the same windows.
    let (mut rect_scan, mut rect_pursuit) = (0, 0);
    for (faft, truth) in fafts.iter().zip(&truths) {
        let mut times = Vec::new();
        let mut result = faft.peaks(scan)?;
        for _ in 0..repeats {
            let begin = Instant::now();
            result = black_box(faft.peaks(scan)?);
            times.push(begin.elapsed().as_secs_f64());
        }
        let mut times_p = Vec::new();
        let mut joint = faft.pursue(pursuit)?;
        for _ in 0..repeats {
            let begin = Instant::now();
            joint = black_box(faft.pursue(pursuit)?);
            times_p.push(begin.elapsed().as_secs_f64());
        }
        let begin = Instant::now();
        black_box(faft.energy()?);
        energy_us.push(1e6 * begin.elapsed().as_secs_f64());
        explained.push(joint.explained());
        bound_ratio.push(joint.bound / joint.floor);
        if let Some(truth) = truth {
            by_scan.seconds.push(median(times));
            by_scan.derivative_evaluations += result.stats.derivative_evaluations;
            by_scan.rect_evaluations += result.stats.rect_evaluations;
            let r: Vec<_> = result
                .peaks
                .iter()
                .map(|p| (p.omega, p.magnitude()))
                .collect();
            by_scan.add(faft, truth, &r);
            by_pursuit.seconds.push(median(times_p));
            by_pursuit.derivative_evaluations += joint.stats.search.derivative_evaluations;
            by_pursuit.rect_evaluations += joint.stats.search.rect_evaluations;
            let r: Vec<_> = joint
                .components
                .iter()
                .map(|c| (c.omega, c.magnitude()))
                .collect();
            by_pursuit.add(faft, truth, &r);
        }
        let rect = curve.transform(faft.window(), Weighting::Rectangular)?;
        rect_scan += rect.peaks(scan)?.peaks.len();
        rect_pursuit += rect.pursue(pursuit)?.components.len();
    }
    // Tracker: the whole sequence, repeated with a fresh tracker each time.
    let mut sequence = Vec::new();
    let mut frames_out = Vec::new();
    for rep in 0..repeats.max(1) {
        let mut tracker = PeakTracker::new(tracker_params)?;
        let mut out = Vec::new();
        let mut per_frame = Vec::new();
        for faft in &fafts {
            let begin = Instant::now();
            let frame = tracker.update(faft)?;
            per_frame.push(begin.elapsed().as_secs_f64());
            out.push(frame);
        }
        sequence.push(per_frame);
        if rep == 0 {
            frames_out = out;
        }
    }
    let mut ids = std::collections::BTreeMap::<u64, usize>::new();
    let (mut tracked_ms, mut scanning_ms) = (Vec::new(), Vec::new());
    for (f, ((faft, truth), frame)) in fafts.iter().zip(&truths).zip(&frames_out).enumerate() {
        match frame.path {
            FramePath::First => paths[0] += 1,
            FramePath::Interval => paths[1] += 1,
            FramePath::Novelty => paths[2] += 1,
            FramePath::Tracked { certified: c } => {
                paths[3] += 1;
                certified += usize::from(c);
            }
        }
        for p in &frame.peaks {
            *ids.entry(p.id).or_default() += 1;
        }
        let ms = 1e3 * median(sequence.iter().map(|s| s[f]).collect());
        if matches!(frame.path, FramePath::Tracked { .. }) {
            tracked_ms.push(ms);
        } else {
            scanning_ms.push(ms);
        }
        if let Some(truth) = truth {
            by_tracker
                .seconds
                .push(median(sequence.iter().map(|s| s[f]).collect()));
            by_tracker.derivative_evaluations += frame.stats.search.derivative_evaluations;
            by_tracker.rect_evaluations += frame.stats.search.rect_evaluations;
            let r: Vec<_> = frame
                .peaks
                .iter()
                .map(|p| (p.omega, p.value.re.hypot(p.value.im)))
                .collect();
            by_tracker.add(faft, truth, &r);
        }
    }
    // Ideal Hann leakage scale for the record.
    let hann = &fafts[0];
    println!(
        "{{\"sample_rate\":{RATE},\"window\":{N},\"hop\":{HOP},\"frames\":{FRAMES},\"scored_frames\":{scored},\"rotations_per_frame\":{:.1},\"truth_partials_per_frame\":{:.2},{},{},{},\"tracker_paths\":{{\"first\":{},\"interval\":{},\"novelty\":{},\"tracked\":{},\"tracked_certified\":{}}},\"tracker_track_ids\":{},\"tracker_ids_in_8_or_more_frames\":{},\"tracker_ms_median\":{{\"tracked\":{:.4},\"scanning\":{:.4}}},\"ms_per_frame_mean\":{{\"scan_newton\":{:.4},\"pursuit\":{:.4},\"tracker\":{:.4}}},\"pursuit_explained_median\":{:.6},\"pursuit_bound_over_floor_median\":{:.3},\"energy_us_median\":{:.2},\"rectangular_reported_per_frame\":{{\"peaks\":{:.2},\"pursuit\":{:.2}}},\"hann_minus3db_halfwidth_bins\":{:.4},\"hann_minus6db_halfwidth_bins\":{:.4}}}",
        rotations as f64 / FRAMES as f64,
        by_scan.truth as f64 / scored as f64,
        by_scan.json("scan_newton", scored),
        by_pursuit.json("pursuit", scored),
        by_tracker.json("tracker", scored),
        paths[0],
        paths[1],
        paths[2],
        paths[3],
        certified,
        ids.len(),
        ids.values().filter(|&&frames| frames >= 8).count(),
        median(tracked_ms.clone()),
        median(scanning_ms.clone()),
        1e3 * mean(&by_scan.seconds),
        1e3 * mean(&by_pursuit.seconds),
        1e3 * mean(&by_tracker.seconds),
        median(explained),
        median(bound_ratio),
        median(energy_us),
        rect_scan as f64 / FRAMES as f64,
        rect_pursuit as f64 / FRAMES as f64,
        hann.kernel_range(3.0)? / hann.resolution(),
        hann.kernel_range(6.0)? / hann.resolution(),
    );
    Ok(())
}
