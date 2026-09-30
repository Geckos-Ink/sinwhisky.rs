//! Private JSON Lines for hierarchical circle fitting and adaptive DTFT peaks.
//! Usage: adaptive_bench N ERROR SPAN ZOOM DEPTH BUDGET REPEATS [COARSE_ERROR] [PRUNE_FLAT]
use sinwhisky::adaptive_faft::{AdaptiveSpectrum, FrequencyParams, StopReason};
use sinwhisky::geometric_faft::{GeometryFrame, GeometryParams, HierarchyParams, WavePrimitive};
use std::f64::consts::{PI, TAU};
use std::hint::black_box;
use std::io::{self, Read, Write};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if !(7..=9).contains(&a.len()) {
        return Err("usage: adaptive_bench N ERROR SPAN ZOOM DEPTH BUDGET REPEATS [COARSE_ERROR] [PRUNE_FLAT]".into());
    }
    let n: usize = a[0].parse()?;
    let error: f64 = a[1].parse()?;
    let span: usize = a[2].parse()?;
    let zoom: usize = a[3].parse()?;
    let depth: usize = a[4].parse()?;
    let budget: usize = a[5].parse()?;
    let repeats: usize = a[6].parse()?;
    let coarse_error: f64 = a
        .get(7)
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(error * 8.0);
    let prune_flat = a.get(8).is_none_or(|v| v == "1");
    if !(2..=4096).contains(&n) || !(1..=9).contains(&repeats) {
        return Err("invalid benchmark sizes".into());
    }
    let params = GeometryParams {
        max_error: error,
        max_span: span,
        min_precision: 12,
        ..GeometryParams::default()
    };
    let coarse_params = GeometryParams {
        max_error: coarse_error,
        min_precision: 8,
        ..params
    };
    let frequency = FrequencyParams {
        max_depth: depth,
        max_evaluations: budget,
        min_width: TAU / n as f64 / 1024.0,
        merge_width: TAU / n as f64 * 0.5,
        relative_floor: 0.02,
        initial_intervals: n / 4,
        prune_flat,
        ..FrequencyParams::default()
    };
    let mut data = String::new();
    io::stdin().read_to_string(&mut data)?;
    let source: Vec<f64> = data
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if source.is_empty() || !source.len().is_multiple_of(n) {
        return Err("expected complete frames".into());
    }
    let mut out = io::BufWriter::new(io::stdout().lock());
    for x in source.chunks_exact(n) {
        let (frame, h) = GeometryFrame::hierarchical(x, params, HierarchyParams::default())?;
        let (before, _) = GeometryFrame::hierarchical(
            x,
            params,
            HierarchyParams {
                merge_passes: 0,
                ..HierarchyParams::default()
            },
        )?;
        let (coarse, _) =
            GeometryFrame::hierarchical(x, coarse_params, HierarchyParams::default())?;
        let prepared = frame.prepare_spectrum(0..n, zoom, true)?;
        let preview = coarse.prepare_spectrum(0..n, zoom, true)?;
        let result = AdaptiveSpectrum::search(&prepared, Some(&preview), frequency)?;
        let direct = AdaptiveSpectrum::search(&prepared, None, frequency)?;
        let mut fits = Vec::new();
        let mut prepares = Vec::new();
        let mut searches = Vec::new();
        let mut fine_searches = Vec::new();
        let mut dp_times = Vec::new();
        let mut dp_count = 0;
        for _ in 0..repeats {
            let at = Instant::now();
            black_box(GeometryFrame::hierarchical(
                black_box(x),
                params,
                HierarchyParams::default(),
            )?);
            fits.push(at.elapsed().as_nanos());
            let at = Instant::now();
            black_box(frame.prepare_spectrum(0..n, zoom, true)?);
            prepares.push(at.elapsed().as_nanos());
            let at = Instant::now();
            black_box(AdaptiveSpectrum::search(
                &prepared,
                Some(&preview),
                frequency,
            )?);
            searches.push(at.elapsed().as_nanos());
            let at = Instant::now();
            black_box(AdaptiveSpectrum::search(&prepared, None, frequency)?);
            fine_searches.push(at.elapsed().as_nanos());
            let at = Instant::now();
            let dp = GeometryFrame::new(
                x,
                GeometryParams {
                    max_span: span.next_power_of_two().min(256),
                    ..params
                },
            )?;
            dp_times.push(at.elapsed().as_nanos());
            dp_count = dp.segments().len();
        }
        let s = frame.stats();
        write!(out,"{{\"fit_ns\":{fits:?},\"prepare_ns\":{prepares:?},\"adaptive_ns\":{searches:?},\"fine_only_ns\":{fine_searches:?},\"dp_ns\":{dp_times:?},\"dp_count\":{dp_count},\"circles\":{},\"lines\":{},\"coarse_count\":{},\"unmerged_count\":{},\"merges\":{},\"splits\":{},\"candidates\":{},\"max_error\":{},\"rmse\":{},\"scale\":{},\"max_span\":{},\"queries\":{{\"coarse\":{},\"fine\":{},\"skipped_fine\":{},\"disagreement_bound\":{},\"budget_exhausted\":{},\"peak_refinements\":{},\"fine_only\":{}}},\"stops\":{{",s.circles,s.lines,coarse.segments().len(),before.segments().len(),h.merges_accepted,h.splits,s.candidates_tested,s.max_error,s.rmse,frame.scale(),h.largest_span,result.stats.coarse_queries,result.stats.fine_queries,result.stats.skipped_fine,result.stats.disagreement_bound,result.stats.budget_exhausted,result.stats.peak_refinements,direct.stats.fine_queries)?;
        for (i, (name, reason)) in [
            ("below_floor", StopReason::BelowFloor),
            ("low_gain", StopReason::LowGain),
            ("width", StopReason::Width),
            ("depth", StopReason::Depth),
            ("budget", StopReason::Budget),
        ]
        .iter()
        .enumerate()
        {
            if i > 0 {
                write!(out, ",")?;
            }
            write!(
                out,
                "\"{name}\":{}",
                result.bands.iter().filter(|b| b.reason == *reason).count()
            )?;
        }
        write!(out, "}},\"peaks\":[")?;
        for (i, p) in result.peaks.iter().enumerate() {
            if i > 0 {
                write!(out, ",")?;
            }
            write!(
                out,
                "{{\"omega\":{},\"value\":[{},{}],\"bracket\":{:?},\"merged\":{}}}",
                p.omega,
                p.value.re,
                p.value.im,
                [p.bracket.0, p.bracket.1],
                p.merged_candidates
            )?;
        }
        write!(out, "],\"points\":[")?;
        for (i, p) in result.points.iter().enumerate() {
            if i > 0 {
                write!(out, ",")?;
            }
            write!(
                out,
                "[{}, {}, {}, {}, {}]",
                p.omega, p.value.re, p.value.im, p.uncertainty, p.verified
            )?;
        }
        write!(out, "],\"models\":[")?;
        for (i, s) in frame.segments().iter().enumerate() {
            if i > 0 {
                write!(out, ",")?;
            }
            write!(
                out,
                "{{\"start\":{},\"end\":{},",
                s.range().start,
                s.range().end
            )?;
            match s.primitive() {
                WavePrimitive::Line {first,last,precision}=>write!(out,"\"kind\":\"line\",\"first\":{first},\"last\":{last},\"bits\":{precision}}}")?,
                WavePrimitive::Circle(c)=>write!(out,"\"kind\":\"circle\",\"cx\":{},\"cy\":{},\"radius\":{},\"x_scale\":{},\"angles\":{:?},\"middle\":{},\"bits\":{}}}",c.center.x,c.center.y,c.radius,c.x_scale,c.angles,c.middle,c.precision)?,
            }
        }
        write!(out, "],\"dense\":[")?;
        for j in 0..n * zoom {
            if j > 0 {
                write!(out, ",")?;
            }
            write!(out, "{}", frame.evaluate(j as f64 / zoom as f64)?)?;
        }
        // Independent reference frequencies include off-bin values and Nyquist.
        write!(out, "],\"check\":[")?;
        for (i, w) in [0.0, PI / 17.231, 0.473829, 1.11361, PI].iter().enumerate() {
            if i > 0 {
                write!(out, ",")?;
            }
            let v = prepared.value(*w)?;
            write!(out, "[{w},{},{}]", v.re, v.im)?;
        }
        writeln!(out, "]}}")?;
    }
    Ok(())
}
