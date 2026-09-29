//! Numeric frames on stdin -> JSON Lines of AFT coefficients and in-process timings.
//! Usage: aft_bench FRAME_SIZE REPEATS MIDPOINT_OVERSAMPLE (0 disables dense output).
use sinwhisky::aft::AngularFrame;
use sinwhisky::AnalysisParams;
use std::hint::black_box;
use std::io::{self, Read, Write};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    let [n, repeats, dense]: [usize; 3] = args
        .try_into()
        .map_err(|_| "usage: aft_bench FRAME_SIZE REPEATS MIDPOINT_OVERSAMPLE")?;
    if !(2..=8192).contains(&n) || !(1..=31).contains(&repeats) || dense > 256 {
        return Err("invalid benchmark sizes".into());
    }
    let mut text = String::new();
    io::stdin().read_to_string(&mut text)?;
    let samples: Vec<f64> = text
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if samples.is_empty() || !samples.len().is_multiple_of(n) {
        return Err("expected complete frames".into());
    }
    let mut out = io::BufWriter::new(io::stdout().lock());
    for samples in samples.chunks_exact(n) {
        let frame = AngularFrame::new(samples, AnalysisParams::original())?;
        black_box(frame.spectrum()?); // Warm the transform before timing it.
        let mut fits = Vec::with_capacity(repeats);
        let mut transforms = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let start = Instant::now();
            let prepared = black_box(AngularFrame::new(
                black_box(samples),
                AnalysisParams::original(),
            )?);
            fits.push(start.elapsed().as_nanos());
            black_box(prepared);
            let start = Instant::now();
            black_box(frame.spectrum()?);
            transforms.push(start.elapsed().as_nanos());
        }
        let spectrum = frame.spectrum()?;
        let stats = frame.stats();
        write!(out,"{{\"fit_ns\":{fits:?},\"aft_ns\":{transforms:?},\"circle_halves\":{},\"line_halves\":{},\"rejected_circle_halves\":{},\"aft\":[",stats.circle_halves,stats.line_halves,stats.rejected_circle_halves)?;
        for (i, z) in spectrum.iter().enumerate() {
            if i > 0 {
                write!(out, ",")?;
            }
            write!(out, "[{},{}]", z.re, z.im)?;
        }
        write!(out, "],\"dense\":[")?;
        for j in 0..n * dense {
            if j > 0 {
                write!(out, ",")?;
            }
            write!(out, "{}", frame.evaluate((j as f64 + 0.5) / dense as f64)?)?;
        }
        writeln!(out, "]}}")?;
    }
    out.flush()?;
    Ok(())
}
