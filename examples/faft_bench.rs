//! Complete numeric frames -> private JSON Lines for the FAFT research harness.
//! Usage: faft_bench N REPEATS MIN_BLOCK MAX_RMSE
use sinwhisky::faft::{transform, Baseline, FaftFrame, FaftParams};
use sinwhisky::spectral::Complex;
use std::hint::black_box;
use std::io::{self, Read, Write};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: faft_bench N REPEATS MIN_BLOCK MAX_RMSE".into());
    }
    let n: usize = args[0].parse()?;
    let repeats: usize = args[1].parse()?;
    let minimum: usize = args[2].parse()?;
    let tolerance: f64 = args[3].parse()?;
    if !(2..=8192).contains(&n) || !(1..=31).contains(&repeats) {
        return Err("invalid sizes".into());
    }
    let params = FaftParams {
        max_rmse: tolerance,
        min_block: minimum,
    };
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
        let input: Vec<_> = samples.iter().map(|&re| Complex { re, im: 0.0 }).collect();
        let frame = FaftFrame::new(samples, params)?;
        black_box(transform(&input, false)?);
        black_box(frame.decode()?);
        let mut kernels = Vec::new();
        let mut fits = Vec::new();
        let mut decodes = Vec::new();
        for _ in 0..repeats {
            let begin = Instant::now();
            black_box(transform(black_box(&input), false)?);
            kernels.push(begin.elapsed().as_nanos());
            let begin = Instant::now();
            black_box(FaftFrame::new(black_box(samples), params)?);
            fits.push(begin.elapsed().as_nanos());
            let begin = Instant::now();
            black_box(frame.decode()?);
            decodes.push(begin.elapsed().as_nanos());
        }
        let stats = frame.stats();
        write!(out,"{{\"kernel_ns\":{kernels:?},\"fit_ns\":{fits:?},\"decode_ns\":{decodes:?},\"circles\":{},\"blocks\":{},\"descriptors\":{},\"line_blocks\":{},\"estimated_model_bits\":{},\"rmse\":{},\"max_error\":{},\"scale\":{},\"models\":[",stats.circles,stats.blocks,stats.descriptors,stats.line_blocks,stats.estimated_model_bits,stats.rmse,stats.max_error,frame.scale())?;
        for (i, b) in frame.blocks().iter().enumerate() {
            if i > 0 {
                write!(out, ",")?;
            }
            let (kind, first, last) = match b.baseline() {
                Baseline::Constant(v) => ("constant", v, v),
                Baseline::Line(l) => ("line", l.first, l.last),
            };
            write!(out,"{{\"start\":{},\"end\":{},\"baseline\":\"{kind}\",\"first\":{first},\"last\":{last},\"circles\":[",b.range().start,b.range().end)?;
            for (j, c) in b.circles().iter().enumerate() {
                if j > 0 {
                    write!(out, ",")?;
                }
                let z = c.precision.coefficient();
                write!(
                    out,
                    "{{\"bin\":{},\"re\":{},\"im\":{},\"bits\":{}}}",
                    c.bin,
                    z.re,
                    z.im,
                    c.precision.bits()
                )?;
            }
            write!(out, "]}}")?;
        }
        write!(out, "],\"spectrum\":[")?;
        for (i, z) in transform(&input, false)?.iter().take(n / 2 + 1).enumerate() {
            if i > 0 {
                write!(out, ",")?;
            }
            write!(out, "[{},{}]", z.re, z.im)?;
        }
        write!(out, "],\"decoded\":{:?}}}", frame.decode()?)?;
        writeln!(out)?;
    }
    Ok(())
}
