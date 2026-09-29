//! Research bridge: numeric stdin -> JSON Lines of existing Rust uniform models.
//! Arguments: block_size (default 256). Samples use normalized amplitude units.
use sinwhisky::geometry::Branch;
use sinwhisky::{Analysis, AnalysisParams, Primitive};
use std::io::{self, Read, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() > 1 {
        return Err("usage: export_circles [block_size]".into());
    }
    let block: usize = args.first().map(|v| v.parse()).transpose()?.unwrap_or(256);
    if !(4..=4096).contains(&block) || !block.is_power_of_two() {
        return Err("block_size must be a power of two in 4..=4096".into());
    }
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let samples: Vec<f64> = input
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let mut out = io::BufWriter::new(io::stdout().lock());
    for samples in samples.chunks(block) {
        let analysis = Analysis::new(samples, AnalysisParams::default())?;
        write!(out, "[")?;
        for (i, model) in analysis.models().iter().enumerate() {
            if i > 0 {
                write!(out, ",")?;
            }
            match model.primitive {
                Primitive::Line { .. } => write!(out, "null")?,
                Primitive::Circle(c) => write!(
                    out,
                    "[{},{},{},{},{},{},{}]",
                    c.anchor,
                    c.aperture,
                    c.circle.center.x,
                    c.circle.center.y,
                    c.circle.radius,
                    if c.circle.branch == Branch::Upper {
                        1
                    } else {
                        -1
                    },
                    c.evaluate_uniform(c.anchor as f64)
                        .ok_or("invalid fitted circle")?
                )?,
            }
        }
        writeln!(out, "]")?;
    }
    out.flush()?;
    Ok(())
}
