use sinwhisky::{resample, Analysis, AnalysisParams, Aperture, ResampleParams};
use std::io::{self, Read, Write};

const USAGE: &str = "Usage: sinwhisky <zoom> [neighbors] [aperture: 0|1] [uniform|angular]\nReads whitespace-separated f64 samples from stdin; writes one sample per line.\nAngular mode requires neighbors=1 and aperture=1 (original range aperture).";

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|s| s == "--help" || s == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    if args.is_empty() || args.len() > 4 {
        return Err(USAGE.into());
    }
    let mut params = ResampleParams::new(args[0].parse()?);
    params.neighbors = args.get(1).map(|v| v.parse()).transpose()?.unwrap_or(1);
    let aperture = match args.get(2).map(String::as_str).unwrap_or("1") {
        "0" => false,
        "1" => true,
        _ => return Err("aperture must be 0 or 1".into()),
    };
    let angular = match args.get(3).map(String::as_str).unwrap_or("uniform") {
        "uniform" => false,
        "angular" => true,
        _ => return Err("mode must be uniform or angular".into()),
    };
    if angular && (!aperture || params.neighbors != 1) {
        return Err("angular mode requires neighbors=1 and aperture=1".into());
    }
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let samples: Vec<f64> = input
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let output = if angular {
        resample(&samples, ResampleParams::original(params.zoom))?
    } else {
        let analysis_params = if aperture {
            AnalysisParams::default()
        } else {
            AnalysisParams {
                aperture: Aperture::Disabled,
                ..AnalysisParams::default()
            }
        };
        Analysis::new(&samples, analysis_params)?.resample(params)?
    };
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for value in output {
        writeln!(stdout, "{value:.17e}")?;
    }
    stdout.flush()?;
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("sinwhisky: {error}");
        std::process::exit(2);
    }
}
