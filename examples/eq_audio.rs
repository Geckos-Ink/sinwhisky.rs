//! Band division, equalization, extrusion and layered coding of real audio
//! (PAPER_EQ_COMPRESSION.md), for listening and byte counts.
//!
//! Usage:
//!   eq_audio COMMAND INPUT.raw CHANNELS RATE OUTPUT_DIR [options]
//!
//! INPUT.raw is interleaved signed 16-bit little-endian PCM (as FFmpeg writes with
//! `-f s16le`). COMMAND is `divide`, `eq`, `extrude`, `compress` or `all`.
//! OUTPUT_DIR must not exist and must be outside this repository; it receives WAV
//! files (PCM16) and `report.json`, which is also printed. Channels are processed
//! independently. Options (frequencies in Hz):
//!   --edges 250,2000      band edges for divide/eq
//!   --gains-db 3,0,-6     eq gains, one per band
//!   --band 40,250         extruded band (0 and the Nyquist rate are open ends)
//!   --crossover brick|cosine:HZ   tone crossover (default brick)
//!   --order 4             residual Butterworth order (zero phase: |H|^2)
//!   --window 2048         analysis window T (hop T/2)
//!   --upper 2000          highest analysed (tonal) frequency
//!   --max-error 16        circle-curve tolerance, PCM16 units
//!   --tolerance 0,16      compress: error bounds to encode (one stream each)
//!   --precision 4         compress: tonal parameter precision, PCM16 units
//! Timings are wall-clock on this machine and are reported, not asserted.
use sinwhisky::bands::{divide, equalize, extrude, BandSplit, Crossover, ToneAnalysis, ToneParams};
use sinwhisky::layered::{decode, encode, LayeredParams};
use std::f64::consts::{PI, TAU};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Options {
    edges: Vec<f64>,
    gains_db: Vec<f64>,
    band: (f64, f64),
    crossover: Option<f64>,
    order: usize,
    window: usize,
    upper: f64,
    max_error: f64,
    tolerances: Vec<u32>,
    precision: f64,
}

fn list<T: std::str::FromStr>(text: &str) -> Result<Vec<T>, String> {
    text.split(',')
        .map(|v| {
            v.trim()
                .parse()
                .map_err(|_| format!("bad number in {text:?}"))
        })
        .collect()
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        edges: vec![250.0, 2000.0],
        gains_db: Vec::new(),
        band: (40.0, 250.0),
        crossover: None,
        order: 4,
        window: 2048,
        upper: 2000.0,
        max_error: 16.0,
        tolerances: vec![0, 16],
        precision: 4.0,
    };
    let mut i = 0;
    while i < args.len() {
        let value = args
            .get(i + 1)
            .ok_or(format!("{} needs a value", args[i]))?;
        let one = |v: &str| v.parse::<f64>().map_err(|_| format!("bad number {v:?}"));
        match args[i].as_str() {
            "--edges" => o.edges = list(value)?,
            "--gains-db" => o.gains_db = list(value)?,
            "--band" => {
                let b: Vec<f64> = list(value)?;
                if b.len() != 2 {
                    return Err("--band needs LO,HI".into());
                }
                o.band = (b[0], b[1]);
            }
            "--crossover" => {
                o.crossover = match value.as_str() {
                    "brick" => None,
                    v if v.starts_with("cosine:") => Some(one(&v[7..])?),
                    _ => return Err("--crossover is brick or cosine:HZ".into()),
                }
            }
            "--order" => o.order = value.parse().map_err(|_| "bad --order")?,
            "--window" => o.window = value.parse().map_err(|_| "bad --window")?,
            "--upper" => o.upper = one(value)?,
            "--max-error" => o.max_error = one(value)?,
            "--tolerance" => o.tolerances = list(value)?,
            "--precision" => o.precision = one(value)?,
            other => return Err(format!("unknown option {other}")),
        }
        i += 2;
    }
    Ok(o)
}

fn write_wav(path: &Path, channels: &[Vec<f64>], rate: u32) -> Result<usize, String> {
    let frames = channels[0].len();
    let data_len = frames * channels.len() * 2;
    let mut bytes = Vec::with_capacity(44 + data_len);
    let block = channels.len() as u16 * 2;
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&(channels.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&rate.to_le_bytes());
    bytes.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
    bytes.extend_from_slice(&block.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(data_len as u32).to_le_bytes());
    let mut clipped = 0;
    for i in 0..frames {
        for ch in channels {
            let v = ch[i].round();
            if !(-32768.0..=32767.0).contains(&v) {
                clipped += 1;
            }
            bytes.extend_from_slice(&(v.clamp(-32768.0, 32767.0) as i16).to_le_bytes());
        }
    }
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(clipped)
}

fn rms(x: &[f64]) -> f64 {
    (x.iter().map(|v| v * v).sum::<f64>() / x.len().max(1) as f64).sqrt()
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 5 {
        return Err("usage: eq_audio COMMAND INPUT.raw CHANNELS RATE OUTPUT_DIR [options]".into());
    }
    let command = args[0].as_str();
    if !["divide", "eq", "extrude", "compress", "all"].contains(&command) {
        return Err(format!("unknown command {command}"));
    }
    let channels: usize = args[2].parse().map_err(|_| "bad CHANNELS")?;
    let rate: u32 = args[3].parse().map_err(|_| "bad RATE")?;
    let output = PathBuf::from(&args[4]);
    let o = parse(&args[5..])?;
    if !(1..=8).contains(&channels) || rate == 0 {
        return Err("CHANNELS must be 1..=8 and RATE positive".into());
    }
    if output.exists() {
        return Err(format!("output {} already exists", output.display()));
    }
    let parent = output.parent().filter(|p| !p.as_os_str().is_empty());
    let parent = parent
        .unwrap_or(Path::new("."))
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .canonicalize()
        .map_err(|e| e.to_string())?;
    if parent.starts_with(&repo) {
        return Err("output must be outside the repository (audio stays private)".into());
    }
    let raw = std::fs::read(&args[1]).map_err(|e| format!("{}: {e}", args[1]))?;
    let frames = raw.len() / (2 * channels);
    if frames < 64 {
        return Err("input is too short".into());
    }
    let pcm: Vec<Vec<i32>> = (0..channels)
        .map(|c| {
            (0..frames)
                .map(|i| {
                    let at = 2 * (i * channels + c);
                    i32::from(i16::from_le_bytes([raw[at], raw[at + 1]]))
                })
                .collect()
        })
        .collect();
    let samples: Vec<Vec<f64>> = pcm
        .iter()
        .map(|c| c.iter().map(|&v| f64::from(v)).collect())
        .collect();
    std::fs::create_dir(&output).map_err(|e| e.to_string())?;
    let hz = |f: f64| (TAU * f / f64::from(rate)).min(PI);
    let params = ToneParams {
        window: o.window,
        upper: hz(o.upper),
        max_error: o.max_error,
        ..ToneParams::default()
    };
    let crossover = match o.crossover {
        None => Crossover::BrickWall,
        Some(w) => Crossover::RaisedCosine { half_width: hz(w) },
    };
    let mut report = String::new();
    let e = |e: sinwhisky::Error| e.to_string();
    write!(
        report,
        "{{\"command\":\"{command}\",\"channels\":{channels},\"rate\":{rate},\"frames\":{frames},\
         \"seconds\":{:.3},\"window\":{},\"upper_hz\":{},\"max_error\":{}",
        frames as f64 / f64::from(rate),
        o.window,
        o.upper,
        o.max_error
    )
    .unwrap();

    let started = Instant::now();
    let mut analyses = Vec::new();
    for x in &samples {
        analyses.push(ToneAnalysis::new(x, params).map_err(e)?);
    }
    let analysis_seconds = started.elapsed().as_secs_f64();
    let stats: Vec<_> = analyses.iter().map(|a| a.stats()).collect();
    write!(
        report,
        ",\"analysis\":{{\"seconds\":{analysis_seconds:.3},\"tones_per_frame\":{:.2},\
         \"rotations\":{},\"explained\":[{}],\"curve_max_error\":{:.3}}}",
        stats.iter().map(|s| s.tones).sum::<usize>() as f64
            / stats.iter().map(|s| s.frames).sum::<usize>().max(1) as f64,
        stats.iter().map(|s| s.rotations).sum::<usize>(),
        analyses
            .iter()
            .map(|a| format!("{:.5}", a.explained()))
            .collect::<Vec<_>>()
            .join(","),
        stats.iter().map(|s| s.curve_max_error).fold(0.0, f64::max)
    )
    .unwrap();

    if matches!(command, "divide" | "eq" | "all") {
        let started = Instant::now();
        let split = BandSplit::new(o.edges.iter().map(|&f| hz(f)).collect(), crossover, o.order)
            .map_err(e)?;
        let divisions: Vec<_> = samples
            .iter()
            .zip(&analyses)
            .map(|(x, a)| divide(x, a, &split))
            .collect::<Result<_, _>>()
            .map_err(e)?;
        let seconds = started.elapsed().as_secs_f64();
        let mut completeness: f64 = 0.0;
        for (x, d) in samples.iter().zip(&divisions) {
            for (i, &v) in x.iter().enumerate() {
                let sum: f64 = (0..d.bands())
                    .map(|b| d.tonal[b][i] + d.residual[b][i])
                    .sum();
                completeness = completeness.max((sum - v).abs());
            }
        }
        let mut bands = Vec::new();
        for b in 0..split.bands() {
            let tracks: Vec<Vec<f64>> = divisions.iter().map(|d| d.band(b)).collect();
            let clipped = write_wav(&output.join(format!("band-{}.wav", b + 1)), &tracks, rate)?;
            let tonal: Vec<f64> = divisions.iter().flat_map(|d| d.tonal[b].clone()).collect();
            let all: Vec<f64> = tracks.concat();
            bands.push(format!(
                "{{\"rms\":{:.2},\"tonal_rms\":{:.2},\"clipped\":{clipped}}}",
                rms(&all),
                rms(&tonal)
            ));
        }
        let tonal: Vec<Vec<f64>> = divisions
            .iter()
            .map(|d| {
                (0..d.tonal[0].len())
                    .map(|i| d.tonal.iter().map(|t| t[i]).sum())
                    .collect()
            })
            .collect();
        let residual: Vec<Vec<f64>> = samples
            .iter()
            .zip(&tonal)
            .map(|(x, t)| x.iter().zip(t).map(|(a, b)| a - b).collect())
            .collect();
        write_wav(&output.join("tonal.wav"), &tonal, rate)?;
        write_wav(&output.join("residual.wav"), &residual, rate)?;
        write!(
            report,
            ",\"divide\":{{\"edges_hz\":{:?},\"seconds\":{seconds:.3},\"max_sum_error\":{completeness:.3e},\"bands\":[{}]}}",
            o.edges,
            bands.join(",")
        )
        .unwrap();
        if matches!(command, "eq" | "all") {
            let gains_db = if o.gains_db.is_empty() {
                // Default: a bass boost and treble cut, audible at a glance.
                (0..split.bands())
                    .map(|b| {
                        if b == 0 {
                            6.0
                        } else if b + 1 == split.bands() {
                            -12.0
                        } else {
                            0.0
                        }
                    })
                    .collect()
            } else {
                o.gains_db.clone()
            };
            let gains: Vec<f64> = gains_db.iter().map(|g| 10f64.powf(g / 20.0)).collect();
            let eq: Vec<Vec<f64>> = divisions
                .iter()
                .map(|d| equalize(d, &gains))
                .collect::<Result<_, _>>()
                .map_err(e)?;
            let clipped = write_wav(&output.join("equalized.wav"), &eq, rate)?;
            write!(
                report,
                ",\"eq\":{{\"gains_db\":{gains_db:?},\"clipped\":{clipped}}}"
            )
            .unwrap();
        }
    }

    if matches!(command, "extrude" | "all") {
        let started = Instant::now();
        let (lo, hi) = (hz(o.band.0), hz(o.band.1));
        let mut isolated = Vec::new();
        let mut remainder = Vec::new();
        for (x, a) in samples.iter().zip(&analyses) {
            let (i, r) = extrude(x, a, lo, hi, crossover, o.order).map_err(e)?;
            isolated.push(i);
            remainder.push(r);
        }
        write_wav(&output.join("isolated.wav"), &isolated, rate)?;
        write_wav(&output.join("remainder.wav"), &remainder, rate)?;
        write!(
            report,
            ",\"extrude\":{{\"band_hz\":[{},{}],\"seconds\":{:.3},\"isolated_rms\":{:.2},\"remainder_rms\":{:.2}}}",
            o.band.0,
            o.band.1,
            started.elapsed().as_secs_f64(),
            rms(&isolated.concat()),
            rms(&remainder.concat())
        )
        .unwrap();
    }

    if matches!(command, "compress" | "all") {
        let pcm_bytes = frames * channels * 2;
        let mut runs = Vec::new();
        for &tolerance in &o.tolerances {
            for tonal in [true, false] {
                let started = Instant::now();
                let params = LayeredParams {
                    tolerance,
                    tone_precision: o.precision,
                    tonal_upper: hz(o.upper),
                    ..LayeredParams::default()
                };
                let mut total = 0;
                let (mut tonal_bytes, mut correction_bytes, mut tracks, mut breakpoints) =
                    (0, 0, 0, 0);
                let (mut worst, mut written, mut residual) = (0, 0, 0.0);
                let mut decoded = Vec::new();
                for (c, (x, a)) in pcm.iter().zip(&analyses).enumerate() {
                    let encoded = encode(x, tonal.then_some(a), params).map_err(e)?;
                    let back = decode(&encoded.bytes).map_err(e)?;
                    if back != encoded.decoded {
                        return Err("decoder did not reproduce the encoder".into());
                    }
                    let s = encoded.stats;
                    if s.max_error > tolerance {
                        return Err("error bound violated".into());
                    }
                    if tonal {
                        let name = format!("tolerance-{tolerance}-channel-{}.swl1", c + 1);
                        std::fs::write(output.join(name), &encoded.bytes)
                            .map_err(|e| e.to_string())?;
                    }
                    total += s.total_bytes;
                    tonal_bytes += s.tonal_bytes;
                    correction_bytes += s.correction_bytes;
                    tracks += s.tracks;
                    breakpoints += s.breakpoints;
                    worst = worst.max(s.max_error);
                    written += s.written;
                    residual += s.residual_rms * s.residual_rms * s.samples as f64;
                    decoded.push(back.iter().map(|&v| f64::from(v)).collect::<Vec<f64>>());
                }
                if tonal {
                    write_wav(
                        &output.join(format!("decoded-tolerance-{tolerance}.wav")),
                        &decoded,
                        rate,
                    )?;
                }
                runs.push(format!(
                    "{{\"tolerance\":{tolerance},\"tonal_layer\":{tonal},\"bytes\":{total},\"percent_of_pcm\":{:.2},\
                     \"kbps\":{:.1},\"tonal_bytes\":{tonal_bytes},\"correction_bytes\":{correction_bytes},\
                     \"tracks\":{tracks},\"breakpoints\":{breakpoints},\"written_fraction\":{:.4},\
                     \"residual_rms\":{:.2},\"max_error\":{worst},\"seconds\":{:.3}}}",
                    100.0 * total as f64 / pcm_bytes as f64,
                    8.0 * total as f64 / (frames as f64 / f64::from(rate)) / 1000.0,
                    written as f64 / (frames * channels) as f64,
                    (residual / (frames * channels) as f64).sqrt(),
                    started.elapsed().as_secs_f64()
                ));
            }
        }
        write!(
            report,
            ",\"compress\":{{\"pcm_bytes\":{pcm_bytes},\"precision\":{},\"runs\":[{}]}}",
            o.precision,
            runs.join(",")
        )
        .unwrap();
    }
    report.push('}');
    std::fs::write(output.join("report.json"), format!("{report}\n")).map_err(|e| e.to_string())?;
    println!("{report}");
    Ok(())
}

fn main() {
    if let Err(message) = run() {
        eprintln!("eq_audio: {message}");
        std::process::exit(2);
    }
}
