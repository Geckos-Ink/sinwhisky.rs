//! Byte counts of SWL2 streams on raw PCM files, for comparing coder variants
//! quickly. No audio is written.
//!
//! Usage:
//!   swl2_bench [--full] [--cascade off] [--tolerance 0,16,128]
//!              [--segment S] [--key K] FILE.raw...
//!   swl2_bench --parametric FILE.raw...
//!
//! Each FILE is interleaved stereo signed 16-bit little-endian PCM. The first
//! form runs `encode_auto` per file and tolerance (each on its own thread; tone
//! analysis only with `--full`), decodes every stream and checks its error
//! bound, and prints bytes per file and the totals per tolerance. The second
//! analyses each channel once and prints parametric streams (gradient groups
//! at precision 16 and 64, alone and with anchors of tolerance 256): the same
//! parameters render the same signal, so their bytes compare description coders
//! at equal SNR. Timings are wall-clock and machine-specific.
use sinwhisky::bands::{ToneAnalysis, ToneParams};
use sinwhisky::layered::{
    decode, encode, encode_auto, AnchorConfig, AutoParams, ChannelConfig, GradientConfig, Stereo,
    StreamConfig, PARAMETRIC,
};
use std::collections::BTreeMap;
use std::time::Instant;

struct Options {
    full: bool,
    cascade: bool,
    parametric: bool,
    tolerances: Vec<u32>,
    segment_exponent: u32,
    key_interval: usize,
    files: Vec<String>,
}

fn options() -> Options {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut o = Options {
        full: false,
        cascade: true,
        parametric: false,
        tolerances: vec![0, 16, 128],
        segment_exponent: 0,
        key_interval: 0,
        files: Vec::new(),
    };
    let mut i = 0;
    while i < args.len() {
        let value = || args.get(i + 1).expect("option value").as_str();
        match args[i].as_str() {
            "--full" => o.full = true,
            "--parametric" => o.parametric = true,
            "--cascade" => {
                o.cascade = value() != "off";
                i += 1;
            }
            "--tolerance" => {
                o.tolerances = value().split(',').map(|v| v.parse().unwrap()).collect();
                i += 1;
            }
            "--segment" => {
                o.segment_exponent = value().parse().unwrap();
                i += 1;
            }
            "--key" => {
                o.key_interval = value().parse().unwrap();
                i += 1;
            }
            file => o.files.push(file.to_string()),
        }
        i += 1;
    }
    o
}

fn main() {
    let o = options();
    if o.parametric {
        parametric_runs(&o.files);
        return;
    }
    let jobs: Vec<(usize, usize)> = (0..o.files.len())
        .flat_map(|f| (0..o.tolerances.len()).map(move |t| (f, t)))
        .collect();
    let results: Vec<(usize, usize, usize, String, f64)> = std::thread::scope(|scope| {
        let handles: Vec<_> = jobs
            .iter()
            .map(|&(f, t)| {
                let o = &o;
                scope.spawn(move || {
                    let (bytes, summary, seconds) = run(&o.files[f], o.tolerances[t], o);
                    (f, t, bytes, summary, seconds)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut totals = vec![(0usize, 0.0f64); o.tolerances.len()];
    for (f, t, bytes, summary, seconds) in results {
        println!(
            "{} tol={} bytes={bytes} {summary} seconds={seconds:.2}",
            o.files[f], o.tolerances[t]
        );
        totals[t].0 += bytes;
        totals[t].1 += seconds;
    }
    for (&tolerance, (bytes, seconds)) in o.tolerances.iter().zip(&totals) {
        println!("TOTAL tol={tolerance} bytes={bytes} seconds={seconds:.2}");
    }
}

fn tone_params() -> ToneParams {
    ToneParams {
        window: 2048,
        upper: 6000.0 / 44100.0 * std::f64::consts::TAU,
        max_error: 16.0,
        ..ToneParams::default()
    }
}

fn read(file: &str) -> Vec<Vec<i32>> {
    let raw = std::fs::read(file).expect("read input");
    let mut channels = vec![Vec::new(), Vec::new()];
    for (k, pair) in raw.as_chunks::<2>().0.iter().enumerate() {
        channels[k % 2].push(i32::from(i16::from_le_bytes(*pair)));
    }
    channels
}

/// One automatic stream: its bytes, a summary (stereo letter per segment, layer
/// bytes per channel) and the encoding time.
fn run(file: &str, tolerance: u32, o: &Options) -> (usize, String, f64) {
    let channels = read(file);
    let params = AutoParams {
        tolerance,
        tones: tone_params(),
        gradients: o.full,
        anchors: o.full,
        cascade: o.cascade,
        segment_exponent: o.segment_exponent,
        key_interval: o.key_interval,
    };
    let clock = Instant::now();
    let (encoded, _) = encode_auto(&channels, &params).expect("encode");
    let seconds = clock.elapsed().as_secs_f64();
    let back = decode(&encoded.bytes).expect("decode");
    assert_eq!(back, encoded.decoded, "decoder disagrees with encoder");
    for (s, b) in channels.iter().zip(&back) {
        for (&a, &v) in s.iter().zip(b) {
            assert!((a - v).unsigned_abs() <= tolerance, "error bound");
        }
    }
    let stereo: String = encoded
        .config
        .segments
        .iter()
        .map(|s| format!("{:?}", s.stereo).chars().next().unwrap())
        .collect();
    let layers: Vec<String> = encoded
        .stats
        .channels
        .iter()
        .map(|c| format!("g{}a{}", c.gradient_bytes, c.anchor_bytes))
        .collect();
    let summary = format!(
        "stereo={stereo} keys={} layers={}",
        encoded.stats.keys,
        layers.join("/")
    );
    (encoded.bytes.len(), summary, seconds)
}

fn parametric_runs(files: &[String]) {
    let analysed: Vec<(Vec<Vec<i32>>, Vec<ToneAnalysis>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = files
            .iter()
            .map(|file| {
                scope.spawn(move || {
                    let channels = read(file);
                    let analyses = std::thread::scope(|inner| {
                        let hs: Vec<_> = channels
                            .iter()
                            .map(|c| {
                                inner.spawn(move || {
                                    let x: Vec<f64> = c.iter().map(|&v| f64::from(v)).collect();
                                    ToneAnalysis::new(&x, tone_params()).expect("analysis")
                                })
                            })
                            .collect();
                        hs.into_iter().map(|h| h.join().unwrap()).collect()
                    });
                    (channels, analyses)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut totals = BTreeMap::new();
    for (file, (channels, analyses)) in files.iter().zip(&analysed) {
        for precision in [16.0, 64.0] {
            for anchors in [None, Some(256.0)] {
                let channel = ChannelConfig {
                    gradients: Some(GradientConfig {
                        lower: 0.0,
                        upper: tone_params().upper,
                        precision,
                        segment_tolerance: 2.0 * precision,
                        min_amplitude: 0.0,
                        max_jump_bins: 1.0,
                    }),
                    anchors: anchors.map(|tolerance: f64| AnchorConfig {
                        cutoff: None,
                        tolerance,
                        step: (0.25 * tolerance).max(1.0),
                        max_gap: 1024,
                    }),
                    ..ChannelConfig::default()
                };
                let config =
                    StreamConfig::single(PARAMETRIC, Stereo::Independent, vec![channel; 2]);
                let refs: Vec<_> = analyses.iter().map(Some).collect();
                let encoded = encode(channels, &[refs], &config).expect("encode");
                let back = decode(&encoded.bytes).expect("decode");
                assert_eq!(back, encoded.decoded, "decoder disagrees with encoder");
                let s = &encoded.stats;
                let g: usize = s.channels.iter().map(|c| c.gradient_bytes).sum();
                let a: usize = s.channels.iter().map(|c| c.anchor_bytes).sum();
                let m: usize = s.channels.iter().map(|c| c.members).sum();
                let label = format!("p={precision} anchors={}", anchors.is_some());
                println!(
                    "{file} {label} bytes={} gradient={g} anchor={a} members={m} snr={:.3}",
                    s.total_bytes, s.snr_db
                );
                let t = totals
                    .entry(label)
                    .or_insert((0usize, 0usize, 0usize, 0.0f64));
                t.0 += s.total_bytes;
                t.1 += g;
                t.2 += a;
                t.3 += s.snr_db;
            }
        }
    }
    for (label, (bytes, g, a, snr)) in totals {
        println!("TOTAL {label} bytes={bytes} gradient={g} anchor={a} snr_sum={snr:.3}");
    }
}
