//! Reference vectors for the ports in `ports/` (WebAssembly SIMD, WebGPU): JSON on stdout.
//! Usage: cargo run --offline --example port_vectors > ports/fixtures/vectors.json
//!
//! Per case: the source samples; the original-angular resampler at several zooms
//! (`ResampleParams::original`); the AFT's halves (`AngularFrame`, original analysis);
//! its unnormalized integral at arbitrary frequencies (`AngularFrame::fourier`); FAFT
//! values on the same curve, rectangular and Hann (`CircleCurve::from_angular`); and
//! Hann peaks with default parameters. Deterministic: no clock, no randomness source.
use sinwhisky::aft::AngularFrame;
use sinwhisky::faft::{CircleCurve, PeakParams, Weighting};
use sinwhisky::spectral::Complex;
use sinwhisky::{resample, AnalysisParams, ResampleParams};
use std::f64::consts::PI;
use std::fmt::Write;

const ZOOMS: [usize; 3] = [1, 3, 7];

/// A small linear congruential generator, so the noise is the same everywhere.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn cases() -> Vec<(&'static str, Vec<f64>)> {
    let mut noise = Lcg(0x5157_4b59);
    let tones: Vec<f64> = (0..256)
        .map(|i| {
            let t = i as f64;
            0.8 * (0.3137 * t + 0.4).sin() + 0.3 * (1.9021 * t).cos() + 0.01 * noise.next()
        })
        .collect();
    let pilot: Vec<f64> = (0..480)
        .map(|i| {
            let t = i as f64;
            let hann = 0.5 - 0.5 * (2.0 * PI * t / 480.0).cos();
            hann * (2.2647 * t).sin() + 0.002 * noise.next()
        })
        .collect();
    // Flat runs, a straight ramp and a step: line halves and rejected circles.
    let mut lines = vec![0.0; 12];
    lines.extend((0..12).map(|i| i as f64 * 0.25));
    lines.extend([3.0, 3.0, -1.0, 2.0, 2.0, 2.0, 0.5, -0.5]);
    let pcm: Vec<f64> = (0..300).map(|_| (noise.next() * 12000.0).round()).collect();
    vec![
        ("tones", tones),
        ("pilot", pilot),
        ("lines", lines),
        ("pcm16", pcm),
        ("short", vec![0.0, 1.0, 0.0, -1.0, 0.5]),
        ("paper", vec![3.0, 7.0, 9.0, 4.0, 6.0]),
    ]
}

fn list(out: &mut String, values: impl IntoIterator<Item = f64>) {
    out.push('[');
    for (i, v) in values.into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write!(out, "{v:?}").unwrap();
    }
    out.push(']');
}

fn complexes(out: &mut String, values: &[Complex]) {
    list(out, values.iter().flat_map(|z| [z.re, z.im]));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = String::from("{\"source\":\"sinwhisky.rs examples/port_vectors.rs\",\"cases\":[");
    for (index, (name, samples)) in cases().into_iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let n = samples.len() as f64;
        write!(out, "{{\"name\":\"{name}\",\"samples\":")?;
        list(&mut out, samples.iter().copied());
        out.push_str(",\"resample\":[");
        for (i, zoom) in ZOOMS.into_iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write!(out, "{{\"zoom\":{zoom},\"output\":")?;
            list(
                &mut out,
                resample(&samples, ResampleParams::original(zoom))?,
            );
            out.push('}');
        }
        out.push(']');
        let frame = AngularFrame::new(&samples, AnalysisParams::original())?;
        let stats = frame.stats();
        write!(
            out,
            ",\"stats\":{{\"circleHalves\":{},\"lineHalves\":{},\"rejectedCircleHalves\":{}}}",
            stats.circle_halves, stats.line_halves, stats.rejected_circle_halves
        )?;
        // Arbitrary frequencies: bins, between bins, DC, near and above pi.
        let omegas: Vec<f64> = [0.0, 0.05, 0.3137, 1.0, 1.9021, 2.2647, 3.0, PI, 4.0, -0.7]
            .into_iter()
            .chain((0..8).map(|k| 2.0 * PI * (k as f64 + 0.37) / n))
            .collect();
        out.push_str(",\"omegas\":");
        list(&mut out, omegas.iter().copied());
        let integrals: Vec<Complex> = omegas
            .iter()
            .map(|&w| frame.fourier(w))
            .collect::<Result<_, _>>()?;
        out.push_str(",\"aftIntegral\":");
        complexes(&mut out, &integrals);
        let curve = CircleCurve::from_angular(&frame)?;
        let window = 0..samples.len();
        let rect = curve.transform(window.clone(), Weighting::Rectangular)?;
        let hann = curve.transform(window, Weighting::Hann)?;
        out.push_str(",\"faftRect\":");
        complexes(&mut out, &rect.values(&omegas)?);
        out.push_str(",\"faftHann\":");
        complexes(&mut out, &hann.values(&omegas)?);
        let position_count = (n * 4.0) as usize;
        out.push_str(",\"curve\":");
        list(
            &mut out,
            (0..position_count).map(|k| frame.evaluate(k as f64 * 0.25 + 0.1).unwrap()),
        );
        if samples.len() >= 16 {
            let search = hann.peaks(PeakParams::default())?;
            out.push_str(",\"peaks\":");
            list(
                &mut out,
                search
                    .peaks
                    .iter()
                    .flat_map(|p| [p.omega, p.value.re, p.value.im, p.curvature]),
            );
        }
        out.push('}');
    }
    out.push_str("]}\n");
    print!("{out}");
    Ok(())
}
