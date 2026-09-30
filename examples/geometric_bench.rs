//! Numeric frames to geometry-only spectra, predictions, timing and private models.
//! Usage: geometric_bench N MAX_ERROR MAX_SPAN PHASE_SEARCH MIN_BITS ZOOM REPEATS [PREVALIDATE]
use sinwhisky::geometric_faft::{GeometryFrame, GeometryParams, WavePrimitive};
use std::hint::black_box;
use std::io::{self, Read, Write};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if !(7..=8).contains(&a.len()) {
        return Err(
            "usage: geometric_bench N MAX_ERROR MAX_SPAN PHASE_SEARCH MIN_BITS ZOOM REPEATS [PREVALIDATE]".into(),
        );
    }
    let n: usize = a[0].parse()?;
    let params = GeometryParams {
        max_error: a[1].parse()?,
        max_span: a[2].parse()?,
        search_phases: a[3].parse::<u8>()? != 0,
        min_precision: a[4].parse()?,
        prevalidate_geometry: a.get(7).map(|s| s == "1").unwrap_or(false),
    };
    let zoom: usize = a[5].parse()?;
    let repeats: usize = a[6].parse()?;
    if !(2..=4096).contains(&n) || !(1..=11).contains(&repeats) {
        return Err("invalid benchmark sizes".into());
    }
    let mut data = String::new();
    io::stdin().read_to_string(&mut data)?;
    let samples: Vec<f64> = data
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if samples.is_empty() || !samples.len().is_multiple_of(n) {
        return Err("expected complete frames".into());
    }
    let mut output = io::BufWriter::new(io::stdout().lock());
    for x in samples.chunks_exact(n) {
        let frame = GeometryFrame::new(x, params)?;
        black_box(frame.spectrum_rotations(zoom)?);
        let mut fits = Vec::new();
        let mut closed_times = Vec::new();
        let mut rotations = Vec::new();
        let mut queries = Vec::new();
        for _ in 0..repeats {
            let start = Instant::now();
            black_box(GeometryFrame::new(black_box(x), params)?);
            fits.push(start.elapsed().as_nanos());
            let start = Instant::now();
            black_box(frame.spectrum_closed(zoom)?);
            closed_times.push(start.elapsed().as_nanos());
            let start = Instant::now();
            black_box(frame.spectrum_rotations(zoom)?);
            rotations.push(start.elapsed().as_nanos());
            let start = Instant::now();
            for bin in [0, 1, 2, 4, 8, 16, 32, 64, 128] {
                black_box(frame.dtft(std::f64::consts::TAU * bin as f64 / n as f64, 0..n)?);
            }
            queries.push(start.elapsed().as_nanos());
        }
        let stats = frame.stats();
        write!(output,"{{\"fit_ns\":{fits:?},\"closed_ns\":{closed_times:?},\"rotation_ns\":{rotations:?},\"nine_query_ns\":{queries:?},\"circles\":{},\"lines\":{},\"pieces\":{},\"candidates\":{},\"max_error\":{},\"rmse\":{},\"estimated_model_bits\":{},\"scale\":{},\"models\":[",stats.circles,stats.lines,stats.angular_pieces,stats.candidates_tested,stats.max_error,stats.rmse,stats.estimated_model_bits,frame.scale())?;
        for (i, s) in frame.segments().iter().enumerate() {
            if i > 0 {
                write!(output, ",")?;
            }
            write!(
                output,
                "{{\"start\":{},\"end\":{},",
                s.range().start,
                s.range().end
            )?;
            match s.primitive() {
                WavePrimitive::Line{first,last,precision}=>write!(output,"\"kind\":\"line\",\"first\":{first},\"last\":{last},\"bits\":{precision}}}")?,
                WavePrimitive::Circle(c)=>write!(output,"\"kind\":\"circle\",\"cx\":{},\"cy\":{},\"radius\":{},\"x_scale\":{},\"angles\":{:?},\"middle\":{},\"bits\":{},\"gain\":{}}}",c.center.x,c.center.y,c.radius,c.x_scale,c.angles,c.middle,c.precision,c.aperture_gain)?,
            }
        }
        write!(output, "],\"closed\":[")?;
        for (i, z) in frame.spectrum_closed(zoom)?.iter().enumerate() {
            if i > 0 {
                write!(output, ",")?;
            }
            write!(output, "[{},{}]", z.re, z.im)?;
        }
        write!(output, "],\"rotations\":[")?;
        for (i, z) in frame.spectrum_rotations(zoom)?.iter().enumerate() {
            if i > 0 {
                write!(output, ",")?;
            }
            write!(output, "[{},{}]", z.re, z.im)?;
        }
        write!(output, "],\"decoded\":{:?},\"dense\":[", frame.decode()?)?;
        for j in 0..n * zoom {
            if j > 0 {
                write!(output, ",")?;
            }
            write!(output, "{}", frame.evaluate(j as f64 / zoom as f64)?)?;
        }
        writeln!(output, "]}}")?;
    }
    Ok(())
}
