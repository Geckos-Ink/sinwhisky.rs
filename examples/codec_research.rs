use sinwhisky::codec::ResearchFrame;
use sinwhisky::{Analysis, AnalysisParams, ResampleParams};

fn main() -> Result<(), sinwhisky::Error> {
    let samples: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.025).sin()).collect();
    let analysis = Analysis::new(&samples, AnalysisParams::default())?;
    let upsampled = analysis.resample(ResampleParams::new(3))?;
    for tolerance in [0.0, 0.001, 0.01, 0.05] {
        let frame = ResearchFrame::from_analysis(&analysis, 44_100, tolerance)?;
        println!(
            "tolerance={tolerance} segments={} max_error={:.8} rms={:.8}",
            frame.segments().len(),
            frame.metrics().max_absolute,
            frame.metrics().root_mean_square
        );
        assert_eq!(frame.decode()?.len(), samples.len());
    }
    println!(
        "Reused {} fitted models for {} output samples and all segment trials.",
        analysis.models().len(),
        upsampled.len()
    );
    println!("Segment counts are not encoded sizes or compression ratios.");
    Ok(())
}
