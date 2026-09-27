//! Experimental in-memory segment representation, NOT a compressed bitstream.
//! Reuses analyzed circles as uniform-x predictors and verifies every covered sample.
use crate::{buffer, Analysis, Error, LocalCircle, Primitive};
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SegmentModel {
    Circle(LocalCircle),
    Line {
        origin: usize,
        value: f64,
        slope: f64,
    },
    Literal(f64),
}
impl SegmentModel {
    pub fn predict(&self, index: usize) -> Option<f64> {
        match *self {
            Self::Circle(c) => c.evaluate_uniform(index as f64),
            Self::Line {
                origin,
                value,
                slope,
            } => {
                let y = value + (index as f64 - origin as f64) * slope;
                y.is_finite().then_some(y)
            }
            Self::Literal(value) => value.is_finite().then_some(value),
        }
    }
}

/// Half-open coverage in original sample units. No gaps, overlaps, or implicit zero fill.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    range: Range<usize>,
    model: SegmentModel,
}
impl Segment {
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
    pub fn model(&self) -> SegmentModel {
        self.model
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ErrorMetrics {
    pub max_absolute: f64,
    pub root_mean_square: f64,
}
impl ErrorMetrics {
    pub fn between(reference: &[f64], actual: &[f64]) -> Result<Self, Error> {
        if reference.len() != actual.len() {
            return Err(Error::LengthMismatch);
        }
        let mut result = Self::default();
        let mut norm: f64 = 0.0;
        for (i, (&a, &b)) in reference.iter().zip(actual).enumerate() {
            let error = (a - b).abs();
            if !a.is_finite() || !b.is_finite() || !error.is_finite() {
                return Err(Error::NonFiniteSample(i));
            }
            result.max_absolute = result.max_absolute.max(error);
            // Scaling first keeps the sum of squares from overflowing on large signals.
            norm = norm.hypot(error / (reference.len() as f64).sqrt());
        }
        result.root_mean_square = norm;
        Ok(result)
    }
}

/// Mono research frame. Metadata and models are retained; original PCM is not.
/// There is no quantization, entropy coding, residual stream, or rate guarantee yet.
#[derive(Debug, Clone)]
pub struct ResearchFrame {
    sample_rate: u32,
    sample_count: usize,
    segments: Vec<Segment>,
    metrics: ErrorMetrics,
}
impl ResearchFrame {
    /// Greedily extend either a line or the already-fitted circle at start+1.
    /// Select longer valid coverage, favoring a line on ties. Literal fallback always progresses.
    /// Error is measured at source samples, not at arbitrary fractional positions.
    pub fn from_analysis(
        analysis: &Analysis,
        sample_rate: u32,
        max_error: f64,
    ) -> Result<Self, Error> {
        if sample_rate == 0 {
            return Err(Error::InvalidParameter("sample_rate"));
        }
        if !max_error.is_finite() || max_error < 0.0 {
            return Err(Error::InvalidParameter("max_error"));
        }
        let samples = analysis.samples();
        let mut segments = buffer(samples.len())?;
        let mut start = 0;
        while start < samples.len() {
            let mut model = SegmentModel::Literal(samples[start]);
            let mut end = start + 1;
            if start + 1 < samples.len() {
                let line = SegmentModel::Line {
                    origin: start,
                    value: samples[start],
                    slope: samples[start + 1] - samples[start],
                };
                let line_end = covered_until(line, samples, start, max_error);
                if line_end > end {
                    model = line;
                    end = line_end;
                }
            }
            // models[start] is the interior triplet centered at start+1.
            if let Some(local) = analysis.models().get(start) {
                if let Primitive::Circle(circle) = local.primitive {
                    let candidate = SegmentModel::Circle(circle);
                    let circle_end = covered_until(candidate, samples, start, max_error);
                    if circle_end >= start + 3 && circle_end > end {
                        model = candidate;
                        end = circle_end;
                    }
                }
            }
            segments.push(Segment {
                range: start..end,
                model,
            });
            start = end;
        }
        let mut frame = Self {
            sample_rate,
            sample_count: samples.len(),
            segments,
            metrics: ErrorMetrics::default(),
        };
        frame.metrics = ErrorMetrics::between(samples, &frame.decode()?)?;
        Ok(frame)
    }
    pub fn decode(&self) -> Result<Vec<f64>, Error> {
        let mut samples = buffer(self.sample_count)?;
        for segment in &self.segments {
            for i in segment.range.clone() {
                samples.push(segment.model.predict(i).ok_or(Error::NonFiniteSample(i))?);
            }
        }
        Ok(samples)
    }
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }
    pub fn metrics(&self) -> ErrorMetrics {
        self.metrics
    }
}

fn covered_until(model: SegmentModel, samples: &[f64], start: usize, tolerance: f64) -> usize {
    let mut end = start;
    while end < samples.len() {
        match model.predict(end) {
            Some(y) if (y - samples[end]).abs() <= tolerance => end += 1,
            _ => break,
        }
    }
    end
}
