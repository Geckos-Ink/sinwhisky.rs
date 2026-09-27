//! Two explicitly distinct interpolation contracts sharing one analysis representation.
use crate::{buffer, Analysis, AnalysisParams, Error, Primitive};
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// Portable C/Python semantics: one y(x) branch, generalized cosine window.
    UniformX,
    /// Recovered 2014 concept: equal angles and the finite-zoom sine window.
    /// Uses adjacent triplets only and repairs degenerate inputs with lines.
    OriginalAngular,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResampleParams {
    pub zoom: usize,
    pub neighbors: usize,
    pub interpolation: Interpolation,
}
impl ResampleParams {
    pub fn new(zoom: usize) -> Self {
        Self {
            zoom,
            neighbors: 1,
            interpolation: Interpolation::UniformX,
        }
    }
    pub fn original(zoom: usize) -> Self {
        Self {
            interpolation: Interpolation::OriginalAngular,
            ..Self::new(zoom)
        }
    }
    pub fn output_len(self, input_len: usize) -> Result<usize, Error> {
        if self.neighbors == 0 {
            return Err(Error::InvalidParameter("neighbors"));
        }
        if self.interpolation == Interpolation::OriginalAngular && self.neighbors != 1 {
            return Err(Error::InvalidParameter(
                "angular interpolation requires neighbors=1",
            ));
        }
        if input_len < 2 {
            return Ok(input_len);
        }
        self.zoom
            .checked_add(1)
            .and_then(|step| (input_len - 1).checked_mul(step))
            .and_then(|n| n.checked_add(1))
            .ok_or(Error::OutputTooLarge)
    }
}

/// Convenience path; use Analysis directly to reuse circles across zooms or codec trials.
pub fn resample(samples: &[f64], params: ResampleParams) -> Result<Vec<f64>, Error> {
    let analysis_params = match params.interpolation {
        Interpolation::UniformX => AnalysisParams::default(),
        Interpolation::OriginalAngular => AnalysisParams::original(),
    };
    Analysis::new(samples, analysis_params)?.resample(params)
}

impl Analysis {
    pub fn resample(&self, params: ResampleParams) -> Result<Vec<f64>, Error> {
        let samples = self.samples();
        let len = params.output_len(samples.len())?;
        let mut output = buffer(len)?;
        if samples.len() < 2 || params.zoom == 0 {
            output.extend_from_slice(samples);
            return Ok(output);
        }
        let q = params.zoom + 1;
        let support = params.neighbors as f64;
        for (i, pair) in samples.windows(2).enumerate() {
            output.push(pair[0]);
            for k in 1..q {
                let t = k as f64 / q as f64;
                let position = i as f64 + t;
                let lo = i.saturating_sub(params.neighbors - 1).max(1);
                let hi = i
                    .saturating_add(params.neighbors)
                    .min(samples.len().saturating_sub(2));
                let mut sum_w = 0.0;
                let mut sum_y = 0.0;
                for anchor in lo..=hi {
                    let model = &self.models()[anchor - 1];
                    let u = position - anchor as f64;
                    if u.abs() >= support {
                        continue;
                    }
                    let prediction = match (params.interpolation, model.primitive) {
                        (Interpolation::UniformX, Primitive::Circle(c)) => {
                            c.evaluate_uniform(position)
                        }
                        (Interpolation::UniformX, Primitive::Line { .. }) => None,
                        (Interpolation::OriginalAngular, Primitive::Circle(c)) => {
                            c.evaluate_angular(position)
                        }
                        (Interpolation::OriginalAngular, Primitive::Line { value, slope }) => {
                            let y = value + u * slope;
                            y.is_finite().then_some(y)
                        }
                    };
                    if let Some(y) = prediction {
                        let weight = match params.interpolation {
                            Interpolation::UniformX => (PI * u / (2.0 * support)).cos(),
                            Interpolation::OriginalAngular => {
                                (PI * (q as f64 / (q as f64 + 1.0)) * u / 2.0).cos()
                            }
                        };
                        if weight > 0.0 {
                            sum_w += weight;
                            sum_y += weight * y;
                        }
                    }
                }
                let blended = sum_y / sum_w;
                let y = if sum_w > 1e-20 && blended.is_finite() {
                    blended
                } else {
                    (1.0 - t) * pair[0] + t * pair[1]
                };
                output.push(y);
            }
        }
        output.push(samples[samples.len() - 1]);
        Ok(output)
    }
}
