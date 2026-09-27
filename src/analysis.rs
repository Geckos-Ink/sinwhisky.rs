//! Fit once, retain the original samples and inspect/reuse every interior model.
use crate::geometry::{arc_angles, Circle, Point};
use crate::{buffer, Error};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Aperture {
    Disabled,
    /// Original range-based aperture. Constant triplets use 1 to avoid zero x-spacing.
    SignalRange,
    Adaptive {
        gain: f64,
        min: f64,
        max: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnalysisParams {
    pub aperture: Aperture,
    pub collinear_epsilon: f64,
}
impl Default for AnalysisParams {
    fn default() -> Self {
        Self {
            aperture: Aperture::Adaptive {
                gain: 1.0,
                min: 1.0,
                max: 12.0,
            },
            collinear_epsilon: 1e-10,
        }
    }
}
impl AnalysisParams {
    /// Original aperture and determinant tolerance, expressed in the 2x2 system.
    pub fn original() -> Self {
        Self {
            aperture: Aperture::SignalRange,
            collinear_epsilon: 4e-6,
        }
    }
    fn validate(self) -> Result<(), Error> {
        if !self.collinear_epsilon.is_finite() || self.collinear_epsilon <= 0.0 {
            return Err(Error::InvalidParameter("collinear_epsilon"));
        }
        if let Aperture::Adaptive { gain, min, max } = self.aperture {
            if !gain.is_finite()
                || gain < 0.0
                || !min.is_finite()
                || min <= 0.0
                || !max.is_finite()
                || max < min
            {
                return Err(Error::InvalidParameter("aperture"));
            }
        }
        Ok(())
    }
}

/// A reusable circle descriptor; anchor is an original sample index, not geometric cx.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalCircle {
    pub anchor: usize,
    pub aperture: f64,
    pub circle: Circle,
    /// Unwrapped angles at anchor-1, anchor, anchor+1.
    pub angles: [f64; 3],
}
impl LocalCircle {
    pub fn evaluate_uniform(&self, position: f64) -> Option<f64> {
        self.circle
            .evaluate((position - self.anchor as f64) * self.aperture)
    }
    /// Equal-angle subdivision on each side of the middle point, only within the triplet.
    pub fn evaluate_angular(&self, position: f64) -> Option<f64> {
        let u = position - self.anchor as f64;
        if !(-1.0..=1.0).contains(&u) {
            return None;
        }
        let (a, b, t) = if u < 0.0 {
            (self.angles[0], self.angles[1], u + 1.0)
        } else {
            (self.angles[1], self.angles[2], u)
        };
        let y = self.circle.center.y + self.circle.radius * (a + t * (b - a)).sin();
        y.is_finite().then_some(y)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Primitive {
    Circle(LocalCircle),
    /// Degenerate/unstable geometry. Uniform mode skips it; angular mode blends this line.
    Line {
        value: f64,
        slope: f64,
    },
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalModel {
    pub anchor: usize,
    pub aperture: f64,
    pub primitive: Primitive,
}

#[derive(Debug, Clone)]
pub struct Analysis {
    samples: Vec<f64>,
    models: Vec<LocalModel>,
    params: AnalysisParams,
}
impl Analysis {
    pub fn new(samples: &[f64], params: AnalysisParams) -> Result<Self, Error> {
        params.validate()?;
        if let Some(i) = samples.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFiniteSample(i));
        }
        let mut owned = buffer(samples.len())?;
        owned.extend_from_slice(samples);
        let mut models = buffer(samples.len().saturating_sub(2))?;
        for (start, triplet) in samples.windows(3).enumerate() {
            let anchor = start + 1;
            let [y0, y1, y2] = [triplet[0], triplet[1], triplet[2]];
            let range = (y0 - y1).abs().max((y1 - y2).abs()).max((y0 - y2).abs());
            let aperture = match params.aperture {
                Aperture::Disabled => 1.0,
                Aperture::SignalRange => {
                    if range > 0.0 {
                        range
                    } else {
                        1.0
                    }
                }
                Aperture::Adaptive { gain, min, max } => {
                    if gain == 0.0 {
                        min
                    } else {
                        (range * gain).clamp(min, max)
                    }
                }
            };
            let points = [
                Point {
                    x: -aperture,
                    y: y0,
                },
                Point { x: 0.0, y: y1 },
                Point { x: aperture, y: y2 },
            ];
            let primitive = match Circle::through(points, params.collinear_epsilon) {
                Some(circle) => Primitive::Circle(LocalCircle {
                    anchor,
                    aperture,
                    circle,
                    angles: arc_angles(circle, points),
                }),
                None => Primitive::Line {
                    value: y1,
                    slope: y2 * 0.5 - y0 * 0.5,
                },
            };
            models.push(LocalModel {
                anchor,
                aperture,
                primitive,
            });
        }
        Ok(Self {
            samples: owned,
            models,
            params,
        })
    }
    pub fn samples(&self) -> &[f64] {
        &self.samples
    }
    pub fn models(&self) -> &[LocalModel] {
        &self.models
    }
    pub fn params(&self) -> AnalysisParams {
        self.params
    }
}
