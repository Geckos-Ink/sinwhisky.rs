//! Fourier experiments on retained predictors. No FFT acceleration or codec claim.
//!
//! Continuous transforms integrate an individual, unblended circle over a finite
//! interval in sample coordinates, with kernel exp(-i * omega * position).
//! They are NOT DFT coefficients. `frame_dft_bin` instead sums integer predictions.
use crate::codec::ResearchFrame;
use crate::geometry::Branch;
use crate::{Error, LocalCircle};
use std::ops::Range;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    fn add(&mut self, other: Self) {
        self.re += other.re;
        self.im += other.im;
    }

    fn rotate(self, phase: f64) -> Self {
        let (s, c) = phase.sin_cos();
        Self {
            re: self.re * c - self.im * s,
            im: self.re * s + self.im * c,
        }
    }

    fn checked(self) -> Result<Self, Error> {
        if self.re.is_finite() && self.im.is_finite() {
            Ok(self)
        } else {
            Err(Error::InvalidParameter("non-finite Fourier result"))
        }
    }
}

fn validate(circle: LocalCircle, interval: &Range<f64>, omega: f64) -> Result<(), Error> {
    if !interval.start.is_finite()
        || !interval.end.is_finite()
        || interval.start >= interval.end
        || !omega.is_finite()
        || !circle.aperture.is_finite()
        || circle.aperture <= 0.0
        || !circle.circle.radius.is_finite()
        || circle.circle.radius <= 0.0
        || !circle.circle.center.x.is_finite()
        || !circle.circle.center.y.is_finite()
        || circle.angles.iter().any(|v| !v.is_finite())
    {
        return Err(Error::InvalidParameter("Fourier interval or circle"));
    }
    Ok(())
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-4 {
        1.0 - x * x / 6.0 + x.powi(4) / 120.0
    } else {
        x.sin() / x
    }
}

/// Analytic finite-interval transform of the piecewise equal-angle circle.
/// Splits at the anchor; requires the interval to lie inside the original triplet.
/// Stable sinc limits handle DC and frequencies equal to either angular slope.
/// Units are amplitude * samples; use omega=TAU*hz/sample_rate and divide by
/// sample_rate for a transform with time measured in seconds.
pub fn angular_fourier(
    circle: LocalCircle,
    interval: Range<f64>,
    omega: f64,
) -> Result<Complex, Error> {
    validate(circle, &interval, omega)?;
    let anchor = circle.anchor as f64;
    if interval.start < anchor - 1.0 || interval.end > anchor + 1.0 {
        return Err(Error::InvalidParameter("angular Fourier extent"));
    }
    let mut result = Complex::default();
    for side in 0..2 {
        let origin = anchor - 1.0 + side as f64;
        let start = interval.start.max(origin);
        let end = interval.end.min(origin + 1.0);
        if start >= end {
            continue;
        }
        let length = end - start;
        let mid = start + length * 0.5;
        let beta = circle.angles[side + 1] - circle.angles[side];
        let theta = circle.angles[side] + beta * (mid - origin);
        let minus = sinc((beta - omega) * length * 0.5);
        let plus = sinc((beta + omega) * length * 0.5);
        let radius = circle.circle.radius;
        result.add(
            Complex {
                re: length
                    * (circle.circle.center.y * sinc(omega * length * 0.5)
                        + radius * 0.5 * theta.sin() * (minus + plus)),
                im: -length * radius * 0.5 * theta.cos() * (minus - plus),
            }
            .rotate(-omega * mid),
        );
    }
    result.checked()
}

/// Numerical transform of a uniform-x semicircle, using composite midpoint
/// quadrature after x=cx+r*sin(theta). The Jacobian removes the endpoint slope
/// singularity. `panels` (1..=1,048,576) controls work, NOT a certified error bound.
/// Compare increasing panel counts, especially at high frequencies/large spans.
/// Does not copy source anchors or apply the resampler's blending window.
pub fn uniform_fourier(
    circle: LocalCircle,
    interval: Range<f64>,
    omega: f64,
    panels: usize,
) -> Result<Complex, Error> {
    validate(circle, &interval, omega)?;
    if !(1..=1_048_576).contains(&panels) {
        return Err(Error::InvalidParameter("Fourier panels"));
    }
    let c = circle.circle;
    let theta = |position: f64| {
        let dx = (position - circle.anchor as f64) * circle.aperture - c.center.x;
        let ratio = dx / c.radius;
        if (-1.0..=1.0).contains(&ratio) {
            Some(ratio.asin())
        } else {
            None
        }
    };
    let start = theta(interval.start).ok_or(Error::InvalidParameter("uniform Fourier extent"))?;
    let end = theta(interval.end).ok_or(Error::InvalidParameter("uniform Fourier extent"))?;
    let step = (end - start) / panels as f64;
    if !step.is_finite() || step <= 0.0 {
        return Err(Error::InvalidParameter("unresolvable Fourier interval"));
    }
    let sign = match c.branch {
        Branch::Upper => 1.0,
        Branch::Lower => -1.0,
    };
    let mut result = Complex::default();
    for i in 0..panels {
        let angle = start + (i as f64 + 0.5) * step;
        let (s, cos) = angle.sin_cos();
        let position = circle.anchor as f64 + (c.center.x + c.radius * s) / circle.aperture;
        let value = c.center.y + sign * c.radius * cos;
        result.add(
            Complex {
                re: value * (c.radius / circle.aperture) * cos * step,
                im: 0.0,
            }
            .rotate(-omega * position),
        );
    }
    result.checked()
}

/// Unnormalized DFT bin of the frame's predicted integer samples, evaluated
/// directly from retained descriptors without materializing PCM. O(N) per bin;
/// a full spectrum is O(N^2), not a replacement for an FFT. No original samples
/// are accessed. Empty frames and bins >= sample_count are rejected.
pub fn frame_dft_bin(frame: &ResearchFrame, bin: usize) -> Result<Complex, Error> {
    let n = frame.sample_count();
    if bin >= n {
        return Err(Error::InvalidParameter("DFT bin"));
    }
    let omega = std::f64::consts::TAU * (bin as f64 / n as f64);
    let mut result = Complex::default();
    for segment in frame.segments() {
        for i in segment.range() {
            result.add(
                Complex {
                    re: segment
                        .model()
                        .predict(i)
                        .ok_or(Error::NonFiniteSample(i))?,
                    im: 0.0,
                }
                .rotate(-omega * i as f64),
            );
        }
    }
    result.checked()
}
