//! Reference preset: analytic integration retained for continuous-model checks.
//! Experimental Angular Fourier Transform of a periodic angular interpolant.
//!
//! This is a continuous Fourier integral, not another algorithm for the DFT.
//! Each unit interval averages the adjacent circles' angular halves. Degenerate
//! or numerically inconsistent halves use that interval's source-pair line.
//! No resampler window, zoom, compression, or FFT approximation is involved.
use crate::spectral::Complex;
use crate::{buffer, Analysis, AnalysisParams, Error, Primitive};
use std::f64::consts::TAU;

#[derive(Debug, Clone, Copy)]
enum Half {
    Arc {
        cy: f64,
        radius: f64,
        theta: f64,
        beta: f64,
        sine: f64,
        cosine: f64,
    },
    Line {
        middle: f64,
        slope: f64,
    },
}

impl Half {
    fn value(self, u: f64) -> f64 {
        match self {
            Self::Arc {
                cy,
                radius,
                theta,
                beta,
                ..
            } => cy + radius * (theta + beta * u).sin(),
            Self::Line { middle, slope } => middle + slope * u,
        }
    }

    // Integral on u in [-1/2,1/2], without the global phase factor.
    fn integral(self, omega: f64) -> Complex {
        let z = omega * 0.5;
        match self {
            Self::Arc {
                cy,
                sine,
                cosine,
                beta,
                ..
            } => {
                let minus = sinc((beta - omega) * 0.5);
                let plus = sinc((beta + omega) * 0.5);
                Complex {
                    re: cy * sinc(z) + sine * (minus + plus) * 0.5,
                    im: -cosine * (minus - plus) * 0.5,
                }
            }
            Self::Line { middle, slope } => Complex {
                re: middle * sinc(z),
                im: slope * 0.5 * sinc_derivative(z),
            },
        }
    }
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-4 {
        1.0 - x * x / 6.0 + x.powi(4) / 120.0
    } else {
        x.sin() / x
    }
}

fn sinc_derivative(x: f64) -> f64 {
    if x.abs() < 1e-3 {
        -x / 3.0 + x.powi(3) / 30.0 - x.powi(5) / 840.0
    } else {
        (x * x.cos() - x.sin()) / (x * x)
    }
}

/// Read-only view of one prepared half on `u in [-1/2, 1/2]` of its unit interval.
/// `Arc`: `center + radius*sin(theta + beta*u)`; `Line`: `middle + slope*u`.
/// The interpolant on each interval is the average of its two halves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AngularHalf {
    Arc {
        center: f64,
        radius: f64,
        theta: f64,
        beta: f64,
    },
    Line {
        middle: f64,
        slope: f64,
    },
}

/// Counts halves, two per source interval. Rejected circles are included in lines.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AngularStats {
    pub circle_halves: usize,
    pub line_halves: usize,
    pub rejected_circle_halves: usize,
}

/// Immutable, prepared periodic angular geometry, independent of transform bins.
/// Period N covers `[0,N]`, including the closing interval from last to first sample.
/// The input's amplitude scale and analysis parameters are explicit: absolute
/// determinant tolerances can change circle/line classification when scaling input.
#[derive(Debug, Clone)]
pub struct AngularFrame {
    intervals: Vec<[Half; 2]>,
    stats: AngularStats,
}

impl AngularFrame {
    /// Fit periodic halos once, then prepare the two halves of every interval.
    /// Empty input is rejected; singleton input becomes a constant period of one.
    /// A fitted half is rejected if its endpoint discrepancy exceeds
    /// 1e-8 * max(1, abs(left), abs(right)); fallback uses the actual source pair.
    pub fn new(samples: &[f64], params: AnalysisParams) -> Result<Self, Error> {
        if samples.is_empty() {
            return Err(Error::InvalidParameter("AFT empty frame"));
        }
        if let Some(i) = samples.iter().position(|value| !value.is_finite()) {
            return Err(Error::NonFiniteSample(i));
        }
        let n = samples.len();
        let mut periodic = buffer(n.checked_add(3).ok_or(Error::OutputTooLarge)?)?;
        periodic.push(samples[n - 1]);
        periodic.extend_from_slice(samples);
        periodic.push(samples[0]);
        periodic.push(samples[1 % n]);
        let analysis = Analysis::new(&periodic, params)?;
        let mut intervals = buffer(n)?;
        let mut stats = AngularStats::default();
        for i in 0..n {
            let left = samples[i];
            let right = samples[(i + 1) % n];
            let line = Half::Line {
                middle: left * 0.5 + right * 0.5,
                slope: right - left,
            };
            let mut halves = [line; 2];
            for (side, half) in halves.iter_mut().enumerate() {
                if let Primitive::Circle(c) = analysis.models()[i + side].primitive {
                    // Left-anchor right half; right-anchor left half.
                    let angle_index = 1 - side;
                    let beta = c.angles[angle_index + 1] - c.angles[angle_index];
                    let theta = c.angles[angle_index] + beta * 0.5;
                    let (s, cos) = theta.sin_cos();
                    let candidate = Half::Arc {
                        cy: c.circle.center.y,
                        radius: c.circle.radius,
                        theta,
                        beta,
                        sine: c.circle.radius * s,
                        cosine: c.circle.radius * cos,
                    };
                    let bound = 1e-8 * left.abs().max(right.abs()).max(1.0);
                    if (candidate.value(-0.5) - left).abs() <= bound
                        && (candidate.value(0.5) - right).abs() <= bound
                    {
                        *half = candidate;
                        stats.circle_halves += 1;
                        continue;
                    }
                    stats.rejected_circle_halves += 1;
                }
                stats.line_halves += 1;
            }
            if !halves.iter().all(|h| h.value(0.0).is_finite()) {
                return Err(Error::InvalidParameter("non-finite AFT geometry"));
            }
            intervals.push(halves);
        }
        Ok(Self { intervals, stats })
    }

    pub fn sample_count(&self) -> usize {
        self.intervals.len()
    }

    pub fn stats(&self) -> AngularStats {
        self.stats
    }

    /// Both prepared halves of every unit interval, in interval order. Exposes the
    /// immutable geometry so other transforms can reuse it without refitting.
    pub fn halves(&self) -> impl Iterator<Item = [AngularHalf; 2]> + '_ {
        self.intervals.iter().map(|pair| {
            pair.map(|half| match half {
                Half::Arc {
                    cy,
                    radius,
                    theta,
                    beta,
                    ..
                } => AngularHalf::Arc {
                    center: cy,
                    radius,
                    theta,
                    beta,
                },
                Half::Line { middle, slope } => AngularHalf::Line { middle, slope },
            })
        })
    }

    /// Evaluate the same interpolant used by `fourier`, periodically extended.
    /// This is useful for independent numerical integration, not a resampling preset.
    pub fn evaluate(&self, position: f64) -> Result<f64, Error> {
        if !position.is_finite() {
            return Err(Error::InvalidParameter("AFT position"));
        }
        let t = position.rem_euclid(self.sample_count() as f64);
        let i = (t.floor() as usize).min(self.sample_count() - 1);
        let u = t - i as f64 - 0.5;
        let halves = self.intervals[i];
        let value = halves[0].value(u) * 0.5 + halves[1].value(u) * 0.5;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::InvalidParameter("AFT evaluation"))
        }
    }

    /// Integral y(t) exp(-i*omega*t) dt on `[0,N]`, in amplitude * sample units.
    /// O(N) work. Use omega=TAU*Hz/sample_rate; divide by sample_rate for seconds.
    pub fn fourier(&self, omega: f64) -> Result<Complex, Error> {
        if !omega.is_finite() {
            return Err(Error::InvalidParameter("AFT frequency"));
        }
        let mut total = Complex::default();
        let mut correction = Complex::default();
        for (i, halves) in self.intervals.iter().enumerate() {
            let a = halves[0].integral(omega);
            let b = halves[1].integral(omega);
            let re = (a.re + b.re) * 0.5;
            let im = (a.im + b.im) * 0.5;
            let (s, c) = (-omega * (i as f64 + 0.5)).sin_cos();
            let value = Complex {
                re: re * c - im * s,
                im: re * s + im * c,
            };
            // Compensated accumulation limits cancellation between intervals.
            let y = Complex {
                re: value.re - correction.re,
                im: value.im - correction.im,
            };
            let next = Complex {
                re: total.re + y.re,
                im: total.im + y.im,
            };
            correction = Complex {
                re: (next.re - total.re) - y.re,
                im: (next.im - total.im) - y.im,
            };
            total = next;
        }
        if total.re.is_finite() && total.im.is_finite() {
            Ok(total)
        } else {
            Err(Error::InvalidParameter("non-finite AFT integral"))
        }
    }

    /// N/2+1 integrals at the usual nonnegative DFT frequencies, k*TAU/N.
    /// O(N²) for the full spectrum. This does not compute a DFT or claim FFT speed.
    /// Unlike an even-length real DFT, the continuous coefficient at pi need not
    /// be real: its negative-frequency partner is distinct, not aliased onto it.
    pub fn spectrum(&self) -> Result<Vec<Complex>, Error> {
        let n = self.sample_count();
        let mut result = buffer(n / 2 + 1)?;
        for k in 0..=n / 2 {
            result.push(self.fourier(TAU * k as f64 / n as f64)?);
        }
        Ok(result)
    }
}
