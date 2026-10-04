//! Circle-based resampling with reusable fitted geometry and experimental segment coding.
//!
//! ```
//! use sinwhisky::{Analysis, AnalysisParams, ResampleParams};
//! let analysis = Analysis::new(&[0.0, 1.0, 0.0], AnalysisParams::default())?;
//! let output = analysis.resample(ResampleParams::new(3))?;
//! assert_eq!(output.len(), 9);
//! assert_eq!(output[4], 1.0);
//! # Ok::<(), sinwhisky::Error>(())
//! ```
#![forbid(unsafe_code)]

pub mod adaptive_faft;
pub mod aft;
pub mod analog;
pub mod analysis;
pub mod bands;
pub mod codec;
pub mod faft;
pub mod geometric_faft;
pub mod geometry;
pub mod layered;
pub mod phasor_fft;
pub mod resample;
pub mod spectral;

pub use analysis::{Analysis, AnalysisParams, Aperture, LocalCircle, LocalModel, Primitive};
pub use resample::{resample, Interpolation, ResampleParams};

/// Invalid input is rejected before fitting or allocating an output buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidParameter(&'static str),
    NonFiniteSample(usize),
    OutputTooLarge,
    LengthMismatch,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidParameter(name) => write!(f, "invalid parameter: {name}"),
            Self::NonFiniteSample(i) => write!(f, "non-finite sample at index {i}"),
            Self::OutputTooLarge => {
                write!(f, "output length or allocation exceeds available capacity")
            }
            Self::LengthMismatch => write!(f, "sample lengths differ"),
        }
    }
}
impl std::error::Error for Error {}

pub(crate) fn buffer<T>(len: usize) -> Result<Vec<T>, Error> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(len)
        .map_err(|_| Error::OutputTooLarge)?;
    Ok(result)
}
