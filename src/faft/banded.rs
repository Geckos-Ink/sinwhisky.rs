//! Two-way banded joint tone fit (PAPER_EQ_COMPRESSION.md, Section 2.4).
//!
//! The joint fit of `Faft::fit` couples components only through the window
//! kernel, a function of their frequency distance that decays away from zero.
//! With components in ascending frequency and the Gram matrix kept within a
//! `reach`, the matrix is banded and its Cholesky factor has the same band. The
//! fit is then one ascending pass (factor and forward substitution, which already
//! gives the explained energy) and one descending pass (back substitution for the
//! amplitudes, and selected inversion of the band for each component's
//! significance). Cost O(P*m^2) for P components and bandwidth m, instead of the
//! dense O(P^3). With an infinite reach it is the dense fit.
use super::{checked, Component, Faft, SpectralFit};
use crate::spectral::Complex;
use crate::{buffer, Error};

impl Faft {
    /// Joint least-squares fit of real tones at `omegas` (finite, nonnegative,
    /// distinct, at most 65536), with Gram entries kept for components at most
    /// `reach` radians per sample apart (`f64::INFINITY` keeps all of them and
    /// equals `fit`). Components are returned in ascending frequency. The residual
    /// energy is evaluated from the full quadratic form of the returned
    /// amplitudes, so `bound` holds at every real frequency even when the band
    /// truncates the Gram matrix. Cost: one transform value per frequency,
    /// O(P*m^2) for the solve and O(P^2) kernel values for a truncated residual.
    pub fn fit_banded(&self, omegas: &[f64], reach: f64) -> Result<SpectralFit, Error> {
        if omegas.len() > 65536
            || omegas.iter().any(|w| !w.is_finite() || *w < 0.0)
            || reach.is_nan()
            || reach <= 0.0
        {
            return Err(Error::InvalidParameter("FAFT banded fit parameters"));
        }
        let mut sorted = omegas.to_vec();
        sorted.sort_by(f64::total_cmp);
        if sorted.windows(2).any(|p| p[0] == p[1]) {
            return Err(Error::InvalidParameter("FAFT fit frequencies"));
        }
        let n = sorted.len();
        let mut table = Vec::with_capacity(self.lengths.len());
        let mut observed = buffer(n)?;
        for &omega in &sorted {
            observed.push(checked(self.centred_value(omega, &mut table))?);
        }
        let energy = self.energy()?;
        // Bandwidth: most components within `reach` above any component.
        let mut width = 0;
        let mut top = 0;
        for p in 0..n {
            top = top.max(p);
            while top + 1 < n && sorted[top + 1] - sorted[p] <= reach {
                top += 1;
            }
            width = width.max(top - p);
        }
        let truncated = width + 1 < n && reach.is_finite();
        let gram = |sine: bool, p: usize, q: usize| {
            let (wp, wq) = (sorted[p], sorted[q]);
            if (wp - wq).abs() > reach {
                return 0.0;
            }
            let minus = self.kernel_derivatives(wp - wq)[0];
            let plus = self.kernel_derivatives(wp + wq)[0];
            0.5 * if sine { minus - plus } else { minus + plus }
        };
        let mut amplitudes = [vec![0.0; n], vec![0.0; n]];
        let mut unique = vec![0.0; n];
        let mut explained = 0.0;
        let mut degenerate = 0;
        let stride = width + 1;
        for (system, sine) in [false, true].into_iter().enumerate() {
            let rhs: Vec<f64> = observed
                .iter()
                .map(|v| if sine { -v.im } else { v.re })
                .collect();
            let band = BandCholesky::factor(n, width, |p, q| gram(sine, p, q))?;
            degenerate += band.dropped;
            // Ascending pass: z = L^-1 b, the coefficients in the Gram-Schmidt
            // basis taken in ascending frequency; |z|^2 is the explained energy.
            let mut z = vec![0.0; n];
            for i in 0..n {
                if !band.active[i] {
                    continue;
                }
                let first = i.saturating_sub(width);
                let mut v = rhs[i];
                for (k, zk) in z.iter().enumerate().take(i).skip(first) {
                    v -= band.at(i, k) * zk;
                }
                z[i] = v / band.at(i, i);
                explained += z[i] * z[i];
            }
            // Descending pass: back substitution, highest component first.
            let x = &mut amplitudes[system];
            for i in (0..n).rev() {
                if !band.active[i] {
                    continue;
                }
                let mut v = z[i];
                for (k, xk) in x.iter().enumerate().take((i + stride).min(n)).skip(i + 1) {
                    v -= band.at(k, i) * xk;
                }
                x[i] = v / band.at(i, i);
            }
            // Selected inversion (Takahashi): diag(G^-1) within the band, also
            // descending. Removing atom p raises the residual by x_p^2/(G^-1)_pp.
            let inverse = band.inverse_diagonal()?;
            for p in 0..n {
                if inverse[p] > 0.0 {
                    unique[p] += x[p] * x[p] / inverse[p];
                }
            }
        }
        let w = self.weight_sum;
        let residual = if truncated {
            // E - 2W a.b + W a.G a over both systems, with the full Gram matrix.
            let mut quadratic = 0.0;
            for (system, sine) in [false, true].into_iter().enumerate() {
                let x = &amplitudes[system];
                for p in 0..n {
                    let b = if sine {
                        -observed[p].im
                    } else {
                        observed[p].re
                    };
                    quadratic -= 2.0 * x[p] * b;
                    for q in 0..n {
                        let (wp, wq) = (sorted[p], sorted[q]);
                        let minus = self.kernel_derivatives(wp - wq)[0];
                        let plus = self.kernel_derivatives(wp + wq)[0];
                        let g = 0.5 * if sine { minus - plus } else { minus + plus };
                        quadratic += x[p] * g * x[q];
                    }
                }
            }
            energy + w * quadratic
        } else {
            energy - w * explained
        };
        let residual_energy = residual.max(0.0);
        let mut components = buffer(n)?;
        for p in 0..n {
            let value = Complex {
                re: 0.5 * amplitudes[0][p],
                im: -0.5 * amplitudes[1][p],
            };
            if !(value.re.is_finite() && value.im.is_finite()) {
                return Err(Error::InvalidParameter("FAFT fit overflow"));
            }
            components.push(Component {
                omega: sorted[p],
                value: self.to_start(sorted[p], value),
                observed: self.to_start(sorted[p], observed[p]),
                significance: (0.5 * unique[p]).sqrt(),
            });
        }
        Ok(SpectralFit {
            components,
            energy,
            residual_energy,
            bound: (residual_energy / w).sqrt(),
            degenerate,
        })
    }
}

impl Faft {
    /// `fit_banded`, then up to `iterations` rounds that move each frequency by one
    /// safeguarded Newton step towards the maximum of its own partial residual
    /// `|a(omega) - sum_{q != p} tone_q(omega)|^2` (the other fitted tones removed
    /// in closed form through the kernel), followed by a banded refit. Steps are
    /// limited to a quarter resolution; it stops once no step exceeds
    /// `tolerance` radians per sample. A coordinate method: cheap (one derivative
    /// evaluation per component and round) and accurate for components a few
    /// resolutions apart, slow for strongly coupled ones (see `pursue`).
    pub fn fit_refined(
        &self,
        omegas: &[f64],
        reach: f64,
        iterations: usize,
        tolerance: f64,
    ) -> Result<SpectralFit, Error> {
        if iterations > 64 || tolerance.is_nan() || tolerance <= 0.0 {
            return Err(Error::InvalidParameter("FAFT refinement parameters"));
        }
        let mut fit = self.fit_banded(omegas, reach)?;
        let limit = 0.25 * self.resolution();
        let half = 0.5 * self.window.len() as f64;
        let mut table = Vec::with_capacity(self.lengths.len());
        for _ in 0..iterations {
            let centred: Vec<(f64, Complex)> = fit
                .components
                .iter()
                .map(|c| (c.omega, super::mul(c.value, super::phase(c.omega * half))))
                .collect();
            let mut moved = Vec::with_capacity(centred.len());
            let mut largest: f64 = 0.0;
            for (p, &(omega, _)) in centred.iter().enumerate() {
                let mut r = self.centered(omega, &mut table)?;
                for (q, &(wq, value)) in centred.iter().enumerate() {
                    if q == p || (omega - wq).abs() > reach {
                        continue;
                    }
                    let below = self.kernel_derivatives(omega - wq);
                    let above = self.kernel_derivatives(omega + wq);
                    let image = super::conj(value);
                    for k in 0..3 {
                        let t = super::add(
                            super::scale(value, below[k]),
                            super::scale(image, above[k]),
                        );
                        r[k] = super::sub(r[k], t);
                    }
                }
                let [g1, g2] = super::power_derivatives(&r);
                let step = if g2 < 0.0 {
                    (-g1 / g2).clamp(-limit, limit)
                } else {
                    0.0
                };
                let next = (omega + step).max(0.0);
                largest = largest.max((next - omega).abs());
                moved.push(next);
            }
            // Two components that converge onto one frequency keep one.
            moved.sort_by(f64::total_cmp);
            moved.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
            fit = self.fit_banded(&moved, reach)?;
            if largest <= tolerance {
                break;
            }
        }
        Ok(fit)
    }
}

/// Cholesky factor of a symmetric positive (semi)definite band matrix, row-major
/// band storage: entry (i, j), i - width <= j <= i, at `i*(width+1) + (i-j)`.
/// Numerically dependent atoms are dropped (row of zeros, unit diagonal), as in
/// the dense solver of `pursuit`.
struct BandCholesky {
    n: usize,
    width: usize,
    lower: Vec<f64>,
    active: Vec<bool>,
    dropped: usize,
}
impl BandCholesky {
    fn factor(n: usize, width: usize, entry: impl Fn(usize, usize) -> f64) -> Result<Self, Error> {
        let stride = width + 1;
        let mut lower = buffer(n.checked_mul(stride).ok_or(Error::OutputTooLarge)?)?;
        lower.resize(n * stride, 0.0);
        let mut active = vec![true; n];
        let mut dropped = 0;
        for i in 0..n {
            let first = i.saturating_sub(width);
            for j in first..=i {
                let mut v = entry(i, j);
                if j < i && !active[j] {
                    continue;
                }
                for k in first.max(j.saturating_sub(width))..j {
                    v -= lower[i * stride + (i - k)] * lower[j * stride + (j - k)];
                }
                if j < i {
                    lower[i * stride + (i - j)] = v / lower[j * stride];
                } else {
                    // Atoms are normalized (diagonal ~1/2); one of negligible norm,
                    // such as the sine atom within a hair of DC, is dropped rather
                    // than fitted with a huge, ill-conditioned amplitude.
                    let diagonal = entry(i, i);
                    if v.is_nan() || diagonal <= 1e-9 || v <= 1e-10 * diagonal {
                        active[i] = false;
                        dropped += 1;
                        for k in first..i {
                            lower[i * stride + (i - k)] = 0.0;
                        }
                        lower[i * stride] = 1.0;
                    } else {
                        lower[i * stride] = v.sqrt();
                    }
                }
            }
        }
        if lower.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidParameter("FAFT banded fit overflow"));
        }
        Ok(Self {
            n,
            width,
            lower,
            active,
            dropped,
        })
    }

    /// L(i, j) for i - width <= j <= i.
    fn at(&self, i: usize, j: usize) -> f64 {
        self.lower[i * (self.width + 1) + (i - j)]
    }

    /// Diagonal of G^-1 by the Takahashi recurrence on the band, from the last
    /// row to the first: Z(j,i) = (delta_ji/L_ii - sum_{k>i} Z(j,k) L(k,i)) / L_ii.
    /// Zero for dropped atoms.
    fn inverse_diagonal(&self) -> Result<Vec<f64>, Error> {
        let (n, width) = (self.n, self.width);
        let stride = width + 1;
        let mut z = buffer(n * stride)?;
        z.resize(n * stride, 0.0);
        let get = |z: &[f64], a: usize, b: usize| {
            let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
            z[hi * stride + (hi - lo)]
        };
        for i in (0..n).rev() {
            if !self.active[i] {
                continue;
            }
            let last = (i + width).min(n - 1);
            let diagonal = self.at(i, i);
            for j in (i..=last).rev() {
                let mut v = if j == i { 1.0 / diagonal } else { 0.0 };
                for k in i + 1..=last {
                    v -= get(&z, j, k) * self.at(k, i);
                }
                z[j * stride + (j - i)] = v / diagonal;
            }
        }
        Ok((0..n).map(|i| z[i * stride]).collect())
    }
}
