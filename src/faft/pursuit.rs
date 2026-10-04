//! Relevant frequencies as a jointly fitted tone model of one FAFT window.
//!
//! A local maximum of |FAFT| is not the amplitude of a component: every component
//! spreads over neighbouring frequencies through the window kernel K (the exact
//! continuous transform of the weight), so a peak's height also contains the
//! leakage of its neighbours, and a component that is not selected leaves its
//! energy in its neighbours' heights and in spurious local maxima (sidelobes).
//! The continuous axis removes grid scalloping, not leakage.
//!
//! This module therefore treats a peak list as a model: real tones
//! `alpha*cos(omega*(t-tc)) + beta*sin(omega*(t-tc))` around the window centre tc,
//! fitted jointly by weighted least squares. Every quantity is closed form: the
//! Gram matrix is built from K, the projections are FAFT values at the selected
//! frequencies, and the residual energy follows from the curve energy. By
//! Cauchy-Schwarz the residual spectrum is bounded at EVERY real frequency by
//! `sqrt(residual_energy/W)` (normalized), which certifies what was not selected.
//!
//! `Faft::pursue` scans once, then seeds from local maxima of the residual
//! spectrum on the same grid (subtracting closed-form kernels, no new transform
//! values). Each seed is refined by Newton on the residual; then every frequency
//! is polished jointly with all amplitudes by damped Gauss-Newton on the full
//! nonlinear least squares (closed-form normal equations from K, K', K'' and
//! the transform's F, F'), and components whose unique energy is below that of a
//! floor-level tone are pruned. It stops when no residual seed remains, the
//! certificate falls below the floor, or a round/capacity limit is reached.
//! `PeakTracker` polishes a frame's components from their predicted frequencies
//! in the next window and rescans only on an interval or when the residual
//! energy grows (novelty).
//!
//! The curve energy is the only quantity integrated numerically: 16-point
//! Gauss-Legendre panels on each support, with at most pi of phase per panel.
use super::{
    add, conj, mul, norm, phase, scale, sub, Faft, PeakParams, PeakStats, Seed, Weighting,
};
use crate::spectral::Complex;
use crate::{buffer, Error};
use std::f64::consts::PI;

const TAYLOR: usize = 9;
/// (-1)^n/(2n+1)!: coefficients of sinc(x) = sin(x)/x in powers of x^2.
const fn sinc_coefficients() -> [f64; TAYLOR] {
    let mut out = [0.0; TAYLOR];
    let mut n = 0;
    let mut factorial = 1.0;
    while n < TAYLOR {
        if n > 0 {
            factorial *= (2 * n) as f64 * (2 * n + 1) as f64;
        }
        out[n] = if n % 2 == 0 { 1.0 } else { -1.0 } / factorial;
        n += 1;
    }
    out
}
const SINC: [f64; TAYLOR] = sinc_coefficients();

/// sinc(x) and its first two derivatives. Taylor series to x^16 below 1/2
/// (truncation < 0.5^18/19!), closed forms above (relative error ~eps/x^2).
fn sinc(x: f64) -> [f64; 3] {
    if x.abs() < 0.5 {
        let y = x * x;
        let (mut s, mut d, mut dd) = (0.0, 0.0, 0.0);
        for n in (0..TAYLOR).rev() {
            s = s * y + SINC[n];
            if n > 0 {
                d = d * y + 2.0 * n as f64 * SINC[n];
                dd = dd * y + (2 * n * (2 * n - 1)) as f64 * SINC[n];
            }
        }
        [s, d * x, dd]
    } else {
        let (sin, cos) = x.sin_cos();
        let s = sin / x;
        let d = (cos - s) / x;
        [s, d, -s - 2.0 * d / x]
    }
}

/// Centred rectangular kernel K(nu) = integral_{-T/2}^{T/2} exp(-i*nu*u) du
/// = T*sinc(nu*T/2), real and even, with its first two nu derivatives.
fn rect_kernel(nu: f64, duration: f64) -> [f64; 3] {
    let half = 0.5 * duration;
    let [s, d, dd] = sinc(nu * half);
    [
        duration * s,
        duration * half * d,
        duration * half * half * dd,
    ]
}

/// 16-point Gauss-Legendre nodes and weights mapped to [0, 1] (weights sum to 1).
fn gauss_legendre() -> [(f64, f64); 16] {
    const N: usize = 16;
    let mut out = [(0.0, 0.0); N];
    for (i, slot) in out.iter_mut().enumerate() {
        let mut x = (PI * (i as f64 + 0.75) / (N as f64 + 0.5)).cos();
        let mut derivative = 1.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=N {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            derivative = N as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / derivative;
            x -= dx;
            if dx.abs() < 1e-16 {
                break;
            }
        }
        let weight = 2.0 / ((1.0 - x * x) * derivative * derivative);
        *slot = (0.5 * (1.0 + x), 0.5 * weight);
    }
    out
}

/// Interval around a peak where |FAFT| stays at or above `level`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeakRange {
    /// Peak frequency the range was measured from, and its normalized magnitude.
    pub omega: f64,
    pub magnitude: f64,
    /// `magnitude * 10^(-drop_db/20)`.
    pub level: f64,
    pub lower: f64,
    pub upper: f64,
    /// True when that side ends at a valley above `level`: the range is shared
    /// with a neighbouring maximum and the edge is the valley's minimum.
    pub lower_valley: bool,
    pub upper_valley: bool,
}
impl PeakRange {
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }
}

/// One fitted tone. Values are normalized like `Faft::value` (a unit sinusoid
/// gives ~0.5) with phase origin at the window start.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Component {
    pub omega: f64,
    /// Jointly fitted complex amplitude `(alpha - i*beta)/2`, rotated to the
    /// window start: the component's own height, with neighbours' leakage removed.
    pub value: Complex,
    /// The transform itself at `omega`: own height plus every neighbour's leakage.
    pub observed: Complex,
    /// The magnitude an isolated tone would need to explain this component's
    /// UNIQUE energy, i.e. the residual increase if it were removed and the
    /// others refitted: `sqrt(dE / 2W)`. It equals the magnitude for an isolated
    /// tone and is smaller when neighbours share the same energy.
    pub significance: f64,
}
impl Component {
    pub fn magnitude(&self) -> f64 {
        norm(self.value)
    }
    /// 20*log10(|observed|/|fitted|): how much of the observed height in dB comes
    /// from (or is cancelled by) other components, the conjugate image and the
    /// model's non-tonal content.
    pub fn leakage_db(&self) -> f64 {
        20.0 * (norm(self.observed) / norm(self.value)).log10()
    }
}

/// Joint least-squares tone fit of a window at given frequencies.
#[derive(Debug, Clone)]
pub struct SpectralFit {
    pub components: Vec<Component>,
    /// `integral y^2 w dt` of the curve in the window (amplitude^2 * samples).
    pub energy: f64,
    /// `integral (y - tones)^2 w dt`, clamped at zero.
    pub residual_energy: f64,
    /// `sqrt(residual_energy / W)`: no normalized residual value exceeds it at
    /// any real frequency (Cauchy-Schwarz).
    pub bound: f64,
    /// Atoms dropped as numerically dependent (e.g. sine atom at omega = 0).
    pub degenerate: usize,
}
impl SpectralFit {
    /// Fraction of the weighted energy explained by the tones.
    pub fn explained(&self) -> f64 {
        if self.energy > 0.0 {
            1.0 - self.residual_energy / self.energy
        } else {
            1.0
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PursuitParams {
    /// Band, scan density, floors (relative to the strongest scanned magnitude),
    /// seed exclusion width (`merge_width`: residual seeds this close to a
    /// component or a rejected position are skipped; components themselves are
    /// pruned by significance, not by distance), component capacity
    /// (`max_peaks`, at most 1024) and Newton limits. `scan_density` is required
    /// and `circle_seeds` must be false.
    pub search: PeakParams,
    /// Residual seeding rounds after the first scan, 1..=64.
    pub max_rounds: usize,
    /// Most joint Gauss-Newton (Levenberg-Marquardt) steps per polishing, 0..=64.
    pub polish_iterations: usize,
    /// Polishing stops, without evaluating the step, once no proposed frequency
    /// step exceeds this many resolutions (2*pi/T). This is the stated frequency
    /// precision of the fit; steps within the energy's rounding are accepted.
    pub polish_tolerance: f64,
}
impl Default for PursuitParams {
    fn default() -> Self {
        Self {
            search: PeakParams {
                max_peaks: 256,
                ..PeakParams::default()
            },
            max_rounds: 6,
            polish_iterations: 16,
            polish_tolerance: 1e-4,
        }
    }
}
impl PursuitParams {
    fn validate(&self) -> Result<(), Error> {
        self.search.validate()?;
        if self.search.scan_density.is_none()
            || self.search.circle_seeds
            || self.search.max_peaks > 1024
            || !(1..=64).contains(&self.max_rounds)
            || self.polish_iterations > 64
            || !(self.polish_tolerance > 0.0 && self.polish_tolerance <= 0.1)
        {
            return Err(Error::InvalidParameter("FAFT pursuit parameters"));
        }
        Ok(())
    }
}

/// Why a pursuit ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PursuitStop {
    /// The window has no energy or no scanned magnitude.
    Silent,
    /// No residual local maximum above the floor produced a new component.
    NoSeeds,
    /// `bound <= floor`: no residual value above the floor exists anywhere.
    Certified,
    /// `max_rounds` reached with seeds remaining.
    Rounds,
    /// `max_peaks` components reached.
    Capacity,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PursuitStats {
    /// Scan, Newton and work counters (`rect_evaluations` counts rectangular
    /// sums; a Hann value costs three).
    pub search: PeakStats,
    pub rounds: usize,
    /// Least-squares solves.
    pub fits: usize,
    /// Accepted and rejected joint Gauss-Newton steps.
    pub polish_steps: usize,
    pub polish_rejected: usize,
    /// Components removed because their significance fell below the floor.
    pub pruned: usize,
    pub degenerate: usize,
}
impl PursuitStats {
    fn absorb(&mut self, other: &PursuitStats) {
        let (a, b) = (&mut self.search, &other.search);
        a.scan_points += b.scan_points;
        a.rect_evaluations += b.rect_evaluations;
        a.seeds += b.seeds;
        a.refined += b.refined;
        a.derivative_evaluations += b.derivative_evaluations;
        a.iterations += b.iterations;
        a.flanks += b.flanks;
        a.unconverged += b.unconverged;
        a.merged += b.merged;
        a.excluded += b.excluded;
        self.rounds += other.rounds;
        self.fits += other.fits;
        self.polish_steps += other.polish_steps;
        self.polish_rejected += other.polish_rejected;
        self.pruned += other.pruned;
        self.degenerate += other.degenerate;
    }
}

#[derive(Debug, Clone)]
pub struct Pursuit {
    /// Fitted components above the floor, sorted by frequency.
    pub components: Vec<Component>,
    pub energy: f64,
    pub residual_energy: f64,
    pub bound: f64,
    /// Strongest scanned magnitude and the resulting floor.
    pub strongest: f64,
    pub floor: f64,
    pub stop: PursuitStop,
    pub stats: PursuitStats,
}
impl Pursuit {
    pub fn explained(&self) -> f64 {
        if self.energy > 0.0 {
            1.0 - self.residual_energy / self.energy
        } else {
            1.0
        }
    }
}

/// Working component: centred, normalized transform and fitted amplitude.
#[derive(Debug, Clone, Copy)]
struct Work {
    omega: f64,
    /// Centred transform F and its frequency derivative F' at `omega`.
    observed: Complex,
    slope: Complex,
    value: Complex,
    /// Caller bookkeeping (the tracker's track index); unused by the pursuit.
    tag: usize,
    /// Needs polishing: new, or a neighbour within the coupling gap changed.
    active: bool,
    /// Where the component was seeded, remembered if it is later rejected.
    origin: f64,
    /// sqrt(unique energy / 2W): see `Component::significance`.
    significance: f64,
    /// Scratch cluster index while polishing.
    group: usize,
}
impl Work {
    fn new(omega: f64, observed: Complex, value: Complex, tag: usize) -> Self {
        Self {
            omega,
            observed,
            slope: Complex::default(),
            value,
            tag,
            active: true,
            origin: omega,
            significance: 0.0,
            group: usize::MAX,
        }
    }
}

impl Faft {
    /// Normalized centred window kernel `K(nu)/W` and its first two derivatives:
    /// the exact transform of the weight, i.e. the shape every tone takes on the
    /// continuous axis. Hann is `1/2 K(nu) + 1/4 K(nu-D) + 1/4 K(nu+D)`, D = 2*pi/T.
    fn kernel_derivatives(&self, nu: f64) -> [f64; 3] {
        let duration = self.window.len() as f64;
        let inverse = 1.0 / self.weight_sum;
        match self.weighting {
            Weighting::Rectangular => rect_kernel(nu, duration).map(|v| v * inverse),
            Weighting::Hann => {
                let shift = self.resolution();
                let (a, b, d) = (
                    rect_kernel(nu, duration),
                    rect_kernel(nu - shift, duration),
                    rect_kernel(nu + shift, duration),
                );
                [0, 1, 2].map(|k| (0.5 * a[k] + 0.25 * (b[k] + d[k])) * inverse)
            }
        }
    }

    /// Normalized window kernel `K(nu)/W` (1 at nu = 0): the response of this
    /// window's weighting to a tone at offset `nu` radians per sample. It is real
    /// and may be negative on rectangular sidelobes.
    pub fn kernel(&self, nu: f64) -> Result<f64, Error> {
        if !nu.is_finite() {
            return Err(Error::InvalidParameter("FAFT kernel frequency"));
        }
        Ok(self.kernel_derivatives(nu)[0])
    }

    /// Half-width of the kernel's main lobe at `drop_db` below its maximum: the
    /// range an isolated tone occupies at that level. The main lobe ends at the
    /// first null, 2*pi/T (rectangular) or 4*pi/T (Hann).
    pub fn kernel_range(&self, drop_db: f64) -> Result<f64, Error> {
        if !drop_db.is_finite() || drop_db <= 0.0 {
            return Err(Error::InvalidParameter("FAFT range level"));
        }
        let level = 10_f64.powf(-drop_db / 20.0);
        let (mut lo, mut hi) = (
            0.0,
            match self.weighting {
                Weighting::Rectangular => self.resolution(),
                Weighting::Hann => 2.0 * self.resolution(),
            },
        );
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if mid <= lo || mid >= hi {
                break;
            }
            if self.kernel_derivatives(mid)[0] >= level {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Ok(lo)
    }

    /// `integral y(t)^2 w(t) dt` over the window, in amplitude^2 * samples.
    /// Stacked rotations are summed before squaring. 16-point Gauss-Legendre
    /// panels with at most pi of phase each (2*|rate| plus the Hann rate), so the
    /// result is exact to rounding; it is a once-per-window scalar, not a
    /// transform value.
    pub fn energy(&self) -> Result<f64, Error> {
        let nodes = gauss_legendre();
        let shift = self.resolution();
        let mut total = 0.0;
        let mut i = 0;
        while i < self.atoms.len() {
            let mut j = i + 1;
            while j < self.atoms.len() && self.atoms[j].start == self.atoms[i].start {
                j += 1;
            }
            let group = &self.atoms[i..j];
            let start = group[0].start as f64;
            let length = group[0].length as f64;
            let rate = group.iter().map(|a| a.rate.abs()).fold(0.0, f64::max);
            let hann = match self.weighting {
                Weighting::Rectangular => 0.0,
                Weighting::Hann => shift,
            };
            let panels = ((2.0 * rate + hann) * length / PI).ceil().max(1.0);
            if !panels.is_finite() || panels > 1e8 {
                return Err(Error::InvalidParameter("FAFT energy panels"));
            }
            let panels = panels as usize;
            let width = length / panels as f64;
            let mut sum = 0.0;
            for p in 0..panels {
                for &(x, weight) in &nodes {
                    let u = width * (p as f64 + x);
                    let y: f64 = group
                        .iter()
                        .map(|a| {
                            let turn = a.rate * u;
                            a.value - 2.0 * a.a * (0.5 * turn).sin().powi(2)
                                + a.b * turn.sin()
                                + a.slope * u
                        })
                        .sum();
                    let w = match self.weighting {
                        Weighting::Rectangular => 1.0,
                        Weighting::Hann => 0.5 - 0.5 * (shift * (start + u)).cos(),
                    };
                    sum += weight * y * y * w;
                }
            }
            total += sum * width;
            i = j;
        }
        if total.is_finite() {
            Ok(total)
        } else {
            Err(Error::InvalidParameter("FAFT energy overflow"))
        }
    }

    /// Centred normalized transform value (window-centre phase origin).
    fn centred_value(&self, omega: f64, table: &mut Vec<Complex>) -> Complex {
        let half = 0.5 * self.window.len() as f64;
        scale(
            mul(self.weighted(omega, table), phase(omega * half)),
            1.0 / self.weight_sum,
        )
    }

    fn to_start(&self, omega: f64, centred: Complex) -> Complex {
        mul(centred, phase(-omega * 0.5 * self.window.len() as f64))
    }

    /// Centred transform `[F, F', F'']` of one fitted tone at frequency `omega`.
    fn tone(&self, w: &Work, omega: f64) -> [Complex; 3] {
        let below = self.kernel_derivatives(omega - w.omega);
        let above = self.kernel_derivatives(omega + w.omega);
        let image = conj(w.value);
        [0, 1, 2].map(|k| add(scale(w.value, below[k]), scale(image, above[k])))
    }

    fn model(&self, comps: &[Work], omega: f64) -> Complex {
        comps.iter().fold(Complex::default(), |sum, w| {
            add(sum, self.tone(w, omega)[0])
        })
    }

    /// Joint weighted least squares of the cosine and sine atoms at each
    /// component's frequency (block diagonal for a symmetric weight). Writes the
    /// fitted amplitudes and returns `(explained normalized projection, dropped)`.
    fn solve(&self, comps: &mut [Work], significance: bool) -> Result<(f64, usize), Error> {
        let n = comps.len();
        let mut explained = 0.0;
        let mut dropped = 0;
        let mut alpha = buffer(n)?;
        alpha.resize(n, 0.0);
        let mut beta = alpha.clone();
        // Diagonals of the inverse Gram blocks: removing atom p raises the
        // residual by x_p^2 / (G^-1)_pp (Schur complement).
        let mut unique = buffer(n)?;
        unique.resize(n, 0.0);
        for (sine, out) in [(false, &mut alpha), (true, &mut beta)] {
            let mut gram = buffer(n * n)?;
            gram.resize(n * n, 0.0);
            let mut rhs = buffer(n)?;
            for p in 0..n {
                let wp = comps[p].omega;
                for q in 0..=p {
                    let wq = comps[q].omega;
                    let minus = self.kernel_derivatives(wp - wq)[0];
                    let plus = self.kernel_derivatives(wp + wq)[0];
                    let g = 0.5 * if sine { minus - plus } else { minus + plus };
                    gram[p * n + q] = g;
                    gram[q * n + p] = g;
                }
                rhs.push(if sine {
                    -comps[p].observed.im
                } else {
                    comps[p].observed.re
                });
            }
            let mut inverse = significance.then(|| vec![0.0; n]);
            dropped += cholesky_solve(&mut gram, &rhs, n, out, inverse.as_deref_mut());
            explained += out.iter().zip(&rhs).map(|(x, b)| x * b).sum::<f64>();
            if let Some(inverse) = inverse {
                for p in 0..n {
                    if inverse[p] > 0.0 {
                        unique[p] += out[p] * out[p] / inverse[p];
                    }
                }
            }
        }
        for (p, w) in comps.iter_mut().enumerate() {
            w.value = Complex {
                re: 0.5 * alpha[p],
                im: -0.5 * beta[p],
            };
            if significance {
                w.significance = (0.5 * unique[p]).sqrt();
            }
            if !(w.value.re.is_finite() && w.value.im.is_finite()) {
                return Err(Error::InvalidParameter("FAFT fit overflow"));
            }
        }
        Ok((explained, dropped))
    }

    fn summarize(&self, energy: f64, explained: f64) -> (f64, f64) {
        let residual = (energy - self.weight_sum * explained).max(0.0);
        (residual, (residual / self.weight_sum).sqrt())
    }

    /// Joint least-squares fit of real tones at `omegas` (finite, nonnegative,
    /// distinct). Each fitted value is the component's own height; `observed` is
    /// the transform there. Cost: one transform value per frequency, O(n^3) solve.
    pub fn fit(&self, omegas: &[f64]) -> Result<SpectralFit, Error> {
        if omegas.len() > 1024 || omegas.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(Error::InvalidParameter("FAFT fit frequencies"));
        }
        let mut sorted = omegas.to_vec();
        sorted.sort_by(f64::total_cmp);
        if sorted.windows(2).any(|p| p[0] == p[1]) {
            return Err(Error::InvalidParameter("FAFT fit frequencies"));
        }
        let mut table = Vec::with_capacity(self.lengths.len());
        let mut comps = buffer(omegas.len())?;
        for &omega in omegas {
            comps.push(Work::new(
                omega,
                super::checked(self.centred_value(omega, &mut table))?,
                Complex::default(),
                0,
            ));
        }
        let energy = self.energy()?;
        let (explained, degenerate) = self.solve(&mut comps, true)?;
        let (residual_energy, bound) = self.summarize(energy, explained);
        Ok(SpectralFit {
            components: comps.iter().map(|w| self.component(w)).collect(),
            energy,
            residual_energy,
            bound,
            degenerate,
        })
    }

    fn component(&self, w: &Work) -> Component {
        Component {
            omega: w.omega,
            value: self.to_start(w.omega, w.value),
            observed: self.to_start(w.omega, w.observed),
            significance: w.significance,
        }
    }

    /// The range around a local maximum at `omega` where |FAFT| stays within
    /// `drop_db` of its value there, measured on the transform itself (not the
    /// ideal kernel). A side that meets a valley above the level stops at the
    /// valley minimum and is flagged: the neighbouring maximum shares that range.
    /// Walk in steps of 2*pi/T/16, then bisection (crossing) or ternary search
    /// (valley); at most 64*2*pi/T on each side.
    pub fn peak_range(&self, omega: f64, drop_db: f64) -> Result<PeakRange, Error> {
        if !omega.is_finite() || !drop_db.is_finite() || drop_db <= 0.0 {
            return Err(Error::InvalidParameter("FAFT range level"));
        }
        let mut table = Vec::with_capacity(self.lengths.len());
        let mut at = |x: f64| norm(self.weighted(x, &mut table)) / self.weight_sum;
        let magnitude = at(omega);
        if !magnitude.is_finite() || magnitude == 0.0 {
            return Err(Error::InvalidParameter("FAFT range at a zero value"));
        }
        let level = magnitude * 10_f64.powf(-drop_db / 20.0);
        let step = self.resolution() / 16.0;
        let mut edge = [(0.0, false); 2];
        for (side, sign) in [(0, -1.0), (1, 1.0)] {
            let (mut previous, mut before) = (omega, magnitude);
            let mut found = None;
            for k in 1..=1024 {
                let x = omega + sign * step * k as f64;
                let m = at(x);
                if !m.is_finite() {
                    return Err(Error::InvalidParameter("FAFT arithmetic overflow"));
                }
                if m < level {
                    // Crossing between previous (>= level) and x (< level).
                    let (mut inside, mut outside) = (previous, x);
                    for _ in 0..60 {
                        let mid = 0.5 * (inside + outside);
                        if at(mid) >= level {
                            inside = mid;
                        } else {
                            outside = mid;
                        }
                    }
                    found = Some((inside, false));
                    break;
                }
                if m > before && k > 1 {
                    // Rising again above the level: a valley lies in the last
                    // two steps. Locate its minimum by ternary search.
                    let (mut a, mut b) = (x - 2.0 * sign * step, x);
                    for _ in 0..80 {
                        let (c1, c2) = (a + (b - a) / 3.0, b - (b - a) / 3.0);
                        if at(c1) <= at(c2) {
                            b = c2;
                        } else {
                            a = c1;
                        }
                    }
                    found = Some((0.5 * (a + b), true));
                    break;
                }
                previous = x;
                before = m;
            }
            edge[side] = found.ok_or(Error::InvalidParameter("FAFT range exceeds search"))?;
        }
        Ok(PeakRange {
            omega,
            magnitude,
            level,
            lower: edge[0].0,
            upper: edge[1].0,
            lower_valley: edge[0].1,
            upper_valley: edge[1].1,
        })
    }

    /// Newton on the residual r: the transform minus the given components. The
    /// criterion is the energy a single real tone at omega explains in
    /// r, `P = (Re r)^2/gc + (Im r)^2/gs` with `gc, gs = (1 +- k(2*omega))/2` the
    /// cosine and sine atom norms: the concentrated least-squares frequency, which
    /// accounts for the tone's own image at -omega. Within one resolution of DC,
    /// where the sine atom vanishes, it is |r|^2. Returns the refined frequency, the
    /// transform and the residual there.
    fn refine_residual(
        &self,
        seed: Seed,
        params: &PeakParams,
        comps: &[Work],
        table: &mut Vec<Complex>,
        stats: &mut PeakStats,
    ) -> Result<Option<(f64, Complex, Complex)>, Error> {
        let mut last = (Complex::default(), Complex::default());
        let concentrated = seed.lo >= self.resolution();
        let found = self.newton(seed, params, stats, |omega| {
            let raw = self.centered(omega, table)?;
            let mut d = raw;
            for w in comps {
                let m = self.tone(w, omega);
                for k in 0..3 {
                    d[k] = sub(d[k], m[k]);
                }
            }
            last = (raw[0], d[0]);
            if !concentrated {
                return Ok(super::power_derivatives(&d));
            }
            let image = self.kernel_derivatives(2.0 * omega);
            // d/domega k(2*omega) = 2k'(2*omega); the atom norms carry a factor 1/2.
            let (c0, c1, c2) = (0.5 * (1.0 + image[0]), image[1], 2.0 * image[2]);
            let atom = |x: [f64; 3], g: [f64; 3]| {
                let [u, u1, u2] = x;
                let [g0, g1, g2] = g;
                [
                    2.0 * u * u1 / g0 - u * u * g1 / (g0 * g0),
                    2.0 * (u1 * u1 + u * u2) / g0 - 4.0 * u * u1 * g1 / (g0 * g0)
                        + u * u * (2.0 * g1 * g1 / (g0 * g0 * g0) - g2 / (g0 * g0)),
                ]
            };
            let cosine = atom([d[0].re, d[1].re, d[2].re], [c0, c1, c2]);
            let sine = atom([d[0].im, d[1].im, d[2].im], [1.0 - c0, -c1, -c2]);
            Ok([cosine[0] + sine[0], cosine[1] + sine[1]])
        })?;
        Ok(found.map(|(omega, _, _)| (omega, last.0, last.1)))
    }

    /// A new component's provisional amplitude from its residual value with the
    /// others fixed: the diagonal 2x2 normal equations of its cosine and sine atoms.
    fn own_amplitude(&self, omega: f64, residual: Complex) -> Complex {
        let image = self.kernel_derivatives(2.0 * omega)[0];
        let (gc, gs) = (0.5 * (1.0 + image), 0.5 * (1.0 - image));
        let alpha = residual.re / gc;
        let beta = if gs > 1e-9 { -residual.im / gs } else { 0.0 };
        Complex {
            re: 0.5 * alpha,
            im: -0.5 * beta,
        }
    }

    /// Centred F and F' at every active component (`observed`, `slope`).
    fn differentiate(
        &self,
        comps: &mut [Work],
        table: &mut Vec<Complex>,
        stats: &mut PursuitStats,
    ) -> Result<(), Error> {
        for w in comps.iter_mut().filter(|w| w.active) {
            let d = self.centered(w.omega, table)?;
            w.observed = d[0];
            w.slope = d[1];
            stats.search.derivative_evaluations += 1;
        }
        Ok(())
    }

    /// Damped Gauss-Newton frequency steps for the joint least squares of all
    /// tones. Atoms of component p (around the window centre, u = t - tc) are
    /// C = cos(w u), S = sin(w u) and D = u(-alpha S + beta C), the derivative of
    /// the tone in w. With k the normalized kernel, every inner product is closed:
    /// <C_p,C_q> = (k(d)+k(s))/2, <S_p,S_q> = (k(d)-k(s))/2, <C_p,S_q> = 0,
    /// <C_p,D_q> = alpha_q (k'(s)-k'(d))/2, <S_p,D_q> = -beta_q (k'(d)+k'(s))/2,
    /// <D_p,D_q> = alpha_p alpha_q (k''(s)-k''(d))/2 - beta_p beta_q (k''(d)+k''(s))/2
    /// (d = w_p - w_q, s = w_p + w_q), and the data side needs F and F' at w_p:
    /// <y,C> = Re F, <y,S> = -Im F, <y,D> = alpha Re F' - beta Im F'. Components
    /// farther apart than `gap` are solved as separate clusters (their kernel
    /// coupling stays in the residual through the global model); clusters
    /// without an active member are left in place.
    fn gauss_newton_steps(
        &self,
        comps: &[Work],
        gap: f64,
        damping: f64,
    ) -> Result<Vec<f64>, Error> {
        let n = comps.len();
        let amp: Vec<(f64, f64)> = comps
            .iter()
            .map(|w| (2.0 * w.value.re, -2.0 * w.value.im))
            .collect();
        let mut steps = buffer(n)?;
        steps.resize(n, 0.0);
        let mut start = 0;
        while start < n {
            let mut end = start + 1;
            while end < n && comps[end].omega - comps[end - 1].omega <= gap {
                end += 1;
            }
            if !comps[start..end].iter().any(|w| w.active) {
                start = end;
                continue;
            }
            let m = end - start;
            let dim = 3 * m;
            let mut hessian = buffer(dim * dim)?;
            hessian.resize(dim * dim, 0.0);
            let mut gradient = buffer(dim)?;
            gradient.resize(dim, 0.0);
            for i in 0..m {
                let p = start + i;
                let (ap, bp) = amp[p];
                let (f, f1) = (comps[p].observed, comps[p].slope);
                let (mut gc, mut gs, mut gd) = (f.re, -f.im, ap * f1.re - bp * f1.im);
                for (q, w) in comps.iter().enumerate() {
                    let (aq, bq) = amp[q];
                    let d = self.kernel_derivatives(comps[p].omega - w.omega);
                    let s = self.kernel_derivatives(comps[p].omega + w.omega);
                    gc -= aq * 0.5 * (d[0] + s[0]);
                    gs -= bq * 0.5 * (d[0] - s[0]);
                    gd -= ap * aq * 0.5 * (d[1] + s[1]) + bp * bq * 0.5 * (d[1] - s[1]);
                    if q < start || q >= end {
                        continue;
                    }
                    let j = q - start;
                    let entries = [
                        (0, 0, 0.5 * (d[0] + s[0])),
                        (1, 1, 0.5 * (d[0] - s[0])),
                        (0, 2, aq * 0.5 * (s[1] - d[1])),
                        (1, 2, -bq * 0.5 * (d[1] + s[1])),
                        (
                            2,
                            2,
                            ap * aq * 0.5 * (s[2] - d[2]) - bp * bq * 0.5 * (d[2] + s[2]),
                        ),
                    ];
                    for (x, y, v) in entries {
                        hessian[(3 * i + x) * dim + 3 * j + y] = v;
                        if x != y {
                            // <D_q, C_p> = <C_p, D_q>: the transposed entry.
                            hessian[(3 * j + y) * dim + 3 * i + x] = v;
                        }
                    }
                }
                gradient[3 * i] = gc;
                gradient[3 * i + 1] = gs;
                gradient[3 * i + 2] = gd;
            }
            for k in 0..dim {
                hessian[k * dim + k] *= 1.0 + damping;
            }
            let mut delta = vec![0.0; dim];
            cholesky_solve(&mut hessian, &gradient, dim, &mut delta, None);
            for i in 0..m {
                steps[start + i] = delta[3 * i + 2];
            }
            start = end;
        }
        if steps.iter().all(|v| v.is_finite()) {
            Ok(steps)
        } else {
            Err(Error::InvalidParameter("FAFT polish overflow"))
        }
    }

    /// Joint refinement of every frequency with its amplitudes: a damped
    /// Gauss-Newton step on the full nonlinear least squares, then the exact
    /// amplitude solve at the new frequencies (variable projection). A step is
    /// kept only if the residual energy does not grow beyond its rounding;
    /// otherwise the damping grows. Steps are limited to half a resolution and to the band. Cost per
    /// step: one derivative evaluation per component and O(cluster^3) algebra.
    /// Returns the residual energy and bound; `comps` stays sorted by frequency.
    #[allow(clippy::too_many_arguments)]
    fn polish(
        &self,
        comps: &mut Vec<Work>,
        params: &PeakParams,
        (iterations, tolerance): (usize, f64),
        energy: f64,
        table: &mut Vec<Complex>,
        stats: &mut PursuitStats,
    ) -> Result<(f64, f64), Error> {
        comps.sort_by(|a, b| a.omega.total_cmp(&b.omega));
        let gap = self.coupling_gap();
        // Whole clusters around active members move together; each cluster gets
        // its own damped loop, so a hard cluster does not stall the others.
        let mut groups = 0;
        let mut start = 0;
        while start < comps.len() {
            let mut end = start + 1;
            while end < comps.len() && comps[end].omega - comps[end - 1].omega <= gap {
                end += 1;
            }
            let active = comps[start..end].iter().any(|w| w.active);
            for w in &mut comps[start..end] {
                w.active = active;
                w.group = if active { groups } else { usize::MAX };
            }
            groups += usize::from(active);
            start = end;
        }
        self.differentiate(comps, table, stats)?;
        let mut summary = self.refit(comps, energy, stats)?;
        for group in 0..groups {
            comps.iter_mut().for_each(|w| w.active = w.group == group);
            summary = self.polish_group(
                comps,
                params,
                (iterations, tolerance),
                (energy, summary),
                table,
                stats,
            )?;
        }
        comps.iter_mut().for_each(|w| {
            w.active = false;
            w.group = usize::MAX;
        });
        Ok(summary)
    }

    /// The damped Gauss-Newton loop of `polish` for the active components.
    fn polish_group(
        &self,
        comps: &mut Vec<Work>,
        params: &PeakParams,
        (iterations, tolerance): (usize, f64),
        (energy, mut summary): (f64, (f64, f64)),
        table: &mut Vec<Complex>,
        stats: &mut PursuitStats,
    ) -> Result<(f64, f64), Error> {
        let resolution = self.resolution();
        let gap = self.coupling_gap();
        let mut damping = 1e-3;
        for _ in 0..iterations {
            if !comps.iter().any(|w| w.active) {
                break;
            }
            let steps = self.gauss_newton_steps(comps, gap, damping)?;
            // Converged once every proposed step is within the tolerance: the
            // components stay and are not evaluated again.
            if comps
                .iter()
                .zip(&steps)
                .all(|(w, v)| !w.active || v.abs() <= tolerance * resolution)
            {
                break;
            }
            let mut trial = comps.clone();
            for (w, step) in trial.iter_mut().zip(&steps) {
                if w.active {
                    w.omega = (w.omega + step.clamp(-0.5 * resolution, 0.5 * resolution))
                        .clamp(params.lower, params.upper);
                }
            }
            trial.sort_by(|a, b| a.omega.total_cmp(&b.omega));
            self.differentiate(&mut trial, table, stats)?;
            let trial_summary = self.refit(&mut trial, energy, stats)?;
            // The residual energy is a difference E - W*explained with rounding of
            // order eps*E: a step within that level is accepted, so convergence is
            // decided by the step tolerance and not by energy rounding (which
            // would cap frequency precision near sqrt(eps)).
            if trial_summary.0 <= summary.0 + 64.0 * f64::EPSILON * energy {
                *comps = trial;
                summary = trial_summary;
                stats.polish_steps += 1;
                damping = (damping * 0.25).max(1e-12);
            } else {
                stats.polish_rejected += 1;
                damping *= 8.0;
                if damping > 1e8 {
                    break;
                }
            }
        }
        Ok(summary)
    }

    /// Distance beyond which tones are polished as separate clusters: the
    /// Hann kernel is below -48 dB beyond 4 resolutions; rectangular sidelobes
    /// decay slowly, so 8 resolutions.
    fn coupling_gap(&self) -> f64 {
        let resolutions = match self.weighting {
            Weighting::Rectangular => 8.0,
            Weighting::Hann => 4.0,
        };
        resolutions * self.resolution()
    }

    fn refit(
        &self,
        comps: &mut [Work],
        energy: f64,
        stats: &mut PursuitStats,
    ) -> Result<(f64, f64), Error> {
        let (explained, dropped) = self.solve(comps, false)?;
        stats.fits += 1;
        stats.degenerate += dropped;
        Ok(self.summarize(energy, explained))
    }

    /// Remove, one at a time, the least significant component while its
    /// significance (unique energy as an isolated magnitude) is below `floor`,
    /// refitting after each removal: leakage of others and duplicates sharing
    /// one tone's energy go, while close but distinct tones stay. Returns the
    /// removed frequencies and seed positions; neighbours become active again.
    fn prune(
        &self,
        comps: &mut Vec<Work>,
        floor: f64,
        energy: f64,
        stats: &mut PursuitStats,
    ) -> Result<(Vec<f64>, (f64, f64)), Error> {
        let mut removed = Vec::new();
        loop {
            let (explained, dropped) = self.solve(comps, true)?;
            stats.fits += 1;
            stats.degenerate += dropped;
            let summary = self.summarize(energy, explained);
            let weakest = comps
                .iter()
                .enumerate()
                .filter(|(_, w)| !(w.significance >= floor && w.significance > 0.0))
                .min_by(|a, b| a.1.significance.total_cmp(&b.1.significance))
                .map(|(i, _)| i);
            let Some(i) = weakest else {
                comps.sort_by(|a, b| a.omega.total_cmp(&b.omega));
                return Ok((removed, summary));
            };
            let gone = comps.remove(i);
            removed.push(gone.omega);
            removed.push(gone.origin);
            let gap = self.coupling_gap();
            for other in comps.iter_mut() {
                if (other.omega - gone.omega).abs() <= gap {
                    other.active = true;
                }
            }
            stats.pruned += 1;
        }
    }

    /// Relevant frequencies as a jointly fitted tone model (see the module docs).
    /// The transform is scanned once on the duration grid; later rounds seed from
    /// local maxima of the residual on that grid, which costs O(components) kernel
    /// evaluations per point and no new transform values. A seed already explained
    /// by components accepted earlier in its round is skipped without Newton work.
    /// Frequencies and amplitudes are then polished jointly and components below
    /// the floor in significance are pruned. A rectangular sidelobe is therefore
    /// explained by its main tone instead of being reported, a weak tone hidden on
    /// a strong tone's flank appears in the residual, and two tones inside one
    /// main lobe can be separated when the data are tones (a model assumption, not
    /// added resolution of the transform).
    pub fn pursue(&self, params: PursuitParams) -> Result<Pursuit, Error> {
        params.validate()?;
        let p = params.search;
        let mut table = Vec::with_capacity(self.lengths.len());
        let mut stats = PursuitStats::default();
        let energy = self.energy()?;
        let density = p.scan_density.unwrap_or(1);
        let (step, values) =
            self.scan_grid(p.lower, p.upper, density, &mut table, &mut stats.search)?;
        let half = 0.5 * self.window.len() as f64;
        let mut grid = buffer(values.len())?;
        let mut omegas = buffer(values.len())?;
        for (k, z) in values.into_iter().enumerate() {
            let omega = p.lower + step * k as f64;
            omegas.push(omega);
            grid.push(super::checked(scale(
                mul(z, phase(omega * half)),
                1.0 / self.weight_sum,
            ))?);
        }
        let strongest = grid.iter().map(|z| norm(*z)).fold(0.0, f64::max);
        let floor = p.floor(strongest);
        let merge = p.merge_width.unwrap_or(0.25 * self.resolution());
        let mut comps: Vec<Work> = Vec::new();
        let mut rejected: Vec<f64> = Vec::new();
        let mut summary = (energy, (energy / self.weight_sum).sqrt());
        let mut stop = PursuitStop::Rounds;
        if energy == 0.0 || strongest == 0.0 {
            stop = PursuitStop::Silent;
        } else {
            let mut magnitude = buffer(grid.len())?;
            magnitude.resize(grid.len(), 0.0);
            for _ in 0..params.max_rounds {
                stats.rounds += 1;
                for (k, slot) in magnitude.iter_mut().enumerate() {
                    *slot = norm(sub(grid[k], self.model(&comps, omegas[k])));
                }
                let mut seeds = super::grid_seeds(&magnitude, p.lower, p.upper, step, floor)?;
                let near = |list: &[f64], w: f64| list.iter().any(|x| (x - w).abs() <= merge);
                seeds.retain(|s| {
                    !near(&rejected, s.omega)
                        && !comps.iter().any(|q| (q.omega - s.omega).abs() <= merge)
                });
                let height = |s: &Seed| {
                    let k = ((s.omega - p.lower) / step)
                        .round()
                        .clamp(0.0, (grid.len() - 1) as f64);
                    magnitude[k as usize]
                };
                seeds.sort_by(|a, b| height(b).total_cmp(&height(a)));
                stats.search.seeds += seeds.len();
                let fixed = comps.len();
                for seed in seeds {
                    if comps.len() >= p.max_peaks {
                        break;
                    }
                    // Components accepted earlier in this round carry provisional
                    // amplitudes: a seed they explain (a sidelobe) needs no Newton.
                    if comps.len() > fixed {
                        let k = ((seed.omega - p.lower) / step)
                            .round()
                            .clamp(0.0, (grid.len() - 1) as f64)
                            as usize;
                        if norm(sub(grid[k], self.model(&comps, omegas[k]))) < floor {
                            stats.search.excluded += 1;
                            continue;
                        }
                    }
                    let found =
                        self.refine_residual(seed, &p, &comps, &mut table, &mut stats.search)?;
                    let Some((omega, raw, residual)) = found else {
                        continue;
                    };
                    if comps.iter().any(|q| (q.omega - omega).abs() <= merge) {
                        stats.search.merged += 1;
                        continue;
                    }
                    let mut work = Work::new(omega, raw, self.own_amplitude(omega, residual), 0);
                    work.origin = seed.omega;
                    comps.push(work);
                }
                if comps.len() == fixed {
                    stop = PursuitStop::NoSeeds;
                    break;
                }
                let iterations = (params.polish_iterations, params.polish_tolerance);
                self.polish(&mut comps, &p, iterations, energy, &mut table, &mut stats)?;
                let (removed, mut s) = self.prune(&mut comps, floor, energy, &mut stats)?;
                if !removed.is_empty() && iterations.0 > 0 {
                    // Removed components had pulled their neighbours (now active):
                    // polish those clusters again.
                    self.polish(&mut comps, &p, iterations, energy, &mut table, &mut stats)?;
                    let (more, again) = self.prune(&mut comps, floor, energy, &mut stats)?;
                    rejected.extend(more);
                    s = again;
                }
                rejected.extend(removed);
                summary = s;
                if summary.1 <= floor {
                    stop = PursuitStop::Certified;
                    break;
                }
                if comps.len() >= p.max_peaks {
                    stop = PursuitStop::Capacity;
                    break;
                }
            }
        }
        if !comps.is_empty() && params.polish_iterations > 0 && stop != PursuitStop::Silent {
            // Final pass over every cluster: distant clusters settled in earlier
            // rounds still carried the weak leakage of models refined later.
            comps.iter_mut().for_each(|w| w.active = true);
            let iterations = (params.polish_iterations, params.polish_tolerance);
            self.polish(&mut comps, &p, iterations, energy, &mut table, &mut stats)?;
            summary = self.prune(&mut comps, floor, energy, &mut stats)?.1;
            if stop == PursuitStop::Certified && summary.1 > floor {
                stop = PursuitStop::NoSeeds;
            }
        }
        comps.sort_by(|a, b| a.omega.total_cmp(&b.omega));
        Ok(Pursuit {
            components: comps.iter().map(|w| self.component(w)).collect(),
            energy,
            residual_energy: summary.0,
            bound: summary.1,
            strongest,
            floor,
            stop,
            stats,
        })
    }
}

/// Cholesky solve of the symmetric `n x n` system, in place. An atom whose pivot
/// falls below 1e-10 of its diagonal (numerically dependent on earlier atoms,
/// e.g. a sine at omega = 0) gets coefficient 0. Returns how many were dropped.
fn cholesky_solve(
    gram: &mut [f64],
    rhs: &[f64],
    n: usize,
    out: &mut [f64],
    inverse_diagonal: Option<&mut [f64]>,
) -> usize {
    let mut dropped = 0;
    let mut active = vec![true; n];
    for i in 0..n {
        let diagonal = gram[i * n + i];
        let mut pivot = diagonal;
        for k in 0..i {
            pivot -= gram[i * n + k] * gram[i * n + k];
        }
        if pivot.is_nan() || pivot <= 1e-10 * diagonal.abs() || diagonal <= 0.0 {
            active[i] = false;
            dropped += 1;
            for k in 0..n {
                gram[i * n + k] = 0.0;
            }
            continue;
        }
        let root = pivot.sqrt();
        gram[i * n + i] = root;
        for j in i + 1..n {
            let mut v = gram[j * n + i];
            for k in 0..i {
                v -= gram[j * n + k] * gram[i * n + k];
            }
            gram[j * n + i] = v / root;
        }
    }
    // diag(G^-1)_i = |L^-1 e_i|^2 over the active atoms (0 for dropped ones).
    if let Some(diagonal) = inverse_diagonal {
        let mut z = vec![0.0; n];
        for i in 0..n {
            diagonal[i] = 0.0;
            if !active[i] {
                continue;
            }
            let mut sum = 0.0;
            for k in i..n {
                z[k] = 0.0;
                if !active[k] {
                    continue;
                }
                let mut v = if k == i { 1.0 } else { 0.0 };
                for m in i..k {
                    v -= gram[k * n + m] * z[m];
                }
                z[k] = v / gram[k * n + k];
                sum += z[k] * z[k];
            }
            diagonal[i] = sum;
        }
    }
    // Forward then backward substitution on the active atoms.
    let mut y = vec![0.0; n];
    for i in 0..n {
        if active[i] {
            let mut v = rhs[i];
            for k in 0..i {
                v -= gram[i * n + k] * y[k];
            }
            y[i] = v / gram[i * n + i];
        }
    }
    for i in (0..n).rev() {
        out[i] = 0.0;
        if active[i] {
            let mut v = y[i];
            for k in i + 1..n {
                v -= gram[k * n + i] * out[k];
            }
            out[i] = v / gram[i * n + i];
        }
    }
    dropped
}

#[derive(Debug, Clone, Copy)]
pub struct TrackerParams {
    /// Search used on scanning frames; its relative floor is the survival floor.
    pub pursuit: PursuitParams,
    /// Relative floor a component must reach to start a track (hysteresis: at
    /// least the survival floor).
    pub birth_floor: f64,
    /// Largest frequency change per frame for continuing a track (radians per
    /// sample); None uses 2*pi/T of the current window.
    pub max_jump: Option<f64>,
    /// Frames a track survives unmatched before it ends.
    pub hold: usize,
    /// A full pursuit runs at least every this many frames (1 = every frame).
    pub rescan_every: usize,
}
impl Default for TrackerParams {
    fn default() -> Self {
        Self {
            pursuit: PursuitParams::default(),
            birth_floor: 0.02,
            max_jump: None,
            hold: 1,
            rescan_every: 8,
        }
    }
}
impl TrackerParams {
    fn validate(&self) -> Result<(), Error> {
        self.pursuit.validate()?;
        if !self.birth_floor.is_finite()
            || self.birth_floor < self.pursuit.search.relative_floor
            || self.birth_floor > 1.0
            || self.max_jump.is_some_and(|j| !j.is_finite() || j <= 0.0)
            || self.rescan_every == 0
        {
            return Err(Error::InvalidParameter("FAFT tracker parameters"));
        }
        Ok(())
    }
}

/// Why a tracker frame ran a full pursuit, or that it did not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FramePath {
    /// No live track yet.
    First,
    /// `rescan_every` frames since the last scan.
    Interval,
    /// The tracked residual energy grew by more than a birth-floor tone's energy.
    Novelty,
    /// Tracked only. `certified`: the residual bound is below the survival floor,
    /// so no residual value above it exists at any frequency.
    Tracked { certified: bool },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackedPeak {
    /// Stable identity across frames.
    pub id: u64,
    pub omega: f64,
    /// Fitted and observed values, as in `Component`.
    pub value: Complex,
    pub observed: Complex,
    /// Frames since the track started, 1 on its first frame.
    pub age: usize,
}

#[derive(Debug, Clone)]
pub struct TrackedFrame {
    /// Tracks matched in this frame, sorted by frequency.
    pub peaks: Vec<TrackedPeak>,
    pub born: usize,
    pub ended: usize,
    pub path: FramePath,
    pub energy: f64,
    pub residual_energy: f64,
    pub bound: f64,
    pub stats: PursuitStats,
}

#[derive(Debug, Clone, Copy)]
struct Track {
    id: u64,
    omega: f64,
    previous: Option<f64>,
    value: Complex,
    observed: Complex,
    age: usize,
    missed: usize,
}
impl Track {
    fn predict(&self, jump: f64) -> f64 {
        self.omega
            + self
                .previous
                .map_or(0.0, |p| (self.omega - p).clamp(-0.5 * jump, 0.5 * jump))
    }
}

/// Peak selection across consecutive windows of one input. Each frame polishes the
/// previous components jointly from their predicted frequencies (no scan),
/// prunes by significance, and accepts the result when the residual certificate
/// is below the survival floor or the residual energy did not grow by the energy
/// of a birth-floor tone over the best residual since the last scan; otherwise,
/// and every `rescan_every` frames, it runs a full `pursue`. Tracks continue within `max_jump`, start only
/// above `birth_floor`, survive down to the pursuit floor and end after `hold`
/// missed frames. The novelty test is a heuristic: a birth that coincides with an
/// equal fall of residual energy elsewhere waits for the next interval scan.
#[derive(Debug, Clone)]
pub struct PeakTracker {
    params: TrackerParams,
    tracks: Vec<Track>,
    next_id: u64,
    since_scan: usize,
    reference: Option<f64>,
}
impl PeakTracker {
    pub fn new(params: TrackerParams) -> Result<Self, Error> {
        params.validate()?;
        Ok(Self {
            params,
            tracks: Vec::new(),
            next_id: 0,
            since_scan: 0,
            reference: None,
        })
    }

    /// Analyse the next window. Frequencies are in radians per source sample, so
    /// windows may advance by any hop; equal durations keep the floors comparable.
    pub fn update(&mut self, faft: &Faft) -> Result<TrackedFrame, Error> {
        let p = self.params.pursuit.search;
        let jump = self.params.max_jump.unwrap_or(faft.resolution());
        let energy = faft.energy()?;
        let mut stats = PursuitStats::default();
        let path = if self.tracks.is_empty() {
            FramePath::First
        } else if self.since_scan + 1 >= self.params.rescan_every {
            FramePath::Interval
        } else {
            match self.tracked(faft, jump, energy, &mut stats)? {
                Some(frame) => return Ok(frame),
                None => FramePath::Novelty,
            }
        };
        let pursuit = faft.pursue(self.params.pursuit)?;
        stats.absorb(&pursuit.stats);
        let birth = p
            .absolute_floor
            .max(self.params.birth_floor * pursuit.strongest);
        let mut order: Vec<&Component> = pursuit.components.iter().collect();
        order.sort_by(|a, b| b.magnitude().total_cmp(&a.magnitude()));
        let predicted: Vec<f64> = self.tracks.iter().map(|t| t.predict(jump)).collect();
        let mut matched = vec![false; self.tracks.len()];
        let mut born = 0;
        let mut fresh = Vec::new();
        for c in order {
            let nearest = predicted
                .iter()
                .enumerate()
                .filter(|&(i, w)| !matched[i] && (w - c.omega).abs() <= jump)
                .min_by(|a, b| (a.1 - c.omega).abs().total_cmp(&(b.1 - c.omega).abs()));
            if let Some((i, _)) = nearest {
                matched[i] = true;
                let t = &mut self.tracks[i];
                t.previous = Some(t.omega);
                t.omega = c.omega;
                t.value = c.value;
                t.observed = c.observed;
                t.age += 1;
                t.missed = 0;
            } else if c.magnitude() >= birth {
                born += 1;
                fresh.push(Track {
                    id: self.next_id,
                    omega: c.omega,
                    previous: None,
                    value: c.value,
                    observed: c.observed,
                    age: 1,
                    missed: 0,
                });
                self.next_id += 1;
            }
        }
        for (t, m) in self.tracks.iter_mut().zip(&matched) {
            if !m {
                t.missed += 1;
            }
        }
        self.tracks.extend(fresh);
        self.since_scan = 0;
        self.reference = Some(pursuit.residual_energy);
        let ended = self.retire();
        Ok(TrackedFrame {
            peaks: self.report(),
            born,
            ended,
            path,
            energy,
            residual_energy: pursuit.residual_energy,
            bound: pursuit.bound,
            stats,
        })
    }

    /// The scan-free path; None asks for a full pursuit (novelty).
    fn tracked(
        &mut self,
        faft: &Faft,
        jump: f64,
        energy: f64,
        stats: &mut PursuitStats,
    ) -> Result<Option<TrackedFrame>, Error> {
        let p = self.params.pursuit.search;
        let mut table = Vec::with_capacity(faft.lengths.len());
        // Start every track at its predicted frequency; amplitudes are re-solved
        // on this window (previous phases belong to another window).
        let mut work = buffer(self.tracks.len())?;
        let mut predicted = buffer(self.tracks.len())?;
        for (i, t) in self.tracks.iter().enumerate() {
            let omega = t.predict(jump).clamp(p.lower, p.upper);
            predicted.push(omega);
            work.push(Work::new(omega, Complex::default(), Complex::default(), i));
        }
        let polish = (
            self.params.pursuit.polish_iterations,
            self.params.pursuit.polish_tolerance,
        );
        faft.polish(&mut work, &p, polish, energy, &mut table, stats)?;
        // A track that wandered beyond the jump limit is not continued here.
        work.retain(|w| (w.omega - predicted[w.tag]).abs() <= jump);
        let strongest = work.iter().map(|w| norm(w.observed)).fold(0.0, f64::max);
        let floor = p.floor(strongest);
        let (_, (residual, bound)) = faft.prune(&mut work, floor, energy, stats)?;
        let certified = bound <= floor;
        let birth = p.absolute_floor.max(self.params.birth_floor * strongest);
        let novel = match self.reference {
            Some(reference) => residual > reference + 2.0 * birth * birth * faft.weight_sum,
            None => true,
        };
        if !certified && novel {
            return Ok(None);
        }
        // Accept: surviving tracks continue; others miss this frame.
        let mut live = vec![false; self.tracks.len()];
        for w in &work {
            live[w.tag] = true;
            let t = &mut self.tracks[w.tag];
            t.previous = Some(t.omega);
            t.omega = w.omega;
            t.value = faft.to_start(w.omega, w.value);
            t.observed = faft.to_start(w.omega, w.observed);
            t.age += 1;
            t.missed = 0;
        }
        for (t, l) in self.tracks.iter_mut().zip(&live) {
            if !l {
                t.missed += 1;
            }
        }
        self.since_scan += 1;
        // Novelty is growth over the best-explained state since the last scan,
        // so a transient scan frame does not loosen the test for later frames.
        self.reference = Some(self.reference.map_or(residual, |r| r.min(residual)));
        let ended = self.retire();
        Ok(Some(TrackedFrame {
            peaks: self.report(),
            born: 0,
            ended,
            path: FramePath::Tracked { certified },
            energy,
            residual_energy: residual,
            bound,
            stats: *stats,
        }))
    }

    fn retire(&mut self) -> usize {
        let hold = self.params.hold;
        let before = self.tracks.len();
        self.tracks.retain(|t| t.missed <= hold);
        before - self.tracks.len()
    }

    fn report(&self) -> Vec<TrackedPeak> {
        let mut out: Vec<TrackedPeak> = self
            .tracks
            .iter()
            .filter(|t| t.missed == 0)
            .map(|t| TrackedPeak {
                id: t.id,
                omega: t.omega,
                value: t.value,
                observed: t.observed,
                age: t.age,
            })
            .collect();
        out.sort_by(|a, b| a.omega.total_cmp(&b.omega));
        out
    }

    /// Live tracks (including those inside their hold), for inspection.
    pub fn live_tracks(&self) -> usize {
        self.tracks.len()
    }
}
