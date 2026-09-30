//! Bounded, nonuniform frequency search over independently centered wave circles.
//! This locates peaks of a finite-window predictor's DTFT, not a unique inverse
//! decomposition into physical oscillators. Heuristic stops are reported explicitly.
use crate::geometric_faft::GeometricSpectrum;
use crate::spectral::Complex;
use crate::{buffer, Error};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct FrequencyParams {
    pub lower: f64,
    pub upper: f64,
    /// Initial equal intervals, independent of output peak count.
    pub initial_intervals: usize,
    pub max_depth: usize,
    /// Hard combined coarse + fine oracle-call budget, including initial probes.
    pub max_evaluations: usize,
    /// Radians per source sample. Refinement and merging have separate limits.
    pub min_width: f64,
    pub merge_width: f64,
    pub absolute_floor: f64,
    pub relative_floor: f64,
    /// Stop sampled-quiet or non-peaked, nearly linear intervals after depth two.
    /// This heuristic can miss an unsampled narrow peak; false uses bounds only.
    pub prune_flat: bool,
    pub relative_gain: f64,
}
impl Default for FrequencyParams {
    fn default() -> Self {
        Self {
            lower: 0.0,
            upper: PI,
            initial_intervals: 128,
            max_depth: 12,
            max_evaluations: 8192,
            min_width: 1e-5,
            merge_width: 1e-3,
            absolute_floor: 0.0,
            relative_floor: 0.01,
            prune_flat: true,
            relative_gain: 0.002,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    BelowFloor,
    LowGain,
    Width,
    Depth,
    Budget,
}
#[derive(Debug, Clone)]
pub struct FrequencyBand {
    pub lower: f64,
    pub upper: f64,
    /// Conservative upper magnitude from nearest sampled points and a centered
    /// derivative bound. Valid for the prepared predictor, not the original audio.
    pub magnitude_upper: f64,
    pub depth: usize,
    pub reason: StopReason,
}
#[derive(Debug, Clone, Copy)]
pub struct FrequencyPoint {
    pub omega: f64,
    pub value: Complex,
    pub uncertainty: f64,
    pub verified: bool,
}
impl FrequencyPoint {
    pub fn magnitude(self) -> f64 {
        self.value.re.hypot(self.value.im)
    }
    fn lower(self) -> f64 {
        (self.magnitude() - self.uncertainty).max(0.0)
    }
    fn upper(self) -> f64 {
        self.magnitude() + self.uncertainty
    }
}
#[derive(Debug, Clone)]
pub struct FrequencyPeak {
    pub omega: f64,
    pub value: Complex,
    /// Neighboring queried frequencies; no statistical frequency error claim.
    pub bracket: (f64, f64),
    pub merged_candidates: usize,
}
#[derive(Debug, Clone, Default)]
pub struct FrequencyStats {
    pub coarse_queries: usize,
    pub fine_queries: usize,
    pub skipped_fine: usize,
    pub disagreement_bound: f64,
    pub observed_peak: f64,
    pub budget_exhausted: bool,
    pub peak_refinements: usize,
}
#[derive(Debug, Clone)]
pub struct AdaptiveSpectrum {
    pub points: Vec<FrequencyPoint>,
    pub peaks: Vec<FrequencyPeak>,
    pub bands: Vec<FrequencyBand>,
    pub stats: FrequencyStats,
}
#[derive(Clone, Copy)]
struct Node {
    points: [FrequencyPoint; 3],
    depth: usize,
}
impl Node {
    fn score(self) -> f64 {
        self.points.iter().map(|p| p.upper()).fold(0.0, f64::max)
    }
    fn width(self) -> f64 {
        self.points[2].omega - self.points[0].omega
    }
}
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Node {}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score()
            .total_cmp(&other.score())
            .then_with(|| self.width().total_cmp(&other.width()))
            .then_with(|| self.points[0].omega.total_cmp(&other.points[0].omega))
            .then_with(|| self.depth.cmp(&other.depth))
    }
}
struct Search<'a> {
    target: &'a GeometricSpectrum,
    coarse: Option<&'a GeometricSpectrum>,
    params: FrequencyParams,
    stats: FrequencyStats,
    cache: BTreeMap<u64, FrequencyPoint>,
}
impl Search<'_> {
    fn floor(&self) -> f64 {
        self.params
            .absolute_floor
            .max(self.params.relative_floor * self.stats.observed_peak)
    }
    fn query(&mut self, omega: f64) -> Result<Option<FrequencyPoint>, Error> {
        if let Some(&p) = self.cache.get(&omega.to_bits()) {
            return Ok(Some(p));
        }
        let used = self.stats.coarse_queries + self.stats.fine_queries;
        if used == self.params.max_evaluations {
            self.stats.budget_exhausted = true;
            return Ok(None);
        }
        let mut point = if let Some(coarse) = self.coarse {
            self.stats.coarse_queries += 1;
            FrequencyPoint {
                omega,
                value: coarse.value_rotations(omega)?,
                uncertainty: self.stats.disagreement_bound,
                verified: false,
            }
        } else {
            self.stats.fine_queries += 1;
            FrequencyPoint {
                omega,
                value: self.target.value_rotations(omega)?,
                uncertainty: 0.0,
                verified: true,
            }
        };
        // Even a skipped fine query retains a certified magnitude interval.
        self.stats.observed_peak = self.stats.observed_peak.max(point.lower());
        if !point.verified {
            if point.upper() < self.floor() {
                self.stats.skipped_fine += 1;
            } else if self.stats.coarse_queries + self.stats.fine_queries
                < self.params.max_evaluations
            {
                point.value = self.target.value_rotations(omega)?;
                point.uncertainty = 0.0;
                point.verified = true;
                self.stats.fine_queries += 1;
                self.stats.observed_peak = self.stats.observed_peak.max(point.lower());
            } else {
                self.stats.budget_exhausted = true;
            }
        }
        self.cache.insert(omega.to_bits(), point);
        Ok(Some(point))
    }
}
impl AdaptiveSpectrum {
    /// Search arbitrary frequencies in `[0,pi]`. Optional coarse geometry avoids
    /// some fine queries using a measured predictor-disagreement bound, including
    /// at zoom >1. The two prepared oracles must use the SAME window/grid/weights.
    /// Refinement uses midpoint arithmetic and magnitude comparisons. Finite-sum
    /// oracles still require trig at new frequencies; no all-arithmetic speed claim.
    pub fn search(
        target: &GeometricSpectrum,
        coarse: Option<&GeometricSpectrum>,
        params: FrequencyParams,
    ) -> Result<Self, Error> {
        if !params.lower.is_finite()
            || !params.upper.is_finite()
            || params.lower < 0.0
            || params.lower >= params.upper
            || params.upper > PI
            || params.initial_intervals == 0
            || params.initial_intervals > 65536
            || params.max_depth > 32
            || !(3..=1_000_000).contains(&params.max_evaluations)
            || !params.min_width.is_finite()
            || params.min_width <= 0.0
            || !params.merge_width.is_finite()
            || params.merge_width < 0.0
            || !params.absolute_floor.is_finite()
            || params.absolute_floor < 0.0
            || !params.relative_floor.is_finite()
            || !(0.0..=1.0).contains(&params.relative_floor)
            || !params.relative_gain.is_finite()
            || params.relative_gain < 0.0
        {
            return Err(Error::InvalidParameter("adaptive frequency parameters"));
        }
        let bound = if let Some(c) = coarse {
            target.difference_bound(c)?
        } else {
            0.0
        };
        let mut state = Search {
            target,
            coarse,
            params,
            cache: BTreeMap::new(),
            stats: FrequencyStats {
                disagreement_bound: bound,
                ..FrequencyStats::default()
            },
        };
        let mut queue = BinaryHeap::new();
        let mut bands = buffer(params.initial_intervals)?;
        let step = (params.upper - params.lower) / params.initial_intervals as f64;
        let mut covered = params.lower;
        for i in 0..params.initial_intervals {
            let left = params.lower + step * i as f64;
            let right = if i + 1 == params.initial_intervals {
                params.upper
            } else {
                params.lower + step * (i + 1) as f64
            };
            let mut points = [FrequencyPoint {
                omega: 0.0,
                value: Complex::default(),
                uncertainty: 0.0,
                verified: false,
            }; 3];
            let mut complete = true;
            for (p, w) in points.iter_mut().zip([left, (left + right) * 0.5, right]) {
                if let Some(v) = state.query(w)? {
                    *p = v;
                } else {
                    complete = false;
                    break;
                }
            }
            if !complete {
                break;
            }
            queue.push(Node { points, depth: 0 });
            covered = right;
        }
        if covered < params.upper {
            bands.push(FrequencyBand {
                lower: covered,
                upper: params.upper,
                magnitude_upper: target.amplitude_bound(),
                depth: 0,
                reason: StopReason::Budget,
            });
        }
        while let Some(node) = queue.pop() {
            let [left, mid, right] = node.points;
            let width = node.width();
            let upper = (node.score() + target.lipschitz_bound() * width * 0.25)
                .min(target.amplitude_bound());
            let peaked = mid.lower() > left.upper() && mid.lower() > right.upper();
            let bend = (mid.magnitude() - (left.magnitude() + right.magnitude()) * 0.5).abs()
                + mid.uncertainty
                + (left.uncertainty + right.uncertainty) * 0.5;
            let reason = if upper < state.floor() {
                Some(StopReason::BelowFloor)
            } else if width <= params.min_width {
                Some(StopReason::Width)
            } else if node.depth >= params.max_depth {
                Some(StopReason::Depth)
            } else if params.prune_flat
                && node.depth >= 2
                && !peaked
                && (node.score() < state.floor()
                    || bend <= params.relative_gain * state.stats.observed_peak)
            {
                Some(StopReason::LowGain)
            } else if state.stats.budget_exhausted {
                Some(StopReason::Budget)
            } else {
                None
            };
            if let Some(reason) = reason {
                bands.push(FrequencyBand {
                    lower: left.omega,
                    upper: right.omega,
                    magnitude_upper: upper,
                    depth: node.depth,
                    reason,
                });
                continue;
            }
            let a = state.query((left.omega + mid.omega) * 0.5)?;
            let b = state.query((mid.omega + right.omega) * 0.5)?;
            if let (Some(a), Some(b)) = (a, b) {
                queue.push(Node {
                    points: [left, a, mid],
                    depth: node.depth + 1,
                });
                queue.push(Node {
                    points: [mid, b, right],
                    depth: node.depth + 1,
                });
            } else {
                bands.push(FrequencyBand {
                    lower: left.omega,
                    upper: right.omega,
                    magnitude_upper: upper,
                    depth: node.depth,
                    reason: StopReason::Budget,
                });
            }
        }
        // A peak may sit at the boundary of two nearly flat monotone intervals.
        // Refine every observed relevant maximum with bounded ternary arithmetic
        // instead of making all high-magnitude flanks arbitrarily dense.
        let snapshot: Vec<_> = state.cache.values().copied().collect();
        for (i, &p) in snapshot.iter().enumerate() {
            if !p.verified || p.magnitude() < state.floor() || p.magnitude() == 0.0 {
                continue;
            }
            let left = i.checked_sub(1).map(|j| snapshot[j]);
            let right = snapshot.get(i + 1).copied();
            if left.is_none_or(|a| p.magnitude() > a.upper())
                && right.is_none_or(|a| p.magnitude() >= a.upper())
            {
                let mut lo = left.map_or(p.omega, |a| a.omega);
                let mut hi = right.map_or(p.omega, |a| a.omega);
                for _ in 0..params.max_depth {
                    if hi - lo <= params.min_width || state.stats.budget_exhausted {
                        break;
                    }
                    let a = state.query(lo + (hi - lo) / 3.0)?;
                    let b = state.query(hi - (hi - lo) / 3.0)?;
                    if let (Some(a), Some(b)) = (a, b) {
                        if !a.verified || !b.verified {
                            break;
                        }
                        state.stats.peak_refinements += 1;
                        if a.magnitude() < b.magnitude() {
                            lo = a.omega;
                        } else {
                            hi = b.omega;
                        }
                    } else {
                        break;
                    }
                }
            }
        }
        let points: Vec<_> = state.cache.into_values().collect();
        let mut candidates = buffer(points.len())?;
        let floor = params
            .absolute_floor
            .max(params.relative_floor * state.stats.observed_peak);
        for (i, &p) in points.iter().enumerate() {
            if !p.verified || p.magnitude() < floor || p.magnitude() == 0.0 {
                continue;
            }
            let left = i.checked_sub(1).map(|j| points[j]);
            let right = points.get(i + 1).copied();
            if left.is_none_or(|a| p.magnitude() > a.upper())
                && right.is_none_or(|a| p.magnitude() >= a.upper())
            {
                candidates.push(FrequencyPeak {
                    omega: p.omega,
                    value: p.value,
                    bracket: (
                        left.map_or(p.omega, |a| a.omega),
                        right.map_or(p.omega, |a| a.omega),
                    ),
                    merged_candidates: 1,
                });
            }
        }
        // Strongest-first nonmaximum suppression avoids transitive merging of
        // a long chain of individually close but collectively separated peaks.
        candidates.sort_by(|a, b| {
            b.value
                .re
                .hypot(b.value.im)
                .total_cmp(&a.value.re.hypot(a.value.im))
        });
        let mut peaks: Vec<FrequencyPeak> = buffer(candidates.len())?;
        for p in candidates {
            if let Some(q) = peaks
                .iter_mut()
                .find(|q| (q.omega - p.omega).abs() <= params.merge_width)
            {
                q.merged_candidates += 1;
            } else {
                peaks.push(p);
            }
        }
        peaks.sort_by(|a, b| a.omega.total_cmp(&b.omega));
        bands.sort_by(|a, b| a.lower.total_cmp(&b.lower));
        Ok(Self {
            points,
            peaks,
            bands,
            stats: state.stats,
        })
    }
}
