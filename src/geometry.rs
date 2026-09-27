//! Geometry in local, aperture-scaled coordinates. No signal buffers or file formats.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Branch {
    Upper,
    Lower,
}

/// Circumcircle with the branch selected at the middle point.
/// The selected branch need not contain both outer points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    pub center: Point,
    pub radius: f64,
    pub branch: Branch,
}

impl Circle {
    /// Solve the same 2x2 system as the portable C/Python implementation.
    pub fn through(points: [Point; 3], epsilon: f64) -> Option<Self> {
        if !epsilon.is_finite()
            || epsilon <= 0.0
            || points.iter().any(|p| !p.x.is_finite() || !p.y.is_finite())
        {
            return None;
        }
        let [p0, p1, p2] = points;
        let a00 = 2.0 * (p1.x - p0.x);
        let a01 = 2.0 * (p1.y - p0.y);
        let a10 = 2.0 * (p2.x - p1.x);
        let a11 = 2.0 * (p2.y - p1.y);
        let b0 = p1.x * p1.x - p0.x * p0.x + (p1.y * p1.y - p0.y * p0.y);
        let b1 = p2.x * p2.x - p1.x * p1.x + (p2.y * p2.y - p1.y * p1.y);
        let det = a00 * a11 - a01 * a10;
        if !det.is_finite() || det.abs() < epsilon {
            return None;
        }
        let cx = (b0 * a11 - a01 * b1) / det;
        let cy = (a00 * b1 - b0 * a10) / det;
        let radius = (cx - p1.x).hypot(cy - p1.y);
        if !cx.is_finite() || !cy.is_finite() || !radius.is_finite() || radius <= 0.0 {
            return None;
        }
        Some(Self {
            center: Point { x: cx, y: cy },
            radius,
            branch: if p1.y >= cy {
                Branch::Upper
            } else {
                Branch::Lower
            },
        })
    }

    /// Evaluate y(x); reject positions outside |x-cx| <= radius.
    pub fn evaluate(&self, x: f64) -> Option<f64> {
        let dx = x - self.center.x;
        let inside = self.radius * self.radius - dx * dx;
        if inside < 0.0 || self.radius <= 0.0 {
            return None;
        }
        let dy = inside.sqrt();
        let y = match self.branch {
            Branch::Upper => self.center.y + dy,
            Branch::Lower => self.center.y - dy,
        };
        y.is_finite().then_some(y)
    }

    pub fn angle(&self, point: Point) -> f64 {
        (point.y - self.center.y).atan2(point.x - self.center.x)
    }
}

/// Unwrap the three angles in the direction that visits the middle point.
/// This also handles circles crossing the atan2 wrap or spanning both branches.
pub(crate) fn arc_angles(circle: Circle, points: [Point; 3]) -> [f64; 3] {
    let tau = std::f64::consts::TAU;
    let a = points.map(|p| circle.angle(p));
    let first = (a[1] - a[0]).rem_euclid(tau);
    let full = (a[2] - a[0]).rem_euclid(tau);
    let direction = if first <= full { 1.0 } else { -1.0 };
    let delta = |from: f64, to: f64| direction * (direction * (to - from)).rem_euclid(tau);
    let middle = a[0] + delta(a[0], a[1]);
    [a[0], middle, middle + delta(a[1], a[2])]
}
