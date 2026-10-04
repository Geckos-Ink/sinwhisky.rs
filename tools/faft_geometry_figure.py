#!/usr/bin/env python3
"""Diagram for PAPER_FAFT.md: from three anchor points to two rotations.

Usage: faft_geometry_figure.py OUTPUT.svg

Standard library only, deterministic, no audio or measured data: the figure is the
exact geometry of one example support. It solves the same 2x2 circle system and
angle unwrapping as src/geometry.rs, asserts that the two angular sweeps are twice
the opposite inscribed angles of the anchor triangle, and draws the plane (left)
next to the equal-angle traversal in time (right). Palette and font follow
tools/faft_figures.py. Published copy: docs/figures/faft-three-points.svg.
"""
import math
import re
import sys
from xml.sax.saxutils import escape

SURFACE, INK, SECONDARY, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

# Example support: L = 5 sample intervals, interior anchor m = 2, aperture alpha.
L, M, ALPHA = 5, 2, 0.32
Y = [0.05, 1.0, 0.45]
P = [(0.0, Y[0]), (M * ALPHA, Y[1]), (L * ALPHA, Y[2])]


def circle(p):
    """Centre, radius and determinant of the circle through three points."""
    (x0, y0), (x1, y1), (x2, y2) = p
    a00, a01, a10, a11 = 2 * (x1 - x0), 2 * (y1 - y0), 2 * (x2 - x1), 2 * (y2 - y1)
    b0 = x1 * x1 - x0 * x0 + y1 * y1 - y0 * y0
    b1 = x2 * x2 - x1 * x1 + y2 * y2 - y1 * y1
    det = a00 * a11 - a01 * a10
    cx, cy = (b0 * a11 - a01 * b1) / det, (a00 * b1 - b0 * a10) / det
    return cx, cy, math.hypot(cx - x1, cy - y1), det


def arc_angles(cx, cy, p):
    """Unwrap the three angles in the direction that visits the middle point."""
    tau = 2 * math.pi
    a = [math.atan2(y - cy, x - cx) for x, y in p]
    direction = 1.0 if (a[1] - a[0]) % tau <= (a[2] - a[0]) % tau else -1.0
    middle = a[0] + direction * ((direction * (a[1] - a[0])) % tau)
    return [a[0], middle, middle + direction * ((direction * (a[2] - a[1])) % tau)]


def interior(p, q, r):
    """Interior angle at vertex p of triangle pqr."""
    v1, v2 = (q[0] - p[0], q[1] - p[1]), (r[0] - p[0], r[1] - p[1])
    return math.atan2(abs(v1[0] * v2[1] - v1[1] * v2[0]), v1[0] * v2[0] + v1[1] * v2[1])


CX, CY, R, DET = circle(P)
TH = arc_angles(CX, CY, P)
G0, G2 = interior(P[0], P[1], P[2]), interior(P[2], P[0], P[1])
B1, B2 = (TH[1] - TH[0]) / M, (TH[2] - TH[1]) / (L - M)
assert abs((TH[1] - TH[0]) - math.copysign(2 * G2, DET)) < 1e-12
assert abs((TH[2] - TH[1]) - math.copysign(2 * G0, DET)) < 1e-12


def theta(t):
    return TH[0] + B1 * t if t <= M else TH[1] + B2 * (t - M)


def curve(t):
    return CY + R * math.sin(theta(t))


class Svg:
    def __init__(self):
        self.parts = []

    def text(self, x, y, value, size=12, anchor='start', fill=SECONDARY, weight=400, italic=False,
             halo=False):
        """'_X' in value renders X as a subscript; halo keeps labels legible over lines."""
        style = ' font-style="italic"' if italic else ''
        if halo:
            style += f' stroke="{SURFACE}" stroke-width="4" stroke-linejoin="round" paint-order="stroke"'
        body = ''.join(
            f'<tspan baseline-shift="sub" font-size="{size * 0.75:.0f}">{escape(part[1:])}</tspan>'
            if part.startswith('_') else escape(part) for part in re.split(r'(_[A-Za-z0-9]+)', value))
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
                          f'fill="{fill}" font-weight="{weight}"{style}>{body}</text>')

    def line(self, a, b, stroke, width=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.parts.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
                          f'stroke="{stroke}" stroke-width="{width}"{d}/>')

    def path(self, points, stroke, width, dash=None):
        d = 'M' + ' L'.join(f'{x:.2f},{y:.2f}' for x, y in points)
        extra = f' stroke-dasharray="{dash}"' if dash else ''
        self.parts.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" '
                          f'stroke-linecap="round" stroke-linejoin="round"{extra}/>')

    def dot(self, p, r, fill):
        self.parts.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="{r}" fill="{fill}"/>')


def render():
    svg = Svg()
    width, height = 900, 470
    # One vertical scale for both panels so equal heights line up.
    y_lo, y_hi = min(CY - R, min(Y)) - 0.06, max(CY + R, max(Y)) + 0.06
    top, bottom, left = 62, 392, 70
    px = (bottom - top) / (y_hi - y_lo)
    x_lo = min(CX - R, 0.0) - 0.06

    def vy(y):
        return bottom - (y - y_lo) * px

    def plane(x, y):  # equal scale on both axes, so the circle stays round
        return (left + (x - x_lo) * px, vy(y))

    x_hi_px = plane(max(CX + R, L * ALPHA) + 0.06, 0)[0]
    t_left, t_right = max(x_hi_px + 120, 520), 800

    def time(t, y):
        return (t_left + t * (t_right - t_left) / L, vy(y))

    svg.parts.append(f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>')
    svg.text(left, 30, 'Plane (horizontal αt, vertical amplitude)', 13, fill=INK, weight=600)
    svg.text(t_left, 30, 'Time: equal angles give two rotations', 13, fill=INK, weight=600)

    # Left panel: circle, radii, anchor triangle and the two traversed arcs.
    svg.path([plane(CX + R * math.cos(a), CY + R * math.sin(a))
              for a in [k * math.tau / 240 for k in range(241)]], GRID, 1.5)
    centre = plane(CX, CY)
    for p in P:
        svg.line(centre, plane(*p), MUTED, 1, '4 3')
    svg.path([plane(*p) for p in P + [P[0]]], SECONDARY, 0.8, '2 3')
    for a, b, colour in [(TH[0], TH[1], BLUE), (TH[1], TH[2], ORANGE)]:
        svg.path([plane(CX + R * math.cos(a + (b - a) * k / 80), CY + R * math.sin(a + (b - a) * k / 80))
                  for k in range(81)], colour, 3.2)

    def angle_mark(vertex, a_from, a_to, radius, colour, label, label_radius, anchor='middle'):
        # Plane angles have y up; SVG has y down.
        svg.path([(vertex[0] + radius * math.cos(a), vertex[1] - radius * math.sin(a))
                  for a in [a_from + (a_to - a_from) * k / 40 for k in range(41)]], colour, 1.6)
        mid = 0.5 * (a_from + a_to)
        svg.text(vertex[0] + label_radius * math.cos(mid), vertex[1] - label_radius * math.sin(mid) + 4,
                 label, 13, anchor, colour, 600, halo=True)

    def direction(frm, to):
        return math.atan2(to[1] - frm[1], to[0] - frm[0])

    def short(a, b):
        return a, a + (b - a + math.pi) % math.tau - math.pi

    angle_mark(centre, TH[0], TH[1], 30, BLUE, 'Θ_1 = 2γ_2', 44, 'end')
    angle_mark(centre, TH[1], TH[2], 22, ORANGE, 'Θ_2 = 2γ_0', 46, 'start')
    angle_mark(plane(*P[2]), *short(direction(P[2], P[0]), direction(P[2], P[1])), 34, BLUE, 'γ_2', 50)
    angle_mark(plane(*P[0]), *short(direction(P[0], P[1]), direction(P[0], P[2])), 34, ORANGE, 'γ_0', 50)
    for k in range(1, 2 * L):
        if k / 2 != M:
            a = theta(k / 2)
            svg.dot(plane(CX + R * math.cos(a), CY + R * math.sin(a)), 3, AQUA)
    for j, (p, (dx, dy)) in enumerate(zip(P, [(-14, -10), (0, -14), (14, -10)])):
        x, y = plane(*p)
        svg.dot((x, y), 4.5, INK)
        svg.text(x + dx, y + dy, f'P_{j}', 13, 'middle', INK, 600)
    svg.dot(centre, 3.5, SECONDARY)
    svg.text(centre[0] + 8, centre[1] + 16, 'C', 13, 'start', SECONDARY, 600, halo=True)
    mid_r = plane(CX + 0.55 * R * math.cos(TH[2]), CY + 0.55 * R * math.sin(TH[2]))
    svg.text(mid_r[0] + 4, mid_r[1] + 16, 'r', 13, 'middle', SECONDARY, 600, italic=True, halo=True)
    base = vy(y_lo) - 2
    svg.line((plane(x_lo, 0)[0], base), (x_hi_px, base), MUTED)
    for x, label in [(0.0, '0'), (M * ALPHA, 'mα'), (CX, 'X_C'), (L * ALPHA, 'Lα')]:
        x_px = plane(x, 0)[0]
        svg.line((x_px, base), (x_px, base + 5), MUTED)
        svg.text(x_px, base + 18, label, 11, 'middle', MUTED)

    # Right panel: the same arc in time, two rotations around the centre height.
    svg.line((t_left, base), (t_right, base), MUTED)
    for t in range(L + 1):
        x_px = time(t, 0)[0]
        svg.line((x_px, base), (x_px, base + 5), MUTED)
        svg.text(x_px, base + 18, str(t), 11, 'middle', MUTED)
    svg.text(t_right, base + 34, 't − s (samples)', 11, 'end', MUTED)
    svg.line(time(M, y_lo), time(M, y_hi), GRID)
    svg.line(time(0, CY), time(L, CY), MUTED, 1, '5 4')
    svg.text(time(L, CY)[0], time(L, CY)[1] - 6, 'centre height c = Y_C', 11, 'end', MUTED)
    svg.path([time(k / 40, curve(k / 40)) for k in range(40 * M + 1)], BLUE, 3.2)
    svg.path([time(M + k / 40, curve(M + k / 40)) for k in range(40 * (L - M) + 1)], ORANGE, 3.2)
    for k in range(1, 2 * L):
        if k / 2 != M:
            svg.dot(time(k / 2, curve(k / 2)), 3, AQUA)
    for t, label, dx, anchor in [(0, 'x[n_0]', -6, 'end'), (M, 'x[n_0+m]', 0, 'middle'),
                                 (L, 'x[n_0+L]', 6, 'start')]:
        p = time(t, curve(t))
        svg.dot(p, 4.5, INK)
        svg.text(p[0] + dx, p[1] - 12, label, 12, anchor, INK)
    svg.text(time(1.0, 0)[0] - 14, vy(curve(1.0)) - 6, 'rate β_1 = Θ_1/m', 13, 'end', BLUE, 600)
    svg.text(time(3.4, 0)[0], vy(curve(3.4)) + 80, 'rate β_2 = Θ_2/(L−m)', 13, 'middle', ORANGE, 600)
    svg.text(t_left, height - 12, 'Aqua dots: steps of ½ sample in time = equal angle steps on each side.',
             11, 'start', MUTED)

    title = 'From three anchor points to two rotations'
    desc = (f'Circle through P0=(0,{Y[0]}), P1=({M}α,{Y[1]}), P2=({L}α,{Y[2]}) with α={ALPHA}: '
            f'centre ({CX:.4f},{CY:.4f}), radius {R:.4f}; sweeps {TH[1] - TH[0]:.4f} and '
            f'{TH[2] - TH[1]:.4f} rad are twice the inscribed angles {G2:.4f} and {G0:.4f}; '
            f'rates {B1:.4f} and {B2:.4f} rad per sample.')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d" font-family="{FONT}">'
            f'<title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc>'
            + ''.join(svg.parts) + '</svg>\n')


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    with open(sys.argv[1], 'w', encoding='utf-8') as out:
        out.write(render())


if __name__ == '__main__':
    main()
