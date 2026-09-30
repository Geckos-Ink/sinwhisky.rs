#!/usr/bin/env python3
"""Static SVG figures for the circle-arc FAFT experiment, from measured data only.

Usage: faft_figures.py OUTPUT_DIR (a directory written by faft_compare.py).
Needs NumPy only. Colors follow one role per entity across figures: blue is FAFT,
orange the sampled/zoom family; categorical order validated for CVD separation.
"""
import json
import math
from pathlib import Path
import sys
from xml.sax.saxutils import escape

import numpy as np

SURFACE, INK, SECONDARY, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, GREEN = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7', '#008300')
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


class Svg:
    def __init__(self, width, height, title, description):
        self.width, self.height, self.parts = width, height, []
        self.head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                     f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d" font-family="{FONT}">'
                     f'<title id="t">{escape(title)}</title><desc id="d">{escape(description)}</desc>'
                     f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>')

    def add(self, text):
        self.parts.append(text)

    def text(self, x, y, value, size=12, anchor='start', fill=SECONDARY, weight=400):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
                 f'fill="{fill}" font-weight="{weight}">{escape(value)}</text>')

    def save(self, path):
        path.write_text(self.head+''.join(self.parts)+'</svg>\n')


def nice_ticks(lo, hi, count=5):
    step = 10**math.floor(math.log10((hi-lo)/count))
    for m in (1, 2, 2.5, 5, 10):
        if (hi-lo)/(step*m) <= count:
            step *= m
            break
    first = math.ceil(lo/step)*step
    return [first+i*step for i in range(int((hi-first)/step+1e-9)+1)]


class Panel:
    """One axis pair. Log axes use decades; gridlines are 1px, solid, recessive."""

    def __init__(self, svg, x, y, w, h, xlim, ylim, xlog=False, ylog=False):
        self.svg, self.x, self.y, self.w, self.h = svg, x, y, w, h
        self.xlim, self.ylim, self.xlog, self.ylog = xlim, ylim, xlog, ylog

    def _map(self, v, lim, log, size, flip):
        a, b = (math.log10(lim[0]), math.log10(lim[1])) if log else lim
        v = math.log10(v) if log else v
        f = (v-a)/(b-a)
        return (1-f)*size if flip else f*size

    def px(self, v):
        return self.x+self._map(v, self.xlim, self.xlog, self.w, False)

    def py(self, v):
        v = min(max(v, self.ylim[0]), self.ylim[1])
        return self.y+self._map(v, self.ylim, self.ylog, self.h, True)

    def axes(self, title, xlabel, ylabel, xticks, yticks, xfmt=str, yfmt=str):
        s = self.svg
        for t in yticks:
            s.add(f'<line x1="{self.x}" x2="{self.x+self.w}" y1="{self.py(t):.1f}" y2="{self.py(t):.1f}" stroke="{GRID}" stroke-width="1"/>')
            s.text(self.x-6, self.py(t)+4, yfmt(t), 11, 'end', MUTED)
        for t in xticks:
            s.add(f'<line x1="{self.px(t):.1f}" x2="{self.px(t):.1f}" y1="{self.y+self.h}" y2="{self.y+self.h+4}" stroke="{MUTED}" stroke-width="1"/>')
            s.text(self.px(t), self.y+self.h+17, xfmt(t), 11, 'middle', MUTED)
        s.add(f'<line x1="{self.x}" x2="{self.x+self.w}" y1="{self.y+self.h}" y2="{self.y+self.h}" stroke="{MUTED}" stroke-width="1"/>')
        s.text(self.x, self.y-12, title, 13, 'start', INK, 600)
        s.text(self.x+self.w/2, self.y+self.h+34, xlabel, 12, 'middle')
        s.add(f'<text transform="translate({self.x-44},{self.y+self.h/2}) rotate(-90)" font-size="12" '
              f'text-anchor="middle" fill="{SECONDARY}">{escape(ylabel)}</text>')

    def line(self, xs, ys, color, width=2, label=None, opacity=1.0):
        pts = ' '.join(f'{self.px(a):.1f},{self.py(b):.1f}' for a, b in zip(xs, ys))
        tip = f'<title>{escape(label)}</title>' if label else ''
        self.svg.add(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" '
                     f'stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}">{tip}</polyline>')

    def dots(self, xs, ys, color, labels=None, r=4):
        for i, (a, b) in enumerate(zip(xs, ys)):
            tip = f'<title>{escape(labels[i])}</title>' if labels else ''
            self.svg.add(f'<circle cx="{self.px(a):.1f}" cy="{self.py(b):.1f}" r="{r}" fill="{color}" '
                         f'stroke="{SURFACE}" stroke-width="2">{tip}</circle>')

    def band(self, xs, lo, hi, color):
        pts = [f'{self.px(a):.1f},{self.py(b):.1f}' for a, b in zip(xs, hi)]
        pts += [f'{self.px(a):.1f},{self.py(b):.1f}' for a, b in zip(xs[::-1], lo[::-1])]
        self.svg.add(f'<polygon points="{" ".join(pts)}" fill="{color}" opacity="0.1"/>')


def legend(svg, x, y, items):
    for label, color in items:
        svg.add(f'<line x1="{x}" x2="{x+18}" y1="{y-4}" y2="{y-4}" stroke="{color}" stroke-width="2" stroke-linecap="round"/>')
        svg.text(x+24, y, label, 12)
        x += 30+7.0*len(label)


def legend_column(svg, x, y, items, step=18):
    for i, (label, color) in enumerate(items):
        yy = y+i*step
        svg.add(f'<line x1="{x}" x2="{x+18}" y1="{yy-4}" y2="{yy-4}" stroke="{color}" stroke-width="2" stroke-linecap="round"/>')
        svg.text(x+24, yy, label, 12)


def envelope(x, y, pixels):
    """Min/max per horizontal pixel so dense spectra stay faithful but small."""
    edges = np.linspace(x[0], x[-1], pixels+1)
    idx = np.clip(np.searchsorted(edges, x, side='right')-1, 0, pixels-1)
    xs, ys = [], []
    for k in range(pixels):
        sel = np.nonzero(idx == k)[0]
        if len(sel) == 0:
            continue
        lo, hi = sel[np.argmin(y[sel])], sel[np.argmax(y[sel])]
        for j in sorted({lo, hi}):
            xs.append(x[j]); ys.append(y[j])
    return np.array(xs), np.array(ys)


def db(v, reference):
    return 20*np.log10(np.maximum(np.asarray(v)/reference, 1e-12))


def zoom_figure(selected, rate, path):
    row = selected['row']
    omega = np.asarray(row['dense_omega'])
    khz = omega*rate/(2*np.pi)/1000
    reference = max(row['dense_faft'])
    svg = Svg(900, 460, 'FAFT versus zoom-sampled transforms of one fitted circle curve',
              'Magnitude in dB versus frequency to three times the source Nyquist. Zoom 1 mirrors '
              'the audio band above 22.05 kHz; zoom 4 follows the exact FAFT; FAFT has no images.')
    p = Panel(svg, 70, 70, 790, 290, (0, khz[-1]), (-120, 0))
    p.axes('Same fitted curve, three transforms · 60 s window, Hann after fitting',
           'Frequency (kHz); 0–22.05 kHz is the recorded band',
           'Magnitude (dB re strongest)', nice_ticks(0, khz[-1], 7), [-120, -90, -60, -30, 0],
           lambda t: f'{t:g}', lambda t: f'{t:g}')
    for f, label in [(rate/2000, 'source Nyquist 22.05 kHz'), (rate/1000, 'zoom-1 period 44.1 kHz')]:
        svg.add(f'<line x1="{p.px(f):.1f}" x2="{p.px(f):.1f}" y1="{p.y}" y2="{p.y+p.h}" stroke="{MUTED}" stroke-width="1"/>')
        svg.text(p.px(f)+4, p.y+p.h-6, label, 11, 'start', MUTED)
    for key, color, label in [('dense_zoom1', ORANGE, 'zoom 1: sampled DTFT'),
                              ('dense_zoom4', AQUA, 'zoom 4: sampled DTFT'),
                              ('dense_faft', BLUE, 'FAFT: exact continuous')]:
        xs, ys = envelope(khz, db(row[key], reference), 790)
        p.line(xs, ys, color, 1.5 if key != 'dense_faft' else 2, label)
    svg.text(p.px(41.5), p.py(-4), 'zoom 1 mirrors the band (images)', 11, 'end', SECONDARY)
    legend(svg, 70, 440, [('FAFT exact continuous', BLUE), ('zoom 1 sampled DTFT', ORANGE),
                          ('zoom 4 sampled DTFT', AQUA)])
    svg.save(path)


def peaks_figure(selected, rate, path, upper_hz=5000):
    row, x = selected['row'], np.asarray(selected['source'])
    n = len(x)
    omega = np.asarray(row['dense_omega'])
    hz = omega*rate/(2*np.pi)
    keep = hz <= upper_hz
    hann = 0.5-0.5*np.cos(2*np.pi*np.arange(n)/n)
    padded = np.abs(np.fft.rfft(x*hann, 64*n))/(n/2)
    fhz = np.arange(len(padded))*rate/(64*n)
    pk = fhz <= upper_hz
    peaks = np.asarray(row['scan']['peaks'], dtype=float).reshape(-1, 4)
    pmag = np.hypot(peaks[:, 1], peaks[:, 2])
    reference = max(max(row['dense_faft']), pmag.max())
    svg = Svg(900, 420, 'Continuous FAFT peaks against the source FFT',
              'Exact FAFT magnitude with Newton-refined peaks, and the 64-times zero-padded FFT of the '
              'Hann-windowed source samples, 0 to 5 kHz. Ticks mark the unpadded 43 Hz DFT bins.')
    strongest = float(peaks[int(np.argmax(pmag)), 0]*rate/(2*np.pi))
    p = Panel(svg, 70, 70, 790, 270, (0, upper_hz/1000), (-60, 0))
    p.axes(f'{len(peaks)} relevant peaks at continuous frequencies · strongest {strongest:.4f} Hz · same window',
           'Frequency (kHz); bottom ticks are the fixed N=1024 DFT bins', 'Magnitude (dB re strongest)',
           nice_ticks(0, upper_hz/1000, 10), [-60, -45, -30, -15, 0], lambda t: f'{t:g}', lambda t: f'{t:g}')
    for k in range(int(upper_hz/(rate/n))+1):
        f = k*rate/n/1000
        svg.add(f'<line x1="{p.px(f):.1f}" x2="{p.px(f):.1f}" y1="{p.y+p.h-5}" y2="{p.y+p.h}" stroke="{MUTED}" stroke-width="1"/>')
    p.line(fhz[pk]/1000, db(padded[pk], reference), ORANGE, 1.5, 'source samples, 64x padded FFT')
    p.line(hz[keep]/1000, db(np.asarray(row['dense_faft'])[keep], reference), BLUE, 2, 'FAFT exact continuous')
    inside = peaks[:, 0]*rate/(2*np.pi) <= upper_hz
    labels = [f'{w*rate/(2*np.pi):.3f} Hz, {20*math.log10(m/reference):.1f} dB'
              for w, m in zip(peaks[inside, 0], pmag[inside])]
    p.dots(peaks[inside, 0]*rate/(2*np.pi)/1000, db(pmag[inside], reference), BLUE, labels)
    legend(svg, 70, 405, [('FAFT exact continuous, Newton-refined peaks', BLUE),
                          ('source samples, 64× zero-padded FFT', ORANGE)])
    svg.save(path)


def cost_figure(report, path):
    svg = Svg(1000, 510, 'Zoom convergence and measured costs',
              'Left: relative difference of zoom-z sampled transforms from the exact FAFT. Middle: '
              'per-frequency cost versus window size. Right: cost to return a relevant-peak list.')
    zs = [1, 2, 4, 8, 16, 64]
    med = [float(np.median([e['zoom_relative_l2'][str(z)]['median'] for e in report['excerpts']])) for z in zs]
    top = [max(e['zoom_relative_l2'][str(z)]['max'] for e in report['excerpts']) for z in zs]
    low = [min(e['zoom_relative_l2'][str(z)]['median'] for e in report['excerpts']) for z in zs]
    a = Panel(svg, 70, 70, 250, 270, (1, 64), (1e-6, 1e-2), True, True)
    a.axes('Zoom only approaches FAFT', 'zoom z (samples per source interval)', 'relative L2 difference',
           zs, [1e-6, 1e-5, 1e-4, 1e-3, 1e-2], lambda t: f'{t:g}', lambda t: f'1e{int(round(math.log10(t)))}')
    a.band(zs, low, top, ORANGE)
    a.line([1, 64], [med[0], med[0]/64**2], MUTED, 1, 'reference slope 1/z²')
    a.line(zs, med, ORANGE, 2, 'median over 63 windows')
    a.dots(zs, med, ORANGE, [f'z={z}: {m:.2e}' for z, m in zip(zs, med)])
    sweep = report['scaling']
    n = [r['n'] for r in sweep]
    b = Panel(svg, 400, 70, 250, 270, (256, 4096), (0.5, 300), True, True)
    b.axes('One arbitrary frequency', 'window N (source samples)', 'µs per frequency', n,
           [1, 10, 100], lambda t: f'{t:g}', lambda t: f'{t:g}')
    series = [('faft_query_us', BLUE, 'FAFT exact'), ('rotation_query_us', ORANGE, 'sampled, rotations'),
              ('closed_query_us', AQUA, 'sampled, closed sums'), ('aft_query_us', YELLOW, 'integral AFT'),
              ('aft_faft_query_us', MAGENTA, 'FAFT on AFT curve')]
    for key, color, label in series:
        b.line(n, [r[key] for r in sweep], color, 2, label)
        b.dots(n, [r[key] for r in sweep], color, [f'{label}, N={r["n"]}: {r[key]:.2f} µs' for r in sweep])
    c = Panel(svg, 730, 70, 250, 270, (0.01, 1000), (0.01, 1000), True, True)
    c.xlim = (256, 4096)
    c.axes('A relevant-peak list', 'window N (source samples)', 'ms per window', n,
           [0.01, 0.1, 1, 10, 100, 1000], lambda t: f'{t:g}', lambda t: f'{t:g}')
    series2 = [('scan_us', BLUE, 'FAFT scan + Newton'), ('adaptive_us', ORANGE, 'earlier adaptive'),
               ('circles_us', VIOLET, 'FAFT circle seeds'), ('numpy_rfft_64x_us', GREEN, 'source FFT ×64 pad')]
    for key, color, label in series2:
        c.line(n, [r[key]/1000 for r in sweep], color, 2, label)
        c.dots(n, [r[key]/1000 for r in sweep], color, [f'{label}, N={r["n"]}: {r[key]/1000:.2f} ms' for r in sweep])
    legend_column(svg, 400, 400, [(label, color) for _, color, label in series])
    legend_column(svg, 730, 400, [(label, color) for _, color, label in series2])
    legend_column(svg, 70, 400, [('median over 63 windows', ORANGE)])
    svg.text(70, 422, 'Band: lowest excerpt median to', 11, 'start', MUTED)
    svg.text(70, 438, 'worst window. Grey: slope 1/z².', 11, 'start', MUTED)
    svg.save(path)


def render(output):
    output = Path(output)
    report = json.loads((output/'report.json').read_text())
    selected = json.loads((output/'private-selected-window.json').read_text())
    zoom_figure(selected, report['rate'], output/'faft-zoom-vs-transform.svg')
    peaks_figure(selected, report['rate'], output/'faft-peaks.svg')
    cost_figure(report, output/'faft-convergence-cost.svg')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    render(sys.argv[1])
