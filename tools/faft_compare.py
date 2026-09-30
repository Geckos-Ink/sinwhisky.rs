#!/usr/bin/env python3
"""Measure the circle-arc FAFT against earlier kernels, controls and independent checks.

Requires NumPy and Cargo. FFmpeg is needed only for non-WAV input; a PCM16 WAV can
be read directly (for example one produced by macOS `afconvert`). Output must be a
new directory outside the repository. Exported rotations, dense spectra and query
values are invertible model data and stay there; copy only report.json and figures.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import wave

import numpy as np

from faft_figures import render

ROOT = Path(__file__).resolve().parents[1]


def read_wav(path, start, seconds, rate):
    """Mono PCM16 excerpt: channel mean rounded to the PCM16 grid (like -ac 1)."""
    with wave.open(str(path), 'rb') as w:
        if w.getsampwidth() != 2 or w.getframerate() != rate:
            raise ValueError('expected PCM16 WAV at the requested rate')
        channels = w.getnchannels()
        begin = int(round(start*rate))
        if begin + int(round(seconds*rate)) > w.getnframes():
            raise ValueError('excerpt beyond end of file')
        w.setpos(begin)
        data = w.readframes(int(round(seconds*rate)))
    x = np.frombuffer(data, dtype='<i2').astype(np.float64).reshape(-1, channels)
    return np.rint(x.mean(axis=1))


def read_ffmpeg(ffmpeg, path, start, seconds, rate):
    result = subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-ss', str(start), '-i', str(path),
                             '-t', str(seconds), '-map', '0:a:0', '-ac', '1', '-ar', str(rate),
                             '-f', 's16le', '-c:a', 'pcm_s16le', 'pipe:1'], capture_output=True, check=True)
    if not result.stdout or len(result.stdout) % 2:
        raise ValueError('empty or incomplete PCM')
    return np.frombuffer(result.stdout, dtype='<i2').astype(np.float64)


def select_windows(pcm, n, hop, count):
    """Raw (unwindowed) windows: FAFT applies Hann analytically AFTER fitting."""
    frames = np.lib.stride_tricks.sliding_window_view(pcm, n)[::hop]
    indices = np.linspace(0, len(frames)-1, count, dtype=int)
    return frames[indices].copy(), indices


def run_bench(binary, frames, error, span, repeats, queries, dense=0):
    n = frames.shape[1]
    text = '\n'.join(' '.join(format(v, '.17g') for v in row) for row in frames)
    args = [str(binary), str(n), str(error), str(span), str(repeats), str(queries)]
    if dense:
        args.append(str(dense))
    proc = subprocess.run(args, input=text, text=True, capture_output=True, check=True)
    rows = [json.loads(line) for line in proc.stdout.splitlines()]
    if len(rows) != len(frames):
        raise ValueError('frame count mismatch')
    for row in rows:
        if len(row['omegas']) != queries or len(row['values']) != queries:
            raise ValueError('query count mismatch')
        if not all(math.isfinite(v) for pair in row['values'] for v in pair):
            raise ValueError('non-finite FAFT value')
    return rows


def curve(row, positions):
    """Independent evaluation of the exported rotations (stacks add)."""
    r = np.asarray(row['model'], dtype=float)
    positions = np.asarray(positions, dtype=float)
    y = np.zeros_like(positions)
    covered = np.zeros(positions.shape, dtype=int)
    for start, length, center, radius, phase, rate, slope in r:
        mask = (positions >= start) & (positions < start+length)
        u = positions[mask]-start
        y[mask] += center + radius*np.sin(phase+rate*u) + slope*u
        covered[mask] += 1
    if np.any(covered == 0):
        raise ValueError('rotation coverage gap')
    return y


class Quadrature:
    """Continuous Hann-weighted FT by Gauss-Legendre per unit interval, normalized
    by n/2 with phase origin at the window start. Independent of the Rust kernel."""

    def __init__(self, row, n, nodes=16):
        x, w = np.polynomial.legendre.leggauss(nodes)
        self.t = (np.arange(n)[:, None] + 0.5*(x[None, :]+1.0)).ravel()
        self.g = curve(row, self.t)*(0.5-0.5*np.cos(2*np.pi*self.t/n))*np.tile(0.5*w, n)/(n/2)

    def __call__(self, omegas):
        return np.exp(-1j*np.outer(np.asarray(omegas, dtype=float), self.t)) @ self.g


def local_peaks(magnitude, floor):
    m = magnitude
    inner = (m[1:-1] >= m[:-2]) & (m[1:-1] > m[2:]) & (m[1:-1] >= floor)
    idx = list(np.nonzero(inner)[0]+1)
    if len(m) > 1 and m[0] > m[1] and m[0] >= floor:
        idx.insert(0, 0)
    return np.asarray(idx, dtype=int)


def match(found, reference, tolerance):
    """Match each reference frequency to its nearest found frequency."""
    found = np.asarray(found, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(reference) == 0 or len(found) == 0:
        return np.zeros(len(reference), dtype=bool), np.array([])
    nearest = np.abs(found[None, :]-reference[:, None]).min(axis=1)
    ok = nearest <= tolerance
    return ok, nearest[ok]


def peaks_of(row, key):
    return np.asarray([p[0] for p in row[key]['peaks']], dtype=float)


def summarize(rows, frames, rate, error):
    n = frames.shape[1]
    resolution = 2*np.pi/n
    hann = 0.5-0.5*np.cos(2*np.pi*np.arange(n)/n)
    s = {'windows': len(rows)}
    source_max, source_sq, quad, local_ok, peaks_checked = 0.0, 0.0, 0.0, 0, 0
    fft_recall, fft_precision, fft_distance = [], [], []
    adaptive_recall, adaptive_precision, adaptive_distance, circle_recall = [], [], [], []
    for x, row in zip(frames, rows):
        e = curve(row, np.arange(n))-x
        source_max = max(source_max, float(np.max(np.abs(e))))
        source_sq += float(np.sum(e**2))
        if source_max > error+1e-9:
            raise ValueError('source error bound violated by exported rotations')
        scan = np.asarray(row['scan']['peaks'], dtype=float).reshape(-1, 4)
        omegas = np.concatenate([scan[:, 0], row['omegas'][:16]])
        rust = np.concatenate([scan[:, 1]+1j*scan[:, 2],
                               np.asarray(row['values'][:16])@np.array([1, 1j])])
        integrate = Quadrature(row, n)
        independent = integrate(omegas)
        scale = max(1e-12, float(np.max(np.abs(rust))))
        quad = max(quad, float(np.max(np.abs(independent-rust))/scale))
        # Every refined peak must be a local maximum of the independently integrated FT.
        h = resolution*1e-3
        for w, z in zip(scan[:, 0], scan[:, 1]+1j*scan[:, 2]):
            side = np.abs(integrate([w-h, w+h]))
            peaks_checked += 1
            local_ok += int(np.all(side < abs(z)) or w == 0.0)
        # Control: 64x zero-padded FFT of the Hann-windowed SOURCE samples.
        padded = np.abs(np.fft.rfft(x*hann, 64*n))/(n/2)
        grid = 2*np.pi*np.arange(len(padded))/(64*n)
        reference = grid[local_peaks(padded, 0.02*padded.max())]
        found = scan[:, 0]
        ok, dist = match(found, reference, 0.5*resolution)
        fft_recall.append(float(np.mean(ok)) if len(ok) else 1.0)
        ok2, _ = match(reference, found, 0.5*resolution)
        fft_precision.append(float(np.mean(ok2)) if len(ok2) else 1.0)
        fft_distance.extend((dist*rate/(2*np.pi)).tolist())
        adaptive = np.asarray([p[0] for p in row['adaptive_peaks']], dtype=float)
        ok, dist = match(adaptive, found, 0.5*resolution)
        adaptive_recall.append(float(np.mean(ok)) if len(ok) else 1.0)
        adaptive_distance.extend((dist*rate/(2*np.pi)).tolist())
        ok, _ = match(found, adaptive, 0.5*resolution)
        adaptive_precision.append(float(np.mean(ok)) if len(ok) else 1.0)
        ok, _ = match(peaks_of(row, 'circle_seeded'), found, 0.25*resolution)
        circle_recall.append(float(np.mean(ok)) if len(ok) else 1.0)
    med = lambda key: float(np.median([r[key] for r in rows]))
    zoom = {}
    for i, (z, _) in enumerate(rows[0]['zoom_relative_l2']):
        values = [r['zoom_relative_l2'][i][1] for r in rows]
        zoom[str(z)] = {'median': float(np.median(values)), 'max': float(np.max(values))}
    s.update({
        'mean_circles': float(np.mean([r['circles'] for r in rows])),
        'mean_lines': float(np.mean([r['lines'] for r in rows])),
        'mean_rotations': float(np.mean([r['rotations'] for r in rows])),
        'source_max_error': source_max,
        'source_rmse': math.sqrt(source_sq/(len(rows)*n)),
        'independent_quadrature_max_relative': quad,
        'peaks_checked_local_max': f'{local_ok}/{peaks_checked}',
        'mean_scan_peaks': float(np.mean([len(r['scan']['peaks']) for r in rows])),
        'mean_circle_seeded_peaks': float(np.mean([len(r['circle_seeded']['peaks']) for r in rows])),
        'mean_adaptive_peaks': float(np.mean([len(r['adaptive_peaks']) for r in rows])),
        'circle_seeded_recall_of_scan_peaks': float(np.mean(circle_recall)),
        'faft_peaks_also_found_by_adaptive': float(np.mean(adaptive_recall)),
        'adaptive_peaks_also_found_by_faft': float(np.mean(adaptive_precision)),
        'adaptive_vs_faft_median_hz': float(np.median(adaptive_distance)) if adaptive_distance else None,
        'source_fft_peak_recall': float(np.mean(fft_recall)),
        'faft_peaks_matching_source_fft': float(np.mean(fft_precision)),
        'source_fft_median_distance_hz': float(np.median(fft_distance)) if fft_distance else None,
        'unconverged_scan_seeds': int(sum(r['scan']['unconverged'] for r in rows)),
        'scan_density': {str(d): {'mean_peaks': float(np.mean([r['densities'][i][1] for r in rows])),
                                  'mean_shared_with_density_2': float(np.mean([r['densities'][i][2] for r in rows])),
                                  'median_us': float(np.median([r['densities'][i][3] for r in rows])/1000)}
                         for i, d in enumerate([1, 2, 4])},
        'scan_derivative_evaluations_mean': float(np.mean([r['scan']['derivative_evaluations'] for r in rows])),
        'circle_derivative_evaluations_mean': float(np.mean([r['circle_seeded']['derivative_evaluations'] for r in rows])),
        'adaptive_queries_mean': float(np.mean([r['adaptive_queries'] for r in rows])),
        'adaptive_budget_exhausted': int(sum(r['adaptive_budget_exhausted'] for r in rows)),
        'aft_same_curve_max_relative_l2': float(max(r['aft_faft_relative_l2'] for r in rows)),
        'sampled_zoom1_vs_faft_median_relative_l2': med('sampled_vs_faft_relative_l2'),
        'zoom_relative_l2': zoom,
        'median_us': {key: med(key+'_ns')/1000 for key in [
            'fit', 'prepare', 'faft_query', 'sampled_prepare', 'closed_query', 'rotation_query',
            'aft_fit', 'aft_query', 'aft_faft_prepare', 'aft_faft_query', 'scan', 'circles', 'adaptive']},
    })
    return s


def timed_fft(x, size, repeats=101):
    np.fft.rfft(x, size)
    t = []
    for _ in range(repeats):
        begin = time.perf_counter_ns()
        np.fft.rfft(x, size)
        t.append(time.perf_counter_ns()-begin)
    return float(np.median(t)/1000)


def build(cargo):
    subprocess.run([cargo, 'build', '--offline', '--release', '--example', 'faft_bench'], cwd=ROOT, check=True)
    meta = subprocess.run([cargo, 'metadata', '--offline', '--no-deps', '--format-version', '1'],
                          cwd=ROOT, capture_output=True, text=True, check=True)
    return Path(json.loads(meta.stdout)['target_directory'])/'release/examples/faft_bench'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--ffmpeg', default=os.environ.get('FFMPEG', 'ffmpeg'))
    p.add_argument('--binary', type=Path)
    p.add_argument('--rate', type=int, default=44100)
    p.add_argument('--starts', type=float, nargs='+', default=[60, 180, 300])
    p.add_argument('--seconds', type=float, default=2)
    p.add_argument('--windows', type=int, default=21)
    p.add_argument('--error', type=float, default=16)
    p.add_argument('--span', type=int, default=128)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--queries', type=int, default=256)
    p.add_argument('--sizes', type=int, nargs='+', default=[256, 512, 1024, 2048, 4096])
    args = p.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if not source.is_file():
        p.error('input recording missing')
    if output.exists() or output == ROOT or ROOT in output.parents:
        p.error('output must be new and outside the repository')
    if source.suffix.lower() == '.wav':
        read = lambda start, seconds: read_wav(source, start, seconds, args.rate)
        decoder = 'wave module (PCM16 WAV input)'
    else:
        ffmpeg = shutil.which(args.ffmpeg)
        if not ffmpeg:
            p.error('non-WAV input needs FFmpeg')
        read = lambda start, seconds: read_ffmpeg(ffmpeg, source, start, seconds, args.rate)
        decoder = subprocess.run([ffmpeg, '-version'], capture_output=True, text=True).stdout.splitlines()[0]
    binary = args.binary or build(shutil.which('cargo') or 'cargo')
    output.mkdir(parents=True)
    n = 1024
    report = {
        'experiment': 'FAFT-circle-arcs-v2', 'date': time.strftime('%Y-%m-%d'),
        'definition': 'Exact continuous FT of the circle-determined curve, Hann applied analytically after fitting; '
                      'omega in radians per source sample; values normalized by T/2.',
        'n': n, 'hop': 512, 'rate': args.rate, 'seconds': args.seconds, 'windows_per_excerpt': args.windows,
        'geometry': {'fit': 'GeometryFrame::hierarchical', 'max_error': args.error, 'max_span': args.span,
                     'min_precision': 12, 'merge_passes': 4},
        'peaks': {'scan_density': 2, 'relative_floor': 0.02, 'newton_tolerance': 1e-12,
                  'merge_width': 'quarter of 2*pi/T'},
        'adaptive_control': 'AdaptiveSpectrum::search on the zoom-1 sampled predictor, N/4 intervals, depth 12, '
                            '8192 oracle calls, 2% floor (earlier survey settings)',
        'input': 'raw mono PCM16 (channel mean rounded), no window before fitting',
        'decoder': decoder,
        'timing': 'Warmed in-process medians; per-query times are total/queries for 256 fixed pseudo-random '
                  'frequencies; peak searches include scan, refinement and merging but not fitting.',
        'environment': {'platform': platform.platform(), 'python': platform.python_version(),
                        'numpy': np.__version__,
                        'rustc': subprocess.run(['rustc', '--version'], capture_output=True, text=True).stdout.strip()},
        'excerpts': [], 'scaling': []}
    first = None
    for start in args.starts:
        print(f'FAFT excerpt at {start:g} s', flush=True)
        pcm = read(start, args.seconds)
        first = pcm if first is None else first
        frames, indices = select_windows(pcm, n, 512, args.windows)
        rows = run_bench(binary, frames, args.error, args.span, args.repeats, args.queries)
        summary = summarize(rows, frames, args.rate, args.error)
        summary.update({'start': start, 'window_indices': indices.tolist(),
                        'numpy_rfft_us': timed_fft(frames[len(frames)//2], n),
                        'numpy_rfft_64x_us': timed_fft(frames[len(frames)//2], 64*n)})
        report['excerpts'].append(summary)
        print(json.dumps({k: summary[k] for k in ['mean_rotations', 'independent_quadrature_max_relative',
                                                  'mean_scan_peaks', 'source_fft_peak_recall',
                                                  'circle_seeded_recall_of_scan_peaks', 'median_us']}), flush=True)
        if start == args.starts[0]:
            middle = frames[len(frames)//2:len(frames)//2+1]
            selected = run_bench(binary, middle, args.error, args.span, 1, args.queries, dense=12288)[0]
            (output/'private-selected-window.json').write_text(json.dumps(
                {'row': selected, 'source': middle[0].tolist()}))
    for size in args.sizes:
        print(f'Size N={size}', flush=True)
        begin = len(first)//2-size//2
        x = first[begin:begin+size][None, :]
        row = run_bench(binary, x, args.error, args.span, 5, args.queries)[0]
        report['scaling'].append({
            'n': size, 'rotations': row['rotations'], 'aft_rotations': row['aft_rotations'],
            'numpy_rfft_us': timed_fft(x[0], size), 'numpy_rfft_64x_us': timed_fft(x[0], 64*size),
            **{key+'_us': row[key+'_ns']/1000 for key in [
                'fit', 'prepare', 'faft_query', 'closed_query', 'rotation_query', 'aft_query',
                'aft_faft_query', 'scan', 'circles', 'adaptive']},
            'scan_peaks': len(row['scan']['peaks']), 'adaptive_peaks': len(row['adaptive_peaks']),
            'adaptive_budget_exhausted': row['adaptive_budget_exhausted']})
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    render(output)
    print(output/'report.json')


if __name__ == '__main__':
    main()
