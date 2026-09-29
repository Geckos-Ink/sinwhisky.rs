#!/usr/bin/env python3
"""Benchmark the analytic Angular Fourier Transform against NumPy's real FFT.

NumPy, Matplotlib, Pillow, FFmpeg, and a Rust release example are required.
PCM and invertible spectral arrays stay in a new directory outside the repository.
Only summary JSON and rendered plots/GIF should be copied into documentation.
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

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import PIL

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'fft': '#315dc5', 'aft': '#d65421', 'linear': '#699469'}


def periodic_hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def window_frames(pcm, n, hop):
    if pcm.ndim != 1 or n < 2 or hop < 1 or len(pcm) < n or not np.all(np.isfinite(pcm)):
        raise ValueError('invalid samples or frame sizes')
    return np.lib.stride_tricks.sliding_window_view(pcm, n)[::hop].copy() * periodic_hann(n)


def rust_frames(binary, frames, repeats=1, dense=0):
    n = frames.shape[1]
    text = '\n'.join(' '.join(format(x, '.17g') for x in row) for row in frames)
    result = subprocess.run([str(binary), str(n), str(repeats), str(dense)],
                            input=text, text=True, capture_output=True, check=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    if len(rows) != len(frames):
        raise ValueError('AFT frame count mismatch')
    for row in rows:
        z = np.asarray(row['aft'], dtype=np.float64)
        if z.shape != (n // 2 + 1, 2) or not np.all(np.isfinite(z)):
            raise ValueError('invalid AFT spectrum')
        if len(row['dense']) != n*dense:
            raise ValueError('invalid dense integration samples')
    return rows


def as_complex(rows):
    a = np.array([r['aft'] for r in rows])
    return a[..., 0] + 1j*a[..., 1]


def timed_fft(frames, repeats=31):
    result = []
    for frame in frames:
        np.fft.rfft(frame)  # Warm up; each timed call allocates its result.
        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            np.fft.rfft(frame)
            elapsed.append(time.perf_counter_ns() - start)
        result.append(elapsed)
    return np.asarray(result)


def dense_integral(values, n, q):
    """Midpoint Riemann sum via FFT, with its half-step phase and dt restored."""
    if len(values) != n*q:
        raise ValueError('dense length mismatch')
    k = np.arange(n//2+1)
    return np.fft.rfft(values)[:len(k)] / q * np.exp(-1j*np.pi*k/(n*q))


def relative_l2(actual, reference):
    denominator = np.linalg.norm(reference)
    return float(np.linalg.norm(actual-reference)/denominator) if denominator else None


def differences(fft, aft, rate, n):
    # These are differences between representations, not error against a known
    # continuous recording. Weak-bin phase and dB ratios are explicitly excluded.
    frequencies = np.fft.rfftfreq(n, 1/rate)
    threshold = np.max(np.abs(fft), axis=1, keepdims=True) * 1e-3
    mask = (np.abs(fft) > threshold) & (np.abs(aft) > threshold)
    mask[:, 0] = False
    if n % 2 == 0:
        mask[:, -1] = False
    ratios = np.abs(aft[mask]) / np.abs(fft[mask])
    phase = np.abs(np.angle(aft[mask] * np.conj(fft[mask]))) * 180/np.pi
    result = {'complex_relative_l2': relative_l2(aft, fft), 'strong_bin_count': int(mask.sum()),
              'strong_bin_magnitude_median_abs_db': float(np.median(np.abs(20*np.log10(ratios)))) if len(ratios) else None,
              'strong_bin_phase_median_abs_degrees': float(np.median(phase)) if len(phase) else None,
              'bands': []}
    for low, high in [(0, 2000), (2000, 8000), (8000, rate/2+1)]:
        band = (frequencies >= low) & (frequencies < high)
        result['bands'].append({'low_hz': low, 'high_hz': min(high, rate/2),
                                'complex_relative_l2': relative_l2(aft[:, band], fft[:, band])})
    peak_fft = np.argmax(np.abs(fft[:, 1:]), axis=1)+1
    peak_aft = np.argmax(np.abs(aft[:, 1:]), axis=1)+1
    result['dominant_bin_agreement_fraction'] = float(np.mean(peak_fft == peak_aft))
    return result


def db(spectrum, n):
    # Plot only interior positive-frequency bins: DC/Nyquist conventions differ
    # for a discrete sequence and a continuous periodic function.
    return 20*np.log10(np.maximum(np.abs(spectrum) * 2/(periodic_hann(n).sum()*32768), 1e-8))


def save_figure(fig, folder, name):
    fig.savefig(folder / f'{name}.png', dpi=160, facecolor='white')
    fig.savefig(folder / f'{name}.svg', facecolor='white')
    plt.close(fig)


def render_plots(report, arrays, output):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'font.family': 'DejaVu Sans'})
    n, rate = report['frame_size'], report['sample_rate']
    freq = np.fft.rfftfreq(n, 1/rate)
    visible = (freq > 0) & (freq < rate/2)
    fig, axes = plt.subplots(len(arrays), 2, figsize=(12, 3.0*len(arrays)), squeeze=False,
                             layout='constrained')
    for row, (excerpt, data) in enumerate(zip(report['excerpts'], arrays)):
        f = np.sqrt(np.mean(np.abs(data['fft'])**2, axis=0))
        a = np.sqrt(np.mean(np.abs(data['aft'])**2, axis=0))
        linear = f * np.sinc(np.arange(len(f))/n)**2
        ax, diff = axes[row]
        for key, values in [('fft', f), ('aft', a), ('linear', linear)]:
            ax.semilogx(freq[visible], db(values, n)[visible], color=COLORS[key],
                        label={'fft': 'Classical FFT', 'aft': 'Analytic AFT', 'linear': 'Linear interpolant FT'}[key],
                        lw=1.6 if key != 'linear' else 1, ls='--' if key == 'linear' else '-')
        ax.set(title=f'Beethoven at {excerpt["start_seconds"]:g} s · RMS over {excerpt["frames"]} windows',
               ylabel='Coefficient magnitude (dBFS)', xlabel='Frequency (Hz)', ylim=(-130, 0), xlim=(freq[1], rate/2))
        ax.grid(alpha=0.2); ax.legend(fontsize=8, loc='lower left')
        ratio = 20*np.log10(np.maximum(a, 1e-30)/np.maximum(f, 1e-30))
        diff.semilogx(freq[visible], ratio[visible], color=COLORS['aft'], label='AFT / FFT')
        diff.semilogx(freq[visible], 40*np.log10(np.sinc(np.arange(len(f))/n))[visible],
                     color=COLORS['linear'], ls='--', label='Linear FT / FFT')
        diff.axhline(0, color='#777777', lw=0.8)
        diff.set(title='Representation difference · same bins and normalization', ylabel='Magnitude ratio (dB)',
                 xlabel='Frequency (Hz)', xlim=(freq[1], rate/2), ylim=(-12, 6))
        diff.grid(alpha=0.2); diff.legend(fontsize=8)
    fig.suptitle('FFT samples versus the continuous angular interpolant', fontsize=15)
    save_figure(fig, output, 'aft-spectra')

    data, excerpt = arrays[0], report['excerpts'][0]
    f, a = db(data['fft'], n)[:, visible], db(data['aft'], n)[:, visible]
    times = excerpt['start_seconds'] + (np.arange(len(f))*report['hop']+n/2)/rate
    step = report['hop']/rate
    extent = (times[0]-step/2, times[-1]+step/2, freq[visible][0]/1000, freq[visible][-1]/1000)
    maximum = math.ceil(max(f.max(), a.max())/10)*10
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True, layout='constrained')
    for ax, values, label in zip(axes[:2], [f, a], ['Classical FFT', 'Analytic AFT']):
        im = ax.imshow(values.T, origin='lower', aspect='auto', extent=extent, cmap='magma', vmin=maximum-80, vmax=maximum)
        ax.set(title=label, ylabel='Frequency (kHz)')
        fig.colorbar(im, ax=ax, label='dBFS', pad=0.01)
    delta = np.ma.masked_where(f < -90, a-f)
    im = axes[2].imshow(delta.T, origin='lower', aspect='auto', extent=extent, cmap='coolwarm', vmin=-1, vmax=1)
    axes[2].set(title='AFT minus FFT · weak bins masked · color saturated at ±1 dB', ylabel='Frequency (kHz)', xlabel='Track time (s)')
    fig.colorbar(im, ax=axes[2], label='dB difference', pad=0.01, extend='both')
    fig.suptitle(f'Beethoven · {n}-sample periodic Hann · hop {report["hop"]}', fontsize=15)
    save_figure(fig, output, 'aft-spectrogram')

    bench = report['scaling']
    sizes = [b['frame_size'] for b in bench]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained')
    for key, label, color in [('fft_median_us', 'NumPy real FFT', COLORS['fft']),
                              ('fit_median_us', 'AFT geometry preparation', '#777777'),
                              ('aft_median_us', 'AFT from prepared geometry', COLORS['aft'])]:
        axes[0].loglog(sizes, [b[key] for b in bench], 'o-', label=label, color=color)
    axes[0].set(title='Measured transform cost · warmed calls', xlabel='Frame size N', ylabel='Median time (µs)')
    axes[0].legend(fontsize=9); axes[0].grid(which='both', alpha=0.2)
    for e in report['excerpts']:
        checks = e['quadrature']
        axes[1].loglog([c['oversample'] for c in checks], [c['complex_relative_l2'] for c in checks], 'o-',
                       label=f'{e["start_seconds"]:g} s excerpt')
    axes[1].set(title='AFT versus dense integration of the same curve', xlabel='Midpoint samples per original interval',
                ylabel='Relative complex L2 discrepancy')
    axes[1].legend(fontsize=9); axes[1].grid(which='both', alpha=0.2)
    fig.suptitle('Runtime and independent numerical verification', fontsize=15)
    save_figure(fig, output, 'aft-benchmark')

    # 48 measured windows, uniformly selected across the first excerpt. No audio.
    selected = np.unique(np.linspace(0, len(f)-1, min(48, len(f))).astype(int))
    fig, (ax, diff) = plt.subplots(2, 1, figsize=(10, 6), layout='constrained')
    line_f, = ax.semilogx(freq[visible], f[0], color=COLORS['fft'], label='Classical FFT', lw=1.4)
    line_a, = ax.semilogx(freq[visible], a[0], color=COLORS['aft'], label='Analytic AFT', lw=1.4)
    line_d, = diff.semilogx(freq[visible], a[0]-f[0], color=COLORS['aft'], label='AFT / FFT (FFT ≥ −90 dBFS)')
    diff.semilogx(freq[visible], 40*np.log10(np.sinc(np.arange(n//2+1)/n))[visible],
                 color=COLORS['linear'], ls='--', label='Linear interpolant / FFT')
    ax.set(ylim=(-120, maximum+5), ylabel='Coefficient magnitude (dBFS)', xlim=(freq[1], rate/2))
    diff.set(ylim=(-1, 0.5), xlabel='Frequency (Hz)', ylabel='Magnitude ratio (dB)', xlim=(freq[1], rate/2))
    for axis in (ax, diff):
        axis.legend(loc='lower left', fontsize=9); axis.grid(alpha=0.2)
    title = ax.set_title('')
    fig.suptitle('AFT / FFT · measured Beethoven windows · silent, slowed animation', fontsize=12)

    def update(index):
        line_f.set_ydata(f[index]); line_a.set_ydata(a[index]); line_d.set_ydata(np.where(f[index] >= -90, a[index]-f[index], np.nan))
        title.set_text(f'Track time {times[index]:.3f} s  |  N={n}, hop={report["hop"]}')
        return line_f, line_a, line_d, title

    animation = FuncAnimation(fig, update, frames=selected, interval=100, blit=False)
    animation.save(output/'aft-evolution.gif', writer=PillowWriter(fps=10), dpi=90)
    plt.close(fig)
    report['animation'] = {'frames': len(selected), 'fps': 10, 'source_start_seconds': float(times[selected[0]]),
                           'source_end_seconds': float(times[selected[-1]]), 'silent': True}


def build_binary():
    cargo = shutil.which('cargo')
    if not cargo:
        raise ValueError('Cargo is required, or pass --binary')
    manifest = str(ROOT/'Cargo.toml')
    subprocess.run([cargo, 'build', '--offline', '--release', '--example', 'aft_bench', '--manifest-path', manifest], check=True)
    metadata = subprocess.run([cargo, 'metadata', '--offline', '--no-deps', '--format-version', '1',
                               '--manifest-path', manifest], capture_output=True, text=True, check=True)
    return Path(json.loads(metadata.stdout)['target_directory'])/'release/examples/aft_bench'


def decode(ffmpeg, source, start, seconds, rate):
    result = subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-ss', str(start), '-i', str(source),
                             '-t', str(seconds), '-map', '0:a:0', '-ac', '1', '-ar', str(rate),
                             '-f', 's16le', '-c:a', 'pcm_s16le', 'pipe:1'], capture_output=True, check=True)
    if not result.stdout or len(result.stdout)%2:
        raise ValueError('empty or incomplete PCM')
    return np.frombuffer(result.stdout, dtype='<i2').astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path, help='new directory outside the repository')
    parser.add_argument('--ffmpeg', default=os.environ.get('FFMPEG', 'ffmpeg'))
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--starts', nargs='+', type=float, default=[60, 180, 300])
    parser.add_argument('--seconds', type=float, default=2)
    parser.add_argument('--rate', type=int, default=44100)
    parser.add_argument('--frame', type=int, default=1024)
    parser.add_argument('--hop', type=int, default=512)
    parser.add_argument('--sizes', type=int, nargs='+', default=[256, 512, 1024, 2048, 4096])
    args = parser.parse_args()
    if (not math.isfinite(args.seconds) or not 0 < args.seconds <= 30
            or not 8000 <= args.rate <= 192000 or not 1 <= args.hop <= args.frame
            or any(not 2 <= n <= 8192 for n in [args.frame, *args.sizes])
            or any(not math.isfinite(s) or s < 0 for s in args.starts)
            or args.seconds*args.rate < max(args.frame, *args.sizes)):
        parser.error('invalid duration, rate, offsets or frame sizes')
    output, source = args.output.resolve(), args.input.resolve()
    if output.exists() or output == ROOT or ROOT in output.parents:
        parser.error('output must be a new directory outside the repository')
    if not source.is_file():
        parser.error('input recording is missing')
    ffmpeg = shutil.which(args.ffmpeg)
    if not ffmpeg:
        parser.error('FFmpeg missing; pass --ffmpeg')
    binary = args.binary or build_binary()
    output.mkdir(parents=True)
    report = {'experiment': 'AFT-v1-periodic-angular-average', 'date': time.strftime('%Y-%m-%d'),
              'sample_rate': args.rate, 'frame_size': args.frame, 'hop': args.hop,
              'requested_seconds': args.seconds, 'channels': 'FFmpeg mono downmix',
              'window': 'periodic Hann applied to PCM16 amplitudes before both transforms',
              'analysis': {'aperture': 'SignalRange', 'collinear_epsilon': 4e-6,
                           'endpoint_relative_tolerance': 1e-8},
              'environment': {'platform': platform.platform(), 'machine': platform.machine(),
                              'python': platform.python_version(), 'numpy': np.__version__,
                              'matplotlib': matplotlib.__version__, 'pillow': PIL.__version__,
                              'ffmpeg': subprocess.run([ffmpeg, '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]},
              'timing': 'Warm in-process calls; includes result allocation, excludes decode, window, text/JSON I/O, plotting; Rust release AFT vs NumPy rfft.',
              'excerpts': [], 'scaling': []}
    cargo = shutil.which('rustc')
    if cargo:
        report['environment']['rustc'] = subprocess.run([cargo, '--version'], capture_output=True, text=True, check=True).stdout.strip()
    cpu = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True) if platform.system() == 'Darwin' else None
    if cpu and cpu.returncode == 0:
        report['environment']['cpu'] = cpu.stdout.strip()
    arrays = []
    first_pcm = None
    for start in args.starts:
        print(f'Computing FFT/AFT windows at {start:g} s...', flush=True)
        pcm = decode(ffmpeg, source, start, args.seconds, args.rate)
        if first_pcm is None:
            first_pcm = pcm
        frames = window_frames(pcm, args.frame, args.hop)
        rows = rust_frames(binary, frames)
        aft, fft = as_complex(rows), np.fft.rfft(frames, axis=1)
        timings = timed_fft(frames)
        linear = fft * np.sinc(np.arange(args.frame//2+1)/args.frame)**2
        excerpt = {'start_seconds': start, 'decoded_samples': len(pcm), 'frames': len(frames),
                   'fft_median_us': float(np.median(timings)/1000),
                   'fit_median_us': float(np.median([r['fit_ns'] for r in rows])/1000),
                   'aft_median_us': float(np.median([r['aft_ns'] for r in rows])/1000),
                   'circle_halves': sum(r['circle_halves'] for r in rows),
                   'line_halves': sum(r['line_halves'] for r in rows),
                   'rejected_circle_halves': sum(r['rejected_circle_halves'] for r in rows),
                   'aft_vs_fft': differences(fft, aft, args.rate, args.frame),
                   'aft_vs_linear_complex_relative_l2': relative_l2(aft, linear), 'quadrature': []}
        # One central frame per excerpt, independently sampled and transformed.
        selected = len(frames)//2
        excerpt['quadrature_frame_index'] = selected
        for q in (8, 32, 128):
            row = rust_frames(binary, frames[selected:selected+1], dense=q)[0]
            numeric = dense_integral(row['dense'], args.frame, q)
            excerpt['quadrature'].append({'oversample': q,
                'complex_relative_l2': relative_l2(numeric, aft[selected]),
                'max_absolute_complex_error_pcm_samples': float(np.max(np.abs(numeric-aft[selected])))})
        arrays.append({'fft': fft, 'aft': aft})
        report['excerpts'].append(excerpt)
        print(f'  {len(frames)} windows; FFT {excerpt["fft_median_us"]:.2f} µs, '
              f'AFT {excerpt["aft_median_us"]:.2f} µs; '
              f'dense-Q128 discrepancy {excerpt["quadrature"][-1]["complex_relative_l2"]:.3g}', flush=True)
    for n in args.sizes:
        print(f'Timing N={n}...', flush=True)
        # Same central location for all sizes; record every repeat for dispersion.
        begin = len(first_pcm)//2-n//2
        frame = first_pcm[begin:begin+n] * periodic_hann(n)
        if len(frame) != n:
            raise ValueError('excerpt too short for benchmark size')
        row = rust_frames(binary, frame[None, :], repeats=11)[0]
        fft_times = timed_fft(frame[None, :], repeats=101)[0]
        report['scaling'].append({'frame_size': n, 'bins': n//2+1,
            'fft_median_us': float(np.median(fft_times)/1000),
            'fit_median_us': float(np.median(row['fit_ns'])/1000),
            'aft_median_us': float(np.median(row['aft_ns'])/1000),
            'fft_ns': fft_times.tolist(), 'fit_ns': row['fit_ns'], 'aft_ns': row['aft_ns']})
    print('Rendering spectrum plots, spectrogram, benchmark and GIF...', flush=True)
    render_plots(report, arrays, output)
    # These complex arrays can reconstruct sampled audio: keep them out of Git.
    np.savez_compressed(output/'private-spectra.npz', **{f'{key}_{i}': value
                          for i, data in enumerate(arrays) for key, value in data.items()})
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(output/'report.json')


if __name__ == '__main__':
    main()
