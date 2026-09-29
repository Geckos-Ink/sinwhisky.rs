#!/usr/bin/env python3
"""Local circle/subcircle/Fourier compression trials (NumPy, FFmpeg, Rust).

SWR1 is a bounded, experimental PCM16 envelope, not a stable audio format.
All size/error results come from serialized bytes decoded without source PCM.
See docs/COMPRESSION.md for the exact experiment and format.
"""
import argparse
import collections
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
import wave
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEADER = struct.Struct('<4sIHHI')
CIRCLE = struct.Struct('<Hddddb')
MAX_FRAMES = 2_000_000
MAX_BLOCK = 4096
METHODS = ('pcm', 'delta', 'line_tree', 'circle_tree', 'line_residual', 'circle_residual',
           'fourier_residual', 'line_fourier_residual', 'circle_fourier_residual')


def pcm_round(values):
    if not np.all(np.isfinite(values)) or np.max(np.abs(values), initial=0) > 1e9:
        raise ValueError('invalid or excessive predictor')
    return np.clip(np.rint(values), -32768, 32767).astype(np.int32)


def predict(model, lo, hi):
    """Uniform-x evaluation in the same aperture coordinates as Rust."""
    x = np.arange(lo, hi, dtype=np.float64)
    if model[0] == 'L':
        return model[1] + (x - lo) * model[2]
    _, anchor, aperture, cx, cy, radius, sign = model
    if (not all(math.isfinite(v) for v in model[1:]) or aperture <= 0
            or radius <= 0 or sign not in (-1, 1)):
        raise ValueError('invalid circle')
    dx = (x - anchor) * aperture - cx
    inside = radius * radius - dx * dx
    if np.any(inside < 0):
        raise ValueError('circle extent')
    return (cy + sign * np.sqrt(inside)) * 32768.0


def model_bytes(model):
    if model[0] == 'L':
        return b'L' + struct.pack('<dd', *model[1:])
    return b'C' + CIRCLE.pack(*model[1:])


def models_for(pcm, circles, lo, hi, allow_circles=True):
    length = hi - lo
    slope = (float(pcm[hi - 1]) - float(pcm[lo])) / max(1, length - 1)
    models = [('L', float(pcm[lo]), slope)]
    if allow_circles and length >= 3:
        for anchor in sorted({lo + 1, (lo + hi - 1) // 2, hi - 2}):
            c = circles[anchor - 1]
            if c is not None:
                models.append(('C', *c[:6]))
    result = []
    for model in models:
        try:
            values = predict(model, lo, hi)
            # Raw source/shorter spans remain available when extrapolation is unstable.
            if np.all(np.isfinite(values)) and np.max(np.abs(values)) <= 1e9:
                result.append((model, pcm_round(values)))
        except ValueError:
            pass
    return result


def compressed_cost(data):
    return len(zlib.compress(data, 9))


def quantized_residual(source, prediction, step):
    residual = np.rint((source.astype(np.float64) - prediction) / step).astype('<i4')
    return residual.tobytes()


def tree_encode(pcm, circles, tolerance, with_residual=False, allow_circles=True):
    step = 2 * tolerance + 1

    def visit(lo, hi):
        candidates = [b'R' + pcm[lo:hi].astype('<i2').tobytes()]
        for model, values in models_for(pcm, circles, lo, hi, allow_circles):
            descriptor = model_bytes(model)
            if np.max(np.abs(pcm[lo:hi] - values)) <= tolerance:
                candidates.append(descriptor)
            if with_residual:
                candidates.append(b'Q' + struct.pack('<H', step) + descriptor
                                  + quantized_residual(pcm[lo:hi], values, step))
        if hi - lo > 4:
            mid = (lo + hi) // 2
            candidates.append(b'S' + visit(lo, mid) + visit(mid, hi))
        return min(candidates, key=compressed_cost)

    return visit(0, len(pcm))


def fourier_descriptor(signal, keep):
    n = len(signal)
    spectrum = np.fft.rfft(signal) / n
    weights = np.full(len(spectrum), 2.0)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0
    # Rank by the energy of the conjugate pair, with deterministic tie order.
    order = np.argsort(-weights * np.abs(spectrum)**2, kind='stable')[:keep]
    indices = sorted(int(i) for i in order)
    payload = b'F' + struct.pack('<H', len(indices))
    quantized = np.zeros_like(spectrum)
    for i in indices:
        record = struct.pack('<Hff', i, spectrum[i].real, spectrum[i].imag)
        _, re, im = struct.unpack('<Hff', record)
        quantized[i] = complex(re, im)
        payload += record
    return payload, np.fft.irfft(quantized * n, n=n)


def fourier_encode(pcm, circles, tolerance, keep, base_mode):
    bases = [(b'', np.zeros(len(pcm), dtype=np.int32))]
    if base_mode != 'none':
        bases = [(model_bytes(model), values) for model, values in
                 models_for(pcm, circles, 0, len(pcm), base_mode == 'circle')]
    candidates = [b'R' + pcm.astype('<i2').tobytes()]
    step = 2 * tolerance + 1
    for descriptor, base in bases:
        fourier, values = fourier_descriptor(pcm - base, keep)
        # B has explicitly additive semantics; its base is rounded/clipped first.
        predictor = b'B' + descriptor + fourier if descriptor else fourier
        prediction = pcm_round(base + values)
        candidates.append(b'Q' + struct.pack('<H', step) + predictor
                          + quantized_residual(pcm, prediction, step))
    return min(candidates, key=compressed_cost)


class Reader:
    def __init__(self, data):
        self.data, self.pos = data, 0
        self.counts = collections.Counter()

    def take(self, n):
        if n < 0 or self.pos + n > len(self.data):
            raise ValueError('truncated stream')
        value = self.data[self.pos:self.pos + n]
        self.pos += n
        return value

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))

    def model(self, tag, lo, hi):
        if tag == b'L':
            model = ('L', *self.unpack('<dd'))
        elif tag == b'C':
            model = ('C', *self.unpack(CIRCLE.format))
        else:
            raise ValueError('expected line/circle')
        self.counts[tag.decode()] += 1
        return pcm_round(predict(model, lo, hi))

    def fourier(self, n):
        self.counts['F'] += 1
        count, = self.unpack('<H')
        if count > n // 2 + 1:
            raise ValueError('too many Fourier bins')
        spectrum = np.zeros(n // 2 + 1, dtype=np.complex128)
        previous = -1
        for _ in range(count):
            index, re, im = self.unpack('<Hff')
            if index <= previous or index >= len(spectrum) or not math.isfinite(re + im):
                raise ValueError('invalid Fourier record')
            if (index == 0 or (n % 2 == 0 and index == n // 2)) and im != 0:
                raise ValueError('complex DC/Nyquist')
            previous = index
            spectrum[index] = complex(re, im)
        return np.fft.irfft(spectrum * n, n=n)

    def predictor(self, lo, hi):
        tag = self.take(1)
        if tag in (b'L', b'C'):
            return self.model(tag, lo, hi)
        if tag == b'F':
            return pcm_round(self.fourier(hi - lo))
        if tag == b'B':
            self.counts['B'] += 1
            base = self.model(self.take(1), lo, hi)
            if self.take(1) != b'F':
                raise ValueError('expected Fourier correction')
            return pcm_round(base + self.fourier(hi - lo))
        raise ValueError('invalid predictor')

    def node(self, lo, hi, depth=0):
        if depth > 12 or lo >= hi:
            raise ValueError('invalid tree coverage')
        tag = self.take(1)
        n = hi - lo
        if tag == b'S':
            if n <= 4:
                raise ValueError('split below minimum leaf')
            self.counts['S'] += 1
            mid = (lo + hi) // 2
            return np.concatenate((self.node(lo, mid, depth + 1), self.node(mid, hi, depth + 1)))
        if tag == b'R':
            self.counts['R'] += 1
            return np.frombuffer(self.take(n * 2), dtype='<i2').astype(np.int32)
        if tag == b'D':
            self.counts['D'] += 1
            values = np.frombuffer(self.take(n * 4), dtype='<i4').astype(np.int64).cumsum()
            if np.any(values < -32768) or np.any(values > 32767):
                raise ValueError('invalid delta PCM')
            return values.astype(np.int32)
        if tag == b'Q':
            self.counts['Q'] += 1
            step, = self.unpack('<H')
            if step == 0 or step % 2 == 0:
                raise ValueError('invalid residual step')
            base = self.predictor(lo, hi)
            residual = np.frombuffer(self.take(n * 4), dtype='<i4').astype(np.int64)
            return np.clip(base + residual * step, -32768, 32767).astype(np.int32)
        if tag in (b'L', b'C'):
            return self.model(tag, lo, hi)
        raise ValueError('invalid node tag')


def validate_metadata(rate, channels, block, frames):
    if (not 1 <= rate <= 384000 or not 1 <= channels <= 2
            or not 4 <= block <= MAX_BLOCK or block & (block - 1)
            or not 1 <= frames <= MAX_FRAMES):
        raise ValueError('invalid or excessive stream metadata')


def decode_stream(data):
    reader = Reader(data)
    magic, rate, channels, block, frames = reader.unpack(HEADER.format)
    if magic != b'SWR1':
        raise ValueError('unknown stream version')
    validate_metadata(rate, channels, block, frames)
    output = np.empty((frames, channels), dtype=np.int32)
    counts = collections.Counter()
    for ch in range(channels):
        for lo in range(0, frames, block):
            n = min(block, frames - lo)
            size, = reader.unpack('<I')
            if not 1 <= size <= 128 * block + 1024:
                raise ValueError('invalid compressed block size')
            packed = reader.take(size)
            inflater = zlib.decompressobj()
            limit = 128 * n + 1024
            raw = inflater.decompress(packed, limit + 1)
            if len(raw) > limit or not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
                raise ValueError('invalid or excessive compressed block')
            node = Reader(raw)
            output[lo:lo+n, ch] = node.node(0, n)
            if node.pos != len(raw):
                raise ValueError('trailing node data')
            counts.update(node.counts)
    if reader.pos != len(data):
        raise ValueError('trailing stream data')
    return output, rate, dict(counts)


def encode_stream(pcm, rate, block, circle_blocks, method, tolerance=0, keep=8):
    if pcm.ndim != 2:
        raise ValueError('PCM must have frame/channel axes')
    frames, channels = pcm.shape
    validate_metadata(rate, channels, block, frames)
    if (method not in METHODS or not 0 <= tolerance <= 32767 or not isinstance(tolerance, int)
            or not 1 <= keep <= block // 2 + 1 or np.any(pcm < -32768)
            or np.any(pcm > 32767) or not np.all(np.isfinite(pcm)) or np.any(pcm != np.rint(pcm))):
        raise ValueError('invalid encoder parameters or PCM16 values')
    output = bytearray(HEADER.pack(b'SWR1', rate, channels, block, frames))
    for ch in range(channels):
        for b, lo in enumerate(range(0, frames, block)):
            source = pcm[lo:lo+block, ch].astype(np.int32)
            circles = circle_blocks[ch][b]
            if len(circles) != max(0, len(source) - 2):
                raise ValueError('model coverage mismatch')
            if method == 'pcm':
                raw = b'R' + source.astype('<i2').tobytes()
            elif method == 'delta':
                raw = b'D' + np.diff(source, prepend=0).astype('<i4').tobytes()
            elif method in ('line_tree', 'circle_tree', 'line_residual', 'circle_residual'):
                raw = tree_encode(source, circles, tolerance, method.endswith('_residual'),
                                  method.startswith('circle'))
            else:
                base = {'fourier_residual': 'none', 'line_fourier_residual': 'line',
                        'circle_fourier_residual': 'circle'}[method]
                raw = fourier_encode(source, circles, tolerance, keep, base)
            packed = zlib.compress(raw, 9)
            output.extend(struct.pack('<I', len(packed)))
            output.extend(packed)
    return bytes(output)


def export_circles(pcm, binary, block):
    channels = []
    for ch in range(pcm.shape[1]):
        source = '\n'.join(format(float(v) / 32768.0, '.17g') for v in pcm[:, ch])
        result = subprocess.run([str(binary), str(block)], input=source, text=True,
                                capture_output=True, check=True)
        blocks = [json.loads(line) for line in result.stdout.splitlines()]
        if len(blocks) != math.ceil(len(pcm) / block):
            raise ValueError('Rust model block count mismatch')
        for b, circles in enumerate(blocks):
            for i, circle in enumerate(circles):
                if circle is not None:
                    if circle[0] != i + 1:
                        raise ValueError('Rust model anchor mismatch')
                    value = predict(('C', *circle[:6]), i + 1, i + 2)[0] / 32768.0
                    if not math.isclose(value, circle[6], abs_tol=1e-12, rel_tol=1e-12):
                        raise ValueError('Python/Rust circle evaluation mismatch')
        channels.append(blocks)
    return channels


def hierarchy_diagnostics(pcm, circles, depth_limit=4):
    """Compare a parent's unchanged subarcs with children selected from source models.

    Subdivision itself adds no information. Neither the parent nor its children
    can certify source error without comparison to the source (or residual data).
    """
    levels = collections.defaultdict(lambda: [0, 0.0, 0.0, 0, 0])

    def visit(lo, hi, depth):
        models = models_for(pcm, circles, lo, hi)
        model, prediction = min(models, key=lambda pair: np.sum((pcm[lo:hi].astype(float) - pair[1])**2))
        if depth >= depth_limit or hi - lo < 8:
            return
        mid = (lo + hi) // 2
        for a, b in [(lo, mid), (mid, hi)]:
            children = models_for(pcm, circles, a, b)
            child, corrected = min(children, key=lambda pair: np.sum((pcm[a:b].astype(float) - pair[1])**2))
            parent = prediction[a-lo:b-lo]
            row = levels[depth]
            row[0] += b - a
            row[1] += float(np.sum((pcm[a:b] - parent).astype(float)**2))
            row[2] += float(np.sum((pcm[a:b] - corrected).astype(float)**2))
            row[3] += int(model[0] == 'C')
            row[4] += int(child[0] == 'C')
            visit(a, b, depth + 1)

    visit(0, len(pcm), 0)
    return levels


def metrics(reference, actual, block):
    error = reference.astype(np.float64) - actual
    power = float(np.mean(reference.astype(np.float64)**2))
    mse = float(np.mean(error**2))
    # Full complex spectral L2 error is equivalent to time-domain L2 by Parseval.
    # Additionally inspect reconstructed block-boundary jumps against the source.
    boundaries = np.arange(block, len(reference), block)
    jumps = error[boundaries] - error[boundaries - 1]
    return {'max_error_pcm16': int(np.max(np.abs(error))),
            'rmse_pcm16': math.sqrt(mse),
            'snr_db': 10 * math.log10(power / mse) if mse and power else None,
            'relative_l2_error': math.sqrt(mse / power) if power else None,
            'max_boundary_jump_error_pcm16': int(np.max(np.abs(jumps), initial=0))}


def write_wav(path, pcm, rate):
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(pcm.shape[1])
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.astype('<i2').tobytes())


def build_exporter():
    cargo = shutil.which('cargo')
    if cargo is None:
        raise ValueError('Cargo is required or pass --exporter')
    subprocess.run([cargo, 'build', '--offline', '--release', '--example', 'export_circles',
                    '--manifest-path', str(ROOT / 'Cargo.toml')], check=True)
    result = subprocess.run([cargo, 'metadata', '--offline', '--no-deps', '--format-version', '1',
                             '--manifest-path', str(ROOT / 'Cargo.toml')], capture_output=True, text=True, check=True)
    return Path(json.loads(result.stdout)['target_directory']) / 'release/examples/export_circles'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path, help='new directory outside the repository')
    parser.add_argument('--starts', type=float, nargs='+', default=[60, 180, 300])
    parser.add_argument('--seconds', type=float, default=1.0)
    parser.add_argument('--rate', type=int, default=44100)
    parser.add_argument('--channels', type=int, choices=[1, 2], default=2)
    parser.add_argument('--block', type=int, default=256)
    parser.add_argument('--tolerances', type=int, nargs='+', default=[0, 16, 128])
    parser.add_argument('--keep', type=int, default=8, help='retained real-DFT bins including conjugates')
    parser.add_argument('--ffmpeg', default=os.environ.get('FFMPEG', 'ffmpeg'))
    parser.add_argument('--exporter', type=Path)
    args = parser.parse_args()
    if (not math.isfinite(args.seconds) or args.seconds <= 0
            or any(not math.isfinite(x) or x < 0 for x in args.starts)
            or any(not 0 <= x <= 32767 for x in args.tolerances)
            or not 1 <= args.keep <= args.block // 2 + 1):
        parser.error('invalid duration, offsets, tolerances, or Fourier count')
    validate_metadata(args.rate, args.channels, args.block, math.ceil(args.seconds * args.rate))
    source, destination = args.input.resolve(), args.output.resolve()
    if not source.is_file():
        parser.error('input is not a file')
    if destination == ROOT or ROOT in destination.parents:
        parser.error('audio artifacts must be outside the repository')
    if destination.exists():
        parser.error('output directory must be new')
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        parser.error('FFmpeg not found; pass --ffmpeg')
    exporter = args.exporter or build_exporter()
    version = subprocess.run([ffmpeg, '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    destination.mkdir(parents=True)
    report = {'format': 'SWR1', 'numpy': np.__version__, 'zlib': zlib.ZLIB_RUNTIME_VERSION,
              'ffmpeg': version, 'rate': args.rate, 'channels': args.channels, 'block': args.block,
              'fourier_bins': args.keep, 'tolerances_pcm16': args.tolerances,
              'normalization': 'PCM16 / 32768 for Rust; no gain; independent channels',
              'excerpts': []}
    for excerpt_index, start in enumerate(args.starts):
        folder = destination / f'excerpt-{excerpt_index}'
        folder.mkdir()
        result = subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-ss', str(start), '-i', str(source),
                                 '-t', str(args.seconds), '-map', '0:a:0', '-ar', str(args.rate), '-ac', str(args.channels),
                                 '-f', 's16le', '-c:a', 'pcm_s16le', 'pipe:1'], capture_output=True, check=True)
        if not result.stdout or len(result.stdout) % (2 * args.channels):
            raise ValueError('empty or incomplete decoded PCM')
        pcm = np.frombuffer(result.stdout, dtype='<i2').reshape(-1, args.channels).astype(np.int32)
        if len(pcm) > MAX_FRAMES:
            raise ValueError('decoded excerpt exceeds frame limit')
        write_wav(folder / 'reference.wav', pcm, args.rate)
        begun = time.perf_counter()
        circles = export_circles(pcm, exporter, args.block)
        analysis_seconds = time.perf_counter() - begun
        levels = collections.defaultdict(lambda: np.zeros(5))
        for ch in range(args.channels):
            for b, lo in enumerate(range(0, len(pcm), args.block)):
                for depth, row in hierarchy_diagnostics(pcm[lo:lo+args.block, ch], circles[ch][b]).items():
                    levels[depth] += row
        hierarchy = [{'depth': depth, 'parent_rmse_pcm16': math.sqrt(row[1]/row[0]),
                      'children_rmse_pcm16': math.sqrt(row[2]/row[0]),
                      'parent_circle_uses': int(row[3]), 'child_circle_uses': int(row[4])}
                     for depth, row in sorted(levels.items())]
        excerpt = {'start_seconds': start, 'frames': len(pcm), 'pcm16_bytes': pcm.nbytes // 2,
                   'analysis_export_seconds': analysis_seconds, 'hierarchy': hierarchy, 'trials': []}
        flac = folder / 'reference.flac'
        subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-n', '-i', str(folder / 'reference.wav'),
                        '-c:a', 'flac', '-compression_level', '8', str(flac)], check=True)
        verified = subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-i', str(flac),
                                   '-f', 's16le', 'pipe:1'], capture_output=True, check=True).stdout
        if verified != pcm.astype('<i2').tobytes():
            raise AssertionError('FLAC baseline failed exact round-trip')
        excerpt['flac_bytes'] = flac.stat().st_size
        for tolerance in args.tolerances:
            for method in METHODS:
                if method in ('pcm', 'delta') and tolerance != args.tolerances[0]:
                    continue
                begun = time.perf_counter()
                encoded = encode_stream(pcm, args.rate, args.block, circles, method, tolerance, args.keep)
                encode_seconds = time.perf_counter() - begun
                begun = time.perf_counter()
                actual, rate, counts = decode_stream(encoded)
                decode_seconds = time.perf_counter() - begun
                errors = metrics(pcm, actual, args.block)
                if rate != args.rate or errors['max_error_pcm16'] > tolerance:
                    raise AssertionError('encoded round-trip violated requested error bound')
                label = f'{method}-e{tolerance}'
                (folder / f'{label}.swr').write_bytes(encoded)
                write_wav(folder / f'{label}.wav', actual, rate)
                trial = {'method': method, 'tolerance_pcm16': tolerance, 'encoded_bytes': len(encoded),
                         'pcm_ratio': len(encoded) / excerpt['pcm16_bytes'],
                         'encode_seconds': encode_seconds, 'decode_seconds': decode_seconds,
                         'selected_nodes': counts, **errors}
                excerpt['trials'].append(trial)
                print(f'offset={start:g} {label}: {len(encoded)} bytes, '
                      f'{trial["pcm_ratio"]:.3f} PCM, error={errors["max_error_pcm16"]}', flush=True)
        report['excerpts'].append(excerpt)
        (destination / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(destination / 'report.json')


if __name__ == '__main__':
    main()
