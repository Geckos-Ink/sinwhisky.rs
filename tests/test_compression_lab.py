"""Optional NumPy research-format tests; real recordings are never required."""
import importlib.util
import pathlib
import struct
import unittest
import zlib

try:
    import numpy as np
except ImportError:
    np = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
if np is not None:
    spec = importlib.util.spec_from_file_location('compression_lab', ROOT / 'tools/compression_lab.py')
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)


@unittest.skipIf(np is None, 'NumPy is required for compression research tests')
class CompressionLabTests(unittest.TestCase):
    def circles(self, pcm, block):
        return [[[None] * max(0, len(pcm[lo:lo+block]) - 2)
                 for lo in range(0, len(pcm), block)] for _ in range(pcm.shape[1])]

    def envelope(self, raw, frames=16):
        packed = zlib.compress(raw)
        return lab.HEADER.pack(b'SWR1', 44100, 1, 16, frames) + struct.pack('<I', len(packed)) + packed

    def test_every_method_round_trips_quantization_and_partial_blocks(self):
        rng = np.random.default_rng(103)
        for n in [1, 2, 3, 17, 64, 79]:
            signals = [np.zeros(n), np.linspace(-500, 500, n),
                       np.rint(16000 * np.sin(np.arange(n) * 0.3)),
                       rng.integers(-32768, 32768, n)]
            for signal in signals:
                pcm = np.column_stack((signal, -np.clip(signal, -32767, 32767))).astype(np.int32)
                circles = self.circles(pcm, 64)
                for method in lab.METHODS:
                    for tolerance in [0, 4, 128]:
                        data = lab.encode_stream(pcm, 44100, 64, circles, method, tolerance)
                        decoded, rate, _ = lab.decode_stream(data)
                        self.assertEqual(rate, 44100)
                        self.assertEqual(decoded.shape, pcm.shape)
                        self.assertLessEqual(np.max(np.abs(pcm - decoded)), tolerance)
                        if tolerance == 0:
                            np.testing.assert_array_equal(decoded, pcm)

    def test_circle_parent_restrictions_and_child_coverage(self):
        n = 128
        model = ('C', 64, 0.01, 0.0, -0.8, 1.0, 1)
        values = lab.predict(model, 0, n)
        np.testing.assert_array_equal(values, np.concatenate((lab.predict(model, 0, 64), lab.predict(model, 64, n))))
        pcm = lab.pcm_round(values).reshape(-1, 1)
        # Same geometry at other anchors requires shifting local cx, not changing radius.
        circles = [[[i, 0.01, (64-i)*0.01, -0.8, 1.0, 1, 0] for i in range(1, n-1)]]
        data = lab.encode_stream(pcm, 44100, 128, [circles], 'circle_tree', 0)
        decoded, _, counts = lab.decode_stream(data)
        np.testing.assert_array_equal(decoded, pcm)
        self.assertGreater(counts.get('C', 0), 0)
        self.assertLess(len(data), pcm.size * 2)
        # Force different local behaviour, including the previously deliberate branch mismatch.
        pcm[64:, 0] = -30000
        for method in ('circle_tree', 'circle_residual'):
            data = lab.encode_stream(pcm, 44100, 128, [circles], method, 1)
            decoded, _, _ = lab.decode_stream(data)
            self.assertLessEqual(np.max(np.abs(decoded - pcm)), 1)

    def test_fourier_phase_conjugacy_quantization_and_linearity(self):
        n = 128
        t = np.arange(n)
        source = 10 + 2000*np.cos(2*np.pi*3*t/n + 0.4) + 500*(-1.0)**t
        descriptor, reconstructed = lab.fourier_descriptor(source, 3)
        reader = lab.Reader(descriptor)
        self.assertEqual(reader.take(1), b'F')
        actual = reader.fourier(n)
        np.testing.assert_array_equal(actual, reconstructed)
        self.assertLess(np.max(np.abs(source-actual)), 0.001)
        predictor = lab.predict(('C', 64, 0.01, 0.0, 0.0, 1.0, 1), 0, n)
        np.testing.assert_allclose(np.fft.rfft(source),
                                   np.fft.rfft(predictor) + np.fft.rfft(source-predictor), atol=1e-9)

    def test_truncated_corrupt_and_oversized_streams_rejected(self):
        pcm = np.arange(19, dtype=np.int32).reshape(-1, 1)
        data = lab.encode_stream(pcm, 44100, 16, self.circles(pcm, 16), 'circle_residual', 0)
        for length in range(len(data)):
            with self.assertRaises((ValueError, zlib.error)):
                lab.decode_stream(data[:length])
        invalid = [data + b'x', b'BAD!' + data[4:],
                   lab.HEADER.pack(b'SWR1', 44100, 1, 16, lab.MAX_FRAMES+1),
                   self.envelope(b'Z'), self.envelope(b'S' * 20),
                   self.envelope(b'R' + b'\0' * 33),
                   self.envelope(b'R' + b'\0' * 100000),
                   self.envelope(b'L' + struct.pack('<dd', float('nan'), 0)),
                   self.envelope(b'C' + lab.CIRCLE.pack(1, 1, 0, 0, -1, 1)),
                   self.envelope(b'Q' + struct.pack('<H', 0)),
                   self.envelope(b'Q' + struct.pack('<H', 1) + b'F' + struct.pack('<H', 1000)),
                   self.envelope(b'Q' + struct.pack('<H', 1) + b'F' + struct.pack('<HHff', 1, 0, 0, 1))]
        for bad in invalid:
            with self.assertRaises((ValueError, zlib.error)):
                lab.decode_stream(bad)

    def test_encoder_parameter_validation(self):
        pcm = np.zeros((16, 1), dtype=np.int32)
        circles = self.circles(pcm, 16)
        for method, tolerance in [('unknown', 0), ('pcm', -1), ('pcm', 32768), ('pcm', 0.1)]:
            with self.assertRaises(ValueError):
                lab.encode_stream(pcm, 44100, 16, circles, method, tolerance)
        for block in [0, 3, 17, 8192]:
            with self.assertRaises(ValueError):
                lab.encode_stream(pcm, 44100, block, circles, 'pcm')
        with self.assertRaises(ValueError):
            lab.encode_stream(pcm+32768, 44100, 16, circles, 'pcm')

    def test_rust_exported_models_decode_and_spectral_residuals(self):
        binary = ROOT / 'target/release/examples/export_circles'
        if not binary.is_file():
            self.skipTest('build cargo example export_circles first')
        source = np.rint(12000 * np.sin(np.arange(517)*0.025)).astype(np.int32).reshape(-1, 1)
        circles = lab.export_circles(source, binary, 256)
        self.assertTrue(any(c is not None for c in circles[0][0]))
        for method in lab.METHODS:
            data = lab.encode_stream(source, 44100, 256, circles, method, 4)
            actual, _, _ = lab.decode_stream(data)
            self.assertLessEqual(np.max(np.abs(source-actual)), 4)
        rows = lab.hierarchy_diagnostics(source[:256, 0], circles[0][0])
        self.assertTrue(rows)
        self.assertGreater(rows[0][1], rows[0][2])


if __name__ == '__main__':
    unittest.main()
