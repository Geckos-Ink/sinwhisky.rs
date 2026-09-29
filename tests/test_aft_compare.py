"""Optional benchmark/normalization checks, with no music recording required."""
import importlib.util
import pathlib
import unittest

try:
    import numpy as np
    import matplotlib
    import PIL
except ImportError:
    np = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
if np is not None:
    spec = importlib.util.spec_from_file_location('aft_compare', ROOT/'tools/aft_compare.py')
    aft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aft)


@unittest.skipIf(np is None, 'NumPy, Matplotlib and Pillow required for AFT benchmark tests')
class AftCompareTests(unittest.TestCase):
    def test_midpoint_fft_phase_and_scale_have_known_sinusoidal_result(self):
        n = 32
        for q in (1, 8, 32):
            t = (np.arange(n*q)+0.5)/q
            signal = 3.0 + 4*np.sin(2*np.pi*3*t/n+0.43)
            coefficients = aft.dense_integral(signal, n, q)
            expected = np.zeros(n//2+1, dtype=complex)
            expected[0] = 3*n
            expected[3] = -2j*n*np.exp(0.43j)
            np.testing.assert_allclose(coefficients, expected, atol=1e-11)
        with self.assertRaises(ValueError):
            aft.dense_integral([0], 32, 8)

    def test_hann_frames_and_zero_signal_metrics(self):
        signal = np.arange(32, dtype=float)
        frames = aft.window_frames(signal, 8, 4)
        self.assertEqual(frames.shape, (7, 8))
        self.assertEqual(aft.periodic_hann(8).sum(), 4.0)
        self.assertGreater(aft.periodic_hann(8)[-1], 0.0)
        np.testing.assert_allclose(frames[2], signal[8:16]*aft.periodic_hann(8))
        for n, hop in [(64, 8), (8, 0), (1, 1)]:
            with self.assertRaises(ValueError):
                aft.window_frames(signal, n, hop)
        zeros = np.zeros((2, 17), dtype=complex)
        differences = aft.differences(zeros, zeros, 44100, 32)
        self.assertIsNone(differences['complex_relative_l2'])
        self.assertIsNone(differences['strong_bin_phase_median_abs_degrees'])

    def test_release_rust_bridge_against_numerical_integration(self):
        binary = ROOT/'target/release/examples/aft_bench'
        if not binary.is_file():
            self.skipTest('build the aft_bench release example first')
        frames = np.array([[1000*np.sin(i*0.35) for i in range(32)], [100.0]*32])
        coarse = aft.rust_frames(binary, frames, dense=8)
        fine = aft.rust_frames(binary, frames, dense=64)
        exact = aft.as_complex(fine)
        for index, row in enumerate(fine):
            numeric = aft.dense_integral(row['dense'], 32, 64)
            self.assertLess(np.linalg.norm(numeric-exact[index]), 0.1)
            if index == 0:
                low = aft.dense_integral(coarse[index]['dense'], 32, 8)
                self.assertLess(np.linalg.norm(numeric-exact[index]), np.linalg.norm(low-exact[index])/40)
            self.assertEqual(row['circle_halves']+row['line_halves'], 64)
        self.assertAlmostEqual(exact[1, 0].real, 3200)
        np.testing.assert_allclose(exact[1, 1:], 0, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
