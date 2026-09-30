"""Independent checks for the circle-arc FAFT comparison tool; no recording needed."""
import pathlib
import sys
import unittest

try:
    import numpy as np
except ImportError:
    np = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
if np is not None:
    sys.path.insert(0, str(ROOT/'tools'))
    import faft_compare as faft


@unittest.skipIf(np is None, 'NumPy required')
class FaftCompareTests(unittest.TestCase):
    def test_independent_curve_and_quadrature(self):
        n, rate, radius, phase = 64, 0.37, 2.5, 0.4
        row = {'model': [[0, n, 1.0, radius, phase, rate, 0.0]]}
        t = np.arange(0, n, 0.5)
        np.testing.assert_allclose(faft.curve(row, t), 1.0+radius*np.sin(phase+rate*t), atol=1e-13)
        with self.assertRaises(ValueError):
            faft.curve({'model': [[0, 4, 0, 0, 0, 0, 0]]}, [5.0])
        # Midpoint rule on a very fine grid is a second independent integrator.
        fine = (np.arange(n*4096)+0.5)/4096
        g = faft.curve(row, fine)*(0.5-0.5*np.cos(2*np.pi*fine/n))/4096/(n/2)
        for omega in [0.0, rate, 1.3, 4.0]:
            expected = np.sum(g*np.exp(-1j*omega*fine))
            self.assertLess(abs(faft.Quadrature(row, n)([omega])[0]-expected), 1e-7)

    def test_matching_and_local_peaks(self):
        ok, distance = faft.match([1.0, 2.0], [1.05, 3.0], 0.1)
        self.assertEqual(ok.tolist(), [True, False])
        np.testing.assert_allclose(distance, [0.05])
        self.assertEqual(faft.local_peaks(np.array([3., 1., 2., 5., 4., 4.]), 1.5).tolist(), [0, 3])

    def test_release_bridge_exact_values_peaks_and_zoom(self):
        binary = ROOT/'target/release/examples/faft_bench'
        if not binary.is_file():
            self.skipTest('build the faft_bench release example')
        n = 256
        t = np.arange(n)
        rng = np.random.default_rng(4)
        frames = np.array([
            np.rint(3000*np.sin(0.2113*t+0.3)+900*np.cos(0.9021*t)+rng.normal(0, 20, n)),
            np.rint(2000*np.sin(0.05*t)*np.sin(1.7*t)),
        ])
        rows = faft.run_bench(binary, frames, 8, 64, 1, 32)
        summary = faft.summarize(rows, frames, 8000, 8)
        self.assertLess(summary['independent_quadrature_max_relative'], 1e-10)
        checked, total = map(int, summary['peaks_checked_local_max'].split('/'))
        self.assertEqual(checked, total)
        self.assertLess(summary['aft_same_curve_max_relative_l2'], 1e-10)
        zoom = summary['zoom_relative_l2']
        self.assertLess(zoom['64']['median'], zoom['1']['median'])
        self.assertLessEqual(summary['source_max_error'], 8)
        strongest = max(rows[0]['scan']['peaks'], key=lambda p: p[1]**2+p[2]**2)
        self.assertAlmostEqual(strongest[0], 0.2113, delta=2e-3)


if __name__ == '__main__':
    unittest.main()
