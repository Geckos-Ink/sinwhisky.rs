"""Independent FAFT synthesis/FFT checks; no source recording required."""
import copy
import pathlib
import sys
import unittest

try:
    import numpy as np
    import matplotlib
    import PIL
except ImportError:
    np = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
if np is not None:
    sys.path.insert(0, str(ROOT/'tools'))
    import faft_compare as faft


@unittest.skipIf(np is None, 'NumPy, Matplotlib and Pillow required')
class FaftCompareTests(unittest.TestCase):
    def test_independent_circle_synthesis_phase_and_coverage(self):
        row = {'scale':2., 'models':[{'start':0,'end':8,'first':1.,'last':1.,
            'circles':[{'bin':2,'re':3.,'im':-4.,'bits':8}]}]}
        t = np.arange(0,8,.25)
        expected = 2*(1+3*np.cos(np.pi*t/2)+4*np.sin(np.pi*t/2))
        np.testing.assert_allclose(faft.project_model(row,t),expected,atol=1e-13)
        with self.assertRaises(ValueError):
            faft.project_model(row,[-1])
        row['models'] *= 2
        with self.assertRaises(ValueError):
            faft.project_model(row,[1])

    def test_release_bridge_numpy_and_measured_error(self):
        binary=ROOT/'target/release/examples/faft_bench'
        if not binary.is_file(): self.skipTest('build faft_bench release example')
        n=128;t=np.arange(n)
        frames=np.array([1000*np.cos(2*np.pi*7*t/n+.5)+.8*np.sin(2*np.pi*15*t/n),
                         10+2*t, np.where(t<n/2,500*np.sin(2*np.pi*3*t/64),500*np.sin(2*np.pi*13*t/64))])
        rows=faft.rust_frames(binary,frames,.01,16)
        for x,row in zip(frames,rows):
            z=np.array(row['spectrum'])
            np.testing.assert_allclose(z[:,0]+1j*z[:,1],np.fft.rfft(x),atol=1e-9)
        self.assertEqual(rows[0]['circles'],2)
        self.assertEqual(rows[1]['circles'],0)
        self.assertEqual(rows[2]['blocks'],2)
        broken=copy.deepcopy(rows[0]);broken['decoded'][0]+=10
        with self.assertRaises(ValueError):
            faft.validate_row(frames[0],broken,.01)
        stats=faft.summary(rows,frames)
        self.assertLess(stats['kernel_vs_numpy_relative_l2'],1e-14)
        self.assertLessEqual(stats['worst_frame_rmse'],.01)


if __name__=='__main__': unittest.main()
