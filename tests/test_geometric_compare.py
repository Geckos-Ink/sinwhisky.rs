"""Independent geometry prediction and spectrum checks, with synthetic signals."""
import copy
import pathlib
import sys
import unittest
try:
    import numpy as np
    import matplotlib
    import PIL
except ImportError:
    np=None
ROOT=pathlib.Path(__file__).resolve().parents[1]
if np is not None:
    sys.path.insert(0,str(ROOT/'tools'))
    import geometric_compare as geo

@unittest.skipIf(np is None,'NumPy, Matplotlib, Pillow required')
class GeometryCompareTests(unittest.TestCase):
    def test_reference_prediction_coverage_and_phase(self):
        row={'scale':10.,'models':[{'start':0,'end':4,'kind':'circle','cy':2.,'radius':3.,
                                  'angles':[-np.pi/2,0,np.pi/2],'middle':2}]}
        t=np.arange(0,4,.125)
        np.testing.assert_allclose(geo.predict(row,t),20-30*np.cos(np.pi*t/4),atol=1e-12)
        with self.assertRaises(ValueError):geo.predict(row,[-1])
        row['models']*=2
        with self.assertRaises(ValueError):geo.predict(row,[1])

    def test_release_geometry_spectra_zoom_and_heldout_points(self):
        binary=ROOT/'target/release/examples/geometric_bench'
        if not binary.is_file():self.skipTest('build geometric_bench release example')
        n=64;t=np.arange(n)
        x=np.array([800*np.sin(.41*t)+50*np.cos(1.1*t),np.full(n,3.),(t-n/2)**2])
        trial={'error':.1,'span':32,'phases':True,'bits':8,'prevalidate':True}
        rows=geo.run_frames(binary,x,trial,zoom=4)
        for source,r in zip(x,rows):
            dense=geo.predict(r,np.arange(n*4)/4)
            np.testing.assert_allclose(geo.as_complex(r['rotations']),np.fft.rfft(dense)[:n//2+1]/4,atol=1e-6)
            self.assertLessEqual(r['max_error'],.1)
        damaged=copy.deepcopy(rows[0]);damaged['rotations'][2][0]+=100
        with self.assertRaises(ValueError):geo.verify(x[0],damaged,trial,4)
        # Sparse measurements do not inherit a held-out prediction guarantee.
        sparse=x[:1,::4]
        sparse_trial={**trial,'error':.01}
        prediction=geo.run_frames(binary,sparse,sparse_trial,zoom=4)[0]
        heldout=np.arange(n)%4!=0
        self.assertGreater(np.max(abs(np.asarray(prediction['dense'])[heldout]-x[0,heldout])),.01)

if __name__=='__main__':unittest.main()
