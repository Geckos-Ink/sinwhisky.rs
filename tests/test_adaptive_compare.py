"""Independent arbitrary-frequency and exported-circle bridge checks."""
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
    import adaptive_compare as adaptive

@unittest.skipIf(np is None,'NumPy, Matplotlib, Pillow required')
class AdaptiveCompareTests(unittest.TestCase):
    def test_direct_hann_queries_and_fft_padding_units(self):
        n=64;q=4;omega=.473829;t=np.arange(n*q)/q;x=np.sin(omega*t)
        queries=adaptive.direct_queries(x,q,[omega,0.,np.pi])
        z=np.fft.rfft(x*adaptive.periodic_hann(len(x)))/(len(x)/2)
        bins=np.arange(len(z))*2*np.pi/n
        np.testing.assert_allclose(adaptive.direct_queries(x,q,bins[[0,1,8,32]]),z[[0,1,8,32]],atol=1e-12)
        self.assertGreater(abs(queries[0]),.49)
        f,s,idx=adaptive.fft_peaks(x,q,8000)
        self.assertLess(abs(f[idx[np.argmax(abs(s[idx]))]]-omega*8000/(2*np.pi)),2.)

    def test_release_models_peaks_precision_and_validation(self):
        binary=ROOT/'target/release/examples/adaptive_bench'
        if not binary.is_file():self.skipTest('build adaptive_bench release example')
        n=128;t=np.arange(n);x=np.array([100*np.sin(.473829*t)+20*np.cos(1.11361*t),np.full(n,3.)])
        rows=adaptive.run_frames(binary,x,error=.001,span=63,zoom=4,repeats=1,coarse_error=.03,budget=6000)
        for source,row in zip(x,rows):
            self.assertLessEqual(row['max_error'],.001)
            self.assertGreater(row['_verification']['checked_frequencies'],30)
            np.testing.assert_allclose(adaptive.predict(row,np.arange(n*4)/4),row['dense'],atol=1e-7)
        self.assertLess(abs(max(rows[0]['peaks'],key=lambda p:abs(complex(*p['value'])))['omega']-.473829),.001)
        damaged=copy.deepcopy(rows[0]);damaged['check'][1][1]+=10
        with self.assertRaises(ValueError):adaptive.verify(x[0],damaged,.001,4,6000)
        damaged=copy.deepcopy(rows[0]);damaged['queries']['fine']=6001
        with self.assertRaises(ValueError):adaptive.verify(x[0],damaged,.001,4,6000)

    def test_small_budget_still_returns_reviewable_partial_search(self):
        binary=ROOT/'target/release/examples/adaptive_bench'
        if not binary.is_file():self.skipTest('build adaptive_bench release example')
        x=np.sin(np.arange(64)*.49)[None,:]
        r=adaptive.run_frames(binary,x,error=.001,budget=7,repeats=1)[0]
        self.assertTrue(r['queries']['budget_exhausted'])
        self.assertGreater(r['stops']['budget'],0)
        self.assertLessEqual(r['queries']['coarse']+r['queries']['fine'],7)

if __name__=='__main__':unittest.main()
