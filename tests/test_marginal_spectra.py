import unittest
import numpy as np
from numpy.testing import assert_allclose
from linbo3.frequency import LIGHT_SPEED_M_PER_FS
from linbo3.marginal_spectra import marginal_densities, peak_normalize, fwhm


class MarginalSpectra(unittest.TestCase):
    def test_separable_jsi_signal_idler_axes_and_gaussian_widths(self):
        omega=np.linspace(1.19,1.21,1201)
        signal=np.exp(-.5*((omega-1.201)/.0004)**2)
        idler=np.exp(-.5*((omega-1.198)/.0007)**2)
        result=marginal_densities(omega,np.outer(idler,signal))
        assert_allclose(peak_normalize(result['signal_omega']),signal/signal.max(),atol=1e-14)
        assert_allclose(peak_normalize(result['idler_omega']),idler/idler.max(),atol=1e-14)
        factor=2*np.sqrt(2*np.log(2))
        self.assertAlmostEqual(fwhm(omega,result['signal_omega']).width/(factor*.0004),1,places=3)
        self.assertAlmostEqual(fwhm(omega,result['idler_omega']).width/(factor*.0007),1,places=3)

    def test_wavelength_jacobian_not_just_relabelled_axis(self):
        omega=np.linspace(1.,2.,101)
        result=marginal_densities(omega,np.ones((101,101)))
        wavelength=result['wavelength_nm']
        self.assertTrue(np.all(np.diff(wavelength)>0))
        expected=(wavelength[0]/wavelength)**2
        assert_allclose(peak_normalize(result['signal_wavelength']),expected)
        assert_allclose(peak_normalize(result['signal_omega']),1)

    def test_density_integral_is_preserved_after_jacobian(self):
        omega=np.linspace(.7,1.7,2001)
        signal=np.exp(-.5*((omega-1.2)/.07)**2)
        result=marginal_densities(omega,np.outer(signal,signal))
        def integral(x,y):return np.sum(np.diff(x)*(y[:-1]+y[1:])/2)
        a=integral(omega,result['signal_omega'])
        b=integral(result['wavelength_nm'],result['signal_wavelength'])
        self.assertLess(abs(a-b)/a,1e-6)

    def test_clipped_and_exact_halfmax_boundaries(self):
        clipped=fwhm(np.arange(5.),np.ones(5))
        self.assertTrue(np.isnan(clipped.width))
        self.assertEqual(clipped.status,'both_edges_clipped')
        exact=fwhm(np.array([0.,1.,2.]),np.array([.5,1.,.5]))
        self.assertEqual(exact.width,2)
        self.assertEqual(exact.status,'ok')

    def test_multiple_peaks_report_main_connected_fwhm(self):
        x=np.linspace(-2,5,4001)
        y=np.exp(-.5*(x/.2)**2)+.8*np.exp(-.5*((x-3)/.2)**2)
        width=fwhm(x,y)
        self.assertEqual(width.intervals,2)
        self.assertEqual(width.status,'multiple_halfmax_intervals')
        self.assertLess(width.width,.5)

    def test_nonuniform_grid_and_empty_spectrum_are_rejected(self):
        with self.assertRaises(ValueError):marginal_densities([1,1.1,1.5],np.ones((3,3)))
        with self.assertRaises(ValueError):fwhm([1,2,3],[0,0,0])


if __name__=='__main__':unittest.main()
