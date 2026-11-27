"""Compact SPDC storage retains exact projections and remains atomic."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from linbo3.artifacts import write_json
from linbo3.case_cache import cache_case
from linbo3.compact_cases import compact_case, remove_full_jsa
from linbo3.dispersion import Mode
from linbo3.marginal_spectra import _densities_from_marginals, fwhm, marginal_densities
from linbo3.pump_hybridization import contribution_interval
from linbo3.spectral_projections import saved_sum_frequency
from linbo3.spdc import calculate_spectrum


class CompactCases(unittest.TestCase):
    def test_convert_legacy_without_changing_spdc_results(self):
        points = 131
        wl = np.linspace(.7, .85, 11)
        spectrum = calculate_spectrum(Mode(wl, 2 + .05 * wl), Mode(2*wl, 2.2 + .03*wl),
                                      Mode(2*wl, 2.1 + .01*wl), points=points)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "case.npz"
            full = root / "jsa_full" / "full.npz"
            signature = dict(case_key="fixture", points=points)
            public, _ = cache_case(path, full, signature, lambda: spectrum)
            legacy = {key: value for key, value in public.items()
                      if key not in {"storage_version", "omega_sum", "sum_frequency_mass"}}
            np.savez_compressed(path, **legacy, IMatrix=spectrum.jsi)
            original = path.read_bytes()
            omega = spectrum.angular_frequency_rad_per_fs
            expected = marginal_densities(omega, spectrum.jsi)
            with np.load(path) as file:
                interval = contribution_interval(*saved_sum_frequency(file))

            write_json(full.with_name("case.json"), signature)
            np.savez_compressed(full, jsa=spectrum.jsa, omega_signal_rad_per_fs=omega,
                                omega_idler_rad_per_fs=omega)
            with self.assertRaises(ValueError):
                remove_full_jsa(path, full, signature)
            self.assertTrue(full.exists())
            with patch.object(Path, "replace", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                compact_case(path, points)
            self.assertEqual(path.read_bytes(), original)

            self.assertTrue(compact_case(path, points))
            self.assertFalse(compact_case(path, points))
            with np.load(path) as file:
                self.assertNotIn("IMatrix", file)
                self.assertEqual(file["K"], spectrum.schmidt_number)
                np.testing.assert_array_equal(file["schmidt_weights"], spectrum.schmidt_weights)
                np.testing.assert_array_equal(file["jsa_preview"], legacy["jsa_preview"])
                np.testing.assert_array_equal(file["plot_jsi"], legacy["plot_jsi"])
                self.assertEqual(contribution_interval(*saved_sum_frequency(file)), interval)
                direct = _densities_from_marginals(file["w_grid"], file["S_s"], file["S_i"])
                for key, exact in expected.items():
                    np.testing.assert_array_equal(direct[key], exact)
                self.assertEqual(fwhm(direct["omega"], direct["signal_omega"]),
                                 fwhm(expected["omega"], expected["signal_omega"]))
            stored, hit = cache_case(path, full, signature, lambda: self.fail("Unexpected SVD"))
            self.assertTrue(hit)
            self.assertEqual(stored["K"], spectrum.schmidt_number)
            self.assertTrue(full.exists())
            self.assertGreater(remove_full_jsa(path, full, signature), 0)
            self.assertFalse(full.exists())
