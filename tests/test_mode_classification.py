"""Numerical classification checks with analytic complex Hermite-Gaussian profiles."""

import unittest
import numpy as np
from numpy.polynomial.hermite import hermval
from numpy.testing import assert_allclose

from linbo3.mode_classification import CrossSection, classify_field, field_grid


class SpatialClassification(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(-2.4, 2.4, 321)
        self.y = np.linspace(-.3, 1.3, 81)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.coordinates = np.array([self.xx.ravel(), self.yy.ravel()])*1e-6
        self.geometry = CrossSection(1., .3, 80.)

    def profile(self, order):
        u = self.xx/.45
        return hermval(u, [0.]*order+[1.])*np.exp(-u**2/2)*np.exp(-((self.yy-.5)/.16)**2/2)

    def classify(self, field):
        return classify_field(self.coordinates, field.reshape(3, -1), 4., self.geometry)[0]

    def test_both_polarizations_and_orders_zero_to_two(self):
        for component, family in enumerate(("TE", "TM")):
            for order in range(3):
                field = np.zeros((3, *self.xx.shape), dtype=complex)
                field[component] = self.profile(order)*np.exp(.83j)
                field[2] = .1j*field[component]
                result = self.classify(field)
                self.assertEqual(result["label"], f"{family}{order}")
                self.assertEqual(result["lateral_peaks"], order+1)
                self.assertEqual(result["status"], "confident")

    def test_ten_percent_hybrid_boundary(self):
        for fraction, hybrid in ((.89, True), (.90, False), (.91, False)):
            field = np.zeros((3, *self.xx.shape), dtype=complex)
            field[0] = np.sqrt(fraction)*self.profile(0)
            field[1] = np.sqrt(1-fraction)*self.profile(0)
            result = self.classify(field)
            self.assertEqual(result["label"].startswith("Hybrid"), hybrid)

    def test_global_phase_and_amplitude_do_not_change_classification(self):
        field = np.zeros((3, *self.xx.shape), dtype=complex)
        field[0] = self.profile(2)
        original = self.classify(field)
        scaled = self.classify(field*7.3*np.exp(1.78j))
        self.assertEqual(original["label"], scaled["label"])
        assert_allclose(original["te_fraction"], scaled["te_fraction"])

    def test_hybrid_is_not_forced_to_pure_te_tm(self):
        field = np.zeros((3, *self.xx.shape), dtype=complex)
        for c, n in ((0, 0), (1, 1)):
            profile = self.profile(n)
            field[c] = profile/np.linalg.norm(profile)*np.exp(c*.7j)
        result = self.classify(field)
        self.assertTrue(result["label"].startswith("Hybrid "))
        self.assertIn("TE0", result["label"])
        self.assertIn("TM1", result["label"])
        self.assertEqual(result["status"], "review")
        self.assertAlmostEqual(result["te_fraction"], .5, places=4)

    def test_longitudinal_field_not_mistaken_for_polarization(self):
        field = np.zeros((3, *self.xx.shape), dtype=complex)
        field[0] = self.profile(0)
        field[2] = 100j*self.profile(0)
        result = self.classify(field)
        self.assertIn("longitudinal_dominated", result["reason"])
        self.assertEqual(result["status"], "review")

    def test_permuting_samples_keeps_coordinates_and_fields_paired(self):
        field = np.zeros((3, self.coordinates.shape[1]), dtype=complex)
        field[1] = self.profile(1).ravel()
        order = np.random.default_rng(7).permutation(field.shape[1])
        result, _ = classify_field(self.coordinates[:, order], field[:, order], 4., self.geometry)
        self.assertEqual(result["label"], "TM1")

    def test_zero_field_and_repeated_coordinates_are_rejected(self):
        field = np.zeros((3, self.coordinates.shape[1]), dtype=complex)
        with self.assertRaises(ValueError):
            self.classify(field.reshape(3, *self.xx.shape))
        with self.assertRaises(ValueError):
            field_grid(np.zeros((2, 8)), np.ones((3, 8)))


if __name__ == "__main__":
    unittest.main()
