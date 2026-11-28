"""Regression checks for coordinate and vector-reflection conventions."""

from pathlib import Path
import unittest

import numpy as np

from linbo3.comsol_models.coupled.analysis import _describe, select_pair, subspace_overlap
from linbo3.mode_classification import CrossSection
from linbo3.modes import ModalSolutions


class CoupledAnalysisTests(unittest.TestCase):
    def test_modes_use_xy_coordinates_and_vector_parity(self):
        x = np.linspace(-2., 2., 81)
        y = np.linspace(-.3, 1., 66)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        left = np.exp(-((xx + .65) / .14)**2 - ((yy - .45) / .1)**2)
        right = np.exp(-((xx - .65) / .14)**2 - ((yy - .45) / .1)**2)
        coordinates = np.stack([xx, yy]).reshape(2, -1) * 1e-6
        order = np.random.default_rng(7).permutation(coordinates.shape[1])
        geometry = CrossSection(.6, .3, 62.5)
        for axis, family in enumerate(("quasi-TE", "quasi-TM")):
            field = np.zeros((2, 3, coordinates.shape[1]), complex)
            field[:, axis] = np.stack([left + right, 1j * (left - right)]).reshape(2, -1)
            raw = ModalSolutions(np.array([1.55, 1.55]), np.array([0, 0]), np.array([18, 33]),
                                 np.array([1.8 + 1e-8j, 1.79]), field[:, :, order],
                                 coordinates[:, order])
            a, b = [_describe(raw, i, .8, 2.15, Path("unused"), geometry, .5) for i in range(2)]
            np.testing.assert_allclose(a.profile.x_um, x)
            np.testing.assert_allclose(a.profile.y_um, y)
            self.assertAlmostEqual(a.component_phase, 1.)
            self.assertAlmostEqual(b.component_phase, -1.)
            self.assertAlmostEqual(a.parity, (-1., 1.)[axis])
            self.assertAlmostEqual(b.parity, (1., -1.)[axis])
            pair, status = select_pair([a, b], family)
            self.assertEqual(status, "paired")
            self.assertEqual([item.solution for item in pair], ([33, 18], [18, 33])[axis])

    def test_mode_subspace_survives_basis_rotation(self):
        a = np.array([1, 0, 0], complex)
        b = np.array([0, 1j, 0], complex)
        self.assertAlmostEqual(subspace_overlap([a, b], [a+b, a-b]), 1.)
        self.assertAlmostEqual(subspace_overlap([a, b], [a, np.array([0, 0, 1])]), 0.)
