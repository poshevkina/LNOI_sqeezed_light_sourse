"""Check normalizations and the sign of power in the 3D workflow."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
from scipy.linalg import eigh

from linbo3.artifacts import file_hash, save_arrays, write_json
from linbo3.comsol_models.coupler_3d.cache import CouplerCase
from linbo3.comsol_models.coupler_3d.inputs import flux_basis
from linbo3.comsol_models.coupler_3d.solver import _extract
from studies.coupler_3d import case_metrics


class CouplerPowerTests(unittest.TestCase):
    def test_input_is_a_coherent_combination(self):
        x = np.linspace(-2e-6, 2e-6, 160)
        y = np.linspace(-1e-6, 1e-6, 81)
        xx, yy = np.meshgrid(x, y)
        left = np.exp(-((xx + .65e-6)/.15e-6)**2 - (yy/.2e-6)**2)
        right = left[:, ::-1]
        electric = np.zeros((2, 3, len(y), len(x)), dtype=complex)
        electric[0, 0] = left+right
        electric[1, 0] = (left-right) * np.exp(.4j)
        _, flux = flux_basis(electric, np.array([7e6, 7e6]), x, y, 1.55)
        np.testing.assert_allclose(flux.sum(axis=0), np.eye(2), atol=1e-12)
        values, weights = eigh(flux[1], flux.sum(axis=0))
        self.assertLess(values[0], 1e-8)
        self.assertGreater(values[1], 1-1e-8)
        self.assertAlmostEqual(np.angle(weights[0, 0]/weights[1, 0]), .4)

    def test_metrics_use_net_input_and_output(self):
        info = dict(z_um=np.array([0., .1, 200.]), power_w=np.array([[1.,0.],[.97,.03],[.2,.78]]),
                    dof=100, elements=20, elapsed_s=10.)
        metrics = case_metrics(info)
        self.assertAlmostEqual(metrics["left_in"], .97)
        self.assertAlmostEqual(metrics["cross_out"], .78/.98)
        self.assertAlmostEqual(metrics["power_drift"], .02)
        self.assertEqual(metrics["dof"], 100)


    def test_sampled_volume_keeps_axis_order_and_signed_power(self):
        interp = Mock()
        def sample():
            x, y, z = np.asarray(interp.setInterpolationCoordinates.call_args.args[0])
            return np.stack((1 + z, -np.where(x < 0, 2., 3.)))[:, None, :]
        interp.getData.side_effect = sample
        with patch("linbo3.comsol_models.coupler_3d.solver._data", return_value=interp), patch.multiple(
            "linbo3.comsol_models.coupler_3d.solver",
            FIELD_SHAPE=(4, 3), MAP_SHAPE=(6, 5), SAMPLE_BATCH_POINTS=42,
            AXIAL_UM=np.array([0., 2., 4.]), SLICE_UM=np.array([2.]),
        ):
            data = _extract(None, "sol1", [(-2., 2.), (-1.5, 1.5)], inlet=1., direction=-1)
        self.assertEqual(data["electric_squared_volume"].shape, (3, 5, 6))
        self.assertEqual(data["electric_squared_volume"].dtype, np.dtype("float32"))
        np.testing.assert_allclose(data["electric_squared"], data["electric_squared_volume"][1:2])
        np.testing.assert_allclose(data["electric_squared_volume"][:, 0, 0], [2., 1.999998, 1.999996])
        np.testing.assert_allclose(data["longitudinal_flux_w_m2"][0, 0], [2., 2., 2., 3., 3., 3.])
        np.testing.assert_allclose(data["power_w"], [[12., 18.]] * 3)

    def test_legacy_cache_is_reused_until_explicit_refresh(self):
        with TemporaryDirectory() as directory:
            case = CouplerCase.__new__(CouplerCase)
            case.folder = Path(directory)
            case.template = case.folder / "model.mph"
            case.excitation, case.layers, case.weights = None, 7, None
            case.field = case.folder / "sections.npz"
            case.manifest = case.folder / "settings.json"
            case.settings = {"version": 1}
            save_arrays(case.field, {"power_w": np.array([0.9])})
            write_json(case.manifest, {"version": 1, "sha256": file_hash(case.field)})
            fields = {"power_w": np.array([0.9]), "electric_squared_volume": np.ones((3, 2, 2)),
                      "longitudinal_flux_w_m2": np.ones((3, 2, 2)), "x_um": np.arange(2),
                      "y_um": np.arange(2)}
            with patch("linbo3.comsol_models.coupler_3d.cache.MAP_SHAPE", (2, 2)), patch(
                "linbo3.comsol_models.coupler_3d.cache.solve_case", return_value=fields,
            ) as solve:
                case.run(solve=True)
                solve.assert_not_called()
                self.assertIsNone(case.read_volume())
                case.run(solve=True, refresh_fields=True)
                solve.assert_called_once()
                case.run(solve=True, refresh_fields=True)
                solve.assert_called_once()
            self.assertEqual(case.read_volume()["electric_squared_volume"].shape, (3, 2, 2))
            self.assertEqual(set(case.read()), {"power_w", "x_um", "y_um"})
