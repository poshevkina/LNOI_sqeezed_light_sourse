"""Python model API and its shared disk cache, with COMSOL mocked."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from linbo3.comsol_models import CoupledWaveguides2D
from linbo3.modes import ModalSolutions


class CoupledModelCache(unittest.TestCase):
    def test_read_only_plan_incremental_solve_and_lazy_fields(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "test.mph"
            model_path.write_bytes(b"mph")
            (root / "n_SiO2.txt").write_text("material", encoding="utf-8")
            model = CoupledWaveguides2D(model_path, results_root=root / "results")
            other_name = dict(width_um=.5, wavelength_um=1.55, mode_count=2)
            raw = ModalSolutions(np.array([1.55, 1.55]), np.array([1, 1]), np.array([1, 2]),
                                 np.array([2+0j, 2.1+0j]), np.ones((2, 3, 4), dtype=complex),
                                 np.zeros((2, 4)))
            with patch.object(model, "_solve", return_value=raw) as solve:
                with redirect_stdout(StringIO()) as info:
                    draft = model.modes(gaps_um=[.8], **other_name, dry_run=True)
                self.assertIn("нет проверенной карты", info.getvalue())
                self.assertFalse(draft.run_dir.exists())
                self.assertFalse((root / "field_maps").exists())
                with self.assertRaisesRegex(RuntimeError, r"g=0\.8 .*--solve"):
                    model.modes(gaps_um=[.8], **other_name)
                solve.assert_not_called()

                first = model.modes(gaps_um=[.8], **other_name, solve=True)
                self.assertEqual(solve.call_count, 1)
                self.assertEqual(solve.call_args.args[1], .8)
                with redirect_stdout(StringIO()) as info:
                    model.modes(gaps_um=[.8, .9], **other_name, dry_run=True)
                self.assertIn("g=0.8", info.getvalue())
                self.assertIn("| кэш", info.getvalue())
                self.assertIn("нет проверенной карты", info.getvalue())

                extended = model.modes(gaps_um=[.8, .9], **other_name, solve=True)
                self.assertEqual(solve.call_count, 2)
                self.assertEqual(solve.call_args.args[1], .9)
                self.assertNotEqual(first.run_dir, extended.run_dir)
                self.assertEqual(model.modes(gaps_um=[.9, .8], **other_name).run_dir, extended.run_dir)
                self.assertEqual(solve.call_count, 2)
                with self.assertRaisesRegex(RuntimeError, "нет|Нет"):
                    model.modes(gaps_um=[.8], **other_name, neff_shift=1.8)
                shifted = model.modes(gaps_um=[.8], **other_name, neff_shift=1.8, solve=True)
                self.assertNotEqual(shifted.run_dir, first.run_dir)
                self.assertEqual(solve.call_count, 3)
                self.assertEqual(solve.call_args.args[4], 1.8)
            with patch("linbo3.comsol_models.cache.load_modal_npz", side_effect=AssertionError("Unexpected field read")):
                table = extended.load_modes()
            self.assertEqual(table.gap_um.tolist(), [.8, .8, .9, .9])
            np.testing.assert_array_equal(extended.load_fields("g_0.9").field, raw.field)
