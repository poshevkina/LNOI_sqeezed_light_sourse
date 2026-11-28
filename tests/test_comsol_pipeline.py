"""Exercise COMSOL export integrity and resume without a running solver."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Event, current_thread
import unittest
from unittest.mock import patch

import numpy as np

from linbo3.__main__ import main
from linbo3.artifacts import file_hash, file_lock, read_json, write_json
from linbo3.comsol import MATERIAL_TABLES
from linbo3.pipeline import run
from linbo3.saved_modes import inspect_modes, load_pair
from test_scientific_workflow import modal_fixture


class ComsolPipeline(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        (self.root / "model.mph").write_bytes(b"model")
        for table in MATERIAL_TABLES.values():
            (self.root / table).write_text("material", encoding="utf-8")
        self.config = self.write_config("one.json", widths=[0.5])
        self.solver = self.enterContext(patch("linbo3.comsol_models.single.model.solve_modal_fields", side_effect=self.solve))

    def write_config(self, name, *, widths):
        path = self.root / name
        write_json(path, dict(
            model="model.mph", output_dir="results", widths_um=widths, geometry={},
            bands={
                "PUMP": dict(start_um=.7, stop_um=.85, points=9),
                "IS": dict(start_um=1.4, stop_um=1.7, points=9),
            },
            modes=None, cores=2, field_grid=[2,2], pump_wavelength_um=.775,
        ))
        return path

    @staticmethod
    def solve(_model, wavelengths, **kwargs):
        kwargs["log"].write_text("complete", encoding="utf-8")
        return modal_fixture("PUMP" if wavelengths[0] < 1 else "IS")

    def test_default_cli_matches_explicit_comsol_and_rejects_all(self):
        outputs = []
        for flags in ([], ["--stage", "comsol"]):
            with self.subTest(flags=flags), patch.object(sys, "argv", ["linbo3", "--config", str(self.config), *flags]):
                with redirect_stdout(StringIO()) as output:
                    main()
                outputs.append(next(line for line in output.getvalue().splitlines() if line.startswith("RESULT_DIR=")))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(self.solver.call_count, 2)
        with patch.object(sys, "argv", ["linbo3", "--config", str(self.config), "--stage", "all"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as failure:
                main()
        self.assertEqual(failure.exception.code, 2)
        self.assertEqual(self.solver.call_count, 2)

    def test_corrupted_field_is_diagnosed_and_only_its_pair_is_repaired(self):
        run_dir = run(self.config)
        width_dir = run_dir / "w_0.5"
        reference = read_json(width_dir / "PUMP_field.json")
        field = width_dir / reference["npz"]
        with np.load(field) as values:
            arrays = {name: values[name] for name in values.files}
        arrays["field"] = arrays["field"] * 2
        np.savez_compressed(field, **arrays)
        self.assertNotEqual(file_hash(field), reference["field_sha256"])
        _, is_fields = load_pair(width_dir, "IS", read_json(run_dir / "config.json")["wavelengths_um"]["IS"])
        self.assertEqual(len(is_fields.neff), 27)
        with patch("linbo3.interpolation_plots.plot_raw"):
            inspection = inspect_modes(run_dir, output_dir=self.root / "inspection")
        pump = inspection.pairs.loc[inspection.pairs.band.eq("PUMP")].iloc[0]
        self.assertEqual(pump.width_um, .5)
        self.assertEqual(pump.analysis_state, "invalid_export")
        self.assertIn("SHA-256", pump.message)
        self.assertEqual(self.solver.call_count, 2)
        self.assertEqual(run(self.config), run_dir)
        self.assertEqual(self.solver.call_count, 3)
        self.assertEqual(file_hash(field), read_json(width_dir / "PUMP_field.json")["field_sha256"])

    def test_resume_rebuilds_csv_from_committed_map(self):
        with patch("linbo3.comsol_models.cache._save_raw_csv", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run(self.config)
        run_dir = next((self.root / "results" / "comsol").iterdir())
        self.assertEqual(read_json(run_dir / "status.json")["state"], "interrupted")
        self.assertFalse((run_dir / "w_0.5" / "PUMP_raw.csv").exists())
        self.assertEqual(self.solver.call_count, 1)
        self.assertEqual(run(self.config), run_dir)
        self.assertTrue((run_dir / "w_0.5" / "PUMP_raw.csv").is_file())
        self.assertEqual(read_json(run_dir / "status.json")["state"], "complete")
        self.assertEqual(self.solver.call_count, 2)
        self.assertEqual(self.solver.call_args_list[-1].args[1][0], 1.4)

    def test_two_runs_share_map_and_identical_run_is_rejected(self):
        other = self.write_config("two.json", widths=[.5,.7])
        started, release, waiting = Event(), Event(), Event()

        def solve(model, wavelengths, **kwargs):
            if kwargs["width_um"] == .5 and wavelengths[0] < 1:
                started.set()
                if not release.wait(10):
                    raise TimeoutError("Test did not release the solver")
            return self.solve(model, wavelengths, **kwargs)

        @contextmanager
        def observe_lock(path, **kwargs):
            if "other-run" in current_thread().name and path.name == ".field.lock" and "w0.5_PUMP" in path.parent.name:
                waiting.set()
            with file_lock(path, **kwargs):
                yield

        self.solver.side_effect = solve
        with (
            patch("linbo3.comsol_jobs.file_lock", side_effect=observe_lock),
            patch("linbo3.comsol_models.cache.file_lock", side_effect=observe_lock),
            ThreadPoolExecutor(1) as first_pool,
            ThreadPoolExecutor(1, thread_name_prefix="other-run") as other_pool,
        ):
            first = first_pool.submit(run, self.config)
            try:
                self.assertTrue(started.wait(10))
                with self.assertRaisesRegex(RuntimeError, "уже запущен"):
                    run(self.config)
                second = other_pool.submit(run, other)
                self.assertTrue(waiting.wait(10))
                self.assertEqual(self.solver.call_count, 1)
            finally:
                release.set()
            self.assertNotEqual(first.result(timeout=10), second.result(timeout=10))
        self.assertEqual(self.solver.call_count, 4)
