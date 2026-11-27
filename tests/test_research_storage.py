"""Research names isolate reports without duplicating scientific calculations."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from linbo3.artifacts import read_json
from linbo3.case_cache import cache_case
from linbo3.dispersion import Mode
from linbo3.study_output import publish_study
from linbo3.interpolation_study import run_study
from linbo3.spdc import calculate_spectrum
from test_scientific_workflow import write_saved_run


class ResearchStorage(unittest.TestCase):
    def test_simultaneous_studies_share_one_case_creation(self):
        with TemporaryDirectory() as tmp, ThreadPoolExecutor(2) as pool:
            base = Path(tmp)
            full = base / "jsa_full" / "full.npz"
            public = base / "cases" / "case.npz"
            signature = dict(case_key="example", points=17)
            wavelengths = np.linspace(.65, .9, 9)
            pump = Mode(wavelengths, 2.1 + .01 * wavelengths)
            signal = Mode(2 * wavelengths, 2.2 + .02 * wavelengths)
            idler = Mode(2 * wavelengths, 2.3 + .03 * wavelengths)
            started, second_started = Event(), Event()

            def compute():
                started.set()
                self.assertTrue(second_started.wait(5))
                return calculate_spectrum(pump, signal, idler, points=17)

            def second():
                second_started.set()
                return cache_case(public, full, signature, lambda: self.fail("Duplicate numerical case"))

            first = pool.submit(cache_case, public, full, signature, compute)
            self.assertTrue(started.wait(5))
            other = pool.submit(second)
            self.assertFalse(first.result(timeout=10)[1])
            self.assertTrue(other.result(timeout=10)[1])

    def test_publication_restores_previous_report_if_rename_fails(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "my-study"
            output.mkdir()
            original = output / "summary.csv"
            original.write_text("previous", encoding="utf-8")
            rename = Path.rename

            def interrupt(path, target):
                if path.name == ".work-my-study":
                    raise KeyboardInterrupt
                return rename(path, target)

            with patch.object(Path, "rename", interrupt), self.assertRaises(KeyboardInterrupt):
                with publish_study(output) as staging:
                    (staging / "summary.csv").write_text("updated", encoding="utf-8")
            self.assertEqual(original.read_text(encoding="utf-8"), "previous")
            self.assertFalse((root / ".previous-my-study").exists())


    def test_named_studies_share_cases_and_resume(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            results = base / "results"
            run = write_saved_run(results / "comsol" / "fixture", widths=(.5,))
            with (
                patch("linbo3.interpolation_plots.plot_raw"),
                patch("linbo3.interpolation_plots.plot_fits"),
                patch("linbo3.interpolation_plots.plot_derivatives"),
                patch("linbo3.interpolation_plots.plot_jsi_comparison"),
                patch("linbo3.interpolation_study.calculate_spectrum", wraps=calculate_spectrum) as solver,
            ):
                first = run_study(run, study_name="fit-one", methods=("poly1",), points=131)
                self.assertEqual(solver.call_count, 2)
                second = run_study(run, study_name="fit-two", methods=("poly1", "poly2"), points=131)
                self.assertEqual(solver.call_count, 4)
                self.assertNotEqual(first.output_dir, second.output_dir)
                self.assertEqual(set(first.summary.case_key),
                                 set(second.summary.loc[second.summary.method == "poly1", "case_key"]))

                self.assertEqual(first.output_dir, results / "studies" / "fit-one")
                previous_summary = (first.output_dir / "summary.csv").read_bytes()
                previous_settings = (first.output_dir / "settings.json").read_bytes()
                stale_figure = first.output_dir / "stale.png"
                stale_figure.touch()

                def stop_second(*args, **kwargs):
                    if stop_second.calls:
                        self.assertEqual((first.output_dir / "summary.csv").read_bytes(), previous_summary)
                        raise KeyboardInterrupt
                    stop_second.calls += 1
                    return calculate_spectrum(*args, **kwargs)

                stop_second.calls = 0
                solver.side_effect = stop_second
                with self.assertRaises(KeyboardInterrupt):
                    run_study(run, study_name="fit-one", methods=("poly3",), points=31)
                self.assertEqual((first.output_dir / "settings.json").read_bytes(), previous_settings)
                self.assertTrue(stale_figure.exists())
                self.assertFalse((results / "studies" / ".work-fit-one").exists())
                solver.reset_mock(side_effect=True)
                resumed = run_study(run, study_name="fit-one", methods=("poly3",), points=31)
                self.assertEqual(solver.call_count, 1)
                self.assertEqual(resumed.output_dir, first.output_dir)
                self.assertEqual(len(resumed.summary), 2)
                self.assertFalse(stale_figure.exists())
                self.assertEqual(read_json(resumed.output_dir / "settings.json")["methods"], ["poly3"])

            first_row = first.summary.iloc[0]
            case_file = (first.output_dir / first_row.npz).resolve()
            source = base / "jsa_full" / case_file.parent.name / "full.npz"
            self.assertFalse(source.exists())
            with np.load(case_file) as case:
                self.assertEqual(case["jsa_preview"].shape, (129, 129))
                self.assertEqual(case["jsa_preview"].dtype, np.dtype("complex64"))
                np.testing.assert_allclose(case["plot_jsi"], np.abs(case["jsa_preview"].astype(np.complex128))**2, rtol=2e-7)
                np.testing.assert_allclose(case["K"], 1 / np.sum(case["schmidt_weights"]**2))
                np.testing.assert_allclose(case["sum_frequency_mass"].sum(), 1)
                self.assertNotIn("IMatrix", case.files)
                self.assertNotIn("fMatrix", case.files)

            # A regular checkout can render verification figures without local JSA or field maps.
            clone = base / "clone" / "results"
            shutil.copytree(results / "studies", clone / "studies")
            shutil.copytree(results / "cases", clone / "cases")
            relative_report = second.output_dir.relative_to(results)
            copied_report = clone / relative_report
            saved = pd.read_csv(copied_report / "summary.csv").iloc[0]
            with np.load(copied_report / saved.npz) as public:
                self.assertEqual(public["sum_frequency_mass"].shape, (261,))
                self.assertEqual(public["jsa_preview"].shape, (129, 129))
            self.assertFalse((clone.parent / "jsa_full").exists())
