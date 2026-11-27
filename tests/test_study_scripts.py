"""Small end-to-end examples for all four Python studies."""

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from linbo3.artifacts import read_json, write_json
from linbo3.spdc import calculate_spectrum
from studies import fit_effect, pulse_series, spatial_single, unclassified
from test_scientific_workflow import write_saved_run


def write_spatial_fields(source):
    model = source / "fake-model.mph"
    with ZipFile(model, "w") as fixture:
        fixture.writestr("dmodel.xml", """
            <model><expressions name="h_core" expr="0.5[um]"/>
            <expressions name="h_film" expr="0.1[um]"/>
            <expressions name="theta" expr="70[deg]"/></model>
        """)
    manifest = read_json(source / "config.json")
    write_json(source / "config.json", dict(manifest, model=model.name, model_path_base="run_dir", run_id="fixture"))
    xx, yy = np.meshgrid(np.linspace(-.65, .65, 41), np.linspace(-.1, .65, 31))
    profile = np.exp(-((xx / .13)**2 + ((yy - .25) / .14)**2)).ravel()
    for path in source.glob("w_*/*_raw.npz"):
        with np.load(path) as saved:
            data = {key: saved[key] for key in saved.files}
        data["coordinates"] = np.array([xx.ravel(), yy.ravel()]) * 1e-6
        data["field"] = data["field"][:, 0, :3, None] * profile[None, None, :]
        np.savez_compressed(path, **data)


class PythonStudies(unittest.TestCase):
    def test_saved_comsol_to_reports_and_shared_spdc_cases(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = write_saved_run(base / "exports", widths=(.5,))
            write_spatial_fields(source)
            results = base / "results"
            fit = fit_effect.run(source, results, points=18)
            with patch("linbo3.sweep.calculate_spectrum", wraps=calculate_spectrum) as solver:
                numeric = unclassified.run(source, results, points=18)
                self.assertEqual(solver.call_count, 0)  # poly6 cases already exist from fit-effect
                spatial = spatial_single.run(source, results, points=18)
                self.assertEqual(solver.call_count, 2)
                series = pulse_series.run(source, results, points=18)
                self.assertEqual(solver.call_count, 16)  # 2000 fs reused, seven new durations × 2 pumps

            for name, report in zip(("fit-effect", "unclassified", "spatial-single", "pulse-series"),
                                    (fit, numeric, spatial, series), strict=True):
                self.assertEqual(report, results / "studies" / name)
                self.assertEqual(read_json(report / "settings.json")["comsol_run"], "fixture")
                summary = pd.read_csv(report / "summary.csv")
                self.assertFalse(summary.empty)
                for row in summary.itertuples():
                    case = (report / row.npz).resolve()
                    self.assertEqual(case.parent.parent, results / "cases")
                    self.assertEqual(read_json(case.with_name("case.json"))["case_key"], row.case_key)
                    self.assertTrue(case.is_file())

            self.assertEqual(pd.read_csv(fit / "summary.csv").method.nunique(), 6)
            classified = pd.read_csv(spatial / "summary.csv")
            self.assertEqual(set(classified.signal_label), {"TM0"})
            self.assertEqual(set(classified.idler_label), {"TE0"})
            summary = pd.read_csv(series / "summary.csv")
            self.assertEqual(set(summary.t_fs), set(pulse_series.WINDOWS))
            self.assertEqual(set(summary.loc[summary.t_fs.eq(2000), "case_key"]), set(classified.case_key))
            for folder, filenames in (
                (fit, ("k_by_fit.png", "k_by_width.png", "fit_errors.csv")),
                (numeric, ("figures/jsi_overview.png", "k_by_width.png", "branches.csv")),
                (spatial, ("marginal_fwhm.csv", "figures/marginals_omega.png", "pump_hybridization_cases.csv")),
                (series, ("K_vs_duration.png", "K_overview.png", "t_250fs/figures/jsi_overview.png")),
            ):
                for name in filenames:
                    self.assertTrue((folder / name).is_file(), folder / name)
