"""Numerical regressions for the omega migration and strict mode selection."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose
from numpy.polynomial import Polynomial

from linbo3.dispersion import Mode
from linbo3.frequency import LIGHT_SPEED_M_PER_FS, omega_to_um, um_to_omega
from linbo3.interpolation import METHODS, fit_dispersion, leave_one_out
from linbo3.modes import ModalSolutions, ModeTrackingError, track_modes
from linbo3.polynomial import fit_study
from linbo3.spdc import calculate_spectrum


def modal_fixture(band="PUMP", imag=1e-9):
    anchor = .775 if band == "PUMP" else 1.55
    wavelengths = np.linspace(.7, .85, 9) * (1 if band == "PUMP" else 2)
    wavelengths[4] = anchor
    rows, fields = [], []
    # Physical branches keep their fields while COMSOL solution numbers swap.
    for point, wl in enumerate(wavelengths):
        order = (0, 1, 2) if point % 2 == 0 else (1, 0, 2)
        for solnum, physical in enumerate(order, 1):
            omega = float(um_to_omega(wl))
            neff = (1.7 + .18 * physical + .02 * omega + .004 * omega**2)
            loss = imag if physical < 2 else 1e-6
            rows.append((wl, point + 1, solnum, neff + 1j * loss))
            field = np.zeros((3, 4), dtype=complex)
            field.flat[physical] = np.exp(1j * point * .7)
            fields.append(field)
    return ModalSolutions(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
                          np.array([r[2] for r in rows]), np.array([r[3] for r in rows]),
                          np.array(fields), np.zeros((2, 4)))


def write_saved_run(directory, widths=(.5, .7), incomplete_width=None):
    """Small saved COMSOL-shaped dataset; no COMSOL dependency."""
    import pandas as pd
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = dict(widths_um=list(widths), pump_wavelength_um=.775, length_m=.12,
                    wavelengths_um={}, model_sha256="synthetic-test-data", stage="comsol")
    pairs = {}
    for width in widths:
        key = f"{width:g}"
        target = directory / f"w_{key}"
        target.mkdir(exist_ok=True)
        pairs[key] = {}
        for band in ("PUMP", "IS"):
            raw = modal_fixture(band)
            manifest["wavelengths_um"][band] = np.unique(raw.wavelength_um).tolist()
            if width == incomplete_width and band == "IS":
                pairs[key][band] = {"state": "failed", "error": "synthetic missing pair"}
                continue
            np.savez_compressed(target / f"{band}_raw.npz", **raw.arrays())
            pd.DataFrame(dict(wavelength_um=raw.wavelength_um, outer_solnum=raw.outer_solnum,
                              solnum=raw.solnum, neff_real=raw.neff.real,
                              neff_imag=raw.neff.imag)).to_csv(target / f"{band}_raw.csv", index=False)
            pairs[key][band] = {"state": "complete"}
    (directory / "config.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "status.json").write_text(json.dumps({"state": "complete", "pairs": pairs}), encoding="utf-8")
    return directory


class FrequencyFits(unittest.TestCase):
    def test_polynomial_and_physical_derivatives(self):
        wavelengths = np.linspace(.7, .85, 30)
        omega = um_to_omega(wavelengths)
        for degree in range(1, 7):
            coefficients = np.array([2.] + [.02 / power for power in range(1, degree + 1)])
            truth = Polynomial(coefficients)
            fitted = fit_dispersion(f"poly{degree}", wavelengths, truth(omega))
            for derivative in (0, 1, 2):
                assert_allclose(fitted.deriv(derivative)(omega), truth.deriv(derivative)(omega),
                                rtol=2e-8, atol=2e-8)
        assert_allclose(omega_to_um(omega), wavelengths)

    def test_leave_one_out_is_in_frequency(self):
        wavelength = np.linspace(1.4, 1.7, 12)
        omega = um_to_omega(wavelength)
        records = list(leave_one_out("poly2", wavelength, 2 + .1 * omega + .03 * omega**2))
        self.assertEqual(len(records), 10)
        assert_allclose([row["error"] for row in records], 0, atol=1e-13)
        with self.assertRaises(ValueError):
            list(leave_one_out("poly6", wavelength[:7], np.ones(7)))

    def test_study_fits_and_numbering(self):
        raw = {band: track_modes(modal_fixture(band), band=band,
                                anchor_um=.775 if band == "PUMP" else 1.55,
                                imag_limit=1e-6, min_overlap=.9, min_margin=.1).modes
               for band in ("PUMP", "IS")}
        study = fit_study({.5: raw})
        self.assertEqual(study.degrees, tuple(range(1, 7)))
        self.assertEqual(study.pump_map[.5]["мода 1"], 1)
        for mode in raw["PUMP"]:
            data = raw["PUMP"][mode]
            assert_allclose(study.fits[.5][2]["PUMP"][mode](um_to_omega(data.wavelength_um)),
                            data.neff, atol=1e-13)


class Tracking(unittest.TestCase):
    def track(self, raw, **kwargs):
        return track_modes(raw, band="PUMP", anchor_um=.775, imag_limit=1e-6,
                           min_overlap=.9, min_margin=.1, **kwargs)

    def test_field_identity_survives_solver_renumbering(self):
        result = self.track(modal_fixture())
        self.assertEqual(len(result.modes), 2)
        rows = [r for r in result.rows if r["mode"] == 0]
        self.assertGreater(len({r["solnum"] for r in rows}), 1)
        self.assertTrue(all(r["passes_imag_limit"] for r in result.rows))
        self.assertLess(max(abs(r["neff_imag"]) for r in result.rows), 1e-6)

    def test_strict_limit_and_manual_choice(self):
        self.assertEqual(len(self.track(modal_fixture()).modes), 2)  # equality excluded
        with self.assertRaises(ModeTrackingError):
            self.track(modal_fixture(), anchor_solnums=[3])
        with self.assertRaises(ModeTrackingError):
            self.track(modal_fixture(imag=1e-6))
        manual = self.track(modal_fixture(), anchor_solnums=[2, 1])
        self.assertGreater(manual.modes[0].neff[4], manual.modes[1].neff[4])

    def test_limit_is_enforced_away_from_anchor(self):
        raw = modal_fixture()
        # A low-loss anchor branch becomes lossy at the longest wavelength.
        raw.neff[(raw.wavelength_um == raw.wavelength_um.max()) & (raw.solnum == 1)] += 2e-6j
        with self.assertRaises(ModeTrackingError):
            self.track(raw)

    def test_ambiguous_fields_stop_tracking(self):
        raw = modal_fixture()
        raw.field[:3] = raw.field[0]
        with self.assertRaises(ModeTrackingError):
            self.track(raw)

    def test_is_requires_exactly_two_anchors(self):
        raw = modal_fixture("IS")
        raw.neff.imag[:] = 1e-9
        with self.assertRaises(ModeTrackingError):
            track_modes(raw, band="IS", anchor_um=1.55, imag_limit=1e-6,
                        min_overlap=.9, min_margin=.1)


class SpectrumCalculation(unittest.TestCase):
    def test_jsa_matches_direct_frequency_formula(self):
        pump = Mode(np.linspace(.7, .85, 20), np.ones(20))
        signal = Mode(np.linspace(1.4, 1.7, 20), np.ones(20))
        fits = (Polynomial([1.9, .08, .015]), Polynomial([1.7, .03, .01]),
                Polynomial([1.8, .06, .02]))
        result = calculate_spectrum(pump, signal, signal, fits=fits, points=31,
                                    half_window_rad_per_fs=.0025, pulse_fwhm_fs=2000, length_m=.12)
        grid = result.angular_frequency_rad_per_fs
        wp = float(um_to_omega(.775))
        wi, ws = grid[:, None], grid[None, :]
        k = lambda fit, omega: fit(omega) * omega / LIGHT_SPEED_M_PER_FS
        dk0 = k(fits[0], wp) - k(fits[1], wp / 2) - k(fits[2], wp / 2)
        mismatch = k(fits[0], ws + wi) - k(fits[1], ws) - k(fits[2], wi) - dk0
        tau = 2000 / (2 * np.sqrt(np.log(2)))
        expected = (np.exp(-(wp - ws - wi)**2 * tau**2 / 2)
                    * np.sinc(mismatch * .12 / (2 * np.pi)) * np.exp(1j * mismatch * .12 / 2))
        assert_allclose(result.jsa, expected, atol=1e-13)
        singular = np.linalg.svd(expected, compute_uv=False)
        weights = singular**2 / np.sum(singular**2)
        self.assertAlmostEqual(result.schmidt_number, 1 / np.sum(weights**2), places=10)


class SavedWorkflow(unittest.TestCase):
    def test_all_rejected_still_produces_readable_empty_reports(self):
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        from linbo3.interpolation_study import run_study
        with tempfile.TemporaryDirectory() as temporary:
            run = write_saved_run(Path(temporary) / "run", widths=(.5,))
            report = run_study(run, points=17, tracking={"imag_limit": 1e-12})
            self.assertTrue(report.summary.empty)
            self.assertTrue(pd.read_csv(report.output_dir / "summary.csv").empty)
            self.assertTrue(pd.read_csv(report.output_dir / "fit_errors.csv").empty)
            self.assertTrue(report.pairs.analysis_state.eq("ambiguous").all())

    def test_all_widths_partial_pair_cache_and_no_schmidt_figures(self):
        import matplotlib
        matplotlib.use("Agg")
        from linbo3.interpolation_study import run_study, saved_study_is_current
        from linbo3.saved_modes import inspect_modes
        with tempfile.TemporaryDirectory() as temporary:
            run = write_saved_run(Path(temporary) / "run", incomplete_width=.7)
            report = run_study(run, points=17)
            self.assertEqual(len(report.summary), 2 * len(METHODS))
            inspection = inspect_modes(run)
            self.assertTrue(saved_study_is_current(run, inspection, 17))
            for width in (.5, .7):
                root = report.output_dir / f"w_{width:g}"
                self.assertTrue((root / "neff_raw.png").is_file())
                self.assertTrue((root / "dispersion_omega.png").is_file())
            self.assertFalse(list(report.output_dir.rglob("schmidt_modes.png")))
            self.assertFalse(list(report.output_dir.rglob("schmidt_weights.png")))
            paths = list(report.output_dir.rglob("poly*.npz"))
            stamps = {path: path.stat().st_mtime_ns for path in paths}
            second = run_study(run, points=17)
            assert_allclose(second.summary.K, report.summary.K)
            self.assertTrue(all(path.stat().st_mtime_ns == stamp for path, stamp in stamps.items()))
            changed = inspect_modes(run, tracking={"imag_limit": 2e-6})
            self.assertFalse(saved_study_is_current(run, changed, 17))


if __name__ == "__main__":
    unittest.main()
