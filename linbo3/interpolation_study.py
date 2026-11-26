from .storage import analysis_root, remove_owned
"""Compare polynomial n_eff(omega) fits using saved COMSOL exports only."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .frequency import LIGHT_SPEED_M_PER_FS, um_to_omega
from .interpolation import METHODS, fit_dispersion, leave_one_out
from .saved_modes import BANDS, TRACKING, inspect_modes
from .spdc import calculate_spectrum
from .sweep import save_arrays


SETTINGS = dict(pump_wavelength_um=0.775, length_m=0.12, pulse_fwhm_fs=2000.0,
                half_window_rad_per_fs=0.0025, points=1501, mode_count=0)
CACHE_VERSION = 2
SUMMARY_COLUMNS = ["width_um", "pump_mode", "signal_mode", "idler_mode", "method", "K",
                   "Lambda_signed_um", "Lambda_um", "delta_k0_per_m", "npz"]


@dataclass(frozen=True)
class StudyReport:
    output_dir: Path
    summary: pd.DataFrame
    pairs: pd.DataFrame


def study_directory(run_dir, points=1501):
    return analysis_root(run_dir)/"comparison"


def source_signature(run_dir):
    """Detect updated/replaced source exports before displaying saved comparisons."""
    run_dir = Path(run_dir)
    payload = []
    for path in sorted(run_dir.glob("w_*/*_raw.*")):
        stat = path.stat()
        payload.append((path.relative_to(run_dir).as_posix(), stat.st_size, stat.st_mtime_ns))
    for name in ("config.json", "status.json"):
        value = json.loads((run_dir / name).read_text(encoding="utf-8-sig"))
        if name == "config.json":
            # Model content is identified by model_sha256; its address is portable.
            value = {key: item for key, item in value.items()
                     if key not in {"model", "model_path_base", "output_dir"}}
        payload.append((name, value))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def saved_study_is_current(run_dir, inspection, points=1501):
    path = study_directory(run_dir, points) / "settings.json"
    if not path.is_file():
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))
    return (saved.get("state") == "complete" and saved.get("cache_version") == CACHE_VERSION
            and saved.get("fit_coordinate") == "omega_rad_per_fs"
            and saved.get("tracking") == inspection.tracking
            and saved.get("anchor_solnums") == inspection.anchor_solnums
            and saved.get("methods") == list(METHODS)
            and saved.get("points") == points
            and saved.get("source_signature") == source_signature(run_dir))


def fit_saved_modes(inspection):
    """Cheap fitting/derivative inspection, without computing JSA or Schmidt decompositions."""
    from .interpolation_plots import plot_fits, plot_derivatives

    all_fits, held_out, training, coefficients = {}, [], [], []
    for width, bands in inspection.modes.items():
        fits = {band: {} for band in bands}
        for band, modes in bands.items():
            for index, mode in modes.items():
                fits[band][index] = {}
                for method in METHODS:
                    poly = fit_dispersion(method, mode.wavelength_um, mode.neff)
                    fits[band][index][method] = poly
                    held_out.extend(dict(width_um=width, band=band, mode=index, method=method, **record)
                                    for record in leave_one_out(method, mode.wavelength_um, mode.neff))
                    residual = poly(um_to_omega(mode.wavelength_um)) - mode.neff
                    training.append(dict(width_um=width, band=band, mode=index, method=method,
                                         points=len(residual), rmse=np.sqrt(np.mean(residual**2)),
                                         max_abs_error=np.abs(residual).max()))
                    for power, value in enumerate(poly.coef):
                        coefficients.append(dict(width_um=width, band=band, mode=index, method=method,
                                                 power=power, coefficient=value, domain_min=poly.domain[0],
                                                 domain_max=poly.domain[1], window_min=poly.window[0],
                                                 window_max=poly.window[1], fit_coordinate="omega_rad_per_fs"))
        all_fits[width] = fits
        directory = inspection.output_dir / f"w_{width:g}"
        directory.mkdir(exist_ok=True)
        plot_fits(directory / "neff_fits.png", width, bands, fits)
        plot_derivatives(directory / "dispersion_omega.png", width, bands, fits)
    output = inspection.output_dir
    pd.DataFrame(training, columns=["width_um", "band", "mode", "method", "points", "rmse",
                                    "max_abs_error"]).to_csv(output / "training_errors.csv", index=False)
    pd.DataFrame(coefficients, columns=["width_um", "band", "mode", "method", "power", "coefficient",
                                       "domain_min", "domain_max", "window_min", "window_max",
                                       "fit_coordinate"]).to_csv(output / "fit_coefficients.csv", index=False)
    if held_out:
        table = pd.DataFrame(held_out)
        table.to_csv(output / "interior_loo.csv", index=False)
        table.assign(squared=table.error**2, absolute=table.error.abs()).groupby(
            ["width_um", "band", "mode", "method"], sort=False).agg(
                rmse=("squared", lambda x: np.sqrt(x.mean())),
                max_abs_error=("absolute", "max"), held_out_points=("error", "size"),
            ).reset_index().to_csv(output / "fit_errors.csv", index=False)
    else:
        pd.DataFrame(columns=["width_um", "band", "mode", "method", "wavelength_um", "omega_rad_per_fs",
                              "observed", "predicted", "error"]).to_csv(output / "interior_loo.csv", index=False)
        pd.DataFrame(columns=["width_um", "band", "mode", "method", "rmse", "max_abs_error",
                              "held_out_points"]).to_csv(output / "fit_errors.csv", index=False)
    return all_fits


def _calculate_case(directory, method, modes, fits, settings):
    digest = hashlib.sha256(json.dumps(
        {"version": CACHE_VERSION, "coordinate": "omega_rad_per_fs", "method": method,
         "c_m_per_fs": LIGHT_SPEED_M_PER_FS, "settings": settings}, sort_keys=True,
    ).encode())
    for mode, poly in zip(modes, fits, strict=True):
        for array in (mode.wavelength_um, mode.neff, poly.coef, poly.domain, poly.window):
            digest.update(np.asarray([len(array)], dtype="<i8").tobytes())
            digest.update(np.asarray(array, dtype="<f8").tobytes())
    target = directory / f"{method}_{digest.hexdigest()[:16]}.npz"
    if target.is_file():
        with np.load(target, allow_pickle=False) as file:
            return target, {key: file[key] for key in file.files}
    spectrum = calculate_spectrum(*modes, fits=fits, **settings)
    omega = spectrum.angular_frequency_rad_per_fs
    indices = np.linspace(0, len(omega)-1, min(len(omega), 451), dtype=int)
    selected_jsa = spectrum.jsa[np.ix_(indices, indices)]
    payload = dict(
        omega_rad_per_fs=omega, wavelength_nm=spectrum.wavelength_nm,
        K=spectrum.schmidt_number, Lambda_signed_um=spectrum.signed_poling_period_um,
        delta_k0_per_m=spectrum.delta_k0_per_m, singular_values=spectrum.singular_values,
        schmidt_weights=spectrum.schmidt_weights, plot_omega_rad_per_fs=omega[indices],
        plot_jsi=np.abs(selected_jsa)**2,
    )
    if spectrum.signal_modes is not None:
        payload.update(signal_modes=spectrum.signal_modes, idler_modes=spectrum.idler_modes)
    save_arrays(target, payload)
    return target, payload


def run_study(run_dir: Path, *, points: int = 1501, anchor_solnums=None,
              tracking=None, schmidt_plots: bool = False) -> StudyReport:
    """All widths and PUMP branches; Schmidt modes/weights figures are opt-in."""
    from .interpolation_plots import plot_jsi_comparison, plot_schmidt

    if type(points) is not int or points < 5:
        raise ValueError("Нужно целое число частотных точек не меньше пяти")
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "config.json").read_text(encoding="utf-8-sig"))
    output = study_directory(run_dir, points)
    inspection = inspect_modes(run_dir, output_dir=output, tracking=tracking, anchor_solnums=anchor_solnums)
    settings = {**SETTINGS, "points": points, "mode_count": 4 if schmidt_plots else 0,
                "pump_wavelength_um": manifest.get("pump_wavelength_um", SETTINGS["pump_wavelength_um"]),
                "length_m": manifest.get("length_m", SETTINGS["length_m"])}
    metadata = {**settings, "methods": list(METHODS), "tracking": inspection.tracking,
                "anchor_solnums": inspection.anchor_solnums, "cache_version": CACHE_VERSION,
                "fit_coordinate": "omega_rad_per_fs", "imag_rule": "strict_at_every_wavelength",
                "source_signature": source_signature(run_dir),
                "model_sha256": manifest.get("model_sha256"), "state": "running"}
    (output / "settings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary = []
    # Reset the table before calculating, so failed/incomplete runs cannot show an old summary.
    pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(output / "summary.csv", index=False)
    fits_by_width = fit_saved_modes(inspection)
    for width, bands in inspection.modes.items():
        if not all(band in bands for band in BANDS):
            continue
        fits = fits_by_width[width]
        for pump_id, pump in bands["PUMP"].items():
            directory = output / f"w_{width:g}" / f"pump_{pump_id}"
            directory.mkdir(exist_ok=True)
            modes = (pump, bands["IS"][0], bands["IS"][1])
            cases = {}
            title = f"{width:g} мкм · PUMP ветвь {pump_id} · 2000 фс"
            for method in METHODS:
                print(f"{title} | {method}", flush=True)
                triplet = (fits["PUMP"][pump_id][method], fits["IS"][0][method], fits["IS"][1][method])
                file, case = _calculate_case(directory, method, modes, triplet, settings)
                cases[method] = case
                summary.append(dict(
                    width_um=width, pump_mode=pump_id, signal_mode=0, idler_mode=1,
                    method=method, K=float(case["K"]), Lambda_signed_um=float(case["Lambda_signed_um"]),
                    Lambda_um=abs(float(case["Lambda_signed_um"])), delta_k0_per_m=float(case["delta_k0_per_m"]),
                    npz=file.relative_to(output).as_posix(),
                ))
            plot_jsi_comparison(directory, title, cases)
            if schmidt_plots:
                plot_schmidt(directory, title, cases)
            pd.DataFrame(summary, columns=SUMMARY_COLUMNS).to_csv(output / "summary.csv", index=False)
    expected={str((output/item['npz']).resolve()) for item in summary}
    for path in output.rglob('*.npz'):
        if str(path.resolve()) not in expected:remove_owned(path,output)
    metadata["state"] = "complete"
    (output / "settings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return StudyReport(output, pd.DataFrame(summary, columns=SUMMARY_COLUMNS), inspection.pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--points", type=int, default=1501)
    args = parser.parse_args()
    from matplotlib import use
    use("Agg")
    report = run_study(args.run_dir, points=args.points)
    print(f"RESULT_DIR={report.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
