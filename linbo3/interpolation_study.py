from .storage import analysis_root
"""Compare polynomial n_eff(omega) fits using saved COMSOL exports only."""

import argparse
from dataclasses import dataclass
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import digest_json, file_hash, read_json, relative_path, results_root_for, utc_now, write_csv, write_json
from .case_cache import cache_case, case_paths
from .frequency import um_to_omega
from .interpolation import METHODS, fit_dispersion, leave_one_out
from .saved_modes import BANDS, TRACKING, inspect_modes
from .spdc import calculate_spectrum
from .study_output import publish_study, report_metadata


SETTINGS = dict(pump_wavelength_um=0.775, length_m=0.12, pulse_fwhm_fs=2000.0,
                half_window_rad_per_fs=0.0025, points=1501, mode_count=0)
CACHE_VERSION = 3
SUMMARY_COLUMNS = ["width_um", "pump_mode", "pump_rank", "signal_mode", "idler_mode", "method", "K",
                   "Lambda_signed_um", "Lambda_um", "delta_k0_per_m", "case_key", "npz"]


@dataclass(frozen=True)
class StudyReport:
    output_dir: Path
    summary: pd.DataFrame
    pairs: pd.DataFrame


def study_directory(run_dir, points=1501):
    return analysis_root(run_dir)/"comparison"


def source_signature(run_dir):
    """Use stable source contents, independent of location and file timestamps."""
    run_dir = Path(run_dir)
    payload = [(path.relative_to(run_dir).as_posix(), file_hash(path))
               for path in sorted(run_dir.glob("w_*/*_raw.*"))]
    for path in sorted(run_dir.glob("w_*/*_field.json")):
        reference = read_json(path)
        reference.pop("npz", None)
        payload.append((path.relative_to(run_dir).as_posix(), reference))
    config = read_json(run_dir / "config.json")
    payload.append({k: v for k, v in config.items()
                    if k not in {"model", "model_path_base", "output_dir"}})
    return digest_json(payload)


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


def fit_saved_modes(inspection, methods=METHODS):
    """Cheap fitting/derivative inspection, without computing JSA or Schmidt decompositions."""
    from .interpolation_plots import plot_fits, plot_derivatives

    all_fits, held_out, training, coefficients = {}, [], [], []
    for width, bands in inspection.modes.items():
        fits = {band: {} for band in bands}
        for band, modes in bands.items():
            for index, mode in modes.items():
                fits[band][index] = {}
                for method in methods:
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


def run_study(
    run_dir: Path, *, points: int = 1501, anchor_solnums=None, tracking=None,
    schmidt_plots: bool = False, study_name: str | None = None, methods=METHODS,
    widths_um=None, pump_modes=None, signal_mode: int = 0, idler_mode: int = 1,
    pulse_fs: float = 2000.0, half_window: float = 0.0025,
    output_dir=None, cache_root=None,
) -> StudyReport:
    """Track saved fields and compare chosen fits, checkpointing each numerical case."""
    if type(points) is not int or points < 5:
        raise ValueError("Нужно целое число частотных точек не меньше пяти")
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError(f"Выбери различные методы из {METHODS}")
    if study_name is not None and output_dir is not None:
        raise ValueError("Укажи имя исследования или явный каталог отчёта")
    run_dir = Path(run_dir).resolve()
    manifest = read_json(run_dir / "config.json")
    root = Path(cache_root).resolve() if cache_root is not None else results_root_for(run_dir) or run_dir.parent
    settings = {**SETTINGS, "points": points, "mode_count": 4 if schmidt_plots else 0,
                "pulse_fwhm_fs": float(pulse_fs), "half_window_rad_per_fs": float(half_window),
                "pump_wavelength_um": manifest.get("pump_wavelength_um", SETTINGS["pump_wavelength_um"]),
                "length_m": manifest.get("length_m", SETTINGS["length_m"])}
    thresholds = {key: manifest.get(key, default) for key, default in TRACKING.items()}
    thresholds.update(tracking or {})
    chosen = manifest.get("anchor_solnums", {}) if anchor_solnums is None else anchor_solnums
    widths = sorted(set(widths_um)) if widths_um is not None else None
    pump_ids = sorted(set(pump_modes)) if pump_modes is not None else None
    if signal_mode == idler_mode or min(signal_mode, idler_mode) < 0:
        raise ValueError("Выбери разные неотрицательные номера signal и idler")
    if widths is not None and (not widths or set(widths) - set(manifest["widths_um"])):
        raise ValueError("Выбери непустой набор ширин из COMSOL-прогона")

    recipe = {**settings, "methods": list(methods), "tracking": thresholds,
              "anchor_solnums": chosen, "widths_um": widths, "pump_modes": pump_ids,
              "signal_mode": signal_mode, "idler_mode": idler_mode, "cache_version": CACHE_VERSION,
              "comsol_run": manifest.get("run_id", run_dir.name)}
    if study_name is None:
        destination = Path(output_dir) if output_dir is not None else study_directory(run_dir, points)
        publication_context = nullcontext(destination)
    else:
        # A name is a single path component even when passed through the Python API.
        if Path(study_name).name != study_name or "/" in study_name or "\\" in study_name:
            raise ValueError("Укажи имя исследования без пути")
        destination = root / "studies" / study_name
        publication_context = publish_study(destination)
    with publication_context as output:
        table, pairs = _calculate_report(run_dir, output, root, recipe, settings,
                                         require_cases=study_name is not None,
                                         schmidt_plots=schmidt_plots)
    return StudyReport(destination, table, pairs)


def _calculate_report(run_dir, output, root, recipe, settings, *, require_cases, schmidt_plots):
    from .interpolation_plots import plot_jsi_comparison, plot_k_by_fit, plot_k_by_width, plot_schmidt

    methods = recipe["methods"]
    signal_mode, idler_mode = recipe["signal_mode"], recipe["idler_mode"]
    pulse_fs = settings["pulse_fwhm_fs"]
    metadata = report_metadata(run_dir, output, **recipe, fit_coordinate="omega_rad_per_fs",
                               imag_rule="strict_at_every_wavelength",
                               source_signature=source_signature(run_dir), state="running")
    publication = output / "settings.json"
    summary_path = output / "summary.csv"
    write_json(publication, metadata)
    summary = []
    try:
        inspection = inspect_modes(run_dir, output_dir=output, tracking=recipe["tracking"],
                                   anchor_solnums=recipe["anchor_solnums"], widths_um=recipe["widths_um"])
        fits_by_width = fit_saved_modes(inspection, methods)
        anchors = inspection.branches.loc[
            inspection.branches["band"].eq("PUMP") & inspection.branches["anchor"].eq(True)
        ].sort_values(["width_um", "neff_real", "mode"], ascending=[True, False, True])
        ranks = {
            (row.width_um, row.mode): rank
            for _, part in anchors.groupby("width_um")
            for rank, row in enumerate(part.itertuples(), start=1)
        }
        for width, bands in inspection.modes.items():
            if not all(band in bands for band in BANDS):
                continue
            if signal_mode not in bands["IS"] or idler_mode not in bands["IS"]:
                raise ValueError(f"w={width:g}: не найдены выбранные IS-моды")
            fits = fits_by_width[width]
            pump_ids = recipe["pump_modes"]
            for pump_id, pump in bands["PUMP"].items():
                if pump_ids is not None and pump_id not in pump_ids:
                    continue
                directory = output / f"w_{width:g}" / f"pump_{pump_id}"
                directory.mkdir(exist_ok=True)
                modes = (pump, bands["IS"][signal_mode], bands["IS"][idler_mode])
                cases = {}
                title = f"{width:g} мкм · PUMP ветвь {pump_id} · {pulse_fs:g} фс"
                for method in methods:
                    triplet = (fits["PUMP"][pump_id][method], fits["IS"][signal_mode][method],
                               fits["IS"][idler_mode][method])
                    path, full, signature = case_paths(root, modes, triplet, settings,
                                                       method=method, width=width)
                    case, reused = cache_case(
                        path, full, signature, lambda: calculate_spectrum(*modes, fits=triplet, **settings),
                    )
                    print(f"{title} | {method}" + (" [кэш]" if reused else ""), flush=True)
                    cases[method] = case
                    summary.append(dict(
                        width_um=width, pump_mode=pump_id, pump_rank=ranks[width, pump_id],
                        signal_mode=signal_mode, idler_mode=idler_mode,
                        method=method, K=float(case["K"]), Lambda_signed_um=float(case["Lambda_signed_um"]),
                        Lambda_um=float(case["Lambda_um"]), delta_k0_per_m=float(case["delta_k0_per_m"]),
                        case_key=signature["case_key"], npz=relative_path(output, path),
                    ))
                plot_jsi_comparison(directory, title, cases)
                if schmidt_plots:
                    plot_schmidt(directory, title, cases)
        table = pd.DataFrame(summary, columns=SUMMARY_COLUMNS)
        if require_cases and table.empty:
            details = inspection.pairs.loc[~inspection.pairs.analysis_state.eq("tracked")]
            raise ValueError("Нет случаев SPDC для публикации. " + " ".join(
                f"w={row.width_um:g}, {row.band}: {row.message or row.analysis_state}"
                for row in details.itertuples()
            ))
        write_csv(summary_path, table)
        plot_k_by_fit(output / "k_by_fit.png", table, methods)
        plot_k_by_width(output / "k_by_width.png", table, methods, sorted(inspection.pairs.width_um.unique()))
    except BaseException as error:
        state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        write_json(publication, {**metadata, "state": state, "error": str(error)})
        raise
    coverage = "complete" if inspection.pairs["analysis_state"].eq("tracked").all() else "partial"
    write_json(publication, {**metadata, "state": "complete", "coverage": coverage, "completed_at": utc_now()})
    return table, inspection.pairs


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
