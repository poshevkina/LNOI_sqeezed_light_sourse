"""Orchestrate and checkpoint the COMSOL-to-figures calculation."""

from pathlib import Path
import traceback
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from .artifacts import digest_json, file_hash, read_json, relative_path, save_arrays, utc_now, write_csv, write_json
from .comsol import MATERIAL_TABLES, solve_modal_fields
from .modes import ModalSolutions, ModeTrackingError, track_modes
from .polynomial import fit_study
from .saved_modes import load_pair
from .spdc import frequency_grid, omega_to_um
from .sweep import PulseConfig, run_sweep


BANDS = ("PUMP", "IS")
EXTRACTION_VERSION = 2


def _read_raw(path):
    with np.load(path, allow_pickle=False) as saved:
        return ModalSolutions(**{name: saved[name] for name in ModalSolutions.__annotations__})


def _save_raw_csv(path, raw):
    data = pd.DataFrame(dict(
        wavelength_um=raw.wavelength_um, outer_solnum=raw.outer_solnum, solnum=raw.solnum,
        neff_real=raw.neff.real, neff_imag=raw.neff.imag,
    ))
    write_csv(path, data, encoding="utf-8-sig", na_rep="NaN")


def _validate_solver_config(config):
    widths = config["widths_um"]
    if not widths or len(set(widths)) != len(widths) or not np.isfinite(widths).all() or min(widths) <= 0:
        raise ValueError("Укажи различные положительные ширины в widths_um")
    field_grid = config["field_grid"]
    if len(field_grid) != 2 or any(type(n) is not int or n < 2 for n in field_grid):
        raise ValueError("field_grid должен содержать два целых размера от 2")
    if type(config["cores"]) is not int or config["cores"] < 1:
        raise ValueError("cores должен быть положительным целым числом")
    modes = config["modes"]
    if modes is not None and (type(modes) is not int or modes < 1):
        raise ValueError("modes должен быть null или положительным целым числом")
    geometry = config["geometry"]
    if not isinstance(geometry, dict) or "w" in geometry:
        raise ValueError("Укажи геометрию в geometry, а ширины в widths_um")
    return sorted(widths)


def _solver_grids(config, *, exact_count):
    pump = config["pump_wavelength_um"]
    bands = {}
    for band, anchor in (("PUMP", pump), ("IS", 2 * pump)):
        span = config["bands"][band]
        start, stop, count = span["start_um"], span["stop_um"], span["points"]
        if not 0 < start < stop or not np.isfinite(stop) or type(count) is not int or count < 2:
            raise ValueError(f"{band}: нужен конечный возрастающий диапазон и минимум две точки")
        if not start <= anchor <= stop:
            raise ValueError(f"{band}: диапазон не покрывает опорную точку {anchor:g} мкм")
        grid = np.linspace(start, stop, count)
        exact = np.flatnonzero(grid == anchor)
        if not exact.size:
            if exact_count:
                if count < 3:
                    raise ValueError(f"{band}: для опорной точки внутри диапазона нужны минимум три точки")
                distance = np.abs(grid[1:-1] - anchor)
                tied = np.isclose(distance, distance.min(), rtol=1e-12, atol=1e-15)
                grid[np.flatnonzero(tied)[0] + 1] = anchor
            else:
                close = np.flatnonzero(np.isclose(grid, anchor, rtol=0, atol=1e-10))
                if len(close):
                    grid[close[0]] = anchor
                else:
                    grid = np.sort(np.append(grid, anchor))
        if not np.all(np.diff(grid) > 0):
            raise ValueError(f"{band}: длины волн должны строго возрастать")
        bands[band] = grid
    return bands


def _validate_analysis_config(config):
    degrees = config["degrees"]
    if not degrees or any(type(degree) is not int or not 1 <= degree <= 6 for degree in degrees) or len(set(degrees)) != len(degrees):
        raise ValueError("Укажи различные целые степени от 1 до 6 в degrees")
    if not 0 <= config["min_overlap"] <= 1 or not 0 <= config["min_margin"] <= 1:
        raise ValueError("Пороги перекрытия и запаса должны лежать от 0 до 1")
    if not np.isfinite(config["imag_limit"]) or config["imag_limit"] <= 0:
        raise ValueError("imag_limit должен быть конечным и положительным")
    pulses = [PulseConfig(**pulse) for pulse in config["pulses"]]
    if not pulses or len({pulse.tag for pulse in pulses}) != len(pulses):
        raise ValueError("Укажи различные настройки импульсов")
    if any(p.t_fs <= 0 or p.dw_half <= 0 or p.n_points < 3 for p in pulses):
        raise ValueError("Длительность и окно должны быть положительны, n_points ≥ 3")
    for band in BANDS:
        span = config["bands"][band]
        if (not 0 < span["start_um"] < span["stop_um"]
                or type(span["points"]) is not int or span["points"] <= max(degrees)):
            raise ValueError(f"{band}: диапазон должен возрастать и покрывать степень полинома")
    return pulses


def _solver_source(model):
    return dict(model_sha256=file_hash(model), material_sha256={
        filename: file_hash(model.parent / filename) for filename in MATERIAL_TABLES.values()
    })


def run(config_file: Path, *, stage: str = "all") -> Path:
    config_file = Path(config_file).resolve(strict=True)
    config = read_json(config_file)
    try:
        if stage not in ("all", "comsol"):
            raise ValueError(f"Неизвестный этап {stage}")
        widths = _validate_solver_config(config)
        grids = _solver_grids(config, exact_count=True)
        pulses = _validate_analysis_config(config) if stage == "all" else None
        model = (config_file.parent / config["model"]).resolve(strict=True)
        output = (config_file.parent / config["output_dir"]).resolve()
    except (KeyError, TypeError) as error:
        raise ValueError(f"Проверь поля файла настроек {config_file}: {error}") from error

    source = _solver_source(model)
    solver = {key: config[key] for key in ("geometry", "modes", "cores", "field_grid", "pump_wavelength_um")}
    identity = dict(**source, **solver, version=EXTRACTION_VERSION, widths_um=widths,
                    wavelengths_um={band: grid.tolist() for band, grid in grids.items()})
    run_id = f"w{widths[0]:g}-{widths[-1]:g}um_{len(widths)}widths__{digest_json(identity)[:16]}"
    run_dir = output / "comsol" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"RESULT_DIR={run_dir}", flush=True)
    manifest = dict(**identity, run_id=run_id, stage="comsol",
                    model=relative_path(run_dir, model), model_path_base="run_dir")
    write_json(run_dir / "config.json", manifest)
    status_path = run_dir / "status.json"
    previous = read_json(status_path) if status_path.is_file() else {}
    status = {"state": "running", "widths": previous.get("widths", {}), "pairs": {
        _width_key(width): previous.get("pairs", {}).get(
            _width_key(width), {band: {"state": "pending"} for band in BANDS}
        ) for width in widths
    }}
    write_json(status_path, status)
    try:
        _run_comsol(config, model, source, output, run_dir, widths, grids, status)
    except BaseException as error:
        status["state"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        status["error"] = str(error)
        write_json(status_path, status)
        raise
    status.update(state="complete", completed_at=utc_now())
    write_json(status_path, status)
    if pulses is not None:
        params = {key: config[key] for key in (
            "degrees", "length_m", "pulses", "min_overlap", "min_margin", "imag_limit", "anchor_solnums"
        )}
        recipe = dict(comsol_run=run_id, **params)
        report = output / "studies" / "cli" / f"{run_id}__{digest_json(recipe)[:16]}"
        write_json(report / "settings.json", {**recipe, "state": "running"})
        _calculate(config, run_dir, report, widths, grids, pulses)
        write_json(report / "settings.json", {**recipe, "state": "complete"})
    print(f"Готово | {run_dir}", flush=True)
    return run_dir


def _calculate(config, run_dir, report, widths, bands, pulses):
    from matplotlib import use
    use("Agg")
    from .study_plots import StudyPlots

    pump = config["pump_wavelength_um"]
    anchors = {"PUMP": pump, "IS": 2 * pump}
    for pulse in pulses:
        grid = frequency_grid(pump, pulse.dw_half, pulse.n_points)
        for band, factor in (("PUMP", 2), ("IS", 1)):
            required = omega_to_um(factor * grid[[0, -1]])
            if required.min() < bands[band][0] - 1e-12 or required.max() > bands[band][-1] + 1e-12:
                raise ValueError(
                    f"{band}, {pulse.t_fs:g} фс: нужно [{required.min():.6g}, "
                    f"{required.max():.6g}] мкм. Расширь диапазон COMSOL или уменьши dw_half."
                )

    raw_modes = {}
    failures = []
    for i, width in enumerate(widths, start=1):
        width_key = _width_key(width)
        width_dir = report / f"w_{width_key}"
        width_dir.mkdir(exist_ok=True)
        label = f"[{i}/{len(widths)}] w={width_key} мкм"
        tracks, diagnostics, errors = {}, [], []
        for band in BANDS:
            name = f"{label} | {band}"
            _, raw = load_pair(run_dir / f"w_{width_key}", band, bands[band])
            print(f"{name} | проверка ветвей", flush=True)
            try:
                tracked = track_modes(
                    raw, band=band, anchor_um=anchors[band], imag_limit=config["imag_limit"],
                    min_overlap=config["min_overlap"], min_margin=config["min_margin"],
                    anchor_solnums=config["anchor_solnums"].get(width_key, {}).get(band),
                )
            except ModeTrackingError as error:
                errors.append(f"w={width_key}: {error}")
                print(errors[-1], flush=True)
                continue
            tracks[band] = tracked.modes
            diagnostics.extend(tracked.rows)
        if diagnostics:
            pd.DataFrame(diagnostics).to_csv(width_dir / "branches.csv", index=False, encoding="utf-8-sig")
        write_json(width_dir / "tracking.json", {"errors": errors})
        if errors:
            failures.extend(errors)
        else:
            raw_modes[width] = tracks

    if failures:
        raise RuntimeError("Проверь ветви и при необходимости задай anchor_solnums. " + " | ".join(failures))

    study = fit_study(raw_modes, degrees=tuple(config["degrees"]), pump_wavelength_um=pump)
    plots = StudyPlots(study, report / "figures", max_plot_points=601, show=False)
    print("Сохранение графиков дисперсии", flush=True)
    plots.plot_dispersion_by_width()
    if len(widths) > 1:
        plots.plot_dispersion_by_degree()
    for pulse in pulses:
        print(f"{pulse.t_fs:g} фс | расчёт спектров и числа Шмидта", flush=True)
        pulse_output = report / pulse.tag
        sweep = run_sweep(study, pulse, pulse_output, length_m=config["length_m"])
        plots=StudyPlots(study,pulse_output/"figures",max_plot_points=601,show=False)
        print(f"{pulse.t_fs:g} фс | сохранение графиков", flush=True)
        plots.plot_jsi_by_width(sweep.cases, pulse)
        plots.plot_spectra_by_width(sweep.cases, pulse)
        if len(widths) > 1:
            plots.plot_jsi_by_degree(sweep.cases, pulse)
            plots.plot_spectra_by_degree(sweep.cases, pulse)
            plots.plot_summary(sweep.summary, pulse)


def _width_key(width):
    return np.format_float_positional(float(width), trim="-")


def _run_comsol(config, model, source, output, run_dir, widths, grids, status):
    failures = []
    completed = 0
    total = len(widths) * len(BANDS)
    for width in widths:
        key = _width_key(width)
        directory = run_dir / f"w_{key}"
        directory.mkdir(exist_ok=True)
        pair_status = status["pairs"][key]
        for band in BANDS:
            label = f"[{completed}/{total} готово] w={key} мкм | {band}"
            pair_status[band] = {"state": "running"}
            write_json(run_dir / "status.json", status)
            diagnostic = directory / f"{band}_error.json"
            try:
                _collect_band(config, model, source, output, directory, width, band, grids[band], label)
            except Exception as error:
                message = f"w={key} мкм, {band}: {error}"
                failures.append(message)
                pair_status[band] = {"state": "failed", "error": str(error)}
                write_json(diagnostic, {"error": str(error), "traceback": traceback.format_exc()})
                print(f"{label} | ошибка: {error}", flush=True)
            else:
                diagnostic.unlink(missing_ok=True)
                pair_status[band] = {"state": "complete"}
                completed += 1
                print(f"[{completed}/{total} готово] w={key} мкм | {band} | сохранено", flush=True)
            write_json(run_dir / "status.json", status)
        status["widths"][key] = "ready" if all(p["state"] == "complete" for p in pair_status.values()) else "failed"
        write_json(run_dir / "status.json", status)
    if failures:
        raise RuntimeError("Незавершённые пары COMSOL: " + " | ".join(failures))


def _validate_raw(raw, grid):
    count = len(raw.wavelength_um)
    if not np.array_equal(np.unique(raw.wavelength_um), grid):
        raise ValueError("В выгрузке отсутствуют или изменились точки сетки")
    if (raw.neff.shape != (count,) or raw.outer_solnum.shape != (count,)
            or raw.solnum.shape != (count,) or raw.field.ndim != 3
            or raw.field.shape[:2] != (count, 3)
            or raw.coordinates.shape != (2, raw.field.shape[2])):
        raise ValueError("Поля, координаты и массивы идентификаторов не согласованы")
    if len(np.unique(np.column_stack((raw.outer_solnum, raw.solnum)), axis=0)) != count:
        raise ValueError("Идентификаторы решений повторяются")


def _collect_band(config, model, source, output, width_dir, width, band, wavelengths, label):
    settings = dict(
        **source, version=EXTRACTION_VERSION, width_um=width, band=band,
        geometry=config["geometry"], wavelengths_um=wavelengths.tolist(), modes=config["modes"],
        cores=config["cores"], field_grid=config["field_grid"],
    )
    nx, ny = settings["field_grid"]
    field_id = (f"w{width:g}_{band}_{wavelengths[0]:g}-{wavelengths[-1]:g}um_"
                f"{len(wavelengths)}pts_{nx}x{ny}__{digest_json(settings)[:16]}")
    directory = output.parent / "field_maps" / field_id
    field_file = directory / "field.npz"
    metadata = directory / "settings.json"
    signature = dict(**settings, field_id=field_id, model_name=model.name)
    raw = None
    if field_file.is_file() and metadata.is_file():
        try:
            stored = read_json(metadata)
            if all(stored.get(key) == value for key, value in settings.items()):
                raw = _read_raw(field_file)
                _validate_raw(raw, wavelengths)
                signature["field_sha256"] = stored["field_sha256"]
        except (ValueError, OSError, KeyError, BadZipFile):
            raw = None
            print(f"{label} | неполный кэш, повторяем расчёт", flush=True)
    if raw is None:
        directory.mkdir(parents=True, exist_ok=True)
        raw = solve_modal_fields(
            model, wavelengths, width_um=width, geometry_parameters=config["geometry"],
            modes=config["modes"], cores=config["cores"], field_grid=tuple(config["field_grid"]),
            log=directory / "comsol.log", label=label,
        )
        _validate_raw(raw, wavelengths)
        if _solver_source(model) != source:
            raise RuntimeError("Модель или таблицы материалов изменились во время расчёта COMSOL")
        save_arrays(field_file, raw.arrays())
        signature["field_sha256"] = file_hash(field_file)
        write_json(metadata, signature)
    else:
        print(f"{label} | готовое решение из кэша", flush=True)
    csv = width_dir / f"{band}_raw.csv"
    _save_raw_csv(csv, raw)
    write_json(width_dir / f"{band}_field.json", {
        **signature, "npz": relative_path(width_dir, field_file),
    })
