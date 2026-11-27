"""Run COMSOL and checkpoint each width and wavelength band."""

from contextlib import ExitStack
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from .artifacts import (
    digest_json, file_hash, file_lock, read_json, relative_path, save_arrays, utc_now, write_csv, write_json,
)
from .comsol import MATERIAL_TABLES, solve_modal_fields
from .saved_modes import BANDS, load_modal_npz


EXTRACTION_VERSION = 2


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


def _solver_grids(config):
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
        if not np.any(grid == anchor):
            if count < 3:
                raise ValueError(f"{band}: для опорной точки внутри диапазона нужны минимум три точки")
            distance = np.abs(grid[1:-1] - anchor)
            tied = np.isclose(distance, distance.min(), rtol=1e-12, atol=1e-15)
            grid[np.flatnonzero(tied)[0] + 1] = anchor
        if not np.all(np.diff(grid) > 0):
            raise ValueError(f"{band}: длины волн должны строго возрастать")
        bands[band] = grid
    return bands


def _solver_source(model):
    return dict(model_sha256=file_hash(model), material_sha256={
        filename: file_hash(model.parent / filename) for filename in MATERIAL_TABLES.values()
    })


def run(config_file: Path, *, stage: str = "comsol") -> Path:
    if stage != "comsol":
        raise ValueError(f"Неизвестный этап {stage}. Анализ запускается через python -m studies.…")
    config_file = Path(config_file).resolve(strict=True)
    config = read_json(config_file)
    try:
        widths = _validate_solver_config(config)
        grids = _solver_grids(config)
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
    with ExitStack() as stack:
        try:
            stack.enter_context(file_lock(run_dir / ".run.lock", wait=False))
        except BlockingIOError as error:
            raise RuntimeError(f"COMSOL-прогон {run_id} уже запущен") from error
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
    print(f"Готово | {run_dir}", flush=True)
    return run_dir


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
    with file_lock(directory / ".field.lock"):
        raw = None
        if field_file.is_file() and metadata.is_file():
            try:
                stored = read_json(metadata)
                if all(stored.get(key) == value for key, value in settings.items()):
                    checksum = stored["field_sha256"]
                    raw = load_modal_npz(field_file, expected_sha256=checksum or "")
                    _validate_raw(raw, wavelengths)
                    signature["field_sha256"] = checksum
            except (ValueError, OSError, KeyError) as error:
                raw = None
                print(f"{label} | неполный или повреждённый кэш: {error}", flush=True)
        if raw is None:
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
