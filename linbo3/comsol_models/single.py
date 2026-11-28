"""The original 2D LNOI waveguide and its PUMP/IS extraction contract."""

from functools import partial
from pathlib import Path

import numpy as np

from ..artifacts import digest_json, file_hash, relative_path
from .cache import ModalTask, run_modal
from .single_solver import MATERIAL_TABLES, solve_modal_fields


EXTRACTION_VERSION = 2


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


class SingleWaveguide2D:
    def __init__(self, model_path, *, results_root="results"):
        self.model_path = Path(model_path).resolve(strict=True)
        self.results_root = Path(results_root).resolve()

    def _fingerprint(self):
        return {
            "model_sha256": file_hash(self.model_path),
            "material_sha256": {name: file_hash(self.model_path.parent / name) for name in MATERIAL_TABLES.values()},
        }

    def modes(self, *, widths_um, bands, pump_wavelength_um, field_grid=(81, 61),
              geometry=None, mode_count=None, cores=2, solve=False, dry_run=False):
        config = dict(widths_um=list(widths_um), bands=bands, pump_wavelength_um=pump_wavelength_um,
                      geometry={} if geometry is None else geometry, modes=mode_count, cores=cores,
                      field_grid=list(field_grid))
        widths = _validate_solver_config(config)
        grids = _solver_grids(config)
        source = self._fingerprint()
        solver = {key: config[key] for key in ("geometry", "modes", "cores", "field_grid", "pump_wavelength_um")}
        identity = dict(**source, **solver, version=EXTRACTION_VERSION, widths_um=widths,
                        wavelengths_um={band: grid.tolist() for band, grid in grids.items()})
        run_id = f"w{widths[0]:g}-{widths[-1]:g}um_{len(widths)}widths__{digest_json(identity)[:16]}"
        run_dir = self.results_root / "comsol" / run_id
        manifest = dict(**identity, run_id=run_id, stage="comsol",
                        model=relative_path(run_dir, self.model_path), model_path_base="run_dir")
        tasks = []
        for width in widths:
            key = np.format_float_positional(float(width), trim="-")
            for band, wavelengths in grids.items():
                settings = dict(
                    **source, version=EXTRACTION_VERSION, width_um=width, band=band,
                    geometry=config["geometry"], wavelengths_um=wavelengths.tolist(), modes=mode_count,
                    cores=cores, field_grid=config["field_grid"],
                )
                nx, ny = field_grid
                field_id = (f"w{width:g}_{band}_{wavelengths[0]:g}-{wavelengths[-1]:g}um_"
                            f"{len(wavelengths)}pts_{nx}x{ny}__{digest_json(settings)[:16]}")
                calculate = partial(solve_modal_fields, self.model_path, wavelengths, width_um=width,
                                    geometry_parameters=config["geometry"], modes=mode_count,
                                    cores=cores, field_grid=tuple(field_grid))
                tasks.append(ModalTask(
                    f"w_{key}", band, f"w={key} мкм | {band}", dict(width_um=width, band=band),
                    self.results_root.parent / "field_maps" / field_id, settings, self.model_path.name,
                    calculate, status_key=key,
                ))
        return run_modal(run_dir, manifest, tasks, source=source, fingerprint=self._fingerprint,
                         solve=solve, dry_run=dry_run, status_summary="widths")
