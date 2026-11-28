"""Model adapter for the 2D LNOI pair with the Study IS and gap g."""

from functools import partial
from pathlib import Path

import numpy as np

from ...artifacts import digest_json, file_hash, relative_path
from ...comsol_session import open_model, solve_study
from ..cache import ModalTask, run_modal
from ..modal import extract_modes
from ..paths import material_directory


MODEL_DIR = Path(__file__).resolve().parent
TEMPLATE_NAME = "coupled_waveguides_2d_clean.mph"
MODEL_PATH = MODEL_DIR / TEMPLATE_NAME
PROFILE = "coupled-waveguides-2d"
EXTRACTION_VERSION = 2
# The ordinary and extraordinary indices are embedded in this MPH.
MATERIALS = {"int1": "n_SiO2.txt"}


class CoupledWaveguides2D:
    def __init__(self, model_path=MODEL_PATH, *, results_root="results"):
        self.model_path = Path(model_path).resolve(strict=True)
        self.results_root = Path(results_root).resolve()

    def _fingerprint(self):
        directory = material_directory(self.model_path, MODEL_DIR)
        return dict(model_sha256=file_hash(self.model_path), material_sha256={
            name: file_hash(directory / name) for name in MATERIALS.values()
        })

    def modes(self, *, width_um, gaps_um, wavelength_um, mode_count, neff_shift=2.0, field_grid=(21, 21),
              cores=2, mesh_scale=1.0, solve=False, dry_run=False, allow_partial=False):
        width, wavelength = float(width_um), float(wavelength_um)
        shift = float(neff_shift)
        gaps = sorted(float(gap) for gap in gaps_um)
        if not gaps or len(set(gaps)) != len(gaps) or not np.isfinite([width, wavelength, *gaps]).all():
            raise ValueError("Укажи конечные ширину, длину волны и различные зазоры")
        if min(width, wavelength, *gaps) <= 0:
            raise ValueError("Ширина, длина волны и зазоры должны быть положительными")
        if not np.isfinite(shift) or shift <= 0:
            raise ValueError("Сдвиг поиска по n_eff должен быть конечным и положительным")
        if not 0 < mesh_scale <= 1:
            raise ValueError("Масштаб размера элементов должен лежать в (0, 1]")
        shape = tuple(field_grid)
        if (type(cores) is not int or cores < 1 or type(mode_count) is not int or mode_count < 1
                or len(shape) != 2 or any(type(n) is not int or n < 2 for n in shape)):
            raise ValueError("Проверь число ядер, число мод и размеры сетки поля")

        source = self._fingerprint()
        identity = dict(**source, profile=PROFILE, version=EXTRACTION_VERSION, width_um=width,
                        wavelengths_um=[wavelength], modes=mode_count, neff_shift=shift,
                        cores=cores, field_grid=list(shape))
        mesh_options = {} if mesh_scale == 1 else {"mesh_scale": mesh_scale}
        identity.update(mesh_options)
        recipe = dict(**identity, gaps_um=gaps)
        run_id = f"w{width:g}_g{gaps[0]:g}-{gaps[-1]:g}um__{digest_json(recipe)[:16]}"
        run_dir = self.results_root / "comsol" / PROFILE / run_id
        manifest = dict(**recipe, run_id=run_id, stage="comsol", model=relative_path(run_dir, self.model_path),
                        model_path_base="run_dir")
        tasks = []
        for gap in gaps:
            key = np.format_float_positional(gap, trim="-")
            settings = dict(**identity, gap_um=gap)
            field_id = f"{PROFILE}_w{width:g}_g{gap:g}um__{digest_json(settings)[:16]}"
            tasks.append(ModalTask(
                f"g_{key}", "MODES", f"w={width:g} мкм | g={key} мкм | λ={wavelength:g} мкм",
                dict(width_um=width, gap_um=gap), self.results_root.parent / "field_maps" / field_id,
                settings, self.model_path.name,
                partial(self._solve, width, gap, wavelength, mode_count, shift, shape, cores, **mesh_options),
            ))
        return run_modal(run_dir, manifest, tasks, source=source, fingerprint=self._fingerprint,
                         solve=solve, dry_run=dry_run, allow_partial=allow_partial)

    def _solve(self, width, gap, wavelength, mode_count, neff_shift, field_grid, cores, log, label, *, mesh_scale=1.0):
        with open_model(self.model_path, cores=cores, label=label) as (client, model):
            directory = material_directory(self.model_path, MODEL_DIR)
            for tag, name in MATERIALS.items():
                function = model.java.component("comp1").func(tag)
                function.set("source", "file")
                function.set("filename", str(directory / name))
            for name, value in (("w", width), ("g", gap), ("lbd0", wavelength)):
                model.parameter(name, f"{value:.16g}[um]")

            if mesh_scale != 1:
                mesh = model.java.component("comp1").mesh("mesh1")
                for tag in ("ftri2", "ftri3"):
                    size = mesh.feature(tag).feature("size1")
                    hmax = float(str(size.getString("hmax")))
                    size.set("hmaxactive", "on")
                    size.set("hmax", str(hmax * mesh_scale))
                mesh.run()

            study = model.java.study("std2")
            if list(study.feature().tags()) != ["mode"] or list(model.java.batch().tags()):
                raise ValueError("Ожидался чистый шаблон с одним Mode Analysis, без Parametric Sweep")
            step = study.feature("mode")
            step.set("neigsactive", "on")
            step.set("neigs", str(mode_count))
            step.set("shiftactive", "on")
            step.set("shift", str(neff_shift))

            print(f"{label} | COMSOL: один зазор, n_eff около {neff_shift:g}, мод: {mode_count}", flush=True)
            solve_study(client, model, "Study IS", log=log, label=label)
            solution, dataset = _solution_dataset(model.java)
            raw = extract_modes(
                model, np.array([wavelength]), component="comp1", geometry="geom1",
                solution_tag=solution, dataset=dataset, field_grid=field_grid,
                label=label, parameter=("g", gap * 1e-6),
            )
            if len(np.unique(raw.outer_solnum)) != 1:
                raise RuntimeError("Ожидалась ровно одна точка расчёта COMSOL")
            if len(raw.neff) != mode_count or not np.isfinite(raw.neff).all() or not np.isfinite(raw.field).any():
                raise RuntimeError(f"Ожидалось {mode_count} мод с полями, получено {len(raw.neff)}")
            if min(field_grid) >= 200:
                _check_field_samples(model.java, dataset, raw)
            return raw


def _solution_dataset(java):
    solutions = [tag for tag in java.sol().tags()
                 if java.sol(tag).isAttached() and java.sol(tag).study() == "std2"]
    if len(solutions) != 1:
        raise RuntimeError("Ожидалось одно решение, связанное с исследованием Mode Analysis")
    solution = str(solutions[0])
    datasets = []
    for tag in java.result().dataset().tags():
        dataset = java.result().dataset(tag)
        if dataset.getType() == "Solution" and dataset.getString("solution") == solution:
            datasets.append(str(tag))
    if len(datasets) != 1:
        raise RuntimeError(f"Ожидался один dataset решения {solution}, найдено: {datasets}")
    return solution, datasets[0]


def _check_field_samples(java, dataset, raw):
    """Evaluate an independent set of points on the two core cross-sections."""
    from jpype import JArray, JInt

    x, y = raw.coordinates * 1e6
    ridge_samples = np.flatnonzero((np.abs(x) < 2.1) & (y > .3) & (y < .6))
    if not len(ridge_samples):
        raise RuntimeError("В координатах COMSOL нет отсчётов в плоскости сердечников")
    points = ridge_samples[np.linspace(len(ridge_samples)-1, 0, 11, dtype=int)]
    interp = java.result().numerical().create("linbo3check", "Interp")
    interp.set("data", dataset)
    interp.set("expr", ["comp1.Ex", "comp1.Ey", "comp1.Ez"])
    interp.setInterpolationCoordinates(raw.coordinates[:, points].tolist())
    if raw.outer_solnum[0]:
        interp.set("outersolnum", JInt(int(raw.outer_solnum[0])))
    interp.set("solnum", JArray(JInt)(raw.solnum.tolist()))
    measured = np.asarray(interp.getData()).transpose(1, 0, 2).astype(complex)
    if interp.isComplex():
        measured.imag = np.asarray(interp.getImagData()).transpose(1, 0, 2)
    expected = raw.field[:, :, points]
    scale = np.max(np.abs(expected))
    if not np.allclose(measured, expected, rtol=1e-8, atol=scale * 1e-10):
        raise RuntimeError("Независимая выборка COMSOL не совпала с сохранённым полем")
