"""Model adapter for the 2D LNOI pair with the Study IS and gap g."""

from functools import partial
from pathlib import Path

import numpy as np

from ..artifacts import digest_json, file_hash, relative_path
from ..comsol_session import open_model, solve_study
from .cache import ModalTask, run_modal
from .modal import extract_modes


PROFILE = "coupled-waveguides-2d"
EXTRACTION_VERSION = 1
# The ordinary and extraordinary indices are embedded in this MPH.
MATERIALS = {"int1": "n_SiO2.txt"}


class CoupledWaveguides2D:
    def __init__(self, model_path, *, results_root="results"):
        self.model_path = Path(model_path).resolve(strict=True)
        self.results_root = Path(results_root).resolve()

    def _fingerprint(self):
        return dict(model_sha256=file_hash(self.model_path), material_sha256={
            name: file_hash(self.model_path.parent / name) for name in MATERIALS.values()
        })

    def modes(self, *, width_um, gaps_um, wavelength_um, mode_count, field_grid=(21, 21),
              cores=2, solve=False, dry_run=False):
        width, wavelength = float(width_um), float(wavelength_um)
        gaps = sorted(float(gap) for gap in gaps_um)
        if not gaps or len(set(gaps)) != len(gaps) or not np.isfinite([width, wavelength, *gaps]).all():
            raise ValueError("Укажи конечные ширину, длину волны и различные зазоры")
        if min(width, wavelength, *gaps) <= 0:
            raise ValueError("Ширина, длина волны и зазоры должны быть положительными")
        shape = tuple(field_grid)
        if (type(cores) is not int or cores < 1 or type(mode_count) is not int or mode_count < 1
                or len(shape) != 2 or any(type(n) is not int or n < 2 for n in shape)):
            raise ValueError("Проверь число ядер, число мод и размеры сетки поля")

        source = self._fingerprint()
        identity = dict(**source, profile=PROFILE, version=EXTRACTION_VERSION, width_um=width,
                        wavelengths_um=[wavelength], modes=mode_count, cores=cores, field_grid=list(shape))
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
                partial(self._solve, width, gap, wavelength, mode_count, shape, cores),
            ))
        return run_modal(run_dir, manifest, tasks, source=source, fingerprint=self._fingerprint,
                         solve=solve, dry_run=dry_run)

    def _solve(self, width, gap, wavelength, mode_count, field_grid, cores, log, label):
        with open_model(self.model_path, cores=cores, label=label) as (client, model):
            for tag, name in MATERIALS.items():
                function = model.java.component("comp1").func(tag)
                function.set("source", "file")
                function.set("filename", str(self.model_path.parent / name))
            for name, value in (("w", width), ("g", gap), ("lbd0", wavelength)):
                model.parameter(name, f"{value:.16g}[um]")

            study = model.java.study("std2")
            parametric = study.feature("param")
            if list(parametric.getStringArray("pname")) != ["g"]:
                raise ValueError("Ожидался параметрический перебор только по зазору g")
            gap_value = f"{gap:.16g}"
            parametric.set("plistarr", [gap_value])
            parametric.set("punit", ["um"])
            if list(parametric.getStringArray("plistarr")) != [gap_value]:
                raise RuntimeError("Не удалось ограничить перебор одним значением g")
            step = study.feature("mode")
            step.set("neigsactive", "on")
            step.set("neigs", str(mode_count))

            # Both sets can contain results saved in the input MPH. Clear them before solving.
            for tag in ("sol16", "sol17"):
                model.java.sol(tag).clearSolutionData()
            print(f"{label} | COMSOL: один зазор g={gap_value} мкм", flush=True)
            solve_study(client, model, "Study IS", log=log, label=label)
            errors = []
            for dataset in ("dset4", "dset3"):
                try:
                    solution = str(model.java.result().dataset(dataset).getString("solution"))
                    raw = extract_modes(
                        model, np.array([wavelength]), component="comp1", geometry="geom1",
                        solution_tag=solution, dataset=dataset, field_grid=field_grid,
                        label=label, parameter=("g", gap * 1e-6),
                    )
                    if len(raw.neff) != mode_count or not np.isfinite(raw.neff).all() or not np.isfinite(raw.field).any():
                        raise RuntimeError(f"Ожидалось {mode_count} мод с полями, получено {len(raw.neff)}")
                    return raw
                except Exception as error:
                    errors.append(f"{dataset}: {error}")
            raise RuntimeError("Свежая выгрузка для указанного зазора не найдена: " + " | ".join(errors))
