"""Java tags and extraction rules for the original LNOI waveguide."""

from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..comsol_session import open_model, solve_study
from ..modes import ModalSolutions
from .modal import extract_modes, import_materials


MATERIAL_TABLES = {"int1": "n_SiO2.txt", "int2": "no_LiNbO3.txt", "int3": "ne_LiNbO3.txt"}


def _import_project_materials(model, model_file):
    import_materials(model, model_file, MATERIAL_TABLES, component="comp1")


@contextmanager
def _solved_model(path, wavelengths_um, width_um, geometry_parameters, modes, cores,
                  *, label="", log=None):
    path = Path(path).resolve(strict=True)
    wavelengths_um = np.asarray(wavelengths_um, dtype=float)
    if (wavelengths_um.ndim != 1 or not wavelengths_um.size
            or not np.isfinite(wavelengths_um).all() or wavelengths_um.min() <= 0):
        raise ValueError("Expected positive finite wavelengths in micrometres")
    if len(np.unique(wavelengths_um)) != len(wavelengths_um):
        raise ValueError("Wavelengths must be distinct")
    parameters = dict(geometry_parameters or {})
    if width_um is not None:
        if "w" in parameters or not np.isfinite(width_um) or width_um <= 0:
            raise ValueError("Specify one positive width via width_um")
        parameters["w"] = f"{width_um:.16g}[um]"
    if cores < 1 or (modes is not None and modes < 1):
        raise ValueError("Cores and requested mode count must be positive")

    with open_model(path, cores=cores, label=label) as (client, model):
        _import_project_materials(model,path)
        unknown = parameters.keys() - model.parameters().keys()
        if unknown:
            raise ValueError(f"Unknown COMSOL parameters: {', '.join(sorted(unknown))}")
        for name, expression in parameters.items():
            model.parameter(name, expression)
        step = model.java.study("std1").feature("mode")
        step.set("plistarr", [" ".join(f"{wavelength:.16g}" for wavelength in wavelengths_um)])
        if modes is not None:
            step.set("neigs", str(modes))
        solve_study(client, model, "Study 1", log=log, label=label)
        yield model


def solve_modes(path, wavelengths_um, *, width_um=None, geometry_parameters=None, modes=None, cores=2):
    """Read the scalar spectrum with the original lightweight interface."""
    with _solved_model(path, wavelengths_um, width_um, geometry_parameters, modes, cores) as model:
        dataset = "Study 1//Solution 1"
        wavelengths = np.atleast_1d(model.evaluate("lbd", dataset=dataset)) * 1e6
        indices = np.atleast_1d(model.evaluate("emw.neff", dataset=dataset))
        slots = {}
        rows = []
        for wavelength, index in zip(wavelengths, indices, strict=True):
            slot = slots.get(wavelength, 0) + 1
            slots[wavelength] = slot
            rows.append((float(wavelength), slot, float(index.real), float(index.imag)))
        return rows


def solve_modal_fields(
    path: Path, wavelengths_um: np.ndarray, *, width_um: float, geometry_parameters: dict,
    modes: int | None, cores: int, field_grid: tuple[int, int], log: Path, label: str,
) -> ModalSolutions:
    """Extract field and global results using the same solution indices."""
    with _solved_model(path, wavelengths_um, width_um, geometry_parameters, modes, cores,
                       log=log, label=label) as model:
        print(f"{label} | выгрузка решений и полей", flush=True)
        return extract_modes(model, wavelengths_um, component="comp1", geometry="geom1",
                             solution_tag="sol1", dataset="dset1", field_grid=field_grid, label=label)
