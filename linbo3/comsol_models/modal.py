"""Shared 2D modal sampling for studies that export n_eff and complex fields."""

import numpy as np

from ..modes import ModalSolutions


def import_materials(model, directory, materials, *, component):
    for filename in materials.values():
        if not (directory / filename).is_file():
            raise FileNotFoundError(directory / filename)
    model.java.modelPath(str(directory))
    for tag, filename in materials.items():
        function = model.java.component(component).func(tag)
        function.set("filename", filename)
        function.importData()


def extract_modes(model, wavelengths_um, *, component, geometry, solution_tag, dataset,
                  field_grid, label, parameter=None):
    from jpype import JArray, JInt

    geom = model.java.component(component).geom(geometry)
    bounds = np.asarray(geom.getBoundingBox()).reshape(2, 2)
    nx, ny = field_grid
    axes = [limits[0] + (np.arange(n) + 0.5) / n * np.diff(limits)[0]
            for limits, n in zip(bounds, (nx, ny), strict=True)]
    coordinates = np.stack(np.meshgrid(*axes, indexing="ij")).reshape(2, -1)

    solution_info = model.java.sol(solution_tag).getSolutioninfo()
    interp = model.java.result().numerical().create("linbo3field", "Interp")
    interp.set("data", dataset)
    interp.set("expr", [f"{component}.{name}" for name in ("Ex", "Ey", "Ez")])
    interp.setInterpolationCoordinates(coordinates.tolist())
    global_eval = model.java.result().numerical().create("linbo3global", "Global")
    global_eval.set("data", dataset)
    global_eval.set("expr", ["lbd", "emw.neff", *([parameter[0]] if parameter else [])])

    tables = []
    for outer in list(solution_info.getOuterSolnum()) or [0]:
        inner = np.array(solution_info.getSolnum(JInt(outer or 1), True), dtype=int)
        if not inner.size:
            raise RuntimeError(f"{label}: COMSOL не вернул идентификаторы решений")
        for feature in (interp, global_eval):
            if outer:
                feature.set("outersolnum", JInt(outer))
            feature.set("solnum", JArray(JInt)(inner.tolist()))
        values = np.asarray(global_eval.getData())[:, :, 0].astype(complex)
        if global_eval.isComplex():
            values.imag = np.asarray(global_eval.getImagData())[:, :, 0]
        if parameter and not np.allclose(values[2], parameter[1], rtol=0, atol=1e-12):
            raise RuntimeError(f"{label}: решение привязано к другому значению {parameter[0]}")
        fields = np.asarray(interp.getData()).transpose(1, 0, 2).astype(complex)
        if interp.isComplex():
            fields.imag = np.asarray(interp.getImagData()).transpose(1, 0, 2)
        if not (values.shape[1] == len(fields) == len(inner)):
            raise RuntimeError(f"{label}: число полей, значений и идентификаторов различается")
        # A shared mask drops sampling points outside the computational domain.
        tables.append((values[0].real * 1e6, np.full(inner.shape, outer), inner, values[1], fields))

    wl, outer, inner, values, fields = map(np.concatenate, zip(*tables, strict=True))
    # Match returned wavelengths to the requested sweep to avoid round-off duplicates.
    nearest = np.abs(wl[:, None] - wavelengths_um).argmin(axis=1)
    expected = wavelengths_um[nearest]
    if not np.allclose(wl, expected, rtol=0, atol=1e-9) or len(np.unique(expected)) != len(wavelengths_um):
        raise RuntimeError(f"{label}: COMSOL вернул неполную или неожиданную сетку длин волн")
    valid_fields = np.isfinite(values) & np.isfinite(fields).any(axis=(1, 2))
    if valid_fields.any():
        mask = np.isfinite(fields[valid_fields]).all(axis=(0, 1))
        if mask.any():
            fields, coordinates = fields[:, :, mask], coordinates[:, mask]
    return ModalSolutions(expected, outer, inner, values, fields, coordinates)
