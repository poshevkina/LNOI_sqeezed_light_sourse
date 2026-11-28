"""COMSOL configuration and sectional flux extraction for the straight coupler."""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np

from ...comsol_session import open_model, solve_study
from ..paths import MATERIALS_DIR
from ..coupled.analysis import ridge_polygons
from ...mode_classification import read_cross_section
from .template import PORTS


SILICA = MATERIALS_DIR / "n_SiO2.txt"
LENGTH_UM = 200.
SLICE_UM = np.array([0.1, 50., 100., 150., 199.9])
AXIAL_UM = np.unique(np.r_[SLICE_UM, np.arange(0., LENGTH_UM+1, 2.)])
FIELD_SHAPE = (152, 125)  # Reference quadrature for reported power.
MAP_SHAPE = (256, 200)
SAMPLE_BATCH_POINTS = 150_000


def _planes(geometry, physics):
    vertices = np.asarray(geometry.getVertexCoord())
    planes = []
    for tag in PORTS:
        entities = physics.feature(tag).selection().entities(2)
        indices = [int(v)-1 for entity in entities for v in geometry.getAdj(2, 0, entity)]
        levels = vertices[2, indices]
        if not levels.size or np.ptp(levels) > 1e-10:
            raise ValueError(f"{tag}: порт не расположен в одной плоскости z")
        planes.append(float(levels[0]))
    if not np.allclose(planes[:2], planes[0], atol=1e-10, rtol=0) or not np.allclose(
            planes[2:], planes[2], atol=1e-10, rtol=0):
        raise ValueError("Обе супермоды должны лежать на одной плоскости каждого торца")
    if not np.isclose(abs(planes[2]-planes[0]), LENGTH_UM * 1e-6, atol=1e-10, rtol=0):
        raise ValueError(f"Неверные координаты входа и выхода: {planes}")
    return planes[0], planes[2]



def _check_ridge_vertices(geometry, shape, excitation, planes):
    vertices = np.asarray(geometry.getVertexCoord()).T
    corners = np.concatenate(ridge_polygons(shape, excitation.width_um, excitation.gap_um)) * 1e-6
    for plane in planes:
        transverse = vertices[np.isclose(vertices[:, 2], plane, atol=1e-10, rtol=0), :2]
        distances = np.linalg.norm(transverse[:, None, :] - corners[None, :, :], axis=2)
        if np.max(distances.min(axis=0)) > 1e-9:
            raise ValueError("Контуры сердечников 2D и 3D не совпадают на торце")


def _port_fields(component, excitation, folder):
    xv, yv = np.meshgrid(excitation.x_m, excitation.y_m, indexing="xy")
    for index, electric in enumerate(excitation.electric, 1):
        names = [f"linE{index}{axis}{part}" for axis in "xyz" for part in ("r", "i")]
        columns = np.column_stack([xv.ravel(), yv.ravel()] + [
            part.ravel() for axis in electric for part in (axis.real, axis.imag)
        ])
        data = folder / f"mode{index}.txt"
        np.savetxt(data, columns, fmt="%.14e")
        interp = component.func().create(f"linbo3_mode{index}", "Interpolation")
        interp.set("source", "file")
        interp.set("filename", str(data.resolve()))
        interp.set("struct", "spreadsheet")
        interp.set("nargs", "2")
        interp.set("funcs", [[name, str(i)] for i, name in enumerate(names, 1)])
        interp.set("argunit", ["m", "m"])
        interp.set("fununit", ["V/m"] * len(names))
        interp.set("extrap", "const")


def _set_ports(model, component, excitation, inlet, outlet, weights):
    direction = int(np.sign(outlet-inlet))
    physics = component.physics("ewbe")
    waves = physics.prop("WaveVector")
    for i, beta in enumerate(excitation.beta, 1):
        model.parameter(f"k0mode{i}", f"{beta:.16g}[1/m]")
        waves.set(f"k{i}", ["0", "0", f"{direction}*k0mode{i}"])

    for port_index, tag in enumerate(PORTS):
        i = port_index % 2
        incoming = port_index < 2
        outward_direction = -direction if incoming else direction
        sign = "+" if outward_direction > 0 else "-"
        expressions = [f"linE{i+1}{axis}r(x,y){sign}i*linE{i+1}{axis}i(x,y)" for axis in "xyz"]
        port = physics.feature(tag)
        port.set("E0", expressions)
        # The fixed propagation constant is a positive magnitude for either outward normal.
        port.set("beta", f"k0mode{i+1}")
        if incoming:
            port.set("Pin", f"{abs(weights[i])**2:.16g}[W]")
            port.set("Thetap", f"{np.angle(weights[i]):.16g}[rad]")
    return direction


def _data(java, solution, expression):
    result = java.result()
    dataset = result.dataset().create("linbo3data", "Solution")
    dataset.set("solution", solution)
    interp = result.numerical().create("linbo3slice", "Interp")
    interp.set("data", "linbo3data")
    interp.set("expr", expression)
    return interp


def _sample_grid(bounds, shape):
    x, y = [lo + (np.arange(n) + .5) * (hi-lo) / n
            for (lo, hi), n in zip(bounds[:2], shape, strict=True)]
    return x, y, np.stack(np.meshgrid(x, y)).reshape(2, -1)


def _extract(java, solution, bounds, inlet, direction):
    intensity = "+".join(f"abs(ewbe.E{axis})^2" for axis in "xyz")
    interp = _data(java, solution, [intensity, "ewbe.Poavz"])
    x, y, map_xy = _sample_grid(bounds, MAP_SHAPE)
    flux_x, flux_y, flux_xy = _sample_grid(bounds, FIELD_SHAPE)
    xy = np.concatenate((map_xy, flux_xy), axis=1)
    map_points = map_xy.shape[1]
    area = (flux_x[1]-flux_x[0]) * (flux_y[1]-flux_y[0])
    volume = np.empty((len(AXIAL_UM), len(y), len(x)), dtype=np.float32)
    poynting = np.empty_like(volume)
    power = np.empty((len(AXIAL_UM), 2))
    planes_per_batch = max(1, SAMPLE_BATCH_POINTS // xy.shape[1])
    for start in range(0, len(AXIAL_UM), planes_per_batch):
        batch = slice(start, start + planes_per_batch)
        z = inlet + direction * AXIAL_UM[batch] * 1e-6
        coords = np.vstack([np.tile(xy, len(z)), np.repeat(z, xy.shape[1])])
        interp.setInterpolationCoordinates(coords.tolist())
        data = np.asarray(interp.getData())[:, 0].reshape(2, len(z), xy.shape[1])
        if not np.isfinite(data).all():
            raise RuntimeError("В выборке 3D-поля есть точки вне расчётной области или пропуски")
        volume[batch] = data[0, :, :map_points].reshape(len(z), len(y), len(x))
        poynting[batch] = direction * data[1, :, :map_points].reshape(len(z), len(y), len(x))
        forward = direction * data[1, :, map_points:].reshape(len(z), len(flux_y), len(flux_x))
        power[batch, 0] = forward[:, :, flux_x < 0].sum(axis=(1, 2)) * area
        power[batch, 1] = forward[:, :, flux_x > 0].sum(axis=(1, 2)) * area

    return dict(x_um=x*1e6, y_um=y*1e6, z_um=AXIAL_UM, slice_um=SLICE_UM,
                power_w=power, electric_squared=volume[np.isin(AXIAL_UM, SLICE_UM)],
                electric_squared_volume=volume, longitudinal_flux_w_m2=poynting,
                inlet_z_um=np.array(inlet*1e6), direction=np.array(direction))


def solve_case(template, excitation, layers, folder, weights=None):
    folder = Path(folder)
    weights = excitation.weights if weights is None else weights
    label = f"3D | g={excitation.gap_um:g} | {excitation.family} | {layers} слоёв"
    with TemporaryDirectory(dir=folder) as sources, open_model(template, cores=4, label=label) as (client, model):
        java = model.java
        component = java.component("comp2")
        model.parameter("w", f"{excitation.width_um:g}[um]")
        model.parameter("g", f"{excitation.gap_um:g}[um]")
        model.parameter("lbd0", f"{excitation.wavelength_um:g}[um]")
        model.parameter("L", f"{LENGTH_UM:g}[um]")
        model.parameter("N_swept_mash", str(layers))
        java.func("int4").set("sourcetype", "user")
        java.func("int4").set("filename", str(SILICA))
        geometry = component.geom("geom2")
        geometry.run()
        bounds = np.asarray(geometry.getBoundingBox()).reshape(3, 2)
        for axis, lo_hi in zip((excitation.x_m, excitation.y_m), bounds[:2], strict=True):
            half_step = (axis[1]-axis[0])/2
            if not np.allclose(lo_hi, [axis[0]-half_step, axis[-1]+half_step], atol=1e-9, rtol=0):
                raise ValueError(f"Границы 3D и 2D сечений различаются: {lo_hi}")
        inlet, outlet = _planes(geometry, component.physics("ewbe"))
        shape = read_cross_section(Path(template).parent, {"model": str(Path(template).resolve())})
        _check_ridge_vertices(geometry, shape, excitation, (inlet, outlet))
        print(f"{label} | вход z={inlet*1e6:g}, выход z={outlet*1e6:g} мкм", flush=True)
        _port_fields(component, excitation, Path(sources))
        direction = _set_ports(model, component, excitation, inlet, outlet, weights)
        mesh = component.mesh("mesh2")
        start = monotonic()
        mesh.run()
        elements = int(mesh.getNumElem())
        solve_study(client, model, str(java.study("std4").label()), log=folder / "comsol.log", label=label)
        elapsed = monotonic() - start
        solutions = [tag for tag in java.sol().tags() if java.sol(tag).isAttached()]
        if len(solutions) != 1:
            raise RuntimeError(f"Ожидалось одно решение Frequency Domain, получено {solutions}")
        solution = solutions[0]
        dof = int(java.sol(solution).getSize()[0])
        print(f"{label} | выборка полей, {elements} элементов, {dof} степеней свободы", flush=True)
        arrays = _extract(java, solution, bounds, inlet, direction)
        return dict(arrays, elements=np.array(elements), dof=np.array(dof), elapsed_s=np.array(elapsed))
