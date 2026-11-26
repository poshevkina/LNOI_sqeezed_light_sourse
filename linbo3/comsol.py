"""Run the project's COMSOL Mode Analysis and extract indexed modal solutions."""

from contextlib import contextmanager
from pathlib import Path
import sys
from threading import Event, Thread
from time import monotonic

import numpy as np

from .modes import ModalSolutions


@contextmanager
def _progress(client, log: Path, label: str, interval: float = 10.0):
    finished = Event()
    start = monotonic()

    def watch():
        position = 0
        while True:
            done = finished.wait(interval)
            if not done:
                print(f"{label} | решатель работает {monotonic() - start:.0f} с", flush=True)
            try:
                with log.open("rb") as stream:
                    stream.seek(position)
                    lines = [line.decode("utf-8", errors="replace").strip() for line in stream]
                    position = stream.tell()
                progress = [line for line in lines if "Current Progress:" in line or "Parameter lbd =" in line]
                warnings = [line for line in lines if line.startswith(("Warning:", "Error:"))]
                for line in (*progress[-4:], *warnings[-4:]):
                    encoding = sys.stdout.encoding or "utf-8"
                    printable = line.encode(encoding, errors="backslashreplace").decode(encoding)
                    print(f"  COMSOL | {printable}", flush=True)
            except OSError:
                pass
            if done:
                break

    try:
        client.java.showProgress(str(log.resolve()))
    except Exception as error:
        print(f"{label} | недоступен журнал COMSOL: {error}", flush=True)
    thread = Thread(target=watch, daemon=True)
    thread.start()
    try:
        yield
    finally:
        finished.set()
        thread.join()
        try:
            client.java.showProgress(False)
        except Exception:
            pass


def _import_project_materials(model,model_file):
    """Override historical file addresses in MPH with local relative filenames."""
    directory=Path(model_file).resolve().parent
    tables={'int1':'n_SiO2.txt','int2':'no_LiNbO3.txt','int3':'ne_LiNbO3.txt'}
    for filename in tables.values():
        if not (directory/filename).is_file():raise FileNotFoundError(directory/filename)
    model.java.modelPath(str(directory))
    for tag,filename in tables.items():
        function=model.java.component('comp1').func(tag)
        function.set('filename',filename)
        function.importData()


@contextmanager
def _solved_model(path, wavelengths_um, width_um, geometry_parameters, modes, cores,
                  *, label="", log=None):
    import mph

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

    if sys.platform == "win32":
        mph.option("session", "stand-alone")
    if label:
        print(f"{label} | загрузка модели", flush=True)
    client = mph.start(cores=cores)
    model = client.load(path)
    try:
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
        if log is None:
            model.solve("Study 1")
        else:
            print(f"{label} | решение COMSOL, {len(wavelengths_um)} длин волн", flush=True)
            with _progress(client, log, label):
                model.solve("Study 1")
        yield model
    finally:
        client.remove(model)


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
    from jpype import JArray, JInt

    with _solved_model(path, wavelengths_um, width_um, geometry_parameters, modes, cores,
                       log=log, label=label) as model:
        print(f"{label} | выгрузка решений и полей", flush=True)
        geom = model.java.component("comp1").geom("geom1")
        bounds = np.asarray(geom.getBoundingBox()).reshape(2, 2)
        nx, ny = field_grid
        axes = [limits[0] + (np.arange(n) + 0.5) / n * np.diff(limits)[0]
                for limits, n in zip(bounds, (nx, ny), strict=True)]
        coordinates = np.stack(np.meshgrid(*axes, indexing="ij")).reshape(2, -1)

        solution_info = model.java.sol("sol1").getSolutioninfo()
        interp = model.java.result().numerical().create("linbo3field", "Interp")
        interp.set("data", "dset1")
        interp.set("expr", ["comp1.Ex", "comp1.Ey", "comp1.Ez"])
        interp.setInterpolationCoordinates(coordinates.tolist())
        global_eval = model.java.result().numerical().create("linbo3global", "Global")
        global_eval.set("data", "dset1")
        global_eval.set("expr", ["lbd", "emw.neff"])

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
