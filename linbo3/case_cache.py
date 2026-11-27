"""Share SPDC numerical cases between studies, independent of their reports."""

import hashlib
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from .artifacts import digest_json, file_lock, read_json, save_arrays, write_json
from .frequency import LIGHT_SPEED_M_PER_FS
from .spectral_projections import sum_frequency_distribution


CASE_VERSION = 1
PREVIEW_POINTS = 129
STORAGE_VERSION = 2


def case_paths(results_root, modes, fits, settings, *, method, width):
    """Return portable public and local paths for an exact numerical case."""
    recipe = dict(version=CASE_VERSION, c_m_per_fs=LIGHT_SPEED_M_PER_FS,
                  fit_coordinate="omega_rad_per_fs", width_um=float(width), method=method,
                  pump_wavelength_um=float(settings["pump_wavelength_um"]),
                  length_m=float(settings["length_m"]), pulse_fwhm_fs=float(settings["pulse_fwhm_fs"]),
                  half_window_rad_per_fs=float(settings["half_window_rad_per_fs"]),
                  points=int(settings["points"]), mode_count=int(settings.get("mode_count", 0)),
                  extrapolate=bool(settings.get("extrapolate", False)))
    key = hashlib.sha256(digest_json(recipe).encode("ascii"))
    for mode, poly in zip(modes, fits, strict=True):
        for array in (mode.wavelength_um, mode.neff, poly.coef, poly.domain, poly.window):
            key.update(np.asarray([len(array)], dtype="<i8").tobytes())
            key.update(np.asarray(array, dtype="<f8").tobytes())
    case_key = key.hexdigest()
    root = Path(results_root)
    # The canonical location depends only on the computation, not report labels.
    label = (f"w{width:g}_{method}_{recipe['pulse_fwhm_fs']:g}fs_"
             f"N{recipe['points']}_DW{recipe['half_window_rad_per_fs']:g}__{case_key[:24]}")
    path = root / "cases" / label / "case.npz"
    full = root.parent / "jsa_full" / label / "full.npz"
    signature = dict(**recipe, case_key=case_key)
    return path, full, signature


def projections(grid, jsi):
    omega_sum, mass = sum_frequency_distribution(grid, jsi)
    step = grid[1] - grid[0]
    return dict(S_s=jsi.sum(axis=0) * step, S_i=jsi.sum(axis=1) * step,
                omega_sum=omega_sum, sum_frequency_mass=mass)


def validate_compact(case, points):
    preview_size = min(points, PREVIEW_POINTS)
    shapes = {
        "w_grid": (points,), "lmb_grid": (points,), "schmidt_weights": (points,),
        "S_s": (points,), "S_i": (points,),
        "preview_indices": (preview_size,), "jsa_preview": (preview_size, preview_size),
        "plot_jsi": (preview_size, preview_size), "omega_sum": (2 * points - 1,),
        "sum_frequency_mass": (2 * points - 1,), "K": (),
    }
    if ("IMatrix" in case or int(case["storage_version"]) != STORAGE_VERSION
            or any(np.shape(case[name]) != shape or not np.isfinite(case[name]).all()
                   for name, shape in shapes.items())):
        raise ValueError("Invalid compact SPDC case")
    expected = np.rint(np.linspace(0, points - 1, preview_size)).astype(int)
    if (not np.array_equal(case["preview_indices"], expected)
            or not np.isclose(case["sum_frequency_mass"].sum(), 1.0)
            or not np.isclose(case["schmidt_weights"].sum(), 1.0)
            or not np.isclose(case["K"], 1 / np.sum(case["schmidt_weights"]**2))):
        raise ValueError("Invalid preview grid or weights")


def _read_case(path, signature):
    metadata = path.with_name("case.json")
    if not path.is_file() or not metadata.is_file():
        return None
    try:
        if read_json(metadata)["case_key"] != signature["case_key"]:
            return None
        with np.load(path, allow_pickle=False) as file:
            public = {key: file[key] for key in file.files if key != "IMatrix"}
            if "IMatrix" in file:
                public.update(projections(public["w_grid"], file["IMatrix"]))
                public["storage_version"] = np.asarray(STORAGE_VERSION)
        validate_compact(public, signature["points"])
        return public
    except (OSError, ValueError, KeyError, BadZipFile):
        return None


def cache_case(path, full, signature, compute, *, reuse=True):
    """Share a completed numerical case, including across simultaneous studies."""
    if reuse and (public := _read_case(path, signature)) is not None:
        return public, True
    with file_lock(full.with_name(".case.lock")):
        if reuse and (public := _read_case(path, signature)) is not None:
            return public, True
        return _save_case(path, signature, compute()), False


def _save_case(path, signature, spectrum):
    grid = spectrum.angular_frequency_rad_per_fs
    jsa = spectrum.jsa
    jsi = spectrum.jsi
    projected = projections(grid, jsi)
    signal, idler = projected["S_s"], projected["S_i"]
    border = (jsi[0, :].sum() + jsi[-1, :].sum()
              + jsi[1:-1, 0].sum() + jsi[1:-1, -1].sum()) / jsi.sum()
    preview = np.rint(np.linspace(0, len(grid)-1, min(len(grid), PREVIEW_POINTS))).astype(int)
    public = dict(
        **projected, storage_version=np.asarray(STORAGE_VERSION),
        w_grid=grid, omega_rad_per_fs=grid, lmb_grid=spectrum.wavelength_nm,
        wavelength_nm=spectrum.wavelength_nm,
        S=spectrum.singular_values, singular_values=spectrum.singular_values,
        normalized_eigenvalues=spectrum.schmidt_weights, schmidt_weights=spectrum.schmidt_weights,
        K=spectrum.schmidt_number, Lambda_um=abs(spectrum.signed_poling_period_um),
        Lambda_signed_um=spectrum.signed_poling_period_um, delta_k0_per_m=spectrum.delta_k0_per_m,
        S_s_norm=signal / signal.max(), S_i_norm=idler / idler.max(),
        border_fraction=border, preview_indices=preview,
        preview_omega_signal_rad_per_fs=grid[preview],
        preview_omega_idler_rad_per_fs=grid[preview],
        jsa_preview=jsa[np.ix_(preview, preview)].astype(np.complex64),
        plot_omega_rad_per_fs=grid[preview], plot_jsi=jsi[np.ix_(preview, preview)],
    )
    if spectrum.signal_modes is not None:
        public.update(signal_modes=spectrum.signal_modes, idler_modes=spectrum.idler_modes)
    validate_compact(public, signature["points"])
    save_arrays(path, public)
    write_json(path.with_name("case.json"), signature)
    return public
