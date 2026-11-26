"""Run polynomial SPDC comparisons and preserve their computed spectra."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import warnings

import numpy as np
import pandas as pd

from .storage import prune_cache, remove_owned
from .polynomial import DispersionStudy, width_label
from .spdc import calculate_spectrum, frequency_grid, omega_to_um, validate_wavelength_range


@dataclass(frozen=True)
class PulseConfig:
    t_fs: float
    dw_half: float
    n_points: int = 1501

    @property
    def tag(self) -> str:
        return f"t_{self.t_fs:g}fs_N{self.n_points}_DW{self.dw_half:g}"


@dataclass(frozen=True)
class SweepResult:
    pulse: PulseConfig
    cases: dict
    summary: pd.DataFrame


def _cache_path(folder, study, width, degree, pump_mode, pulse, length_m, save_jsa):
    digest = hashlib.sha256()
    settings = dict(
        version=2, fit_coordinate="omega_rad_per_fs", pulse=asdict(pulse),
        width=width, degree=degree, length_m=length_m,
        wavelength_um=study.pump_wavelength_um, save_jsa=save_jsa,
        pump_mode=pump_mode, signal_mode=study.signal_mode, idler_mode=study.idler_mode,
    )
    digest.update(json.dumps(settings, sort_keys=True).encode())
    for branch, block in (("PUMP", pump_mode), ("IS", study.signal_mode), ("IS", study.idler_mode)):
        mode = study.raw[width][branch][block]
        poly = study.fits[width][degree][branch][block]
        for values in (mode.wavelength_um, mode.neff, poly.coef, poly.domain, poly.window):
            digest.update(np.asarray([len(values)], dtype="<i8").tobytes())
            digest.update(np.asarray(values, dtype="<f8").tobytes())
    return folder / f"{pulse.tag}_d{degree}_p{pump_mode}_{digest.hexdigest()[:20]}.npz"


def _compute_case(study, width, degree, pump_mode, pulse, length_m, save_jsa):
    raw = study.raw[width]
    fits = study.fits[width][degree]
    spectrum = calculate_spectrum(
        raw["PUMP"][pump_mode], raw["IS"][study.signal_mode], raw["IS"][study.idler_mode],
        pump_wavelength_um=study.pump_wavelength_um, length_m=length_m,
        pulse_fwhm_fs=pulse.t_fs, half_window_rad_per_fs=pulse.dw_half, points=pulse.n_points,
        fits=(fits["PUMP"][pump_mode], fits["IS"][study.signal_mode], fits["IS"][study.idler_mode]),
    )
    jsi = spectrum.jsi
    grid = spectrum.angular_frequency_rad_per_fs
    step = grid[1] - grid[0]
    spectrum_s, spectrum_i = jsi.sum(axis=0) * step, jsi.sum(axis=1) * step
    border = (jsi[0, :].sum() + jsi[-1, :].sum()
              + jsi[1:-1, 0].sum() + jsi[1:-1, -1].sum()) / jsi.sum()
    payload = dict(
        IMatrix=jsi, w_grid=grid, lmb_grid=spectrum.wavelength_nm,
        S=spectrum.singular_values, normalized_eigenvalues=spectrum.schmidt_weights,
        K=spectrum.schmidt_number, Lambda_um=abs(spectrum.signed_poling_period_um),
        Lambda_signed_um=spectrum.signed_poling_period_um, delta_k0_per_m=spectrum.delta_k0_per_m,
        S_s=spectrum_s, S_i=spectrum_i, S_s_norm=spectrum_s / spectrum_s.max(),
        S_i_norm=spectrum_i / spectrum_i.max(), border_fraction=border,
    )
    if save_jsa:
        payload["fMatrix"] = spectrum.jsa
    return payload


def save_arrays(path, payload):
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_sweep(
    study: DispersionStudy,
    pulse: PulseConfig,
    output_dir: Path,
    *,
    length_m: float = 0.12,
    save_jsa: bool = False,
    reuse_cache: bool = True,
    progress: bool = True,
) -> SweepResult:
    grid = frequency_grid(study.pump_wavelength_um, pulse.dw_half, pulse.n_points)
    endpoints = grid[[0, -1]]
    for width, bands in study.raw.items():
        for band, modes, omega in (
            ("PUMP", study.pump_map[width].values(), 2 * endpoints),
            ("IS", {study.signal_mode, study.idler_mode}, endpoints),
        ):
            for mode in modes:
                validate_wavelength_range(bands[band][mode], omega_to_um(omega),
                                          f"{width_label(width)}, {band}[{mode}]")

    output_dir = Path(output_dir)
    cache = output_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    study.fit_errors.to_csv(output_dir / "fit_errors.csv", index=False, encoding="utf-8-sig")
    study.mode_matching.to_csv(output_dir / "mode_matching.csv", index=False, encoding="utf-8-sig")
    cases = {}
    rows = []
    expected=[]
    for width, pump_modes in study.pump_map.items():
        cases[width] = {}
        for degree in study.degrees:
            cases[width][degree] = {}
            for label, block in pump_modes.items():
                path = _cache_path(cache, study, width, degree, block, pulse, length_m, save_jsa)
                expected.append(path)
                cached = reuse_cache and path.is_file()
                if not cached:
                    save_arrays(path, _compute_case(study, width, degree, block, pulse, length_m, save_jsa))
                with np.load(path, allow_pickle=False) as stored:
                    result = {key: stored[key] for key in stored.files if key not in {"IMatrix", "fMatrix"}}
                result["cache_path"] = path
                result["pump_mode"] = block
                cases[width][degree][label] = result
                rows.append(dict(
                    t_fs=pulse.t_fs, width_um=width, degree=degree, pump_label=label, pump_mode=block,
                    signal_mode=study.signal_mode, idler_mode=study.idler_mode, N_POINTS=pulse.n_points,
                    DW=pulse.dw_half, **{k: float(result[k]) for k in (
                        "K", "Lambda_um", "Lambda_signed_um", "delta_k0_per_m", "border_fraction"
                    )},
                ))
                if progress:
                    note = " [кэш]" if cached else ""
                    print(f"{width_label(width)} | d={degree} | PUMP {label} | K={result['K']:.5f}{note}",
                          flush=True)
    prune_cache(cache,expected)
    # The sweep now represents one current pulse/window/grid configuration.
    for path in output_dir.glob('summary_t_*.csv'):remove_owned(path,output_dir)
    if (output_dir/'figures').exists():remove_owned(output_dir/'figures',output_dir)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / f"summary_{pulse.tag}.csv", index=False, encoding="utf-8-sig")
    if summary["border_fraction"].max() > 0.01:
        warnings.warn("Более 1% дискретной JSI находится на границе хотя бы одной сетки", stacklevel=2)
    return SweepResult(pulse, cases, summary)


def check_convergence(
    study: DispersionStudy,
    pulse: PulseConfig,
    *,
    width: float | None,
    degree: int,
    pump_label: str,
    point_counts: tuple[int, ...],
    length_m: float = 0.12,
) -> pd.DataFrame:
    """Compare K at fixed frequency half-window for a single polynomial case."""
    block = study.pump_map[width][pump_label]
    rows = []
    for points in point_counts:
        config = PulseConfig(pulse.t_fs, pulse.dw_half, points)
        result = _compute_case(study, width, degree, block, config, length_m, save_jsa=False)
        rows.append(dict(N_POINTS=points, DW=pulse.dw_half, K=result["K"],
                         border_fraction=result["border_fraction"]))
    return pd.DataFrame(rows)
