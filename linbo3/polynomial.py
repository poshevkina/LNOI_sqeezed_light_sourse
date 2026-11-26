"""Polynomial dispersion fits and mode numbering for the waveguide study."""

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd

from .dispersion import Mode, load_concatenated_modes
from .frequency import um_to_omega
from .interpolation import DEGREES, fit_dispersion


WIDTH_FOLDER = re.compile(r"w\s*=\s*(\d+(?:[.,]\d+)?)")
FILENAMES = {"PUMP": "PUMP neff.txt", "IS": "IS neff.txt"}


def width_label(width: float | None) -> str:
    return f"w={width:g} мкм" if width is not None else "Локальная пара"


@dataclass(frozen=True)
class DispersionStudy:
    raw: dict[float | None, dict[str, dict[int, Mode]]]
    fits: dict[float | None, dict[int, dict[str, dict[int, Polynomial]]]]
    fit_errors: pd.DataFrame
    mode_matching: pd.DataFrame
    pump_map: dict[float | None, dict[str, int]]
    degrees: tuple[int, ...]
    signal_mode: int
    idler_mode: int
    pump_wavelength_um: float

    @property
    def widths(self) -> tuple[float | None, ...]:
        return tuple(self.raw)

    @property
    def pump_labels(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(label for modes in self.pump_map.values() for label in modes))


def load_study(
    base: Path,
    *,
    degrees: tuple[int, ...] = DEGREES,
    widths: tuple[float, ...] | None = None,
    local_width_um: float | None = None,
    signal_mode: int = 0,
    idler_mode: int = 1,
    pump_wavelength_um: float = 0.775,
) -> DispersionStudy:
    """Read paired TXT exports from width folders or one pair directly in base."""
    base = Path(base)
    if local_width_um is not None and (not np.isfinite(local_width_um) or local_width_um <= 0):
        raise ValueError("Ширина локальной пары должна быть положительной")

    folders = {}
    for folder in sorted(base.iterdir()):
        match = WIDTH_FOLDER.fullmatch(folder.name)
        if not folder.is_dir() or match is None:
            continue
        width = float(match[1].replace(",", "."))
        if widths is not None and width not in widths:
            continue
        if width <= 0 or width in folders:
            raise ValueError(f"{folder.name}: недопустимая или повторная ширина")
        folders[width] = folder
    if not folders and all((base / name).is_file() for name in FILENAMES.values()):
        if widths is None or local_width_um in widths:
            folders[local_width_um] = base
    if widths is not None and (absent := set(widths) - folders.keys()):
        raise FileNotFoundError(f"{base}: нет данных для ширин {sorted(absent)}")

    raw, incomplete = {}, []
    for width, folder in sorted(folders.items()):
        paths = {branch: folder / name for branch, name in FILENAMES.items()}
        if absent_files := [path.name for path in paths.values() if not path.is_file()]:
            incomplete.append(f"{folder.name}: {', '.join(absent_files)}")
            continue
        raw[width] = {branch: load_concatenated_modes(path) for branch, path in paths.items()}
    if incomplete:
        message = "Нет полной пары выгрузок в папках " + "; ".join(incomplete)
        if widths is not None:
            raise FileNotFoundError(message)
        warnings.warn(message, stacklevel=2)
    if not raw:
        raise FileNotFoundError(f"{base}: не найдены пары IS neff.txt и PUMP neff.txt")
    return fit_study(raw, degrees=degrees, signal_mode=signal_mode, idler_mode=idler_mode,
                     pump_wavelength_um=pump_wavelength_um)


def fit_study(
    raw: dict[float | None, dict[str, dict[int, Mode]]],
    *,
    degrees: tuple[int, ...] = DEGREES,
    signal_mode: int = 0,
    idler_mode: int = 1,
    pump_wavelength_um: float = 0.775,
) -> DispersionStudy:
    """Fit the already identified modes of each width."""
    if (not degrees or len(set(degrees)) != len(degrees)
            or any(type(d) is not int or d not in DEGREES for d in degrees)):
        raise ValueError("Выбери различные целые степени полинома от 1 до 6")
    fits, pump_map = {}, {}
    errors, matching = [], []
    for width, modes in raw.items():
        if missing_is := {signal_mode, idler_mode} - modes["IS"].keys():
            raise ValueError(f"{width_label(width)}: отсутствуют IS-моды {sorted(missing_is)}")
        fits[width] = {}
        for degree in degrees:
            fits[width][degree] = {"PUMP": {}, "IS": {}}
            for branch, blocks in modes.items():
                for block, mode in blocks.items():
                    wl = mode.wavelength_um
                    if len(wl) <= degree:
                        raise ValueError(f"{width_label(width)}, {branch}, блок {block}: мало точек для степени {degree}")
                    omega = um_to_omega(wl)
                    poly = fit_dispersion(f"poly{degree}", wl, mode.neff)
                    fits[width][degree][branch][block] = poly
                    residual = mode.neff - poly(omega)
                    errors.append(dict(
                        width_um=width, degree=degree, branch=branch, mode=block, points=len(wl),
                        lambda_min_um=wl[0], lambda_max_um=wl[-1],
                        fit_coordinate="omega_rad_per_fs", omega_min_rad_per_fs=omega.min(),
                        omega_max_rad_per_fs=omega.max(),
                        RMSE=np.sqrt(np.mean(residual**2)), max_error=np.max(np.abs(residual)),
                    ))

        ranked = []
        for block, mode in modes["PUMP"].items():
            if not mode.wavelength_um[0] <= pump_wavelength_um <= mode.wavelength_um[-1]:
                raise ValueError(f"{width_label(width)}, PUMP, блок {block}: нет данных при {pump_wavelength_um:g} мкм")
            ranked.append((block, float(np.interp(pump_wavelength_um, mode.wavelength_um, mode.neff))))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        if any(np.isclose(a[1], b[1], rtol=0, atol=1e-10) for a, b in zip(ranked, ranked[1:])):
            warnings.warn(f"{width_label(width)}: совпадают n_eff на длине волны накачки, сохранён порядок блоков",
                          stacklevel=2)
        pump_map[width] = {}
        for rank, (block, index) in enumerate(ranked, start=1):
            label = f"мода {rank}"
            pump_map[width][label] = block
            matching.append(dict(width_um=width, pump_label=label, pump_rank=rank,
                                 pump_mode=block, neff_at_pump=index))

    return DispersionStudy(raw, fits, pd.DataFrame(errors), pd.DataFrame(matching),
                           pump_map, degrees, signal_mode, idler_mode, pump_wavelength_um)
