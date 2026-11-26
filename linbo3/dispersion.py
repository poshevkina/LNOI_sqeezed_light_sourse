"""Read effective-index exports with wavelengths in micrometres."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Mode:
    wavelength_um: np.ndarray
    neff: np.ndarray

    def omega_polynomial(self, degree: int = 4):
        """Return n_eff(omega); omega is in rad/fs."""
        from .interpolation import fit_dispersion
        return fit_dispersion(f"poly{degree}", self.wavelength_um, self.neff)

    def interpolator(self, *, extrapolate: bool = False):
        """Compatibility wavelength view of a polynomial fitted in omega."""
        from .frequency import um_to_omega
        poly = self.omega_polynomial(min(4, len(self.wavelength_um) - 1))

        def evaluate(wavelength_um):
            wl = np.asarray(wavelength_um)
            values = poly(um_to_omega(wl))
            if not extrapolate:
                values = np.where((wl >= self.wavelength_um[0]) & (wl <= self.wavelength_um[-1]),
                                  values, np.nan)
            return values
        return evaluate


def _read_table(path: Path) -> np.ndarray:
    table = np.loadtxt(path, comments=("#", "%"), ndmin=2, encoding="utf-8-sig")
    if table.shape[1] != 2 or not np.isfinite(table).all() or np.any(table[:, 0] <= 0):
        raise ValueError(f"{path}: expected two finite columns with positive wavelengths")
    return table


def _mode(table: np.ndarray, label: str) -> Mode:
    wavelength, neff = table.T
    if len(wavelength) < 2 or np.any(np.diff(wavelength) <= 0):
        raise ValueError(f"{label}: expected at least two strictly increasing wavelengths")
    return Mode(wavelength, neff)


def load_concatenated_modes(path: Path) -> dict[int, Mode]:
    """Assign mode numbers by block order. Each block has increasing wavelength."""
    table = _read_table(path)
    boundaries = np.flatnonzero(np.diff(table[:, 0]) <= 0) + 1
    return {
        index: _mode(block, f"{path} mode {index}")
        for index, block in enumerate(np.split(table, boundaries))
    }
