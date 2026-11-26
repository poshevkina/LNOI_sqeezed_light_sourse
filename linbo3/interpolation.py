"""Least-squares polynomials of Re(n_eff) in angular frequency, rad/fs."""

import numpy as np
from numpy.polynomial import Polynomial
from .frequency import um_to_omega


DEGREES = tuple(range(1, 7))
METHODS = tuple(f"poly{degree}" for degree in DEGREES)
LABELS = {f"poly{degree}": f"Полином {degree}" for degree in DEGREES}


def fit_dispersion(method, wavelength, neff):
    """Accept wavelength samples; return a Polynomial callable at omega (rad/fs)."""
    if method not in METHODS:
        raise ValueError(f"Выбери метод из {METHODS}; получен {method!r}")
    degree = int(method[4:])
    wavelength = np.asarray(wavelength, dtype=float)
    neff = np.asarray(neff)
    if (wavelength.ndim != 1 or neff.shape != wavelength.shape
            or not np.isfinite(wavelength).all() or np.any(wavelength <= 0)
            or not np.isfinite(neff).all() or np.iscomplexobj(neff)):
        raise ValueError("Нужны конечные вещественные n_eff и положительные длины волн")
    if len(wavelength) <= degree or len(np.unique(wavelength)) != len(wavelength):
        raise ValueError(f"Для степени {degree} нужны хотя бы {degree + 1} различных точек")
    # Keep the scaled domain/window: evaluation and deriv() include that scaling.
    return Polynomial.fit(um_to_omega(wavelength), neff, degree)


def leave_one_out(method, wavelength, neff):
    """Hold out each interior wavelength; fit and predict in omega without extrapolation."""
    wavelength, neff = np.asarray(wavelength), np.asarray(neff)
    degree = int(method[4:]) if method in METHODS else None
    if degree is None or len(wavelength) < degree + 2:
        raise ValueError(f"Для LOO метода {method} нужны хотя бы {(degree or 0) + 2} точки")
    order = np.argsort(wavelength)
    wavelength, neff = wavelength[order], neff[order]
    mask = np.ones(len(wavelength), dtype=bool)
    for index in range(1, len(wavelength) - 1):
        mask[index] = False
        omega = float(um_to_omega(wavelength[index]))
        predicted = float(fit_dispersion(method, wavelength[mask], neff[mask])(omega))
        mask[index] = True
        yield dict(wavelength_um=wavelength[index], omega_rad_per_fs=omega, observed=neff[index],
                   predicted=predicted, error=predicted - neff[index])
