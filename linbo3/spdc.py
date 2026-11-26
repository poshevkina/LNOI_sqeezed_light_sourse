"""Low-gain SPDC on a uniform angular-frequency grid."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh, svdvals

from .dispersion import Mode
from .frequency import LIGHT_SPEED_M_PER_FS, omega_to_um, um_to_omega


def frequency_grid(pump_wavelength_um: float, half_window_rad_per_fs: float, points: int):
    parameters = (pump_wavelength_um, half_window_rad_per_fs)
    if not np.isfinite(parameters).all() or min(parameters) <= 0 or points < 2:
        raise ValueError("Wavelength and frequency half-window must be positive; use at least two points")
    center = np.pi * LIGHT_SPEED_M_PER_FS / (pump_wavelength_um * 1e-6)
    if half_window_rad_per_fs >= center:
        raise ValueError("Frequency window must stay strictly above zero")
    return np.linspace(center - half_window_rad_per_fs, center + half_window_rad_per_fs, points)


def validate_wavelength_range(mode: Mode, wavelength, label: str) -> None:
    lower, upper = mode.wavelength_um[[0, -1]]
    requested_min, requested_max = np.min(wavelength), np.max(wavelength)
    if requested_min < lower - 1e-12 or requested_max > upper + 1e-12:
        raise ValueError(
            f"{label}: requested {requested_min:.6g}–{requested_max:.6g} µm, "
            f"exported {lower:.6g}–{upper:.6g} µm"
        )


@dataclass(frozen=True)
class Spectrum:
    wavelength_nm: np.ndarray
    jsa: np.ndarray
    signed_poling_period_um: float
    schmidt_number: float
    angular_frequency_rad_per_fs: np.ndarray
    singular_values: np.ndarray
    schmidt_weights: np.ndarray
    delta_k0_per_m: float
    signal_modes: np.ndarray | None = None
    idler_modes: np.ndarray | None = None

    @property
    def jsi(self) -> np.ndarray:
        return np.abs(self.jsa) ** 2


def calculate_spectrum(
    pump: Mode,
    signal: Mode,
    idler: Mode,
    *,
    pump_wavelength_um: float = 0.775,
    length_m: float = 0.12,
    pulse_fwhm_fs: float = 2000.0,
    half_window_rad_per_fs: float = 0.002,
    points: int = 1501,
    extrapolate: bool = False,
    fits: tuple[Callable, Callable, Callable] | None = None,
    mode_count: int = 0,
) -> Spectrum:
    """Return JSA[idler, signal]; supplied fits accept omega in rad/fs, in PUMP/s/i order."""
    if mode_count < 0:
        raise ValueError("mode_count must be nonnegative")
    if not np.isfinite((length_m, pulse_fwhm_fs)).all() or min(length_m, pulse_fwhm_fs) <= 0:
        raise ValueError("Length and pulse duration must be positive")

    grid = frequency_grid(pump_wavelength_um, half_window_rad_per_fs, points)
    omega_pump = 2 * np.pi * LIGHT_SPEED_M_PER_FS / (pump_wavelength_um * 1e-6)
    if fits is None:
        fits = tuple(mode.omega_polynomial(min(4, len(mode.wavelength_um) - 1))
                     for mode in (pump, signal, idler))
    pump_fit, signal_fit, idler_fit = fits

    if not extrapolate:
        endpoints = grid[[0, -1]]
        validate_wavelength_range(pump, omega_to_um(2 * endpoints), "PUMP")
        validate_wavelength_range(signal, omega_to_um(endpoints), "signal")
        validate_wavelength_range(idler, omega_to_um(endpoints), "idler")

    def wavevector(fit: Callable, omega):
        return fit(omega) * omega / LIGHT_SPEED_M_PER_FS

    omega_i, omega_s = grid[:, None], grid[None, :]
    delta_k0 = float(
        wavevector(pump_fit, omega_pump)
        - wavevector(signal_fit, omega_pump / 2)
        - wavevector(idler_fit, omega_pump / 2)
    )
    mismatch = (wavevector(pump_fit, omega_s + omega_i)
                - wavevector(signal_fit, omega_s) - wavevector(idler_fit, omega_i) - delta_k0)
    tau_fs = pulse_fwhm_fs / (2 * np.sqrt(np.log(2)))
    delta = omega_pump - omega_s - omega_i
    amplitude = np.exp(-delta**2 * tau_fs**2 / 2) * np.sinc(mismatch * length_m / (2 * np.pi))
    jsa = amplitude * np.exp(1j * mismatch * length_m / 2)
    del delta, mismatch, amplitude

    signal_modes = idler_modes = None
    if mode_count:
        # JSA[omega_i, omega_s] = U @ diag(s) @ Vh.
        # Recover the requested rows of Vh without computing a second eigensystem.
        gram = jsa @ jsa.conj().T
        if not np.isfinite(gram).all():
            raise ValueError("The joint spectral amplitude has non-finite values")
        eigenvalues, vectors = eigh(gram, driver="evd", overwrite_a=True, check_finite=False)
        weights = np.maximum(eigenvalues[::-1], 0)
        singular = np.sqrt(weights)
        rank = np.count_nonzero(weights > weights[0] * np.finfo(float).eps * len(grid))
        count = min(mode_count, rank)
        idler_modes = vectors[:, ::-1][:, :count].copy()
        signal_modes = ((idler_modes.conj().T @ jsa) / singular[:count, None]).T.copy()
    else:
        singular = svdvals(jsa, overwrite_a=False)
        weights = singular**2
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("The joint spectral amplitude has zero or non-finite norm")
    probabilities = weights / total
    return Spectrum(
        wavelength_nm=omega_to_um(grid) * 1e3,
        jsa=jsa,
        signed_poling_period_um=2 * np.pi / delta_k0 * 1e6 if delta_k0 != 0 else np.inf,
        schmidt_number=float(1 / np.sum(probabilities**2)),
        angular_frequency_rad_per_fs=grid,
        singular_values=singular,
        schmidt_weights=probabilities,
        delta_k0_per_m=delta_k0,
        signal_modes=signal_modes,
        idler_modes=idler_modes,
    )
