"""Shared frequency convention: wavelengths in µm, angular frequency in rad/fs."""

import numpy as np

# Preserve the rounded value used in the original calculations.
LIGHT_SPEED_M_PER_FS = 3e-7


def um_to_omega(wavelength_um):
    return 2 * np.pi * LIGHT_SPEED_M_PER_FS / (np.asarray(wavelength_um) * 1e-6)


def omega_to_um(omega):
    return 2 * np.pi * LIGHT_SPEED_M_PER_FS / np.asarray(omega) * 1e6
