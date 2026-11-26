"""Plot effective-index samples and joint spectral intensity."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .dispersion import Mode
from .spdc import Spectrum


def save_dispersion(path: Path, modes: dict[int, Mode], band: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for mode_id, mode in modes.items():
        wavelength = np.linspace(mode.wavelength_um[0], mode.wavelength_um[-1], 500)
        line, = ax.plot(wavelength, mode.interpolator()(wavelength), label=f"Mode {mode_id}")
        ax.scatter(mode.wavelength_um, mode.neff, s=18, color=line.get_color())
    ax.set(xlabel="Wavelength (µm)", ylabel="Effective refractive index", title=band)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_jsi(path: Path, spectrum: Spectrum, pump_mode: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    wavelength = spectrum.wavelength_nm
    image = ax.pcolormesh(wavelength, wavelength, spectrum.jsi, cmap="inferno", shading="auto")
    ax.set(
        xlabel="Signal wavelength (nm)",
        ylabel="Idler wavelength (nm)",
        title=f"Pump mode {pump_mode} · K = {spectrum.schmidt_number:.4f}",
    )
    fig.colorbar(image, ax=ax, label="Joint spectral intensity")
    fig.savefig(path, dpi=150)
    plt.close(fig)
