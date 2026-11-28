"""Field maps and coupling figures for the 2D directional coupler."""

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Polygon
import numpy as np

from .analysis import ridge_polygons



def draw_cross_section(ax, outlines, film_y, *, dark):
    edge, halo = ("white", "#222222") if dark else ("#151515", "white")
    effects = [patheffects.withStroke(linewidth=2.5, foreground=halo)]
    for outline in outlines:
        ax.add_patch(Polygon(outline, fill=False, edgecolor=edge, linewidth=1.2,
                             path_effects=effects, zorder=4))
    ax.axhline(film_y, color=edge, linewidth=.8, linestyle="--", alpha=.8,
               path_effects=effects, zorder=3)


def plot_modes(path, modes, geometry, width, *, diagnostic=False):
    gap = modes[0].gap
    outlines = ridge_polygons(geometry, width, gap)
    fig, axes = plt.subplots(len(modes), 2, figsize=(9, 2.05 * len(modes) + .4),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    densities, components = [], []
    for item in modes:
        x, y, electric = item.profile.x_um, item.profile.y_um, item.profile.electric
        density = np.sum(abs(electric)**2, axis=0)
        area = (x[1] - x[0]) * (y[1] - y[0])
        scale = np.sqrt(np.nansum(density) * area)
        densities.append(density / scale**2)
        component = electric[np.argmax(item.fractions[:2])]
        amplitude = np.where(x[None, :] > 0, abs(component), np.nan)
        row, col = np.unravel_index(np.nanargmax(amplitude), amplitude.shape)
        components.append((component * np.exp(-1j*np.angle(component[row, col]))).real / scale)
    max_density = max(np.nanmax(density) for density in densities)
    max_amplitude = max(np.nanmax(abs(component)) for component in components)

    for row, (item, density, component) in enumerate(zip(modes, densities, components, strict=True)):
        x, y = item.profile.x_um, item.profile.y_um
        heat = axes[row, 0].pcolormesh(x, y, density / max_density, shading="nearest",
                                      cmap="inferno", vmin=0, vmax=1)
        signed = axes[row, 1].pcolormesh(x, y, component / max_amplitude, shading="nearest",
                                        cmap="RdBu_r", vmin=-1, vmax=1)
        parity = "even" if item.parity >= .9 else "odd" if item.parity <= -.9 else "mixed"
        phase = "in phase" if item.component_phase >= .9 else "opposite phase" if item.component_phase <= -.9 else "mixed phase"
        axis = ("x", "y")[np.argmax(item.fractions[:2])]
        axes[row, 0].set_ylabel("y (µm)")
        prefix = f"{item.polarization} · {item.key} · " if diagnostic else ""
        axes[row, 0].set_title(fr"{prefix}$|E|^2$ · vector-{parity} · $n_{{eff}}$={item.neff.real:.8f}",
                               fontsize=10 if diagnostic else None)
        axes[row, 1].set_title(fr"$Re(E_{axis})$ · {phase}")
        for col, ax in enumerate(axes[row]):
            draw_cross_section(ax, outlines, geometry.film_height_um, dark=col == 0)
            ax.set_aspect("equal")
            ax.set_xlim(-gap/2-width-.7, gap/2+width+.7)
            ax.set_ylim(-.55, 1.15)
    for ax in axes[-1]:
        ax.set_xlabel("x (µm)")
    fig.colorbar(heat, ax=axes[:, 0], label=r"Relative $|E|^2$")
    fig.colorbar(signed, ax=axes[:, 1], label=r"Relative $Re(E)$")
    label = "Low-loss localized candidates" if diagnostic else modes[0].polarization
    fig.suptitle(f"{label}   w = {width:g} µm   g = {gap:g} µm")
    fig.savefig(path, dpi=175)
    plt.close(fig)


def plot_coupling(output, pairs, wavelength_um):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), layout="constrained")
    transfer, transfer_axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True, layout="constrained")
    z = np.linspace(0, 200, 401)
    for i, (family, color) in enumerate((("quasi-TE", "C0"), ("quasi-TM", "C1"))):
        table = [row for row in pairs if row["family"] == family]
        gap = np.array([row["gap_um"] for row in table])
        delta = np.array([row["delta_neff"] for row in table])
        length = np.array([row["L_pi_um"] for row in table])
        measured = np.isfinite(delta) & (delta > 0)
        resolved = np.isfinite(length)
        axes[0].plot(gap, np.where(resolved, delta, np.nan), "o-", color=color, label=family)
        axes[0].scatter(gap[measured & ~resolved], delta[measured & ~resolved], edgecolors=color,
                        facecolors="none", marker="o")
        axes[1].plot(gap, length, "o-", color=color, label=family)
        for g, splitting in zip(gap[resolved], delta[resolved], strict=True):
            transfer_axes[i].plot(z, np.sin(np.pi * splitting * z / wavelength_um)**2, label=f"{g:g}")
        ax = transfer_axes[i]
        ax.set(title=family, xlabel="Propagation distance z (µm)", ylim=(0, 1.02))
        if resolved.any():
            ax.legend(title="g (µm)", ncol=2, fontsize=7)
    for ax, label in zip(axes, (r"$|\Delta\mathrm{Re}(n_{eff})|$", r"$L_\pi$ (µm)"), strict=True):
        ax.set(xlabel="Top gap g (µm)", ylabel=label, yscale="log")
        ax.legend()
        ax.grid(alpha=.2)
    transfer_axes[0].set_ylabel("Power fraction in the other guide")
    fig.suptitle("Open markers: splitting unresolved or awaiting verification")
    transfer.suptitle("Straight coupler · low-loss two-mode estimate")
    fig.savefig(output / "splitting_and_length.png", dpi=170)
    transfer.savefig(output / "power_transfer.png", dpi=170)
    plt.close(fig)
    plt.close(transfer)
