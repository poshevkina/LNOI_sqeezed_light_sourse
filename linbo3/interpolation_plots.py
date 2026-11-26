"""Disk figures for the dispersion and Schmidt-mode comparisons."""

from matplotlib import pyplot as plt
import numpy as np

from .interpolation import LABELS, METHODS
from .frequency import um_to_omega


def _save(fig, path):
    try:
        fig.savefig(path, dpi=140)
    finally:
        plt.close(fig)


def plot_raw(path, width, raw, branches, *, imag_limit=1e-6, anchors=None):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for row, band in enumerate(("PUMP", "IS")):
        table = raw.get(band)
        for col, column in enumerate(("neff_real", "neff_imag")):
            ax = axes[row, col]
            ax.set(xlabel="Длина волны, мкм", ylabel=("Re" if col == 0 else "Im") + "(n_eff)", title=band)
            if table is None:
                ax.text(.5, .5, "Нет выгрузки", ha="center", transform=ax.transAxes)
                continue
            ax.scatter(table.wavelength_um, table[column], s=10, color="0.65", label="COMSOL")
            if "eligible" in table:
                good = table.loc[table.eligible]
                ax.scatter(good.wavelength_um, good[column], s=12, color="#426b34",
                           label="Проходят порог и проверку поля")
            tracked = branches.get(band)
            if tracked is not None:
                for mode, branch in tracked.groupby("mode"):
                    ax.plot(branch.wavelength_um, branch[column], ".-", label=f"Ветвь {mode}")
            if col == 1:
                ax.set_yscale("symlog", linthresh=1e-9)
                ax.axhline(imag_limit, color="crimson", ls="--", lw=1, label="Границы отбора (строго)")
                ax.axhline(-imag_limit, color="crimson", ls="--", lw=1)
            if anchors is not None:
                ax.axvline(anchors[band], color="black", ls=":", lw=1, label="Опорная λ")
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    fig.suptitle(f"Ширина {width:g} мкм · решения COMSOL")
    _save(fig, path)


def plot_fits(path, width, bands, fits):
    entries = [(band, mode, data) for band, modes in bands.items() for mode, data in modes.items()]
    if not entries:
        return
    fig, axes = plt.subplots((len(entries) + 1) // 2, 2, figsize=(13, 3.4 * ((len(entries) + 1) // 2)),
                             squeeze=False, layout="constrained")
    for ax, (band, mode, data) in zip(axes.ravel(), entries):
        x = um_to_omega(data.wavelength_um)
        dense = np.linspace(x.min(), x.max(), 600)
        ax.scatter(x, data.neff, s=10, color="black", zorder=10, label="COMSOL")
        for method in METHODS:
            ax.plot(dense, fits[band][mode][method](dense), lw=1.1, label=LABELS[method])
        ax.set(title=f"{band} · ветвь {mode}", xlabel="ω, рад/фс", ylabel="Re(n_eff)")
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.grid(alpha=.2)
    for ax in axes.ravel()[len(entries):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=8)
    fig.suptitle(f"Ширина {width:g} мкм · аппроксимации")
    _save(fig, path)


def plot_derivatives(path, width, bands, fits):
    """Show each tracked branch and its first two analytical omega derivatives."""
    entries = [(band, mode, data) for band, modes in bands.items() for mode, data in modes.items()]
    if not entries:
        return
    labels = ("Re(n_eff)", "dn_eff/dω, фс/рад", "d²n_eff/dω², фс²/рад²")
    fig, axes = plt.subplots(len(entries), 3, figsize=(15, 2.8 * len(entries)),
                             squeeze=False, layout="constrained")
    for row, (band, mode, data) in enumerate(entries):
        omega = um_to_omega(data.wavelength_um)
        dense = np.linspace(omega.min(), omega.max(), 600)
        for order, ax in enumerate(axes[row]):
            for method in METHODS:
                ax.plot(dense, fits[band][mode][method].deriv(order)(dense), lw=1.15, label=LABELS[method])
            if order == 0:
                ax.scatter(omega, data.neff, s=9, color="black", zorder=10)
            ax.set(xlabel="ω, рад/фс", ylabel=labels[order], title=f"{band} · ветвь {mode}")
            ax.ticklabel_format(axis="y", useOffset=False)
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=6, fontsize=8)
    fig.suptitle(f"Ширина {width:g} мкм · дисперсия и производные по ω")
    _save(fig, path)


def plot_schmidt(output, title, cases):
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), layout="constrained", sharex=True)
    for method in METHODS:
        case = cases[method]
        omega = case["omega_rad_per_fs"]
        delta = omega - (omega[0] + omega[-1]) / 2
        spacing = omega[1] - omega[0]
        for col, role in enumerate(("signal", "idler")):
            vectors = case[f"{role}_modes"]
            for idx in range(vectors.shape[1]):
                axes[idx, col].plot(delta, np.abs(vectors[:, idx]) ** 2 / spacing,
                                    lw=1.15, label=LABELS[method])
    for idx, row in enumerate(axes, start=1):
        for col, ax in enumerate(row):
            ax.set_title(f"{('Signal', 'Idler')[col]} · мода {idx}")
            ax.set_ylabel("Плотность, фс/рад")
            ax.grid(alpha=.2)
    for ax in axes[-1]:
        ax.set_xlabel("Отстройка частоты, рад/фс")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=8)
    fig.suptitle(title)
    _save(fig, output / "schmidt_modes.png")

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for method in METHODS:
        weights = cases[method]["schmidt_weights"][:30]
        ax.semilogy(np.arange(1, len(weights) + 1), np.where(weights > 0, weights, np.nan),
                    ".-", label=f"{LABELS[method]} · K={float(cases[method]['K']):.3f}")
    ax.set(title=title, xlabel="Номер моды", ylabel="Нормированный вес")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    _save(fig, output / "schmidt_weights.png")


def plot_jsi_comparison(output, title, cases):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained", sharex=True, sharey=True)
    for ax, method in zip(axes.ravel(), METHODS):
        case = cases[method]
        grid = case["plot_omega_rad_per_fs"]
        detuning = grid - (grid[0] + grid[-1]) / 2
        im = ax.pcolormesh(detuning, detuning, case["plot_jsi"], cmap="inferno",
                           vmin=0, vmax=1, shading="auto")
        ax.set(title=f"{LABELS[method]} · K={float(case['K']):.4f}",
               xlabel="Signal Δω, рад/фс", ylabel="Idler Δω, рад/фс")
        ax.label_outer()
    fig.colorbar(im, ax=axes.ravel()[:len(METHODS)], label="JSI")
    fig.suptitle(title)
    _save(fig, output / "jsi_comparison.png")
