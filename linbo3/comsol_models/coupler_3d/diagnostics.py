"""Field, power and mesh diagnostics derived from saved 3D coupler samples."""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from ...artifacts import write_csv
from ..coupled.analysis import ridge_polygons
from ..coupled.plots import draw_cross_section


def _finish(fig, path):
    try:
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def _fraction(data):
    return data["power_w"][:, 1] / data["power_w"].sum(axis=1)


def plot_power(output, groups, target_layers):
    available = [(family, cases) for family, cases in groups.items() if cases]
    if not available:
        return
    fig, axes = plt.subplots(2, len(available), figsize=(5.5*len(available), 6.4),
                             squeeze=False, layout="constrained")
    balance, balances = plt.subplots(2, len(available), figsize=(5.5*len(available), 6.4),
                                   squeeze=False, layout="constrained")
    for col, (family, cases) in enumerate(available):
        finest, (case, reference) = max(cases.items())
        selected = target_layers if target_layers in cases else finest
        selected_data = cases[selected][1]
        for layers, (_, data) in sorted(cases.items()):
            z = data["z_um"]
            cross = _fraction(data)
            axes[0, col].plot(z, cross*100, label=f"{layers} layers")
            if layers != finest:
                mask = (z >= reference["z_um"][0]) & (z <= reference["z_um"][-1])
                baseline = np.interp(z[mask], reference["z_um"], _fraction(reference))
                axes[1, col].plot(z[mask], (cross[mask]-baseline)*100, label=f"{layers} − {finest}")
            power = data["power_w"].sum(axis=1)
            entrance = int(np.argmin(abs(z-.1)))
            balances[1, col].plot(z, (power/power[entrance]-1)*100, label=f"{layers} layers")
        z = reference["z_um"]
        ideal = np.sin(np.diff(case.excitation.beta)[0]*z*1e-6/2)**2
        axes[0, col].plot(z, ideal*100, "--", color=".25", label="2D prediction")
        axes[0, col].set(title=family, ylabel="Cross power (%)")
        axes[1, col].set(ylabel=f"Difference vs {finest} layers (pp)", xlabel="Distance from input (µm)")
        powers = selected_data["power_w"]
        for y, label in zip((*powers.T, powers.sum(axis=1)), ("Left", "Right", "Total"), strict=True):
            balances[0, col].plot(selected_data["z_um"], y, label=label)
        balances[0, col].set(title=f"{family} · {selected} layers", ylabel="Power (W)")
        balances[1, col].set(ylabel="Total flux change from input (%)", xlabel="Distance from input (µm)")
        for ax in (*axes[:, col], *balances[:, col]):
            ax.grid(alpha=.22)
            if ax.lines:
                ax.legend(fontsize=8)
    _finish(fig, output/"transfer_and_mesh_difference.png")
    _finish(balance, output/"power_balance.png")

    common = [(family, cases[target_layers][1]) for family, cases in available if target_layers in cases]
    if len(common) == 2:
        (first, a), (second, b) = common
        z = a["z_um"]
        cross_a = _fraction(a)
        cross_b = np.interp(z, b["z_um"], _fraction(b))
        fig, axes = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True, layout="constrained")
        for power, family in ((cross_a, first), (cross_b, second)):
            axes[0].plot(z, np.where(power > 0, power*100, np.nan), label=family)
        axes[0].set(yscale="log", ylabel="Cross power (%)", title=f"Same axial mesh · {target_layers} layers")
        axes[0].legend()
        axes[1].plot(z, (cross_a-cross_b)*100)
        axes[1].axhline(0, color=".5", linewidth=.8)
        axes[1].set(ylabel=f"{first} − {second} (pp)", xlabel="Distance from input (µm)")
        _finish(fig, output/"polarization_selectivity.png")


def _grid_steps(x, y):
    return float(x[1]-x[0]), float(y[1]-y[0])


def plot_volume_details(output, case, data, volume, geometry):
    intensity = volume["electric_squared_volume"]
    flux = volume["longitudinal_flux_w_m2"]
    x, y, distance = (data[key] for key in ("x_um", "y_um", "z_um"))
    dx, dy = _grid_steps(x, y)
    def projected(array):
        return array.sum(axis=1, dtype=np.float64)*dy, array.sum(axis=2, dtype=np.float64)*dx

    ix, iy = projected(intensity)
    fx, fy = projected(flux)
    peak = max(float(intensity.max()), np.finfo(np.float32).tiny)
    extent = max(float(np.max(abs(fx))), float(np.max(abs(fy))), np.finfo(np.float32).tiny)
    density_extent = max(float(ix.max()), float(iy.max()), np.finfo(np.float32).tiny)
    outlines = ridge_polygons(geometry, case.excitation.width_um, case.excitation.gap_um)
    indices = np.unique(np.rint(np.linspace(0, len(distance)-1, 12)).astype(int))
    atlas, axes = plt.subplots(3, 4, figsize=(12, 6.8), sharex=True, sharey=True, layout="constrained")
    for index, ax in zip(indices, axes.flat):
        im = ax.pcolormesh(x, y, intensity[index]/peak, shading="nearest", cmap="inferno",
                           norm=LogNorm(vmin=1e-5, vmax=1.))
        draw_cross_section(ax, outlines, geometry.film_height_um, dark=True)
        ax.set(title=f"{distance[index]:g} µm", xlim=(-3, 3), ylim=(-.8, 1.4), aspect="equal")
    for ax in list(axes.flat)[len(indices):]:
        ax.set_visible(False)
    atlas.supxlabel("x (µm)")
    atlas.supylabel("y (µm)")
    atlas.colorbar(im, ax=axes, label=r"Relative $|E|^2$")
    _finish(atlas, output/"transverse_atlas.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, layout="constrained")
    for row, (axis, im_data, flux_data, coordinate) in enumerate(((x, ix, fx, "x"), (y, iy, fy, "y"))):
        heat = axes[row, 0].pcolormesh(distance, axis, im_data.T/density_extent, shading="nearest",
                                      cmap="inferno", norm=LogNorm(vmin=1e-5, vmax=1.))
        signed = axes[row, 1].pcolormesh(distance, axis, flux_data.T/extent, shading="nearest",
                                        cmap="RdBu_r", norm=SymLogNorm(.01, vmin=-1., vmax=1.))
        axes[row, 0].set_ylabel(f"{coordinate} (µm)")
        axes[row, 1].set_ylabel(f"{coordinate} (µm)")
    axes[0, 0].set_title(r"Projected $|E|^2$")
    axes[0, 1].set_title("Projected longitudinal flux")
    fig.supxlabel("Distance from input (µm)")
    fig.colorbar(heat, ax=axes[:, 0], label="Relative intensity")
    fig.colorbar(signed, ax=axes[:, 1], label="Relative signed flux")
    _finish(fig, output/"power_and_intensity_projections.png")

    weight = ix.sum(axis=1)*dx
    centroid_x = np.divide(ix @ x * dx, weight, out=np.full(len(weight), np.nan), where=weight > 0)
    centroid_y = np.divide(iy @ y * dy, weight, out=np.full(len(weight), np.nan), where=weight > 0)
    square = np.einsum("ijk,ijk->i", intensity, intensity, dtype=np.float64)*dx*dy
    area = np.divide(weight**2, square, out=np.full(len(weight), np.nan), where=square > 0)
    # Negative local flux includes interference and is not a port reflectance.
    backward = np.sum(np.maximum(-flux, 0), axis=(1, 2), dtype=np.float64)
    absolute = np.sum(np.abs(flux), axis=(1, 2), dtype=np.float64)
    backward_share = np.divide(backward, absolute, out=np.full(len(weight), np.nan), where=absolute > 0)
    dense_power = np.column_stack([fx[:, x < 0].sum(axis=1), fx[:, x > 0].sum(axis=1)])*dx*1e-12
    reference = data["power_w"]
    dense_total = dense_power.sum(axis=1)
    dense_fraction = np.divide(dense_power[:, 1], dense_total, out=np.full(len(weight), np.nan),
                               where=dense_total != 0)
    flux_error = dense_power-reference
    write_csv(output/"spatial_metrics.csv", pd.DataFrame(dict(
        distance_um=distance, intensity_centroid_x_um=centroid_x, intensity_centroid_y_um=centroid_y,
        intensity_effective_area_um2=area, negative_local_flux_fraction=backward_share,
        dense_left_w=dense_power[:, 0], dense_right_w=dense_power[:, 1],
        reference_left_w=reference[:, 0], reference_right_w=reference[:, 1],
        cross_fraction_difference=dense_fraction-_fraction(data),
    )))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, layout="constrained")
    axes[0, 0].plot(distance, centroid_x, label="x")
    axes[0, 0].plot(distance, centroid_y, label="y")
    axes[0, 0].set(ylabel="Intensity centroid (µm)")
    axes[0, 0].legend()
    axes[0, 1].plot(distance, area)
    axes[0, 1].set(ylabel="Intensity effective area (µm²)")
    axes[1, 0].plot(distance, backward_share*100)
    axes[1, 0].set(ylabel="Negative local flux (% of absolute flux)")
    axes[1, 1].plot(distance, flux_error[:, 0], label="Left")
    axes[1, 1].plot(distance, flux_error[:, 1], label="Right")
    axes[1, 1].set(ylabel="Dense minus reference integration (W)")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=.22)
    fig.supxlabel("Distance from input (µm)")
    _finish(fig, output/"field_and_flux_checks.png")


def plot_slice_comparison(output, first, last, geometry):
    lower, a = first
    upper, b = last
    distances = np.intersect1d(a["slice_um"], b["slice_um"])
    x = b["x_um"][(b["x_um"] >= a["x_um"][0]) & (b["x_um"] <= a["x_um"][-1])]
    y = b["y_um"][(b["y_um"] >= a["y_um"][0]) & (b["y_um"] <= a["y_um"][-1])]
    if not len(distances) or min(len(x), len(y)) < 2:
        return
    yy, xx = np.meshgrid(y, x, indexing="ij")
    area = (x[1]-x[0])*(y[1]-y[0])
    outlines = ridge_polygons(geometry, upper.excitation.width_um, upper.excitation.gap_um)
    comparisons = []
    for distance in distances:
        def shape(data):
            i = np.flatnonzero(data["slice_um"] == distance)[0]
            field = RegularGridInterpolator((data["y_um"], data["x_um"]), data["electric_squared"][i])((yy, xx))
            return field / (field.sum()*area)

        p, q = shape(a), shape(b)
        comparisons.append((p, q, q-p))
    common_max = max(max(p.max(), q.max()) for p, q, _ in comparisons)
    residual_max = max(float(np.finfo(float).eps), max(np.max(abs(d)) for _, _, d in comparisons))
    fig, axes = plt.subplots(len(distances), 3, figsize=(10, 1.9*len(distances)+.8),
                             squeeze=False, sharex=True, sharey=True, layout="constrained")
    rows = []
    for row, (distance, (p, q, diff)) in enumerate(zip(distances, comparisons, strict=True)):
        for col, density in enumerate((p, q)):
            im = axes[row, col].pcolormesh(x, y, density/common_max, cmap="inferno",
                                          shading="nearest", vmin=0., vmax=1.)
        delta = axes[row, 2].pcolormesh(x, y, diff/common_max, cmap="RdBu_r", shading="nearest",
                                      vmin=-residual_max/common_max, vmax=residual_max/common_max)
        axes[row, 0].set_ylabel(f"{distance:g} µm\ny (µm)")
        for col, ax in enumerate(axes[row]):
            draw_cross_section(ax, outlines, geometry.film_height_um, dark=col < 2)
            ax.set(xlim=(-3, 3), ylim=(-.8, 1.4), aspect="equal")
        rows.append(dict(distance_um=distance, coarse_layers=lower.layers, fine_layers=upper.layers,
                         shape_l1=float(abs(diff).sum()*area),
                         intensity_shape_similarity=float(np.sqrt(p*q).sum()*area)))
    for ax, label in zip(axes[0], (f"{lower.layers} layers", f"{upper.layers} layers",
                                  f"{upper.layers} − {lower.layers}"), strict=True):
        ax.set_title(label)
    for ax in axes[-1]:
        ax.set_xlabel("x (µm)")
    fig.colorbar(im, ax=axes[:, :2], label="Relative normalized intensity")
    fig.colorbar(delta, ax=axes[:, 2], label="Relative signed difference")
    _finish(fig, output/"transverse_mesh_difference.png")
    write_csv(output/"transverse_mesh_difference.csv", pd.DataFrame(rows))
