"""Launch one guide of the straight 3D LNOI coupler with a pair of 2D supermodes."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    # Direct execution puts the studies directory on the import path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import numpy as np
import pandas as pd

from linbo3.artifacts import atomic_path, read_json, relative_path, write_csv, write_json
from linbo3.comsol_models.coupled.analysis import ridge_polygons
from linbo3.comsol_models.coupled.plots import draw_cross_section
from linbo3.comsol_models.coupler_3d.cache import CouplerCase
from linbo3.comsol_models.coupler_3d.inputs import load_excitation, validate_materials
from linbo3.comsol_models.coupler_3d.template import TEMPLATE, build_template
from linbo3.mode_classification import read_cross_section
from linbo3.study_output import publish_study


PROJECT = Path(__file__).resolve().parents[1]
FAMILIES = ("quasi-TE", "quasi-TM")
PILOT_GAP = .8
TARGET_GAP = 1.6
INPUT_LIMIT = .95
MESH_TOLERANCE = .02


def case_metrics(data):
    powers = data["power_w"]
    total = powers.sum(axis=1)
    if not np.isfinite(total).all() or np.min(total) <= 0:
        raise ValueError("Продольный поток должен быть положителен от входа к выходу")
    incoming = int(np.argmin(abs(data["z_um"]-.1)))
    return dict(left_in=float(powers[incoming, 0]/total[incoming]),
                cross_out=float(powers[-1, 1]/total[-1]), power_in_w=float(total[incoming]),
                power_out_w=float(total[-1]), power_drift=float(np.max(abs(total/total[incoming]-1))),
                layers=None, elements=int(data["elements"]), dof=int(data["dof"]),
                elapsed_s=float(data["elapsed_s"]))


def plot_case(output, case, data, geometry):
    excitation = case.excitation
    x, y, z = data["x_um"], data["y_um"], data["z_um"]
    powers = data["power_w"]
    _, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    left, right = powers.T
    ideal = np.sin((excitation.beta[0]-excitation.beta[1]) * z * 1e-6/2)**2
    axes[0].plot(z, right/(left+right), label="COMSOL 3D")
    axes[0].plot(z, ideal, "--", label="2D two-mode prediction")
    axes[0].set(ylabel="Cross power fraction", ylim=(0, 1.03))
    axes[1].plot(z, left, label="Left half")
    axes[1].plot(z, right, label="Right half")
    axes[1].plot(z, left+right, ":", color="0.25", label="Net flux")
    axes[1].set(ylabel="Power (W)")
    for ax in axes:
        ax.set_xlabel("Distance from input (µm)")
        ax.legend()
    stem = f"g_{excitation.gap_um:g}_{excitation.family}_{case.layers}layers"
    axes[0].figure.savefig(output / f"{stem}_power.png", dpi=165)
    plt.close(axes[0].figure)
    write_csv(output / f"{stem}_power.csv", pd.DataFrame(
        dict(z_um=z, left_w=left, right_w=right, cross_fraction=right/(left+right), prediction_2d=ideal)))

    slices = data["electric_squared"]
    fig, axes = plt.subplots(len(slices), 1, figsize=(6.5, 2*len(slices)), sharex=True,
                             squeeze=False, layout="constrained")
    outlines = ridge_polygons(geometry, excitation.width_um, excitation.gap_um)
    limit = slices.max()
    for ax, image, distance in zip(axes[:, 0], slices, data["slice_um"], strict=True):
        color = ax.pcolormesh(x, y, image / limit, shading="nearest", cmap="inferno", vmin=0., vmax=1.)
        draw_cross_section(ax, outlines, geometry.film_height_um, dark=True)
        ax.set(xlim=(-excitation.gap_um/2-1.2, excitation.gap_um/2+1.2), ylim=(-.65, 1.25),
               ylabel="y (µm)", title=f"Distance from input {distance:g} µm", aspect="equal")
    axes[-1, 0].set_xlabel("x (µm)")
    fig.colorbar(color, ax=axes[:, 0], label=r"Relative $|E|^2$")
    fig.savefig(output / f"{stem}_fields.png", dpi=165)
    plt.close(fig)


def plot_volume(path, case, data, volume):
    intensity = volume["electric_squared_volume"]
    flux = volume["longitudinal_flux_w_m2"]
    x, y, z = data["x_um"], data["y_um"], data["z_um"]
    iy = int(np.argmax(intensity.sum(axis=(0, 2), dtype=np.float64)))
    center = (case.excitation.gap_um + case.excitation.width_um) / 2
    left, right = [int(np.argmin(abs(x-position))) for position in (-center, center)]
    intensity_peak = max(float(intensity.max()), np.finfo(np.float32).tiny)
    flux_peak = max(float(np.max(abs(flux))), np.finfo(np.float32).tiny)
    intensity_norm = LogNorm(vmin=1e-5, vmax=1.)
    flux_norm = SymLogNorm(linthresh=.01, vmin=-1., vmax=1.)
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.3), sharex=True, layout="constrained")

    heat = axes[0, 0].pcolormesh(z, x, (intensity[:, iy] / intensity_peak).T,
                                shading="nearest", cmap="inferno", norm=intensity_norm)
    signed = axes[0, 1].pcolormesh(z, x, (flux[:, iy] / flux_peak).T,
                                  shading="nearest", cmap="RdBu_r", norm=flux_norm)
    axes[0, 0].set(title=fr"$|E|^2$ at y = {y[iy]:.2f} µm", ylabel="x (µm)")
    axes[0, 1].set(title=fr"Signed forward flux at y = {y[iy]:.2f} µm", ylabel="x (µm)")
    for ax in axes[0]:
        ax.axhline(0, color=".6", linewidth=.7, linestyle=":")

    for ax, index in zip(axes[1], (left, right), strict=True):
        ax.pcolormesh(z, y, (intensity[:, :, index] / intensity_peak).T,
                      shading="nearest", cmap="inferno", norm=intensity_norm)
        ax.set(title=fr"$|E|^2$ at x = {x[index]:.2f} µm",
               ylabel="y (µm)", xlabel="Distance from input (µm)")
    fig.colorbar(heat, ax=[axes[0, 0], *axes[1]], label=r"Relative $|E|^2$")
    fig.colorbar(signed, ax=axes[0, 1], label="Relative signed power flux")
    try:
        with atomic_path(path) as temporary:
            fig.savefig(temporary, dpi=160)
    finally:
        plt.close(fig)


def study_geometry(model, source):
    model_2d = (source / read_json(source/"config.json")["model"]).resolve()
    geometry = read_cross_section(source, {"model": str(model_2d)})
    if geometry != read_cross_section(model.parent, {"model": str(model)}):
        raise ValueError("Размеры слоя и углы стенок 2D и 3D различаются")
    validate_materials(model, model_2d)
    return geometry


def run(model, source, results, *, solve, dry_run, pilot, refresh_fields=False, check_28=False):
    geometry = study_geometry(model, source)
    cache = results.parent / "field_maps"
    inputs = {(gap, family): load_excitation(source, gap, family)
              for gap in (PILOT_GAP, TARGET_GAP) for family in FAMILIES
              if not pilot or gap == PILOT_GAP}
    cases = {}

    def collect(gap, family, layers, *, allow_solve=True):
        case = CouplerCase(cache, model, inputs[gap, family], layers)
        label = f"g={gap:g} {family} · {layers} слоёв"
        modes = ", ".join(identifier for _, _, identifier in case.excitation.refs)
        ready = case.ready()
        state = "кэш с объёмом поля" if ready and case.has_volume() else (
            "кэш с поперечными срезами" if ready else "нет проверенного результата")
        print(f"{label} | моды {modes} | {state} | {case.folder}", flush=True)
        if dry_run:
            return None
        if not allow_solve and not ready:
            cases[gap, family, layers] = (case, None, dict(status="not_requested"))
            return None
        try:
            data = case.run(solve and allow_solve, refresh_fields=refresh_fields and allow_solve)
            metrics = case_metrics(data) if data is not None else {}
        except Exception as error:
            print(f"{label} | {error}", flush=True)
            cases[gap, family, layers] = (case, None, dict(status="error", diagnostic=str(error)))
            return None
        status = "calculated" if data is not None else "missing"
        cases[gap, family, layers] = (case, data, dict(metrics, status=status))
        if solve and allow_solve and data is not None and case.has_volume():
            preview = case.folder / "diagnostics.png"
            if not preview.is_file() or preview.stat().st_mtime_ns < case.field.stat().st_mtime_ns:
                try:
                    plot_volume(preview, case, data, case.read_volume())
                    print(f"Карта поля | {preview}", flush=True)
                except (OSError, RuntimeError, ValueError) as error:
                    print(f"Данные сохранены | карта пока недоступна: {error}", flush=True)
        return metrics or None

    for family in FAMILIES:
        if dry_run:
            for gap in (PILOT_GAP, TARGET_GAP):
                if (gap, family) in inputs:
                    layers_to_check = (7, 14, 28) if gap == PILOT_GAP and check_28 else (7, 14)
                    for layers in layers_to_check:
                        collect(gap, family, layers)
            continue
        coarse = collect(PILOT_GAP, family, 7)
        fine = collect(PILOT_GAP, family, 14)
        if coarse is None or fine is None:
            continue
        chosen = 7
        if abs(coarse["cross_out"]-fine["cross_out"]) > MESH_TOLERANCE:
            finer = collect(PILOT_GAP, family, 28, allow_solve=check_28)
            if finer is None or abs(fine["cross_out"]-finer["cross_out"]) > MESH_TOLERANCE:
                cases[PILOT_GAP, family, 14][2]["status"] = "axial_mesh_unresolved"
                continue
            coarse, fine = fine, finer
            chosen = 14
        if min(coarse["left_in"], fine["left_in"]) < INPUT_LIMIT:
            cases[PILOT_GAP, family, chosen][2]["status"] = "input_not_localized"
            continue
        cases[PILOT_GAP, family, chosen][2]["status"] = "accepted"
        if not pilot:
            target = collect(TARGET_GAP, family, chosen)
            if target is not None:
                cases[TARGET_GAP, family, chosen][2]["status"] = (
                    "accepted" if target["left_in"] >= INPUT_LIMIT else "input_not_localized")
    if dry_run:
        return 0
    target = results / "studies" / ("coupler-3d-pilot" if pilot else "coupler-3d")
    with publish_study(target) as output:
        rows = []
        for (gap, family, layers), (case, data, metrics) in cases.items():
            ideal = np.sin((case.excitation.beta[0]-case.excitation.beta[1])*100e-6)**2
            row = dict(metrics, gap_um=gap, family=family, layers=layers, prediction_cross_200um=ideal,
                       input_left_2d=case.excitation.p_left_2d, cache=relative_path(output, case.folder))
            for i, (reference, checksum, identifier) in enumerate(case.excitation.refs, 1):
                row.update({f"reference_{i}": relative_path(output, reference), f"mode_id_{i}": identifier,
                            f"sha256_{i}": checksum})
            if data is not None:
                plot_case(output, case, data, geometry)
                volume = case.read_volume()
                if volume is not None:
                    diagnostic = f"g_{gap:g}_{family}_{layers}layers_diagnostics.png"
                    plot_volume(output / diagnostic, case, data, volume)
                    row["field_diagnostics"] = diagnostic
                row["field_coverage"] = "sampled_volume" if volume is not None else "five_slices"
                row["map_nx"], row["map_ny"] = len(data["x_um"]), len(data["y_um"])
                row["map_nz"] = len(data["z_um"]) if volume is not None else len(data["slice_um"])
            rows.append(row)
        completed = {(row["gap_um"], row["family"]) for row in rows if row["status"] == "accepted"}
        for gap, family in inputs.keys() - {(row["gap_um"], row["family"]) for row in rows}:
            rows.append(dict(gap_um=gap, family=family, status="awaiting_pilot"))
        write_csv(output / "summary.csv", pd.DataFrame(rows))
        write_json(output / "config.json", dict(
            model=relative_path(output, model), source_2d=relative_path(output, source),
            gaps_um=[PILOT_GAP] if pilot else [PILOT_GAP, TARGET_GAP],
            width_um=.5, wavelength_um=1.55, length_um=200,
            mesh_delta_limit=MESH_TOLERANCE, input_left_limit=INPUT_LIMIT,
            allow_28_layers=check_28,
        ))
    print(f"REPORT_DIR={target}", flush=True)
    return len(inputs.keys() - completed)


def main():
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=TEMPLATE)
    parser.add_argument("--results-root", type=Path, default=PROJECT/"results")
    parser.add_argument("--source-2d", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--check-28", action="store_true",
                        help="Allow a new 28-layer verification if 7 and 14 layers disagree")
    parser.add_argument("--refresh-fields", action="store_true",
                        help="With --solve, recalculate cases whose cache lacks the current diagnostic volume")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--solve", action="store_true")
    actions.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.refresh_fields and not args.solve:
        parser.error("--refresh-fields требуется запускать вместе с --solve")
    try:
        results = args.results_root.resolve()
        source = (args.source_2d or results/"studies"/"coupled-modes").resolve()
        if args.solve and args.model == TEMPLATE and not TEMPLATE.is_file():
            build_template()
        pending = run(args.model.resolve(strict=True), source, results, solve=args.solve,
                      dry_run=args.dry_run, pilot=args.pilot, refresh_fields=args.refresh_fields,
                      check_28=args.check_28)
        if args.solve and pending:
            parser.exit(1, f"Осталось подтвердить случаев: {pending}. См. summary.csv.\n")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
