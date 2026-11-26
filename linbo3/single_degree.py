from .storage import analysis_root
"""Dedicated selected-degree workflow; comparison notebooks keep their original behavior."""

import json
from pathlib import Path

from .interpolation_study import source_signature
from .polynomial import fit_study
from .saved_modes import inspect_modes
from .study_plots import StudyPlots
from .sweep import PulseConfig, run_sweep

DEGREE = 6


def single_directory(run_dir, points=1501, degree=6):
    if type(degree) is not int or not 1 <= degree <= 6:
        raise ValueError("Степень должна быть целым числом от 1 до 6")
    return analysis_root(run_dir)/"single"


def saved_single_is_current(run_dir, inspection, points=1501, degree=6, *, require_overview=True):
    path = single_directory(run_dir, points, degree) / "settings.json"
    if not path.is_file():
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))
    return (saved.get("state") == "complete" and saved.get("degree") == degree
            and (not require_overview or (saved.get("layout_version") == 4
                 and (single_directory(run_dir, points, degree) / "figures/jsi_overview.png").is_file()))
            and saved.get("points") == points and saved.get("tracking") == inspection.tracking
            and saved.get("anchor_solnums") == inspection.anchor_solnums
            and saved.get("source_signature") == source_signature(run_dir))


def run_single_study(run_dir, *, points=1501, tracking=None, anchor_solnums=None, degree=6):
    """Fit and calculate JSI/K only for the selected degree, all confirmed widths and PUMP modes."""
    run_dir = Path(run_dir)
    output = single_directory(run_dir, points, degree)
    inspection = inspect_modes(run_dir, output_dir=output, tracking=tracking, anchor_solnums=anchor_solnums)
    manifest = json.loads((run_dir / "config.json").read_text(encoding="utf-8-sig"))
    settings = dict(state="running", degree=degree, layout_version=4, fit_coordinate="omega_rad_per_fs", points=points,
                    tracking=inspection.tracking, anchor_solnums=inspection.anchor_solnums,
                    source_signature=source_signature(run_dir))
    (output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    complete = {width: bands for width, bands in inspection.modes.items()
                if all(band in bands for band in ("PUMP", "IS"))}
    if not complete:
        raise ValueError("Нет ширин с подтверждёнными PUMP и IS. Проверь pairs.csv в папке анализа.")
    study = fit_study(complete, degrees=(degree,), pump_wavelength_um=manifest["pump_wavelength_um"])
    pulse = PulseConfig(t_fs=2000, dw_half=0.0025, n_points=points)
    sweep = run_sweep(study, pulse, output, length_m=manifest.get("length_m", .12))
    sweep.summary.assign(method=f"poly{degree}").to_csv(output / "summary.csv", index=False)
    plots = StudyPlots(study, output / "figures", max_plot_points=451, show=False)
    plot_jsi_overview(plots, sweep.cases, pulse, degree, manifest["widths_um"])
    settings["state"] = "complete"
    (output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return output


def plot_jsi_overview(plots, cases, pulse, degree, widths):
    """One figure: each width is a row, PUMP ranks increase left to right."""
    import matplotlib.pyplot as plt
    # No row for a rejected IS/PUMP band (including more than two admissible IS modes).
    widths = [width for width in sorted(widths) if cases.get(width, {}).get(degree)]
    if not widths:
        raise ValueError("Нет рассчитанных ширин для построения JSI")
    labels = sorted(plots.study.pump_labels, key=lambda label: int(label.split()[-1]))
    fig, axes = plt.subplots(len(widths), len(labels), squeeze=False,
                             figsize=(4.2 * len(labels), 3.5 * len(widths)), layout="constrained")
    try:
        for row, width in enumerate(widths):
            row_mesh = None
            for col, label in enumerate(labels):
                ax = axes[row, col]
                case = cases.get(width, {}).get(degree, {}).get(label)
                if case is None:
                    ax.text(.5, .5, "Нет подтверждённых данных", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"w={width:g} мкм · PUMP {label}")
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                row_mesh = plots.jsi_panel(ax, case)
                ax.set_title(f"w={width:g} мкм · PUMP {label}\nK={float(case['K']):.4f}")
            if row_mesh is not None:
                fig.colorbar(row_mesh, ax=axes[row, :].tolist(), label="JSI", shrink=.85)
        fig.suptitle(f"JSI · полином степени {degree} по ω · {pulse.t_fs:g} фс")
        plots.figure_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plots.figure_dir / "jsi_overview.png", dpi=140)
    finally:
        plt.close(fig)


def refresh_single_overview(run_dir, inspection, points=1501, degree=6):
    """Redraw the figure using existing NPZ files only; never calculate spectra here."""
    import numpy as np
    from .sweep import _cache_path
    if not saved_single_is_current(run_dir, inspection, points, degree, require_overview=False):
        raise ValueError("Нет актуальных численных данных. Выполни раздел 5 с RUN_STUDY=True.")
    run_dir = Path(run_dir)
    output = single_directory(run_dir, points, degree)
    manifest = json.loads((run_dir / "config.json").read_text(encoding="utf-8-sig"))
    complete = {width: bands for width, bands in inspection.modes.items()
                if all(band in bands for band in ("PUMP", "IS"))}
    if not complete:
        raise ValueError("Нет подтверждённых PUMP/IS для построения JSI")
    study = fit_study(complete, degrees=(degree,), pump_wavelength_um=manifest["pump_wavelength_um"])
    pulse = PulseConfig(t_fs=2000, dw_half=.0025, n_points=points)
    cases = {}
    for width, mapping in study.pump_map.items():
        cases[width] = {degree: {}}
        for label, block in mapping.items():
            path = _cache_path(output / "cache", study, width, degree, block, pulse,
                               manifest.get("length_m", .12), False)
            if not path.is_file():
                raise FileNotFoundError("Нет кэша JSI. Выполни раздел 5 с RUN_STUDY=True: " + str(path))
            with np.load(path, allow_pickle=False) as saved:
                cases[width][degree][label] = dict(cache_path=path, K=float(saved["K"]))
    plots = StudyPlots(study, output / "figures", max_plot_points=451, show=False)
    plot_jsi_overview(plots, cases, pulse, degree, manifest["widths_um"])
    path = output / "settings.json"
    settings = json.loads(path.read_text(encoding="utf-8"))
    settings["layout_version"] = 4
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return output / "figures/jsi_overview.png"
