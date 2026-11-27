"""Compare spatially classified SPDC across a series of pump durations."""

from pathlib import Path

import pandas as pd

from linbo3.artifacts import relative_path, utc_now, write_json
from linbo3.marginal_spectra import plot_marginal_spectra
from linbo3.pulse_series import plot_duration_summary
from linbo3.pump_hybridization import audit_pump_hybridization
from linbo3.spatial_workflow import plot_spatial_jsi, plot_spatial_k, write_spatial_sweep
from linbo3.study_output import publish_study, report_metadata
from linbo3.sweep import PulseConfig
from studies._common import PROJECT_ROOT, classify_spatial, study_main


STUDY_NAME = "pulse-series"
RESULTS_ROOT = PROJECT_ROOT / "results"
COMSOL_RUN = None
DEGREE = 6
SIGNAL_LABEL = "TM0"
IDLER_LABEL = "TE0"
POINTS = 2000
# Original half-windows in rad/fs, chosen along a line in 1/t.
WINDOWS = {
    250: .015,
    500: .007857142857142856,
    750: .005476190476190476,
    1000: .004285714285714285,
    1250: .0035714285714285718,
    1500: .003095238095238095,
    1750: .0027551020408163266,
    2000: .0025,
}
SPECTRAL_FRACTION = .99
MIXING_LIMIT = .10
CLASSIFICATION = dict(imag_limit=1e-6, polarization_threshold=.9, peak_prominence=.08, peak_height=.10)
TRACKING = dict(imag_limit=1e-6, min_overlap=.9, min_margin=.1)
SPECTRA_AT_FS = (250,)  # The combined K figures always include every duration.


def plot_width_summary(path, spatial, pulses):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots((len(pulses) + 3) // 4, 4, figsize=(20, 5 * ((len(pulses) + 3) // 4)),
                             squeeze=False, layout="constrained")
    handles = {}
    try:
        for ax, (pulse, folder, audit) in zip(axes.flat, pulses):
            plot_spatial_k(folder, spatial, hybridization=audit, ax=ax,
                           title=f"t={pulse.t_fs:g} фс · DW=±{pulse.dw_half:.5g}")
            for handle, label in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(label, handle)
            ax.get_legend().remove()
        for ax in list(axes.flat)[len(pulses):]:
            ax.set_visible(False)
        fig.legend(list(handles.values()), list(handles), loc="outside lower center", ncol=3)
        fig.suptitle("Число Шмидта от ширины для разных длительностей")
        fig.savefig(path, dpi=100)
    finally:
        plt.close(fig)


def run(source, results_root, *, points=POINTS):
    output = Path(results_root) / "studies" / STUDY_NAME
    if not WINDOWS:
        raise ValueError("Задай хотя бы одну длительность и полуширину окна")
    started_at = utc_now()
    with publish_study(output) as work:
        spatial = classify_spatial(source, work, degree=DEGREE, signal_label=SIGNAL_LABEL,
                                   idler_label=IDLER_LABEL, classification=CLASSIFICATION, tracking=TRACKING)
        reports, summaries = [], []
        for duration, half_window in sorted(WINDOWS.items()):
            print(f"t={duration:g} фс | DW=±{half_window:.7g} рад/фс", flush=True)
            pulse = PulseConfig(duration, half_window, points)
            folder = work / f"t_{duration:g}fs"
            sweep, summary = write_spatial_sweep(source, spatial, pulse, folder, cache_root=results_root)
            audit = audit_pump_hybridization(source, spatial, folder, points=points,
                                            fraction=SPECTRAL_FRACTION, mixing_limit=MIXING_LIMIT)
            if duration in SPECTRA_AT_FS:
                plot_spatial_jsi(folder, spatial, cases=sweep.cases, title=f"JSI · t={duration:g} фс")
                plot_marginal_spectra(source, spatial, folder, points=points, title=f"t={duration:g} фс")
            reports.append((pulse, folder, audit))
            table = summary.merge(
                audit.cases[["width_um", "pump_label", "pump_interval_hybrid", "checked_points", "status"]],
                on=["width_um", "pump_label"], validate="one_to_one",
            )
            table["npz"] = table["npz"].map(lambda name: relative_path(work, (folder / name).resolve()))
            summaries.append(table)
        summary = pd.concat(summaries, ignore_index=True)
        summary.to_csv(work / "summary.csv", index=False)
        plot_width_summary(work / "K_overview.png", spatial, reports)
        plot_duration_summary(summary, work / "K_vs_duration.png")
        settings = report_metadata(
            source, work, identity=reports[0][2].identity, windows=WINDOWS, points=points,
            classification=CLASSIFICATION, spectra_at_fs=SPECTRA_AT_FS, spectral_fraction=SPECTRAL_FRACTION,
            mixing_limit=MIXING_LIMIT, state="complete", completed_at=utc_now(),
        )
        write_json(work / "settings.json", {**settings, "started_at": started_at})
    return output


def main():
    study_main(__doc__, run, points=POINTS, results_root=RESULTS_ROOT, comsol_run=COMSOL_RUN)


if __name__ == "__main__":
    main()
