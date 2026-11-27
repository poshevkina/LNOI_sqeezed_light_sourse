"""Compare K across widths by local PUMP rank and one polynomial degree."""

from pathlib import Path
import sys


if __package__ in (None, ""):
    # Direct execution puts the studies directory on the import path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from linbo3.artifacts import read_json, utc_now, write_json
from linbo3.interpolation_plots import plot_k_by_width
from linbo3.interpolation_study import source_signature
from linbo3.polynomial import fit_study
from linbo3.saved_modes import BANDS, inspect_modes
from linbo3.single_degree import plot_jsi_overview
from linbo3.study_output import publish_study, report_metadata
from linbo3.study_plots import StudyPlots
from linbo3.sweep import PulseConfig, run_sweep
from studies._common import PROJECT_ROOT, study_main


STUDY_NAME = "unclassified"
RESULTS_ROOT = PROJECT_ROOT / "results"
COMSOL_RUN = None
DEGREE = 6
SIGNAL_MODE = 0
IDLER_MODE = 1
POINTS = 1501
PULSE_FS = 2000.0
HALF_WINDOW = .0025
TRACKING = {}
ANCHOR_SOLNUMS = None


def run(source, results_root, *, points=POINTS):
    source = Path(source)
    output = Path(results_root) / "studies" / STUDY_NAME
    manifest = read_json(source / "config.json")
    pulse = PulseConfig(PULSE_FS, HALF_WINDOW, points)
    started_at = utc_now()
    with publish_study(output) as work:
        inspection = inspect_modes(source, output_dir=work, tracking=TRACKING, anchor_solnums=ANCHOR_SOLNUMS)
        raw = {width: bands for width, bands in inspection.modes.items() if all(band in bands for band in BANDS)}
        if not raw:
            diagnostics = " ".join(f"w={row.width_um:g}, {row.band}: {row.message or row.analysis_state}"
                                   for row in inspection.pairs.itertuples())
            raise ValueError("Нет ширин с подтверждёнными PUMP и IS. " + diagnostics)
        study = fit_study(raw, degrees=(DEGREE,), signal_mode=SIGNAL_MODE, idler_mode=IDLER_MODE,
                          pump_wavelength_um=manifest.get("pump_wavelength_um", .775))
        sweep = run_sweep(study, pulse, work, length_m=manifest.get("length_m", .12), cache_root=results_root)
        method = f"poly{DEGREE}"
        summary = sweep.summary.assign(method=method).merge(
            study.mode_matching[["width_um", "pump_mode", "pump_rank"]],
            on=["width_um", "pump_mode"], validate="many_to_one",
        )
        summary.to_csv(work / "summary.csv", index=False)
        plots = StudyPlots(study, work / "figures", max_plot_points=451, show=False)
        plots.plot_dispersion_by_width()
        plot_k_by_width(work / "k_by_width.png", summary, [method], sorted(manifest["widths_um"]))
        plot_jsi_overview(plots, sweep.cases, pulse, DEGREE, manifest["widths_um"])
        settings = report_metadata(
            source, work, degree=DEGREE, pulse_fs=PULSE_FS, half_window=HALF_WINDOW, points=points,
            signal_mode=SIGNAL_MODE, idler_mode=IDLER_MODE,
            tracking=inspection.tracking, anchor_solnums=inspection.anchor_solnums,
            source_signature=source_signature(source), state="complete", completed_at=utc_now(),
        )
        write_json(work / "settings.json", {**settings, "started_at": started_at})
    return output


def main():
    study_main(__doc__, run, points=POINTS, results_root=RESULTS_ROOT, comsol_run=COMSOL_RUN)


if __name__ == "__main__":
    main()
