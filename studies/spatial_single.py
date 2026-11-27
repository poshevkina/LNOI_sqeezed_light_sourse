"""Calculate spatially classified SPDC for one pump duration."""

from pathlib import Path
import sys


if __package__ in (None, ""):
    # Direct execution puts the studies directory on the import path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linbo3.artifacts import utc_now, write_json
from linbo3.marginal_spectra import plot_marginal_spectra
from linbo3.pump_hybridization import audit_pump_hybridization
from linbo3.spatial_workflow import plot_spatial_dispersion, plot_spatial_jsi, plot_spatial_k, write_spatial_sweep
from linbo3.study_output import publish_study, report_metadata
from linbo3.sweep import PulseConfig
from studies._common import PROJECT_ROOT, classify_spatial, study_main


STUDY_NAME = "spatial-single"
RESULTS_ROOT = PROJECT_ROOT / "results"
COMSOL_RUN = None
DEGREE = 6
SIGNAL_LABEL = "TM0"
IDLER_LABEL = "TE0"
POINTS = 1501
PULSE_FS = 2000.0
HALF_WINDOW = .0025
SPECTRAL_FRACTION = .99
MIXING_LIMIT = .10
CLASSIFICATION = dict(imag_limit=1e-6, polarization_threshold=.9, peak_prominence=.08, peak_height=.10)
TRACKING = dict(imag_limit=1e-6, min_overlap=.9, min_margin=.1)


def run(source, results_root, *, points=POINTS):
    output = Path(results_root) / "studies" / STUDY_NAME
    pulse = PulseConfig(PULSE_FS, HALF_WINDOW, points)
    started_at = utc_now()
    with publish_study(output) as work:
        spatial = classify_spatial(source, work, degree=DEGREE, signal_label=SIGNAL_LABEL,
                                   idler_label=IDLER_LABEL, classification=CLASSIFICATION, tracking=TRACKING)
        sweep, summary = write_spatial_sweep(source, spatial, pulse, work, cache_root=results_root)
        audit = audit_pump_hybridization(source, spatial, work, points=points,
                                        fraction=SPECTRAL_FRACTION, mixing_limit=MIXING_LIMIT)
        plot_spatial_dispersion(spatial, work / "figures", show=False)
        plot_spatial_jsi(work, spatial, cases=sweep.cases)
        plot_spatial_k(work, spatial, hybridization=audit, show=False)
        plot_marginal_spectra(source, spatial, work, points=points)
        summary.merge(audit.cases[["width_um", "pump_label", "pump_interval_hybrid", "checked_points", "status"]],
                      on=["width_um", "pump_label"], validate="one_to_one").to_csv(work / "spdc_and_hybridization.csv",
                                                                                  index=False)
        settings = report_metadata(
            source, work, identity=audit.identity, pulse_fs=PULSE_FS, half_window=HALF_WINDOW,
            points=points, classification=CLASSIFICATION, spectral_fraction=SPECTRAL_FRACTION,
            mixing_limit=MIXING_LIMIT, state="complete", completed_at=utc_now(),
        )
        write_json(work / "settings.json", {**settings, "started_at": started_at})
    return output


def main():
    study_main(__doc__, run, points=POINTS, results_root=RESULTS_ROOT, comsol_run=COMSOL_RUN)


if __name__ == "__main__":
    main()
