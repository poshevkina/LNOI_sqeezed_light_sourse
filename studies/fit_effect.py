"""Compare the effect of dispersion fits on the Schmidt number."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    # Direct execution puts the studies directory on the import path.
    sys.path.insert(0, str(PROJECT_ROOT))

from linbo3.interpolation import METHODS
from linbo3.interpolation_study import run_study
from studies._common import study_main


STUDY_NAME = "fit-effect"
RESULTS_ROOT = PROJECT_ROOT / "results"
COMSOL_RUN = None  # None selects the last completed run; or set its directory name.
FIT_METHODS = METHODS  # Polynomials 1–6 in the current project's omega convention.
WIDTHS_UM = None      # None includes every width with supported branches.
PUMP_MODES = None     # None includes every tracked pump mode.
SIGNAL_MODE = 0
IDLER_MODE = 1
PULSE_FS = 2000.0
HALF_WINDOW_RAD_PER_FS = 0.0025
POINTS = 1501


def run(source, results_root, *, points=POINTS, widths_um=WIDTHS_UM, study_name=STUDY_NAME):
    report = run_study(
        source, points=points, study_name=study_name, cache_root=results_root, methods=FIT_METHODS,
        widths_um=widths_um, pump_modes=PUMP_MODES, signal_mode=SIGNAL_MODE,
        idler_mode=IDLER_MODE, pulse_fs=PULSE_FS, half_window=HALF_WINDOW_RAD_PER_FS,
    )
    return report.output_dir


def main():
    study_main(__doc__, run, points=POINTS, results_root=RESULTS_ROOT, comsol_run=COMSOL_RUN)


if __name__ == "__main__":
    main()
