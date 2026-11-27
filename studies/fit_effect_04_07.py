"""Compare polynomial fits for waveguide widths from 0.4 to 0.7 micrometres."""

from pathlib import Path
import sys


if __package__ in (None, ""):
    # Direct execution puts the studies directory on the import path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linbo3.artifacts import read_json
from studies import fit_effect
from studies._common import study_main


STUDY_NAME = "fit-effect-w04-w07"
WIDTHS_UM = (0.4, 0.5, 0.6, 0.7)


def run(source, results_root, *, points=fit_effect.POINTS):
    status = read_json(Path(source) / "status.json")
    pending = [
        f"w={width:g}, {band}"
        for width in WIDTHS_UM
        for band in ("PUMP", "IS")
        if status["pairs"].get(f"{width:g}", {}).get(band, {}).get("state") != "complete"
    ]
    if pending:
        raise ValueError("Сначала закончи выгрузку COMSOL: " + ", ".join(pending))
    return fit_effect.run(source, results_root, points=points, widths_um=WIDTHS_UM,
                          study_name=STUDY_NAME)


def main():
    study_main(__doc__, run, points=fit_effect.POINTS, results_root=fit_effect.RESULTS_ROOT,
               required_widths=WIDTHS_UM)


if __name__ == "__main__":
    main()
