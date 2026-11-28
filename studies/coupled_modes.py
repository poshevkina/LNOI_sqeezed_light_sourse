"""Read or calculate the modes of the 2D pair of LNOI waveguides."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    # Direct execution puts the studies directory on the import path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linbo3.comsol_models import CoupledWaveguides2D


PROJECT = Path(__file__).resolve().parents[1]
WIDTH_UM = 0.5
GAPS_UM = (0.8,)
WAVELENGTH_UM = 1.55
MODE_COUNT = 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    models = sorted(PROJECT.glob("*modes LNOI X-cut.mph"))
    parser.add_argument("--model", type=Path, default=models[0] if len(models) == 1 else None)
    parser.add_argument("--results-root", type=Path, default=PROJECT / "results")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--solve", action="store_true", help="Calculate missing points")
    action.add_argument("--dry-run", action="store_true", help="Show the requested points and cache status")
    args = parser.parse_args()
    if args.model is None:
        parser.error("Укажи путь к модели соседних волноводов через --model")
    try:
        model = CoupledWaveguides2D(args.model, results_root=args.results_root)
        result = model.modes(width_um=WIDTH_UM, gaps_um=GAPS_UM, wavelength_um=WAVELENGTH_UM,
                             mode_count=MODE_COUNT, solve=args.solve, dry_run=args.dry_run)
        if not args.dry_run:
            print(result.load_modes().to_string(index=False))
            print(f"RESULT_DIR={result.run_dir}")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
