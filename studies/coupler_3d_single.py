"""Run one 3D LNOI coupler case at a chosen number of axial layers."""

import argparse
from pathlib import Path
import sys
from shutil import copyfile

if __package__ in (None, ""):
    # Direct execution puts the studies directory on the import path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import pandas as pd

from linbo3.artifacts import relative_path, write_csv
from linbo3.comsol_models.coupler_3d.cache import CouplerCase
from linbo3.comsol_models.coupler_3d.inputs import load_excitation
from linbo3.comsol_models.coupler_3d.template import TEMPLATE, build_template
from linbo3.study_output import publish_study
from studies.coupler_3d import FAMILIES, PROJECT, case_metrics, plot_case, plot_volume, study_geometry


LAYERS = 16
GAP_UM = 0.8
FAMILY = "quasi-TE"
MODEL = TEMPLATE
RESULTS_ROOT = PROJECT / "results"
SOURCE_2D = RESULTS_ROOT / "studies" / "coupled-modes"


def run(model, source, results, *, layers, gap, family, solve=False, dry_run=False):
    if type(layers) is not int or layers < 1:
        raise ValueError("Число продольных слоёв должно быть положительным целым")
    if not model.is_file() and model == TEMPLATE and solve:
        build_template()
    geometry = study_geometry(model, source)
    excitation = load_excitation(source, gap, family)
    case = CouplerCase(results.parent / "field_maps", model, excitation, layers)
    print(f"g={gap:g} мкм | {family} | {layers} продольных слоёв", flush=True)
    print("2D-моды: " + ", ".join(identifier for _, _, identifier in excitation.refs), flush=True)
    print(f"{'Готовый кэш' if case.ready() else 'Нет готового кэша'} | {case.folder}", flush=True)
    if dry_run:
        return
    data = case.run(solve=solve)
    if data is None:
        print("Для расчёта этой точки запусти сценарий с --solve.", flush=True)
        return

    volume = case.read_volume()
    if volume is not None:
        preview = case.folder / "diagnostics.png"
        plot_volume(preview, case, data, volume)
        print(f"Карта поля | {preview}", flush=True)

    gap_label = format(gap, "f").rstrip("0").rstrip(".").replace(".", "p")
    target = results / "studies" / f"coupler-3d-single-g{gap_label}-{family.lower()}-{layers}layers"
    with publish_study(target) as output:
        plot_case(output, case, data, geometry)
        if volume is not None:
            copyfile(preview, output / "diagnostics.png")
        rows = []
        for count in sorted({7, 14, layers}):
            comparison = case if count == layers else CouplerCase(case.folder.parent, model, excitation, count)
            saved = data if count == layers else comparison.run()
            if saved is None:
                continue
            rows.append(dict(case_metrics(saved), layers=count, gap_um=gap, family=family,
                             requested=count == layers, cache=relative_path(output, comparison.folder)))
        write_csv(output / "summary.csv", pd.DataFrame(rows))
    print(f"REPORT_DIR={target}", flush=True)


def main():
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--gap", type=float, default=GAP_UM, help="Top gap in micrometres")
    parser.add_argument("--family", choices=FAMILIES, default=FAMILY)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Show the selected case and cache location")
    action.add_argument("--solve", action="store_true", help="Solve the selected case if it is absent")
    args = parser.parse_args()
    try:
        run(MODEL, SOURCE_2D, RESULTS_ROOT, layers=args.layers, gap=args.gap, family=args.family,
            solve=args.solve, dry_run=args.dry_run)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
