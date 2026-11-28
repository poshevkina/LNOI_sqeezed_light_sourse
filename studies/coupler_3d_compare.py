"""Run TE and TM sequentially, or compare the saved cases without COMSOL."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    # Direct execution puts the studies directory on the import path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
import pandas as pd

from linbo3.artifacts import relative_path, write_csv, write_json
from linbo3.comsol_models.coupler_3d.cache import CouplerCase
from linbo3.comsol_models.coupler_3d.diagnostics import plot_power, plot_slice_comparison, plot_volume_details
from linbo3.comsol_models.coupler_3d.inputs import load_excitation
from linbo3.study_output import publish_study
from studies.coupler_3d import FAMILIES, case_metrics, plot_case, plot_volume, study_geometry
from studies.coupler_3d_single import GAP_UM, LAYERS, MODEL, RESULTS_ROOT, SOURCE_2D
from studies.coupler_3d_single import run as run_single


def analyze(model, source, results, *, layers, gap):
    geometry = study_geometry(model, source)
    groups, coverage, summaries = {}, [], []
    for family in FAMILIES:
        excitation = load_excitation(source, gap, family)
        groups[family] = {}
        for count in sorted({7, 14, layers}):
            case = CouplerCase(results.parent / "field_maps", model, excitation, count)
            data = case.run()
            status = "missing" if data is None else (
                "sampled_volume" if case.has_volume() else "five_slices")
            coverage.append(dict(family=family, layers=count, requested=count == layers, coverage=status))
            if data is not None:
                groups[family][count] = (case, data)

    gap_label = format(gap, "f").rstrip("0").rstrip(".").replace(".", "p")
    target = results / "studies" / f"coupler-3d-compare-g{gap_label}-{layers}layers"
    with publish_study(target) as output:
        plot_power(output, groups, layers)
        for family, cases in groups.items():
            if not cases:
                continue
            representative = layers if layers in cases else max(cases)
            case, data = cases[representative]
            folder = output / family.lower()
            folder.mkdir()
            plot_case(folder, case, data, geometry)
            volume = case.read_volume()
            if volume is not None:
                plot_volume(folder/"longitudinal_sections.png", case, data, volume)
                plot_volume_details(folder, case, data, volume, geometry)
            if len(cases) > 1:
                plot_slice_comparison(folder, cases[min(cases)], cases[max(cases)], geometry)

            finest = max(cases)
            reference_out = case_metrics(cases[finest][1])["cross_out"]
            for count, (saved_case, saved) in sorted(cases.items()):
                metrics = case_metrics(saved)
                ideal = np.sin(np.diff(saved_case.excitation.beta)[0]*saved["z_um"][-1]*1e-6/2)**2
                summaries.append(dict(
                    metrics, family=family, layers=count, representative=count == representative,
                    cross_out_2d=ideal, cross_vs_2d_pp=100*(metrics["cross_out"]-ideal),
                    cross_vs_finest_pp=100*(metrics["cross_out"]-reference_out),
                    reference_layers=finest, cache=relative_path(output, saved_case.folder),
                ))
        write_csv(output/"coverage.csv", pd.DataFrame(coverage))
        if summaries:
            write_csv(output/"summary.csv", pd.DataFrame(summaries))
        write_json(output/"config.json", dict(gap_um=gap, requested_layers=layers,
                   source_2d=relative_path(output, source), model=relative_path(output, model)))
    print(f"ANALYSIS_DIR={target}", flush=True)
    return sum(layers not in cases for cases in groups.values())


def main():
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--gap", type=float, default=GAP_UM)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--solve", action="store_true", help="Solve both polarizations sequentially, then analyze")
    action.add_argument("--dry-run", action="store_true", help="Show the two requested cases")
    args = parser.parse_args()
    if args.layers < 1:
        parser.error("--layers must be positive")
    try:
        if args.dry_run or args.solve:
            for family in FAMILIES:
                try:
                    run_single(MODEL, SOURCE_2D, RESULTS_ROOT, layers=args.layers, gap=args.gap,
                               family=family, solve=args.solve, dry_run=args.dry_run)
                except Exception as error:
                    if not args.solve:
                        raise
                    print(f"{family} | {error}", flush=True)
        if not args.dry_run:
            missing = analyze(MODEL, SOURCE_2D, RESULTS_ROOT, layers=args.layers, gap=args.gap)
            if args.solve and missing:
                parser.exit(1, f"Не готовы результаты для {missing} поляризаций. См. coverage.csv.\n")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
