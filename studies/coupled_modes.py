"""Compare guided supermodes of a fixed-width pair over a series of gaps."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    # Direct execution puts the studies directory on the import path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from linbo3.artifacts import read_json, relative_path, write_csv, write_json
from linbo3.comsol_models.coupled import MODEL_PATH, CoupledWaveguides2D
from linbo3.comsol_models.coupled.analysis import (
    IMAG_LIMIT, OVERLAP_LIMIT, PARITY_LIMIT, POLARIZATION_LIMIT, RIDGE_LIMIT,
    describe_candidates, load_mode_field, overlap, select_pair, subspace_overlap,
)
from linbo3.comsol_models.coupled.plots import plot_coupling, plot_modes
from linbo3.mode_classification import read_cross_section
from linbo3.study_output import publish_study


PROJECT = Path(__file__).resolve().parents[1]
WIDTH_UM = 0.5
GAPS_UM = tuple(g / 10 for g in range(6, 31, 2))
WAVELENGTH_UM = 1.55
MODE_COUNT = 8
SHIFTS = (2.15, 2.05)
FIELD_GRID = (241, 201)
DENSE_GRID = (482, 402)
MESH_SCALE = .7
FAMILIES = ("quasi-TE", "quasi-TM")


def _runs(model, gaps, *, solve, dry_run, grid=FIELD_GRID, scale=1., shifts=SHIFTS, count=MODE_COUNT):
    runs = {}
    for shift in shifts:
        kwargs = dict(width_um=WIDTH_UM, gaps_um=gaps, wavelength_um=WAVELENGTH_UM,
                      mode_count=count, neff_shift=shift, field_grid=grid, mesh_scale=scale)
        try:
            runs[shift] = model.modes(**kwargs, solve=solve, dry_run=dry_run, allow_partial=True)
        except (OSError, ValueError, RuntimeError) as error:
            if not solve:
                raise
            print(error, flush=True)
            runs[shift] = model.modes(**kwargs, allow_partial=True)
    return runs


def _references(runs, gap):
    refs = []
    for shift, run in runs.items():
        reference = run.run_dir / f"g_{gap:g}" / "MODES_field.json"
        if reference.is_file():
            refs.append((shift, reference))
    return refs


def _candidate_row(item, duplicate, output):
    return dict(gap_um=item.gap, shift=item.shift, outer_solnum=item.outer, solnum=item.solution,
                neff_real=item.neff.real, neff_imag=item.neff.imag, family=item.polarization,
                Ex_fraction=item.fractions[0], Ey_fraction=item.fractions[1], Ez_fraction=item.fractions[2],
                ridge_fraction=item.ridge_fraction, ridge_balance=item.ridge_balance,
                vector_parity=item.parity, dominant_component_phase_cos=item.component_phase,
                eligible=item.eligible, duplicate_of=duplicate,
                field_reference=relative_path(output, item.reference))


def _splitting(pair):
    return abs(pair[0].neff.real - pair[1].neff.real)


def _compare(checked, geometry, base_pairs):
    rows = []
    for name, gap, runs in checked:
        modes, _ = describe_candidates(_references(runs, gap), gap, geometry, WIDTH_UM)
        for family in FAMILIES:
            old = base_pairs.get((gap, family))
            new, status = select_pair(modes, family)
            comparable = old is not None and new is not None
            match = (subspace_overlap([mode.local for mode in old], [mode.local for mode in new])
                     if comparable else np.nan)
            ready = comparable and match >= OVERLAP_LIMIT and status == "paired"
            rows.append(dict(check=name, gap_um=gap, family=family,
                             status="matched" if ready else "unverified",
                             subspace_overlap=match, baseline_delta_neff=_splitting(old) if old else np.nan,
                             checked_delta_neff=_splitting(new) if new else np.nan,
                             change=abs(_splitting(new)-_splitting(old)) if ready else np.nan))
    return rows


def open_mode(report, *, gap_um, shift, outer_solnum, solnum):
    """Open one saved mode without starting COMSOL or loading other gaps."""
    report = Path(report)
    table = pd.read_csv(report / "candidates.csv")
    selected = table[(table.gap_um == gap_um) & (table["shift"] == shift)
                     & (table.outer_solnum == outer_solnum) & (table.solnum == solnum)]
    if len(selected) != 1:
        raise ValueError("Не найдено единственное поле для указанных зазора, сдвига и ID")
    return load_mode_field(report / selected.iloc[0].field_reference,
                           outer_solnum=outer_solnum, solnum=solnum)


def build_report(model, runs, pilot, checks, gaps, *, name="coupled-modes"):
    geometry = read_cross_section(model.model_path.parent, {"model": str(model.model_path)})
    target = model.results_root / "studies" / name
    rows, pairs, coverage, control = [], [], [], []
    previous, anchors = {}, {}
    with publish_study(target) as output:
        for gap in gaps:
            available = dict(_references(runs, gap))
            if gap == .8:
                available = dict(_references(pilot, gap)) | available
            refs = [(shift, available[shift]) for shift in SHIFTS if shift in available]
            coverage.append(dict(gap_um=gap, shifts_available=len(refs),
                                 missing_shifts=" ".join(str(shift) for shift in SHIFTS if shift not in available)))
            unique, candidates = describe_candidates(refs, gap, geometry, WIDTH_UM)
            rows.extend(_candidate_row(item, duplicate, output) for item, duplicate in candidates)
            has_missing_pair = False
            for family in FAMILIES:
                pair, status = select_pair(unique, family)
                track = np.nan
                prior = previous.pop(family, None)
                ids = ("", "")
                delta = signed_delta = np.nan
                if pair is not None:
                    delta = _splitting(pair)
                    if status == "paired":
                        signed_delta = pair[0].neff.real - pair[1].neff.real
                    ids = tuple(item.key for item in pair)
                    if prior is not None:
                        track = subspace_overlap([mode.local for mode in prior], [mode.local for mode in pair])
                        if track < OVERLAP_LIMIT and status == "paired":
                            status = "branch_break"
                    previous[family] = pair
                    if gap in (.8, 3.) and status == "paired":
                        anchors[gap, family] = pair
                    plot_modes(output / f"g_{gap:g}_{family}.png", pair, geometry, WIDTH_UM)
                else:
                    has_missing_pair = True
                    if not refs:
                        status = "missing"
                pairs.append(dict(gap_um=gap, family=family, status=status, first_id=ids[0], second_id=ids[1],
                                  neighbor_overlap=track, delta_neff=delta, even_minus_odd_neff=signed_delta))
            if has_missing_pair:
                localized = [mode for mode in unique if abs(mode.neff.imag) <= IMAG_LIMIT
                             and mode.ridge_fraction >= RIDGE_LIMIT]
                if localized:
                    plot_modes(output / f"g_{gap:g}_candidates.png", localized, geometry, WIDTH_UM, diagnostic=True)
            if gap == .8:
                legacy = _runs(model, [.8], solve=False, dry_run=False, grid=(21, 21), shifts=(2.,), count=2)
                old, _ = describe_candidates(_references(legacy, .8), .8, geometry, WIDTH_UM)
                for item in old:
                    nearest = min(unique, key=lambda mode: abs(mode.neff - item.neff), default=None)
                    control.append(dict(old_id=item.key, old_neff_real=item.neff.real, old_neff_imag=item.neff.imag,
                                        new_id=nearest.key if nearest else "",
                                        delta_neff=abs(item.neff-nearest.neff) if nearest else np.nan))
        verification = _compare(checks, geometry, anchors)
        for row in pairs:
            mesh_checks = [item for item in verification if item["family"] == row["family"]
                           and item["check"] == "finer_mesh" and item["status"] == "matched"]
            point_checks = [item for item in mesh_checks if item["gap_um"] == row["gap_um"]]
            errors = [item["change"] for item in point_checks or mesh_checks]
            bound = max([1e-10, *errors]) if errors else np.nan
            row["mesh_delta_neff_change"] = bound
            row["mesh_check_at_gap"] = bool(point_checks)
            row["L_pi_um"] = np.nan
            row["P_cross_200um"] = np.nan
            row["P_cross_max_to_200um"] = np.nan
            if row["status"] == "paired":
                if not np.isfinite(bound):
                    row["status"] = "awaiting_mesh_check"
                elif row["delta_neff"] <= 2 * bound:
                    row["status"] = "unresolved_splitting"
                else:
                    row["L_pi_um"] = WAVELENGTH_UM / (2 * row["delta_neff"])
                    angle = np.pi * row["delta_neff"] * 200 / WAVELENGTH_UM
                    row["P_cross_200um"] = np.sin(angle)**2
                    row["P_cross_max_to_200um"] = 1. if row["L_pi_um"] <= 200 else row["P_cross_200um"]
        candidate_columns = ["gap_um", "shift", "outer_solnum", "solnum", "neff_real", "neff_imag",
                             "family", "Ex_fraction", "Ey_fraction", "Ez_fraction", "ridge_fraction",
                             "ridge_balance", "vector_parity", "dominant_component_phase_cos",
                             "eligible", "duplicate_of", "field_reference"]
        for name, data, columns in (("candidates", rows, candidate_columns), ("pairs", pairs, None),
                                    ("coverage", coverage, None), ("verification", verification, None),
                                    ("legacy_check", control, ["old_id","old_neff_real","old_neff_imag","new_id","delta_neff"])):
            write_csv(output / f"{name}.csv", pd.DataFrame(data, columns=columns))
        write_json(output / "config.json", dict(
            width_um=WIDTH_UM, wavelength_um=WAVELENGTH_UM, gaps_um=list(gaps),
            shifts=list(SHIFTS), mode_count=MODE_COUNT, field_grid=list(FIELD_GRID),
            model=relative_path(output, model.model_path), refined_mesh_scale=MESH_SCALE,
            polarization_limit=POLARIZATION_LIMIT, imag_limit=IMAG_LIMIT, ridge_limit=RIDGE_LIMIT,
            parity_limit=PARITY_LIMIT, overlap_limit=OVERLAP_LIMIT,
            input_runs={str(shift): relative_path(output, run.run_dir) for shift, run in runs.items()},
        ))
        if any(np.isfinite(row["delta_neff"]) for row in pairs):
            plot_coupling(output, pairs, WAVELENGTH_UM)
    print(f"REPORT_DIR={target}", flush=True)
    return sum(row["shifts_available"] != len(SHIFTS) for row in coverage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--results-root", type=Path, default=PROJECT / "results")
    parser.add_argument("--pilot", action="store_true", help="Study only gap 0.8 µm")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--solve", action="store_true", help="Calculate missing points")
    action.add_argument("--dry-run", action="store_true", help="Show the requested points and cache status")
    args = parser.parse_args()
    try:
        model = CoupledWaveguides2D(args.model, results_root=args.results_root)
        options = dict(solve=args.solve, dry_run=args.dry_run)
        pilot = _runs(model, [.8], **options)
        dense = _runs(model, [.8], **options, grid=DENSE_GRID)
        fine_pilot = _runs(model, [.8], **options, scale=MESH_SCALE)
        runs = pilot if args.pilot else _runs(model, GAPS_UM, **options)
        checks = [("dense_sampling", .8, dense), ("finer_mesh", .8, fine_pilot)]
        if not args.pilot:
            fine_end = _runs(model, [3.], **options, scale=MESH_SCALE)
            checks.append(("finer_mesh", 3., fine_end))
        if not args.dry_run:
            name = "coupled-modes-pilot" if args.pilot else "coupled-modes"
            missing = build_report(model, runs, pilot, checks, [.8] if args.pilot else GAPS_UM, name=name)
            if args.solve and missing:
                parser.exit(1, f"Не все сдвиги расчёта готовы для {missing} зазоров. См. coverage.csv.\n")
            if args.solve and any(len(_references(check_runs, gap)) < len(SHIFTS)
                                  for _, gap, check_runs in checks):
                parser.exit(1, "Не все контрольные расчёты готовы. См. verification.csv.\n")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
