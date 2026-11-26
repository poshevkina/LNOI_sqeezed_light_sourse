from .storage import analysis_root
"""Inspect completed COMSOL exports, with auditable selection and no solver calls."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .modes import ModalSolutions, ModeTrackingError, selection_masks, track_modes


TRACKING = dict(imag_limit=1e-6, min_overlap=0.9, min_margin=0.1)
BANDS = ("PUMP", "IS")
BRANCH_COLUMNS = ["width_um", "band", "mode", "wavelength_um", "outer_solnum", "solnum",
                  "neff_real", "neff_imag", "overlap", "overlap_margin", "anchor", "passes_imag_limit"]


@dataclass(frozen=True)
class ModeInspection:
    output_dir: Path
    modes: dict
    pairs: pd.DataFrame
    selection: pd.DataFrame
    branches: pd.DataFrame
    tracking: dict
    anchor_solnums: dict


def load_pair(directory, band, wavelengths):
    table = pd.read_csv(directory / f"{band}_raw.csv", float_precision="round_trip")
    with np.load(directory / f"{band}_raw.npz", allow_pickle=False) as file:
        raw = ModalSolutions(**{name: file[name] for name in ModalSolutions.__annotations__})
    for column in ("outer_solnum", "solnum"):
        if not np.array_equal(table[column], getattr(raw, column)):
            raise ValueError(f"{band}: идентификаторы решений в CSV и NPZ различаются")
    for column, values in (("wavelength_um", raw.wavelength_um), ("neff_real", raw.neff.real),
                           ("neff_imag", raw.neff.imag)):
        if len(table[column]) != len(values) or not np.allclose(table[column], values, rtol=0, atol=1e-12,
                                                               equal_nan=True):
            raise ValueError(f"{band}: столбец {column} не соответствует NPZ")
    if not np.array_equal(np.unique(raw.wavelength_um), wavelengths):
        raise ValueError(f"{band}: сетка длин волн не соответствует манифесту")
    return table, raw


def inspect_modes(run_dir, *, output_dir=None, tracking=None, anchor_solnums=None):
    """Plot every width/band, including rejected points and failed or incomplete pairs."""
    from .interpolation_plots import plot_raw

    run_dir = Path(run_dir)
    output = Path(output_dir) if output_dir is not None else analysis_root(run_dir) / "mode_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run_dir / "config.json").read_text(encoding="utf-8-sig"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8-sig"))
    thresholds = {key: manifest.get(key, default) for key, default in TRACKING.items()}
    thresholds.update(tracking or {})
    overrides = manifest.get("anchor_solnums", {}) if anchor_solnums is None else anchor_solnums
    pump = manifest.get("pump_wavelength_um", 0.775)
    anchors = {"PUMP": pump, "IS": 2 * pump}
    states = status.get("pairs")
    if states is None:
        states = {f"{width:g}": {band: {"state": "complete"} for band in BANDS}
                  for width in manifest["widths_um"]}

    all_modes, coverage, raw_frames, selection_frames, branch_frames = {}, [], [], [], []
    for key, bands in sorted(states.items(), key=lambda pair: float(pair[0])):
        width = float(key)
        directory = output / f"w_{width:g}"
        directory.mkdir(exist_ok=True)
        raw_tables, branches, tracks = {}, {}, {}
        for band in BANDS:
            state = bands.get(band, {})
            row = dict(width_um=width, band=band, comsol_state=state.get("state", "missing"),
                       analysis_state="not_ready", message=state.get("error", ""), n_branches=0)
            coverage.append(row)
            if state.get("state") != "complete":
                continue
            try:
                table, raw = load_pair(run_dir / f"w_{key}", band, manifest["wavelengths_um"][band])
                usable, passes_imag, eligible = selection_masks(raw, thresholds["imag_limit"])
            except (FileNotFoundError, ValueError, KeyError, OSError) as error:
                row.update(analysis_state="invalid_export", message=str(error))
                continue
            audit = table.assign(width_um=width, band=band, field_usable=usable,
                                 passes_imag_limit=passes_imag, eligible=eligible,
                                 abs_neff_imag=np.abs(raw.neff.imag),
                                 anchor=np.isclose(raw.wavelength_um, anchors[band], atol=1e-10, rtol=0))
            chosen = overrides.get(key, {}).get(band)
            audit["anchor_candidate"] = audit.anchor & audit.eligible
            if chosen is not None:
                audit["anchor_candidate"] &= audit.solnum.isin(chosen)
            audit["selected"] = False
            try:
                tracked = track_modes(raw, band=band, anchor_um=anchors[band],
                                      **thresholds, anchor_solnums=chosen)
            except ModeTrackingError as error:
                row.update(analysis_state="ambiguous", message=str(error))
            else:
                tracks[band] = tracked.modes
                branches[band] = pd.DataFrame(tracked.rows)
                branch_frames.append(branches[band].assign(width_um=width))
                selected_ids = set(zip(branches[band].outer_solnum, branches[band].solnum))
                audit["selected"] = [(outer, inner) in selected_ids
                                     for outer, inner in zip(raw.outer_solnum, raw.solnum)]
                row.update(analysis_state="tracked", n_branches=len(tracked.modes))
            raw_tables[band] = audit
            raw_frames.append(table.assign(width_um=width, band=band))
            selection_frames.append(audit)
        # Even an entirely unavailable width gets a figure explaining the missing data.
        plot_raw(directory / "neff_raw.png", width, raw_tables, branches,
                 imag_limit=thresholds["imag_limit"], anchors=anchors)
        if tracks:
            all_modes[width] = tracks
    pairs = pd.DataFrame(coverage, columns=["width_um", "band", "comsol_state", "analysis_state",
                                           "message", "n_branches"])
    branch_table = (pd.concat(branch_frames, ignore_index=True) if branch_frames
                    else pd.DataFrame(columns=BRANCH_COLUMNS))
    selection = (pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame())
    pairs.to_csv(output / "pairs.csv", index=False, encoding="utf-8-sig")
    branch_table.to_csv(output / "branches.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(output / "selection_audit.csv", index=False, encoding="utf-8-sig")
    if raw_frames:
        pd.concat(raw_frames, ignore_index=True).to_csv(output / "neff_raw.csv", index=False)
    (output / "tracking_settings.json").write_text(json.dumps({
        "tracking": thresholds, "anchor_solnums": overrides, "imag_rule": "strict_at_every_wavelength",
    }, indent=2), encoding="utf-8")
    return ModeInspection(output, all_modes, pairs, selection, branch_table, thresholds, overrides)
