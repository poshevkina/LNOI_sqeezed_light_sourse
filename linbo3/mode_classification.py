from .storage import analysis_root
"""Central-wavelength quasi-TE/TM classification from complex transverse fields.

Convention for this xy cross-section, propagation along z: TE is predominantly
Ex (parallel to the film), TM predominantly Ey (normal to the film). Labels are
shape diagnostics, not COMSOL solution numbers or a ranking by effective index.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .saved_modes import load_pair


@dataclass(frozen=True)
class CrossSection:
    core_height_um: float
    film_height_um: float
    sidewall_angle_deg: float


@dataclass(frozen=True)
class ClassificationReport:
    output_dir: Path
    table: pd.DataFrame
    figures: tuple[Path, ...]


def _literal(expression, kind):
    match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\[([^]]+)\]\s*", expression)
    scales = {"m": 1e6, "um": 1., "nm": .001} if kind == "length" else {"deg": 1.}
    if match is None or match[2] not in scales:
        raise ValueError(f"Не удалось прочитать параметр геометрии {expression!r}; укажи CrossSection явно")
    return float(match[1]) * scales[match[2]]


def read_cross_section(run_dir, manifest):
    model = Path(manifest["model"])
    if not model.is_absolute():
        base = Path(run_dir) if manifest.get("model_path_base") == "run_dir" else Path(run_dir) / "../.."
        model = (base / model).resolve()
    with ZipFile(model) as archive:
        root = ET.fromstring(archive.read("dmodel.xml"))
    params = {node.get("name"): node.get("expr") for node in root.iter("expressions")}
    params.update(manifest.get("geometry", {}))
    core = _literal(params["h_core"], "length")
    film = (core - _literal(params["h_trap"], "length") if params.get("h_film") == "h_core-h_trap"
            else _literal(params["h_film"], "length"))
    return CrossSection(core, film, _literal(params["theta"], "angle"))


def _axis_weights(axis):
    if len(axis) < 3 or not np.all(np.diff(axis) > 0):
        raise ValueError("Для классификации нужны хотя бы три координаты по каждой оси")
    edges = np.r_[axis[0] - (axis[1]-axis[0])/2, (axis[1:]+axis[:-1])/2,
                  axis[-1] + (axis[-1]-axis[-2])/2]
    return np.diff(edges)


def field_grid(coordinates, field):
    """Map actual samples to xy; missing samples remain NaN, never synthesized."""
    xy = np.asarray(coordinates) * 1e6  # COMSOL geometry coordinates are metres in this model.
    if xy.ndim != 2 or xy.shape[0] != 2 or field.shape != (3, xy.shape[1]):
        raise ValueError("Ожидались coordinates[2,N] и комплексное поле [3,N]")
    x, y = np.unique(xy[0]), np.unique(xy[1])
    ix, iy = np.searchsorted(x, xy[0]), np.searchsorted(y, xy[1])
    if len(np.unique(iy * len(x) + ix)) != xy.shape[1]:
        raise ValueError("Координаты поля повторяются")
    grid = np.full((3, len(y), len(x)), np.nan + 1j*np.nan)
    grid[:, iy, ix] = field
    return x, y, grid


def _peaks(profile, prominence, height):
    # Padding allows a maximum at an actual end sample to be diagnosed as truncated later.
    return find_peaks(np.r_[0., profile, 0.], prominence=prominence, height=height)[0] - 1


def classify_field(coordinates, field, width_um, geometry, *, polarization_threshold=.9,
                   peak_prominence=.08, peak_height=.1):
    """Classify a sampled field; n = lateral lobe count - 1, no neff ranking."""
    if not .5 < polarization_threshold <= 1 or not 0 < peak_prominence < 1 or not 0 < peak_height < 1:
        raise ValueError("Нужны 0.5 < polarization_threshold ≤ 1 и пороги пиков между 0 и 1")
    h, film, angle = geometry.core_height_um, geometry.film_height_um, geometry.sidewall_angle_deg
    if not (0 <= film < h and 0 < angle < 90 and width_um > 0):
        raise ValueError("Проверь высоту ядра/плёнки, ширину и угол стенки")
    x, y, grid = field_grid(coordinates, field)
    dx, dy = _axis_weights(x), _axis_weights(y)
    xx, yy = np.meshgrid(x, y)
    valid = np.isfinite(grid).all(axis=0)
    core = (yy >= 0) & (yy <= h) & (np.abs(xx) <= width_um/2 + (h-yy)/np.tan(np.deg2rad(angle)))
    if np.count_nonzero(core & valid) < 6 or np.count_nonzero(valid & core) < .9*np.count_nonzero(core):
        raise ValueError("Недостаточно отсчётов поля в ядре для классификации")
    intensity = np.abs(grid)**2
    area = dy[:, None] * dx[None, :]
    powers = np.array([np.sum(a[core & valid] * area[core & valid]) for a in intensity])
    if not np.isfinite(powers).all() or powers[:2].sum() <= 0:
        raise ValueError("Нет конечного ненулевого поперечного поля в ядре")
    fraction = float(powers[0] / powers[:2].sum())
    dominant = 0 if fraction >= .5 else 1
    purity = max(fraction, 1-fraction)
    # A strip through the waveguide height: integrate over y, resolve lateral lobes along x.
    strip = (y >= 0) & (y <= h)
    roi = np.abs(x) <= width_um/2 + h/np.tan(np.deg2rad(angle)) + 1.5
    profiles, peaks, stable = [], [], []
    for component in (0, 1):
        profile = np.nansum(intensity[component, strip] * dy[strip, None], axis=0)
        profile[~roi] = 0
        if profile.max() <= 0:
            profile = np.zeros_like(profile)
        else:
            profile /= profile.max()
        positions = _peaks(profile, peak_prominence, peak_height)
        positions = positions[(positions >= 0) & (positions < len(x))]
        counts = [len(_peaks(profile, level, peak_height)) for level in
                  (peak_prominence*.5, peak_prominence, min(.95, peak_prominence*1.5))]
        profiles.append(profile); peaks.append(positions)
        stable.append(len(set(counts)) == 1 and len(positions) > 0)
    lobes = len(peaks[dominant])
    order = lobes - 1
    family = "TE" if dominant == 0 else "TM"
    candidate = f"{family}{order}" if lobes else "Unknown"
    support = int(np.count_nonzero(profiles[dominant] >= peak_height))
    samples_per_lobe = support/max(1, lobes)
    truncated = bool(np.any(peaks[dominant] == 0) or np.any(peaks[dominant] == len(x)-1))
    reasons = []
    if powers[2]/powers.sum() > .5:
        reasons.append("longitudinal_dominated")
    if purity < polarization_threshold-1e-12:
        reasons.append("mixed_polarization")
    if not stable[dominant]:
        reasons.append("unstable_peak_count")
    if samples_per_lobe < 3:
        reasons.append("coarse_sampling")
    if truncated:
        reasons.append("truncated_profile")
    if purity < polarization_threshold-1e-12:
        alternatives = [f"{('TE','TM')[k]}{len(peaks[k])-1}" for k in (dominant, 1-dominant) if len(peaks[k])]
        label = "Hybrid " + "/".join(alternatives)
    else:
        label = candidate
    if any(reason != "mixed_polarization" for reason in reasons):
        label += " ?"
    record = dict(label=label, candidate_label=candidate, family=family, lateral_order=order,
                  lateral_peaks=lobes, te_fraction=fraction, polarization_purity=purity,
                  longitudinal_fraction=float(powers[2]/powers.sum()),
                  ex_peaks=len(peaks[0]), ey_peaks=len(peaks[1]), stable_peaks=bool(stable[dominant]),
                  samples_per_lobe=samples_per_lobe, core_samples=int(np.count_nonzero(core & valid)),
                  dx_um=float(np.median(dx)), dy_um=float(np.median(dy)),
                  status="confident" if not reasons else "review", reason=";".join(reasons),
                  peak_x_um=json.dumps(x[peaks[dominant]].tolist()))
    plot = dict(x=x, y=y, grid=grid, profile=profiles[dominant], peaks=peaks[dominant], dominant=dominant)
    return record, plot


def _draw_width(path, width, entries, geometry, field_view):
    import matplotlib.pyplot as plt
    ncols = max(1, *(len(entries[band]) for band in ("IS", "PUMP")))
    fig = plt.figure(figsize=(4.4*ncols, 5.7), layout="constrained")
    outer = fig.add_gridspec(2, 1)
    axes, color_axes = [], []
    for band in ("IS", "PUMP"):
        count = max(1, len(entries[band]))
        inner = outer[len(axes)].subgridspec(1, count+1, width_ratios=[1]*count+[.035])
        axes.append([fig.add_subplot(inner[0, col]) for col in range(count)])
        color_axes.append(fig.add_subplot(inner[0, -1]))
    half = max(1.6, width/2 + geometry.core_height_um/np.tan(np.deg2rad(geometry.sidewall_angle_deg)) + .8)
    try:
        for row, band in enumerate(("IS", "PUMP")):
            row_mesh = None
            for col in range(max(1, len(entries[band]))):
                ax = axes[row][col]
                if col >= len(entries[band]):
                    ax.set_visible(False)
                    continue
                record, plot = entries[band][col]
                x, y, grid = plot["x"], plot["y"], plot["grid"]
                amplitude = (np.sqrt(np.nansum(np.abs(grid)**2, axis=0)) if field_view == "total"
                             else np.abs(grid[plot["dominant"]]))
                maximum = np.nanmax(amplitude)
                amplitude = amplitude/maximum if maximum > 0 else amplitude
                row_mesh = ax.pcolormesh(x, y, amplitude, shading="gouraud", cmap="inferno", vmin=0, vmax=1)
                h, film = geometry.core_height_um, geometry.film_height_um
                ywall = np.array([film, h]); xwall = width/2 + (h-ywall)/np.tan(np.deg2rad(geometry.sidewall_angle_deg))
                for sign in (-1, 1):
                    ax.plot(sign*xwall, ywall, "w--", lw=.8)
                ax.plot([-width/2, width/2], [h, h], "w--", lw=.8)
                ax.axhline(film, color="white", ls="--", lw=.6, alpha=.7)
                ax.axhline(0, color="white", ls="--", lw=.6, alpha=.7)
                ax.set(xlim=(-half, half), ylim=(-.4, h+.7), xlabel="x, мкм",
                       ylabel=f"{band} · {record['wavelength_um']*1000:g} нм\ny, мкм" if col == 0 else "y, мкм")
                ax.set_aspect("equal")
                ax.set_title(f"{record['label']} · solnum {record['solnum']}\n"
                             f"n_eff={record['neff_real']:.5f} · f_TE={record['te_fraction']:.0%}", fontsize=10,
                             color="crimson" if record['polarization_purity'] < .9-1e-12 else "black")
                inset = ax.inset_axes([.61, .05, .36, .25])
                inset.plot(x, plot["profile"], color="#256aa6", lw=1)
                positions = plot["peaks"]
                inset.plot(x[positions], plot["profile"][positions], "r.", ms=4)
                inset.set(xlim=(-half, half), ylim=(-.05, 1.15), xticks=[], yticks=[])
                inset.text(.03, .96, f"∫|E{('x','y')[plot['dominant']]}|²dy · пиков: {record['lateral_peaks']}",
                           fontsize=6, va="top", transform=inset.transAxes,
                           bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=.2))
            visible = [axes[row][c] for c in range(len(entries[band]))]
            if row_mesh is not None:
                fig.colorbar(row_mesh, cax=color_axes[row], label="|E| / max|E|" if field_view == "total" else "|E_dom| / max|E_dom|")
            elif not entries[band]:
                color_axes[row].set_visible(False)
                ax = axes[row][0]; ax.set_visible(True); ax.set_axis_off()
                ax.text(.5, .5, f"{band}: нет допустимых опорных решений", ha="center", va="center", transform=ax.transAxes)
        fig.suptitle(f"w={width:g} мкм · классификация по полям на центральной частоте\n"
                     "Номер = число боковых пиков − 1; красным: Hybrid (примесь >10%); ? требует проверки", fontsize=12)
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def classify_run(run_dir, *, imag_limit=1e-6, polarization_threshold=.9,
                 peak_prominence=.08, peak_height=.1, geometry=None, branches=None,
                 field_view="total", output_dir=None):
    """Classify all low-loss anchor solutions, even if branch tracking failed.

    Does not select a signal/idler pair or modify existing tracking/SPDC results.
    """
    if not np.isfinite(imag_limit) or imag_limit <= 0 or field_view not in ("total", "dominant"):
        raise ValueError("Нужен положительный imag_limit и field_view='total' или 'dominant'")
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir/"config.json").read_text(encoding="utf-8-sig"))
    status = json.loads((run_dir/"status.json").read_text(encoding="utf-8-sig"))
    geometry = geometry or read_cross_section(run_dir, manifest)
    output = Path(output_dir) if output_dir else analysis_root(run_dir)/"mode_classification"
    output.mkdir(parents=True, exist_ok=True)
    anchors = {"IS": 2*manifest["pump_wavelength_um"], "PUMP": manifest["pump_wavelength_um"]}
    mapping = {}
    if branches is not None and not branches.empty:
        for item in branches.loc[branches["anchor"]].to_dict("records"):
            mapping[(item["width_um"], item["band"], item["outer_solnum"], item["solnum"])] = item["mode"]
    records, figures, errors = [], [], []
    for width in sorted(manifest["widths_um"]):
        key = f"{width:g}"; entries = {"IS": [], "PUMP": []}
        for band in entries:
            pair_state = status.get("pairs", {}).get(key, {}).get(band, {}).get("state")
            if pair_state is not None and pair_state != "complete":
                errors.append(dict(width_um=width, band=band, message=f"COMSOL: {pair_state}"))
                continue
            try:
                _, raw = load_pair(run_dir/f"w_{key}", band, manifest["wavelengths_um"][band])
                indices = np.flatnonzero(np.isclose(raw.wavelength_um, anchors[band], rtol=0, atol=1e-10))
                if not len(indices):
                    raise ValueError("Нет точной опорной длины волны")
            except (OSError, ValueError, KeyError) as error:
                errors.append(dict(width_um=width, band=band, message=str(error)))
                continue
            for i in indices:
                record = dict(width_um=width, band=band, wavelength_um=float(raw.wavelength_um[i]),
                              outer_solnum=int(raw.outer_solnum[i]), solnum=int(raw.solnum[i]),
                              neff_real=float(raw.neff[i].real), neff_imag=float(raw.neff[i].imag))
                record["internal_mode"] = mapping.get((width, band, record["outer_solnum"], record["solnum"]))
                if not np.isfinite(raw.neff[i]) or abs(raw.neff[i].imag) >= imag_limit:
                    record.update(label="Excluded", status="excluded", reason="imag_limit_or_nonfinite_neff")
                    records.append(record); continue
                try:
                    classification, plot = classify_field(raw.coordinates, raw.field[i], width, geometry,
                        polarization_threshold=polarization_threshold, peak_prominence=peak_prominence,
                        peak_height=peak_height)
                except ValueError as error:
                    record.update(label="Unknown", status="review", reason=str(error))
                    records.append(record); continue
                record.update(classification)
                entries[band].append((record, plot)); records.append(record)
            # A one-index description cannot uniquely identify two modes with the same label.
            for item, _ in entries[band]:
                if item["status"] == "confident" and sum(
                    other["candidate_label"] == item["candidate_label"] for other, _ in entries[band]) > 1:
                    item.update(label=item["candidate_label"]+" ?", status="review", reason="duplicate_spatial_label")
            entries[band].sort(key=lambda e: ({"TM": 0, "TE": 1}.get(e[0].get("family"), 2),
                                              e[0].get("lateral_order", 0), e[0]["solnum"]))
        path = output/f"central_modes_w_{key}.png"
        _draw_width(path, width, entries, geometry, field_view)
        figures.append(path)
    table = pd.DataFrame(records)
    table.to_csv(output/"classification.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors, columns=["width_um", "band", "message"]).to_csv(output/"errors.csv", index=False, encoding="utf-8-sig")
    (output/"settings.json").write_text(json.dumps(dict(algorithm_version=1, anchors_um=anchors,
        geometry=asdict(geometry), imag_limit=imag_limit, polarization_threshold=polarization_threshold,
        peak_prominence=peak_prominence, peak_height=peak_height, field_view=field_view,
        convention="xy cross-section; z propagation; TE: Ex; TM: Ey; lateral order = lobes - 1"), indent=2), encoding="utf-8")
    return ClassificationReport(output, table, tuple(figures))
