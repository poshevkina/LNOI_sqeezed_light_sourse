"""Identify and follow eigenmodes using sampled complex vector fields."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .dispersion import Mode


@dataclass(frozen=True)
class ModalSolutions:
    wavelength_um: np.ndarray
    outer_solnum: np.ndarray
    solnum: np.ndarray
    neff: np.ndarray
    field: np.ndarray
    coordinates: np.ndarray

    def arrays(self):
        return vars(self)


@dataclass(frozen=True)
class TrackedModes:
    modes: dict[int, Mode]
    rows: list[dict]


class ModeTrackingError(ValueError):
    pass


def selection_masks(raw: ModalSolutions, imag_limit: float):
    """The same strict loss and finite-field criteria are used at every wavelength."""
    if not np.isfinite(imag_limit) or imag_limit <= 0:
        raise ValueError("imag_limit должен быть конечным и положительным")
    field = raw.field.reshape(len(raw.neff), -1)
    norms = np.linalg.norm(field, axis=1)
    usable = np.isfinite(raw.neff) & np.isfinite(norms) & (norms > 0)
    passes_imag = np.isfinite(raw.neff) & (np.abs(raw.neff.imag) < imag_limit)
    return usable, passes_imag, usable & passes_imag


def track_modes(
    raw: ModalSolutions,
    *,
    band: str,
    anchor_um: float,
    imag_limit: float,
    min_overlap: float,
    min_margin: float,
    anchor_solnums: list[int] | None = None,
) -> TrackedModes:
    """Follow eligible fields in both directions; never bypass the strict loss limit."""
    if not 0 <= min_overlap <= 1 or not 0 <= min_margin <= 1:
        raise ValueError("Пороги перекрытия и запаса должны лежать от 0 до 1")
    wavelengths = np.unique(raw.wavelength_um)
    matches = np.flatnonzero(np.isclose(wavelengths, anchor_um, atol=1e-10, rtol=0))
    if len(matches) != 1:
        raise ModeTrackingError(f"{band}: нет единственной опорной точки {anchor_um:g} мкм")
    center = int(matches[0])
    groups = [np.flatnonzero(raw.wavelength_um == wl) for wl in wavelengths]
    anchor = groups[center]
    if len(np.unique(raw.solnum[anchor])) != len(anchor):
        raise ModeTrackingError(f"{band}: идентификаторы решений в опорной точке неоднозначны")

    field = raw.field.reshape(len(raw.neff), -1)
    norms = np.linalg.norm(field, axis=1)
    usable, passes_imag, eligible = selection_masks(raw, imag_limit)

    def describe(indices):
        return ", ".join(f"{raw.solnum[i]}: {raw.neff[i]:.8g}" for i in indices)

    if anchor_solnums is None:
        selected = anchor[eligible[anchor]]
        selected = selected[np.argsort(raw.neff[selected].real, kind="stable")]
    else:
        available = {int(raw.solnum[i]): i for i in anchor if eligible[i]}
        try:
            selected = np.array([available[number] for number in anchor_solnums], dtype=int)
        except KeyError as error:
            raise ModeTrackingError(
                f"{band} {anchor_um:g} мкм: выбранное решение {error} недоступно "
                f"или не проходит |Im(n_eff)| < {imag_limit:g} и проверку поля. "
                f"Решения в опорной точке: {describe(anchor)}"
            ) from error
    if (not len(selected) or len(np.unique(selected)) != len(selected)
            or (band == "IS" and len(selected) != 2)):
        raise ModeTrackingError(
            f"{band} {anchor_um:g} мкм: выбрано {len(selected)} решений. "
            f"Опорные решения (solnum: neff): {describe(anchor)}"
        )

    vectors = np.zeros_like(field, dtype=complex)
    vectors[usable] = field[usable] / norms[usable, None]
    indices = np.empty((len(wavelengths), len(selected)), dtype=int)
    indices[center] = selected
    scores = np.ones(indices.shape)
    margins = np.ones(indices.shape)
    directions = (range(center + 1, len(wavelengths)), range(center - 1, -1, -1))
    for direction in directions:
        prev = selected
        for point in direction:
            candidates = groups[point][eligible[groups[point]]]
            if len(candidates) < len(prev):
                raise ModeTrackingError(
                    f"{band} {wavelengths[point]:g} мкм: недостаточно решений "
                    f"с |Im(n_eff)| < {imag_limit:g} и конечным ненулевым полем. "
                    f"Доступны {describe(candidates)}"
                )
            overlap = np.minimum(1.0, np.abs(vectors[prev].conj() @ vectors[candidates].T) ** 2)
            rows, cols = linear_sum_assignment(overlap, maximize=True)
            chosen = overlap[rows, cols]
            alternatives = overlap.copy()
            alternatives[rows, cols] = 0
            margin = chosen - alternatives.max(axis=1)
            uncertain = (chosen < min_overlap) | (margin < min_margin)
            if np.any(uncertain):
                details = ", ".join(
                    f"{raw.solnum[prev[r]]} → {raw.solnum[candidates[c]]}: "
                    f"перекрытие {score:.3f}, запас {gap:.3f}"
                    for r, c, score, gap in zip(rows[uncertain], cols[uncertain],
                                                chosen[uncertain], margin[uncertain], strict=True)
                )
                raise ModeTrackingError(f"{band} {wavelengths[point]:g} мкм: неоднозначная ветвь ({details})")
            prev = candidates[cols]
            indices[point] = prev
            scores[point] = chosen
            margins[point] = margin

    modes = {}
    records = []
    for block, track in enumerate(indices.T):
        modes[block] = Mode(wavelengths, raw.neff[track].real)
        for point, row in enumerate(track):
            records.append(dict(
                band=band, mode=block, wavelength_um=wavelengths[point],
                outer_solnum=int(raw.outer_solnum[row]), solnum=int(raw.solnum[row]),
                neff_real=raw.neff[row].real, neff_imag=raw.neff[row].imag,
                overlap=scores[point, block], overlap_margin=margins[point, block],
                anchor=point == center, passes_imag_limit=bool(passes_imag[row]),
            ))
    return TrackedModes(modes, records)
