"""Geometry, polarization and reflection checks for the symmetric 2D coupler."""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ...artifacts import read_json
from ...mode_classification import field_grid
from ...saved_modes import load_modal_npz


POLARIZATION_LIMIT = .90
IMAG_LIMIT = 1e-6
RIDGE_LIMIT = .08
PARITY_LIMIT = .90
OVERLAP_LIMIT = .85
DUPLICATE_NEFF = 1e-7


@dataclass(frozen=True)
class Profile:
    x_um: np.ndarray
    y_um: np.ndarray
    electric: np.ndarray


@dataclass(frozen=True)
class Candidate:
    gap: float
    shift: float
    outer: int
    solution: int
    neff: complex
    reference: Path
    profile: Profile
    polarization: str
    fractions: np.ndarray
    ridge_fraction: float
    ridge_balance: float
    parity: float
    component_phase: float
    local: np.ndarray

    @property
    def key(self):
        return f"{self.shift:g}:{self.outer}:{self.solution}"

    @property
    def eligible(self):
        return (np.isfinite(self.neff) and abs(self.neff.imag) <= IMAG_LIMIT
                and self.ridge_fraction >= RIDGE_LIMIT and self.ridge_balance >= .20
                and self.polarization in ("quasi-TE", "quasi-TM"))


def load_raw(reference):
    reference = Path(reference)
    info = read_json(reference)
    return load_modal_npz(reference.parent / info["npz"], expected_sha256=info["field_sha256"])


def load_mode_field(reference, *, outer_solnum, solnum):
    """Load one complex E[x,y,z] profile, shaped (3, len(y_um), len(x_um))."""
    raw = load_raw(reference)
    indices = np.flatnonzero((raw.outer_solnum == outer_solnum) & (raw.solnum == solnum))
    if len(indices) != 1:
        raise ValueError(f"Ожидалось одно решение {outer_solnum}/{solnum}, получено {len(indices)}")
    return Profile(*field_grid(raw.coordinates, raw.field[indices[0]]))


def ridge_polygons(geometry, width, gap):
    """Coordinates of the two trapezoids in geom1, in micrometres."""
    h, film = geometry.core_height_um, geometry.film_height_um
    side = (h - film) / np.tan(np.deg2rad(geometry.sidewall_angle_deg))
    inner = gap / 2
    right = np.array([[inner-side, film], [inner, h], [inner+width, h], [inner+width+side, film]])
    return right * [-1, 1], right


def overlap(a, b):
    a, b = a.ravel(), b.ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    denominator = np.sqrt(np.sum(abs(a)**2) * np.sum(abs(b)**2))
    return np.sum(a.conj() * b) / denominator if denominator else 0j


def _local_field(profile, geometry, width, gap):
    x = np.linspace(-.65, .65, 65) + (gap + width) / 2
    y = np.linspace(geometry.film_height_um-.25, geometry.core_height_um+.35, 46)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    interpolate = RegularGridInterpolator(
        (profile.y_um, profile.x_um), profile.electric.transpose(1, 2, 0),
        bounds_error=False, fill_value=np.nan,
    )
    return interpolate((yy, xx))


def _describe(raw, index, gap, shift, reference, geometry, width):
    profile = Profile(*field_grid(raw.coordinates, raw.field[index]))
    x, y, electric = profile.x_um, profile.y_um, profile.electric
    if len(x) < 3 or len(y) < 3 or not np.allclose(x, -x[::-1], atol=1e-8, rtol=0):
        raise ValueError("Сетка x не симметрична относительно середины двух сердечников")
    if not (np.allclose(np.diff(x), np.diff(x)[0]) and np.allclose(np.diff(y), np.diff(y)[0])):
        raise ValueError("Ожидались равномерные шаги по x и y")
    power = np.nansum(abs(electric)**2, axis=(1, 2))
    total = power.sum()
    fractions = power / total if total else np.zeros(3)
    axis = int(np.argmax(power[:2]))
    transverse = power[0] + power[1]
    polarization = ("quasi-TE", "quasi-TM")[axis] if transverse and power[axis]/transverse >= POLARIZATION_LIMIT else "mixed"

    _, right = ridge_polygons(geometry, width, gap)
    taper = (right[1, 1] - y[:, None]) / np.tan(np.deg2rad(geometry.sidewall_angle_deg))
    ridge = ((y[:, None] >= right[0, 1]) & (y[:, None] <= right[1, 1])
             & (x >= right[1, 0] - taper) & (x <= right[2, 0] + taper))
    intensity = np.nansum(abs(electric)**2, axis=0)
    right_power = intensity[ridge].sum()
    left_power = intensity[ridge[:, ::-1]].sum()
    captured = left_power + right_power
    reflected = electric[:, :, ::-1] * np.array([-1, 1, 1])[:, None, None]
    return Candidate(
        gap, shift, int(raw.outer_solnum[index]), int(raw.solnum[index]), raw.neff[index],
        reference, profile, polarization, fractions, captured / total if total else 0.,
        min(left_power, right_power) / captured if captured else 0.,
        overlap(electric, reflected).real,
        overlap(electric[axis][ridge], electric[axis, :, ::-1][ridge]).real,
        _local_field(profile, geometry, width, gap),
    )


def describe_candidates(references, gap, geometry, width):
    """Keep every solution and flag duplicates returned by another search shift."""
    unique, candidates = [], []
    for shift, reference in references:
        raw = load_raw(reference)
        for index in range(len(raw.neff)):
            item = _describe(raw, index, gap, shift, reference, geometry, width)
            duplicate = next((other.key for other in unique
                              if other.shift != shift and abs(other.neff-item.neff) <= DUPLICATE_NEFF
                              and abs(overlap(other.profile.electric, item.profile.electric)) >= .995), "")
            if not duplicate:
                unique.append(item)
            candidates.append((item, duplicate))
    return unique, candidates


def _basis(matrix):
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    return u[:, :2] if len(s) == 2 and s[1] > s[0] * 1e-8 else None


def subspace_overlap(left, right):
    """Smallest principal overlap of two complex, two-mode subspaces."""
    a = np.column_stack([item.ravel() for item in left])
    b = np.column_stack([item.ravel() for item in right])
    finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    qa, qb = _basis(a[finite]), _basis(b[finite])
    return float(np.linalg.svd(qa.conj().T @ qb, compute_uv=False)[-1]) if qa is not None and qb is not None else 0.


def _reflection_subspace(first, second):
    a = np.stack([first.profile.electric, second.profile.electric])
    reflected = a[:, :, :, ::-1] * np.array([-1, 1, 1])[None, :, None, None]
    a = a.reshape(2, -1).T
    r = reflected.reshape(2, -1).T
    finite = np.isfinite(a).all(axis=1) & np.isfinite(r).all(axis=1)
    q = _basis(a[finite])
    qr = _basis(r[finite])
    if q is None or qr is None or np.linalg.svd(q.conj().T @ qr, compute_uv=False)[-1] < PARITY_LIMIT:
        return False
    transform, *_ = np.linalg.lstsq(a[finite], q, rcond=None)
    reflection = q.conj().T @ (r[finite] @ transform)
    values = np.linalg.eigvalsh((reflection + reflection.conj().T)/2)
    return values[0] <= -PARITY_LIMIT and values[1] >= PARITY_LIMIT


def select_pair(candidates, polarization):
    """Return same-family even/odd modes, or preserve an explicit diagnostic."""
    family = [item for item in candidates if item.eligible and item.polarization == polarization]
    pairs = []
    for a, b in combinations(family, 2):
        even, odd = sorted((a, b), key=lambda item: -item.parity)
        if even.parity >= PARITY_LIMIT and odd.parity <= -PARITY_LIMIT:
            if abs(overlap(even.local, odd.local)) >= OVERLAP_LIMIT:
                pairs.append((even, odd))
    if len(pairs) == 1:
        return pairs[0], "paired"
    if len(pairs) > 1:
        return None, "ambiguous_pair"
    # An eigensolver can return an arbitrary basis within a nearly degenerate pair.
    possible = [item for item in candidates if item.polarization == polarization
                and abs(item.neff.imag) <= IMAG_LIMIT and item.ridge_fraction >= RIDGE_LIMIT]
    for a, b in combinations(possible, 2):
        if abs(a.neff - b.neff) <= DUPLICATE_NEFF and _reflection_subspace(a, b):
            return (a, b), "degenerate_subspace"
    return None, "no_confirmed_pair"
