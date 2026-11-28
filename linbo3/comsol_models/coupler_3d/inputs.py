"""Load paired 2D fields and choose a one-sided excitation by flux."""

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.constants import c, mu_0
from scipy.linalg import eigh

from ...artifacts import read_json
from ..coupled.analysis import load_mode_field


@dataclass(frozen=True)
class Excitation:
    gap_um: float
    family: str
    width_um: float
    wavelength_um: float
    x_m: np.ndarray
    y_m: np.ndarray
    electric: np.ndarray
    beta: np.ndarray
    weights: np.ndarray
    p_left_2d: float
    flux: np.ndarray
    refs: tuple


def flux_gram(electric, magnetic, mask, area):
    matrix = np.empty((2, 2), complex)
    for i in range(2):
        for j in range(2):
            ex, ey = electric[j, :2, mask].T
            hx, hy = magnetic[i, :2, mask].T
            matrix[i, j] = np.sum(ex * hy.conj() - ey * hx.conj()) * area
    return (matrix + matrix.conj().T) / 4


def flux_basis(electric, beta, x_m, y_m, wavelength_um):
    """Forward power-normalized basis and left/right Hermitian flux matrices."""
    dy, dx = np.gradient(electric, y_m, x_m, axis=(2, 3), edge_order=2)
    dz = -1j * beta[:, None, None, None] * electric
    omega = 2 * np.pi * c / (wavelength_um * 1e-6)
    magnetic = 1j / (omega * mu_0) * np.stack(
        [dy[:, 2] - dz[:, 1], dz[:, 0] - dx[:, 2], dx[:, 1] - dy[:, 0]], axis=1,
    )
    left = np.broadcast_to(x_m < 0, electric.shape[-2:])
    area = (x_m[1]-x_m[0]) * (y_m[1]-y_m[0])
    flux = np.stack([flux_gram(electric, magnetic, mask, area) for mask in (left, ~left)])
    power = flux.sum(axis=0).diagonal().real
    if (power <= 0).any():
        raise ValueError("Ожидались положительные мощности 2D-мод вдоль +z")
    normalize = np.sqrt(power)
    return electric / normalize[:, None, None, None], flux / np.outer(normalize, normalize)


def load_excitation(source, gap, family):
    source = Path(source).resolve()
    config = read_json(source / "config.json")
    pairs = pd.read_csv(source / "pairs.csv")
    row, = pairs[(pairs.gap_um == gap) & (pairs.family == family) & (pairs.status == "paired")].itertuples()
    candidates = pd.read_csv(source / "candidates.csv")
    refs, fields, beta = [], [], []
    for identifier in (row.first_id, row.second_id):
        shift, outer, solution = identifier.split(":")
        item, = candidates[(candidates.gap_um == gap) & (candidates["shift"] == float(shift))
                           & (candidates.outer_solnum == int(outer))
                           & (candidates.solnum == int(solution))].itertuples()
        ref = (source / item.field_reference).resolve()
        settings = read_json(ref)
        profile = load_mode_field(ref, outer_solnum=int(outer), solnum=int(solution))
        if fields and (not np.array_equal(profile.x_um, fields[0].x_um)
                       or not np.array_equal(profile.y_um, fields[0].y_um)):
            raise ValueError("Сетки двух супермод не совпадают")
        refs.append((ref, settings["field_sha256"], identifier))
        fields.append(profile)
        beta.append(2 * np.pi * item.neff_real / (config["wavelength_um"] * 1e-6))
    x_m, y_m = fields[0].x_um * 1e-6, fields[0].y_um * 1e-6
    beta = np.array(beta)
    electric = np.array([field.electric for field in fields])
    if not np.isfinite(electric).all():
        raise ValueError("Портовая сетка 2D содержит пропуски")
    electric, flux = flux_basis(electric, beta, x_m, y_m, config["wavelength_um"])
    fractions, vectors = eigh(flux[1], flux.sum(axis=0))
    weights = vectors[:, 0]
    weights *= np.exp(-1j * np.angle(weights[0]))
    weights /= np.linalg.norm(weights)
    return Excitation(gap, family, config["width_um"], config["wavelength_um"],
                      x_m, y_m, electric, beta, weights, float(1-fractions[0]), flux, tuple(refs))


def validate_materials(template, source_2d):
    """Compare LN optical tensor and embedded interpolation tables in the two MPH files."""
    def signature(path):
        with ZipFile(path) as archive:
            root = ElementTree.fromstring(archive.read("dmodel.xml"))
        entries = {}
        for function in root.iter("FunctionFeature"):
            props = {node.get("name"): node.get("value") or node.get("valueMatrix")
                     for node in function.findall("propertyValue")}
            name = props.get("p:funcname")
            if name in {"no_LiNbO3", "ne_LiNbO3"} and props.get("p:table"):
                entries[name] = props["p:table"]
        material, = [item for item in root.iter("Material") if item.get("name") == "LiNbO3"]
        entries["tensor"] = sorted(value for node in material.iter() for value in node.attrib.values()
                                   if "no_LiNbO3(lbd)^2" in value and "ne_LiNbO3(lbd)^2" in value)
        return entries
    left, right = signature(template), signature(source_2d)
    if left != right or len(left) != 3 or not left["tensor"]:
        raise ValueError("В 2D и 3D различаются таблицы LiNbO3 или оптический тензор")
