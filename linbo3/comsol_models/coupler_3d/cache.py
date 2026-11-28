"""Content-addressed sectional results for a single 3D solve."""

from pathlib import Path

import numpy as np

from ...artifacts import digest_json, file_hash, file_lock, read_json, save_arrays, write_json
from .solver import AXIAL_UM, FIELD_SHAPE, LENGTH_UM, MAP_SHAPE, SILICA, SLICE_UM, solve_case

VOLUME_KEYS = ("electric_squared_volume", "longitudinal_flux_w_m2")


class CouplerCase:
    def __init__(self, cache_root, template, excitation, layers, weights=None):
        self.template = Path(template)
        self.excitation = excitation
        self.layers = layers
        self.weights = excitation.weights if weights is None else weights
        self.settings = dict(
            version=1, model_sha256=file_hash(template), silica_sha256=file_hash(SILICA),
            gap_um=excitation.gap_um, width_um=excitation.width_um, length_um=LENGTH_UM,
            wavelength_um=excitation.wavelength_um, family=excitation.family,
            modes=[[checksum, identifier] for _, checksum, identifier in excitation.refs],
            beta=excitation.beta.tolist(), weights=[[float(a.real), float(a.imag)] for a in self.weights],
            layers=layers, axial_um=AXIAL_UM.tolist(), slice_um=SLICE_UM.tolist(), field_shape=list(FIELD_SHAPE),
        )
        identity = digest_json(self.settings)[:16]
        self.folder = Path(cache_root) / f"coupler3d_g{excitation.gap_um:g}_{excitation.family}_{layers}layers__{identity}"
        self.field = self.folder / "sections.npz"
        self.manifest = self.folder / "settings.json"

    def ready(self):
        try:
            stored = read_json(self.manifest)
            return self.field.is_file() and stored == dict(self.settings, sha256=file_hash(self.field))
        except (OSError, ValueError):
            return False

    def has_volume(self, *, min_shape=None):
        try:
            with np.load(self.field, allow_pickle=False) as npz:
                if not all(key in npz.files for key in VOLUME_KEYS):
                    return False
                if min_shape is None:
                    return True
                return len(npz["x_um"]) >= min_shape[0] and len(npz["y_um"]) >= min_shape[1]
        except (OSError, KeyError, ValueError):
            return False

    def read(self):
        with np.load(self.field, allow_pickle=False) as npz:
            return {key: npz[key] for key in npz.files if key not in VOLUME_KEYS}

    def read_volume(self):
        with np.load(self.field, allow_pickle=False) as npz:
            if not all(key in npz.files for key in VOLUME_KEYS):
                return None
            return {key: npz[key] for key in VOLUME_KEYS}

    def run(self, solve=False, *, refresh_fields=False):
        def usable():
            return self.ready() and (not refresh_fields or self.has_volume(min_shape=MAP_SHAPE))

        if usable():
            return self.read()
        if not solve:
            return None
        with file_lock(self.folder / ".case.lock"):
            if not usable():
                try:
                    arrays = solve_case(self.template, self.excitation, self.layers, self.folder, self.weights)
                    save_arrays(self.field, arrays)
                    write_json(self.manifest, dict(self.settings, sha256=file_hash(self.field)))
                except Exception as error:
                    write_json(self.folder / "error.json", {"error": str(error)})
                    raise
                (self.folder / "error.json").unlink(missing_ok=True)
            return self.read()
