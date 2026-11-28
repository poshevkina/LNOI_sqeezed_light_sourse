"""Shared cache and lazy results for COMSOL modal exports."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import file_hash, file_lock, read_json, relative_path, save_arrays, write_csv, write_json
from ..comsol_jobs import execute
from ..modes import ModalSolutions
from ..saved_modes import load_modal_npz


@dataclass(frozen=True)
class ModalTask:
    group: str
    name: str
    label: str
    columns: dict
    directory: Path
    settings: dict
    model_name: str
    calculate: Callable[..., ModalSolutions]
    status_key: str | None = None

    @property
    def fields(self):
        return self.directory / "field.npz"


@dataclass(frozen=True)
class ModalRun:
    run_dir: Path
    tasks: tuple[ModalTask, ...]

    def load_modes(self):
        frames = []
        for task in self.tasks:
            path = self.run_dir / task.group / f"{task.name}_raw.csv"
            frames.append(pd.read_csv(path, float_precision="round_trip").assign(**task.columns))
        return pd.concat(frames, ignore_index=True)

    def load_fields(self, case: str, band: str = "MODES"):
        reference = self.run_dir / case / f"{band}_field.json"
        stored = read_json(reference)
        return load_modal_npz(reference.parent / stored["npz"], expected_sha256=stored["field_sha256"])


def _valid_checksum(task):
    try:
        stored = read_json(task.directory / "settings.json")
        if all(stored.get(key) == value for key, value in task.settings.items()):
            checksum = stored["field_sha256"]
            if checksum == file_hash(task.fields):
                return checksum
    except (OSError, ValueError, KeyError):
        pass
    return None


def _export_exists(task, directory, checksum):
    try:
        reference = read_json(directory / f"{task.name}_field.json")
        return (reference["field_sha256"] == checksum
                and (directory / reference["npz"]).resolve() == task.fields.resolve()
                and (directory / f"{task.name}_raw.csv").is_file())
    except (OSError, ValueError, KeyError):
        return False


def _validate_raw(raw, wavelengths):
    count = len(raw.wavelength_um)
    if not np.array_equal(np.unique(raw.wavelength_um), wavelengths):
        raise ValueError("В выгрузке отсутствуют или изменились точки сетки")
    if (raw.neff.shape != (count,) or raw.outer_solnum.shape != (count,)
            or raw.solnum.shape != (count,) or raw.field.ndim != 3
            or raw.field.shape[:2] != (count, 3)
            or raw.coordinates.shape != (2, raw.field.shape[2])):
        raise ValueError("Поля, координаты и массивы идентификаторов не согласованы")
    if len(np.unique(np.column_stack((raw.outer_solnum, raw.solnum)), axis=0)) != count:
        raise ValueError("Идентификаторы решений повторяются")


def _save_raw_csv(path, raw):
    table = pd.DataFrame(dict(
        wavelength_um=raw.wavelength_um, outer_solnum=raw.outer_solnum, solnum=raw.solnum,
        neff_real=raw.neff.real, neff_imag=raw.neff.imag,
    ))
    write_csv(path, table, encoding="utf-8-sig", na_rep="NaN")


def run_modal(run_dir, manifest, tasks, *, source, fingerprint, solve=False, dry_run=False, status_summary="groups",
              allow_partial=False):
    result = ModalRun(run_dir, tuple(tasks))
    if dry_run or not solve:
        checksums = [_valid_checksum(task) for task in result.tasks]
        if dry_run:
            print(f"MODEL={(run_dir / manifest['model']).resolve()} | RUN={run_dir}")
            for task, checksum in zip(result.tasks, checksums, strict=True):
                print(f"{task.label} | {'кэш' if checksum else 'нет проверенной карты'}")
            return result
        missing = [task.label for task, checksum in zip(result.tasks, checksums, strict=True) if not checksum]
        if missing and allow_partial:
            return result
        if missing:
            raise RuntimeError("Нет готовых карт для " + ", ".join(missing) + ". Для досчёта добавь --solve.")
        if ((run_dir / "status.json").is_file() and read_json(run_dir / "status.json")["state"] == "complete"
                and all(_export_exists(task, run_dir / task.group, checksum)
                        for task, checksum in zip(result.tasks, checksums, strict=True))):
            return result

    def collect(task, directory, label):
        with file_lock(task.directory / ".field.lock"):
            checksum = _valid_checksum(task)
            if checksum:
                print(f"{label} | готовое решение из кэша", flush=True)
                if _export_exists(task, directory, checksum):
                    return
                raw = load_modal_npz(task.fields, expected_sha256=checksum)
                _validate_raw(raw, task.settings["wavelengths_um"])
            else:
                if not solve:
                    raise RuntimeError("Кэш стал недоступен. Для досчёта добавь --solve.")
                if task.fields.is_file():
                    print(f"{label} | неполный или повреждённый кэш", flush=True)
                raw = task.calculate(log=task.directory / "comsol.log", label=label)
                _validate_raw(raw, task.settings["wavelengths_um"])
                if fingerprint() != source:
                    raise RuntimeError("Модель или таблицы материалов изменились во время расчёта COMSOL")
                save_arrays(task.fields, raw.arrays())
                checksum = file_hash(task.fields)
                write_json(task.directory / "settings.json", {
                    **task.settings, "field_id": task.directory.name,
                    "model_name": task.model_name, "field_sha256": checksum,
                })
            _save_raw_csv(directory / f"{task.name}_raw.csv", raw)
            write_json(directory / f"{task.name}_field.json", {
                **task.settings, "field_id": task.directory.name, "model_name": task.model_name,
                "field_sha256": checksum, "npz": relative_path(directory, task.fields),
            })

    execute(run_dir, manifest, result.tasks, collect, summary_key=status_summary)
    return result
