"""Inspect or compact saved SPDC cases without invoking either solver."""

import argparse
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np

from .artifacts import atomic_path, file_lock, read_json
from .case_cache import STORAGE_VERSION, projections, validate_compact


def legacy_jsi_size(path):
    with ZipFile(path) as archive:
        try:
            return archive.getinfo("IMatrix.npy").compress_size
        except KeyError:
            return 0


def compact_case(path, points):
    """Verify the replacement before committing it with a single file rename."""
    with np.load(path, allow_pickle=False) as source:
        saved = {key: source[key] for key in source.files if key != "IMatrix"}
        if "IMatrix" not in source:
            validate_compact(saved, points)
            return False
        jsi = source["IMatrix"]
    projected = projections(saved["w_grid"], jsi)
    for key in ("S_s", "S_i"):
        if not np.array_equal(saved[key], projected[key]):
            raise ValueError(f"Saved marginal differs from the full matrix: {key}")
    idx = saved["preview_indices"]
    if not np.array_equal(saved["plot_jsi"], jsi[np.ix_(idx, idx)]):
        raise ValueError("Saved preview differs from the full matrix")
    saved.update(projected, storage_version=np.asarray(STORAGE_VERSION))
    validate_compact(saved, points)
    with atomic_path(path) as temporary:
        np.savez_compressed(temporary, **saved)
        with np.load(temporary, allow_pickle=False) as check:
            validate_compact(check, points)
            if set(check.files) != saved.keys() or any(
                not np.array_equal(check[key], value, equal_nan=True) for key, value in saved.items()
            ):
                raise ValueError("Compact copy failed verification")
    return True


def remove_full_jsa(path, full, signature):
    if not full.is_file():
        return 0
    if full.is_symlink() or full.resolve() != full:
        raise ValueError(f"Expected local JSA file: {full}")
    with np.load(path, allow_pickle=False) as case:
        validate_compact(case, signature["points"])
        if read_json(full.with_name("case.json")) != signature:
            raise ValueError("Full and compact case metadata differ")
        idx = case["preview_indices"]
        with np.load(full, allow_pickle=False) as raw:
            preview = raw["jsa"][np.ix_(idx, idx)].astype(np.complex64)
            if (not np.array_equal(preview, case["jsa_preview"])
                    or any(not np.array_equal(raw[key], case["w_grid"])
                           for key in ("omega_signal_rad_per_fs", "omega_idler_rad_per_fs"))):
                raise ValueError("Full JSA differs from the compact preview")
    size = full.stat().st_size
    full.unlink()
    return size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true", help="Verify and replace legacy case NPZ files")
    action.add_argument("--remove-full-jsa", action="store_true",
                        help="Permanently remove full JSA files whose compact copies pass verification")
    args = parser.parse_args()
    root = args.results_root.resolve()
    changed = failures = gained = total = legacy = estimate = jsa_count = jsa_bytes = 0
    for path in sorted((root / "cases").glob("*/case.npz")):
        full = root.parent / "jsa_full" / path.parent.name / "full.npz"
        total += 1
        try:
            signature = read_json(path.with_name("case.json"))
            if path.resolve() != path or path.is_symlink():
                raise ValueError(f"Expected local case file: {path}")
            size = legacy_jsi_size(path)
            legacy += bool(size)
            if size:
                # The compact payload adds two vectors of 2*N-1 float64 samples.
                estimate += max(0, size - (2 * signature["points"] - 1) * 16)
            if full.is_file():
                jsa_count += 1
                jsa_bytes += full.stat().st_size
            if args.apply or args.remove_full_jsa:
                with file_lock(full.with_name(".case.lock")):
                    if args.apply:
                        before = path.stat().st_size
                        if compact_case(path, signature["points"]):
                            changed += 1
                            gained += before - path.stat().st_size
                    else:
                        removed = remove_full_jsa(path, full, signature)
                        changed += bool(removed)
                        gained += removed
        except (OSError, ValueError, KeyError, IndexError, BadZipFile) as error:
            failures += 1
            print(f"{path.parent.name}: {error}", flush=True)
    print(f"Cases: {total}; with full JSI: {legacy}; estimated savings: {estimate / 1e6:.1f} MB")
    print(f"Existing full JSA: {jsa_count}; size: {jsa_bytes / 1e6:.1f} MB")
    if args.apply or args.remove_full_jsa:
        print(f"Changed: {changed}; reclaimed: {gained / 1e6:.1f} MB; errors: {failures}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
