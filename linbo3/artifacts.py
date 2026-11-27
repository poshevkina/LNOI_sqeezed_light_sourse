"""Portable paths and atomic writes for published calculation outputs."""

from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep

import numpy as np


def digest_json(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as file:
        return json.load(file)


def relative_path(base, target):
    return Path(os.path.relpath(target, base)).as_posix()


@contextmanager
def file_lock(path, *, wait=True):
    """Hold an OS lock released automatically when the process exits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        lock.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if not wait:
                        raise BlockingIOError(errno.EAGAIN, "Lock already held", path) from error
                    sleep(.1)
        else:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as file:
        temporary = Path(file.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_arrays(path, payload):
    with atomic_path(path) as temporary:
        np.savez_compressed(temporary, **payload)


def write_csv(path, table, **kwargs):
    with atomic_path(path) as temporary:
        table.to_csv(temporary, index=False, **kwargs)


def results_root_for(path):
    """Resolve results/ from a path inside one of its COMSOL runs or studies."""
    path = Path(path).resolve()
    for parent in (path, *path.parents):
        if parent.parent.name == "comsol" and (parent / "config.json").is_file():
            return parent.parent.parent
        if parent.parent.name == "studies" and (parent.parent.parent / "comsol").is_dir():
            return parent.parent.parent
    return None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def latest_directory(root, manifest_name):
    manifests = list(Path(root).glob(f"*/{manifest_name}"))
    def freshness(path):
        value = read_json(path)
        return value.get("completed_at", value.get("started_at", "")), path.stat().st_mtime_ns, path.name

    return max(manifests, key=freshness).parent if manifests else None
