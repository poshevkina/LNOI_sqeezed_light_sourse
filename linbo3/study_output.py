"""Publish one portable report per study after its work succeeds."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
import re
from shutil import rmtree

from .artifacts import file_lock, read_json, relative_path, utc_now


@contextmanager
def publish_study(output):
    """Build beside the report so relative case links survive publication."""
    output = Path(output)
    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", output.name) is None:
        raise ValueError("Имя исследования задаётся латиницей, цифрами и дефисом")
    output = output.parent.resolve() / output.name
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.with_name(f".work-{output.name}")
    backup = output.with_name(f".previous-{output.name}")
    lock = output.with_name(f".lock-{output.name}")
    with ExitStack() as stack:
        try:
            stack.enter_context(file_lock(lock, wait=False))
        except BlockingIOError as error:
            raise RuntimeError(f"Исследование уже запущено: {output.name}") from error
        for path in (output, work, backup):
            if path.resolve() != path or path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                raise ValueError(f"Ожидается обычная папка отчёта: {path}")
        if backup.exists():
            if output.exists():
                rmtree(backup)
            else:
                backup.rename(output)
        if work.exists():
            rmtree(work)
        work.mkdir()
        try:
            yield work
            try:
                if output.exists():
                    output.rename(backup)
                work.rename(output)
            except BaseException:
                if backup.exists() and not output.exists():
                    backup.rename(output)
                raise
            if backup.exists():
                rmtree(backup)
        finally:
            if work.exists():
                rmtree(work)


def report_metadata(run_dir, output, **settings):
    manifest = read_json(Path(run_dir) / "config.json")
    return {
        **settings,
        "comsol_run": manifest.get("run_id", Path(run_dir).name),
        "comsol_path": relative_path(output, run_dir),
        "model_sha256": manifest.get("model_sha256"),
        "pump_wavelength_um": manifest.get("pump_wavelength_um", .775),
        "length_m": manifest.get("length_m", .12),
        "started_at": utc_now(),
    }
