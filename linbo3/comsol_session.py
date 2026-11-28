"""Shared MPh session management and COMSOL progress reporting."""

from contextlib import contextmanager
from pathlib import Path
import sys
from threading import Event, Thread
from time import monotonic


@contextmanager
def _progress(client, log: Path, label: str, interval: float = 10.0):
    finished = Event()
    start = monotonic()

    def watch():
        position = 0
        while True:
            done = finished.wait(interval)
            if not done:
                print(f"{label} | решатель работает {monotonic() - start:.0f} с", flush=True)
            try:
                with log.open("rb") as stream:
                    stream.seek(position)
                    lines = [line.decode("utf-8", errors="replace").strip() for line in stream]
                    position = stream.tell()
                progress = [line for line in lines if "Current Progress:" in line or "Parameter lbd =" in line]
                warnings = [line for line in lines if line.startswith(("Warning:", "Error:"))]
                for line in (*progress[-4:], *warnings[-4:]):
                    encoding = sys.stdout.encoding or "utf-8"
                    printable = line.encode(encoding, errors="backslashreplace").decode(encoding)
                    print(f"  COMSOL | {printable}", flush=True)
            except OSError:
                pass
            if done:
                break

    try:
        client.java.showProgress(str(log.resolve()))
    except Exception as error:
        print(f"{label} | недоступен журнал COMSOL: {error}", flush=True)
    thread = Thread(target=watch, daemon=True)
    thread.start()
    try:
        yield
    finally:
        finished.set()
        thread.join()
        try:
            client.java.showProgress(False)
        except Exception:
            pass


@contextmanager
def open_model(path, *, cores, label=""):
    import mph

    if sys.platform == "win32":
        mph.option("session", "stand-alone")
    if label:
        print(f"{label} | загрузка модели", flush=True)
    client = mph.start(cores=cores)
    model = client.load(Path(path).resolve(strict=True))
    try:
        yield client, model
    finally:
        client.remove(model)


def solve_study(client, model, study, *, log, label):
    if log is None:
        model.solve(study)
    else:
        print(f"{label} | решение COMSOL", flush=True)
        with _progress(client, log, label):
            model.solve(study)
