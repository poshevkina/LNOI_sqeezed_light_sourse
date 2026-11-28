"""Run-level locking and status shared by COMSOL model adapters."""

from contextlib import ExitStack
import traceback

from .artifacts import file_lock, read_json, utc_now, write_json


def execute(run_dir, manifest, tasks, collect, *, summary_key):
    """Publish case status only after the adapter has saved its artifacts."""
    with ExitStack() as stack:
        try:
            stack.enter_context(file_lock(run_dir / ".run.lock", wait=False))
        except BlockingIOError as error:
            raise RuntimeError(f"COMSOL-прогон {run_dir.name} уже запущен") from error

        print(f"RESULT_DIR={run_dir}", flush=True)
        write_json(run_dir / "config.json", manifest)
        status_path = run_dir / "status.json"
        previous = read_json(status_path) if status_path.is_file() else {}
        pairs = {}
        for task in tasks:
            group = task.status_key or task.group
            state = previous.get("pairs", {}).get(group, {}).get(task.name, {"state": "pending"})
            pairs.setdefault(group, {})[task.name] = state
        status = {"state": "running", summary_key: dict(previous.get(summary_key, {})), "pairs": pairs}
        write_json(status_path, status)
        failures = []
        completed = 0
        try:
            for task in tasks:
                label = f"[{completed}/{len(tasks)} готово] {task.label}"
                group = task.status_key or task.group
                state = pairs[group]
                state[task.name] = {"state": "running"}
                write_json(status_path, status)
                directory = run_dir / task.group
                directory.mkdir(exist_ok=True)
                diagnostic = directory / f"{task.name}_error.json"
                try:
                    collect(task, directory, label)
                except Exception as error:
                    failures.append(f"{task.label}: {error}")
                    state[task.name] = {"state": "failed", "error": str(error)}
                    write_json(diagnostic, {"error": str(error), "traceback": traceback.format_exc()})
                    print(f"{label} | ошибка: {error}", flush=True)
                else:
                    diagnostic.unlink(missing_ok=True)
                    state[task.name] = {"state": "complete"}
                    completed += 1
                    print(f"[{completed}/{len(tasks)} готово] {task.label} | сохранено", flush=True)
                states = {pair["state"] for pair in state.values()}
                if states <= {"complete", "failed"}:
                    status[summary_key][group] = "failed" if "failed" in states else "ready"
                write_json(status_path, status)
            if failures:
                raise RuntimeError("Незавершённые пары COMSOL: " + " | ".join(failures))
        except BaseException as error:
            status.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          error=str(error))
            write_json(status_path, status)
            raise
        status.update(state="complete", completed_at=utc_now())
        write_json(status_path, status)
    print(f"Готово | {run_dir}", flush=True)
