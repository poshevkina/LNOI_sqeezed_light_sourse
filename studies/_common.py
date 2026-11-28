"""Terminal entry points for studies on saved COMSOL data."""

import argparse
from pathlib import Path

from linbo3.artifacts import read_json
from linbo3.mode_classification import classify_run
from linbo3.spatial_workflow import prepare_spatial_study


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_comsol_run(results_root, name=None, *, required_widths=()):
    directory = Path(results_root) / "comsol"
    if name is not None:
        candidate = (directory / name).resolve()
        if candidate.parent != directory.resolve() or not (candidate / "config.json").is_file():
            raise ValueError(f"Не найден COMSOL-прогон: {name}")
        return candidate
    completed = [(path, read_json(path)) for path in directory.glob("*/status.json")]
    required = set(required_widths)
    completed = [(path, status) for path, status in completed if status.get("state") == "complete"
                 and (not required or required.issubset(read_json(path.parent / "config.json")["widths_um"]))]
    if not completed:
        expected = f" с ширинами {', '.join(f'{w:g}' for w in sorted(required))} мкм" if required else ""
        raise FileNotFoundError(f"Нет готового прогона{expected} в {directory}")
    path, _ = max(completed, key=lambda entry: (
        entry[1].get("completed_at", ""), entry[0].stat().st_mtime_ns, entry[0].parent.name,
    ))
    return path.parent


def study_main(description, run, *, points, results_root, comsol_run=None, required_widths=(), prepare_comsol=None):
    from matplotlib import use

    parser = argparse.ArgumentParser(description=description)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--comsol-run", help="Имя папки в results/comsol")
    source.add_argument("--comsol-dir", type=Path, help="Явный путь к сохранённому прогону")
    parser.add_argument("--results-root", type=Path, default=results_root, help="Корень отчётов и кэша")
    parser.add_argument("--points", type=int, default=points, help="Размер частотной сетки")
    if prepare_comsol is not None:
        source.add_argument("--solve", action="store_true", help="Досчитать недостающие точки в COMSOL")
        source.add_argument("--dry-run", action="store_true", help="Показать точки и состояние кэша")
    args = parser.parse_args()
    if args.points < 5:
        parser.error("--points: нужно минимум 5 точек")
    use("Agg")
    try:
        if prepare_comsol is not None and not (args.comsol_dir or args.comsol_run or comsol_run):
            prepared = prepare_comsol(args.results_root, solve=args.solve, dry_run=args.dry_run)
            if args.dry_run:
                return
            source = prepared.run_dir
        else:
            source = args.comsol_dir or resolve_comsol_run(
                args.results_root, args.comsol_run or comsol_run, required_widths=required_widths,
            )
        output = run(source.resolve(), args.results_root.resolve(), points=args.points)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Ошибка: {error}\n")
    print(f"RESULT_DIR={output.resolve()}", flush=True)


def classify_spatial(source, work, *, degree, signal_label, idler_label, classification, tracking):
    classified = classify_run(source, output_dir=work / "classification", **classification)
    if classified.table.empty:
        raise ValueError("В выгрузках отсутствуют центральные решения для классификации")
    spatial = prepare_spatial_study(source, classified.table, degree=degree,
                                    signal_label=signal_label, idler_label=idler_label, tracking=tracking)
    spatial.branches.to_csv(work / "branches.csv", index=False)
    spatial.diagnostics.to_csv(work / "spatial_diagnostics.csv", index=False)
    return spatial
