"""Build a single-point template from the archived two-waveguide MPH."""

import argparse
from pathlib import Path

from ...artifacts import atomic_path
from ...comsol_session import open_model

from ..paths import material_directory
from .model import MODEL_DIR, TEMPLATE_NAME


def _remove_all(nodes):
    for tag in list(nodes.tags()):
        if tag in list(nodes.tags()):
            nodes.remove(tag)


def build_template(source, destination):
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    if destination.suffix.lower() != ".mph" or destination.exists():
        raise ValueError("Укажи новый файл .mph для шаблона")
    directory = material_directory(destination, MODEL_DIR)
    if not (directory / "n_SiO2.txt").is_file():
        raise FileNotFoundError(directory / "n_SiO2.txt")

    with atomic_path(destination) as temporary:
        with open_model(source, cores=2) as (_, model):
            java = model.java
            if list(java.study().tags()) != ["std2"]:
                raise ValueError("Ожидался исходный вариант соседних волноводов с исследованием std2")
            study = java.study("std2")
            if list(study.feature().tags()) != ["param", "mode"]:
                raise ValueError("Ожидались шаги param и mode в исходной модели")

            study.feature().remove("param")
            _remove_all(java.batch())
            _remove_all(java.result())
            _remove_all(java.result().numerical())
            _remove_all(java.result().dataset())
            _remove_all(java.sol())
            study.setGenPlots(False)
            study.showAutoSequences("sol")

            solutions = list(java.sol().tags())
            jobs = list(java.batch().tags())
            if len(solutions) != 1 or jobs:
                raise RuntimeError(f"Ожидалась одна цепочка без перебора, получено: {solutions}, задачи: {jobs}")
            solver = java.sol(solutions[0])
            if not solver.isAttached() or solver.study() != "std2":
                solver.attach("std2")
            if not any(solver.feature(tag).getType() == "Eigenvalue" for tag in solver.feature().tags()):
                raise RuntimeError("В новой цепочке отсутствует решатель собственных значений")
            datasets = java.result().dataset()
            if not list(datasets.tags()):
                dataset = datasets.create("linbo3modes", "Solution")
                dataset.set("solution", solutions[0])
            tags = list(datasets.tags())
            if len(tags) != 1 or java.result().dataset(tags[0]).getString("solution") != solutions[0]:
                raise RuntimeError("Ожидался один dataset нового решателя")

            java.modelPath(str(directory))
            material = java.component("comp1").func("int1")
            material.set("source", "file")
            material.set("filename", "n_SiO2.txt")
            model.save(temporary)
            print(f"Mode Analysis: std2/mode | solver: {solutions[0]} | dataset: {tags[0]}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    target = args.output or args.source.with_name(TEMPLATE_NAME)
    try:
        print(build_template(args.source, target))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
