"""Compatibility entry point for the original JSON configuration."""

from pathlib import Path

from .artifacts import read_json
from .comsol_models.single import SingleWaveguide2D


def run(config_file: Path, *, stage: str = "comsol") -> Path:
    if stage != "comsol":
        raise ValueError(f"Неизвестный этап {stage}. Анализ запускается через python -m studies.…")
    config_file = Path(config_file).resolve(strict=True)
    config = read_json(config_file)
    try:
        model = SingleWaveguide2D(config_file.parent / config["model"],
                                  results_root=config_file.parent / config["output_dir"])
        result = model.modes(widths_um=config["widths_um"], bands=config["bands"],
                             pump_wavelength_um=config["pump_wavelength_um"],
                             field_grid=config["field_grid"], geometry=config["geometry"],
                             mode_count=config["modes"], cores=config["cores"], solve=True)
    except (KeyError, TypeError) as error:
        raise ValueError(f"Проверь поля файла настроек {config_file}: {error}") from error
    return result.run_dir
