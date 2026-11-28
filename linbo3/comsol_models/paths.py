"""Resolve material tables for bundled and external MPH files."""

from pathlib import Path


MATERIALS_DIR = Path(__file__).resolve().parent / "materials"


def material_directory(model_path, bundled_directory):
    directory = Path(model_path).resolve().parent
    return MATERIALS_DIR if directory == bundled_directory else directory
