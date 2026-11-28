"""Public COMSOL adapters used by research scripts."""

from .coupled import CoupledWaveguides2D
from .single import SingleWaveguide2D

__all__ = ["CoupledWaveguides2D", "SingleWaveguide2D"]
