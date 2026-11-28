"""Compatibility entry point for the coupled 2D template builder."""

from .coupled.template import TEMPLATE_NAME, build_template, main

__all__ = ["TEMPLATE_NAME", "build_template", "main"]


if __name__ == "__main__":
    main()
