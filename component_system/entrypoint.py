"""Standalone entrypoint for the component_system baseline."""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from component_system.training.mainline import run_mainline_training


def main() -> None:
    run_mainline_training()


if __name__ == "__main__":
    main()
