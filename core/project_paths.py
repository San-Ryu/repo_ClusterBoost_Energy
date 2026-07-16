"""Shared project path helpers."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("KIER_DATA_ROOT", Path.home() / "data")).expanduser()


def data_path(*parts: str) -> Path:
    return DATA_ROOT.joinpath(*parts)


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
