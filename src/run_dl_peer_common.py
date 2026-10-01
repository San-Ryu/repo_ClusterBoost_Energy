from __future__ import annotations

import os
import sys
from pathlib import Path


def _default_python(root: Path) -> str:
    candidates = [
        root / ".venv-dl" / "bin" / "python",
        root / ".venv" / "bin" / "python",
        Path(sys.executable),
    ]
    for path in candidates:
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return "python"


def run_peer_grid_model(model_name: str) -> None:
    root = Path(__file__).resolve().parent.parent
    python_bin = os.environ.get("PYTHON_BIN") or _default_python(root)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    env.setdefault("TF_NUM_INTEROP_THREADS", "1")

    cmd = [
        python_bin,
        "src/model_dl_peer_grid_cv.py",
        "--model",
        model_name,
        *sys.argv[1:],
    ]
    os.execvpe(python_bin, cmd, env)
