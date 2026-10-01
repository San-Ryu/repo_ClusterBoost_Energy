from __future__ import annotations

import os
import sys
from pathlib import Path


def _default_python(root: Path, family: str) -> str:
    candidates = []
    if family == "dl":
        candidates.append(root / ".venv-dl" / "bin" / "python")
    candidates.append(root / ".venv" / "bin" / "python")
    candidates.append(Path(sys.executable))
    for path in candidates:
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return "python"


def run_legacy_aggregate_model(family: str, model_name: str, default_results_dir: str) -> None:
    """Execute the legacy aggregate runner with stable model/env defaults."""
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["FAMILY"] = family
    env["MODEL"] = model_name
    env.setdefault("SCOPE", "both")
    env.setdefault("RESOLUTION", "1H")
    env.setdefault("RESULTS_DIR", default_results_dir)
    env.setdefault("PYTHON_BIN", _default_python(root, family))
    env.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    env.setdefault("TF_NUM_INTEROP_THREADS", "1")

    cmd = ["bash", "src/run_legacy_aggregate.sh", *sys.argv[1:]]
    os.execvpe("bash", cmd, env)
