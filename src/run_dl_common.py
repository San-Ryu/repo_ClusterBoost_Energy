from __future__ import annotations

import os
import sys
from pathlib import Path


def run_model(model_name: str) -> None:
    """
    Execute model_dl_single.py with a stable DL runtime context.
    - Prefer .venv-dl python when present
    - Apply conservative thread/runtime env defaults
    """
    root = Path(__file__).resolve().parent.parent
    venv_dl_python = root / ".venv-dl" / "bin" / "python"
    if not venv_dl_python.exists():
        print(
            "[ERROR] .venv-dl not found. Create DL runtime first:\n"
            "  python3.10 -m venv .venv-dl\n"
            "  source .venv-dl/bin/activate\n"
            "  python -m pip install --upgrade pip\n"
            "  python -m pip install \"numpy<2.0\" pandas scikit-learn matplotlib seaborn tqdm\n"
            "  python -m pip install tensorflow-macos==2.15.0 tensorflow-metal==1.1.0 keras==2.15.0\n"
            "  python -c \"import tensorflow as tf; print(tf.__version__)\""
        )
        sys.exit(2)

    python_bin = str(venv_dl_python)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    env.setdefault("TF_NUM_INTEROP_THREADS", "1")

    cmd = [
        python_bin,
        "src/model_dl_single.py",
        "--model",
        model_name,
        *sys.argv[1:],
    ]
    os.execvpe(python_bin, cmd, env)

