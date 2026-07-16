import os
import sys

os.execvp(
    sys.executable,
    [sys.executable, "src/model_ml_single.py", "--model", "RandomForest", *sys.argv[1:]],
)

