#!/bin/sh
set -eu
cd "$(dirname "$0")"
PY=${DEL_PYTHON:-../venv/bin/python}
TORCH_LIB=$("$PY" -c 'import importlib.util; from pathlib import Path; s=importlib.util.find_spec("torch"); print(Path(s.origin).parent / "lib" if s else "")')
if [ -n "$TORCH_LIB" ]; then
    export DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export POLARS_MAX_THREADS=4
"$PY" -u prepare.py
"$PY" -u features.py training validation
"$PY" -u train.py
"$PY" -u similarity.py
"$PY" -u export_round.py
"$PY" -u verify.py
