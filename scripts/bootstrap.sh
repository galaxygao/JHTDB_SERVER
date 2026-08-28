#!/usr/bin/env bash
set -euo pipefail
set +x

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: bootstrap.sh is only for the SciServer Linux container." >&2
  exit 1
fi

if [[ "$PROJECT_ROOT" != /home/idies/workspace/Storage/*/persistent/JHU_DATA ]]; then
  echo "ERROR: project must be stored in the SciServer persistent volume: $PROJECT_ROOT" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
if [[ ! -d .venv ]]; then
  python -m venv --system-site-packages .venv
fi
source .venv/bin/activate
python -m pip install --editable .
python - <<'PY'
import importlib

required = (
    "filelock", "giverny", "numcodecs", "numpy", "plotly", "yaml",
    "rich", "scipy", "streamlit", "zarr",
)
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
if missing:
    raise SystemExit(
        "Missing packages in the selected SciServer image: " + ", ".join(missing)
    )
PY
python -m pip check
python -m compileall -q src
echo "SciServer environment ready: $PROJECT_ROOT/.venv"
