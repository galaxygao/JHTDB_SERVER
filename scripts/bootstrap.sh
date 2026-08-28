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

# isotropic1024coarse still needs the functional legacy runtime bundled with
# Essentials 4.0.  Check the image before pip can change the environment.
python - <<'PY'
import sys

if sys.version_info < (3, 9):
    raise SystemExit("Python 3.9 or newer is required.")

try:
    import pyJHTDB  # noqa: F401
    from giverny.turbulence_dataset import turb_dataset  # noqa: F401
    from giverny.turbulence_toolkit import getCutout  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "The selected image does not provide the functional legacy JHTDB "
        "runtime required by isotropic1024coarse. Use SciServer Essentials 4.0. "
        f"Underlying import error: {exc}"
    ) from exc
PY

if [[ -x .venv/bin/python ]]; then
  BASE_PYTHON="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  VENV_PYTHON="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  if [[ "$BASE_PYTHON" != "$VENV_PYTHON" ]]; then
    echo "ERROR: existing .venv uses Python ${VENV_PYTHON:-unknown}, but this image uses Python $BASE_PYTHON." >&2
    echo "Move the old .venv aside, then run bootstrap.sh again." >&2
    exit 1
  fi
fi

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

try:
    importlib.import_module("pyJHTDB")
    from giverny.turbulence_dataset import turb_dataset
    from giverny.turbulence_toolkit import getCutout
except ImportError as exc:
    raise SystemExit(
        "The selected image does not provide the functional legacy pyJHTDB "
        "runtime required by isotropic1024coarse. Use SciServer Essentials 4.0. "
        f"Underlying import error: {exc}"
    ) from exc
PY
python -m pip check
python -m compileall -q src
echo "SciServer environment ready: $PROJECT_ROOT/.venv"
