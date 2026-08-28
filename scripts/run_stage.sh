#!/usr/bin/env bash
set -euo pipefail
set +x

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${JHTDB_PIPELINE_CONFIG:-$PROJECT_ROOT/configs/pipeline.yaml}"

if [[ $# -lt 1 ]]; then
  echo "Usage: run_stage.sh single-frame --time-index N [--sigma-grid S]" >&2
  exit 2
fi

STAGE="$1"
shift
if [[ "$STAGE" != "single-frame" ]]; then
  echo "ERROR: unsupported stage '$STAGE'; the first release supports single-frame only." >&2
  exit 2
fi

cd "$PROJECT_ROOT"
source .venv/bin/activate

mkdir -p state/logs
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="$PROJECT_ROOT/state/logs/${STAGE}_${STAMP}.log"

python -m jhtdb_pipeline single-frame "$@" --config "$CONFIG_PATH" 2>&1 | tee "$LOG_PATH"
