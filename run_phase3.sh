#!/usr/bin/env bash

set -euo pipefail

project_root="$(
  cd "$(
    dirname "${BASH_SOURCE[0]}"
  )" \
  && pwd
)"

cd "$project_root"

if [ -n "${PHASE3_PYTHON:-}" ]
then
  python_bin="$PHASE3_PYTHON"
elif [ -x "venv_bonus/bin/python" ]
then
  python_bin="venv_bonus/bin/python"
elif command -v python3 >/dev/null 2>&1
then
  python_bin="$(
    command -v python3
  )"
elif command -v python >/dev/null 2>&1
then
  python_bin="$(
    command -v python
  )"
else
  echo "No suitable Python interpreter was found." >&2
  exit 1
fi

export DO_NOT_TRACK=1
export PREFECT_SERVER_ANALYTICS_ENABLED=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

exec "$python_bin" \
  scripts/validate_phase3_delivery.py \
  "$@"
