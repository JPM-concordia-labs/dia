#!/usr/bin/env bash
set -euo pipefail
DIA_ROOT="/home/joao/work/external/dia"
exec "$DIA_ROOT/.venv/bin/python" "$DIA_ROOT/cli.py" "$@"
