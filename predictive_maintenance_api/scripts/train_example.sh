#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m app.train --data data/cmapss_like.csv --output-dir artifacts
