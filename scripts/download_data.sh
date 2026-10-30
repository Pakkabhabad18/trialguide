#!/usr/bin/env bash
# Download TREC CT 2021, TREC CT 2022, SIGIR 2016 (via the TrialGPT repo) and the
# 1,020 expert criterion labels into ./data/raw. Safe to re-run; files are checksummed.
# Usage: bash scripts/download_data.sh [--with-trial-info]
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m trialguide.data.download "$@"
