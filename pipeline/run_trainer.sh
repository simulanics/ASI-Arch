#!/usr/bin/env bash
set -euo pipefail

# 1) (Optional) activate your virtual‑env if you have one
#if [ -f ".venv/bin/activate" ]; then
#  source .venv/bin/activate
#fi

# 2) Run the trainer
#    you can pass through any args via "$@" if you update test_trainer.py to accept them
python3 "$(dirname "$0")/test_trainer.py" "$@"
