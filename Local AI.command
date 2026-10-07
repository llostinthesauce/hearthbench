#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
show_error() {
    status=$?
    if [[ "$status" -ne 0 && -t 0 ]]; then
        echo "Local AI exited with an error (code $status). See the message above."
        read -r -p "Press Enter to close." || true
    fi
}
trap show_error EXIT
if [[ ! -x .venv/bin/python ]]; then
    echo "The local environment is missing. Follow One-time setup in README.md."
    exit 1
fi
.venv/bin/python llm.py serve "$@"
