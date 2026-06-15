#!/usr/bin/env bash
# Re-run formal HumanEval (tok8192, no prompt truncation). Wrapper for rerun_humaneval_tok8192.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/rerun_humaneval_tok8192.sh
