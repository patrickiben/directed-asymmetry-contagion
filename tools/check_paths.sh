#!/usr/bin/env bash
# Absolute-path / machine-specific-path guard.
#
# WHY: a hardcoded /Users/<name>/... or C:\... path is the classic reason a reviewer's
# fresh clone fails to run. This greps the tracked Python for machine-specific paths and
# fails if any survive outside comments/docstrings-of-record. Runs in CI on a cold runner.
#
# Usage: bash tools/check_paths.sh [dir]   (default: repo root)
set -uo pipefail
ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"

# Machine-specific absolute paths that must never appear in executable code.
# We DO allow them inside triple-quoted "Run: ..." docstrings-of-record, so we match only
# lines that look like assignments / calls, not prose. To stay simple and strict, we flag
# ALL occurrences and let the author whitelist any legitimate doc mention with  # noqa: path
PATTERN='(/Users/[A-Za-z0-9._-]+/|/home/[A-Za-z0-9._-]+/|C:\\\\|/private/tmp/|/tmp/lsa_venv)'

echo "Absolute-path guard over: $ROOT"
echo "=============================================================================="
HITS=$(grep -rnoE "$PATTERN" "$ROOT" \
        --include='*.py' --include='*.sh' --include='*.yml' --include='*.yaml' \
        --exclude='check_paths.sh' \
        2>/dev/null | grep -v 'noqa: path')

if [ -n "$HITS" ]; then
  echo "$HITS" | sed 's/^/>FLAG< /'
  N=$(echo "$HITS" | wc -l | tr -d ' ')
  echo "=============================================================================="
  echo "SUMMARY: $N machine-specific path(s) found. Replace with Path(__file__).resolve().parent-relative"
  echo "         paths, or annotate an intentional doc mention with  # noqa: path"
  exit 1
fi
echo "  ok  no machine-specific absolute paths in executable code"
echo "=============================================================================="
exit 0
