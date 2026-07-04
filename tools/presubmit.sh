#!/usr/bin/env bash
# One-command pre-submission gate. Run this against the manuscript you are about to submit.
#
#   bash tools/presubmit.sh /path/to/SUBMISSION_folder
#
# Runs every deterministic check in one pass and reports a combined verdict:
#   [code ]  absolute-path guard          (reproduction: no machine-specific paths)
#   [code ]  property / invariant tests   (correctness: GFEVD / VAR / null invariants)
#   [paper]  citation-integrity gate      (integrity: every reference resolves; no fabrication/hijack)
#   [paper]  AI-artifact / hidden-text     (integrity: no leftover AI boilerplate or placeholders)
#
# The code checks also run in CI on every push; this bundles them with the manuscript-level
# checks so you get a single green light before hitting submit. Exit 0 = all clear.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SUB="${1:-}"
PY="${PYTHON:-python3}"
fail=0

hr() { printf '%s\n' "------------------------------------------------------------------------------"; }
run() { echo; echo ">> $1"; hr; shift; "$@"; local rc=$?; if [ $rc -ne 0 ]; then fail=1; echo "   [FAILED rc=$rc]"; fi; return 0; }

echo "=============================================================================="
echo " PRE-SUBMISSION GATE"
echo " repo:       $REPO"
echo " manuscript: ${SUB:-<none given — running code checks only>}"
echo "=============================================================================="

run "code  · absolute-path guard"        bash "$REPO/tools/check_paths.sh" "$REPO"
run "code  · property / invariant tests" env PYTHONPATH="$REPO/pilot_cross_tier" "$PY" -m pytest "$REPO/tests" -q
run "code  · symbolic exact oracle (GFEVD)" "$PY" "$REPO/tools/verify_symbolic.py"

if [ -n "$SUB" ]; then
  MS="$SUB/manuscript.tex";        [ -f "$MS" ] || MS="$(ls "$SUB"/*.tex 2>/dev/null | grep -iv supplement | head -1)"
  SI="$SUB/supplementary_information.tex"
  PDF="$SUB/manuscript.pdf";       [ -f "$PDF" ] || PDF="$(ls "$SUB"/*.pdf 2>/dev/null | grep -iv supplement | head -1)"
  if [ -n "${MS:-}" ] && [ -f "$MS" ]; then
    mkdir -p "$REPO/_reports"
    run "paper · citation-integrity gate" "$PY" "$REPO/tools/check_citations.py" "$MS" --json "$REPO/_reports/citation_report.json"
    SCRUB_ARGS=("$MS"); [ -f "$SI" ] && SCRUB_ARGS+=("$SI")
    if [ -f "$PDF" ]; then
      run "paper · AI-artifact / hidden-text" "$PY" "$REPO/tools/scrub_ai_artifacts.py" "${SCRUB_ARGS[@]}" --pdf "$PDF"
    else
      run "paper · AI-artifact scan"          "$PY" "$REPO/tools/scrub_ai_artifacts.py" "${SCRUB_ARGS[@]}"
    fi
  else
    echo; echo "   [skip] no manuscript .tex found under $SUB"
  fi
fi

echo; echo "=============================================================================="
if [ $fail -eq 0 ]; then
  echo " RESULT: ALL CLEAR — deterministic pre-submission checks pass"
else
  echo " RESULT: ATTENTION NEEDED — one or more checks reported items above"
fi
echo "=============================================================================="
exit $fail
