#!/usr/bin/env bash
# One command to reproduce the out-of-sample real-data lead. CPU-only, <30s.
set -e
python3 -m pip install -q -r requirements.txt
echo ">>> Financial arm (offline, bundled vendor data):"
python3 reproduce_oos.py --offline
echo
echo ">>> Full run incl. live CDC influenza (needs internet):"
python3 reproduce_oos.py || echo "(influenza fetch failed — offline result above is the reproducible financial arm)"
