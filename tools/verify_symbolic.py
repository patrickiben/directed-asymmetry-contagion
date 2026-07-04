#!/usr/bin/env python3
"""
Independent SymPy exact-rational re-derivation of the Diebold-Yilmaz GFEVD connectedness engine,
checked against the numpy engine on a fixed rational fixture.

WHY: a second implementation in EXACT arithmetic catches transcription and algebra bugs that
same-code unit tests cannot — the property tests assert invariants the engine must satisfy, but
they do not prove the engine computes the RIGHT number. This does, on a fixture, from the
Pesaran-Shin formula written independently.

Mirrors ~/Documents/consilience/scripts/verify_symbolic.py (the Neuro_Atlas run), retargeted to
`lsa_capstone.gfevd` / `connectedness`. Run: python3 tools/verify_symbolic.py   (needs sympy)
"""
import sys
from pathlib import Path
import sympy as sp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pilot_cross_tier"))
import lsa_capstone as L

H = 4
# fixture: a 3-node VAR(1) with an ASYMMETRIC non-negative coupling and an SPD innovation cov,
# all exact rationals.
Phi = sp.Matrix([[sp.Rational(1, 5), sp.Rational(1, 10), sp.Rational(1, 20)],
                 [0,                 sp.Rational(1, 5),  sp.Rational(1, 10)],
                 [sp.Rational(1, 10), 0,                 sp.Rational(1, 5)]])
Sig = sp.Matrix([[1,                sp.Rational(1, 5),  0],
                 [sp.Rational(1, 5), 1,                 sp.Rational(1, 10)],
                 [0,                sp.Rational(1, 10), 1]])
N = Phi.shape[0]

# ---- independent symbolic GFEVD (Pesaran-Shin), exact rationals ----
A = [sp.eye(N)]
for _h in range(1, H):
    A.append(Phi * A[-1])                       # A_h = Phi^h
th = sp.zeros(N, N)
for i in range(N):
    den = sum((A[h] * Sig * A[h].T)[i, i] for h in range(H))
    for j in range(N):
        num = sum((A[h] * Sig)[i, j] ** 2 for h in range(H))
        th[i, j] = num / Sig[j, j] / den
# row-normalise (matches the engine's final `th / th.sum(1)`)
th_norm = sp.Matrix([[th[i, j] / sum(th[i, k] for k in range(N)) for j in range(N)] for i in range(N)])

# symbolic connectedness (TO/FROM/NET/total), exact
d = [th_norm[i, i] for i in range(N)]
TO_s = [(sum(th_norm[r, c] for r in range(N)) - d[c]) * 100 for c in range(N)]      # column sums (off-diag)
FROM_s = [(sum(th_norm[r, c] for c in range(N)) - d[r]) * 100 for r in range(N)]    # row sums (off-diag)
NET_s = [TO_s[k] - FROM_s[k] for k in range(N)]
total_s = (sum(th_norm) - sum(d)) / N * 100

# ---- engine (float) on the same fixture ----
Phi_f = np.array(Phi.tolist(), dtype=float)
Sig_f = np.array(Sig.tolist(), dtype=float)
th_e = L.gfevd(Phi_f, Sig_f, H=H)
TO_e, FROM_e, NET_e, total_e = L.connectedness(th_e)

th_s = np.array(th_norm.evalf().tolist(), dtype=float)
net_s = np.array([float(x) for x in NET_s])

dmax_th = float(np.max(np.abs(th_s - th_e)))
dmax_net = float(np.max(np.abs(net_s - NET_e)))
dtotal = abs(float(total_s) - float(total_e))

print(f"max |theta_symbolic - theta_engine|   = {dmax_th:.2e}")
print(f"max |NET_symbolic  - NET_engine|      = {dmax_net:.2e}")
print(f"|total_symbolic - total_engine|       = {dtotal:.2e}")
print(f"symbolic NET (exact) = {[str(sp.nsimplify(x)) for x in NET_s]}")

TOL = 1e-12
assert dmax_th < TOL, "SymPy GFEVD re-derivation DISAGREES with the engine"
assert dmax_net < TOL, "SymPy connectedness re-derivation DISAGREES with the engine"
assert dtotal < TOL, "SymPy total-connectedness re-derivation DISAGREES with the engine"
print("OK: independent exact-rational re-derivation matches the engine to < 1e-12")
