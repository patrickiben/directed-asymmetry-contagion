"""
Verify the 2008-equity-crash network reproduces on REDISTRIBUTABLE Stooq data.
USAGE:
  1. Download the 8 daily CSVs from Stooq (see PROPOSED_equity_resourcing_kit.md) into a folder
     named  data_redist/  next to this script. Filenames can be SPX.csv / ^spx_d.csv / spx.csv etc.
  2. Run:   python3 PROPOSED_verify_equity_redist.py
It rebuilds the network with the SAME pipeline as equity_deep.py (non-negative VAR on weekly decline-stress,
ridge=2e-2, Diebold-Yilmaz GFEVD) and checks reproduction. PASS = US is the net-transmitter and DY total ~81%.
"""
import sys, glob, re
import numpy as np, pandas as pd
from pathlib import Path
BASE = Path(__file__).parent
sys.path.insert(0, str(BASE.parent / "pilot_cross_tier"))
import lsa_capstone as L
RIDGE = 2e-2
DATADIR = BASE / "data_redist"

# market <- any filename containing the Stooq stem
STEMS = {"spx": "US", "ukx": "UK", "dax": "Germany", "cac": "France",
         "nkx": "Japan", "hsi": "Hong Kong", "aex": "Neth.", "bvp": "Brazil"}
def find(stem):
    for f in glob.glob(str(DATADIR / "*")):
        if stem in Path(f).name.lower().replace("^", ""):
            return f
    return None

if not DATADIR.exists():
    sys.exit(f"Make the folder {DATADIR} and put the 8 Stooq CSVs in it (see the kit).")
cols = {}
for stem, mkt in STEMS.items():
    f = find(stem)
    if f is None:
        print(f"  [{mkt:11s}] MISSING (need a file containing '{stem}')"); continue
    df = pd.read_csv(f)
    dcol = [c for c in df.columns if c.lower() == "date"][0]
    ccol = [c for c in df.columns if c.lower() == "close"][0]
    s = pd.Series(df[ccol].astype(float).values, index=pd.to_datetime(df[dcol])).dropna()
    cols[mkt] = s
    print(f"  [{mkt:11s}] {Path(f).name}  ({len(s)} rows)")
if len(cols) < 8:
    sys.exit(f"\nOnly {len(cols)}/8 markets loaded — download the missing CSVs and re-run.")

P = pd.DataFrame(cols).sort_index().resample("W").last().loc["2007-01-01":"2010-06-30"].dropna(how="any")
names = list(P.columns); N = len(names)
print(f"\nPanel: {N} markets x {len(P)} weeks ({P.index[0].date()}..{P.index[-1].date()})")
ret = 100 * np.log(P).diff().dropna(); stress = (-ret).clip(lower=0)
Phi, c, Sig = L.fit_var_nonneg(stress.values, ridge=RIDGE)
TO, FROM, NET, tot = L.connectedness(L.gfevd(Phi, Sig))
order = np.argsort(-NET); trans = names[int(np.argmax(NET))]
print(f"\n(1) DY connectedness (redistributable data): total = {tot:.0f}%   [committed ~81%]")
for i in order:
    print(f"     {names[i]:11s} TO={TO[i]:5.1f} FROM={FROM[i]:5.1f} NET={NET[i]:+6.1f}  {'transmitter' if NET[i]>0 else 'receiver'}")

# symmetrization-collapse check (directed-cause signature; committed +49 -> -23)
def proj(a, x, B): a = np.clip(a, 0, x); s = a.sum(); return a * (B / s) if s > B else a
def stepf(Ph, cc, x, a, rng): return np.clip(Ph @ np.clip(x - a, 0, None) + cc + 0.05 * rng.standard_normal(len(x)), 0, None)
def greedy(x, B):
    a = np.zeros_like(x); rem = B
    for i in np.argsort(-x):
        g = min(x[i], rem); a[i] = g; rem -= g
        if rem <= 1e-9: break
    return a
def alloc(sc, x, B): w = np.clip(sc, 0, None); return proj(B * w / w.sum(), x, B) if w.sum() > 0 else np.zeros_like(x)
def rescale(Ph, rho=1.06): ev = max(abs(np.linalg.eigvals(Ph))); return Ph * (rho / ev) if ev > 1e-6 else Ph
S0 = stress.loc["2008-09-01":"2008-11-15"].mean().values + 0.5
def adv(Ph):
    _, _, NETa, _ = L.connectedness(L.gfevd(Ph, Sig)); Phr = rescale(Ph); sc = np.clip(NETa, 0, None); out = {}
    for nm, fn in [("none", None), ("g", "g"), ("t", "t")]:
        tt = []
        for sd in range(12):
            rng = np.random.default_rng(20 + sd); x = S0.copy(); acc = 0.0
            for t in range(20):
                a = np.zeros_like(x) if fn is None else (greedy(x, 2.0) if fn == "g" else alloc(sc, x, 2.0))
                x = stepf(Phr, c, x, a, rng); acc += x.sum()
            tt.append(acc)
        out[nm] = np.mean(tt)
    b = out["none"]; return 100 * (1 - out["t"] / b) - 100 * (1 - out["g"] / b)
print("\n(2) symmetrization sweep (transmitter-minus-greedy advantage)   [committed +49 -> -23]:")
for al in [1.0, 0.75, 0.5, 0.25, 0.0]:
    Pa = al * Phi + (1 - al) * 0.5 * (Phi + Phi.T)
    print(f"     alpha={al:.2f}  advantage={adv(Pa):+6.1f}")

ok = (trans == "US") and (75 <= tot <= 87)
print(f"\n==> {'PASS' if ok else 'CHECK'}: transmitter={trans} (want US), DY total={tot:.0f}% (want ~81%).")
print("If PASS, deposit data_redist/ in your repo and change the equity Data-availability source to Stooq.")
