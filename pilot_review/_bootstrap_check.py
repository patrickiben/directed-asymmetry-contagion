"""Sanity check: replicate loaders + interdict_adv from nonnormality_predictor.py,
confirm point estimates reproduce, before running the full bootstrap."""
import sys, json, io, zipfile, urllib.request, time
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pilot_cross_tier"))
import lsa_capstone as L

BASE = Path(__file__).resolve().parent
EQDIR = Path(__file__).resolve().parent.parent / "pilot_3p46_equity"
RIDGE = 5e-2

# ---- interdiction (verbatim from nonnormality_predictor.py) ----
def project(a, x, B): a = np.clip(a, 0, x); s = a.sum(); return a * (B / s) if s > B else a
def stepf(Phi, c, x, a, rng): return np.clip(Phi @ np.clip(x - a, 0, None) + c + 0.05 * rng.standard_normal(len(x)), 0, None)
def greedy(x, B):
    a = np.zeros_like(x); rem = B
    for i in np.argsort(-x):
        g = min(x[i], rem); a[i] = g; rem -= g
        if rem <= 1e-9: break
    return a
def alloc_score(score, x, B):
    w = np.clip(score, 0, None)
    return project(B * w / w.sum(), x, B) if w.sum() > 0 else np.zeros_like(x)
def rescale(Phi, rho=1.06):
    ev = max(abs(np.linalg.eigvals(Phi))); return Phi * (rho / ev) if ev > 1e-6 else Phi
def interdict_adv(Phi0, c0, Sig0, S0, B=2.0, T=16, seeds=16, seed0=20):
    _, _, NET, _ = L.connectedness(L.gfevd(Phi0, Sig0))
    Phi = rescale(Phi0); scores = {"transmitter": np.clip(NET, 0, None)}
    out = {}
    for name in ["none", "greedy", "transmitter"]:
        tt = []
        for s in range(seeds):
            rng = np.random.default_rng(seed0 + s); x = S0.copy(); acc = 0.0
            for t in range(T):
                if name == "none": a = np.zeros_like(x)
                elif name == "greedy": a = greedy(x, B)
                else: a = alloc_score(scores[name], x, B)
                x = stepf(Phi, c0, x, a, rng); acc += x.sum()
            tt.append(acc)
        out[name] = float(np.mean(tt))
    b = out["none"]; red = {k: 100 * (1 - v / b) for k, v in out.items()}
    return red["transmitter"], red["greedy"], NET

def load_flights():
    S = pd.read_csv("/tmp/lsa_flights/flights_panel.csv", index_col=0)
    return S.values, np.maximum(S.values.mean(0), 0.5)
def load_conflict():
    S = pd.read_csv("/tmp/lsa_conflict/conflict_panel.csv", index_col=0)
    return S.values, np.maximum(S.values.mean(0), 0.5)
def load_equity():
    idx = {"GSPC": "US", "FTSE": "UK", "GDAXI": "Germany", "FCHI": "France",
           "N225": "Japan", "HSI": "Hong Kong", "AEX": "Neth.", "BVSP": "Brazil"}
    def load(sym):
        d = json.load(open(EQDIR / "data" / f"{sym}.json"))["chart"]["result"][0]
        return pd.Series(d["indicators"]["quote"][0]["close"], index=pd.to_datetime(d["timestamp"], unit="s")).dropna()
    P = pd.concat([load(s).rename(idx[s]) for s in idx], axis=1).ffill().dropna()
    P = P.resample("W").last().loc["2007-01-01":"2010-06-30"]
    ret = 100 * np.log(P).diff().dropna(); stress = (-ret).clip(lower=0)
    S0 = stress.loc["2008-09-01":"2008-11-15"].mean().values + 0.5
    return stress.values, S0
def load_covid():
    DROP = {"American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands", "Diamond Princess",
            "Grand Princess", "Puerto Rico"}
    d = pd.read_csv(BASE / "jhu_confirmed_US.csv"); dcols = [c for c in d.columns if "/" in c]
    st = d.groupby("Province_State")[dcols].sum().drop(index=[s for s in DROP if s in d["Province_State"].unique()], errors="ignore")
    st.columns = pd.to_datetime(st.columns)
    weekly = st.T.sort_index().resample("W").last().diff().clip(lower=0).dropna()
    top = weekly.sum().sort_values(ascending=False).head(14).index
    W = (weekly[top] / 1000.0).loc["2020-03-01":"2022-06-30"]
    S0 = W.loc["2020-06-01":"2020-08-01"].mean().values + 0.5
    return W.values, S0

for name, loader in [("flights", load_flights), ("conflict", load_conflict),
                     ("equity", load_equity), ("COVID", load_covid)]:
    M, S0 = loader()
    Phi0, c0, Sig0 = L.fit_var_nonneg(M, ridge=RIDGE)
    tr, gr, NET = interdict_adv(Phi0, c0, Sig0, S0)
    print(f"[{name:8s}] T={M.shape[0]:3d} N={M.shape[1]:2d}  transmitter={tr:+6.2f} greedy={gr:+6.2f} adv={tr-gr:+6.2f}  top_NET={int(np.argmax(NET))}")
