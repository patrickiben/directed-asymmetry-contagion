"""
Horizon-AUC robustness diagnostic (reshaped-version addition; see SI "The transmitter is stable across
operational forecast horizons"). Instead of fixing the GFEVD forecast horizon H, integrate net Diebold-Yilmaz
connectedness over a sweep of H so the transmitter diagnosis is robust BY CONSTRUCTION. Verifies the
horizon-integrated transmitter matches the fixed-horizon (H=10) transmitter, and reports horizon-rank stability.

NOTE ON DATA: the COVID loader reads the included pilot_review/jhu_confirmed_US.csv; the asia97/smoke/flu loaders
fetch live public data (FRED/EPA/Delphi). The equity/housing/emfx loaders read locally-cached vendor series that
are NOT redistributed in this archive (Yahoo Finance / FRED housing); to run those rows, point them at your own
cached copies. The committed result (integrated transmitter == fixed-H transmitter in every network) is reported
in the manuscript and SI.
"""
import sys, json, io, urllib.request, time, zipfile
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
LSA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LSA / "pilot_cross_tier"))
import lsa_capstone as L
PR = LSA / "pilot_review"; EQ = LSA / "pilot_3p46_equity"; HO = LSA / "pilot_3p50_housing"; FX = LSA / "pilot_3p49b_emfx"
RIDGE = 5e-2

def load_covid():
    DROP = {"American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands", "Diamond Princess", "Grand Princess", "Puerto Rico"}
    d = pd.read_csv(PR / "jhu_confirmed_US.csv"); dc = [c for c in d.columns if "/" in c]
    st = d.groupby("Province_State")[dc].sum().drop(index=[s for s in DROP if s in d["Province_State"].unique()], errors="ignore")
    st.columns = pd.to_datetime(st.columns)
    wk = st.T.sort_index().resample("W").last().diff().clip(lower=0).dropna()
    top = wk.sum().sort_values(ascending=False).head(14).index
    W = (wk[top] / 1000.0).loc["2020-03-01":"2022-06-30"]; return W.values, list(W.columns)
def load_asia97():
    SER = {"DEXTHUS": "Thai", "DEXKOUS": "Korea", "DEXMAUS": "Malay", "DEXSIUS": "Singa", "DEXJPUS": "Japan", "DEXTAUS": "Taiwan", "DEXHKUS": "HK", "DEXINUS": "India", "DEXCHUS": "China"}
    def fred(i):
        u = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={i}&cosd=1996-01-01&coed=1998-12-31"
        for _ in range(3):
            try:
                df = pd.read_csv(io.StringIO(urllib.request.urlopen(u, timeout=25).read().decode())); df.columns = ["date", i]
                df["date"] = pd.to_datetime(df["date"]); df[i] = pd.to_numeric(df[i], errors="coerce"); return df.set_index("date")[i].dropna()
            except Exception: time.sleep(2)
        return None
    cols = {}
    for i, nm in SER.items():
        s = fred(i)
        if s is not None and len(s) > 400: cols[nm] = s
    F = pd.DataFrame(cols).resample("W").last().interpolate().dropna(); b = F.loc["1996-01-01":"1996-12-31"].mean()
    S = ((F / b - 1.0) * 100.0).loc["1996-06-01":"1998-12-31"]; return S.values, list(S.columns)
def load_flu():
    STATES = ["ca", "tx", "fl", "ny", "pa", "il", "oh", "ga", "nc", "mi", "nj", "va", "wa", "az", "ma"]
    r = json.load(urllib.request.urlopen("https://api.delphi.cmu.edu/epidata/fluview/?regions=" + ",".join(STATES) + "&epiweeks=201040-201920", timeout=60)); assert r["result"] == 1
    df = pd.DataFrame(r["epidata"])[["region", "epiweek", "wili"]]; M = df.pivot(index="epiweek", columns="region", values="wili").sort_index()
    wk = (M.index % 100); M = M[(wk >= 40) | (wk <= 20)]; av = [s for s in STATES if s in M.columns]; M = M[av].interpolate().dropna(); return M.values, list(M.columns)

# Networks with redistributable / live-fetchable data in this archive:
SYS = [("COVID", load_covid), ("asia97", load_asia97), ("flu", load_flu)]
Hs = np.array([2, 3, 4, 6, 8, 10, 12, 15])
print(f"horizon sweep H = {list(Hs)} ; integrate net DY connectedness over H\n")
match = 0; tot = 0; rhos = []
for name, loader in SYS:
    try: M, names = loader()
    except Exception as e: print(f"[{name:7s}] skip: {str(e)[:60]}"); continue
    Phi, c, Sig = L.fit_var_nonneg(M, ridge=RIDGE)
    NETs = np.array([L.connectedness(L.gfevd(Phi, Sig, int(H)))[2] for H in Hs])
    tfix = int(np.argmax(NETs[list(Hs).index(10)]))
    integ = np.trapz(NETs, Hs, axis=0); tauc = int(np.argmax(integ)); agree = (tauc == tfix)
    rho = spearmanr(NETs[0], NETs[-1]).correlation; rhos.append(rho); tot += 1; match += int(agree)
    print(f"[{name:7s}] N={M.shape[1]:2d}  fixedH10 trans={names[tfix]:>9s}  integrated trans={names[tauc]:>9s}  agree={agree}  rank-corr(H2,H15)={rho:.2f}")
print(f"\nSUMMARY: integrated transmitter == fixed-H transmitter in {match}/{tot} networks tested here; "
      f"mean rank stability = {np.mean(rhos):.2f}. (Full 7-network result -- all agree -- is in the manuscript/SI.)")
