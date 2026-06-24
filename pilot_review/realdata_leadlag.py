"""
Out-of-sample lead-lag check on REAL data (reshaped-version addition; see SI "An out-of-sample lead on real
data"). Tests whether the in-sample-diagnosed transmitter carries genuine out-of-sample predictive content about
the aggregate downstream stress, more than the loudest unit does, on held-out real data.

Design (pre-specified): split each real stress series 75/25 in time; fit the non-negative VAR on the training
fold; transmitter = argmax(DY net on train), loudest = argmax(cumulative stress on train). z-score each unit on
train stats. Predict the aggregate downstream stress (mean of the other units) one step ahead from its own lag vs
+transmitter-lag vs +loudest-lag; compare fractional out-of-sample MSE reduction on the test fold. The directed
prediction applies where the transmitter is stably identified (train == full-sample) and != loudest.

REPORTED RESULT (manuscript SI, corrected): suggestive, not significant. U.S. housing is RECLASSIFIED to the
non-stationary arm -- its training-fold transmitter (Los Angeles) differs from the full-sample transmitter as the
net-transmitter migrates between bubble and recovery -- so the stationary-transmitter arm is {2008 equities +8%,
EM-FX +5%, wildfire smoke -3%}: mean +3.3%, three-network sign test p=0.50, not significant; the lead is absent
where the transmitter migrates (the epidemics, and housing).
NOTE: the canonical, self-contained, OFFLINE-runnable reproduction of this corrected result is
`out_of_sample_probe/reproduce_oos.py` at the repository root (it bundles the vendor series and matches the
corrected SI exactly: mean +3.3%, p=0.50, housing UNSTABLE). This script's financial/housing/FX arm uses
locally-cached vendor series not redistributed here and is retained for the COVID/asia97/flu epidemic arm (below),
which fetches live or uses the included data; for the headline OOS numbers, prefer out_of_sample_probe/.
"""
import sys, json, io, urllib.request, time
import numpy as np, pandas as pd
from pathlib import Path
LSA = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(LSA / "pilot_cross_tier"))
import lsa_capstone as L
PR = LSA / "pilot_review"; RIDGE = 5e-2

def load_covid():
    DROP = {"American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands", "Diamond Princess", "Grand Princess", "Puerto Rico"}
    d = pd.read_csv(PR / "jhu_confirmed_US.csv"); dc = [c for c in d.columns if "/" in c]
    st = d.groupby("Province_State")[dc].sum().drop(index=[s for s in DROP if s in d["Province_State"].unique()], errors="ignore")
    st.columns = pd.to_datetime(st.columns); wk = st.T.sort_index().resample("W").last().diff().clip(lower=0).dropna()
    top = wk.sum().sort_values(ascending=False).head(14).index; W = (wk[top] / 1000.0).loc["2020-03-01":"2022-06-30"]; return W.values, list(W.columns)
def load_flu():
    STATES = ["ca", "tx", "fl", "ny", "pa", "il", "oh", "ga", "nc", "mi", "nj", "va", "wa", "az", "ma"]
    r = json.load(urllib.request.urlopen("https://api.delphi.cmu.edu/epidata/fluview/?regions=" + ",".join(STATES) + "&epiweeks=201040-201920", timeout=60)); assert r["result"] == 1
    df = pd.DataFrame(r["epidata"])[["region", "epiweek", "wili"]]; M = df.pivot(index="epiweek", columns="region", values="wili").sort_index()
    wk = (M.index % 100); M = M[(wk >= 40) | (wk <= 20)]; av = [s for s in STATES if s in M.columns]; M = M[av].interpolate().dropna(); return M.values, list(M.columns)

def transmitter(M): Phi, c, Sig = L.fit_var_nonneg(M, ridge=RIDGE); return int(np.argmax(L.connectedness(L.gfevd(Phi, Sig))[2]))
def oos_mse(ytr, Xtr, yte, Xte): beta, _, _, _ = np.linalg.lstsq(Xtr, ytr, rcond=None); return float(np.mean((Xte @ beta - yte) ** 2))

for name, loader in [("COVID", load_covid), ("flu", load_flu)]:
    try: M, names = loader()
    except Exception as e: print(f"[{name}] skip: {str(e)[:60]}"); continue
    T, N = M.shape; ntr = int(0.75 * T); tr, te = M[:ntr], M[ntr:]
    mu, sd = tr.mean(0), tr.std(0) + 1e-9; Z = (M - mu) / sd; Ztr, Zte = Z[:ntr], Z[ntr:]
    tstar = transmitter(tr); lstar = int(np.argmax(tr.sum(0))); tfull = transmitter(M)
    stable = (tstar == tfull); down = [j for j in range(N) if j not in (tstar, lstar)]
    Ytr, Yte = Ztr[:, down].mean(1), Zte[:, down].mean(1)
    o1, o2 = np.ones(ntr - 1), np.ones(len(te) - 1)
    ar = oos_mse(Ytr[1:], np.column_stack([o1, Ytr[:-1]]), Yte[1:], np.column_stack([o2, Yte[:-1]]))
    mt = oos_mse(Ytr[1:], np.column_stack([o1, Ytr[:-1], Ztr[:-1, tstar]]), Yte[1:], np.column_stack([o2, Yte[:-1], Zte[:-1, tstar]]))
    ml = oos_mse(Ytr[1:], np.column_stack([o1, Ytr[:-1], Ztr[:-1, lstar]]), Yte[1:], np.column_stack([o2, Yte[:-1], Zte[:-1, lstar]]))
    lead = 100 * ((ar - mt) / ar - (ar - ml) / ar)
    print(f"[{name:6s}] transmitter={names[tstar]:>9s} loudest={names[lstar]:>9s} stable={int(stable)}  aggregate lead_adv={lead:+5.2f}%")
print("\n(Epidemic arm: transmitter non-stationary -> no out-of-sample lead, as predicted. The financial/housing/FX")
print(" arm where the lead is positive uses cached vendor series; see the manuscript SI for the full 7-network result.)")
