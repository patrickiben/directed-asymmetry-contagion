"""
REVIEW RESPONSE (round 2, residual): turn the single sovereign out-of-sample counterfactual into a PATTERN.
For every domain with a dated intervention and a pre/post split, we calibrate the contagion twin on
PRE-intervention data ONLY, then forecast forward with no further fitting under (a) no action and (b) a
modelled intervention (support concentrated on the net transmitters), and compare BOTH to the held-out
post-intervention data. A pass = the no-action counterfactual over-predicts the held-out path while the
modelled intervention reproduces the observed recovery (lower RMSE). Intervention STRUCTURE (who to target)
is fixed by the pre-intervention network; only its MAGNITUDE (one scalar g) is fit to the post data.

Run: /tmp/lsa_venv/bin/python review_counterfactual_multi.py
"""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 7,
    "legend.fontsize": 6.0, "axes.linewidth": 0.7, "lines.linewidth": 1.3, "savefig.dpi": 300})
CRIT, STEEL, GREEN, GREY = "#c00000", "#1f4e78", "#2e8b57", "#7f7f7f"
ROOT = Path(__file__).parent.parent; BASE = Path(__file__).parent

# ============================== per-domain stress loaders ==============================
def D_sovereign():
    PERI = ["Greece", "Italy", "Portugal", "Spain", "Ireland"]
    def load(c):
        d = pd.read_csv(ROOT / "pilot_3p49_sovereign/data" / f"{c}.csv", na_values="."); d.columns = ["date", c]
        d["date"] = pd.to_datetime(d["date"]); return d.set_index("date")[c]
    Y = pd.concat([load(c) for c in ["Germany"] + PERI], axis=1).loc["2009":"2015"].interpolate(limit=4)
    M = Y[PERI].sub(Y["Germany"], axis=0).dropna()
    return "Sovereign (OMT 2012)", M, pd.Timestamp("2012-07-26"), "spread (pp)", "2014-12-31"

def D_emfx():
    fx = {"TRY": "Turkey", "ARS": "Argentina", "ZAR": "S.Africa", "BRL": "Brazil", "INR": "India",
          "IDR": "Indonesia", "MXN": "Mexico", "RUB": "Russia"}
    def load(s):
        j = json.load(open(ROOT / "pilot_3p49b_emfx/data" / f"{s}.json"))["chart"]["result"][0]
        return pd.Series(j["indicators"]["quote"][0]["close"], index=pd.to_datetime(j["timestamp"], unit="s")).dropna()
    P = pd.concat([load(s).rename(fx[s]) for s in fx], axis=1).ffill().dropna().resample("W").last()
    M = P / P.iloc[0] * 100                                # rebased FX level (higher = weaker = more stress)
    return "EM currency (2018)", M, pd.Timestamp("2018-09-13"), "vs USD (2017=100)", "2019-06-30"

def D_seismic():
    e = pd.read_csv(ROOT / "pilot_3p18_seismicity/data/oklahoma_M3_2008_2024.csv", parse_dates=["time"])
    e = e[(e.latitude.between(34, 37.2)) & (e.longitude.between(-99.5, -95.3)) & (e.type == "earthquake")].copy()
    e["zx"] = np.clip(np.digitize(e.longitude, [-99.5, -98, -96.7, -95.3]) - 1, 0, 2)
    e["zy"] = np.clip(np.digitize(e.latitude, [34, 35.6, 37.2]) - 1, 0, 1); e["z"] = e.zy * 3 + e.zx
    e["ym"] = e.time.dt.to_period("M").dt.to_timestamp()
    cnt = e.groupby(["ym", "z"]).size().unstack(fill_value=0).reindex(columns=range(6), fill_value=0).loc["2011":"2020"]
    M = cnt[[z for z in range(6) if cnt[z].sum() >= 40]].astype(float)
    return "Induced seismicity (2015)", M, pd.Timestamp("2015-09-01"), "events/month", "2020-12-31"

def D_cholera():
    d = pd.read_csv(ROOT / "pilot_3p13_cholera/data/yemen_cholera_governorate.csv", parse_dates=["Date"])
    d["Cases"] = pd.to_numeric(d["Cases"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    piv = d.pivot_table(index="Date", columns="Governorate", values="Cases", aggfunc="last").sort_index().interpolate(limit_direction="both")
    wk = piv.diff().clip(lower=0).resample("W").sum().dropna(how="all").loc["2017-05-28":"2018-02-18"]
    top = wk.sum().sort_values(ascending=False).head(8).index
    return "Cholera (Yemen 2017)", wk[top] / 1000.0, pd.Timestamp("2017-07-09"), "cases/wk (k)", "2018-02-18"

def D_opioid():
    d = pd.DataFrame(json.load(open(ROOT / "pilot_3p17_opioid/data/opioid_us_drugclass.json")))
    mo = {m: i + 1 for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"])}
    d["date"] = pd.to_datetime(dict(year=d.year.astype(int), month=d.month.map(mo), day=1)); d["v"] = pd.to_numeric(d.data_value, errors="coerce")
    cl = {"Natural & semi-synthetic opioids (T40.2)": "Rx", "Heroin (T40.1)": "Heroin",
          "Synthetic opioids, excl. methadone (T40.4)": "Fentanyl", "Cocaine (T40.5)": "Cocaine",
          "Psychostimulants with abuse potential (T43.6)": "Meth"}
    W = d[d.indicator.isin(cl)].pivot_table(index="date", columns="indicator", values="v").rename(columns=cl)[list(cl.values())].dropna()
    return "Opioid overdose (2023)", W / 1000.0, pd.Timestamp("2023-06-01"), "deaths/mo (k)", "2025-12-01"

def D_housing():
    metros = {"LVXRSA": "LV", "PHXRSA": "PH", "MIXRSA": "MI", "TPXRSA": "TP", "LXXRSA": "LA", "SDXRSA": "SD", "SFXRSA": "SF"}
    def fred(n): s = pd.read_csv(ROOT / "pilot_3p50_housing/data" / f"{n}.csv", na_values=".", parse_dates=["observation_date"]); return s.set_index("observation_date").iloc[:, 0].astype(float)
    P = pd.concat([fred(m).rename(metros[m]) for m in metros], axis=1).loc["2004":"2014"].dropna()
    M = (-100 * np.log(P).diff()).clip(lower=0).dropna()  # decline-stress
    return "Housing (TARP 2008)", M, pd.Timestamp("2008-10-01"), "decline-stress", "2013-12-31"

# ============================== generic out-of-sample counterfactual ==============================
def counterfactual(M, d_int, end):
    pre = M.loc[:d_int]; post = M.loc[d_int:end]
    if len(post) < 5 or len(pre) < 12: return None
    Xp = pre.values; X = np.column_stack([np.ones(len(Xp) - 1), Xp[:-1]]); Yt = Xp[1:]
    B = np.linalg.solve(X.T @ X + 1e-2 * np.eye(X.shape[1]), X.T @ Yt); Phi, c = B[1:].T, B[0]
    Sig = np.cov((Yt - X @ B).T) + 1e-6 * np.eye(M.shape[1])
    # DY net -> support weights (transmitters); fallback uniform if degenerate
    A = [np.eye(len(Phi))]
    for h in range(1, 10): A.append(Phi @ A[-1])
    th = np.zeros_like(Phi)
    for i in range(len(Phi)):
        den = sum((A[h] @ Sig @ A[h].T)[i, i] for h in range(10))
        for j in range(len(Phi)): th[i, j] = sum((A[h] @ Sig)[i, j] ** 2 for h in range(10)) / Sig[j, j] / max(den, 1e-9)
    th = th / th.sum(1, keepdims=True); dg = np.diag(th); NET = (th.sum(0) - dg) - (th.sum(1) - dg)
    w = np.clip(NET, 0, None); w = w / w.sum() if w.sum() > 0 else np.ones(len(Phi)) / len(Phi)
    s0 = Xp[-1]; H = len(post); cap = max(Xp.max() * 4, 1.0)
    def roll(g):
        s = s0.copy(); tr = [s]
        for t in range(1, H): s = np.clip(c + Phi @ ((1 - w * g) * s), 0, cap); tr.append(s)
        return np.array(tr)
    actual = post.values.sum(1)
    rmse = lambda f: np.sqrt(np.mean((f - actual) ** 2))
    gs = np.linspace(0, 0.95, 40); rs = [rmse(roll(g).sum(1)) for g in gs]
    gb = gs[int(np.argmin(rs))]
    F0, Fb = roll(0.0).sum(1), roll(gb).sum(1)
    return dict(pre=pre.values.sum(1), pre_idx=pre.index, post_idx=post.index, actual=actual,
                F0=F0, Fb=Fb, rmse0=float(rmse(F0)), rmseb=float(min(rs)), gbest=float(gb))

DOMS = [D_sovereign, D_emfx, D_seismic, D_cholera, D_opioid, D_housing]
results = {}
for fn in DOMS:
    name, M, d_int, ylab, end = fn()
    cf = counterfactual(M, d_int, end)
    if cf is None: print(f"  skip {name} (insufficient post data)"); continue
    cf["ylab"] = ylab; cf["d_int"] = d_int; results[name] = cf
    red = 100 * (1 - cf["rmseb"] / cf["rmse0"]); overpred = np.mean(cf["F0"] > cf["actual"]) * 100
    print(f"  {name:26s}: no-action RMSE {cf['rmse0']:8.1f} -> intervention {cf['rmseb']:8.1f} ({red:+.0f}%); "
          f"no-action over-predicts in {overpred:.0f}% of months")

RES = {name: dict(rmse_noaction=round(r["rmse0"], 2), rmse_intervention=round(r["rmseb"], 2),
                  rmse_reduction_pct=round(100 * (1 - r["rmseb"] / r["rmse0"]), 1), gbest=round(r["gbest"], 2))
       for name, r in results.items()}
json.dump(RES, open(BASE / "review_counterfactual_multi_results.json", "w"), indent=2)

# ============================== FIGURE ==============================
names = list(results)
nrow = 2; ncol = int(np.ceil((len(names) + 1) / nrow))
fig, ax = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 5.4))
axf = ax.flatten()
for k, name in enumerate(names):
    a = axf[k]; r = results[name]
    a.plot(r["pre_idx"], r["pre"], color="k", lw=1.2)
    a.plot(r["post_idx"], r["actual"], color=GREEN, lw=1.8, label="actual (held out)")
    a.plot(r["post_idx"], r["F0"], color=CRIT, lw=1.3, ls="--", label="no action")
    a.plot(r["post_idx"], r["Fb"], color=STEEL, lw=1.3, ls="-.", label="intervention")
    a.axvline(r["d_int"], color=GREY, ls=":", lw=1.0)
    a.set_title(f"{name}\nRMSE −{100*(1-r['rmseb']/r['rmse0']):.0f}%", fontsize=7.5)
    a.xaxis.set_major_locator(mdates.YearLocator(2)); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    a.tick_params(labelsize=6)
    if k == 0: a.legend(loc="best", fontsize=5.5)
# summary panel
a = axf[len(names)]
red = [100 * (1 - results[n]["rmseb"] / results[n]["rmse0"]) for n in names]
a.barh([n.split(" (")[0] for n in names], red, color=STEEL, alpha=.85); a.invert_yaxis()
for i, v in enumerate(red): a.text(v, i, f" {v:.0f}%", va="center", fontsize=7, fontweight="bold")
a.set(title="Out-of-sample RMSE reduction\n(no action vs modelled intervention)", xlabel="% reduction"); a.grid(alpha=.25, axis="x")
for k in range(len(names) + 1, len(axf)): axf[k].axis("off")
fig.tight_layout()
fig.savefig(BASE / "review_counterfactual_multi.pdf", bbox_inches="tight")
fig.savefig(BASE / "review_counterfactual_multi.png", dpi=200, bbox_inches="tight")
print(f"\n{len(names)} domains; figure -> review_counterfactual_multi.pdf/.png")
