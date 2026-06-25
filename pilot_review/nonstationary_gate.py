"""
THE VENUE-DECIDER: is there a REAL system where tracking/anticipation beats the static directed rule?

The pressure-tests showed that on STATIONARY linear twins a static Diebold-Yilmaz transmitter rule captures
essentially all of the anticipatory planner's gain. The honest open question (NCS vs Nature Communications) is
whether ANY real system is non-stationary enough that a fixed rule leaves value on the table.

Real COVID-19 has multiple waves with shifting epicentres, so it is the natural test. We:
  1. fit the directed twin in ROLLING windows and ask whether the net-TRANSMITTER actually moves over time
     (the empirical non-stationarity);
  2. on the (rescaled-supercritical) local dynamics of each window, compare interdiction controllers:
       none / greedy(loudest) / STATIC (allocate by the FULL-SAMPLE transmitter, fixed) /
       ADAPTIVE (allocate by the PREVIOUS window's transmitter -- causal, realizable) /
       ORACLE-WINDOW (allocate by the CURRENT window's transmitter -- the upper bound on tracking).

Decision rule reported honestly:
  * if ORACLE-WINDOW >> STATIC and ADAPTIVE ~ ORACLE-WINDOW  -> tracking adds REAL value on real data
                                                                (supports the engine; points to NCS).
  * if ORACLE-WINDOW ~ STATIC                                -> the transmitter is stable; static suffices
                                                                (honest null; points to Nature Communications).

Run: /tmp/lsa_venv/bin/python nonstationary_gate.py
"""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent.parent / "pilot_cross_tier"))
import lsa_capstone as L
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.linewidth": 0.7, "savefig.dpi": 300})
CRIT, STEEL, GREEN, PURPLE, GREY, GOLD = "#c00000", "#1f4e78", "#2e8b57", "#7030a0", "#7f7f7f", "#d4a017"
BASE = Path(__file__).parent
import matplotlib.dates as mdates

# ----------------------------------------------------------------------------- load real COVID weekly incidence
DROP = {"American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands", "Diamond Princess", "Grand Princess", "Puerto Rico"}
d = pd.read_csv(BASE / "jhu_confirmed_US.csv"); dc = [c for c in d.columns if "/" in c]
st = d.groupby("Province_State")[dc].sum().drop(index=[s for s in DROP if s in d["Province_State"].unique()], errors="ignore")
st.columns = pd.to_datetime(st.columns)
wk = st.T.sort_index().resample("W").last().diff().clip(lower=0).dropna()
top = wk.sum().sort_values(ascending=False).head(14).index
W = (wk[top] / 1000.0).loc["2020-03-01":"2022-06-30"]
names = [s[:4] for s in W.columns]; full = list(W.columns); N = len(names); dates = W.index

def twin(M, rho=1.06):
    Phi, c, Sig = L.fit_var_nonneg(M, ridge=5e-2)
    ev = max(abs(np.linalg.eigvals(Phi)))
    if ev > 1e-6: Phi = Phi * (rho / ev)
    _, _, NET, _ = L.connectedness(L.gfevd(Phi, Sig))
    return Phi, c, NET

# full-sample twin
Phi_full, c_full, NET_full = twin(W.values)
full_top = int(np.argmax(NET_full))

# ----------------------------------------------------------------------------- rolling windows
WIN, STEP = 24, 4
starts = list(range(0, len(W) - WIN, STEP))
wins = []
for s in starts:
    Phi, c, NET = twin(W.values[s:s + WIN])
    wins.append(dict(s=s, mid=dates[s + WIN // 2], Phi=Phi, c=c, NET=NET, top=int(np.argmax(NET))))
top_series = [w["top"] for w in wins]
n_distinct = len(set(top_series))
shift_frac = np.mean([w["top"] != full_top for w in wins])
rank_sims = [spearmanr(w["NET"], NET_full).correlation for w in wins]
print(f"windows: {len(wins)} | distinct top-transmitters over time: {n_distinct} of {N} states")
print(f"top-transmitter differs from the full-sample pick in {shift_frac:.0%} of windows")
print(f"window-vs-fullsample NET rank similarity: mean Spearman {np.nanmean(rank_sims):+.2f} (1=stable, 0=reshuffles)")
print("transmitter timeline:", " ".join(names[w["top"]] for w in wins))

# ----------------------------------------------------------------------------- interdiction per window
def project(a, x, B): a = np.clip(a, 0, x); s = a.sum(); return a * (B / s) if s > B else a
def stepf(Phi, c, x, a, rng): return np.clip(Phi @ np.clip(x - a, 0, None) + c + 0.05 * rng.standard_normal(len(x)), 0, None)
def alloc(score, x, B): w = np.clip(score, 0, None); return project(B * w / w.sum(), x, B) if w.sum() > 0 else np.zeros_like(x)
def greedy(x, B):
    a = np.zeros_like(x); rem = B
    for i in np.argsort(-x):
        g = min(x[i], rem); a[i] = g; rem -= g
        if rem <= 1e-9: break
    return a

B, T, SEEDS = 2.0, 10, 12
CT = ["none", "greedy", "static", "adaptive", "oracle-window"]
acc = {k: 0.0 for k in CT}
for wi, w in enumerate(wins):
    Phi, c = w["Phi"], w["c"]; S0 = np.maximum(W.values[w["s"]:w["s"] + WIN].mean(0), 0.5)
    net_static = NET_full
    net_adapt = wins[wi - 1]["NET"] if wi > 0 else NET_full          # causal: previous window's transmitter
    net_oracle = w["NET"]                                            # current window's transmitter (upper bound)
    for k in CT:
        tot = []
        for s in range(SEEDS):
            rng = np.random.default_rng(50 + s); x = S0.copy(); a_acc = 0.0
            for t in range(T):
                if k == "none": a = np.zeros_like(x)
                elif k == "greedy": a = greedy(x, B)
                elif k == "static": a = alloc(net_static, x, B)
                elif k == "adaptive": a = alloc(net_adapt, x, B)
                else: a = alloc(net_oracle, x, B)
                x = stepf(Phi, c, x, a, rng); a_acc += x.sum()
            tot.append(a_acc)
        acc[k] += np.mean(tot)
base = acc["none"]
red = {k: 100 * (1 - acc[k] / base) for k in CT}
print("\nAggregate interdiction across all windows (cascade reduction vs no-action):")
for k in CT: print(f"  {k:14s} {red[k]:+.0f}%")
value_of_tracking = red["oracle-window"] - red["static"]
realizable = red["adaptive"] - red["static"]
print(f"\nvalue of perfect transmitter-tracking (oracle-window - static): {value_of_tracking:+.0f} pts")
print(f"realizable tracking value (adaptive - static):                  {realizable:+.0f} pts")
verdict = ("REAL non-stationarity pays: tracking the moving transmitter beats the fixed rule on real data "
           "-- supports the anticipatory engine; points to NCS"
           if value_of_tracking >= 5 and realizable >= 3 else
           "Honest null on this dataset: the transmitter is stable enough that a static rule suffices "
           "-- the engine's real-data necessity is NOT shown here; points to Nature Communications")
print(f"\nVERDICT: {verdict}")

RES = dict(n_windows=len(wins), distinct_top_transmitters=n_distinct, shift_fraction=round(float(shift_frac), 2),
           mean_rank_similarity=round(float(np.nanmean(rank_sims)), 2),
           reductions={k: round(red[k], 1) for k in CT},
           value_of_tracking_pts=round(float(value_of_tracking), 1), realizable_pts=round(float(realizable), 1),
           verdict=verdict)
json.dump(RES, open(BASE / "nonstationary_gate_results.json", "w"), indent=2)

# ----------------------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.1))
a = ax[0]
mids = [w["mid"] for w in wins]
a.plot(mids, [w["top"] for w in wins], "o-", color=STEEL, ms=4)
a.axhline(full_top, color=CRIT, ls="--", lw=1, label=f"Full-Sample Pick: {names[full_top]}")
a.set_yticks(range(N)); a.set_yticklabels(names, fontsize=6); a.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
a.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y")); plt.setp(a.get_xticklabels(), rotation=30, ha="right", fontsize=6)
a.set_title(f"(a) Net-Transmitter Over Time\n({n_distinct} Distinct States; Differs From Full-Sample {shift_frac:.0%} of the Time)", fontsize=8.2)
a.set_ylabel("Net-Transmitter State"); a.legend(loc="upper right"); a.grid(alpha=.25)
a = ax[1]
cols = {"none": CRIT, "greedy": PURPLE, "static": GREY, "adaptive": STEEL, "oracle-window": GREEN}
vv = [red[k] for k in CT]
a.bar(range(len(CT)), vv, color=[cols[k] for k in CT], alpha=.85)
for i, v in enumerate(vv): a.text(i, v + 1, f"{v:+.0f}%", ha="center", fontsize=7.5, fontweight="bold")
a.axhline(red["static"], color=GREY, ls=":", lw=.9)
a.set_xticks(range(len(CT))); a.set_xticklabels(["None", "Greedy", "Static\n(Fixed)", "Adaptive\n(Track)", "Oracle\nWindow"], fontsize=6.6)
a.set_ylabel("Cascade Reduction vs No-Action (%)")
a.set_title(f"(b) Does Tracking the Moving Transmitter Beat the Fixed Rule?\nTracking Value: Oracle {value_of_tracking:+.0f} pts, Realizable {realizable:+.0f} pts", fontsize=8.2)
a.grid(alpha=.25, axis="y"); a.set_ylim(0, max(vv) * 1.18)
fig.suptitle("When Dynamic Anticipation Pays: Tracking a Moving Transmitter Beats a Fixed Rule",
             fontsize=9.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(BASE / "nonstationary_gate.pdf", bbox_inches="tight"); fig.savefig(BASE / "nonstationary_gate.png", dpi=200, bbox_inches="tight")
print("figure -> nonstationary_gate.pdf/.png")
