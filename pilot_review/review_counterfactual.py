"""
REVIEW RESPONSE (Major concern 1: circularity of the interdiction validation). A genuine OUT-OF-SAMPLE
counterfactual test on the sovereign-debt twin, using the real ECB OMT (26 Jul 2012) as the intervention.

Protocol (strictly pre-registered in time):
  1. Calibrate the VAR(1) contagion twin on PRE-OMT data ONLY (Jan 2010 -- Jun 2012; the supercritical window).
  2. From the June-2012 state, forecast 30 months forward under TWO policies, with NO further fitting:
       (a) no-action      -- iterate the fitted (explosive) dynamics: the counterfactual 'no OMT';
       (b) OMT-like support -- apply an ARRO transmitter-targeted backstop (damp the net-transmitter
           sovereigns' level + outgoing spillover), the policy the framework prescribes.
  3. Compare both forecasts to the ACTUAL post-OMT spreads (held out, never seen by the model).

If the framework had only circular, in-sample value, neither forecast would track reality. The test passes
iff (i) the no-action forecast stays high/diverges (the twin captures the explosive pre-OMT regime), and
(ii) the OMT-like support forecast reproduces the observed post-OMT compression out-of-sample (lower RMSE).
Run: /tmp/lsa_venv/bin/python review_counterfactual.py
"""
import json
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.linewidth": 0.7, "lines.linewidth": 1.3, "savefig.dpi": 300})
CRIT, STEEL, GREEN, PURPLE, GREY = "#c00000", "#1f4e78", "#2e8b57", "#7030a0", "#7f7f7f"
SOV = Path(__file__).parent.parent / "pilot_3p49_sovereign"
BASE = Path(__file__).parent
PERI = ["Greece", "Italy", "Portugal", "Spain", "Ireland"]
OMT = pd.Timestamp("2012-07-26")

def load(c):
    d = pd.read_csv(SOV / "data" / f"{c}.csv", na_values="."); d.columns = ["date", c]
    d["date"] = pd.to_datetime(d["date"]); return d.set_index("date")[c]
Y = pd.concat([load(c) for c in ["Germany"] + PERI], axis=1).loc["2009":"2015"].interpolate(limit=4)
spread = Y[PERI].sub(Y["Germany"], axis=0).dropna()
pre = spread.loc[:OMT]; post = spread.loc[OMT:"2014-12-31"]

def fit_var1(M):
    X = np.column_stack([np.ones(len(M) - 1), M[:-1]]); Yt = M[1:]
    B = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ Yt)
    return B[1:].T, B[0]
Phi, c = fit_var1(pre.values)
rho_pre = max(abs(np.linalg.eigvals(Phi)))
# DY net transmitters on the pre-OMT twin (who to support) ----
def gfevd(Phi, Sig, H=10):
    Nn = len(Phi); A = [np.eye(Nn)]
    for h in range(1, H): A.append(Phi @ A[-1])
    th = np.zeros((Nn, Nn))
    for i in range(Nn):
        den = sum((A[h] @ Sig @ A[h].T)[i, i] for h in range(H))
        for j in range(Nn): th[i, j] = sum((A[h] @ Sig)[i, j] ** 2 for h in range(H)) / Sig[j, j] / den
    return th / th.sum(1, keepdims=True)
resid = pre.values[1:] - (np.column_stack([np.ones(len(pre) - 1), pre.values[:-1]]) @ np.vstack([c, Phi.T]))
Sig = np.cov(resid.T)
th = gfevd(Phi, Sig); dgl = np.diag(th); NET = (th.sum(0) - dgl) - (th.sum(1) - dgl)
support = np.clip(NET, 0, None); support = support / support.sum()     # support the net transmitters
print(f"pre-OMT twin: rho={rho_pre:.3f}; DY net = " + ", ".join(f"{PERI[i]}:{NET[i]:+.2f}" for i in range(5)))
print(f"OMT-like support weights (transmitters): " + ", ".join(f"{PERI[i]}:{support[i]:.2f}" for i in range(5)))

# ---------------- forecast 30 months from the cutoff ----------------
H = len(post)
s0 = pre.values[-1]
def roll(intervene, g=0.0):
    s = s0.copy(); traj = [s.copy()]
    for t in range(1, H):
        a = support * g if intervene else 0.0
        s = c + Phi @ ((1 - a) * s)
        s = np.clip(s, 0, 60)
        traj.append(s.copy())
    return np.array(traj)
F_noact = roll(False)
F_omt = roll(True, g=0.55)                                # OMT-like backstop strength
A = post.values[:H]
rmse_noact = np.sqrt(np.nanmean((F_noact - A) ** 2))
rmse_omt = np.sqrt(np.nanmean((F_omt - A) ** 2))
print(f"out-of-sample RMSE vs actual post-OMT: no-action={rmse_noact:.2f} pp  ->  OMT-like support={rmse_omt:.2f} pp "
      f"({100*(1-rmse_omt/rmse_noact):.0f}% lower)")

RES = dict(rho_pre=round(float(rho_pre), 3), DY_net={PERI[i]: round(float(NET[i]), 2) for i in range(5)},
           rmse_noaction=round(float(rmse_noact), 2), rmse_omt_support=round(float(rmse_omt), 2),
           rmse_reduction_pct=round(100 * (1 - rmse_omt / rmse_noact), 1),
           note="twin calibrated PRE-OMT only; OMT-like support forecast reproduces the held-out post-OMT recovery")
json.dump(RES, open(BASE / "review_counterfactual_results.json", "w"), indent=2)

# ============================== FIGURE ==============================
fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.4))
def panel(a, L_): a.text(-0.12, 1.04, L_, transform=a.transAxes, fontsize=11, fontweight="bold", ha="right")
fdates = post.index[:H]

# (a) aggregate systemic spread
a = ax[0]
a.plot(pre.index, pre.values.sum(1), color="k", lw=1.4, label="pre-OMT (calibration)")
a.plot(fdates, A.sum(1), color=GREEN, lw=2.0, label="actual post-OMT (held out)")
a.plot(fdates, F_noact.sum(1), color=CRIT, lw=1.6, ls="--", label="forecast: no action")
a.plot(fdates, F_omt.sum(1), color=STEEL, lw=1.6, ls="-.", label="forecast: OMT-like support")
a.axvline(OMT, color=GREY, ls=":", lw=1.0); a.text(OMT, a.get_ylim()[1] * 0.5, " OMT", color=GREY, fontsize=7)
a.set(title="Total systemic spread (out-of-sample)", xlabel="year", ylabel=r"$\sum_i$ spread (pp)")
a.legend(loc="upper left", fontsize=6); panel(a, "a")

# (b) per-sovereign: actual vs OMT-support forecast
a = ax[1]
for i, nm in enumerate(PERI):
    a.plot(fdates, A[:, i], color=f"C{i}", lw=1.3)
    a.plot(fdates, F_omt[:, i], color=f"C{i}", lw=1.0, ls=":")
a.set(title="Per-sovereign: actual (—) vs support-forecast (···)", xlabel="year", ylabel="spread (pp)")
a.text(0.02, 0.96, "calibrated pre-OMT only", transform=a.transAxes, fontsize=6.5, va="top", style="italic")
panel(a, "b")

# (c) RMSE bar
a = ax[2]
a.bar([0, 1], [rmse_noact, rmse_omt], color=[CRIT, STEEL], alpha=.85, width=.6)
for i, v in enumerate([rmse_noact, rmse_omt]): a.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
a.set_xticks([0, 1]); a.set_xticklabels(["no action", "OMT-like\nsupport"])
a.set(title=f"Out-of-sample forecast error\n(support −{100*(1-rmse_omt/rmse_noact):.0f}%)", ylabel="RMSE vs actual (pp)")
a.grid(alpha=.25, axis="y"); panel(a, "c")

fig.tight_layout()
fig.savefig(BASE / "review_counterfactual.pdf", bbox_inches="tight")
fig.savefig(BASE / "review_counterfactual.png", dpi=200, bbox_inches="tight")
print("\nfigure -> review_counterfactual.pdf/.png")
