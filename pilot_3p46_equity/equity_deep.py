"""
Tier III deep dive — GLOBAL EQUITY-MARKET CONTAGION (Domain 3.46, liquidity flash-crash / market
contagion), the 2008 crash, CAPSTONE PARITY with the sovereign-debt pilot. This is the canonical
Diebold-Yilmaz setting: cross-market return/decline spillovers, whose total connectedness spikes when the
separating boundaries between national markets dissolve (kappa_B -> 1, chi -> 0). Eight national indices
(Yahoo Finance, daily, 2007-2010).

  (1) criticality crossing — rolling TOTAL Diebold-Yilmaz connectedness kappa_B(t) spikes through the 2008
      crash (markets move as one) and recedes after the Oct-2008 coordinated central-bank response;
  (2) directed network — static DY connectedness: which national market TRANSMITS the crash (expect the US);
  (3) ARRO interdiction — a controllable rescaled-explosive decline-stress twin; the learned world-model
      targets the transmitter market and is compared to the myopic 'support the hardest-hit market'.

Run: python3 equity_deep.py
"""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "pilot_cross_tier"))
import lsa_capstone as L
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.0, "axes.linewidth": 0.7, "lines.linewidth": 1.3, "savefig.dpi": 300})
CRIT, STEEL, GREEN, PURPLE, GREY = "#c00000", "#1f4e78", "#2e8b57", "#7030a0", "#7f7f7f"
BASE = Path(__file__).parent

idx = {"GSPC": "US", "FTSE": "UK", "GDAXI": "Germany", "FCHI": "France",
       "N225": "Japan", "HSI": "Hong Kong", "AEX": "Neth.", "BVSP": "Brazil"}
def load(sym):
    d = json.load(open(BASE / "data" / f"{sym}.json"))["chart"]["result"][0]
    s = pd.Series(d["indicators"]["quote"][0]["close"], index=pd.to_datetime(d["timestamp"], unit="s")).dropna()
    return s
P = pd.concat([load(s).rename(idx[s]) for s in idx], axis=1).ffill().dropna()
P = P.resample("W").last().loc["2007-01-01":"2010-06-30"]
names = list(P.columns); N = len(names)
ret = 100 * np.log(P).diff().dropna()
stress = (-ret).clip(lower=0)                          # weekly % decline (0 if up) = contagion stress
INT = pd.Timestamp("2008-10-08")                       # coordinated central-bank cuts / TARP

# ---------------- (1) criticality: rolling TOTAL DY connectedness ----------------
WIN = 26
def var_fit(M, lam=3e-2):
    X = np.column_stack([np.ones(len(M) - 1), M[:-1]]); Y = M[1:]
    Pn = np.eye(X.shape[1]); Pn[0, 0] = 0
    B = np.linalg.solve(X.T @ X + lam * len(M) * Pn, X.T @ Y)
    E = Y - X @ B
    return B[1:].T, np.cov(E.T) + 1e-6 * np.eye(M.shape[1])
def roll_totalconn(M, win):
    out = []
    for e in range(win, len(M) + 1):
        Phi, Sig = var_fit(M[e - win:e]); _, _, _, tot = L.connectedness(L.gfevd(Phi, Sig)); out.append(tot)
    return np.array(out)
tc = roll_totalconn(stress.values, WIN); tcd = stress.index[WIN - 1:]
pre = tc[tcd <= INT]; post = tc[tcd > "2009-07-01"]
print(f"(1) criticality: total connectedness {tc.min():.0f}% -> peak {tc.max():.0f}% (crisis) -> {post.mean():.0f}% (post)")

# ---------------- (2) directed network on decline-stress ----------------
Phi_s, c_s, Sig_s = L.fit_var_nonneg(stress.values, ridge=2e-2)
th = L.gfevd(Phi_s, Sig_s); TO, FROM, NET, tot = L.connectedness(th)
rank = [names[i] for i in np.argsort(-NET)]
print(f"(2) DY connectedness (decline stress) total={tot:.0f}%")
for i in np.argsort(-NET):
    print(f"     {names[i]:10s} TO={TO[i]:5.1f} FROM={FROM[i]:5.1f} NET={NET[i]:+6.1f}  {'transmitter' if NET[i]>0 else 'receiver'}")
print(f"     transmitter->receiver: {rank}")

# ---------------- (3) ARRO interdiction ----------------
S0 = stress.loc["2008-09-01":"2008-11-15"].mean().values + 0.5
print("(3) ARRO interdiction on rescaled decline-stress twin (16 seeds):")
ID = L.run_interdiction(Phi_s, c_s, Sig_s, S0, names, target_rho=1.05, budget=2.0, T_ep=24, seeds=16, H=4, train_eps=400, steps=3000)
base = ID["summary"]["none"]["mean"]
for m in ID["order"]:
    s = ID["summary"][m]; print(f"     {m:11s} {s['mean']:9.1f} +/- {s['sd']:5.1f}  ({100*(1-s['mean']/base):+.0f}%)")
print(f"     corr(DY-net, support): learned={ID['corr_dy_alloc']['learned-MPC']:+.2f}")

dd = (P / P.cummax() - 1) * 100
RES = dict(markets=names, conn_min=round(float(tc.min()), 1), conn_peak=round(float(tc.max()), 1),
           conn_post=round(float(post.mean()), 1), max_drawdown=round(float(dd.min().min()), 1),
           DY_total=round(float(tot), 1), DY_net={names[i]: round(float(NET[i]), 1) for i in range(N)},
           transmitter_rank=rank,
           interdiction={m: {k: round(v, 1) for k, v in ID["summary"][m].items()} for m in ID["order"]},
           corr_dy_alloc={k: round(v, 2) for k, v in ID["corr_dy_alloc"].items()})
json.dump(RES, open(BASE / "equity_deep_results.json", "w"), indent=2)

# ============================== FIGURE (2x3 Nature) ==============================
order = ID["order"]; colors = {"none": CRIT, "random": GREY, "greedy": PURPLE, "learned-MPC": STEEL, "oracle-MPC": GREEN}
fig, ax = plt.subplots(2, 3, figsize=(7.3, 5.0))
def panel(a, L_, x=-0.13, y=1.02): a.text(x, y, L_, transform=a.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")

# (a) index levels (rebased)
a = ax[0, 0]
R = P / P.iloc[0] * 100
for nm in names:
    a.plot(R.index, R[nm], lw=0.9, label=nm)
a.axvline(INT, color=GREEN, ls=":", lw=1.2)
a.set(title="National equity indices (2007 = 100)", xlabel="year", ylabel="index")
a.legend(loc="lower left", ncol=2, fontsize=5.3); panel(a, "a")

# (b) criticality: rolling total connectedness
a = ax[0, 1]
a.fill_between(tcd, tc.min(), tc, color=CRIT, alpha=.12)
a.plot(tcd, tc, color=STEEL, lw=1.6)
a.axvline(INT, color=GREEN, ls=":", lw=1.2); a.text(INT, tc.min()+2, " Oct 2008\n cuts/TARP", color=GREEN, fontsize=6, ha="right")
a.set(title="Criticality: total DY connectedness", xlabel="year", ylabel=r"$\kappa_\mathcal{B}$: total connectedness (%)")
panel(a, "b")

# (c) drawdowns
a = ax[0, 2]
for nm in names:
    a.plot(dd.index, dd[nm], lw=0.8)
a.axvline(INT, color=GREEN, ls=":", lw=1.2)
a.set(title="Drawdowns (the crash)", xlabel="year", ylabel="drawdown from peak (%)")
panel(a, "c")

# (d) directed network
a = ax[1, 0]
oi = np.argsort(NET); cols = [CRIT if NET[i] > 0 else STEEL for i in oi]
a.barh([names[i] for i in oi], NET[oi], color=cols, alpha=.85); a.axvline(0, color="k", lw=.8)
a.set(title="Net directional connectedness\n(Diebold–Yılmaz; >0 = transmitter)", xlabel="NET = TO − FROM (%)")
a.grid(alpha=.25, axis="x"); panel(a, "d")

# (e) interdiction
a = ax[1, 1]
means = [ID["summary"][m]["mean"] for m in order]; sds = [ID["summary"][m]["sd"] for m in order]
a.bar(range(len(order)), means, yerr=sds, color=[colors[m] for m in order], alpha=.85, capsize=2.5)
for i, m in enumerate(order):
    a.text(i, means[i] + sds[i] + base * .02, f"{100*(1-means[i]/base):+.0f}%", ha="center", fontsize=6)
a.set_xticks(range(len(order))); a.set_xticklabels(order, rotation=25, ha="right")
a.set(title="ARRO interdiction (16 seeds)", ylabel="cumulative stress (lower=better)")
a.grid(alpha=.25, axis="y"); panel(a, "e")

# (f) allocation vs DY-net
a = ax[1, 2]
x = np.arange(N); w = 0.38
al_j = np.array(ID["alloc"]["learned-MPC"]); al_g = np.array(ID["alloc"]["greedy"])
a.bar(x - w/2, al_g, w, label="greedy (loudest)", color=PURPLE, alpha=.8)
a.bar(x + w/2, al_j, w, label="ARRO (learned-MPC)", color=STEEL, alpha=.85)
a.set_xticks(x); a.set_xticklabels(names, rotation=45, ha="right", fontsize=5.5)
a.set(title=f"Support allocation (corr ARRO,DY = {ID['corr_dy_alloc']['learned-MPC']:+.2f})", ylabel=r"mean support $a_i$")
a.legend(loc="upper right"); a.grid(alpha=.25, axis="y")
at = a.twinx(); at.plot(x, NET, "kD--", ms=3, lw=0.9); at.set_ylabel("DY net (%)", fontsize=6.5)
panel(a, "f")

fig.tight_layout()
fig.savefig(BASE / "equity_deep.pdf", bbox_inches="tight")
fig.savefig(BASE / "equity_deep.png", dpi=200, bbox_inches="tight")
print("\nfigure -> equity_deep.pdf/.png ; results -> equity_deep_results.json")
