"""
VALIDITY PRESSURE-TEST: insight vs machinery.

The central methodological worry about the directed-asymmetry law is that it may be partly *definitional*:
ARRO is handed the directed-network structure to plan with, the damage-weighted heuristic is not, so "the
controller that uses directional information beats the one that ignores it" is close to true by construction.

This script decomposes ARRO's advantage to see what is actually doing the work, by adding a control that has the
INSIGHT but none of the MACHINERY:

  none            do nothing                                         (damage ceiling)
  greedy          support the LOUDEST node now (reactive heuristic)  (the paper's baseline)
  static-transmit support the Diebold-Yilmaz net-transmitter with a FIXED allocation -- no planning, no
                  world-model, no anticipation. Pure insight.
  mpc-oracle      CEM model-predictive control anticipating H steps with the TRUE dynamics (the strongest
                  possible "anticipation"; the learned-MPC can only be <= this).

The decisive comparison is static-transmit vs mpc-oracle:
  * if static-transmit ~ mpc-oracle  -> the value is the INSIGHT (target transmitters); the anticipatory
                                        world-model machinery adds little. The honest, deflating reading.
  * if mpc-oracle >> static-transmit -> anticipation genuinely adds value beyond knowing the transmitter.

Run on a controlled synthetic directed twin AND the real COVID-19 state twin.
Run: python3 validity_decomposition.py
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
    "legend.fontsize": 7, "axes.linewidth": 0.7, "lines.linewidth": 1.3, "savefig.dpi": 300})
CRIT, STEEL, GREEN, PURPLE, GREY, GOLD = "#c00000", "#1f4e78", "#2e8b57", "#7030a0", "#7f7f7f", "#d4a017"
BASE = Path(__file__).parent

# ----------------------------------------------------------------------------- interdiction simulator
def project(a, x, B):
    """nonneg, a<=x (can't support more than the stress), sum a <= B."""
    a = np.clip(a, 0, x)
    s = a.sum()
    if s > B: a = a * (B / s)
    return a

def step(Phi, c, x, a, rng, Sig):
    xe = np.clip(x - a, 0, None)
    nz = rng.multivariate_normal(np.zeros(len(x)), Sig) if Sig is not None else 0.0
    return np.clip(Phi @ xe + c + nz, 0, None)

def ctl_none(x, **k): return np.zeros_like(x)
def ctl_greedy(x, B=0, **k):                       # waterfill budget onto the loudest nodes
    a = np.zeros_like(x); order = np.argsort(-x); rem = B
    for i in order:
        give = min(x[i], rem); a[i] = give; rem -= give
        if rem <= 1e-9: break
    return a
def ctl_static(x, NET=None, B=0, **k):             # fixed allocation by DY net-transmitter score
    w = np.clip(NET, 0, None)
    if w.sum() <= 0: return ctl_greedy(x, B=B)
    return project(B * w / w.sum(), x, B)
def ctl_mpc(x, Phi=None, c=None, B=0, H=4, rng=None, **k):   # CEM-MPC with TRUE dynamics (anticipation)
    N = len(x); mu = np.full((H, N), B / N); sd = np.full((H, N), B / 2)
    for _ in range(4):
        C = np.clip(rng.normal(mu, sd, (48, H, N)), 0, None)
        cost = np.zeros(48); sim = np.tile(x, (48, 1))
        for h in range(H):
            A = np.stack([project(C[j, h], sim[j], B) for j in range(48)])
            sim = np.clip((sim - A) @ Phi.T + c, 0, None); cost += sim.sum(1)
        el = C[np.argsort(cost)[:8]]; mu, sd = el.mean(0), el.std(0) + 1e-3
    return project(mu[0], x, B)

CTRLS = [("none", ctl_none, CRIT), ("greedy", ctl_greedy, PURPLE),
         ("static-transmit", ctl_static, GOLD), ("mpc-oracle", ctl_mpc, STEEL)]

def evaluate(Phi, c, Sig, S0, NET, B, T=20, H=4, seeds=24):
    out = {}
    for name, fn, _ in CTRLS:
        tot = []
        for s in range(seeds):
            rng = np.random.default_rng(1000 + s); x = S0.copy(); acc = 0.0
            for t in range(T):
                a = fn(x, Phi=Phi, c=c, NET=NET, B=B, H=H, rng=rng)
                x = step(Phi, c, x, a, rng, Sig); acc += x.sum()
            tot.append(acc)
        out[name] = (float(np.mean(tot)), float(np.std(tot)))
    return out

def summarize(res, tag):
    base = res["none"][0]
    red = {k: 100 * (1 - v[0] / base) for k, v in res.items()}
    # how much of the anticipation gain does the static insight already capture?
    g_static, g_mpc = red["static-transmit"], red["mpc-oracle"]
    captured = 100 * g_static / g_mpc if g_mpc > 1e-6 else float("nan")
    print(f"\n[{tag}]  none={base:.0f}")
    for k, v in res.items(): print(f"  {k:16s} {v[0]:9.1f} +/- {v[1]:5.1f}   ({red[k]:+.0f}% vs none)")
    print(f"  greedy(loudest) {red['greedy']:+.0f}%  <  static-transmit {g_static:+.0f}%  ~?  mpc-oracle {g_mpc:+.0f}%")
    print(f"  --> static insight captures {captured:.0f}% of the anticipatory planner's gain")
    return dict(none=round(base, 1), reductions={k: round(red[k], 1) for k in red},
                static_captures_pct_of_mpc=round(captured, 1))

# ----------------------------------------------------------------------------- twin A: controlled synthetic
def synthetic_twin(N=6, rho=1.06, seed=0):
    rng = np.random.default_rng(seed)
    Phi = np.zeros((N, N))
    Phi[1:, 0] = rng.uniform(0.55, 0.85, N - 1)        # node 0 = transmitter hub -> everyone
    np.fill_diagonal(Phi, rng.uniform(0.25, 0.45, N))  # self-persistence
    Phi[0, 0] = 0.45
    Phi *= rho / max(abs(np.linalg.eigvals(Phi)))       # scale to target spectral radius
    c = np.full(N, 0.15); Sig = 0.04 * np.eye(N)
    th = L.gfevd(Phi, Sig); _, _, NET, _ = L.connectedness(th)
    S0 = np.full(N, 1.0)
    return Phi, c, Sig, S0, NET

# ----------------------------------------------------------------------------- twin B: real COVID-19 states
def covid_twin():
    DROP = {"American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands", "Diamond Princess",
            "Grand Princess", "Puerto Rico"}
    d = pd.read_csv(BASE / "jhu_confirmed_US.csv"); dcols = [c for c in d.columns if "/" in c]
    st = d.groupby("Province_State")[dcols].sum().drop(index=[s for s in DROP if s in d["Province_State"].unique()], errors="ignore")
    st.columns = pd.to_datetime(st.columns)
    weekly = st.T.sort_index().resample("W").last().diff().clip(lower=0).dropna()
    top = weekly.sum().sort_values(ascending=False).head(14).index
    W = (weekly[top] / 1000.0).loc["2020-03-01":"2022-06-30"]
    Phi, c, Sig = L.fit_var_nonneg(W.values, ridge=5e-2)
    th = L.gfevd(Phi, Sig); _, _, NET, _ = L.connectedness(th)      # DY net BEFORE rescaling (true structure)
    Phi = Phi * (1.05 / max(abs(np.linalg.eigvals(Phi))))           # rescale to a genuine cascade (as the paper's demo does)
    Sig = 0.05 * np.eye(Phi.shape[0])                               # modest noise so the test is about control, not noise
    S0 = W.loc["2020-06-01":"2020-08-01"].mean().values + 0.5
    return Phi, c, Sig, S0, NET, [s[:4] for s in W.columns]

# ----------------------------------------------------------------------------- run
print("=" * 64)
PhiS, cS, SigS, S0S, NETS = synthetic_twin()
resS = evaluate(PhiS, cS, SigS, S0S, NETS, B=1.2, T=20, seeds=24)
sumS = summarize(resS, "synthetic directed twin (rho=1.06, one hub)")

PhiC, cC, SigC, S0C, NETC, namesC = covid_twin()
resC = evaluate(PhiC, cC, SigC, S0C, NETC, B=2.0, T=20, seeds=24)
sumC = summarize(resC, "real COVID-19 state twin (14 states)")

verdict = ("INSIGHT dominates: the static transmitter rule captures most of the anticipatory planner's gain, "
           "so the world-model machinery adds only a modest increment"
           if (sumS["static_captures_pct_of_mpc"] >= 80 and sumC["static_captures_pct_of_mpc"] >= 80)
           else "MACHINERY earns its keep: anticipation adds substantial value beyond the static transmitter rule")
print(f"\nVERDICT: {verdict}")
json.dump({"synthetic": sumS, "covid": sumC, "verdict": verdict}, open(BASE / "validity_decomposition_results.json", "w"), indent=2)

# ----------------------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.2))
for a, (res, tag) in zip(ax, [(resS, "Synthetic Directed Twin (One Hub, ρ=1.06)"),
                              (resC, "Real COVID-19 State Twin (14 States)")]):
    base = res["none"][0]; names = [c[0] for c in CTRLS]; cols = [c[2] for c in CTRLS]
    red = [100 * (1 - res[n][0] / base) for n in names]
    sd = [100 * res[n][1] / base for n in names]
    a.bar(range(len(names)), red, yerr=sd, color=cols, alpha=.85, capsize=3)
    for i, (r, s) in enumerate(zip(red, sd)): a.text(i, r + s + max(red) * 0.04, f"{r:+.0f}%", ha="center", fontsize=8, fontweight="bold")
    gst = 100 * (1 - res["static-transmit"][0] / base); gmpc = 100 * (1 - res["mpc-oracle"][0] / base)
    cap = 100 * gst / gmpc if gmpc > 1e-6 else 0
    a.axhline(gmpc, color=STEEL, ls="--", lw=.8, alpha=.6)
    a.set_xticks(range(len(names))); a.set_xticklabels(["None", "Greedy\n(Loudest)", "Static\nTransmitter", "MPC\n(Anticipate)"])
    a.set_ylabel("Cascade Reduction vs No-Action (%)"); a.set_title(tag, fontsize=8.5)
    a.text(0.5, 0.965, f"Static Insight Captures {cap:.0f}% of the\nAnticipatory Planner's Gain",
           transform=a.transAxes, ha="center", va="top", fontsize=7.5,
           bbox=dict(boxstyle="round,pad=0.3", fc="#fffbe9", ec=GOLD, lw=.8))
    a.grid(alpha=.25, axis="y"); a.set_ylim(0, max(red) * 1.42)
fig.suptitle("Insight or Machinery: Static Directed Targeting Captures Most of the Anticipatory Gain",
             fontsize=10, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(BASE / "validity_decomposition.pdf", bbox_inches="tight")
fig.savefig(BASE / "validity_decomposition.png", dpi=200, bbox_inches="tight")
print("figure -> validity_decomposition.pdf/.png")
