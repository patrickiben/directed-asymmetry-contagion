"""
REVIEW RESPONSE (Major concern 2: the linear test bed makes the learned world-model trivial). Here the twin
is NONLINEAR -- a gated tipping node (an LSA index-1 saddle): the transmitter hub's self-excitation is dormant
below a threshold and IGNITES super-linearly above it. A linear VAR cannot represent the gate, so a
linear-model controller mis-anticipates the cascade. We show the NONLINEAR learned world-model (a small MLP,
the JEPA class) captures the gate and yields markedly better interdiction than a linear oracle, approaching
the true-model oracle.

Controllers (CEM receding-horizon MPC unless noted): none; greedy (support the loudest); linear-oracle-MPC
(plans with a VAR fit to exploration -- misspecified); learned-MPC (nonlinear MLP world-model, trained on the
SAME exploration); true-oracle-MPC (plans with the real nonlinear dynamics, an upper bound). 24 seeds.
Run: python3 review_nonlinear.py
"""
import json
import numpy as np
import autograd.numpy as anp
from autograd import grad
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.linewidth": 0.7, "lines.linewidth": 1.3, "savefig.dpi": 300})
CRIT, STEEL, GREEN, PURPLE, GREY, GOLD = "#c00000", "#1f4e78", "#2e8b57", "#7030a0", "#7f7f7f", "#b8862b"
BASE = Path(__file__).parent
np.seterr(over="ignore", invalid="ignore")

# ---------------- nonlinear LOUD-DECOY vs HIDDEN-GATE twin ----------------
# Node 0: a QUIET tipping node (low now) that drifts up and, past a gate, IGNITES and cascades to nodes 2..5.
# Node 1: a LOUD but BENIGN decoy (high stress, persistent, no outgoing cascade).
# A linear model sees node 1 as the costly one and node 0 as nearly decoupled (its coupling is purely
# nonlinear, gate-mediated) -> it spends the budget on the decoy and lets node 0 ignite: the WRONG policy.
N = 6; CAP = 80.0; T_EP = 22; BUDGET = 0.9
THETA, GAMMA, WID, DRIFT, BETA, PHI00, DECOY = 6.0, 2.0, 0.8, 0.65, 1.2, 0.85, 0.90
c = np.array([0.3, 1.2, 0.6, 0.6, 0.6, 0.6]); SELF = np.array([0, 0, 0.4, 0.4, 0.4, 0.4]); NOISE = 0.25
S0 = np.array([2.5, 12.0, 2.0, 2.0, 2.0, 2.0])          # node 0 quiet (will ignite), node 1 LOUD benign decoy

def _ig(x0): return GAMMA / (1 + np.exp(-(x0 - THETA) / WID)) * x0
def env_step(s, a, rng):
    se = (1 - a) * s; ig = _ig(se[0]); s2 = np.empty(N)
    s2[0] = c[0] + PHI00 * se[0] + DRIFT + ig
    s2[1] = c[1] + DECOY * se[1]
    s2[2:] = c[2:] + SELF[2:] * se[2:] + BETA * ig
    return np.clip(s2 + rng.normal(0, NOISE, N), 0, CAP)
def env_det_batch(s, a):                                # deterministic nonlinear, (pop,N)
    se = (1 - a) * s; ig = _ig(se[:, 0]); s2 = c + SELF * se
    s2[:, 0] = c[0] + PHI00 * se[:, 0] + DRIFT + ig
    s2[:, 1] = c[1] + DECOY * se[:, 1]
    s2[:, 2:] = s2[:, 2:] + BETA * ig[:, None]
    return np.clip(s2, 0, CAP)

def softmax_budget(L):
    e = np.exp(L - L.max(-1, keepdims=True)); return np.minimum(BUDGET * e / e.sum(-1, keepdims=True), 1.0)
def cem(cost, H, rng, pop=64, elite=10, iters=4):
    mean = np.zeros((H, N)); std = np.ones((H, N))
    for _ in range(iters):
        Lz = rng.normal(mean, std, size=(pop, H, N)); A = softmax_budget(Lz)
        idx = np.argsort(cost(A))[:elite]; mean = Lz[idx].mean(0); std = Lz[idx].std(0) + 1e-2
    return softmax_budget(mean[0])

# ---- exploration data (random policy) ----
def explore(n_ep, seed):
    Sx, Ax, Sy = [], [], []
    for e in range(n_ep):
        rng = np.random.default_rng(seed + e); s = S0.copy()
        for t in range(T_EP):
            a = softmax_budget(rng.normal(0, 1, N)); s2 = env_step(s, a, rng)
            Sx.append(s); Ax.append(a); Sy.append(s2); s = s2
    return np.array(Sx), np.array(Ax), np.array(Sy)
Sx, Ax, Sy = explore(220, 1000)
SE = (1 - Ax) * Sx                                       # effective (post-support) state

# ---- linear oracle: VAR  s' = c_hat + Phi_hat @ se ----
Xl = np.column_stack([np.ones(len(SE)), SE]); Bl = np.linalg.solve(Xl.T @ Xl + 1e-3 * np.eye(N + 1), Xl.T @ Sy)
c_lin, Phi_lin = Bl[0], Bl[1:].T
def lin_batch(s, a):
    se = (1 - a) * s; return np.clip(c_lin + se @ Phi_lin.T, 0, CAP)

# ---- learned nonlinear world-model: MLP  se -> s' ----
H1 = 32
def init(seed):
    r = np.random.default_rng(seed)
    return [r.normal(0, .3, (N, H1)), np.zeros(H1), r.normal(0, .3, (H1, H1)), np.zeros(H1), r.normal(0, .3, (H1, N)), np.zeros(N)]
def mlp(P, se):
    h = anp.tanh(anp.dot(se, P[0]) + P[1]); h = anp.tanh(anp.dot(h, P[2]) + P[3]); return anp.dot(h, P[4]) + P[5]
def train_mlp(seed=0, steps=1500):
    P = init(seed); m = [np.zeros_like(p) for p in P]; v = [np.zeros_like(p) for p in P]
    SEn = anp.array(SE); Syn = anp.array(Sy); Mn = len(SE); r = np.random.default_rng(seed)
    def loss(P, idx): return anp.mean((mlp(P, SEn[idx]) - Syn[idx]) ** 2)
    gl = grad(loss)
    for t in range(1, steps + 1):
        idx = r.integers(0, Mn, 256); g = gl(P, idx)
        for k in range(len(P)):
            m[k] = .9 * m[k] + .1 * g[k]; v[k] = .999 * v[k] + .001 * g[k] ** 2
            P[k] = P[k] - 3e-3 * (m[k] / (1 - .9 ** t)) / (np.sqrt(v[k] / (1 - .999 ** t)) + 1e-8)
    return P
Pmlp = train_mlp()
def mlp_batch(s, a):
    se = (1 - a) * s; return np.clip(np.asarray(mlp(Pmlp, se)), 0, CAP)

# prediction accuracy near the tipping point (se0 > THETA)
tip = SE[:, 0] > THETA
err_lin = np.sqrt(np.mean((np.clip(c_lin + SE @ Phi_lin.T, 0, CAP)[tip] - Sy[tip]) ** 2))
err_mlp = np.sqrt(np.mean((np.asarray(mlp(Pmlp, SE))[tip] - Sy[tip]) ** 2))
print(f"1-step prediction RMSE near tipping: linear VAR={err_lin:.2f}  vs  learned MLP={err_mlp:.2f}")

# ---- controllers + rollout cost ----
def cost_model(model, s0, H):
    def cost(A):
        s = np.tile(s0, (A.shape[0], 1)); tot = np.zeros(A.shape[0])
        for k in range(H):
            tot += s.sum(1); s = model(s, A[:, k, :])
        return tot
    return cost
H = 4
# focused world-model comparison: none vs linear-world-model vs learned-world-model vs true-model oracle
ctrls = {
    "none": lambda s, rng: np.zeros(N),
    "linear-MPC": lambda s, rng: cem(cost_model(lin_batch, s, H), H, rng),
    "learned-MPC": lambda s, rng: cem(cost_model(mlp_batch, s, H), H, rng),
    "true-oracle": lambda s, rng: cem(cost_model(env_det_batch, s, H), H, rng),
}
def episode(ctrl, seed):
    rng = np.random.default_rng(seed); s = S0.copy(); tot = 0.0
    for t in range(T_EP):
        a = ctrl(s, rng); tot += s.sum(); s = env_step(s, a, rng)
    return tot
SEEDS = range(24)
res = {m: np.array([episode(ctrls[m], 5000 + i) for i in SEEDS]) for m in ctrls}
base = res["none"].mean()
for m in ctrls:
    print(f"  {m:12s} {res[m].mean():8.1f} +/- {res[m].std():5.1f}  ({100*(1-res[m].mean()/base):+.0f}%)")
gain = 100 * (res["linear-MPC"].mean() - res["learned-MPC"].mean()) / res["linear-MPC"].mean()
print(f"  -> learned-MPC beats linear-MPC by {gain:.0f}% (closes {100*(res['linear-MPC'].mean()-res['learned-MPC'].mean())/(res['linear-MPC'].mean()-res['true-oracle'].mean()):.0f}% of the gap to the true oracle)")

RES = dict(pred_rmse_tipping=dict(linear=round(float(err_lin), 2), learned=round(float(err_mlp), 2)),
           interdiction={m: dict(mean=round(float(res[m].mean()), 1), sd=round(float(res[m].std()), 1),
                                 pct=round(100 * (1 - res[m].mean() / base), 1)) for m in ctrls},
           learned_beats_linear_pct=round(float(gain), 1))
json.dump(RES, open(BASE / "review_nonlinear_results.json", "w"), indent=2)

# ============================== FIGURE ==============================
fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.5))
def panel(a, L_): a.text(-0.13, 1.04, L_, transform=a.transAxes, fontsize=11, fontweight="bold", ha="right")

# (a) the gate: true vs linear-model next-hub-value
a = ax[0]
xx = np.linspace(0, 14, 100)
true_hub = c[0] + PHI00 * xx + DRIFT + GAMMA / (1 + np.exp(-(xx - THETA) / WID)) * xx
lin_hub = c_lin[0] + Phi_lin[0, 0] * xx
mlp_hub = np.array([np.asarray(mlp(Pmlp, np.r_[x, np.full(N - 1, 1.0)]))[0] for x in xx])
a.plot(xx, true_hub, color="k", lw=1.8, label="true (nonlinear gate)")
a.plot(xx, lin_hub, color=CRIT, lw=1.4, ls="--", label="linear VAR (misspecified)")
a.plot(xx, mlp_hub, color=STEEL, lw=1.4, ls="-.", label="learned MLP")
a.axvline(THETA, color=GOLD, ls=":", lw=1.0); a.text(THETA + .2, a.get_ylim()[1] * .2, " ignition\n threshold", color=GOLD, fontsize=6)
a.set(title="The hub-ignition gate the linear model misses", xlabel="hub state $s_0$", ylabel="next hub value")
a.legend(loc="upper left", fontsize=6); panel(a, "a")

# (b) interdiction outcomes (controllers only; no-action annotated, it is off-scale)
a = ax[1]
order = ["linear-MPC", "learned-MPC", "true-oracle"]; cols = {"linear-MPC": GOLD, "learned-MPC": STEEL, "true-oracle": GREEN}
a.bar(range(len(order)), [res[m].mean() for m in order], yerr=[res[m].std() for m in order],
      color=[cols[m] for m in order], alpha=.85, capsize=3)
for i, m in enumerate(order): a.text(i, res[m].mean() + res[m].std() + 8, f"{res[m].mean():.0f}", ha="center", fontsize=7, fontweight="bold")
a.set_xticks(range(len(order))); a.set_xticklabels(["linear\nworld-model", "learned\nworld-model", "true-model\noracle"], fontsize=7)
a.set(title=f"Interdiction (24 seeds): learned −{gain:.0f}% vs linear", ylabel="cumulative stress (lower=better)")
a.text(0.02, 0.96, f"no action: {base:.0f} (off scale)", transform=a.transAxes, fontsize=6.5, va="top", color=CRIT, style="italic")
a.grid(alpha=.25, axis="y"); panel(a, "b")

# (c) prediction error near tipping
a = ax[2]
a.bar([0, 1], [err_lin, err_mlp], color=[CRIT, STEEL], alpha=.85, width=.6)
for i, v in enumerate([err_lin, err_mlp]): a.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
a.set_xticks([0, 1]); a.set_xticklabels(["linear\nVAR", "learned\nMLP"])
a.set(title="1-step prediction RMSE near the tipping point", ylabel="RMSE")
a.grid(alpha=.25, axis="y"); panel(a, "c")

fig.tight_layout()
fig.savefig(BASE / "review_nonlinear.pdf", bbox_inches="tight")
fig.savefig(BASE / "review_nonlinear.png", dpi=200, bbox_inches="tight")
print("\nfigure -> review_nonlinear.pdf/.png")
