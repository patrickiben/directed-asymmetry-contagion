"""
Shared capstone-parity analysis for the cross-tier deep dives (housing Tier III, opioid Tier I).
Mirrors the sovereign-debt pipeline (pilot_3p49_sovereign): a non-negative VAR(1) directed-contagion
twin  ->  Diebold-Yilmaz generalized-FEVD connectedness (who transmits)  ->  rolling spectral-radius
criticality  ->  ARRO interdiction (CEM-MPC + a learned JEPA world-model, 5 controllers x seeds).

The domain drivers supply a multivariate STRESS series M (T x N) for calibration/network/criticality
and a crisis-onset state S0 for the controllable interdiction. Run with the pinned environment (see RUN.md).
"""
import numpy as np
import autograd.numpy as anp
from autograd import grad
try:
    from scipy.optimize import lsq_linear
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

np.seterr(over="ignore", invalid="ignore", divide="ignore")

# ============================== TWIN CALIBRATION ==============================
def ridge_var1(M, lam=2e-2):
    """Plain ridge VAR(1) -> Phi (fast; used for rolling spectral radius)."""
    X = np.column_stack([np.ones(len(M) - 1), M[:-1]]); Y = M[1:]
    P = np.eye(X.shape[1]); P[0, 0] = 0
    B = np.linalg.solve(X.T @ X + lam * len(M) * P, X.T @ Y)
    return B[1:].T

def fit_var_nonneg(M, ridge=5e-2):
    """VAR(1) with NON-NEGATIVE off-diagonal couplings (clean contagion kernel a_ij>=0);
    diagonal (own dynamics) and intercept free. Ridge-regularised. Returns Phi, c, Sigma."""
    T, N = M.shape
    X = np.column_stack([np.ones(T - 1), M[:-1]]); Y = M[1:]
    Preg = np.eye(N + 1); Preg[0, 0] = 0.0
    Aaug = np.vstack([X, np.sqrt(ridge * (T - 1)) * Preg])
    Phi = np.zeros((N, N)); c = np.zeros(N); R = np.zeros_like(Y)
    for i in range(N):
        baug = np.concatenate([Y[:, i], np.zeros(N + 1)])
        lb = np.full(N + 1, -np.inf); ub = np.full(N + 1, np.inf)
        for j in range(N):
            if j != i:
                lb[1 + j] = 0.0
        if HAVE_SCIPY:
            coef = lsq_linear(Aaug, baug, bounds=(lb, ub), max_iter=400, tol=1e-10).x
        else:
            coef = np.linalg.lstsq(Aaug, baug, rcond=None)[0]
            for j in range(N):
                if j != i and coef[1 + j] < 0:
                    coef[1 + j] = 0.0
        c[i] = coef[0]; Phi[i] = coef[1:]; R[:, i] = Y[:, i] - X @ coef
    Sigma = np.cov(R.T)
    return Phi, c, Sigma

def spectral_radius(Phi):
    return float(max(abs(np.linalg.eigvals(Phi))))

def rolling_rho(M, win, lam=2e-2):
    """Rolling spectral radius rho(t)=kappa_B(t) (aligned to window-end indices win-1 .. T-1)."""
    return np.array([spectral_radius(ridge_var1(M[e - win:e], lam)) for e in range(win, len(M) + 1)])

# ============================== DIEBOLD-YILMAZ ===============================
def gfevd(Phi, Sigma, H=10):
    """Generalized FEVD (Pesaran-Shin) for a VAR(1): theta_ij = share of i's H-step FE variance from j."""
    N = Phi.shape[0]; A = [np.eye(N)]
    for h in range(1, H):
        A.append(Phi @ A[-1])
    th = np.zeros((N, N))
    for i in range(N):
        den = sum((A[h] @ Sigma @ A[h].T)[i, i] for h in range(H))
        for j in range(N):
            th[i, j] = sum((A[h] @ Sigma)[i, j] ** 2 for h in range(H)) / Sigma[j, j] / den
    return th / th.sum(1, keepdims=True)

def connectedness(theta):
    """Returns TO%, FROM%, NET% (>0 = net transmitter), total connectedness index %."""
    d = np.diag(theta)
    TO = (theta.sum(0) - d) * 100
    FROM = (theta.sum(1) - d) * 100
    total = (theta.sum() - d.sum()) / theta.shape[0] * 100
    return TO, FROM, TO - FROM, total

# ============================== ARRO INTERDICTION ============================
def run_interdiction(Phi0, c, Sigma, S0, names, target_rho=1.05, budget=1.5, cap_mult=6.0,
                     T_ep=24, seeds=16, H=4, noise_scale=0.30, train_eps=200, steps=2000,
                     greedy_gain=3.0, verbose=True):
    """ARRO interdiction on a rescaled-explosive contagion twin (state = per-node stress).
    Action a_i in [0,1] under a budget = targeted SUPPORT (lowers node i's stress + its outgoing
    spillover). 5 controllers: none/random/greedy(loudest)/learned-MPC(CEM over a learned world-model)/oracle-MPC(true twin).
    Returns a results dict including per-method cumulative-stress mean/sd, allocations, DY_NET, traj."""
    N = len(names)
    r0 = spectral_radius(Phi0)
    Phi = Phi0 * (target_rho / r0) if r0 > 0 else Phi0.copy()
    CAP = float(np.max(S0)) * cap_mult
    NOISE = noise_scale * np.sqrt(np.clip(np.diag(Sigma), 0, None))
    th = gfevd(Phi, Sigma); d = np.diag(th)
    DY_NET = ((th.sum(0) - d) - (th.sum(1) - d)) * 100
    off = Phi.copy(); np.fill_diagonal(off, 0); OUT_SPILL = off.sum(0)
    S0 = np.asarray(S0, float)

    def env_step(s, a, rng):
        s_eff = (1 - a) * s
        return np.clip(c + Phi @ s_eff + rng.normal(0, NOISE), 0, CAP)

    def expected_batch(s, a):
        return np.clip(c + (Phi @ ((1 - a) * s).T).T, 0, CAP)

    def softmax_budget(logits):
        e = np.exp(logits - logits.max(-1, keepdims=True))
        return np.minimum(budget * e / e.sum(-1, keepdims=True), 1.0)

    def cem_plan(cost, rng, pop=48, elite=8, iters=4):
        mean = np.zeros((H, N)); std = np.ones((H, N))
        for _ in range(iters):
            L = rng.normal(mean, std, size=(pop, H, N)); A = softmax_budget(L)
            idx = np.argsort(cost(A))[:elite]; mean = L[idx].mean(0); std = L[idx].std(0) + 1e-2
        return softmax_budget(mean[0])

    def oracle_cost(s0):
        def cost(A):
            pop = A.shape[0]; s = np.tile(s0, (pop, 1)); tot = np.zeros(pop)
            for k in range(H):
                tot += s.sum(1); s = expected_batch(s, A[:, k, :])
            return tot
        return cost

    ctrl_none = lambda s, hist, rng: np.zeros(N)
    ctrl_random = lambda s, hist, rng: softmax_budget(rng.normal(0, 1, N))
    ctrl_greedy = lambda s, hist, rng: softmax_budget(s * greedy_gain)
    ctrl_oracle = lambda s, hist, rng: cem_plan(oracle_cost(s), rng)

    # ---- JEPA world-model (pragmatic, learned from random-policy exploration) ----
    W = 3; OBS = N * W; D = 10; Hh = 24
    shapes = dict(We=(OBS, D), be=(D,), Wp1=(D + N, Hh), bp1=(Hh,), Wp2=(Hh, D), bp2=(D,), Wr=(D, 1), br=(1,))
    sizes = {k: int(np.prod(v)) for k, v in shapes.items()}

    def unpack(v, lib=anp):
        o, out = 0, {}
        for k, sh in shapes.items():
            out[k] = lib.reshape(v[o:o + sizes[k]], sh); o += sizes[k]
        return out

    def enc(P, O): return anp.tanh(anp.dot(O, P["We"]) + P["be"])

    def predl(P, z, a):
        x = anp.concatenate([z, a], axis=-1)
        return anp.tanh(anp.dot(anp.tanh(anp.dot(x, P["Wp1"]) + P["bp1"]), P["Wp2"]) + P["bp2"])

    def collect(n_eps, seed0):
        O, A, Onx, C = [], [], [], []
        for e in range(n_eps):
            rng = np.random.default_rng(seed0 + e); s = S0.copy(); buf = [S0.copy()] * W
            for t in range(T_ep):
                a = softmax_budget(rng.normal(0, 1, N)); o = np.concatenate(buf[-W:])
                s = env_step(s, a, rng); buf.append(s.copy())
                O.append(o); A.append(a); Onx.append(np.concatenate(buf[-W:])); C.append(s.sum())
        return np.array(O), np.array(A), np.array(Onx), np.array(C, float)

    def train_jepa(data, seed=0):
        O, A, Onx, C = data
        Om, Os = O.mean(0), O.std(0) + 1e-6; Cm, Cs = C.mean(), C.std() + 1e-6
        On = anp.array(np.clip((O - Om) / Os, -8, 8)); An = anp.array(A)
        Onn = anp.array(np.clip((Onx - Om) / Os, -8, 8)); Cn = anp.array((C - Cm) / Cs)
        rng = np.random.default_rng(seed); Mn = On.shape[0]
        v = np.concatenate([rng.normal(0, 1 / np.sqrt(sh[0] if len(sh) == 2 else 1), sizes[k]) for k, sh in shapes.items()])
        def loss(v, vt, idx):
            P, Pt = unpack(v), unpack(vt)
            z = enc(P, On[idx]); zt = enc(Pt, Onn[idx])
            jepa = anp.mean(anp.sum((predl(P, z, An[idx]) - zt) ** 2, axis=1))
            rd = anp.mean((anp.dot(z, P["Wr"])[:, 0] + P["br"][0] - Cn[idx]) ** 2)
            std = anp.sqrt(anp.var(z, axis=0) + 1e-4)
            return jepa + rd + 0.05 * anp.mean(anp.maximum(0.0, 1.0 - std))
        gl = grad(loss); ma = np.zeros_like(v); va = np.zeros_like(v)
        for t in range(1, steps + 1):
            idx = rng.integers(0, Mn, size=256); g = np.asarray(gl(v, v.copy(), idx))
            ma = 0.9 * ma + 0.1 * g; va = 0.999 * va + 0.001 * g * g
            v = v - 3e-3 * (ma / (1 - 0.9 ** t)) / (np.sqrt(va / (1 - 0.999 ** t)) + 1e-8)
        return dict(P={k: np.asarray(val) for k, val in unpack(v, np).items()}, Om=Om, Os=Os, Cm=Cm, Cs=Cs)

    def jepa_ctrl(J):
        P = J["P"]
        def fn(s, hist, rng):
            obs = np.concatenate(hist[-W:]); z0 = np.tanh(np.clip((obs - J["Om"]) / J["Os"], -8, 8) @ P["We"] + P["be"])
            def cost(A):
                pop = A.shape[0]; z = np.tile(z0, (pop, 1)); tot = np.zeros(pop)
                for k in range(H):
                    tot += np.maximum((z @ P["Wr"])[:, 0] + P["br"][0], -3) * J["Cs"] + J["Cm"]
                    x = np.concatenate([z, A[:, k, :]], axis=1)
                    z = np.tanh(np.tanh(x @ P["Wp1"] + P["bp1"]) @ P["Wp2"] + P["bp2"])
                return tot
            return cem_plan(cost, rng)
        return fn

    def episode(ctrl, seed):
        rng = np.random.default_rng(seed); s = S0.copy(); buf = [S0.copy()] * W
        straj, atraj = [], []
        for t in range(T_ep):
            a = ctrl(s, buf, rng); straj.append(s.copy()); atraj.append(a.copy())
            s = env_step(s, a, rng); buf.append(s.copy())
        return dict(total=float(np.array(straj).sum()), straj=np.array(straj), atraj=np.array(atraj))

    SEEDS = list(range(seeds))
    methods = {"none": ctrl_none, "random": ctrl_random, "greedy": ctrl_greedy, "oracle-MPC": ctrl_oracle}
    res = {m: [episode(ct, s) for s in SEEDS] for m, ct in methods.items()}
    if verbose:
        for m in methods:
            tot = np.array([r["total"] for r in res[m]])
            print(f"  {m:11s}: cumulative stress {tot.mean():9.1f} +/- {tot.std():6.1f}")
    J = train_jepa(collect(train_eps, 30000))
    res["learned-MPC"] = [episode(jepa_ctrl(J), s) for s in SEEDS]
    if verbose:
        tot = np.array([r["total"] for r in res["learned-MPC"]])
        print(f"  {'learned-MPC':11s}: cumulative stress {tot.mean():9.1f} +/- {tot.std():6.1f}")

    order = ["none", "random", "greedy", "learned-MPC", "oracle-MPC"]
    alloc = {m: np.array([np.mean([r["atraj"][:, i].mean() for r in res[m]]) for i in range(N)]) for m in order}
    corr = {}
    for m in ["learned-MPC", "oracle-MPC"]:
        corr[m] = float(np.corrcoef(DY_NET, alloc[m])[0, 1])
    summary = {m: dict(mean=float(np.mean([r["total"] for r in res[m]])),
                       sd=float(np.std([r["total"] for r in res[m]]))) for m in order}
    return dict(names=names, rho_target=spectral_radius(Phi), DY_NET=DY_NET, OUT_SPILL=OUT_SPILL,
                summary=summary, alloc={m: alloc[m].tolist() for m in order}, corr_dy_alloc=corr,
                order=order, res=res)
