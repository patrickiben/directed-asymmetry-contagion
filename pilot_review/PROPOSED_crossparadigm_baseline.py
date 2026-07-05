"""
PROPOSED (discharges referee LSA-NOV-2 -- NOT wired into the manuscript).

The benchmark set was entirely within the VAR/GFEVD paradigm (directed estimators + undirected
centralities on the same fitted network + a planner on the same surrogate). This adds three
CROSS-PARADIGM strong baselines from the source-identification / network-control literatures the
Introduction invokes, and runs each as a COMPETING interdiction controller through the identical
cascade simulator used for transmitter-targeting:

  * effective-distance source centrality (Brockmann & Helbing 2013)   -- arrival-time source ranking
  * collective influence  (Morone & Makse 2015, morone2015)           -- optimal-percolation influencers
  * structural-controllability driver nodes (Liu, Slotine & Barabasi 2011, liu2011) -- max-matching drivers

For each, budget is allocated by the method's node score and the same rescaled-explosive cascade is
simulated; we report the cascade reduction vs no-action, alongside transmitter-targeting (DY-net) and
the reactive loudest-node baseline. The claim to test: transmitter-targeting matches or exceeds these
deployed cross-paradigm methods on the two anchor networks (2008 equities, held-out COVID-19), so the
advantage is not an artefact of a within-paradigm comparison.

Self-contained + OFFLINE (bundled equity CSV + jhu_confirmed_US.csv). Deterministic.
Run:  python3 PROPOSED_crossparadigm_baseline.py   -> PROPOSED_crossparadigm_baseline.json
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, maximum_bipartite_matching

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "pilot_cross_tier"))
import lsa_capstone as L

TARGET_RHO, T_EP, SEEDS = 1.05, 16, 24
np.seterr(all="ignore")

# ---------------------------------------------------------- cascade simulator (same as bootstrap_ci.py)
def project(a, x, B):
    a = np.clip(a, 0, x); s = a.sum(); return a * (B / s) if s > B else a
def stepf(Phi, c, x, a, rng):
    return np.clip(Phi @ np.clip(x - a, 0, None) + c + 0.05 * rng.standard_normal(len(x)), 0, None)
def greedy(x, B):
    a = np.zeros_like(x); rem = B
    for i in np.argsort(-x):
        g = min(x[i], rem); a[i] = g; rem -= g
        if rem <= 1e-9: break
    return a
def alloc_score(score, x, B):
    w = np.clip(score, 0, None); return project(B * w / w.sum(), x, B) if w.sum() > 0 else np.zeros_like(x)
def rescale(Phi, rho=TARGET_RHO):
    ev = max(abs(np.linalg.eigvals(Phi))); return Phi * (rho / ev) if ev > 1e-6 else Phi

def run_controller(Phi, c, S0, B, kind, score=None):
    tot = []
    for s in range(SEEDS):
        rng = np.random.default_rng(100 + s); x = S0.copy(); acc = 0.0
        for _t in range(T_EP):
            if kind == "none":     a = np.zeros_like(x)
            elif kind == "greedy": a = greedy(x, B)
            else:                  a = alloc_score(score, x, B)
            x = stepf(Phi, c, x, a, rng); acc += x.sum()
        tot.append(acc)
    return float(np.mean(tot))

# ---------------------------------------------------------- cross-paradigm node scores
def s_transmitter(Phi, Sig):
    _TO, _FROM, NET, _tot = L.connectedness(L.gfevd(Phi, Sig)); return np.clip(NET, 0, None)

def s_effective_distance(Phi):
    """Brockmann-Helbing: directed edge n->m has effective length 1 - log P(m|n), with
    P = column-normalized outflow; source centrality of i = -mean shortest effective distance
    from i to every other node (a better spreader reaches everyone at smaller effective distance)."""
    F = np.clip(Phi.copy(), 0, None); np.fill_diagonal(F, 0.0)
    col = F.sum(0); P = np.divide(F, col, out=np.zeros_like(F), where=col > 0)   # P[m,n] = flux to m from n
    N = Phi.shape[0]
    G = np.zeros((N, N))                                                         # 0 = no edge (sparse)
    for n in range(N):
        for m in range(N):
            if P[m, n] > 0:
                G[n, m] = 1.0 - np.log(P[m, n])                                  # length of n -> m
    D = shortest_path(csr_matrix(G), method="D", directed=True)
    D[np.isinf(D)] = np.nan
    reach = -np.nanmean(np.where(np.eye(N) == 1, np.nan, D), axis=1)             # higher = better source
    if not np.isfinite(reach).all():
        fill = np.nanmin(reach[np.isfinite(reach)]) if np.isfinite(reach).any() else 0.0
        reach[~np.isfinite(reach)] = fill
    return reach - reach.min()

def s_collective_influence(Phi, q=0.6):
    F = np.clip(Phi.copy(), 0, None); np.fill_diagonal(F, 0.0)
    thr = np.quantile(F[F > 0], q) if (F > 0).any() else 0.0
    A = (F > thr).astype(float)                       # A[i,j]=1: j influences i
    kout = A.sum(0)                                    # out-degree of j = #i it influences
    CI = np.zeros(len(kout))
    for j in range(len(kout)):
        nbrs = np.where(A[:, j] > 0)[0]                # direct out-neighbors of j
        CI[j] = max(kout[j] - 1, 0) * sum(max(kout[m] - 1, 0) for m in nbrs)
    return CI

def s_structural_controllability(Phi, q=0.6):
    F = np.clip(Phi.copy(), 0, None); np.fill_diagonal(F, 0.0)
    thr = np.quantile(F[F > 0], q) if (F > 0).any() else 0.0
    N = Phi.shape[0]
    B = np.zeros((N, N))                               # B[j,i]=1 if edge j->i
    for i in range(N):
        for j in range(N):
            if i != j and F[i, j] > thr: B[j, i] = 1.0
    try:
        match = maximum_bipartite_matching(csr_matrix(B), perm_type="column")   # matched row per column i
        driver = np.array([1.0 if match[i] < 0 else 0.2 for i in range(N)])      # unmatched in-node = driver
    except Exception:
        driver = np.ones(N)
    return driver

# ---------------------------------------------------------- panels
def equity():
    P = pd.read_csv(BASE.parent / "pilot_3p46_equity" / "equity_weekly_close_2007_2010.csv",
                    parse_dates=["week_ending"]).set_index("week_ending")
    P = P.resample("W").last().loc["2007-01-01":"2010-06-30"].dropna(how="any")
    S = (-(100 * np.log(P).diff().dropna())).clip(lower=0)
    Phi, c, Sig = L.fit_var_nonneg(S.values, ridge=2e-2)
    S0 = S.loc["2008-09-01":"2008-11-15"].mean().values + 0.5
    return "2008 equities", [x.split(" (")[0] for x in S.columns], Phi, c, Sig, S0

def covid():
    DROP = {"American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands",
            "Diamond Princess", "Grand Princess", "Puerto Rico"}
    d = pd.read_csv(BASE / "jhu_confirmed_US.csv"); dc = [c for c in d.columns if "/" in c]
    st = d.groupby("Province_State")[dc].sum().drop(
        index=[s for s in DROP if s in d["Province_State"].unique()], errors="ignore")
    st.columns = pd.to_datetime(st.columns)
    wk = st.T.sort_index().resample("W").last().diff().clip(lower=0).dropna()
    top = wk.sum().sort_values(ascending=False).head(14).index
    W = (wk[top] / 1000.0).loc["2020-03-01":"2022-06-30"]
    Phi, c, Sig = L.fit_var_nonneg(W.values, ridge=5e-2)
    S0 = W.loc["2020-06-01":"2020-08-01"].mean().values + 0.5
    return "COVID-19", list(top), Phi, c, Sig, S0

def assess(name, names, Phi, c, Sig, S0):
    Phi = rescale(Phi); B = 0.20 * float(S0.sum())
    base = run_controller(Phi, c, S0, B, "none")
    methods = {
        "transmitter (DY-net, ours)": ("directed", s_transmitter(Phi, Sig)),
        "loudest (reactive)":         ("reactive", None),
        "effective-distance":         ("cross-paradigm", s_effective_distance(Phi)),
        "collective-influence":       ("cross-paradigm", s_collective_influence(Phi)),
        "structural-controllability": ("cross-paradigm", s_structural_controllability(Phi)),
    }
    rows = []
    for m, (fam, sc) in methods.items():
        val = run_controller(Phi, c, S0, B, "greedy" if m.startswith("loudest") else "score", sc)
        rows.append((m, fam, 100 * (1 - val / base) if base > 0 else 0.0))
    rows.sort(key=lambda r: -r[2])
    print(f"\n[{name}]  N={len(names)}  budget={B:.2f}  cascade reduction vs no-action:")
    for m, fam, red in rows:
        print(f"    {red:+6.1f}%   {m:32s} [{fam}]")
    tr = next(r[2] for r in rows if r[0].startswith("transmitter"))
    cross = [r for r in rows if r[1] == "cross-paradigm"]
    best_cross = max(cross, key=lambda r: r[2])
    verdict = ("matches/exceeds all cross-paradigm baselines" if tr >= best_cross[2] - 0.5 else
               f"beaten by {best_cross[0]} ({best_cross[2]:+.1f}% vs {tr:+.1f}%)")
    print(f"  -> transmitter-targeting {verdict}")
    return dict(network=name, N=len(names), budget=round(B, 2),
                reductions={m: round(red, 2) for m, _f, red in rows},
                transmitter_vs_best_crossparadigm=verdict)

def main():
    out = [assess(*equity()), assess(*covid())]
    json.dump(out, open(BASE / "PROPOSED_crossparadigm_baseline.json", "w"), indent=1)
    print("\n[wrote PROPOSED_crossparadigm_baseline.json]")

if __name__ == "__main__":
    main()
