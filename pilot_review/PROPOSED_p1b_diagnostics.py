"""
PROPOSED P1b (does NOT modify committed scripts/manuscript). Two more referee-panel robustness re-runs,
reusing the committed lsa_capstone + the twin simulator in PROPOSED_p1_robustness:

  (A) STRONG CROSS-PARADIGM BASELINE. The existing controllers are all VAR-derived (out-strength, impulse
      spillover, correlation centrality). The panel asked for an established, different-paradigm influential-node
      measure. We add PageRank-influence and eigenvector-influence (computed on the fitted |Phi| influence graph)
      as controllers and test whether the Diebold-Yilmaz net-transmitter still beats them.
  (B) DEGREES-OF-FREEDOM / LAMBDA SENSITIVITY. For the p~N~T regime, report params vs T, the residual-covariance
      condition number, an effective-dof for the ridge fit, and whether the net-transmitter IDENTITY is stable as
      the ridge lambda varies across an order of magnitude.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "pilot_cross_tier"))
import lsa_capstone as L
sys.path.insert(0, str(Path(__file__).parent))
import PROPOSED_p1_robustness as P1

def influence_graph(Phi):
    W = np.abs(Phi).copy(); np.fill_diagonal(W, 0.0); return W   # W[i,j] = strength of j -> i

def pagerank_influence(Phi, d=0.85, it=200):
    # PageRank on the REVERSED graph (W^T): a node scores high if it INFLUENCES influential nodes (a source measure)
    W = influence_graph(Phi).T
    n = W.shape[0]; cs = W.sum(0); P = np.where(cs > 0, W / np.where(cs == 0, 1, cs), 1.0 / n)
    r = np.ones(n) / n
    for _ in range(it): r = d * P @ r + (1 - d) / n
    return r

def eigen_influence(Phi):
    W = influence_graph(Phi).T                       # out-influence direction
    vals, vecs = np.linalg.eig(W)
    v = np.abs(vecs[:, int(np.argmax(vals.real))].real); return v / (v.sum() + 1e-12)

def cross_paradigm(M, names, seeds=16):
    Phi, c, Sig = L.fit_var_nonneg(M, ridge=5e-2)
    _, _, NET, tot = L.connectedness(L.gfevd(Phi, Sig))
    S0 = np.maximum(M.mean(0), 0.5)
    scores = {"transmitter": np.clip(NET, 0, None),
              "pagerank": pagerank_influence(Phi),
              "eigencentrality": eigen_influence(Phi)}
    red = P1.interdict(P1.rescale(Phi), c, S0, scores, seeds=seeds)
    adv = {k: round(red[k] - red["greedy"], 1) for k in scores}
    return {"transmitter": names[int(np.argmax(NET))],
            "pagerank_top": names[int(np.argmax(scores["pagerank"]))],
            "eigen_top": names[int(np.argmax(scores["eigencentrality"]))],
            "advantage_over_greedy": adv,
            "transmitter_minus_pagerank": round(red["transmitter"] - red["pagerank"], 1),
            "transmitter_minus_eigen": round(red["transmitter"] - red["eigencentrality"], 1)}

def dof_lambda(M, names):
    T, N = M.shape
    params = N * N + N                                # Phi + intercept per equation
    Phi, c, Sig = L.fit_var_nonneg(M, ridge=5e-2)
    cond = float(np.linalg.cond(Sig))
    # effective dof of the ridge fit: tr[X(X'X+lam P)^-1 X'] summed structure -> approximate via ridge hat trace
    X = np.column_stack([np.ones(T - 1), M[:-1]]); lam = 5e-2
    Preg = np.eye(N + 1); Preg[0, 0] = 0.0
    H = X @ np.linalg.solve(X.T @ X + lam * (T - 1) * Preg, X.T)
    edof = float(np.trace(H)) * N                     # per-equation hat-trace x N equations
    # lambda sensitivity of the transmitter IDENTITY across an order of magnitude
    lams = [5e-3, 1.6e-2, 5e-2, 1.6e-1, 5e-1]; trans = []
    for lm in lams:
        Ph, cc, Sg = L.fit_var_nonneg(M, ridge=lm)
        _, _, NETl, _ = L.connectedness(L.gfevd(Ph, Sg)); trans.append(names[int(np.argmax(NETl))])
    return {"T": T, "N": N, "params": params, "params_over_T": round(params / T, 2),
            "Sigma_condition_number": round(cond, 1), "effective_dof": round(edof, 1),
            "transmitter_across_lambda": dict(zip([str(x) for x in lams], trans)),
            "transmitter_stable": len(set(trans)) == 1}

def run(M, names, label):
    M = np.asarray(M, float)
    cp = cross_paradigm(M, names); dl = dof_lambda(M, names)
    print(f"\n[P1b :: {label}]  T={dl['T']} N={dl['N']} params={dl['params']} (params/T={dl['params_over_T']})")
    print(f"  (A) cross-paradigm: DY-transmitter={cp['transmitter']} | PageRank-top={cp['pagerank_top']} | eigen-top={cp['eigen_top']}")
    print(f"      advantage over greedy: {cp['advantage_over_greedy']}")
    print(f"      transmitter - pagerank = {cp['transmitter_minus_pagerank']} ; transmitter - eigen = {cp['transmitter_minus_eigen']}")
    print(f"  (B) Sigma cond#={dl['Sigma_condition_number']} | eff-dof={dl['effective_dof']} | transmitter stable across lambda x100: {dl['transmitter_stable']}")
    print(f"      transmitter(lambda): {dl['transmitter_across_lambda']}")
    return {"label": label, "cross_paradigm": cp, "dof_lambda": dl}
