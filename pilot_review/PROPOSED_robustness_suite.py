"""
PROPOSED (candidate robustness addition -- NOT wired into the manuscript).

Tier-2 statistical robustness suite for the directed-asymmetry headline (2008-equity network,
US net-transmitter). Ports the four checks from the consilience/Neuro_Atlas run
(~/Documents/_neuro_atlas_build/robustness_suite.py) to the DY-connectedness estimand:

  1. Stationary block bootstrap of net connectedness -> point CI + standard error.
  2. Type-S / Type-M design analysis (Gelman-Carlin retrodesign) on the US transmitter, whose
     bootstrap CI barely excludes zero -- quantifies sign-error and exaggeration risk if the
     true net-transmission is small.
  3. TOST equivalence of the NON-transmitter nodes against a smallest-effect-of-interest, i.e.
     "is the rest of the ranking statistically equivalent to zero net transmission?"
  4. Benjamini-YEKUTIELI FDR (valid under dependence -- the edges come from one VAR, so plain
     Benjamini-Hochberg is not obviously valid) across the per-node net-transmitter tests.

Self-contained and OFFLINE (bundled 2008-equity CSV). Deterministic. Writes PROPOSED_robustness_suite.json.
Run:  python3 PROPOSED_robustness_suite.py [--B 3000]
"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "pilot_cross_tier"))
import lsa_capstone as L

CSV = BASE.parent / "pilot_3p46_equity" / "equity_weekly_close_2007_2010.csv"
RIDGE = 2e-2
B = int(sys.argv[sys.argv.index("--B") + 1]) if "--B" in sys.argv else 3000
SEED = 20260703
SESOI = 5.0        # smallest net-connectedness (percentage points) worth caring about
rng = np.random.default_rng(SEED)

def load_stress():
    P = pd.read_csv(CSV, parse_dates=["week_ending"]).set_index("week_ending")
    P = P.resample("W").last().loc["2007-01-01":"2010-06-30"].dropna(how="any")
    stress = (-(100 * np.log(P).diff().dropna())).clip(lower=0)
    return stress.values, list(stress.columns)

def net(M):
    Phi, c, Sig = L.fit_var_nonneg(M, ridge=RIDGE)
    _TO, _FROM, NET, _tot = L.connectedness(L.gfevd(Phi, Sig))
    return NET

def stationary_indices(T, Lmean, rng):
    p = 1.0 / Lmean; idx = np.empty(T, int); cur = rng.integers(T)
    for t in range(T):
        idx[t] = cur
        cur = rng.integers(T) if rng.random() < p else (cur + 1) % T
    return idx

def retrodesign(true_effect, se, M=200000, z_crit=1.96):
    """Gelman-Carlin: estimator ~ Normal(true, se); significance = |est| > z_crit*se (CI excludes 0)."""
    est = rng.normal(true_effect, se, M)
    sig = np.abs(est) > z_crit * se
    if sig.sum() == 0:
        return {"true_net": true_effect, "power": 0.0, "type_s": None, "type_m": None}
    return {"true_net": true_effect, "power": round(float(sig.mean()), 3),
            "type_s": round(float(np.mean(np.sign(est[sig]) != np.sign(true_effect))), 4),
            "type_m": round(float(np.mean(np.abs(est[sig])) / abs(true_effect)), 2)}

def tost(est, se, sesoi):
    p_lower = stats.norm.sf((est - (-sesoi)) / se)     # H0: true <= -SESOI
    p_upper = stats.norm.cdf((est - sesoi) / se)       # H0: true >=  SESOI
    p = max(p_lower, p_upper)
    return float(p), bool(p < 0.05)

def by_fdr(pvals, alpha=0.05):
    """Benjamini-Yekutieli: valid under arbitrary dependence."""
    m = len(pvals); order = np.argsort(pvals); ranked = np.array(pvals)[order]
    cm = np.sum(1.0 / np.arange(1, m + 1))
    thresh = (np.arange(1, m + 1) / (m * cm)) * alpha
    below = ranked <= thresh
    kmax = int(np.max(np.where(below)[0])) + 1 if below.any() else 0
    survivors = set(order[:kmax].tolist())
    return kmax, cm, survivors

def main():
    M, names = load_stress()
    T, N = M.shape
    Lblk = max(2, int(round(np.sqrt(T))))
    NET0 = net(M)
    us = int(np.argmax(NET0))

    # 1) bootstrap
    boot = np.full((B, N), np.nan)
    for b in range(B):
        try:
            boot[b] = net(M[stationary_indices(T, Lblk, rng)])
        except Exception:
            pass
    se = np.nanstd(boot, axis=0)
    lo, hi = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)
    p_one_sided = np.nanmean(boot <= 0, axis=0)          # small => strong net transmitter

    # 2) design analysis on US
    se_us = float(se[us])
    design = [retrodesign(t, se_us) for t in (5.0, 10.0, float(round(NET0[us], 1)), 40.0)]

    # 3) TOST for non-US nodes
    tost_rows = []
    n_equiv = 0
    for i in range(N):
        if i == us:
            continue
        p, eq = tost(float(NET0[i]), float(se[i]), SESOI)
        n_equiv += eq
        tost_rows.append(dict(node=names[i], net=round(float(NET0[i]), 2), se=round(float(se[i]), 2),
                              tost_p=round(p, 4), equivalent_to_zero=eq))

    # 4) BY-FDR across per-node net-transmitter tests
    kmax, cm, survivors = by_fdr(list(p_one_sided))
    fdr_rows = [dict(node=names[i], net=round(float(NET0[i]), 2), p=round(float(p_one_sided[i]), 4),
                     survives_BY_FDR=bool(i in survivors)) for i in np.argsort(-NET0)]

    out = {
        "network": "2008 equities (weekly decline-stress)", "n_nodes": N, "T": int(T), "bootstrap_B": B,
        "block_mean": Lblk, "transmitter": names[us],
        "bootstrap": {"us_net": round(float(NET0[us]), 2), "us_ci95": [round(float(lo[us]), 2), round(float(hi[us]), 2)],
                      "us_se": round(se_us, 2), "us_fraction_negative": round(float(p_one_sided[us]), 3)},
        "design_analysis": {"note": "Gelman-Carlin retrodesign; SE from the stationary bootstrap; significance = CI excludes 0",
                            "by_assumed_true_net": design},
        "tost_equivalence": {"SESOI_net_pct": SESOI, "n_nonUS_equivalent_to_zero": n_equiv,
                             "n_nonUS": N - 1, "per_node": tost_rows},
        "benjamini_yekutieli_fdr": {"family_size": N, "c_m": round(float(cm), 3),
                                    "n_survive_0.05": kmax, "survivors": [names[i] for i in survivors],
                                    "note": "one-sided net-transmitter p per node; BY is valid under the edge dependence induced by a single VAR",
                                    "per_node": fdr_rows},
    }
    print(json.dumps(out, indent=1))
    json.dump(out, open(BASE / "PROPOSED_robustness_suite.json", "w"), indent=1)
    print("\n[wrote PROPOSED_robustness_suite.json]")

if __name__ == "__main__":
    main()
