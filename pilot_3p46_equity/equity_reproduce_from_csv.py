"""
Self-contained 2008-equity reproduction from the REDISTRIBUTABLE weekly CSV
(equity_weekly_close_2007_2010.csv) -- no daily vendor JSON required.

This is the archive's one-command equity path. It reproduces, from the deposited CSV alone:

  * the directed connectedness:      DY total ~81%, US net-transmitter ~+21%
  * the published interdiction pair: transmitter-targeting ~+42%, loudest-node ~-7%
  * the published alpha=1 advantage: ~+49 points

These are the connectedness and primary interdiction-pair headlines; this mode does not
reproduce the full symmetrization curve or every supplementary analysis.

The interdiction figures are produced by `interdict_adv`, whose canonical definition lives in
pilot_review/bootstrap_ci.py (and verbatim in nonnormality_predictor.py / _bootstrap_check.py).
It is inlined below so this script stays self-contained. Its budget is the archive-wide default
B = 2.0, shared with the five transfer scripts, bootstrap_ci, directedness_null, groundtruth_gap,
nonnormality_predictor, si_rho_sensitivity and validity_decomposition.

NOTE ON THE TWO BUDGETS. PROPOSED_crossparadigm_baseline.py is the one script that instead uses a
budget-proportional rule (B = 0.2*sum(S0) ~ 9.02 here) so that its cross-paradigm table is
like-for-like internally; at that budget the loudest-node figure is ~-22% rather than ~-7%.
Transmitter-targeting is near budget-invariant (+42.0% at B=2.0 vs +42.3% at B=9.02); the reactive
controller is not (-6.8% vs -24.2%). Both signs hold at both budgets, so the qualitative result is
budget-independent while the magnitudes are not. The SI states this.

A second ARRO/planner comparison (none/random/greedy/learned-MPC/oracle-MPC) is also
run below as a secondary check. It is a different controller family with its own budget, so its
magnitudes are NOT the published pair -- only the ordering is informative there.

Run:  python3 equity_reproduce_from_csv.py            (faithful: 16 seeds, full training; a few min)
      python3 equity_reproduce_from_csv.py --fast     (quick check: fewer seeds/steps)
      python3 equity_reproduce_from_csv.py --headline (connectedness + published pair only; seconds)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "pilot_cross_tier"))
import lsa_capstone as L

FAST = "--fast" in sys.argv
HEADLINE_ONLY = "--headline" in sys.argv
CSV = BASE / "equity_weekly_close_2007_2010.csv"
RIDGE = 2e-2          # operating penalty for this twin (best-conditioned panel); see SI


# ------------------------------------------------------------------ interdict_adv (verbatim from
# ------------------------------------------------------------------ pilot_review/bootstrap_ci.py)
def _project(a, x, B):
    a = np.clip(a, 0, x); s = a.sum(); return a * (B / s) if s > B else a


def _stepf(Phi, c, x, a, rng):
    return np.clip(Phi @ np.clip(x - a, 0, None) + c + 0.05 * rng.standard_normal(len(x)), 0, None)


def _greedy(x, B):
    a = np.zeros_like(x); rem = B
    for i in np.argsort(-x):
        g = min(x[i], rem); a[i] = g; rem -= g
        if rem <= 1e-9: break
    return a


def _alloc_score(score, x, B):
    w = np.clip(score, 0, None)
    return _project(B * w / w.sum(), x, B) if w.sum() > 0 else np.zeros_like(x)


def _rescale(Phi, rho=1.06):
    ev = max(abs(np.linalg.eigvals(Phi))); return Phi * (rho / ev) if ev > 1e-6 else Phi


def interdict_adv(Phi0, c0, Sig0, S0, B=2.0, T=16, seeds=16, seed0=20):
    """transmitter% and greedy% cascade reduction (vs no-action) on the rescaled twin."""
    _, _, NET, _ = L.connectedness(L.gfevd(Phi0, Sig0)); Phi = _rescale(Phi0)
    scores = {"transmitter": np.clip(NET, 0, None)}; out = {}
    for name in ["none", "greedy", "transmitter"]:
        tt = []
        for s in range(seeds):
            rng = np.random.default_rng(seed0 + s); x = S0.copy(); acc = 0.0
            for _t in range(T):
                a = np.zeros_like(x) if name == "none" else (
                    _greedy(x, B) if name == "greedy" else _alloc_score(scores[name], x, B))
                x = _stepf(Phi, c0, x, a, rng); acc += x.sum()
            tt.append(acc)
        out[name] = float(np.mean(tt))
    b = out["none"]
    if b <= 0: return 0.0, 0.0
    red = {k: 100 * (1 - v / b) for k, v in out.items()}
    return red["transmitter"], red["greedy"]


def main():
    P = pd.read_csv(CSV, parse_dates=["week_ending"]).set_index("week_ending")
    P = P.resample("W").last().loc["2007-01-01":"2010-06-30"].dropna(how="any")
    stress = (-(100 * np.log(P).diff().dropna())).clip(lower=0)
    names = list(stress.columns)

    Phi, c, Sig = L.fit_var_nonneg(stress.values, ridge=RIDGE)
    TO, FROM, NET, tot = L.connectedness(L.gfevd(Phi, Sig))
    us = int(np.argmax(NET))
    print("=" * 70)
    print("CONNECTEDNESS (from the redistributable weekly CSV):")
    print(f"  DY total connectedness index = {tot:.1f}%          [paper: ~81%]")
    print(f"  net-transmitter = {names[us]}  (net {NET[us]:+.1f}%)     [paper: US, ~+21%]")
    ok_conn = abs(tot - 81) < 3 and names[us].startswith("US") and abs(NET[us] - 21) < 3
    print(f"  -> connectedness reproduces: {'YES' if ok_conn else 'CHECK'}")

    S0 = stress.loc["2008-09-01":"2008-11-15"].mean().values + 0.5
    tr, gr = interdict_adv(Phi, c, Sig, S0)
    print("=" * 70)
    print("PUBLISHED INTERDICTION PAIR (interdict_adv, archive default B=2.0):")
    print(f"  transmitter-targeting      = {tr:+6.1f}%          [paper: ~+42%]")
    print(f"  loudest-node (reactive)    = {gr:+6.1f}%          [paper: ~-7%]")
    print(f"  advantage at alpha=1       = {tr - gr:+6.1f} pts        [paper: ~+49]")
    ok_int = abs(tr - 42) < 2.5 and abs(gr + 7) < 2.5
    print(f"  -> published magnitudes reproduce: {'YES' if ok_int else 'CHECK'}")
    if not (ok_conn and ok_int):
        raise SystemExit("Equity headline reproduction failed its numeric checks.")

    if HEADLINE_ONLY:
        print("=" * 70)
        print("Connectedness and the published interdiction pair reproduce from this CSV. (--headline mode)")
        return

    seeds = 6 if FAST else 16
    steps = 800 if FAST else 3000
    train = 120 if FAST else 400
    print("=" * 70)
    print(f"SECONDARY: ARRO/planner family on the CSV twin (seeds={seeds}; different action rule,")
    print("           so magnitudes are NOT the published pair -- read the ordering only):")
    ID = L.run_interdiction(Phi, c, Sig, S0, names, target_rho=1.05, budget=2.0, T_ep=24,
                            seeds=seeds, H=4, train_eps=train, steps=steps, verbose=False)
    base = ID["summary"]["none"]["mean"]
    red = {m: 100 * (1 - ID["summary"][m]["mean"] / base) for m in ID["order"]}
    for m in ID["order"]:
        print(f"  {m:12s} {red[m]:+6.0f}%")
    print("=" * 70)
    print("The connectedness + published interdiction pair + alpha=1 advantage")
    print("reproduce from this deposited CSV; no daily vendor series is required.")


if __name__ == "__main__":
    main()
