"""
PROPOSED (candidate robustness addition -- NOT wired into the manuscript).

Bootstrap confidence intervals on the Diebold-Yilmaz CONNECTEDNESS quantities themselves
(net connectedness per node and the total connectedness index), so every reported
net-transmitter carries an uncertainty band and one can answer the reflexive referee
question: "is this net-transmitter edge distinguishable from zero?"

This complements the existing `bootstrap_ci.py`, which bootstraps the transmitter IDENTITY
and the controller ADVANTAGE but does not attach a CI to the connectedness numbers.

Method: stationary block bootstrap (Politis & Romano 1994) of the multivariate stress
panel -- resample contiguous wrap-around blocks of whole time-rows with geometric block
lengths (mean L ~ sqrt(T)), preserving contemporaneous structure across series and the
lag-1 dynamics the VAR(1) reads -- then refit fit_var_nonneg and recompute connectedness.

Self-contained and OFFLINE: uses the bundled 2008-equity weekly-close CSV, the paper's
headline network (US net-transmitter, DY total ~81%). Deterministic (fixed seed).

Run:  python3 PROPOSED_connectedness_bootstrap.py [--K 500]
Writes: PROPOSED_connectedness_bootstrap.{json,pdf,png}
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "pilot_cross_tier"))
import lsa_capstone as L
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = BASE.parent / "pilot_3p46_equity" / "equity_weekly_close_2007_2010.csv"
RIDGE = 2e-2
K = int(sys.argv[sys.argv.index("--K") + 1]) if "--K" in sys.argv else 500
SEED = 20260703

# ------------------------------------------------------------------ panel (verbatim equity pipeline)
def load_stress():
    P = pd.read_csv(CSV, parse_dates=["week_ending"]).set_index("week_ending")
    P = P.resample("W").last().loc["2007-01-01":"2010-06-30"].dropna(how="any")
    ret = 100 * np.log(P).diff().dropna()
    stress = (-ret).clip(lower=0)              # weekly % decline (0 if up) = contagion stress
    return stress.values, list(stress.columns)

def net_and_tci(M):
    Phi, c, Sig = L.fit_var_nonneg(M, ridge=RIDGE)
    TO, FROM, NET, tot = L.connectedness(L.gfevd(Phi, Sig))
    return NET, tot

# ------------------------------------------------------------------ stationary bootstrap
def stationary_indices(T, L_mean, rng):
    """Politis-Romano stationary bootstrap: geometric blocks, wrap-around."""
    p = 1.0 / L_mean
    idx = np.empty(T, dtype=int)
    t = 0
    cur = rng.integers(T)
    while t < T:
        idx[t] = cur
        t += 1
        if rng.random() < p:
            cur = rng.integers(T)          # start a new block
        else:
            cur = (cur + 1) % T            # extend the block (wrap)
    return idx

def main():
    M, names = load_stress()
    T, N = M.shape
    Lblk = max(2, int(round(np.sqrt(T))))
    NET0, TCI0 = net_and_tci(M)

    rng = np.random.default_rng(SEED)
    NETb = np.full((K, N), np.nan)
    TCIb = np.full(K, np.nan)
    for k in range(K):
        idx = stationary_indices(T, Lblk, rng)
        try:
            NETb[k], TCIb[k] = net_and_tci(M[idx])
        except Exception:
            pass  # a degenerate resample: leave NaN, drop from percentiles

    lo, hi = np.nanpercentile(NETb, 2.5, axis=0), np.nanpercentile(NETb, 97.5, axis=0)
    tci_lo, tci_hi = np.nanpercentile(TCIb, 2.5), np.nanpercentile(TCIb, 97.5)
    p_pos = np.nanmean(NETb > 0, axis=0)       # P(net transmitter) across resamples

    print(f"Stationary block bootstrap: K={K}, mean block L={Lblk}, T={T}, N={N}")
    print(f"Total connectedness index: {TCI0:.1f}%  95% CI [{tci_lo:.1f}, {tci_hi:.1f}]")
    print("=" * 78)
    print(f"{'node':22s} {'NET%':>7s} {'95% CI':>18s} {'P(NET>0)':>9s}  verdict")
    order = np.argsort(-NET0)
    rows = []
    for i in order:
        excl0 = lo[i] > 0 or hi[i] < 0
        verdict = ("net TRANSMITTER (CI>0)" if lo[i] > 0 else
                   "net receiver (CI<0)" if hi[i] < 0 else
                   "not distinguishable from 0")
        print(f"{names[i]:22.22s} {NET0[i]:+7.1f}  [{lo[i]:+6.1f}, {hi[i]:+6.1f}]   {p_pos[i]:7.2f}  {verdict}")
        rows.append(dict(node=names[i], net=round(float(NET0[i]), 2),
                         ci_lo=round(float(lo[i]), 2), ci_hi=round(float(hi[i]), 2),
                         p_net_positive=round(float(p_pos[i]), 3),
                         distinguishable_from_zero=bool(excl0)))
    print("=" * 78)
    top = names[int(np.argmax(NET0))]
    top_excl = lo[int(np.argmax(NET0))] > 0
    print(f"Headline: top net-transmitter = {top}; its 95% CI "
          f"{'EXCLUDES' if top_excl else 'includes'} zero "
          f"-> {'distinguishable from zero' if top_excl else 'NOT distinguishable (report as such)'}")

    out = dict(network="2008 equities (weekly decline-stress)", K=K, block_mean=Lblk, T=int(T),
               tci=round(float(TCI0), 2), tci_ci=[round(float(tci_lo), 2), round(float(tci_hi), 2)],
               nodes=rows, top_transmitter=top, top_ci_excludes_zero=bool(top_excl))
    json.dump(out, open(BASE / "PROPOSED_connectedness_bootstrap.json", "w"), indent=2)

    # forest plot: NET with 95% CI, zero line
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = np.arange(N)
    o = order[::-1]
    ax.errorbar(NET0[o], y, xerr=[NET0[o] - lo[o], hi[o] - NET0[o]], fmt="o", color="#1f4e79",
                ecolor="#8aa9c7", capsize=3, lw=1.4, ms=5)
    ax.axvline(0, color="#b03030", lw=1.1, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([names[i] for i in o], fontsize=9)
    ax.set_xlabel("Net Connectedness (TO − FROM, %)  —  95% Stationary-Bootstrap CI")
    ax.set_title("Bootstrap Uncertainty on Directed Net-Transmitter Strength\n"
                 "2008 Equity-Crash Network (K=%d)" % K, fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(BASE / f"PROPOSED_connectedness_bootstrap.{ext}", dpi=140)
    print(f"[wrote PROPOSED_connectedness_bootstrap.json/.pdf/.png]")

if __name__ == "__main__":
    main()
