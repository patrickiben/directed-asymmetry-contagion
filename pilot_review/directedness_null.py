"""
DATA-LEVEL DIRECTEDNESS NULL for the transmitter-targeting controller advantage.

For each of the 7 systems we ask: does the observed transmitter-over-greedy advantage
exceed what a DIRECTION-DESTROYING null produces?  Two complementary nulls:

(a) TIME-REVERSAL.  Refit the VAR on the time-reversed multivariate series M[::-1] and
    recompute interdict_adv.  Genuine forward-in-time directionality should largely vanish
    under reversal (the causal arrow flips / smears).  Reported as a single reversed-advantage
    value per system (point null, no distribution -> descriptive, not a p-value).

(b) SYMMETRIC-SURROGATE PARAMETRIC BOOTSTRAP.  Build the symmetrized kernel
    Phi_sym = 0.5*(Phi + Phi^T) (rho-matched back to the original spectral radius so the
    cascade energy is preserved), simulate ~150 surrogate VAR(1) series driven by RESAMPLED
    fit residuals, refit each surrogate with the SAME fit_var_nonneg(ridge=5e-2), and recompute
    interdict_adv.  Phi_sym is normal-by-construction => destroys the directed (antisymmetric)
    part while preserving total coupling magnitude.  The advantages of the refit surrogates form
    the null distribution; the observed advantage's upper-tail p-value =
        (1 + #{null >= observed}) / (1 + n_surr).

Reuses, verbatim, the loaders + interdict_adv + fit from nonnormality_predictor.py.

Run: python3 directedness_null.py
Outputs: directedness_null.json + directedness_null.png/.pdf
"""
import sys, json, time
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent / "pilot_cross_tier"))

# import the SHARED machinery verbatim (loaders, interdict_adv, rescale, RIDGE, lsa_capstone as L)
import nonnormality_predictor as NN          # noqa  (runs its analysis once on import; harmless, we only need its defs)
import lsa_capstone as L
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RIDGE = NN.RIDGE
interdict_adv = NN.interdict_adv

# ------------------------------------------------------------------ systems (loader + published advantage + verdict)
PUBLISHED_ADV = {  # transmitter% - greedy% as reported / recomputed in the pipeline
    "asia97": 7.3, "smoke": 6.2, "flu": 9.9, "flights": 3.9,
    "conflict": 1.2, "equity": None, "COVID": 31.7,   # equity recomputed below
}
VERDICT = {"asia97": "confirm", "smoke": "confirm", "flu": "refine", "flights": "falsify",
           "conflict": "null", "equity": "confirm", "COVID": "confirm"}

LOADERS = {
    "asia97": NN.load_asia97, "smoke": NN.load_smoke, "flu": NN.load_flu,
    "flights": NN.load_flights, "conflict": NN.load_conflict,
    "equity": NN.load_equity, "COVID": NN.load_covid,
}
ORDER = ["asia97", "smoke", "flu", "flights", "conflict", "equity", "COVID"]

N_SURR = 150
SIM_SEED = 7

# ------------------------------------------------------------------ helpers
def fit(M):
    """Same fit used everywhere: VAR(1) non-neg off-diagonal, ridge=5e-2. Returns Phi,c,Sig,resid."""
    Phi, c, Sig = L.fit_var_nonneg(M, ridge=RIDGE)
    # reconstruct one-step residuals R = Y - (X @ coef) with the fitted Phi,c
    T, Nn = M.shape
    X = M[:-1]; Y = M[1:]
    pred = X @ Phi.T + c
    R = Y - pred
    return Phi, c, Sig, R

def adv_of(Phi, c, Sig, S0):
    tr, gr = interdict_adv(Phi, c, Sig, S0)
    return float(tr - gr), float(tr), float(gr)

def symmetrize_rho_matched(Phi):
    """Phi_sym = 0.5*(Phi+Phi^T), rescaled so its spectral radius == rho(Phi).
    Off-diagonals clipped >=0 to stay inside the non-neg contagion class (then re-symmetrized)."""
    Ps = 0.5 * (Phi + Phi.T)
    # keep non-negativity of off-diagonals (contagion); symmetric clip is still symmetric
    off = Ps.copy(); np.fill_diagonal(off, 0.0)
    off = np.clip(off, 0.0, None)
    Ps = off + np.diag(np.diag(Ps))
    rho0 = float(max(abs(np.linalg.eigvals(Phi))))
    rhoS = float(max(abs(np.linalg.eigvals(Ps))))
    if rhoS > 1e-9:
        Ps = Ps * (rho0 / rhoS)
    return Ps

def simulate(Phi, c, R, T, S0_state, rng, burn=40):
    """Simulate a VAR(1) path x_{t+1} = Phi x_t + c + e_t, e_t resampled (with replacement) from
    residual rows R. Non-negativity clamp (matches the contagion data convention: stresses>=0).
    Returns a (T x N) panel comparable to the original M."""
    N = Phi.shape[0]
    nR = R.shape[0]
    x = np.maximum(S0_state.astype(float), 0.0)
    out = np.empty((T + burn, N))
    for t in range(T + burn):
        e = R[rng.integers(0, nR)]
        x = Phi @ x + c + e
        x = np.maximum(x, 0.0)
        out[t] = x
    return out[burn:]

# ------------------------------------------------------------------ main loop
def run_system(name):
    loader = LOADERS[name]
    M, S0 = loader()
    T, Nn = M.shape
    Phi0, c0, Sig0, R0 = fit(M)

    # --- OBSERVED advantage (recompute consistently for ALL systems so the null is on the same scale)
    obs_adv, obs_tr, obs_gr = adv_of(Phi0, c0, Sig0, S0)

    # --- (a) TIME-REVERSAL: refit on reversed series, recompute advantage
    Mr = M[::-1].copy()
    Phir, cr, Sigr, _ = fit(Mr)
    rev_adv, rev_tr, rev_gr = adv_of(Phir, cr, Sigr, S0)

    # --- (b) SYMMETRIC-SURROGATE bootstrap
    Phi_sym = symmetrize_rho_matched(Phi0)
    # state seed for simulation: use the data's own mean state so dynamics are exercised
    S0_state = np.maximum(M.mean(0), 0.5)
    rng = np.random.default_rng(SIM_SEED + hash(name) % 10_000)
    null_adv = []
    n_fail = 0
    for b in range(N_SURR):
        Msur = simulate(Phi_sym, c0, R0, T, S0_state, rng)
        # guard against degenerate (all-constant / non-finite) surrogates
        if not np.all(np.isfinite(Msur)) or Msur.std() < 1e-8:
            n_fail += 1
            continue
        Phib, cb, Sigb, _ = fit(Msur)
        # use the SAME steady seed S0 for interdiction so only the kernel differs
        a_b, _, _ = adv_of(Phib, cb, Sigb, S0)
        if np.isfinite(a_b):
            null_adv.append(a_b)
        else:
            n_fail += 1
    null_adv = np.array(null_adv)

    n_eff = len(null_adv)
    # upper-tail p-value: how often does the symmetric (direction-destroyed) null match/beat observed
    p_val = float((1 + np.sum(null_adv >= obs_adv)) / (1 + n_eff)) if n_eff else float("nan")
    null_mean = float(np.mean(null_adv)) if n_eff else float("nan")
    null_sd = float(np.std(null_adv, ddof=1)) if n_eff > 1 else float("nan")
    null_q = (np.percentile(null_adv, [2.5, 50, 97.5]).tolist() if n_eff else [None, None, None])
    # observed percentile within null
    obs_pct = float(100.0 * np.mean(null_adv < obs_adv)) if n_eff else float("nan")
    z = float((obs_adv - null_mean) / null_sd) if (n_eff > 1 and null_sd > 0) else float("nan")

    return dict(
        system=name, verdict=VERDICT[name], N=int(Nn), T=int(T),
        published_advantage=PUBLISHED_ADV[name],
        observed=dict(advantage=round(obs_adv, 3), transmitter=round(obs_tr, 3), greedy=round(obs_gr, 3)),
        time_reversal=dict(advantage=round(rev_adv, 3), transmitter=round(rev_tr, 3), greedy=round(rev_gr, 3),
                           retained_fraction=round(rev_adv / obs_adv, 3) if abs(obs_adv) > 1e-6 else None),
        symmetric_surrogate=dict(
            n_surrogates=N_SURR, n_effective=int(n_eff), n_failed=int(n_fail),
            null_mean=round(null_mean, 3), null_sd=round(null_sd, 3),
            null_ci95=[round(q, 3) if q is not None else None for q in null_q],
            observed_percentile=round(obs_pct, 2), z_score=round(z, 3) if np.isfinite(z) else None,
            p_value=round(p_val, 4) if np.isfinite(p_val) else None,
        ),
        _null_samples=null_adv.tolist(),  # kept for the figure, stripped before json (kept actually; small)
    )

def main():
    t0 = time.time()
    results = {}
    for name in ORDER:
        try:
            r = run_system(name)
            results[name] = r
            ss = r["symmetric_surrogate"]; tr = r["time_reversal"]
            print(f"[{name:8s}] {r['verdict']:8s} obs={r['observed']['advantage']:+7.2f}  "
                  f"REV={tr['advantage']:+7.2f} (ret {tr['retained_fraction']})  "
                  f"SYMnull mean={ss['null_mean']:+6.2f} sd={ss['null_sd']:.2f}  "
                  f"obs_pct={ss['observed_percentile']:5.1f}%  p={ss['p_value']}  (n={ss['n_effective']})")
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = dict(system=name, error=str(e))
            print(f"[{name}] ERROR: {e}")

    # ----------------------------------------------------------- save json (keep null samples; file stays small)
    out = dict(
        description="Data-level directedness null for the transmitter-targeting controller advantage.",
        method=dict(
            time_reversal="refit VAR on M[::-1]; recompute interdict_adv; report retained advantage fraction",
            symmetric_surrogate=("simulate %d VAR(1) surrogates from rho-matched Phi_sym=0.5*(Phi+Phi^T) "
                                 "with resampled fit residuals; refit each (ridge=%.0e); null dist of "
                                 "transmitter-greedy advantage; upper-tail p=(1+#{null>=obs})/(1+n)") % (N_SURR, RIDGE),
            interdiction="shared interdict_adv (rho=1.06 rescale + simulator) from nonnormality_predictor.py",
            ridge=RIDGE,
        ),
        systems=results,
    )
    json.dump(out, open(BASE / "directedness_null.json", "w"), indent=2)
    print(f"\nsaved -> {BASE/'directedness_null.json'}   ({time.time()-t0:.0f}s)")

    # ----------------------------------------------------------- figure
    make_figure(results)
    return results

def make_figure(results):
    matplotlib.rcParams.update({"font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "savefig.dpi": 300})
    VCOL = {"confirm": "#2e8b57", "refine": "#d4a017", "falsify": "#c00000", "null": "#7f7f7f"}
    syss = [s for s in ORDER if "error" not in results[s]]
    n = len(syss)
    ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.7 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, syss):
        r = results[name]
        col = VCOL[r["verdict"]]
        null = np.array(r["_null_samples"])
        obs = r["observed"]["advantage"]
        rev = r["time_reversal"]["advantage"]
        p = r["symmetric_surrogate"]["p_value"]
        if len(null):
            ax.hist(null, bins=22, color=col, alpha=0.40, edgecolor="white", lw=0.3)
        ax.axvline(obs, color=col, lw=2.2, zorder=5, label=f"observed {obs:+.1f}")
        ax.axvline(rev, color="#333", lw=1.4, ls=":", zorder=4, label=f"time-reversed {rev:+.1f}")
        ax.axvline(0, color="#bbb", lw=0.8, ls="-", zorder=1)
        ttl_p = "p<0.007" if (p is not None and p <= 1.0 / (1 + r["symmetric_surrogate"]["n_effective"])) else \
                (f"p={p:.3f}" if p is not None else "p=n/a")
        ax.set_title(f"{name}  ({r['verdict']})\n{ttl_p}", fontsize=9.5,
                     color=col if r["verdict"] in ("confirm",) else "#222")
        ax.set_xlabel("transmitter - greedy advantage (pts)")
        ax.set_ylabel("surrogate count")
        ax.legend(fontsize=6.6, loc="upper right", framealpha=0.7)
        ax.grid(alpha=0.2)

    for ax in axes[len(syss):]:
        ax.axis("off")

    fig.suptitle("Directedness null: observed transmitter advantage vs symmetric (direction-destroying) surrogate distribution\n"
                 "solid = observed | dotted = time-reversed refit | shaded = rho-matched Phi_sym bootstrap (n=%d)" % N_SURR,
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(BASE / "directedness_null.png", dpi=200, bbox_inches="tight")
    fig.savefig(BASE / "directedness_null.pdf", bbox_inches="tight")
    print(f"figure -> {BASE/'directedness_null.png'} / .pdf")

if __name__ == "__main__":
    main()
