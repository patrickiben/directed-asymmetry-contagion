"""
PROPOSED (discharges referee LSA-RIG-5 / the Second Brain 'degrees-of-freedom strain' item -- NOT
wired into the manuscript).

Per twin, the non-negative ridge-VAR(1) estimates ~N^2 transition coefficients plus an NxN residual
covariance from as few as ~90 observations, then builds a generalized FEVD on that covariance. The
referee asks for: (1) the residual-covariance condition number; (2) an effective degrees-of-freedom
for the ridge fit vs the sample size; (3) lambda justified by sensitivity -- does the net-transmitter
RANKING change as lambda moves an order of magnitude; and (4) where p>=T, a stated-property shrinkage
(Ledoit-Wolf) estimator, checking whether the top-transmitter verdict changes.

Twins covered offline: 2008 equities (well-conditioned anchor) and held-out COVID-19. The fragile I(1)
1997 Asian-FX transfer twin is included when FRED is reachable (the point of the exercise). Smoke/flights
run identically once their panels are built (same fit_var_nonneg call).

Run:  python3 PROPOSED_df_diagnostics.py        -> PROPOSED_df_diagnostics.json
"""
import sys, io, json, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "pilot_cross_tier"))
import lsa_capstone as L

RIDGE = 5e-2
np.seterr(all="ignore")

# ------------------------------------------------------------------ panels
def panel_equity():
    P = pd.read_csv(BASE.parent / "pilot_3p46_equity" / "equity_weekly_close_2007_2010.csv",
                    parse_dates=["week_ending"]).set_index("week_ending")
    P = P.resample("W").last().loc["2007-01-01":"2010-06-30"].dropna(how="any")
    S = (-(100 * np.log(P).diff().dropna())).clip(lower=0)
    return S.values, [c.split(" (")[0] for c in S.columns], RIDGE

def panel_covid():
    DROP = {"American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands",
            "Diamond Princess", "Grand Princess", "Puerto Rico"}
    d = pd.read_csv(BASE / "jhu_confirmed_US.csv"); dc = [c for c in d.columns if "/" in c]
    st = d.groupby("Province_State")[dc].sum().drop(
        index=[s for s in DROP if s in d["Province_State"].unique()], errors="ignore")
    st.columns = pd.to_datetime(st.columns)
    wk = st.T.sort_index().resample("W").last().diff().clip(lower=0).dropna()
    top = wk.sum().sort_values(ascending=False).head(14).index
    W = (wk[top] / 1000.0).loc["2020-03-01":"2022-06-30"]
    return W.values, list(top), 5e-2

def panel_asia_fx():
    SER = {"DEXTHUS": "Thai", "DEXKOUS": "Korea", "DEXMAUS": "Malay", "DEXSIUS": "Singa", "DEXJPUS": "Japan",
           "DEXTAUS": "Taiwan", "DEXHKUS": "HK", "DEXINUS": "India", "DEXCHUS": "China"}
    def fred(i):
        u = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={i}&cosd=1996-01-01&coed=1998-12-31"
        for _ in range(3):
            try:
                df = pd.read_csv(io.StringIO(urllib.request.urlopen(u, timeout=25).read().decode()))
                df.columns = ["date", i]; df["date"] = pd.to_datetime(df["date"])
                df[i] = pd.to_numeric(df[i], errors="coerce"); return df.set_index("date")[i].dropna()
            except Exception:
                time.sleep(2)
        return None
    cols = {}
    for i, nm in SER.items():
        s = fred(i)
        if s is not None and len(s) > 400:
            cols[nm] = s
    if len(cols) < 4:
        return None
    FX = pd.DataFrame(cols).resample("W").last().interpolate().dropna()
    base = FX.loc["1996-01-01":"1996-12-31"].mean()
    S = ((FX / base - 1.0) * 100.0).loc["1996-06-01":"1998-12-31"]
    return S.values, list(S.columns), 5e-2

# ------------------------------------------------------------------ diagnostics
def residuals(M, Phi, c):
    X = M[:-1]; Y = M[1:]
    return Y - (X @ Phi.T + c)

def effective_df_per_eq(M, ridge):
    """Ridge hat-trace: trace(X (X'X + ridge*(T-1)*P)^-1 X'), P penalizes all but the intercept.
    Upper bound on the true (non-negativity-constrained) df."""
    T = len(M); X = np.column_stack([np.ones(T - 1), M[:-1]])
    P = np.eye(X.shape[1]); P[0, 0] = 0.0
    G = X.T @ X + ridge * (T - 1) * P
    H_trace = np.trace(X @ np.linalg.solve(G, X.T))
    return float(H_trace)

def top_transmitter(M, ridge):
    Phi, c, Sig = L.fit_var_nonneg(M, ridge=ridge)
    _TO, _FROM, NET, _tot = L.connectedness(L.gfevd(Phi, Sig))
    return int(np.argmax(NET)), Phi, c, Sig, NET

def ledoit_wolf_cov(R):
    """Ledoit-Wolf linear shrinkage toward a scaled-identity target (self-contained, no sklearn)."""
    n, p = R.shape
    Rc = R - R.mean(0)
    S = (Rc.T @ Rc) / n
    mu = np.trace(S) / p
    target = mu * np.eye(p)
    d2 = np.sum((S - target) ** 2)
    b2 = 0.0
    for t in range(n):
        b2 += np.sum((np.outer(Rc[t], Rc[t]) - S) ** 2)
    b2 = b2 / (n ** 2)
    b2 = min(b2, d2)
    delta = b2 / d2 if d2 > 0 else 0.0
    return (1 - delta) * S + delta * target, float(delta)

def diagnose(name, M, names, ridge):
    T, N = M.shape
    p_params = N * (N + 1)                       # transition + intercept coefficients
    us, Phi, c, Sig, NET = top_transmitter(M, ridge)
    cond = float(np.linalg.cond(Sig))
    edf_eq = effective_df_per_eq(M, ridge)
    R = residuals(M, Phi, c)

    # (3) lambda sensitivity of the net-transmitter ranking
    lam_grid = [ridge / 10, ridge / 3.16, ridge, ridge * 3.16, ridge * 10]
    tops = []
    for lam in lam_grid:
        ui, *_ = top_transmitter(M, lam)
        tops.append(names[ui])
    lam_stable = len(set(tops)) == 1

    # (4) Ledoit-Wolf shrinkage GFEVD -> does the top transmitter change?
    Sig_lw, delta = ledoit_wolf_cov(R)
    _TO, _FROM, NET_lw, _tot = L.connectedness(L.gfevd(Phi, Sig_lw))
    us_lw = int(np.argmax(NET_lw))
    lw_stable = (us_lw == us)

    out = dict(twin=name, N=N, T=int(T), obs_per_eq=int(T - 1), params=p_params,
               p_ge_T=bool(p_params >= T), transmitter=names[us],
               residual_cov_condition_number=round(cond, 1),
               effective_df_per_eq=round(edf_eq, 2), max_df_per_eq=N + 1,
               lambda_grid=[round(x, 4) for x in lam_grid], top_across_lambda=tops,
               transmitter_stable_across_lambda=lam_stable,
               ledoit_wolf_shrinkage=round(delta, 3), top_under_ledoit_wolf=names[us_lw],
               verdict_stable_under_shrinkage=lw_stable)
    print(f"\n[{name}] N={N} T={T}  params={p_params} {'>=' if p_params>=T else '<'} T"
          f"  transmitter={names[us]}")
    print(f"  residual-cov condition number = {cond:,.0f}")
    print(f"  effective df / equation = {edf_eq:.1f} of max {N+1}  (obs/eq = {T-1})")
    print(f"  net-transmitter across lambda {[round(x,4) for x in lam_grid]}: {tops}  -> {'STABLE' if lam_stable else 'CHANGES'}")
    print(f"  Ledoit-Wolf shrinkage delta={delta:.3f}; top under shrinkage = {names[us_lw]}  -> {'unchanged' if lw_stable else 'CHANGES'}")
    return out

def main():
    twins = [("2008 equities", panel_equity), ("COVID-19", panel_covid), ("1997 Asian FX", panel_asia_fx)]
    results = []
    for name, builder in twins:
        try:
            got = builder()
        except Exception as e:  # noqa
            print(f"[{name}] panel build failed: {e}"); continue
        if got is None:
            print(f"[{name}] skipped (data unavailable, e.g. FRED unreachable)"); continue
        M, names, ridge = got
        results.append(diagnose(name, M, names, ridge))
    json.dump(results, open(BASE / "PROPOSED_df_diagnostics.json", "w"), indent=1)
    print("\n[wrote PROPOSED_df_diagnostics.json]")

if __name__ == "__main__":
    main()
