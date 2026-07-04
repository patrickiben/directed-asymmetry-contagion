"""
PROPOSED (addresses the referee finding P1 -- NOT wired into the manuscript).

The SI stationarity table (tab:stationarity) reports ADF/KPSS only for the five transfer twins.
The 2008-equity and COVID-19 networks -- which carry primary confirmatory weight -- are absent,
and COVID is elsewhere described as non-stationary while its static full-sample GFEVD is used as a
confirmatory anchor. This script computes the SAME ADF/KPSS battery for the exact series each of
those two VARs is fit on, so the two missing rows can be added to the table with REAL numbers.

Per series: ADF (H0: unit root; p<0.05 => stationary in levels) and KPSS (H0: stationarity; p<0.05
=> non-stationary). "Both tests" = ADF-stationary AND KPSS-stationary. "Needs differencing" =
ADF fails to reject a unit root. Matches the reporting of the existing five-twin table.

Self-contained and OFFLINE (bundled equity CSV + jhu_confirmed_US.csv). Needs statsmodels.
Run:  python3 PROPOSED_stationarity_equity_covid.py     -> PROPOSED_stationarity_equity_covid.json
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

warnings.simplefilter("ignore")   # KPSS interpolation notices when the stat is outside the p-value table
BASE = Path(__file__).resolve().parent
ALPHA = 0.05

def equity_stress():
    P = pd.read_csv(BASE.parent / "pilot_3p46_equity" / "equity_weekly_close_2007_2010.csv",
                    parse_dates=["week_ending"]).set_index("week_ending")
    P = P.resample("W").last().loc["2007-01-01":"2010-06-30"].dropna(how="any")
    return (-(100 * np.log(P).diff().dropna())).clip(lower=0)          # decline-stress, the modeled series

def covid_panel():
    DROP = {"American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands",
            "Diamond Princess", "Grand Princess", "Puerto Rico"}
    d = pd.read_csv(BASE / "jhu_confirmed_US.csv"); dc = [c for c in d.columns if "/" in c]
    st = d.groupby("Province_State")[dc].sum().drop(
        index=[s for s in DROP if s in d["Province_State"].unique()], errors="ignore")
    st.columns = pd.to_datetime(st.columns)
    wk = st.T.sort_index().resample("W").last().diff().clip(lower=0).dropna()
    top = wk.sum().sort_values(ascending=False).head(14).index
    return (wk[top] / 1000.0).loc["2020-03-01":"2022-06-30"]           # exactly covid_twin()'s W

def test_series(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if np.ptp(x) == 0:                      # constant series: degenerate, treat as non-stationary/undefined
        return dict(adf_p=None, kpss_p=None, adf_stationary=False, kpss_stationary=False, both=False, note="constant")
    adf_p = float(adfuller(x, autolag="AIC", regression="c")[1])
    kpss_p = float(kpss(x, regression="c", nlags="auto")[1])
    adf_s = adf_p < ALPHA                    # reject unit root -> stationary
    kpss_s = kpss_p >= ALPHA                 # fail to reject stationarity -> stationary
    return dict(adf_p=round(adf_p, 4), kpss_p=round(kpss_p, 4),
                adf_stationary=bool(adf_s), kpss_stationary=bool(kpss_s), both=bool(adf_s and kpss_s), note="")

def summarize(name, panel):
    rows = {c: test_series(panel[c].values) for c in panel.columns}
    N = len(rows)
    adf_stat = sum(r["adf_stationary"] for r in rows.values())
    both = sum(r["both"] for r in rows.values())
    needs_diff = sum((not r["adf_stationary"]) for r in rows.values())
    print(f"\n{name}: N={N}, T={len(panel)}  ({panel.index[0].date()}..{panel.index[-1].date()})")
    print(f"  ADF-stationary: {adf_stat}/{N}   Both tests: {both}/{N}   Needs differencing: {needs_diff}/{N}")
    for c, r in rows.items():
        flag = "stat" if r["adf_stationary"] else "UNIT-ROOT"
        print(f"    {c:22.22s} ADF p={r['adf_p']}  KPSS p={r['kpss_p']}  -> {flag}{' (both)' if r['both'] else ''}")
    return dict(name=name, N=N, T=len(panel), adf_stationary=adf_stat, both_tests=both,
                needs_differencing=needs_diff, per_series=rows)

def main():
    out = {"alpha": ALPHA,
           "method": "ADF (H0 unit root, autolag=AIC, regression=c) + KPSS (H0 stationarity, regression=c)",
           "networks": [summarize("2008 equities (decline-stress)", equity_stress()),
                        summarize("COVID-19 (weekly new cases)", covid_panel())]}
    json.dump(out, open(BASE / "PROPOSED_stationarity_equity_covid.json", "w"), indent=1)
    print("\n[wrote PROPOSED_stationarity_equity_covid.json]")

if __name__ == "__main__":
    main()
