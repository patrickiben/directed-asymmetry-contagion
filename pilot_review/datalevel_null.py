"""
DATA-LEVEL DIRECTEDNESS NULL (referee request).

The paper's symmetrization null perturbs the *fitted* Phi (alpha*Phi + (1-alpha)*sym). Referees want a
DATA-LEVEL null: build, for each twin, a principled surrogate of the PANEL that
  (i)  PRESERVES each series' marginal distribution and its own autocorrelation, but
  (ii) DESTROYS directed (lead-lag) CROSS-series structure,
refit fit_var_nonneg(ridge=5e-2) on each surrogate, and ask whether the OBSERVED directedness of the
fitted kernel is beyond what this direction-destroyed null produces.

SURROGATE CHOICE -- independent per-series circular shift (primary):
  Each column j is circularly shifted by an independent random lag tau_j ~ Uniform{1..T-1}. A circular
  shift is a pure re-indexing of the SAME values, so it preserves the marginal EXACTLY and (up to the
  small wrap-around seam) the series' own autocorrelation. Because the columns are shifted by DIFFERENT
  lags, any lead-lag alignment between series -- the substrate of directed (Granger-type) coupling -- is
  destroyed, along with contemporaneous cross-correlation. This is the cleanest "direction-destroyed"
  null for non-negative, non-Gaussian stress series (no Gaussianization, unlike phase randomization).

  We also report a TIME-REVERSAL surrogate as a secondary check. Joint time reversal preserves marginals
  and the autocovariance function exactly and FLIPS the arrow of lead-lag; it is a weaker null (it
  transposes rather than erases directed structure) and is reported only for context.

DIRECTEDNESS STATISTIC D (two definitions, both reported):
  D_asym = ||Phi - Phi^T||_F / ||Phi||_F          (antisymmetry fraction; ties to the symmetrization null)
  D_net  = max_i |NET_i|  of the Diebold-Yilmaz net-connectedness (loudest transmitter/receiver imbalance)

For each system: refit on K>=300 surrogates -> null distribution of D -> one-sided p-value
  p = (1 + #{D_surrogate >= D_observed}) / (K + 1)        (is observed directedness BEYOND the null?)
and the percentile of the observed D within the null.

Run: /tmp/lsa_venv/bin/python3 datalevel_null.py
"""
import sys, json, io, urllib.request, time, zipfile
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "pilot_cross_tier"))
import lsa_capstone as L
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
EQDIR = BASE.parent / "pilot_3p46_equity"
NN = Path("/tmp/lsa_nn")
RIDGE = 5e-2
K = 300                      # surrogates per system (>=300 as requested)
H_GFEVD = 10                 # gfevd horizon (engine default)
SEED = 12345

matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.7, "savefig.dpi": 300})

# verdict class assigned in each system's own script (for colour + expectation)
VERDICT = {"asia97": "confirm", "smoke": "confirm", "flu": "refine", "flights": "falsify",
           "conflict": "null", "equity": "symmetric", "COVID": "confirm"}
VCOL = {"confirm": "#2e8b57", "refine": "#d4a017", "falsify": "#c00000",
        "null": "#7f7f7f", "symmetric": "#1f4e78"}

# ============================================================ directedness statistics
def D_stats(Phi, Sig):
    """Return (D_asym, D_net). D_asym = ||Phi-Phi^T||_F/||Phi||_F; D_net = max|DY net|."""
    fro = float(np.linalg.norm(Phi, "fro"))
    d_asym = float(np.linalg.norm(Phi - Phi.T, "fro") / fro) if fro > 1e-12 else 0.0
    th = L.gfevd(Phi, Sig, H=H_GFEVD)
    _, _, NET, _ = L.connectedness(th)
    d_net = float(np.max(np.abs(NET)))
    return d_asym, d_net

# ============================================================ surrogate generators
def surr_circshift(M, rng):
    """Independent per-series circular shift: column j rolled by tau_j ~ U{1..T-1}."""
    T, N = M.shape
    out = np.empty_like(M)
    for j in range(N):
        tau = int(rng.integers(1, T))          # 1..T-1 (avoid identity)
        out[:, j] = np.roll(M[:, j], tau)
    return out

def surr_timereverse(M):
    """Joint time reversal (all series reversed together)."""
    return M[::-1].copy()

# ============================================================ cached / refetched panel loaders
# 6 of 7 panels reproduce the cached Phi bit-exactly from cached data; asia97 FRED dropped a series, so
# we cache the original 9-currency panel to /tmp/lsa_nn/panel_asia97.csv when available and reuse it.
def load_flights():
    return pd.read_csv("/tmp/lsa_flights/flights_panel.csv", index_col=0).values

def load_conflict():
    return pd.read_csv("/tmp/lsa_conflict/conflict_panel.csv", index_col=0).values

def load_equity():
    idx = {"GSPC": "US", "FTSE": "UK", "GDAXI": "Germany", "FCHI": "France",
           "N225": "Japan", "HSI": "Hong Kong", "AEX": "Neth.", "BVSP": "Brazil"}
    def load(sym):
        d = json.load(open(EQDIR / "data" / f"{sym}.json"))["chart"]["result"][0]
        return pd.Series(d["indicators"]["quote"][0]["close"],
                         index=pd.to_datetime(d["timestamp"], unit="s")).dropna()
    P = pd.concat([load(s).rename(idx[s]) for s in idx], axis=1).ffill().dropna().resample("W").last().loc["2007-01-01":"2010-06-30"]
    return (-(100 * np.log(P).diff().dropna())).clip(lower=0).values

def load_covid():
    DROP = {"American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands", "Diamond Princess",
            "Grand Princess", "Puerto Rico"}
    d = pd.read_csv(BASE / "jhu_confirmed_US.csv"); dcols = [c for c in d.columns if "/" in c]
    st = d.groupby("Province_State")[dcols].sum().drop(index=[s for s in DROP if s in d["Province_State"].unique()], errors="ignore")
    st.columns = pd.to_datetime(st.columns)
    weekly = st.T.sort_index().resample("W").last().diff().clip(lower=0).dropna()
    top = weekly.sum().sort_values(ascending=False).head(14).index
    return (weekly[top] / 1000.0).loc["2020-03-01":"2022-06-30"].values

def load_smoke():
    cache = NN / "panel_smoke.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0).values
    STATES = ["New York", "New Jersey", "Pennsylvania", "Connecticut", "Massachusetts", "Rhode Island", "Vermont",
              "New Hampshire", "Maine", "Ohio", "Michigan", "Illinois", "Wisconsin", "Minnesota", "Indiana", "Maryland", "Virginia"]
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen("https://aqs.epa.gov/aqsweb/airdata/daily_88101_2023.zip", timeout=120).read()))
    csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
    raw = pd.read_csv(z.open(csv_name), usecols=["State Name", "Date Local", "Arithmetic Mean"])
    raw = raw[raw["State Name"].isin(STATES)]; raw["Date Local"] = pd.to_datetime(raw["Date Local"])
    daily = raw.groupby(["State Name", "Date Local"])["Arithmetic Mean"].mean().reset_index()
    P = daily.pivot(index="Date Local", columns="State Name", values="Arithmetic Mean").sort_index()
    P = P.loc["2023-05-01":"2023-07-31"].interpolate().dropna(axis=1, how="any")
    return P[[s for s in STATES if s in P.columns]].values

def load_flu():
    cache = NN / "panel_flu.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0).values
    STATES = ["ca", "tx", "fl", "ny", "pa", "il", "oh", "ga", "nc", "mi", "nj", "va", "wa", "az", "ma"]
    url = "https://api.delphi.cmu.edu/epidata/fluview/?regions=" + ",".join(STATES) + "&epiweeks=201040-201920"
    r = json.load(urllib.request.urlopen(url, timeout=60)); assert r["result"] == 1, r.get("message")
    df = pd.DataFrame(r["epidata"])[["region", "epiweek", "wili"]]
    M = df.pivot(index="epiweek", columns="region", values="wili").sort_index()
    wk = (M.index % 100); M = M[(wk >= 40) | (wk <= 20)]
    return M[[s for s in STATES if s in M.columns]].interpolate().dropna().values

def load_asia97():
    """Use the cached original 9-currency panel (panel_asia97.csv).
    The live FRED endpoint is the only non-reproducible source; if the cache is missing this raises so the
    system is recorded honestly rather than silently substituting a different (8-currency) panel."""
    cache = NN / "panel_asia97.csv"
    if not cache.exists():
        raise FileNotFoundError("panel_asia97.csv not cached and FRED unreachable this run "
                                "(cached 9x9 Phi/Sig exist, but the data-level null needs the raw panel)")
    return pd.read_csv(cache, index_col=0).values

LOADERS = [("asia97", load_asia97), ("smoke", load_smoke), ("flu", load_flu),
           ("flights", load_flights), ("conflict", load_conflict), ("equity", load_equity), ("COVID", load_covid)]

# ============================================================ run
def pctile_and_p(obs, null):
    null = np.asarray(null, float)
    pct = float(100.0 * np.mean(null < obs))                 # % of null strictly below observed
    p = float((1 + np.sum(null >= obs)) / (len(null) + 1))   # one-sided: obs beyond null?
    return pct, p

results = {}
panels = {}
print(f"DATA-LEVEL DIRECTEDNESS NULL  (K={K} per system, ridge={RIDGE}, primary surrogate = per-series circular shift)\n")
for name, loader in LOADERS:
    try:
        M = loader()
    except Exception as e:
        print(f"[{name:8s}] LOAD FAILED: {e!r}")
        results[name] = {"error": repr(e)}
        continue
    M = np.asarray(M, float)
    panels[name] = M
    Phi, c, Sig = L.fit_var_nonneg(M, ridge=RIDGE)
    # verify vs cached Phi where available
    cached = NN / f"Phi_{name}.npy"
    md = None
    if cached.exists():
        Pc = np.load(cached)
        if Pc.shape == Phi.shape:
            md = float(np.abs(Phi - Pc).max())
    d_asym_obs, d_net_obs = D_stats(Phi, Sig)

    # deterministic per-system seed (avoid Python's randomized hash() so runs reproduce exactly)
    name_seed = int.from_bytes(name.encode(), "little") % 100000
    rng = np.random.default_rng(SEED + name_seed)
    null_asym, null_net = [], []
    for k in range(K):
        Ms = surr_circshift(M, rng)
        Ps, cs, Ss = L.fit_var_nonneg(Ms, ridge=RIDGE)
        a, n = D_stats(Ps, Ss)
        null_asym.append(a); null_net.append(n)
    # time-reversal secondary (single surrogate -> we report its D, not a distribution)
    Mtr = surr_timereverse(M)
    Ptr, ctr, Str = L.fit_var_nonneg(Mtr, ridge=RIDGE)
    d_asym_tr, d_net_tr = D_stats(Ptr, Str)

    pct_a, p_a = pctile_and_p(d_asym_obs, null_asym)
    pct_n, p_n = pctile_and_p(d_net_obs, null_net)

    results[name] = dict(
        N=int(M.shape[1]), T=int(M.shape[0]), verdict=VERDICT[name], refit_maxdiff_vs_cached=md,
        D_asym=dict(observed=round(d_asym_obs, 4), null_mean=round(float(np.mean(null_asym)), 4),
                    null_sd=round(float(np.std(null_asym)), 4), null_p95=round(float(np.percentile(null_asym, 95)), 4),
                    observed_pctile=round(pct_a, 1), p_value=round(p_a, 4),
                    null=[round(float(x), 4) for x in null_asym], timereverse=round(d_asym_tr, 4)),
        D_net=dict(observed=round(d_net_obs, 3), null_mean=round(float(np.mean(null_net)), 3),
                   null_sd=round(float(np.std(null_net)), 3), null_p95=round(float(np.percentile(null_net, 95)), 3),
                   observed_pctile=round(pct_n, 1), p_value=round(p_n, 4),
                   null=[round(float(x), 3) for x in null_net], timereverse=round(d_net_tr, 3)),
    )
    md_s = "n/a" if md is None else f"{md:.1e}"
    print(f"[{name:8s}] N={M.shape[1]:2d} T={M.shape[0]:3d} ({VERDICT[name]:9s})  refit~cached:{md_s}")
    print(f"            D_asym obs={d_asym_obs:.3f}  null {np.mean(null_asym):.3f}+/-{np.std(null_asym):.3f}  "
          f"pctile={pct_a:5.1f}  p={p_a:.3f}   [revD={d_asym_tr:.3f}]")
    print(f"            D_net  obs={d_net_obs:6.2f}  null {np.mean(null_net):6.2f}+/-{np.std(null_net):5.2f}  "
          f"pctile={pct_n:5.1f}  p={p_n:.3f}   [revD={d_net_tr:.2f}]")

# ============================================================ save json
good = {k: v for k, v in results.items() if "error" not in v}
json.dump(dict(K=K, ridge=RIDGE, gfevd_H=H_GFEVD, seed=SEED,
               primary_surrogate="independent per-series circular shift",
               secondary_surrogate="joint time-reversal",
               D_definitions={"D_asym": "||Phi-Phi^T||_F/||Phi||_F", "D_net": "max_i |DY net_i|"},
               p_value_def="one-sided (1+#{null>=obs})/(K+1); small p = observed directedness beyond null",
               systems=results),
          open(BASE / "datalevel_null.json", "w"), indent=2)
print("\nsaved -> datalevel_null.json")

# ============================================================ figure
order = [k for k in ["asia97", "smoke", "equity", "COVID", "flu", "flights", "conflict"] if k in good]
fig, axes = plt.subplots(2, len(order), figsize=(2.0 * len(order), 5.4), sharey="row")
if len(order) == 1:
    axes = axes.reshape(2, 1)

for col, name in enumerate(order):
    r = good[name]; col_c = VCOL[r["verdict"]]
    for row, key, lab in [(0, "D_asym", r"$D_{asym}=\|\Phi-\Phi^\top\|_F/\|\Phi\|_F$"),
                          (1, "D_net", r"$D_{net}=\max_i|NET_i|$")]:
        ax = axes[row, col]
        nd = np.array(good[name][key]["null"]); obs = good[name][key]["observed"]
        ax.hist(nd, bins=22, color="#bbbbbb", edgecolor="white", lw=0.3, alpha=0.9)
        ax.axvline(obs, color=col_c, lw=2.0, zorder=5)
        p = good[name][key]["p_value"]; pct = good[name][key]["observed_pctile"]
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "ns"))
        ax.text(0.96, 0.95, f"p={p:.3f}\n{sig}", transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, fontweight="bold", color=col_c,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col_c, lw=0.8, alpha=0.9))
        ax.tick_params(labelsize=6)
        if col == 0:
            ax.set_ylabel(("antisymmetry\n" if row == 0 else "max |net|\n") + "surrogate count", fontsize=7.5)
        if row == 0:
            ax.set_title(f"{name}\n({r['verdict']})", fontsize=8, color=col_c, fontweight="bold")
missing = [k for k in VERDICT if k not in good]
miss_note = ("  (asia97 omitted: FRED unreachable this run)" if "asia97" in missing else "")
fig.suptitle("Data-level directedness null — observed fitted-kernel directedness (coloured line) vs the "
             "direction-destroyed per-series circular-shift surrogate null (grey)\n"
             "Top row D_asym = antisymmetry fraction (saturated/noisy); bottom row D_net = max |DY net imbalance| "
             "(the transmitter statistic). One-sided p = (1+#{null>=obs})/(K+1)." + miss_note,
             fontsize=9.5, fontweight="bold", y=1.005)
# x-axis labels on bottom row
for col, name in enumerate(order):
    axes[0, col].set_xlabel(r"$D_{asym}$", fontsize=7); axes[1, col].set_xlabel(r"$D_{net}$", fontsize=7)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(BASE / "datalevel_null.pdf", bbox_inches="tight")
fig.savefig(BASE / "datalevel_null.png", dpi=200, bbox_inches="tight")
print("figure -> datalevel_null.pdf / .png")

# ============================================================ compact summary table
print("\n================ SUMMARY (D_asym primary) ================")
print(f"{'system':9s} {'verdict':9s} {'D_obs':>7s} {'null_mn':>7s} {'pctile':>6s} {'p':>6s}")
for name in order:
    r = good[name]["D_asym"]
    print(f"{name:9s} {good[name]['verdict']:9s} {r['observed']:7.3f} {r['null_mean']:7.3f} "
          f"{r['observed_pctile']:6.1f} {r['p_value']:6.3f}")
