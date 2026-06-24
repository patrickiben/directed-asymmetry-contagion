"""
TWO analyses building on nonnormality_predictor.py machinery (reused, not re-derived):

(A) ADVANTAGE-vs-GAP REGRESSION
    Per system: structural gap = NET[transmitter] - NET[loudest], where
      transmitter = argmax(DY NET)  (net spillover transmitter)
      loudest     = argmax(mean stress)  (the 'greedy' target)
    Regress controller advantage (transmitter% - greedy%, recomputed consistently via interdict_adv)
    on the gap across all 7 systems (Pearson + Spearman). Figure: advantage vs gap, labelled, coloured
    by verdict, fit line + r.
    HONEST FLAG: when transmitter == loudest the gap == 0 AND advantage ~ 0 by construction (flu),
    so the relationship is partly STRUCTURAL -> operationalisation/illustration of the law, NOT an
    independent test.

(B) GROUND-TRUTH RECOVERY TABLE
    Per system: DY net-transmitter node vs the INDEPENDENTLY-DOCUMENTED origin, match? (web-confirmed).

Reuses loaders + interdict_adv from nonnormality_predictor.py and the cached kernels in /tmp/lsa_nn.
Run: /tmp/lsa_venv/bin/python3 groundtruth_gap.py
"""
import sys, json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent / "pilot_cross_tier"))
import lsa_capstone as L
# import the SHARED machinery exactly as defined (loaders, interdiction, fit)
import nonnormality_predictor as NN

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RIDGE = 5e-2
NNDIR = Path("/tmp/lsa_nn")

# node-name vectors per system (in the SAME column order the loaders produce)
def names_asia97():
    SER = {"DEXTHUS": "Thai", "DEXKOUS": "Korea", "DEXMAUS": "Malay", "DEXSIUS": "Singa", "DEXJPUS": "Japan",
           "DEXTAUS": "Taiwan", "DEXHKUS": "HK", "DEXINUS": "India", "DEXCHUS": "China"}
    return list(SER.values())  # loader keeps only series len>400; we re-derive from M shape below if needed
def names_smoke():
    return ["New York", "New Jersey", "Pennsylvania", "Connecticut", "Massachusetts", "Rhode Island", "Vermont",
            "New Hampshire", "Maine", "Ohio", "Michigan", "Illinois", "Wisconsin", "Minnesota", "Indiana",
            "Maryland", "Virginia"]
def names_flu():
    return ["ca", "tx", "fl", "ny", "pa", "il", "oh", "ga", "nc", "mi", "nj", "va", "wa", "az", "ma"]
def names_equity():
    return ["US", "UK", "Germany", "France", "Japan", "Hong Kong", "Neth.", "Brazil"]

# systems whose node order can be read straight off a cached csv / loader's columns
import pandas as pd
def names_flights():
    return list(pd.read_csv("/tmp/lsa_flights/flights_panel.csv", index_col=0).columns)
def names_conflict():
    return list(pd.read_csv("/tmp/lsa_conflict/conflict_panel.csv", index_col=0).columns)
def names_covid():
    # reconstruct EXACTLY as load_covid picks the top-14 states (same code path)
    DROP = {"American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands", "Diamond Princess",
            "Grand Princess", "Puerto Rico"}
    d = pd.read_csv(BASE / "jhu_confirmed_US.csv"); dcols = [c for c in d.columns if "/" in c]
    st = d.groupby("Province_State")[dcols].sum().drop(index=[s for s in DROP if s in d["Province_State"].unique()], errors="ignore")
    st.columns = pd.to_datetime(st.columns)
    weekly = st.T.sort_index().resample("W").last().diff().clip(lower=0).dropna()
    top = weekly.sum().sort_values(ascending=False).head(14).index
    W = (weekly[top] / 1000.0).loc["2020-03-01":"2022-06-30"]
    return list(W.columns)

# for asia97/equity the loader can silently drop a column; reconstruct names that match M's columns
def names_asia97_safe(M):
    full = names_asia97()
    # loader builds columns in dict-iteration order, dropping series that failed/short.
    # Safest: re-run the loader path is not needed; M.shape[1] tells us how many survived.
    # We reload the FX frame to get the actual surviving names.
    SER = {"DEXTHUS": "Thai", "DEXKOUS": "Korea", "DEXMAUS": "Malay", "DEXSIUS": "Singa", "DEXJPUS": "Japan",
           "DEXTAUS": "Taiwan", "DEXHKUS": "HK", "DEXINUS": "India", "DEXCHUS": "China"}
    import io, urllib.request, time
    def fred(i, tries=3):
        u = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={i}&cosd=1996-01-01&coed=1998-12-31"
        for _ in range(tries):
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
        if s is not None and len(s) > 400: cols[nm] = nm
    surv = list(cols.values())
    if len(surv) == M.shape[1]:
        return surv
    return full[:M.shape[1]]

NAMEFN = {
    "asia97": None,  # handled via safe reload
    "smoke": names_smoke, "flu": names_flu, "flights": names_flights,
    "conflict": names_conflict, "equity": names_equity, "covid": names_covid,
}

VERDICT = {  # equity relabelled 'symmetric' in NN; map to a verdict class for colouring (treat as confirm-family)
    "asia97": "confirm", "smoke": "confirm", "flu": "refine", "flights": "falsify",
    "conflict": "null", "equity": "symmetric", "COVID": "confirm",
}

SYSTEMS = [
    ("asia97",   NN.load_asia97),
    ("smoke",    NN.load_smoke),
    ("flu",      NN.load_flu),
    ("flights",  NN.load_flights),
    ("conflict", NN.load_conflict),
    ("equity",   NN.load_equity),
    ("COVID",    NN.load_covid),
]

rows = []
gt_rows = []
for name, loader in SYSTEMS:
    M, S0 = loader()
    Phi0, c0, Sig0 = L.fit_var_nonneg(M, ridge=RIDGE)
    # NET from the SAME pipeline interdict_adv uses
    TO, FROM, NET, total = L.connectedness(L.gfevd(Phi0, Sig0))
    # node names
    if name == "asia97":
        nm = names_asia97_safe(M)
    else:
        nm = NAMEFN[name.lower()]()
    if len(nm) != M.shape[1]:
        nm = [f"n{i}" for i in range(M.shape[1])]

    transmitter_i = int(np.argmax(NET))
    # 'loudest' = greedy target = argmax mean stress.  S0 is the seeded initial stress used by the
    # simulator (== mean stress, clipped to >=0.5 for most loaders); use M.mean(0) as the canonical
    # mean-stress ranking (matches the 'loudest node' definition).
    meanstress = M.mean(0)
    loudest_i = int(np.argmax(meanstress))

    gap = float(NET[transmitter_i] - NET[loudest_i])  # structural gap (>=0 by construction)

    # advantage: recompute consistently for ALL systems via the shared interdiction (same NET, twin)
    tr, gr = NN.interdict_adv(Phi0, c0, Sig0, S0, seed0=20)
    adv = float(tr - gr)

    rows.append(dict(system=name, transmitter=nm[transmitter_i], loudest=nm[loudest_i],
                     same_node=bool(transmitter_i == loudest_i),
                     NET_transmitter=round(float(NET[transmitter_i]), 4),
                     NET_loudest=round(float(NET[loudest_i]), 4),
                     gap=round(gap, 4), advantage=round(adv, 2),
                     transmitter_pct=round(tr, 2), greedy_pct=round(gr, 2),
                     verdict=VERDICT[name]))
    gt_rows.append(dict(system=name, dy_transmitter=nm[transmitter_i], verdict=VERDICT[name]))
    print(f"[{name:8s}] transmitter={nm[transmitter_i]:14s} loudest={nm[loudest_i]:14s} "
          f"same={transmitter_i==loudest_i!s:5s} gap={gap:+7.3f} adv={adv:+6.2f} "
          f"(tr={tr:+.1f} gr={gr:+.1f})  [{VERDICT[name]}]")

# ===================== regression: advantage ~ gap =====================
gap_arr = np.array([r["gap"] for r in rows])
adv_arr = np.array([r["advantage"] for r in rows])
pr, pp = pearsonr(gap_arr, adv_arr)
sr, sp = spearmanr(gap_arr, adv_arr)
b1, b0 = np.polyfit(gap_arr, adv_arr, 1)
print(f"\n=== ADVANTAGE ~ GAP (n={len(rows)}) ===")
print(f"Pearson r = {pr:+.3f} (p={pp:.3f})   Spearman rho = {sr:+.3f} (p={sp:.3f})")
print(f"OLS: advantage = {b0:+.2f} + {b1:+.2f} * gap")
# robustness: drop the structural-zero (flu) point
mask = gap_arr > 1e-6
pr_x, pp_x = pearsonr(gap_arr[mask], adv_arr[mask]) if mask.sum() > 2 else (np.nan, np.nan)
sr_x, sp_x = spearmanr(gap_arr[mask], adv_arr[mask]) if mask.sum() > 2 else (np.nan, np.nan)
print(f"excl gap==0 systems (n={int(mask.sum())}): Pearson r={pr_x:+.3f} (p={pp_x:.3f}) Spearman rho={sr_x:+.3f}")

# ===================== figure =====================
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "savefig.dpi": 300})
VCOL = {"confirm": "#2e8b57", "refine": "#d4a017", "falsify": "#c00000",
        "null": "#7f7f7f", "symmetric": "#1f4e78"}
fig, ax = plt.subplots(figsize=(6.6, 5.0))
for r in rows:
    ax.scatter(r["gap"], r["advantage"], s=85, color=VCOL[r["verdict"]], edgecolor="k", lw=0.7, zorder=3)
    ax.annotate(r["system"], (r["gap"], r["advantage"]), textcoords="offset points",
                xytext=(7, 4), fontsize=8.5)
xs = np.linspace(gap_arr.min(), gap_arr.max(), 50)
ax.plot(xs, b0 + b1 * xs, color="#444", lw=1.3, ls="--", zorder=2)
ax.set_xlabel("structural gap  NET[transmitter] - NET[loudest]")
ax.set_ylabel("controller advantage  (transmitter - greedy, pts)")
ax.set_title("Advantage vs structural gap (operationalisation of the law)")
ax.grid(alpha=.25)
ax.text(0.04, 0.96, f"Pearson r = {pr:+.2f} (p={pp:.2f})\nSpearman $\\rho$ = {sr:+.2f} (p={sp:.2f})",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#888", lw=.7))
handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="k",
                  markersize=8, label=v) for v, c in VCOL.items()]
ax.legend(handles=handles, loc="lower right", fontsize=7.5, title="verdict", title_fontsize=8)
fig.tight_layout()
fig.savefig(BASE / "groundtruth_gap.pdf", bbox_inches="tight")
fig.savefig(BASE / "groundtruth_gap.png", dpi=200, bbox_inches="tight")
print("figure -> groundtruth_gap.pdf/.png")

# ===================== save json (analysis A; B filled with documented origins below) =====================
out = dict(
    analysis_A_advantage_vs_gap=dict(
        ridge=RIDGE, n_systems=len(rows),
        per_system=rows,
        regression=dict(pearson_r=round(float(pr), 4), pearson_p=round(float(pp), 4),
                        spearman_rho=round(float(sr), 4), spearman_p=round(float(sp), 4),
                        ols_intercept=round(float(b0), 4), ols_slope=round(float(b1), 4),
                        excl_gap0=dict(n=int(mask.sum()),
                                       pearson_r=None if np.isnan(pr_x) else round(float(pr_x), 4),
                                       pearson_p=None if np.isnan(pp_x) else round(float(pp_x), 4),
                                       spearman_rho=None if np.isnan(sr_x) else round(float(sr_x), 4))),
        honest_flag=("Partly structural: when transmitter==loudest the gap==0 and advantage~0 by "
                     "construction (flu). This is an operationalisation/illustration of the law, NOT "
                     "an independent test."),
    ),
    analysis_B_groundtruth_recovery=dict(per_system=gt_rows),  # documented origins appended in step 2
)
json.dump(out, open(BASE / "groundtruth_gap.json", "w"), indent=2)
print("partial json -> groundtruth_gap.json (analysis B origins appended after web confirmation)")
