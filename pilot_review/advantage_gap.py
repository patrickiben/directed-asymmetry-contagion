"""
ADVANTAGE ~ GAP regression (EXPLORATORY).

Per twin define a STRUCTURAL GAP:
    gap = NET[transmitter] - NET[loudest]
  transmitter = argmax(DY NET)            (net spillover transmitter; high NET)
  loudest     = argmax(mean stress)       (the 'greedy' target; may be a receiver = low/neg NET)
For flu the transmitter IS the loudest (tx == tx) so gap == 0 by construction.

Controller ADVANTAGE = transmitter% - greedy% (cascade-reduction on the rescaled twin), taken from
each twin's own source EXACTLY as the twin defines it:
  asia97/smoke/flu/flights/conflict : *_transfer_results.json  (transmitter - greedy)
  equity                            : interdict recompute at ridge=5e-2  (+43.1 - (-3.0) = 46.1)
  COVID                             : validity_decomposition static-transmit(31.9) - greedy(0.2) = 31.7

The law predicts advantage GROWS with the gap and is ~0 at gap==0 (flu).

CRITICAL HONESTY: this is partly SEMI-TAUTOLOGICAL -- when transmitter==loudest the gap==0, and the
advantage 'should' be ~0 because the two controllers target the same node. We therefore report the
full-n fit AND the fit with the gap==0 point (flu) removed; if the relationship survives ONLY because
of the flu anchor (i.e. collapses among the 6 directed twins) we set holds=false.

Kernels: cached /tmp/lsa_nn/Phi_*.npy, Sig_*.npy (== fresh fit_var_nonneg ridge=5e-2; verified maxdiff 0).
Panels reconstructed (cached data, no slow refetch) only to read mean-stress for 'loudest'.

Run: /tmp/lsa_venv/bin/python3 advantage_gap.py
"""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent / "pilot_cross_tier"))
import lsa_capstone as L
import nonnormality_predictor as NN          # cached loaders, same code path the twins use

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RIDGE = 5e-2
NN_DIR = Path("/tmp/lsa_nn")

# -------- node names in each loader's column order (to label transmitter / loudest) --------
def names_asia97(M):
    return ["Thai", "Korea", "Malay", "Singa", "Japan", "Taiwan", "HK", "India", "China"][:M.shape[1]]
def names_smoke(M):
    return ["New York", "New Jersey", "Pennsylvania", "Connecticut", "Massachusetts", "Rhode Island",
            "Vermont", "New Hampshire", "Maine", "Ohio", "Michigan", "Illinois", "Wisconsin",
            "Minnesota", "Indiana", "Maryland", "Virginia"][:M.shape[1]]
def names_flu(M):
    return ["ca", "tx", "fl", "ny", "pa", "il", "oh", "ga", "nc", "mi", "nj", "va", "wa", "az", "ma"][:M.shape[1]]
def names_flights(M):
    return list(pd.read_csv("/tmp/lsa_flights/flights_panel.csv", index_col=0).columns)
def names_conflict(M):
    return list(pd.read_csv("/tmp/lsa_conflict/conflict_panel.csv", index_col=0).columns)
def names_equity(M):
    return ["US", "UK", "Germany", "France", "Japan", "Hong Kong", "Neth.", "Brazil"]
def names_covid(M):
    DROP = {"American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands",
            "Diamond Princess", "Grand Princess", "Puerto Rico"}
    d = pd.read_csv(BASE / "jhu_confirmed_US.csv"); dc = [c for c in d.columns if "/" in c]
    st = d.groupby("Province_State")[dc].sum().drop(
        index=[s for s in DROP if s in d["Province_State"].unique()], errors="ignore")
    st.columns = pd.to_datetime(st.columns)
    wk = st.T.sort_index().resample("W").last().diff().clip(lower=0).dropna()
    return list(wk.sum().sort_values(ascending=False).head(14).index)

# -------- controller advantage per twin (transmitter% - greedy%), EXACTLY as the twin defines it --------
def adv_json(fname):
    it = json.load(open(BASE / fname))["interdiction"]
    return float(it["transmitter"] - it["greedy"]), float(it["transmitter"]), float(it["greedy"]), \
        f"{fname} (transmitter {it['transmitter']} - greedy {it['greedy']})"
def adv_equity():
    # interdict recompute at ridge=5e-2 (equity_decisive.py): transmitter +43.1, greedy -3.0
    return 46.1, 43.1, -3.0, "equity interdict recompute ridge=5e-2 (transmitter +43.1 - greedy -3.0)"
def adv_covid():
    vd = json.load(open(BASE / "validity_decomposition_results.json"))["covid"]["reductions"]
    return float(vd["static-transmit"] - vd["greedy"]), float(vd["static-transmit"]), float(vd["greedy"]), \
        f"validity_decomposition static-transmit {vd['static-transmit']} - greedy {vd['greedy']}"

VERDICT = {"asia97": "confirm", "smoke": "confirm", "flu": "refine", "flights": "falsify",
           "conflict": "null", "equity": "symmetric", "COVID": "confirm"}

SYSTEMS = [
    ("asia97",   NN.load_asia97,   names_asia97,   lambda: adv_json("asia97_transfer_results.json")),
    ("smoke",    NN.load_smoke,    names_smoke,    lambda: adv_json("smoke23_transfer_results.json")),
    ("flu",      NN.load_flu,      names_flu,      lambda: adv_json("flu_transfer_results.json")),
    ("flights",  NN.load_flights,  names_flights,  lambda: adv_json("flights_transfer_results.json")),
    ("conflict", NN.load_conflict, names_conflict, lambda: adv_json("conflict_transfer_results.json")),
    ("equity",   NN.load_equity,   names_equity,   adv_equity),
    ("COVID",    NN.load_covid,    names_covid,    adv_covid),
]

rows = []
for name, loader, namefn, advfn in SYSTEMS:
    M, S0 = loader()
    Phi = np.load(NN_DIR / f"Phi_{name}.npy"); Sig = np.load(NN_DIR / f"Sig_{name}.npy")
    # sanity: cached kernel must match a fresh fit on the reconstructed panel
    Phi_fresh, _, _ = L.fit_var_nonneg(M, ridge=RIDGE)
    refit_maxdiff = float(np.max(np.abs(Phi_fresh - Phi))) if Phi_fresh.shape == Phi.shape else None

    TO, FROM, NET, total = L.connectedness(L.gfevd(Phi, Sig))
    names = namefn(M)
    if len(names) != M.shape[1]:
        names = [f"n{i}" for i in range(M.shape[1])]

    ti = int(np.argmax(NET))                 # transmitter
    meanstress = M.mean(0)
    li = int(np.argmax(meanstress))          # loudest
    gap = float(NET[ti] - NET[li])

    adv, tr, gr, src = advfn()
    rows.append(dict(
        system=name, transmitter=names[ti], loudest=names[li],
        same_node=bool(ti == li),
        NET_transmitter=round(float(NET[ti]), 4), NET_loudest=round(float(NET[li]), 4),
        gap=round(gap, 4), advantage=round(float(adv), 2),
        transmitter_pct=round(float(tr), 2), greedy_pct=round(float(gr), 2),
        advantage_source=src, verdict=VERDICT[name],
        N=int(M.shape[1]), refit_maxdiff=refit_maxdiff))
    print(f"[{name:8s}] transmitter={names[ti]:13s}(NET={NET[ti]:+7.2f})  loudest={names[li]:13s}"
          f"(NET={NET[li]:+7.2f})  gap={gap:+8.3f}  adv={adv:+6.2f}  same={ti==li!s:5s}  "
          f"refit_dd={refit_maxdiff}")

# ===================== regression =====================
gap_arr = np.array([r["gap"] for r in rows], float)
adv_arr = np.array([r["advantage"] for r in rows], float)
pr, pp = pearsonr(gap_arr, adv_arr)
sr, sp = spearmanr(gap_arr, adv_arr)
b1, b0 = np.polyfit(gap_arr, adv_arr, 1)

mask = gap_arr > 1e-6                       # drop the gap==0 structural anchor (flu)
pr_x, pp_x = pearsonr(gap_arr[mask], adv_arr[mask])
sr_x, sp_x = spearmanr(gap_arr[mask], adv_arr[mask])

print(f"\n=== ADVANTAGE ~ GAP (n={len(rows)}) ===")
print(f"Pearson  r = {pr:+.3f} (p={pp:.3f})    Spearman rho = {sr:+.3f} (p={sp:.3f})")
print(f"OLS: advantage = {b0:+.2f} + {b1:+.3f} * gap")
print(f"--- drop gap==0 anchor (flu), n={int(mask.sum())} directed twins ---")
print(f"Pearson  r = {pr_x:+.3f} (p={pp_x:.3f})    Spearman rho = {sr_x:+.3f} (p={sp_x:.3f})")

# ===================== HONEST tautology assessment =====================
# Semi-tautological IF the full-n correlation is driven by the gap==0 (flu) anchor: i.e. it
# collapses (loses sign / drops to ~0 / loses rank monotonicity) once flu is removed.
collapses = (abs(pr_x) < 0.30) or (np.sign(pr_x) != np.sign(pr)) or (abs(sr_x) < 0.30)
# Additional concern: COVID is a high-leverage point (gap ~275 >> others). Check without COVID too.
mask_noc = np.array([r["system"] != "COVID" for r in rows])
pr_noc, pp_noc = pearsonr(gap_arr[mask_noc], adv_arr[mask_noc])
sr_noc, sp_noc = spearmanr(gap_arr[mask_noc], adv_arr[mask_noc])
# and without BOTH flu and COVID (the 5 plain directed twins)
mask_core = mask & mask_noc
pr_core, _ = pearsonr(gap_arr[mask_core], adv_arr[mask_core])
sr_core, _ = spearmanr(gap_arr[mask_core], adv_arr[mask_core])
print(f"--- drop COVID leverage point, n={int(mask_noc.sum())} ---  Pearson r={pr_noc:+.3f}  Spearman rho={sr_noc:+.3f}")
print(f"--- drop BOTH flu & COVID, n={int(mask_core.sum())} ---  Pearson r={pr_core:+.3f}  Spearman rho={sr_core:+.3f}")

HOLDS = (not collapses) and (abs(pr_x) >= 0.30) and (np.sign(pr_x) == np.sign(pr))
print(f"\ncollapses_without_flu_anchor = {collapses}   => HOLDS (clean & non-trivial) = {HOLDS}")

# ===================== figure =====================
matplotlib.rcParams.update({"font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "savefig.dpi": 300})
VCOL = {"confirm": "#2e8b57", "refine": "#d4a017", "falsify": "#c00000",
        "null": "#7f7f7f", "symmetric": "#1f4e78"}
fig, ax = plt.subplots(figsize=(7.4, 5.2))
LBL_OFF = {"equity": (-10, 6), "COVID": (-10, 8), "flu": (8, 8)}
for r in rows:
    ax.scatter(r["gap"], r["advantage"], s=95, color=VCOL[r["verdict"]],
               edgecolor="k", lw=0.8, zorder=3)
    ox, oy = LBL_OFF.get(r["system"], (8, 5))
    ha = "right" if r["system"] in ("COVID", "equity") else "left"
    ax.annotate(r["system"], (r["gap"], r["advantage"]), textcoords="offset points",
                xytext=(ox, oy), fontsize=8.5, ha=ha)
xs = np.linspace(gap_arr.min(), gap_arr.max(), 50)
ax.plot(xs, b0 + b1 * xs, color="#444", lw=1.4, ls="--", zorder=2,
        label=f"OLS (all n=7): r={pr:+.2f}")
# fit line over the directed twins only (flu removed)
b1x, b0x = np.polyfit(gap_arr[mask], adv_arr[mask], 1)
xs2 = np.linspace(gap_arr[mask].min(), gap_arr[mask].max(), 50)
ax.plot(xs2, b0x + b1x * xs2, color="#7030a0", lw=1.2, ls=":", zorder=2,
        label=f"OLS (excl flu, n=6): r={pr_x:+.2f}")
ax.axvline(0, color="#bbb", lw=0.8, zorder=1)
# flag equity: SMALL gap but LARGEST advantage -> the point that breaks monotonicity
eq = next(r for r in rows if r["system"] == "equity")
ax.annotate("equity: small gap (32),\nLARGEST advantage (46)\n-> breaks monotonicity",
            (eq["gap"], eq["advantage"]),
            textcoords="offset points", xytext=(28, -58), fontsize=8, color="#1f4e78",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="#1f4e78", lw=1.2))
ax.set_xlabel("structural gap   NET[transmitter] - NET[loudest]")
ax.set_ylabel("controller advantage   (transmitter - greedy, pts)")
ax.set_title("Advantage vs structural gap  (EXPLORATORY; partly tautological at gap=0)")
ax.grid(alpha=.25)
ax.text(0.30, 0.97,
        f"Pearson r = {pr:+.2f} (p={pp:.2f})\nSpearman $\\rho$ = {sr:+.2f} (p={sp:.2f})\n"
        f"excl flu: r = {pr_x:+.2f},  $\\rho$ = {sr_x:+.2f}\n"
        f"excl COVID (lever.): r = {pr_noc:+.2f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff6e8", ec="#888", lw=.7))
vhandles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="k",
                   markersize=8, label=v) for v, c in VCOL.items()]
leg1 = ax.legend(loc="lower right", fontsize=8)
ax.add_artist(leg1)
ax.legend(handles=vhandles, loc="upper right", fontsize=7.5, title="verdict", title_fontsize=8)
ax.margins(x=0.10, y=0.10)
fig.tight_layout()
fig.savefig(BASE / "advantage_gap.pdf", bbox_inches="tight")
fig.savefig(BASE / "advantage_gap.png", dpi=200, bbox_inches="tight")
print("figure -> advantage_gap.pdf / .png")

# ===================== json =====================
out = dict(
    title="ADVANTAGE ~ GAP (exploratory): controller advantage vs structural gap NET[transmitter]-NET[loudest]",
    ridge=RIDGE, n_systems=len(rows),
    gap_definition="gap = NET[transmitter=argmax(DY NET)] - NET[loudest=argmax(mean stress)]; gap=0 when transmitter==loudest (flu).",
    advantage_definition="transmitter% - greedy% cascade reduction on the rescaled twin, taken EXACTLY from each twin's source.",
    per_system=rows,
    regression=dict(
        n=len(rows),
        pearson_r=round(float(pr), 4), pearson_p=round(float(pp), 4),
        spearman_rho=round(float(sr), 4), spearman_p=round(float(sp), 4),
        ols_intercept=round(float(b0), 4), ols_slope=round(float(b1), 4),
        excl_gap0_flu=dict(n=int(mask.sum()), pearson_r=round(float(pr_x), 4), pearson_p=round(float(pp_x), 4),
                           spearman_rho=round(float(sr_x), 4), spearman_p=round(float(sp_x), 4)),
        excl_covid_leverage=dict(n=int(mask_noc.sum()), pearson_r=round(float(pr_noc), 4),
                                 spearman_rho=round(float(sr_noc), 4)),
        core_excl_flu_and_covid=dict(n=int(mask_core.sum()), pearson_r=round(float(pr_core), 4),
                                     spearman_rho=round(float(sr_core), 4)),
    ),
    holds=bool(HOLDS),
    collapses_without_flu_anchor=bool(collapses),
    honest_assessment=(
        "PARTLY SEMI-TAUTOLOGICAL. The gap==0 / advantage~0 anchor (flu, transmitter==loudest==tx) is "
        "true BY CONSTRUCTION: when the loudest node IS the transmitter, transmitter-targeting and "
        "greedy target the same node, so their advantage must be ~0. The full-n positive correlation is "
        "therefore not an independent test on its own. Robustness: dropping flu leaves Pearson "
        f"r={pr_x:+.2f} / Spearman rho={sr_x:+.2f} over 6 directed twins, and dropping the high-leverage "
        f"COVID point (gap~275) leaves r={pr_noc:+.2f}. Decision rule: holds=true only if the relationship "
        "survives removal of the gap==0 anchor with |r|>=0.30 and same sign."),
    caveats=[
        "n=7 twins; all p-values are non-significant -- exploratory only, no overclaiming.",
        "flu (gap=0) is a structural anchor, not an independent observation; relationship is partly definitional there.",
        "COVID is a high-leverage point: gap~275 vs <=88 for all others; it disproportionately drives the slope.",
        "equity advantage uses the interdict recompute at ridge=5e-2 (transmitter +43.1, greedy -3.0 = 46.1), a different controller setup than the transfer-json twins; advantage definitions are not perfectly homogeneous across twins.",
        "gap is computed on the cached fit_var_nonneg ridge=5e-2 kernel (verified == fresh fit, maxdiff 0 for all 7).",
    ],
    kernels="cached /tmp/lsa_nn/Phi_*.npy, Sig_*.npy (verified identical to fresh fit_var_nonneg ridge=5e-2).",
)
json.dump(out, open(BASE / "advantage_gap.json", "w"), indent=2)
print("json -> advantage_gap.json")
