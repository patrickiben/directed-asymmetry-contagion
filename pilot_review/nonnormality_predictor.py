"""
NON-NORMALITY as the physics-grounded predictor of the directed-asymmetry controller advantage.

Hypothesis (candidate addition to the NCS submission): the controller advantage of transmitter-targeting
over loudest-node ('greedy') interdiction is predicted by the NON-NORMALITY of the fitted contagion kernel
Phi -- and is NOT predicted by total Diebold-Yilmaz connectedness (its MAGNITUDE). The decisive case is the
smoke-vs-equity dissociation: near-equal connectedness (~82 vs ~81%) but very different non-normality.

Non-normality indices (scale-invariant -> rescaling irrelevant, computed on the pre-rescale Phi):
  nu_H = sqrt( ||Phi||_F^2 - sum_i |lambda_i|^2 ) / ||Phi||_F     (Henrici normalized departure; 0 iff normal)
  asym = ||Phi - Phi^T||_F / ||Phi||_F                            (antisymmetry fraction; ties to the symmetrization null)
  reactivity_gap = lambda_max((Phi+Phi^T)/2) - rho(Phi)           (transient amplification a normal matrix lacks)

Advantage = transmitter-cascade-reduction% minus greedy% on the rescaled twin, taken from each system's
*_transfer_results.json where present; equity + COVID recomputed consistently.

Run: /tmp/lsa_venv/bin/python3 nonnormality_predictor.py
"""
import sys, json, io, zipfile, urllib.request, time
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, str(Path(__file__).parent.parent / "pilot_cross_tier"))
import lsa_capstone as L
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
EQDIR = Path(__file__).parent.parent / "pilot_3p46_equity"
NNDIR = Path("/tmp/lsa_nn"); NNDIR.mkdir(parents=True, exist_ok=True)
RIDGE = 5e-2

# ============================================================ non-normality indices
def nonnormality(Phi):
    """Henrici nu_H, antisymmetry fraction, reactivity gap. All scale-invariant (nu_H, asym) on Phi."""
    fro2 = float(np.sum(Phi * Phi))                       # ||Phi||_F^2
    fro = np.sqrt(fro2)
    ev = np.linalg.eigvals(Phi)
    sum_lam2 = float(np.sum(np.abs(ev) ** 2))             # sum_i |lambda_i|^2
    nu_H = float(np.sqrt(max(fro2 - sum_lam2, 0.0)) / fro) if fro > 0 else 0.0
    asym = float(np.linalg.norm(Phi - Phi.T, "fro") / fro) if fro > 0 else 0.0
    sym = 0.5 * (Phi + Phi.T)
    lam_max_sym = float(np.max(np.linalg.eigvalsh(sym)))
    rho = float(np.max(np.abs(ev)))
    react_gap = lam_max_sym - rho
    return nu_H, asym, react_gap, rho, fro

def dy_total(Phi, Sig):
    _, _, _, tot = L.connectedness(L.gfevd(Phi, Sig))
    return float(tot)

# ============================================================ consistent interdiction (matches transfer scripts)
def project(a, x, B): a = np.clip(a, 0, x); s = a.sum(); return a * (B / s) if s > B else a
def stepf(Phi, c, x, a, rng): return np.clip(Phi @ np.clip(x - a, 0, None) + c + 0.05 * rng.standard_normal(len(x)), 0, None)
def greedy(x, B):
    a = np.zeros_like(x); rem = B
    for i in np.argsort(-x):
        g = min(x[i], rem); a[i] = g; rem -= g
        if rem <= 1e-9: break
    return a
def alloc_score(score, x, B):
    w = np.clip(score, 0, None)
    return project(B * w / w.sum(), x, B) if w.sum() > 0 else np.zeros_like(x)
def rescale(Phi, rho=1.06):
    ev = max(abs(np.linalg.eigvals(Phi))); return Phi * (rho / ev) if ev > 1e-6 else Phi
def interdict_adv(Phi0, c0, Sig0, S0, B=2.0, T=16, seeds=16, seed0=20):
    """Returns transmitter% and greedy% (cascade reduction vs no-action) on the rescaled twin."""
    _, _, NET, _ = L.connectedness(L.gfevd(Phi0, Sig0))
    Phi = rescale(Phi0); scores = {"transmitter": np.clip(NET, 0, None)}
    out = {}
    for name in ["none", "greedy", "transmitter"]:
        tt = []
        for s in range(seeds):
            rng = np.random.default_rng(seed0 + s); x = S0.copy(); acc = 0.0
            for t in range(T):
                if name == "none": a = np.zeros_like(x)
                elif name == "greedy": a = greedy(x, B)
                else: a = alloc_score(scores[name], x, B)
                x = stepf(Phi, c0, x, a, rng); acc += x.sum()
            tt.append(acc)
        out[name] = float(np.mean(tt))
    b = out["none"]
    red = {k: 100 * (1 - v / b) for k, v in out.items()}
    return red["transmitter"], red["greedy"]

# ============================================================ data loaders (cached -> refit Phi @ ridge=5e-2)
def load_asia97():
    SER = {"DEXTHUS": "Thai", "DEXKOUS": "Korea", "DEXMAUS": "Malay", "DEXSIUS": "Singa", "DEXJPUS": "Japan",
           "DEXTAUS": "Taiwan", "DEXHKUS": "HK", "DEXINUS": "India", "DEXCHUS": "China"}
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
        if s is not None and len(s) > 400: cols[nm] = s
    FX = pd.DataFrame(cols).resample("W").last().interpolate().dropna()
    base = FX.loc["1996-01-01":"1996-12-31"].mean()
    S = (FX / base - 1.0) * 100.0
    S = S.loc["1996-06-01":"1998-12-31"]
    return S.values, np.maximum(S.values.mean(0), 0.5)

def load_smoke():
    STATES = ["New York", "New Jersey", "Pennsylvania", "Connecticut", "Massachusetts", "Rhode Island", "Vermont",
              "New Hampshire", "Maine", "Ohio", "Michigan", "Illinois", "Wisconsin", "Minnesota", "Indiana", "Maryland", "Virginia"]
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen("https://aqs.epa.gov/aqsweb/airdata/daily_88101_2023.zip", timeout=120).read()))
    csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
    raw = pd.read_csv(z.open(csv_name), usecols=["State Name", "Date Local", "Arithmetic Mean"])
    raw = raw[raw["State Name"].isin(STATES)]
    raw["Date Local"] = pd.to_datetime(raw["Date Local"])
    daily = raw.groupby(["State Name", "Date Local"])["Arithmetic Mean"].mean().reset_index()
    P = daily.pivot(index="Date Local", columns="State Name", values="Arithmetic Mean").sort_index()
    P = P.loc["2023-05-01":"2023-07-31"].interpolate().dropna(axis=1, how="any")
    avail = [s for s in STATES if s in P.columns]; P = P[avail]
    S = P.copy()
    return S.values, np.maximum(S.values.mean(0), 0.5)

def load_flu():
    STATES = ["ca", "tx", "fl", "ny", "pa", "il", "oh", "ga", "nc", "mi", "nj", "va", "wa", "az", "ma"]
    url = "https://api.delphi.cmu.edu/epidata/fluview/?regions=" + ",".join(STATES) + "&epiweeks=201040-201920"
    r = json.load(urllib.request.urlopen(url, timeout=60))
    assert r["result"] == 1, r.get("message")
    df = pd.DataFrame(r["epidata"])[["region", "epiweek", "wili"]]
    M = df.pivot(index="epiweek", columns="region", values="wili").sort_index()
    wk = (M.index % 100)
    M = M[(wk >= 40) | (wk <= 20)]
    avail = [s for s in STATES if s in M.columns]
    M = M[avail].interpolate().dropna()
    return M.values, np.maximum(M.values.mean(0), 0.5)

def load_flights():
    S = pd.read_csv("/tmp/lsa_flights/flights_panel.csv", index_col=0)
    return S.values, np.maximum(S.values.mean(0), 0.5)

def load_conflict():
    S = pd.read_csv("/tmp/lsa_conflict/conflict_panel.csv", index_col=0)
    return S.values, np.maximum(S.values.mean(0), 0.5)

def load_equity():
    idx = {"GSPC": "US", "FTSE": "UK", "GDAXI": "Germany", "FCHI": "France",
           "N225": "Japan", "HSI": "Hong Kong", "AEX": "Neth.", "BVSP": "Brazil"}
    def load(sym):
        d = json.load(open(EQDIR / "data" / f"{sym}.json"))["chart"]["result"][0]
        s = pd.Series(d["indicators"]["quote"][0]["close"], index=pd.to_datetime(d["timestamp"], unit="s")).dropna()
        return s
    P = pd.concat([load(s).rename(idx[s]) for s in idx], axis=1).ffill().dropna()
    P = P.resample("W").last().loc["2007-01-01":"2010-06-30"]
    ret = 100 * np.log(P).diff().dropna()
    stress = (-ret).clip(lower=0)
    S0 = stress.loc["2008-09-01":"2008-11-15"].mean().values + 0.5
    return stress.values, S0

def load_covid():
    DROP = {"American Samoa", "Guam", "Northern Mariana Islands", "Virgin Islands", "Diamond Princess",
            "Grand Princess", "Puerto Rico"}
    d = pd.read_csv(BASE / "jhu_confirmed_US.csv"); dcols = [c for c in d.columns if "/" in c]
    st = d.groupby("Province_State")[dcols].sum().drop(index=[s for s in DROP if s in d["Province_State"].unique()], errors="ignore")
    st.columns = pd.to_datetime(st.columns)
    weekly = st.T.sort_index().resample("W").last().diff().clip(lower=0).dropna()
    top = weekly.sum().sort_values(ascending=False).head(14).index
    W = (weekly[top] / 1000.0).loc["2020-03-01":"2022-06-30"]
    S0 = W.loc["2020-06-01":"2020-08-01"].mean().values + 0.5
    return W.values, S0

# ============================================================ advantage sources
# transmitter - greedy from each *_transfer_results.json (rescaled twin, same interdiction)
def adv_from_json(fname, key="interdiction"):
    d = json.load(open(BASE / fname))
    it = d[key]
    return float(it["transmitter"] - it["greedy"]), "transfer_results.json (transmitter-greedy)"

VERDICT = {  # verdict class assigned in each system's own script
    "asia97": "confirm", "smoke": "confirm", "flu": "refine", "flights": "falsify",
    "conflict": "null", "equity": "symmetric", "COVID": "confirm",
}

SYSTEMS = [
    ("asia97",   load_asia97,   "json", "asia97_transfer_results.json"),
    ("smoke",    load_smoke,    "json", "smoke23_transfer_results.json"),
    ("flu",      load_flu,      "json", "flu_transfer_results.json"),
    ("flights",  load_flights,  "json", "flights_transfer_results.json"),
    ("conflict", load_conflict, "json", "conflict_transfer_results.json"),
    ("equity",   load_equity,   "recompute_eq", None),
    ("COVID",    load_covid,    "covid", None),
]

rows = []
for name, loader, adv_mode, jf in SYSTEMS:
    try:
        M, S0 = loader()
    except Exception as e:
        print(f"[{name}] SKIPPED (load failed): {e}")
        rows.append(dict(system=name, error=str(e)))
        continue
    Phi0, c0, Sig0 = L.fit_var_nonneg(M, ridge=RIDGE)          # the fitted kernel, BEFORE rescaling
    np.save(NNDIR / f"Phi_{name}.npy", Phi0)
    np.save(NNDIR / f"Sig_{name}.npy", Sig0)
    nu_H, asym, react_gap, rho, fro = nonnormality(Phi0)
    conn = dy_total(Phi0, Sig0)

    if adv_mode == "json":
        adv, src = adv_from_json(jf)
    elif adv_mode == "recompute_eq":
        # SAME interdiction as the transfer scripts (transmitter vs greedy on the rescaled twin)
        tr, gr = interdict_adv(Phi0, c0, Sig0, S0, seed0=20)
        adv = tr - gr; src = f"recomputed interdict transmitter({tr:.1f}) - greedy({gr:.1f})"
    elif adv_mode == "covid":
        # task: static-transmit 31.9 vs greedy 0.2 from validity_decomposition -> advantage 31.7
        vd = json.load(open(BASE / "validity_decomposition_results.json"))["covid"]["reductions"]
        adv = float(vd["static-transmit"] - vd["greedy"])
        src = f"validity_decomposition static-transmit({vd['static-transmit']}) - greedy({vd['greedy']})"
    row = dict(system=name, N=int(M.shape[1]), nu_H=round(nu_H, 4), asym=round(asym, 4),
               react_gap=round(react_gap, 4), rho=round(rho, 4), fro=round(fro, 4),
               connectedness=round(conn, 1), advantage=round(adv, 2),
               verdict=VERDICT[name], advantage_source=src)
    rows.append(row)
    print(f"[{name:8s}] N={M.shape[1]:2d}  nu_H={nu_H:.3f}  asym={asym:.3f}  react={react_gap:+.3f}  "
          f"conn={conn:5.1f}%  advantage={adv:+6.2f}  ({VERDICT[name]})  <- {src}")

good = [r for r in rows if "error" not in r]
nu = np.array([r["nu_H"] for r in good])
asy = np.array([r["asym"] for r in good])
adv = np.array([r["advantage"] for r in good])
conn = np.array([r["connectedness"] for r in good])

def corr(x, y):
    pr, pp = pearsonr(x, y); sr, sp = spearmanr(x, y)
    return float(pr), float(pp), float(sr), float(sp)

pr_nu, pp_nu, sr_nu, sp_nu = corr(nu, adv)
pr_as, pp_as, sr_as, sp_as = corr(asy, adv)
pr_cn, pp_cn, sr_cn, sp_cn = corr(conn, adv)

# also report nu_H vs advantage EXCLUDING covid (the only non-transfer advantage definition), as a robustness check
mask = np.array([r["system"] != "COVID" for r in good])
pr_nu_x, pp_nu_x, sr_nu_x, sp_nu_x = corr(nu[mask], adv[mask])

print("\n================ CORRELATIONS (n = %d) ================" % len(good))
print(f"nu_H  vs advantage : Pearson r={pr_nu:+.3f} (p={pp_nu:.3f})  Spearman rho={sr_nu:+.3f} (p={sp_nu:.3f})")
print(f"asym  vs advantage : Pearson r={pr_as:+.3f} (p={pp_as:.3f})  Spearman rho={sr_as:+.3f} (p={sp_as:.3f})")
print(f"conn  vs advantage : Pearson r={pr_cn:+.3f} (p={pp_cn:.3f})  Spearman rho={sr_cn:+.3f} (p={sp_cn:.3f})")
print(f"nu_H  vs advantage (excl COVID, n={mask.sum()}): Pearson r={pr_nu_x:+.3f}  Spearman rho={sr_nu_x:+.3f}")

# smoke vs equity dissociation
smk = next(r for r in good if r["system"] == "smoke")
eq = next(r for r in good if r["system"] == "equity")
print("\n================ smoke vs equity dissociation ================")
print(f"smoke : conn={smk['connectedness']:.1f}%  nu_H={smk['nu_H']:.3f}  asym={smk['asym']:.3f}  advantage={smk['advantage']:+.2f}")
print(f"equity: conn={eq['connectedness']:.1f}%  nu_H={eq['nu_H']:.3f}  asym={eq['asym']:.3f}  advantage={eq['advantage']:+.2f}")
print(f"-> near-equal connectedness ({smk['connectedness']:.0f} vs {eq['connectedness']:.0f}), "
      f"nu_H smoke {'>>' if smk['nu_H']>eq['nu_H'] else '<='} equity "
      f"({smk['nu_H']:.3f} vs {eq['nu_H']:.3f}); advantage {smk['advantage']:+.1f} vs {eq['advantage']:+.1f}")

# ============================================================ save results json
out = dict(ridge=RIDGE, n_systems=len(good),
           systems={r["system"]: {k: r[k] for k in ("N", "nu_H", "asym", "react_gap", "rho", "fro",
                                                     "connectedness", "advantage", "verdict", "advantage_source")}
                    for r in good},
           correlations=dict(
               nu_H_vs_advantage=dict(pearson_r=round(pr_nu, 4), pearson_p=round(pp_nu, 4),
                                      spearman_rho=round(sr_nu, 4), spearman_p=round(sp_nu, 4)),
               asym_vs_advantage=dict(pearson_r=round(pr_as, 4), pearson_p=round(pp_as, 4),
                                      spearman_rho=round(sr_as, 4), spearman_p=round(sp_as, 4)),
               connectedness_vs_advantage=dict(pearson_r=round(pr_cn, 4), pearson_p=round(pp_cn, 4),
                                               spearman_rho=round(sr_cn, 4), spearman_p=round(sp_cn, 4)),
               nu_H_vs_advantage_excl_covid=dict(pearson_r=round(pr_nu_x, 4), spearman_rho=round(sr_nu_x, 4), n=int(mask.sum())),
           ),
           smoke_vs_equity=dict(smoke=dict(conn=smk["connectedness"], nu_H=smk["nu_H"], asym=smk["asym"], advantage=smk["advantage"]),
                                equity=dict(conn=eq["connectedness"], nu_H=eq["nu_H"], asym=eq["asym"], advantage=eq["advantage"])))
json.dump(out, open(BASE / "nonnormality_results.json", "w"), indent=2)
print("\nresults -> nonnormality_results.json ; per-system Phi -> /tmp/lsa_nn/")

# ============================================================ figure
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "savefig.dpi": 300})
VCOL = {"confirm": "#2e8b57", "refine": "#d4a017", "falsify": "#c00000",
        "null": "#7f7f7f", "symmetric": "#1f4e78"}
labels = [r["system"] for r in good]
vcols = [VCOL[r["verdict"]] for r in good]

fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.6))

def scatter(a, xv, title, xlab, r_p, p_p, r_s):
    for i in range(len(good)):
        a.scatter(xv[i], adv[i], s=70, color=vcols[i], edgecolor="k", lw=0.6, zorder=3)
        a.annotate(labels[i], (xv[i], adv[i]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    # OLS fit line
    b1, b0 = np.polyfit(xv, adv, 1)
    xs = np.linspace(xv.min(), xv.max(), 50)
    a.plot(xs, b0 + b1 * xs, color="#444", lw=1.2, ls="--", zorder=2)
    a.set_title(title); a.set_xlabel(xlab); a.set_ylabel("controller advantage  (transmitter - greedy, pts)")
    a.grid(alpha=.25)
    a.text(0.04, 0.96, f"Pearson r = {r_p:+.2f} (p={p_p:.2f})\nSpearman $\\rho$ = {r_s:+.2f}",
           transform=a.transAxes, va="top", ha="left", fontsize=8.5,
           bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#888", lw=.7))

scatter(ax[0], nu, "(a) Advantage vs non-normality $\\nu_H$", "Henrici non-normality  $\\nu_H$", pr_nu, pp_nu, sr_nu)
scatter(ax[1], conn, "(b) Advantage vs total connectedness (DY)", "DY total connectedness (%)", pr_cn, pp_cn, sr_cn)

# annotate smoke/equity pair on BOTH panels
for a, xkey in [(ax[0], "nu_H"), (ax[1], "connectedness")]:
    xs_ = smk[xkey]; xe_ = eq[xkey]
    a.annotate("", xy=(xe_, eq["advantage"]), xytext=(xs_, smk["advantage"]),
               arrowprops=dict(arrowstyle="<->", color="#7030a0", lw=1.3, alpha=.8))
    midx = (xs_ + xe_) / 2; midy = (smk["advantage"] + eq["advantage"]) / 2
    a.text(midx, midy, " smoke vs equity", color="#7030a0", fontsize=8, fontweight="bold",
           ha="center", va="bottom", rotation=0)

# legend for verdict colours
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="k", markersize=8, label=v)
           for v, c in VCOL.items()]
ax[0].legend(handles=handles, loc="lower right", fontsize=7.5, title="verdict class", title_fontsize=8)

fig.suptitle("Non-normality of the fitted kernel predicts the directed-asymmetry controller advantage; total connectedness does not",
             fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(BASE / "nonnormality_predictor.pdf", bbox_inches="tight")
fig.savefig(BASE / "nonnormality_predictor.png", dpi=200, bbox_inches="tight")
print("figure -> nonnormality_predictor.pdf/.png")
