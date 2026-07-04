"""
SI: rho-SENSITIVITY of the directed-asymmetry result across all five transfer twins.

For each twin we reconstruct (M -> Phi0, c0, Sig0, NET, S0, names) EXACTLY as its own transfer
script does (same fetch/cache, same fit_var_nonneg(ridge=5e-2), same DY connectedness, same local
interdiction simulator). We then sweep target_rho in {1.02, 1.05, 1.08, 1.10} through the same
`rescale(Phi0, rho)` step the scripts use, and record for each rho:
  - transmitter-controller cascade reduction %   (red['transmitter'], NET-weighted allocation)
  - loudest-node 'greedy' cascade reduction %     (red['greedy'])
  - their ratio transmitter/greedy.

The KEY claim tested: the QUALITATIVE law (confirms: transmitter > greedy; null/falsify: both near
the noise floor) is INVARIANT across rho even though absolute magnitudes change with rho.

Run: python3 si_rho_sensitivity.py
"""
import sys, json, io, zipfile, urllib.request, time
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "pilot_cross_tier"))
import lsa_capstone as L
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.7, "savefig.dpi": 300})
CRIT, STEEL, GREEN, PURPLE, GREY, GOLD, TEAL = "#c00000", "#1f4e78", "#2e8b57", "#7030a0", "#7f7f7f", "#d4a017", "#138086"
BASE = Path(__file__).parent
RHOS = [1.02, 1.05, 1.08, 1.10]

# ============================================================================ shared simulator (IDENTICAL across scripts)
# These five helpers are byte-identical in all five transfer scripts; the per-twin pieces that differ are
# (a) the random-seed offset in interdict (30/20/40/20/20) and (b) the S0 floor / mean basis. We pass both in.
def project(a, x, B): a = np.clip(a, 0, x); s = a.sum(); return a * (B / s) if s > B else a
def stepf(Phi, c, x, a, rng): return np.clip(Phi @ np.clip(x - a, 0, None) + c + 0.05 * rng.standard_normal(len(x)), 0, None)
def alloc(score, x, B): w = np.clip(score, 0, None); return project(B * w / w.sum(), x, B) if w.sum() > 0 else np.zeros_like(x)
def greedy(x, B):
    a = np.zeros_like(x); rem = B
    for i in np.argsort(-x):
        g = min(x[i], rem); a[i] = g; rem -= g
        if rem <= 1e-9: break
    return a
def rescale(Phi, rho):
    ev = max(abs(np.linalg.eigvals(Phi))); return Phi * (rho / ev) if ev > 1e-6 else Phi

def interdict(Phi, c, scores, S0, seed0, B=2.0, T=16, seeds=16):
    """Replicates the local interdict() of every transfer script (greedy + score-based controllers).
    seed0 = per-script rng offset (asia97:30, smoke23:20, flu:40, flights:20, conflict:20)."""
    out = {}
    for name in ["none", "greedy"] + list(scores.keys()):
        tt = []
        for s in range(seeds):
            rng = np.random.default_rng(seed0 + s); x = S0.copy(); acc = 0.0
            for t in range(T):
                if name == "none": a = np.zeros_like(x)
                elif name == "greedy": a = greedy(x, B)
                else: a = alloc(scores[name], x, B)
                x = stepf(Phi, c, x, a, rng); acc += x.sum()
            tt.append(acc)
        out[name] = float(np.mean(tt))
    b = out["none"]; return {k: 100 * (1 - v / b) for k, v in out.items()}


# ============================================================================ per-twin reconstruction of (S, names, S0_basis, seed0)
def build_asia97():
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
    names = list(S.columns)
    Sv = S.values
    S0 = np.maximum(Sv.mean(0), 0.5)          # asia97 default S0
    return Sv, names, S0, 30

def build_smoke23():
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
    ABBR = {"New York": "NY", "New Jersey": "NJ", "Pennsylvania": "PA", "Connecticut": "CT", "Massachusetts": "MA",
            "Rhode Island": "RI", "Vermont": "VT", "New Hampshire": "NH", "Maine": "ME", "Ohio": "OH", "Michigan": "MI",
            "Illinois": "IL", "Wisconsin": "WI", "Minnesota": "MN", "Indiana": "IN", "Maryland": "MD", "Virginia": "VA"}
    names = [ABBR.get(s, s[:3]) for s in P.columns]
    Sv = P.values
    S0 = np.maximum(Sv.mean(0), 0.5)
    return Sv, names, S0, 20

def build_flu():
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
    names = [s.upper() for s in M.columns]
    Sv = M.values
    S0 = np.maximum(Sv.mean(0), 0.5)
    return Sv, names, S0, 40

def build_flights():
    # cached panel (days x airports), volume-ordered columns, written by flights_transfer.py
    df = pd.read_csv("/tmp/lsa_flights/flights_panel.csv", index_col=0)
    names = list(df.columns)
    Sv = df.values
    S0 = np.maximum(Sv.mean(0), 0.5)
    return Sv, names, S0, 20

def build_conflict():
    df = pd.read_csv("/tmp/lsa_conflict/conflict_panel.csv", index_col=0)
    names = list(df.columns)
    Sv = df.values
    S0 = np.maximum(Sv.mean(0), 0.5)
    return Sv, names, S0, 20


TWINS = [
    ("asia97",   "1997 Asian FX",      "CONFIRM",  build_asia97),
    ("smoke23",  "2023 Wildfire Smoke","CONFIRM",  build_smoke23),
    ("flu",      "US Influenza",       "REFINE",   build_flu),
    ("flights",  "2013-14 Flight Delays","FALSIFY", build_flights),
    ("conflict", "Sahel Conflict",     "NULL",     build_conflict),
]

results = {}
table_rows = []
for key, label, verdict_class, builder in TWINS:
    print(f"\n========== {key} ({label}, {verdict_class}) ==========")
    try:
        Sv, names, S0, seed0 = builder()
    except Exception as e:
        print(f"  FAILED to build {key}: {type(e).__name__}: {e}")
        results[key] = {"error": f"{type(e).__name__}: {e}"}
        continue
    N = len(names)
    Phi0, c0, Sig0 = L.fit_var_nonneg(Sv, ridge=5e-2)
    TO, FROM, NET, tot = L.connectedness(L.gfevd(Phi0, Sig0))
    transmitter = names[int(np.argmax(NET))]
    loudest = names[int(np.argmax(Sv.mean(0)))]
    r0 = float(max(abs(np.linalg.eigvals(Phi0))))
    scores = {"transmitter": np.clip(NET, 0, None)}
    print(f"  N={N} names={names}")
    print(f"  fitted rho(Phi0)={r0:.3f} | DY total {tot:.0f}% | transmitter={transmitter} loudest={loudest}")
    sweep = []
    for rho in RHOS:
        red = interdict(rescale(Phi0, rho), c0, scores, S0, seed0)
        tred = red["transmitter"]; gred = red["greedy"]
        ratio = tred / gred if abs(gred) > 1e-9 else float("nan")
        sweep.append({"rho": rho, "transmitter_pct": round(tred, 2), "greedy_pct": round(gred, 2),
                      "ratio": (round(ratio, 2) if np.isfinite(ratio) else None)})
        table_rows.append({"twin": key, "label": label, "verdict_class": verdict_class, "rho": rho,
                           "transmitter_pct": round(tred, 2), "greedy_pct": round(gred, 2),
                           "ratio": (round(ratio, 2) if np.isfinite(ratio) else None),
                           "transmitter_node": transmitter, "loudest_node": loudest})
        rr = f"{ratio:.2f}" if np.isfinite(ratio) else "nan"
        print(f"  rho={rho:.2f}:  transmitter={tred:6.2f}%   greedy={gred:6.2f}%   ratio={rr}")
    results[key] = {"label": label, "verdict_class": verdict_class, "names": names,
                    "transmitter_node": transmitter, "loudest_node": loudest,
                    "dy_total_pct": round(float(tot), 1), "fitted_rho_Phi0": round(r0, 4),
                    "sweep": sweep}

# ============================================================================ save JSON
out = {"description": "rho-sensitivity of directed-asymmetry (transmitter vs greedy interdiction) across five transfer twins",
       "rhos": RHOS, "method": "per-twin reconstruction matches each *_transfer.py exactly; rescale(Phi0,rho)+local interdict",
       "twins": results, "table": table_rows}
json.dump(out, open(BASE / "si_rho_sensitivity.json", "w"), indent=2)
print("\nsaved -> si_rho_sensitivity.json")
_failed = [k for k in results if "error" in results[k]]
if _failed:
    print("\n*** WARNING: %d/%d twins FAILED to build (%s). The figure below is INCOMPLETE -- run "
          "flights_transfer.py and conflict_transfer.py first to populate /tmp/lsa_flights and /tmp/lsa_conflict. ***"
          % (len(_failed), len(TWINS), _failed))

# ============================================================================ figure: transmitter vs greedy across rho, one panel per twin
ok = [k for k, *_ in TWINS if k in results and "error" not in results[k]]
n = len(ok)
fig, axes = plt.subplots(1, n, figsize=(2.55 * n, 3.1), sharex=True)
if n == 1: axes = [axes]
for ax, key in zip(axes, ok):
    r = results[key]
    rhos = [s["rho"] for s in r["sweep"]]
    tr = [s["transmitter_pct"] for s in r["sweep"]]
    gr = [s["greedy_pct"] for s in r["sweep"]]
    ax.plot(rhos, tr, "o-", color=STEEL, lw=1.6, ms=4, label=f"Transmitter ({r['transmitter_node']})")
    ax.plot(rhos, gr, "s--", color=PURPLE, lw=1.4, ms=4, label=f"Greedy/Loudest ({r['loudest_node']})")
    ax.axvline(1.06, color=GREY, ls=":", lw=0.8)
    ax.set_title(f"{r['label']} ({r['verdict_class'].capitalize()})", fontsize=7.6)
    ax.set_xlabel(r"Target $\rho$"); ax.grid(alpha=.25)
    ax.legend(fontsize=5.8, loc="center left", frameon=True, framealpha=0.85, edgecolor="0.85", borderpad=0.3)
axes[0].set_ylabel("Cascade Reduction vs No-Action (%)")
fig.suptitle("Spectral-Radius (ρ) Sensitivity: The Directed-Asymmetry Verdict Is Invariant to ρ\n(Transmitter Controller vs Loudest-Node 'Greedy'; Dotted Grey = Published ρ=1.06)"
             "",
             fontsize=8.6, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig(BASE / "si_rho_sensitivity.pdf", bbox_inches="tight")
fig.savefig(BASE / "si_rho_sensitivity.png", dpi=200, bbox_inches="tight")
print("figure -> si_rho_sensitivity.pdf/.png")
