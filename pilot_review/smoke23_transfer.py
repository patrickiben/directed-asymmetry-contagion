"""
FOURTH pre-registered transfer test (smoke23_PREREGISTRATION.md) — 2023 Canadian wildfire-smoke air-quality
disaster. A new domain CLASS (environmental disaster) with a directed PM2.5 contagion network across US states.

Same pipeline. Verdict scored against the pre-registered criteria. Honest either way.

Run: /tmp/lsa_venv/bin/python smoke23_transfer.py
"""
import sys, json, io, zipfile, urllib.request
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent.parent / "pilot_cross_tier"))
import lsa_capstone as L
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.linewidth": 0.7, "savefig.dpi": 300})
CRIT, STEEL, GREEN, PURPLE, GREY, GOLD, TEAL = "#c00000", "#1f4e78", "#2e8b57", "#7030a0", "#7f7f7f", "#d4a017", "#138086"
BASE = Path(__file__).parent
import matplotlib.dates as mdates

# ----------------------------------------------------------------------------- fetch EPA daily PM2.5 (2023)
STATES = ["New York", "New Jersey", "Pennsylvania", "Connecticut", "Massachusetts", "Rhode Island", "Vermont",
          "New Hampshire", "Maine", "Ohio", "Michigan", "Illinois", "Wisconsin", "Minnesota", "Indiana", "Maryland", "Virginia"]
print("fetching EPA daily PM2.5 2023 (~9 MB)...")
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
names = [ABBR.get(s, s[:3]) for s in P.columns]; N = len(names); dates = P.index
S = P.copy()                                                     # stress = daily PM2.5 (ug/m3)
print(f"smoke23: {N} states, {len(S)} days (May-Jul 2023); peak daily PM2.5 by state:")
print("  " + "  ".join(f"{names[i]}:{S.iloc[:,i].max():.0f}" for i in range(N)))

# ----------------------------------------------------------------------------- diagnose
Phi0, c0, Sig0 = L.fit_var_nonneg(S.values, ridge=5e-2)
TO, FROM, NET, tot = L.connectedness(L.gfevd(Phi0, Sig0))
transmitter = names[int(np.argmax(NET))]; loudest = names[int(np.argmax(S.mean().values))]
print(f"DY total connectedness {tot:.0f}% | net-transmitter = {transmitter} | loudest (mean PM2.5) = {loudest}")
print(f"top transmitters: {[names[i] for i in np.argsort(-NET)[:3]]} | receivers: {[names[i] for i in np.argsort(NET)[:3]]}")

# ----------------------------------------------------------------------------- simulator + methods
def project(a, x, B): a = np.clip(a, 0, x); s = a.sum(); return a * (B / s) if s > B else a
def stepf(Phi, c, x, a, rng): return np.clip(Phi @ np.clip(x - a, 0, None) + c + 0.05 * rng.standard_normal(len(x)), 0, None)
def alloc(score, x, B): w = np.clip(score, 0, None); return project(B * w / w.sum(), x, B) if w.sum() > 0 else np.zeros_like(x)
def greedy(x, B):
    a = np.zeros_like(x); rem = B
    for i in np.argsort(-x):
        g = min(x[i], rem); a[i] = g; rem -= g
        if rem <= 1e-9: break
    return a
def ctl_mpc(x, Phi, c, B, H, rng):
    n = len(x); mu = np.full((H, n), B / n); sd = np.full((H, n), B / 2)
    for _ in range(4):
        C = np.clip(rng.normal(mu, sd, (48, H, n)), 0, None); cost = np.zeros(48); sim = np.tile(x, (48, 1))
        for h in range(H):
            A = np.stack([project(C[j, h], sim[j], B) for j in range(48)])
            sim = np.clip((sim - A) @ Phi.T + c, 0, None); cost += sim.sum(1)
        el = C[np.argsort(cost)[:8]]; mu, sd = el.mean(0), el.std(0) + 1e-3
    return project(mu[0], x, B)
def s_corr(Phi, Sig):
    Sx = Sig.copy()
    for _ in range(300): Sx = Phi @ Sx @ Phi.T + Sig
    d = np.sqrt(np.clip(np.diag(Sx), 1e-9, None)); Cc = np.abs(Sx / np.outer(d, d)); np.fill_diagonal(Cc, 0); return Cc.sum(1)
def s_varout(Phi): A = np.abs(Phi).copy(); np.fill_diagonal(A, 0); return A.sum(0)
def s_spill(Phi, H=15):
    M = np.zeros_like(Phi); Pp = np.eye(len(Phi))
    for h in range(H + 1): M += np.abs(Pp); Pp = Pp @ Phi
    np.fill_diagonal(M, 0); return M.sum(0)
def rescale(Phi, rho=1.06):
    ev = max(abs(np.linalg.eigvals(Phi))); return Phi * (rho / ev) if ev > 1e-6 else Phi
def interdict(Phi, c, scores, B=2.0, T=16, H=4, seeds=16):
    S0 = np.maximum(S.values.mean(0), 0.5); out = {}
    for name in ["none", "greedy", "mpc"] + list(scores.keys()):
        tt = []
        for s in range(seeds):
            rng = np.random.default_rng(20 + s); x = S0.copy(); acc = 0.0
            for t in range(T):
                if name == "none": a = np.zeros_like(x)
                elif name == "greedy": a = greedy(x, B)
                elif name == "mpc": a = ctl_mpc(x, Phi, c, B, H, rng)
                else: a = alloc(scores[name], x, B)
                x = stepf(Phi, c, x, a, rng); acc += x.sum()
            tt.append(acc)
        out[name] = float(np.mean(tt))
    b = out["none"]; return {k: 100 * (1 - v / b) for k, v in out.items()}

scores = {"corr": s_corr(Phi0, Sig0), "var-out": s_varout(Phi0), "spillover": s_spill(Phi0), "transmitter": np.clip(NET, 0, None)}
red = interdict(rescale(Phi0), c0, scores)
print("\nInterdiction (cascade reduction vs no-action):")
for k in ["none", "greedy", "corr", "var-out", "spillover", "transmitter", "mpc"]: print(f"  {k:12s} {red[k]:+.0f}%")
alphas = [1.0, 0.75, 0.5, 0.25, 0.0]; adv = []
for a in alphas:
    Pa = a * Phi0 + (1 - a) * 0.5 * (Phi0 + Phi0.T); _, _, NETa, _ = L.connectedness(L.gfevd(Pa, Sig0))
    ra = interdict(rescale(Pa), c0, {"transmitter": np.clip(NETa, 0, None)}, seeds=10)
    adv.append(ra["transmitter"] - ra["greedy"])
print("Symmetrization null (transmitter - greedy advantage):", [round(v, 1) for v in adv])

# ----------------------------------------------------------------------------- verdict
trans_vs_loud = transmitter != loudest
margin = red["transmitter"] / max(red["greedy"], 1e-6)
directed = adv[0] >= 5 and adv[-1] < adv[0] * 0.6
if trans_vs_loud and red["transmitter"] >= 1.8 * red["greedy"] and directed:
    verdict = (f"CONFIRMED (loudest != transmitter): the smoke transmitter is {transmitter} while the loudest state is "
               f"{loudest}; transmitter-targeting beats greedy x{margin:.1f}, directed beats undirected, and the edge collapses "
               f"under symmetrization -- a confirmatory win on a new domain class (environmental disaster).")
elif (not trans_vs_loud) and directed:
    verdict = (f"REFINES (like flu): {transmitter} is BOTH transmitter and loudest, so the gap is absent and transmitter-"
               f"targeting ~ greedy (x{margin:.1f}) despite a directed network -- consistent with the sharpened law.")
elif not directed:
    verdict = (f"SYMMETRIC NULL (like 2008 equity): the smoke plume is a near-symmetric common factor (DY {tot:.0f}%, "
               f"symmetrization barely changes the edge), so transmitter-targeting ~ greedy -- the no-leverage case the law predicts.")
else:
    verdict = "MIXED: reported honestly (see numbers)."
print(f"\ntransmitter={transmitter}, loudest={loudest} (gap: {trans_vs_loud}); margin x{margin:.1f}; directed:{directed}")
print(f"VERDICT: {verdict}")

RES = dict(states=names, n_days=int(len(S)), dy_total=round(float(tot), 1), transmitter=transmitter, loudest=loudest,
           peak_pm25={names[i]: round(float(S.iloc[:, i].max()), 1) for i in range(N)},
           interdiction={k: round(red[k], 1) for k in red}, symmetrization_advantage=[round(v, 1) for v in adv],
           transmitter_not_loudest=bool(trans_vs_loud), arro_over_greedy_x=round(float(margin), 2), verdict=verdict)
json.dump(RES, open(BASE / "smoke23_transfer_results.json", "w"), indent=2)

# ----------------------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(12.2, 4.0))
a = ax[0]
for i in range(N): a.plot(dates, S.iloc[:, i], lw=0.9, alpha=.8, label=names[i])
a.axvspan(pd.Timestamp("2023-06-06"), pd.Timestamp("2023-06-09"), color=GREY, alpha=.2)
a.text(pd.Timestamp("2023-06-07"), a.get_ylim()[1]*.85, "NYC\norange", fontsize=6, color=CRIT)
a.xaxis.set_major_locator(mdates.MonthLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
a.set_title("(a) Daily PM2.5 by state (May-Jul 2023)", fontsize=8.3); a.set_ylabel("PM2.5 (ug/m3)"); a.legend(ncol=3, fontsize=5, loc="upper left")
a = ax[1]; oi = np.argsort(NET); cols = [CRIT if NET[i] > 0 else STEEL for i in oi]
a.barh([names[i] for i in oi], NET[oi], color=cols, alpha=.85); a.axvline(0, color="k", lw=.7)
a.set_title(f"(b) Directed smoke network\ntransmitter={transmitter}, loudest={loudest}", fontsize=8.3); a.set_xlabel("net connectedness (%)"); a.tick_params(axis="y", labelsize=6); a.grid(alpha=.25, axis="x")
a = ax[2]; ks = ["none", "greedy", "corr", "var-out", "spillover", "transmitter", "mpc"]
cc = [CRIT, PURPLE, GREY, TEAL, GREEN, STEEL, GOLD]; vv = [red[k] for k in ks]
a.bar(range(len(ks)), vv, color=cc, alpha=.85)
for i, v in enumerate(vv): a.text(i, v + .5, f"{v:+.0f}%", ha="center", fontsize=6.6, fontweight="bold")
a.set_xticks(range(len(ks))); a.set_xticklabels(["none", "greedy", "corr", "var\nout", "spill", "trans\nmitter", "mpc"], fontsize=6.2)
a.set_ylabel("cascade reduction vs no-action (%)"); a.set_title("(c) Interdiction + benchmark", fontsize=8.3); a.grid(alpha=.25, axis="y")
fig.suptitle(f"Fourth pre-registered transfer test — 2023 wildfire-smoke air quality  ·  {verdict.split(':')[0].split('(')[0].strip()}",
             fontsize=10, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(BASE / "smoke23_transfer.pdf", bbox_inches="tight"); fig.savefig(BASE / "smoke23_transfer.png", dpi=200, bbox_inches="tight")
print("figure -> smoke23_transfer.pdf/.png")
