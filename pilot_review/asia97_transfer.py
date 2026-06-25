"""
THIRD pre-registered transfer test (asia97_PREREGISTRATION.md) — the 1997 Asian financial crisis (FX contagion).

A candidate *confirmatory* domain: a crisis with a clear quiet origin (Thailand) distinct from its loudest
casualties. Same pipeline as the rest of the paper. Verdict scored against the pre-registered criteria.

Run: /tmp/lsa_venv/bin/python asia97_transfer.py
"""
import sys, json, io, urllib.request, time
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent.parent / "pilot_cross_tier"))
import lsa_capstone as L
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.linewidth": 0.7, "savefig.dpi": 300})
CRIT, STEEL, GREEN, PURPLE, GREY, GOLD, TEAL = "#c00000", "#1f4e78", "#2e8b57", "#7030a0", "#7f7f7f", "#d4a017", "#138086"
BASE = Path(__file__).parent

# ----------------------------------------------------------------------------- fetch FRED FX
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
print("fetched:", list(cols.keys()))
FX = pd.DataFrame(cols).resample("W").last().interpolate().dropna()
base = FX.loc["1996-01-01":"1996-12-31"].mean()
S = (FX / base - 1.0) * 100.0                                   # % depreciation from 1996 baseline (stress)
S = S.loc["1996-06-01":"1998-12-31"]
names = list(S.columns); N = len(names)
print(f"1997 Asian FX: {N} currencies, {len(S)} weeks; peak depreciation by currency:")
print("  " + "  ".join(f"{c}:{S[c].max():.0f}%" for c in names))

# ----------------------------------------------------------------------------- diagnose
Phi0, c0, Sig0 = L.fit_var_nonneg(S.values, ridge=5e-2)
TO, FROM, NET, tot = L.connectedness(L.gfevd(Phi0, Sig0))
transmitter = names[int(np.argmax(NET))]
loudest = names[int(np.argmax(S.mean().values))]               # highest mean depreciation
print(f"DY total connectedness {tot:.0f}% | net-transmitter = {transmitter} | loudest (mean deprec.) = {loudest}")
print(f"top transmitters: {[names[i] for i in np.argsort(-NET)[:3]]} | receivers: {[names[i] for i in np.argsort(NET)[:3]]}")

# ----------------------------------------------------------------------------- simulator + methods (as in benchmark)
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
    M = np.zeros_like(Phi); P = np.eye(len(Phi))
    for h in range(H + 1): M += np.abs(P); P = P @ Phi
    np.fill_diagonal(M, 0); return M.sum(0)
def rescale(Phi, rho=1.06):
    ev = max(abs(np.linalg.eigvals(Phi))); return Phi * (rho / ev) if ev > 1e-6 else Phi
def interdict(Phi, c, scores, B=2.0, T=16, H=4, seeds=16, S0=None):
    if S0 is None: S0 = np.maximum(S.values.mean(0), 0.5)
    out = {}
    for name in ["none", "greedy", "mpc"] + list(scores.keys()):
        tot_ = []
        for s in range(seeds):
            rng = np.random.default_rng(30 + s); x = S0.copy(); acc = 0.0
            for t in range(T):
                if name == "none": a = np.zeros_like(x)
                elif name == "greedy": a = greedy(x, B)
                elif name == "mpc": a = ctl_mpc(x, Phi, c, B, H, rng)
                else: a = alloc(scores[name], x, B)
                x = stepf(Phi, c, x, a, rng); acc += x.sum()
            tot_.append(acc)
        out[name] = float(np.mean(tot_))
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

# ----------------------------------------------------------------------------- verdict vs pre-registration
trans_vs_loud = transmitter != loudest
margin = red["transmitter"] / max(red["greedy"], 1e-6)
directed = adv[0] >= 5 and adv[-1] < adv[0] * 0.6
trans_is_asean = transmitter in ("Thai", "Malay", "Korea", "Singa")
if trans_is_asean and trans_vs_loud and red["transmitter"] >= 1.8 * red["greedy"] and directed:
    verdict = (f"CONFIRMED (loudest != transmitter): the crisis transmitter is {transmitter} (an early-crisis ASEAN "
               f"currency) while the loudest casualty is {loudest}; transmitter-targeting beats greedy x{margin:.1f}, and the "
               f"edge collapses under symmetrization -- a second COVID-style "
               f"confirmation on a genuinely new (financial) domain.")
elif trans_is_asean and (not trans_vs_loud) and directed:
    verdict = (f"REFINES (like flu): {transmitter} is BOTH the transmitter and the loudest, so the loudest-transmitter gap "
               f"is absent and transmitter-targeting ~ greedy (x{margin:.1f}) despite a strongly directed network -- "
               f"consistent with the sharpened law.")
elif not trans_is_asean:
    verdict = (f"AGAINST PREDICTION: the net-transmitter is {transmitter}, not an early-crisis ASEAN currency -- the "
               f"pre-registered structural prediction was wrong; reported honestly.")
else:
    verdict = "MIXED: see numbers; reported honestly."
print(f"\ntransmitter={transmitter}, loudest={loudest} (gap: {trans_vs_loud}); margin x{margin:.1f}; directed:{directed}")
print(f"VERDICT: {verdict}")

RES = dict(currencies=names, n_weeks=int(len(S)), dy_total=round(float(tot), 1), transmitter=transmitter, loudest=loudest,
           peak_deprec={c: round(float(S[c].max()), 1) for c in names},
           interdiction={k: round(red[k], 1) for k in red}, symmetrization_advantage=[round(v, 1) for v in adv],
           transmitter_not_loudest=bool(trans_vs_loud), arro_over_greedy_x=round(float(margin), 2), verdict=verdict)
json.dump(RES, open(BASE / "asia97_transfer_results.json", "w"), indent=2)

# ----------------------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(12.2, 4.0))
a = ax[0]
for c in names: a.plot(S.index, S[c], lw=1.0, alpha=.85, label=c)
a.axvline(pd.Timestamp("1997-07-02"), color=CRIT, ls="--", lw=.9); a.text(pd.Timestamp("1997-07-05"), a.get_ylim()[1]*.8, "Baht\nFloat", fontsize=6, color=CRIT)
a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); a.tick_params(axis="x", labelrotation=20)
a.set_title("(a) % Depreciation vs 1996 Baseline", fontsize=8.3); a.set_ylabel("Depreciation (%)"); a.legend(ncol=2, fontsize=5.5, loc="upper left")
a = ax[1]; oi = np.argsort(NET); cols = [CRIT if NET[i] > 0 else STEEL for i in oi]
a.barh([names[i] for i in oi], NET[oi], color=cols, alpha=.85); a.axvline(0, color="k", lw=.7)
a.set_title(f"(b) Directed FX Network\nTransmitter={transmitter}, Loudest={loudest}", fontsize=8.3); a.set_xlabel("Net Connectedness (%)"); a.grid(alpha=.25, axis="x")
a = ax[2]; ks = ["none", "greedy", "corr", "var-out", "spillover", "transmitter", "mpc"]
cc = [CRIT, PURPLE, GREY, TEAL, GREEN, STEEL, GOLD]; vv = [red[k] for k in ks]
a.bar(range(len(ks)), vv, color=cc, alpha=.85); a.set_ylim(top=max(vv) * 1.18)
for i, v in enumerate(vv): a.text(i, v + max(vv) * 0.025, f"{v:+.0f}%", ha="center", fontsize=6.6, fontweight="bold")
a.set_xticks(range(len(ks))); a.set_xticklabels(["None", "Greedy", "Corr", "Var\nOut", "Spill", "Trans-\nmitter", "MPC"], fontsize=6.2)
a.set_ylabel("Cascade Reduction vs No-Action (%)"); a.set_title("(c) Interdiction + Benchmark", fontsize=8.3); a.grid(alpha=.25, axis="y")
fig.suptitle(f"Third Prediction-First Transfer Test — 1997 Asian FX Crisis  ·  {verdict.split(':')[0].split('(')[0].strip().capitalize()}",
             fontsize=10, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(BASE / "asia97_transfer.pdf", bbox_inches="tight"); fig.savefig(BASE / "asia97_transfer.png", dpi=200, bbox_inches="tight")
print("figure -> asia97_transfer.pdf/.png")
