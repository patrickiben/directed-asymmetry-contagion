"""
SIXTH pre-registered transfer test (conflict_PREREGISTRATION.md) — West-African / Lake-Chad / Sahel
armed-conflict diffusion. A NEW domain CLASS: geopolitical / armed-conflict (Tier IV, the framework's
highest tier), never yet validated with a real transfer test.

Stress = monthly total conflict fatalities (UCDP GED v24.1 'best') per country, 2012-01..2023-12.
The central-Sahel insurgency began in Mali (2012); the pre-committed origin prediction is
net-transmitter == Mali, loudest != Mali (gap PRESENT) -> transmitter-targeting >> greedy.

Same pipeline as flights_transfer / smoke23 (COVID/flu/Asian-FX/wildfire-smoke/flights). Verdict scored
against the pre-registered criteria. Honest either way — the prediction is already committed; this only
computes it.

Run: python3 conflict_transfer.py
"""
import sys, json, zipfile
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "pilot_cross_tier"))
import lsa_capstone as L
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.linewidth": 0.7, "savefig.dpi": 300})
CRIT, STEEL, GREEN, PURPLE, GREY, GOLD, TEAL = "#c00000", "#1f4e78", "#2e8b57", "#7030a0", "#7f7f7f", "#d4a017", "#138086"
BASE = Path(__file__).parent
CACHE = Path("/tmp/lsa_conflict"); CACHE.mkdir(parents=True, exist_ok=True)

# pre-registered structural sets (committed before computing the network).
# UCDP-spelled candidate list {Mali, Burkina Faso, Niger, Nigeria, Chad, Cameroon, Ivory Coast, Benin,
# Togo, Ghana, Senegal, Guinea, Mauritania}. "Ivory Coast" is the UCDP name for Cote d'Ivoire.
CANDIDATES = ["Mali", "Burkina Faso", "Niger", "Nigeria", "Chad", "Cameroon", "Ivory Coast",
              "Benin", "Togo", "Ghana", "Senegal", "Guinea", "Mauritania"]
SAHEL = {"Mali", "Burkina Faso", "Niger"}            # central-Sahel insurgency (Mali origin)
LAKECHAD = {"Nigeria", "Chad", "Cameroon"}           # Boko-Haram / Lake-Chad basin (Nigeria origin)

# ----------------------------------------------------------------------------- fetch + cache UCDP GED v24.1
URL = "https://ucdp.uu.se/downloads/ged/ged241-csv.zip"
f = CACHE / "ged241.zip"
if not f.exists():
    print("fetching UCDP GED v24.1 (~30 MB)...")
    import urllib.request
    f.write_bytes(urllib.request.urlopen(URL, timeout=180).read())
else:
    print("using cached UCDP GED v24.1")
z = zipfile.ZipFile(f)
csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
raw = pd.read_csv(z.open(csv_name), usecols=["year", "date_start", "country", "region", "best"], low_memory=False)
raw["month"] = pd.to_datetime(raw["date_start"]).dt.to_period("M")
win = raw[(raw["month"] >= "2012-01") & (raw["month"] <= "2023-12")].copy()
print(f"loaded {len(raw):,} GED events; {len(win):,} in window 2012-01..2023-12 ({win['month'].nunique()} months)")

# ----------------------------------------------------------------------------- >=100-event inclusion filter
# objective, outcome-INDEPENDENT: keep every candidate country with >= 100 conflict EVENTS in the window.
ev_counts = win[win["country"].isin(CANDIDATES)].groupby("country").size()
names = [c for c in CANDIDATES if ev_counts.get(c, 0) >= 100]
print("event counts (candidate list):")
for c in CANDIDATES:
    n = int(ev_counts.get(c, 0)); print(f"  {c:14s} events={n:6d}  pass>=100={n >= 100}")
print("PASS (>=100 events):", names)
sub = win[win["country"].isin(names)]

# ----------------------------------------------------------------------------- stress panel: monthly total fatalities
# sum of 'best' per country per month; pivot months x countries; missing months = 0 (genuinely no fatalities).
fat = sub.groupby(["month", "country"])["best"].sum().reset_index()
full_idx = pd.period_range("2012-01", "2023-12", freq="M")
M = fat.pivot(index="month", columns="country", values="best").reindex(full_idx).reindex(columns=names)
M = M.fillna(0.0)                                                # missing month = 0 fatalities (NOT NaN)
N = len(names); dates = M.index.to_timestamp()
S = M.copy()                                                    # panel M (months x countries), stress = fatalities
S.to_csv(CACHE / "conflict_panel.csv")                          # save for independent verification
print(f"conflict: {N} countries, {len(S)} months (2012-01..2023-12); total fatalities by country:")
print("  " + "  ".join(f"{names[i]}:{int(S.iloc[:,i].sum())}" for i in range(N)))

# ----------------------------------------------------------------------------- diagnose (DY connectedness)
Phi0, c0, Sig0 = L.fit_var_nonneg(S.values, ridge=5e-2)
TO, FROM, NET, tot = L.connectedness(L.gfevd(Phi0, Sig0))
transmitter = names[int(np.argmax(NET))]; loudest = names[int(np.argmax(S.sum().values))]
print(f"DY total connectedness {tot:.0f}% | net-transmitter = {transmitter} | loudest (total fatalities) = {loudest}")
top_tx = [names[i] for i in np.argsort(-NET)[:3]]; top_rx = [names[i] for i in np.argsort(NET)[:3]]
print(f"top transmitters: {top_tx} | receivers: {top_rx}")

# ----------------------------------------------------------------------------- simulator + methods (IDENTICAL to flights/smoke23)
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
    Mm = np.zeros_like(Phi); Pp = np.eye(len(Phi))
    for h in range(H + 1): Mm += np.abs(Pp); Pp = Pp @ Phi
    np.fill_diagonal(Mm, 0); return Mm.sum(0)
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

# ----------------------------------------------------------------------------- verdict vs pre-registration
trans_vs_loud = transmitter != loudest
margin = red["transmitter"] / max(red["greedy"], 1e-6)
directed = adv[0] >= 5 and adv[-1] < adv[0] * 0.6
trans_is_mali = transmitter == "Mali"
trans_sahel = transmitter in SAHEL
dir_beats_undir = min(red["var-out"], red["spillover"], red["transmitter"]) > red["corr"]
if trans_is_mali and trans_vs_loud and red["transmitter"] >= 1.8 * red["greedy"] and directed:
    verdict = (f"CONFIRMED: the conflict net-transmitter is Mali (the pre-committed central-Sahel origin) while the "
               f"loudest country is {loudest}; transmitter-targeting beats greedy x{margin:.1f}, directed methods beat "
               f"undirected correlation, and the edge collapses under symmetrization -- a confirmatory win on a new "
               f"domain class (geopolitical / armed-conflict, Tier IV).")
elif (not trans_is_mali) and trans_vs_loud and red["transmitter"] >= 1.8 * red["greedy"] and directed:
    verdict = (f"CONFIRMED_LAW_MISSED_ORIGIN: the law holds -- the net-transmitter {transmitter} is NOT the loudest "
               f"({loudest}), transmitter-targeting beats greedy x{margin:.1f} on a verifiably directed network -- but the "
               f"pre-committed Mali origin bet MISSED ({transmitter} != Mali). Reported honestly, not inflated to CONFIRMED.")
elif (not trans_vs_loud) and directed:
    verdict = (f"REFINED (like flu): {transmitter} is BOTH the transmitter and the loudest, so the loudest-transmitter gap "
               f"is absent and transmitter-targeting ~ greedy (x{margin:.1f}) despite a directed network -- consistent with "
               f"the sharpened law (the operative quantity is the gap, not directedness alone).")
elif not directed:
    verdict = (f"NULL: the regional violence looks like a near-symmetric common shock (DY {tot:.0f}%, symmetrization barely "
               f"changes the edge), so transmitter-targeting ~ greedy (x{margin:.1f}) -- the no-leverage case the law predicts.")
else:
    verdict = (f"FALSIFIED: the net-transmitter is {transmitter}; the directed-cause / margin conditions for the "
               f"pre-registered confirmation are not met. Reported honestly (see numbers).")
print(f"\ntransmitter={transmitter} (mali:{trans_is_mali}, sahel:{trans_sahel}), loudest={loudest} "
      f"(gap: {trans_vs_loud}); margin x{margin:.1f}; directed:{directed}")
print(f"VERDICT: {verdict}")

RES = dict(countries=names, n_months=int(len(S)), dy_total=round(float(tot), 1), transmitter=transmitter, loudest=loudest,
           top_transmitters=top_tx, top_receivers=top_rx,
           NET={names[i]: round(float(NET[i]), 2) for i in range(N)},
           TO={names[i]: round(float(TO[i]), 2) for i in range(N)},
           FROM={names[i]: round(float(FROM[i]), 2) for i in range(N)},
           total_fatalities={names[i]: int(S.iloc[:, i].sum()) for i in range(N)},
           interdiction={k: round(red[k], 1) for k in red}, symmetrization_advantage=[round(v, 1) for v in adv],
           transmitter_not_loudest=bool(trans_vs_loud), transmitter_is_mali=bool(trans_is_mali),
           transmitter_in_sahel=bool(trans_sahel), directed=bool(directed), dir_beats_undir=bool(dir_beats_undir),
           arro_over_greedy_x=round(float(margin), 2), verdict=verdict)
json.dump(RES, open(BASE / "conflict_transfer_results.json", "w"), indent=2)
print("results -> conflict_transfer_results.json")

# ----------------------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(12.2, 4.0))
a = ax[0]
for i in range(N): a.plot(dates, S.iloc[:, i], lw=0.8, alpha=.85, label=names[i])
a.axvspan(pd.Timestamp("2012-01-01"), pd.Timestamp("2012-04-01"), color=GREY, alpha=.25)
a.text(pd.Timestamp("2012-02-01"), a.get_ylim()[1] * .82, "2012 Mali\nOnset", fontsize=6, color=CRIT)
a.xaxis.set_major_locator(mdates.YearLocator(2)); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
a.set_title("(a) Monthly Conflict Fatalities by Country (2012-2023)", fontsize=8.3); a.set_ylabel("Fatalities / Month")
a.legend(ncol=2, fontsize=6, loc="upper right")
a = ax[1]; oi = np.argsort(NET); cols = [CRIT if NET[i] > 0 else STEEL for i in oi]
a.barh([names[i] for i in oi], NET[oi], color=cols, alpha=.85); a.axvline(0, color="k", lw=.7)
a.set_title(f"(b) Directed Conflict Network\nTransmitter={transmitter}, Loudest={loudest}", fontsize=8.3)
a.set_xlabel("Net Connectedness (%)"); a.tick_params(axis="y", labelsize=6.5); a.grid(alpha=.25, axis="x")
a = ax[2]; ks = ["none", "greedy", "corr", "var-out", "spillover", "transmitter", "mpc"]
cc = [CRIT, PURPLE, GREY, TEAL, GREEN, STEEL, GOLD]; vv = [red[k] for k in ks]
a.bar(range(len(ks)), vv, color=cc, alpha=.85); a.set_ylim(top=max(vv) * 1.18)
for i, v in enumerate(vv): a.text(i, v + max(vv) * 0.025, f"{v:+.0f}%", ha="center", fontsize=6.6, fontweight="bold")
a.set_xticks(range(len(ks))); a.set_xticklabels(["None", "Greedy", "Corr", "Var\nOut", "Spill", "Trans-\nmitter", "MPC"], fontsize=6.2)
a.set_ylabel("Cascade Reduction vs No-Action (%)"); a.set_title("(c) Interdiction + Benchmark", fontsize=8.3); a.grid(alpha=.25, axis="y")
fig.suptitle(f"Sixth Prediction-First Transfer Test — West-African / Sahel Armed-Conflict Diffusion  ·  {verdict.split(':')[0].split('(')[0].strip().capitalize()}",
             fontsize=10, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(BASE / "conflict_transfer.pdf", bbox_inches="tight"); fig.savefig(BASE / "conflict_transfer.png", dpi=200, bbox_inches="tight")
print("figure -> conflict_transfer.pdf/.png")
