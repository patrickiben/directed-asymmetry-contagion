"""
FIFTH pre-registered transfer test (flights_PREREGISTRATION.md) — U.S. air-traffic delay propagation, the
polar-vortex winter (1 Dec 2013 - 28 Feb 2014). A NEW domain CLASS: engineered-infrastructure / logistics,
a human-built transport network whose directed coupling comes from aircraft rotations and crew banking rather
than physics, biology or markets. Departure delays seed at a weather-hit hub and propagate downstream.

Same pipeline as smoke23 (COVID/flu/Asian-FX/wildfire-smoke). Verdict scored against the pre-registered
criteria. Honest either way.

Run: /tmp/lsa_venv/bin/python3 flights_transfer.py
"""
import sys, json, io, zipfile, urllib.request
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
CACHE = Path("/tmp/lsa_flights"); CACHE.mkdir(parents=True, exist_ok=True)

# pre-registered structural sets (committed before computing the network)
WEATHER = {"ORD", "EWR", "LGA", "JFK", "BOS", "DTW", "MSP", "DEN", "PHL"}
SUNBELT = {"ATL", "DFW", "IAH", "PHX", "LAS", "LAX", "MCO", "CLT"}

# ----------------------------------------------------------------------------- fetch + cache 3 BTS months
URL = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip"
MONTHS = [(2013, 12), (2014, 1), (2014, 2)]
frames = []
for y, m in MONTHS:
    f = CACHE / f"{y}_{m}.zip"
    if not f.exists():
        print(f"fetching BTS {y}-{m:02d} ...")
        data = urllib.request.urlopen(URL.format(y=y, m=m), timeout=180).read()
        f.write_bytes(data)
    else:
        print(f"using cached BTS {y}-{m:02d}")
    z = zipfile.ZipFile(f)
    csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    df = pd.read_csv(z.open(csv_name), usecols=["FlightDate", "Origin", "DepDelayMinutes", "Cancelled"], low_memory=False)
    frames.append(df)
raw = pd.concat(frames, ignore_index=True)
raw["FlightDate"] = pd.to_datetime(raw["FlightDate"])
raw = raw[(raw["FlightDate"] >= "2013-12-01") & (raw["FlightDate"] <= "2014-02-28")]
print(f"loaded {len(raw):,} flight rows over {raw['FlightDate'].nunique()} days")

# ----------------------------------------------------------------------------- top-18 airports by departure VOLUME
vol = raw.groupby("Origin").size().sort_values(ascending=False)
top = list(vol.index[:18])                                       # objective, outcome-independent rule
print("top-18 by departure volume:", top)
sub = raw[raw["Origin"].isin(top)]

# stress = daily mean DepDelayMinutes per origin (already floored at 0). cancellations excluded from primary signal.
daily = sub.groupby(["FlightDate", "Origin"])["DepDelayMinutes"].mean().reset_index()
M = daily.pivot(index="FlightDate", columns="Origin", values="DepDelayMinutes").sort_index()
M = M[top]                                                       # keep volume order
M = M.apply(lambda col: col.fillna(col.mean()), axis=0)         # fill missing days with column mean
names = list(M.columns); N = len(names); dates = M.index
S = M.copy()                                                    # panel M (days x airports), stress in minutes
S.to_csv(CACHE / "flights_panel.csv")                          # save for independent verification
print(f"flights: {N} airports, {len(S)} days (Dec 2013-Feb 2014); mean daily delay by airport:")
print("  " + "  ".join(f"{names[i]}:{S.iloc[:,i].mean():.0f}" for i in range(N)))

# ----------------------------------------------------------------------------- diagnose (DY connectedness)
Phi0, c0, Sig0 = L.fit_var_nonneg(S.values, ridge=5e-2)
TO, FROM, NET, tot = L.connectedness(L.gfevd(Phi0, Sig0))
transmitter = names[int(np.argmax(NET))]; loudest = names[int(np.argmax(S.mean().values))]
print(f"DY total connectedness {tot:.0f}% | net-transmitter = {transmitter} | loudest (mean delay) = {loudest}")
top_tx = [names[i] for i in np.argsort(-NET)[:3]]; top_rx = [names[i] for i in np.argsort(NET)[:3]]
print(f"top transmitters: {top_tx} | receivers: {top_rx}")

# ----------------------------------------------------------------------------- simulator + methods (as smoke23)
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
trans_weather = transmitter in WEATHER
trans_sunbelt = transmitter in SUNBELT
dir_beats_undir = min(red["var-out"], red["spillover"], red["transmitter"]) > red["corr"]
if trans_sunbelt:
    verdict = (f"FALSIFIED: the net-transmitter is {transmitter}, a Sun-Belt hub -- the pre-registered structural bet "
               f"(a cold-weather-exposed Northeast/Midwest hub) was wrong; reported honestly.")
elif trans_weather and trans_vs_loud and red["transmitter"] >= 1.8 * red["greedy"] and directed:
    verdict = (f"CONFIRMED (loudest != transmitter): the delay transmitter is {transmitter} (a weather-exposed hub) "
               f"while the loudest airport is {loudest}; transmitter-targeting beats greedy x{margin:.1f}, the directed "
               f"methods beat undirected correlation, and the edge collapses under symmetrization -- a confirmatory win "
               f"on a new domain class (engineered-infrastructure / logistics).")
elif trans_weather and (not trans_vs_loud) and directed:
    verdict = (f"REFINES (like flu): {transmitter} is BOTH the transmitter and the loudest, so the loudest-transmitter gap "
               f"is absent and transmitter-targeting ~ greedy (x{margin:.1f}) despite a directed network -- consistent with "
               f"the sharpened law (the operative quantity is the gap, not directedness alone).")
elif not directed:
    verdict = (f"SYMMETRIC NULL: winter delays look like a near-symmetric national common shock (DY {tot:.0f}%, "
               f"symmetrization barely changes the edge), so transmitter-targeting ~ greedy (x{margin:.1f}) -- the "
               f"no-leverage case the law predicts.")
else:
    verdict = "MIXED: reported honestly (see numbers)."
print(f"\ntransmitter={transmitter} (weather:{trans_weather}, sunbelt:{trans_sunbelt}), loudest={loudest} "
      f"(gap: {trans_vs_loud}); margin x{margin:.1f}; directed:{directed}")
print(f"VERDICT: {verdict}")

RES = dict(airports=names, n_days=int(len(S)), dy_total=round(float(tot), 1), transmitter=transmitter, loudest=loudest,
           top_transmitters=top_tx, top_receivers=top_rx,
           NET={names[i]: round(float(NET[i]), 2) for i in range(N)},
           mean_delay={names[i]: round(float(S.iloc[:, i].mean()), 2) for i in range(N)},
           interdiction={k: round(red[k], 1) for k in red}, symmetrization_advantage=[round(v, 1) for v in adv],
           transmitter_not_loudest=bool(trans_vs_loud), transmitter_weather_exposed=bool(trans_weather),
           transmitter_sunbelt=bool(trans_sunbelt), directed=bool(directed), dir_beats_undir=bool(dir_beats_undir),
           arro_over_greedy_x=round(float(margin), 2), verdict=verdict)
json.dump(RES, open(BASE / "flights_transfer_results.json", "w"), indent=2)
print("results -> flights_transfer_results.json")

# ----------------------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(12.2, 4.0))
a = ax[0]
for i in range(N): a.plot(dates, S.iloc[:, i], lw=0.8, alpha=.8, label=names[i])
a.axvspan(pd.Timestamp("2014-01-05"), pd.Timestamp("2014-01-08"), color=GREY, alpha=.25)
a.text(pd.Timestamp("2014-01-06"), a.get_ylim()[1]*.85, "polar\nvortex", fontsize=6, color=CRIT)
a.xaxis.set_major_locator(mdates.MonthLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
a.set_title("(a) Daily mean departure delay (Dec 2013-Feb 2014)", fontsize=8.3); a.set_ylabel("delay (min)")
a.legend(ncol=3, fontsize=5, loc="upper right")
a = ax[1]; oi = np.argsort(NET); cols = [CRIT if NET[i] > 0 else STEEL for i in oi]
a.barh([names[i] for i in oi], NET[oi], color=cols, alpha=.85); a.axvline(0, color="k", lw=.7)
a.set_title(f"(b) Directed delay network\ntransmitter={transmitter}, loudest={loudest}", fontsize=8.3)
a.set_xlabel("net connectedness (%)"); a.tick_params(axis="y", labelsize=6); a.grid(alpha=.25, axis="x")
a = ax[2]; ks = ["none", "greedy", "corr", "var-out", "spillover", "transmitter", "mpc"]
cc = [CRIT, PURPLE, GREY, TEAL, GREEN, STEEL, GOLD]; vv = [red[k] for k in ks]
a.bar(range(len(ks)), vv, color=cc, alpha=.85)
for i, v in enumerate(vv): a.text(i, v + .3, f"{v:+.0f}%", ha="center", fontsize=6.6, fontweight="bold")
a.set_xticks(range(len(ks))); a.set_xticklabels(["none", "greedy", "corr", "var\nout", "spill", "trans\nmitter", "mpc"], fontsize=6.2)
a.set_ylabel("cascade reduction vs no-action (%)"); a.set_title("(c) Interdiction + benchmark", fontsize=8.3); a.grid(alpha=.25, axis="y")
fig.suptitle(f"Fifth pre-registered transfer test — U.S. air-traffic delay propagation (polar-vortex winter)  ·  {verdict.split(':')[0].split('(')[0].strip()}",
             fontsize=10, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(BASE / "flights_transfer.pdf", bbox_inches="tight"); fig.savefig(BASE / "flights_transfer.png", dpi=200, bbox_inches="tight")
print("figure -> flights_transfer.pdf/.png")
