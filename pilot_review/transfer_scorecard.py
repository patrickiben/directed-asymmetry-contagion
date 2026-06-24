#!/usr/bin/env python3
"""Six-test transfer scorecard for the NCS Article: the five prediction-first
out-of-domain tests (COVID is the held-out anchor, shown in the benchmark/validity
figures). Visualises the law: connectedness magnitude does NOT predict controllability
(smoke 82%, flu 78%, flights 77% cluster yet diverge); directed asymmetry does."""
import json, matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

R = str(Path(__file__).resolve().parent)
def load(f): return json.load(open(f"{R}/{f}"))

# (label, class, file, verdict)  -- ordered by domain class
rows = [
    ("Asian FX crisis\n1997, 9 currencies",      "financial",      "asia97_transfer_results.json",   "CONFIRM"),
    ("Wildfire smoke\n2023, 17 states",          "environmental",  "smoke23_transfer_results.json",  "CONFIRM"),
    ("Influenza\n2010-19, 14 states",            "epidemic",       "flu_transfer_results.json",      "REFINE"),
    ("Flight delays\n2013-14, 18 airports",      "infrastructure", "flights_transfer_results.json",  "FALSIFY"),
    ("Armed conflict\n2012-23, 6 states",        "geopolitical",   "conflict_transfer_results.json", "NULL"),
]
COL = {"CONFIRM": "#2C7A2C", "REFINE": "#D9A521", "FALSIFY": "#B23A3A", "NULL": "#B23A3A"}

data = []
for lab, cls, f, verdict in rows:
    d = load(f)
    inter = d["interdiction"]
    data.append(dict(lab=lab, cls=cls, verdict=verdict,
                     dy=d["dy_total"], greedy=inter["greedy"], transmit=inter["transmitter"],
                     transmitter=str(d["transmitter"]), loudest=str(d["loudest"]),
                     gap=bool(d.get("transmitter_not_loudest"))))

n = len(data)
y = np.arange(n)[::-1]  # top-to-bottom in listed order
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.4), gridspec_kw=dict(width_ratios=[1.0, 1.35], wspace=0.45))

# --- Panel A: network connectedness ---
for yi, d in zip(y, data):
    axA.barh(yi, d["dy"], color="#5B7DB1", height=0.55, zorder=3)
    axA.text(d["dy"] + 1.5, yi, f"{d['dy']:.0f}%", va="center", ha="left", fontsize=9, zorder=4)
axA.axvspan(75, 84, color="0.85", zorder=0)  # the high-connectedness cluster band
axA.text(79.5, n - 0.35, "smoke, flu, flights\ncluster here", fontsize=7.0, ha="center", va="bottom", color="0.35")
axA.set_xlim(0, 100); axA.set_ylim(-0.7, n - 0.3)
axA.set_yticks(y)
axA.set_yticklabels([d["lab"] for d in data], fontsize=8.2)
for tick, d in zip(axA.get_yticklabels(), data):
    tick.set_color(COL[d["verdict"]])
axA.set_xlabel("Diebold–Yılmaz total\nconnectedness (%)", fontsize=9)
axA.set_title("a   How connected", fontsize=10, loc="left", fontweight="bold")
axA.spines[["top", "right"]].set_visible(False)

# --- Panel B: cascade reduction, loudest-node heuristic vs transmitter-targeting ---
h = 0.33
for yi, d in zip(y, data):
    axB.barh(yi + h/1.7, d["transmit"], color=COL[d["verdict"]], height=h, zorder=3, label="_")
    axB.barh(yi - h/1.7, d["greedy"], color="0.72", height=h, zorder=3, label="_")
axB.set_xlim(0, 108); axB.set_ylim(-0.7, n - 0.3)
axB.set_yticks([])
axB.set_xticks([0, 20, 40, 60, 80, 100])
axB.set_xlabel("Cascade reduction vs no action (%)\non the calibrated supercritical VAR surrogate", fontsize=9)
axB.set_title("b   Does transmitter-targeting beat the loudest-node rule", fontsize=10, loc="left", fontweight="bold")
axB.spines[["top", "right", "left"]].set_visible(False)
# verdict chips at far right, clear of the bars
for yi, d in zip(y, data):
    axB.text(107, yi, d["verdict"], va="center", ha="right", fontsize=7.6, fontweight="bold",
             color="white", bbox=dict(boxstyle="round,pad=0.25", fc=COL[d["verdict"]], ec="none"))

legend = [Patch(fc="0.72", label="loudest-node heuristic"),
          Patch(fc="#444444", label="transmitter-targeting"),
          Patch(fc=COL["CONFIRM"], label="confirm"),
          Patch(fc=COL["REFINE"], label="refine"),
          Patch(fc=COL["FALSIFY"], label="falsify / null")]
axB.legend(handles=legend, loc="lower right", bbox_to_anchor=(1.0, -0.40), ncol=3,
           fontsize=7.0, frameon=False, handlelength=1.2, columnspacing=1.1)

fig.suptitle("Five prediction-first transfer tests: directed asymmetry, not connectedness magnitude, decides controllability",
             fontsize=10.5, y=1.0, x=0.02, ha="left", fontweight="bold")
fig.savefig(f"{R}/transfer_scorecard.pdf", bbox_inches="tight", dpi=200)
fig.savefig(f"{R}/transfer_scorecard.png", bbox_inches="tight", dpi=150)
print("wrote transfer_scorecard.pdf/.png")
for d in data:
    print(f"  {d['cls']:<15} dy={d['dy']:>5.1f}  greedy={d['greedy']:>5.1f}  transmit={d['transmit']:>5.1f}  {d['transmitter']}->{d['loudest']} gap={d['gap']}  {d['verdict']}")
