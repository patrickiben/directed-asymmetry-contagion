# Prediction-first pre-registration — directed-asymmetry diagnostic (new prospective tests)

**Purpose.** This is a *register-before-you-compute* document. It commits, in advance, the predicted directed
structure and the falsification criteria for three NEW contagion networks that are **not** in the current manuscript.
Registering it on OSF (which freezes a public timestamp) converts the paper's "prediction-first, not pre-registered"
protocol into a genuinely prospective one for the next round, and removes the single caveat a methods referee leans on.

**HARD RULE for this to count.** Do **not** fit any VAR, compute any connectedness, or look at the multivariate
panels for these three systems until *after* this document is registered. If you have already inspected a panel,
drop that test and substitute a fresh one. These predictions are drafted from public, documented facts about each
episode and the directed-asymmetry regularity — they are genuine forecasts and may be wrong (that is the point; the
manuscript already reports two falsifications).

**How to register (5 minutes).** osf.io → New Project → upload this file → *Registrations* → "Register" with the
"OSF Preregistration" template (or "Open-Ended Registration"). Cite the resulting `osf.io/49kn7` URL in the manuscript's
prediction-first paragraph and in the Data/Code-availability statement. **Then** run each test with the frozen pipeline
below and add the confirm/refine/falsify outcome to the ledger.

---

## The frozen pipeline (identical to the manuscript; no degrees of freedom)
For each network: build the multivariate weekly (or per-event-step) **stress** series (declines/incidence/detections,
clipped at zero); fit a **non-negative VAR(1)** with ridge regularization; compute the **generalized FEVD
Diebold–Yılmaz** connectedness; the **transmitter** is the unit of maximum net (to − from) connectedness, read at the
**horizon-integrated** net connectedness (`∫ net_i(H) dH`, H∈{2…15}) so the horizon is not a free choice; the
**loudest** unit is the one of largest cumulative realized stress. Interdict on the surrogate rescaled to ρ=1.05–1.06,
comparing transmitter-targeting vs the reactive loudest-node rule; isolate directedness with the **symmetrization
null** (each Φ(α) re-rescaled to the common ρ). Effect sizes are surrogate quantities, reported with that qualifier.

## The decision rule (committed)
- **Confirm** if (a) the predicted transmitter is the net-transmitter, (b) transmitter ≠ loudest as predicted, and
  (c) transmitter-targeting beats the loudest-node rule on the surrogate with the advantage collapsing under
  symmetrization.
- **Refine** if the network is strongly directed but the transmitter coincides with the loudest unit (no gap), so the
  advantage is near-null for the *structural* reason the regularity already admits.
- **Falsify** if the predicted transmitter is a net *receiver*, or the network is near-symmetric / weakly coupled
  (total connectedness low or the symmetrization null flat), so no directed edge exists to exploit.
- A directed network on which transmitter-targeting **loses**, or a symmetric one on which it **wins with an advantage
  that survives symmetrization**, breaks the regularity outright.

---

## Test 1 — 2023 U.S. regional-banking contagion (financial)
- **Units / data:** daily (→weekly) equity prices of ~10–12 U.S. regional banks plus the KRE regional-bank ETF,
  1 Jan – 30 Jun 2023 (public; Stooq/FRED/any provider). Stress = weekly % price decline, clipped at zero.
- **Documented origin:** **Silicon Valley Bank** (collapsed 10 Mar 2023; Signature Bank 12 Mar) — the trigger.
- **Documented loudest (worst realized):** **First Republic** (lost the most and failed last, 1 May 2023), i.e.
  downstream of the initial shock.
- **COMMITTED PREDICTION:** SVB (and the earliest-failing banks) are **net-transmitters**; First Republic is a
  **net-receiver**; transmitter ≠ loudest, so transmitter-targeting beats loudest-node and the edge collapses under
  symmetrization → **confirm**.
- **Genuine risk of falsification:** the 2023 episode may instead read as a **near-symmetric common shock** (a
  duration/rate-driven repricing hitting all regionals at once), in which case the network is symmetric and every
  controller sits near the noise floor → **falsify** (the flight-delay failure mode). I do not know which; that is the
  test.

## Test 2 — 2024 U.S. H5N1 spread in dairy cattle (epidemic, spatial)
- **Units / data:** state-level weekly counts of H5N1-affected dairy herds, USDA APHIS, Mar 2024 – end 2024 (public).
  Stress = weekly new-affected-herd counts (or per-capita), clipped at zero; top ~12 affected states.
- **Documented origin:** **Texas** (first confirmed dairy-cattle detections, late Mar 2024, Panhandle).
- **Documented loudest (worst realized):** **California** (largest dairy industry; large later-2024 outbreak), i.e.
  affected later and downstream of the initial Texas/Midwest spread.
- **COMMITTED PREDICTION:** Texas (and the early-affected states) are **net-transmitters**; California is a
  **net-receiver**; transmitter ≠ loudest → transmitter-targeting beats loudest-node, edge collapses under
  symmetrization → **confirm**.
- **Genuine risk:** cattle-movement diffusion may be too sparse / thin at monthly–state resolution to fit a stable
  directed VAR (the Sahel-conflict failure mode) → **falsify** via low connectedness / unstable transmitter.

## Test 3 — 2024 Canadian-wildfire smoke over the U.S. (environmental) — replication
- **Units / data:** daily (→ event-window) PM₂.₅ across ~15 northern/eastern U.S. states during the 2024 Canadian
  wildfire-smoke intrusions, EPA AirData (public). Stress = daily PM₂.₅. **Different fire year from the manuscript's
  2023 smoke network — a replication of the mechanism, not the same data.**
- **Documented origin:** the **upwind entry states** (upper-Midwest / Great-Lakes corridor where Canadian smoke first
  arrives), which transmit downwind.
- **Documented loudest (worst realized):** a **downwind, high-population state** that accumulates the highest mean
  PM₂.₅ but is not the source.
- **COMMITTED PREDICTION:** an upwind state is the **net-transmitter**; the loudest downwind state is a
  **net-receiver**; transmitter ≠ loudest → transmitter-targeting beats loudest-node, edge collapses under
  symmetrization → **confirm** (this is the manuscript's confirmed wildfire mechanism; a *failure* to replicate would
  be informative and must be reported).
- **Genuine risk:** a weaker 2024 smoke season may give a low-connectedness, near-symmetric network → **refine/falsify**.

---

## Ledger (fill after running each registered test)
| Test | Predicted transmitter | Predicted loudest | Net-transmitter (observed) | Loudest (observed) | DY total | Advantage (T−loud) | Symmetrization collapse | Outcome |
|---|---|---|---|---|---|---|---|---|
| 1 — 2023 banks | SVB | First Republic | | | | | | |
| 2 — 2024 H5N1 | Texas | California | | | | | | |
| 3 — 2024 smoke | upwind state | downwind state | | | | | | |

**Reporting commitment.** Whatever the outcomes — including falsifications or failed replications — they are added to
the manuscript's confirm/refine/falsify record (or a follow-up) exactly as the existing five prediction-first tests
are. The value of this registration is that it is timestamped *before* the networks are computed.
