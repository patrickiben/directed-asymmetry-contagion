# Pre-registration — fifth out-of-domain transfer test: U.S. air-traffic delay propagation (an engineered-infrastructure / logistics network)

**Registered: 2026-06-09, BEFORE computing the connectedness network or any interdiction outcome.**
(Per `DOMAIN_EXPANSION_PROTOCOL.md`. Track record so far: COVID (confirm, epidemic), 1997 Asian FX (confirm,
financial), 2023 wildfire smoke (confirm, environmental), US flu (refine). This fifth domain is a NEW domain
*class* — an **engineered-infrastructure / logistics** network — chosen because all four prior tests are
natural, biological or financial contagions; none is a human-built transport system whose directed structure
comes from operational coupling rather than physics, biology or markets.)

## Domain & data
- **System:** the U.S. national airspace as a directed delay-contagion network. Departure delays seed at a
  disrupted hub (weather, congestion) and propagate **downstream** along aircraft rotations and crew/connection
  banking — a hub that goes down in the morning exports delay to the airports its aircraft fly to that afternoon.
  Air-traffic flow management already practices *anticipatory* interdiction here: Ground Delay Programs and
  ground stops hold traffic *bound for or originating at* the disrupted transmitter hub, rather than reacting at
  whichever airport currently shows the largest delay.
- **Data:** Bureau of Transportation Statistics (BTS) Reporting-Carrier On-Time Performance, public, no key
  (`transtats.bts.gov/PREZIP/...` monthly zips). **Window: 1 Dec 2013 – 28 Feb 2014** (the "polar-vortex"
  winter, a season with strong, geographically-localised weather disruption — chosen a priori for high directed
  signal, before any network was computed).
- **Units:** the **top ~18 airports by departure volume** in the window (an objective, outcome-independent
  selection rule — not hand-picked from the results).
- **Stress variable:** daily mean departure delay in minutes per origin airport (`DepDelayMinutes`, already
  floored at 0). Cancellations handled as a robustness note, not the primary signal.
- **Twin:** the same non-negative VAR + Diebold–Yılmaz pipeline, rescaled to a supercritical cascade for the
  interdiction test (identical to COVID, flu, the Asian FX crisis and the wildfire-smoke test).

## The prediction (committed before computing the network)
Delays propagate as a **directed** front: a weather-exposed hub that is hit first exports delay downstream along
its rotations, so the network should be directionally asymmetric, not a symmetric national common shock. The
genuine test is that the same winter also has *common-mode* storms that raise delays nationwide at once (which
would instead make the network symmetric), and that the busiest hubs are congested receivers as well as
transmitters.

- **My falsifiable bet (H-confirm):** the net-transmitter is a **cold-weather-exposed Northeast/Midwest hub**
  (one of ORD, EWR, LGA, JFK, BOS, DTW, MSP, DEN, PHL), **not** a Sun-Belt hub (ATL, DFW, IAH, PHX, LAS, LAX,
  MCO, CLT). The loudest airport by mean delay is a *different*, congested hub that is a net **receiver**
  (loudest ≠ transmitter, gap PRESENT) → **transmitter-targeting ≫ loudest-node heuristic** (≥ 2× cascade
  reduction). A confirmatory win, like COVID, the Asian FX crisis and the wildfire smoke.
- **Live alternative 1 (H-symmetric):** winter delays are a near-symmetric national common shock → no
  exploitable directed structure → **ARRO ≈ greedy** (the 2008-equity-crash null).
- **Live alternative 2 (H-refine):** directed, but the net-transmitter is *itself* the loudest hub → **ARRO ≈
  greedy** (the U.S.-influenza refinement: the operative quantity is the loudest-vs-transmitter gap, not
  directedness alone).

## Binding success criteria (fixed now)
1. **Net-transmitter** = a cold-weather-exposed Northeast/Midwest hub (per the list above), not a Sun-Belt hub
   and not a peripheral low-volume airport.
2. **If loudest ≠ transmitter:** transmitter-targeting beats the loudest-node heuristic by ≥ 2× cascade
   reduction.
3. **If loudest = transmitter, or the network is near-symmetric:** ARRO ≈ greedy (refinement / null).
4. **Benchmark:** directed influence measures (net-connectedness, raw out-strength, spillover) beat undirected
   correlation centrality and the loudest-node heuristic.
5. **Symmetrization null:** the transmitter-targeting advantage collapses as the twin is symmetrized.

**Falsified iff:** the net-transmitter is a Sun-Belt or peripheral airport; or transmitter-targeting ≫ greedy on
a network whose advantage the symmetrization null cannot explain (an advantage with no directed cause).

## Results (filled in by `flights_transfer.py` AFTER the above was committed)

### RESULT (2026-06-09) — FALSIFIED

- **Data:** top-18 U.S. airports by departure volume, 90 days (1 Dec 2013 – 28 Feb 2014, the polar-vortex
  winter); BTS On-Time Performance, daily mean `DepDelayMinutes` per origin. 1,419,290 flight rows; top-18 by an
  objective outcome-independent volume rule.
- **Diagnosis:** DY total connectedness **77%** (HIGH — the national airspace co-moves strongly). Net-transmitter
  = **LAS (Las Vegas)**; top transmitters **LAS, PHX, LAX** (all Sun-Belt / West-Coast, warm). Loudest by mean
  delay = **ORD (28 min)**, with EWR (26) and JFK (25) — the weather-exposed hubs the bet named — but these sit
  near-zero or **negative on NET**: they are congested **receivers**, not transmitters.
- **Loudest-vs-transmitter gap:** PRESENT but **wrong-signed for the bet** — the loud hubs (ORD/EWR/JFK) are
  receivers, but the transmitter is a Sun-Belt airport, not the committed cold-weather hub.
- **Interdiction:** every controller barely moves the cascade — greedy +1.2%, corr +1.9%, var-out +4.9%,
  spillover **+7.1% (best)**, transmitter +5.1%, mpc +5.5%. transmitter-vs-greedy margin = **×4.21** — a
  small-denominator **ratio artifact** (5.1% / 1.2%, both near the noise floor), NOT a real directed advantage.
- **Symmetrization null:** transmitter−greedy advantage = **[3.9, −0.8, −5.3, −4.9, −4.3]** — adv0 = 3.9 is
  **below the ≥5 directedness floor**, the null is **non-monotone** and goes **negative** under symmetrization
  (the opposite of a collapsing directed edge). `directed = False`.
- **VERDICT — FALSIFIED**, on two independent pre-registered clauses. (1) The net-transmitter is **LAS, a
  Sun-Belt airport** in the pre-committed set {ATL,DFW,IAH,PHX,LAS,LAX,MCO,CLT}, and the entire top-3 is
  Sun-Belt/West-Coast — directly violating "Falsified iff the net-transmitter is a Sun-Belt or peripheral
  airport." (2) transmitter ≫ greedy (×4.2) with **no directed cause** in the null (adv0 = 3.9 < 5; edge goes
  negative under symmetrization) — the second clause, "transmitter ≫ greedy with an advantage the symmetrization
  null cannot explain."
- **DIRECTEDNESS, NOT CONNECTEDNESS MAGNITUDE, IS WHAT FAILS.** High DY (77%) coexists with essentially no
  directed leverage: the cascade is a near-symmetric national common shock (winter storms hitting many hubs at
  once), the 2008-equity-style symmetric null behind a big connectedness number — the mirror image of the
  wildfire-smoke case, where equally high connectedness (82%) *was* directed and the controller won.
- **Robustness (skeptic-verified, 3/3 independent reproductions, no leakage):** net-transmitter = LAS is
  invariant across ridge 1e-3..1.0, first-60/last-60/full sub-windows, top-15 (by volume and by mean delay),
  2-/3-day aggregation and log1p; the verdict does NOT hinge on the polar-vortex week (dropping it flips
  LAS→PHX, still Sun-Belt). The only weather-hub result (DEN) appears solely under weekly aggregation, a
  degenerate T=14 < N=18 over-parameterized fit that is itself unstable (flips to PHX/MCO under reasonable
  ridge/window changes) and does not overturn the verdict.
- **Honest caveats:** (i) thin panel — T≈90 days vs N=18 airports is a low T/N ratio; the nonneg-VAR(1) +
  ridge(5e-2) is regularized but the directed structure is fragile at this resolution. (ii) The ×4.2 margin must
  NOT be read as a win — it is a ratio of two tiny near-noise-floor effects with `directed=False`. (iii) The null
  is non-monotone (overshoots negative), stronger evidence against a directed edge than a clean monotone decay.
  (iv) Cancellations were excluded per spec; they are highest at exactly the weather hubs (EWR/LGA/ORD ~9–10%)
  and including them could plausibly shift NET toward the Northeast — a **registered robustness note, not run**
  in the primary stress (skeptic imputation stress at 180/300 min kept LAS as transmitter). (v) Unmodeled
  exogeneity confound — warm low-weather airports may score as transmitters because their delays are
  idiosyncratic/exogenous (mechanical, congestion, ATC-flow) and read by the VAR as sources, while congested
  weather hubs import variance.

*(see `flights_transfer_results.json` and `flights_transfer.png`)*
