# Pre-registration — fourth out-of-domain transfer test: the 2023 Canadian wildfire-smoke air-quality disaster

**Registered: 2026-06-09, BEFORE computing the connectedness network or any interdiction outcome.**
(Per `DOMAIN_EXPANSION_PROTOCOL.md`. Track record so far: COVID (confirm), 1997 Asian FX (confirm), US flu
(refine). This fourth domain is a NEW domain *class* — an environmental disaster — chosen because the paper's
only prior environmental case (stratospheric ozone) was a single field with no directed network to interdict.)

## Domain & data
- **System:** the June 2023 Canadian wildfire smoke that blanketed the U.S. Northeast and Midwest, as a
  directed-contagion network of fine-particulate pollution (PM2.5) across U.S. states.
- **Data:** EPA AirData daily PM2.5 (parameter 88101), 2023, public, no key; ~15–18 Northeast/Midwest/Great-Lakes
  states; May–July 2023 (the smoke episodes plus baseline). Stress = state-mean daily PM2.5; weekly or daily.
- **Twin:** the same non-negative VAR + Diebold–Yılmaz pipeline, rescaled to a supercritical cascade for the
  interdiction test (as for COVID, flu and the Asian FX crisis).

## The prediction (committed before computing the network)
Wildfire smoke is *wind-transported*, so PM2.5 should propagate as a moving front rather than rise uniformly,
giving a **directed** network in which the upwind / earliest-hit states transmit to the downwind ones. This is a
genuine test, because the same event also has a strong *common-plume* component (a regional haze that lifts
everyone's PM2.5 together), which would instead make it symmetric.

- **My falsifiable bet (H-confirm):** the directed smoke front dominates, the net-transmitter is an upwind /
  early-hit / Great-Lakes-or-border state, and the loudest (peak-PM2.5) state is a different, downwind population
  centre — so loudest ≠ transmitter → **ARRO ≫ greedy** (a confirmatory win, like COVID and the Asian FX crisis).
- **Live alternative 1 (H-symmetric):** the plume is a near-symmetric common factor → no exploitable directed
  structure → **ARRO ≈ greedy** (the 2008-equity-crash null).
- **Live alternative 2 (H-refine):** directed, but the transmitter is itself the loudest state → **ARRO ≈ greedy**
  (the US-influenza refinement).

## Binding success criteria (fixed now)
1. **Net-transmitter** = an upwind / early-hit / northern-border state (not a peripheral unaffected one).
2. **If loudest ≠ transmitter:** ARRO (transmitter-targeting) beats greedy by ≥ 2× cascade reduction.
3. **If loudest = transmitter, or the network is near-symmetric:** ARRO ≈ greedy.
4. **Benchmark:** directed influence measures beat undirected centrality and the loudest-node heuristic.
5. **Symmetrization null:** the transmitter-targeting advantage collapses as the twin is symmetrized.

**Falsified iff:** ARRO ≫ greedy on a network whose advantage the symmetrization null cannot explain, or the
net-transmitter is an unaffected/peripheral state.

## Results (filled in by `smoke23_transfer.py` AFTER the above was committed)

### RESULT (2026-06-09) — CONFIRMED
- **Data:** 17 Northeast/Midwest/Great-Lakes states, 92 days (May–Jul 2023); EPA AirData daily PM2.5 (88101).
- **Diagnosis:** DY total connectedness **82%** (high — a strong common plume). Net-transmitter = **Wisconsin**
  (top transmitters WI, IL, IN — the upper-Midwest/Great-Lakes states the late-June plume reached first); loudest
  by mean PM2.5 = **Michigan**. The loudest-vs-transmitter GAP is **PRESENT** (WI transmits, MI is a loud
  receiver) — exactly the pre-registered bet (H-confirm).
- **Interdiction:** greedy +4%, transmitter-targeting **+10% (×2.5)**, mpc +10%; net-connectedness (+10) and raw
  out-strength (+10) beat undirected correlation centrality (+4) and the loudest-node heuristic (+4).
- **Symmetrization null:** transmitter−greedy advantage +6 → −8 pts as the twin is symmetrized — causally directed.
- **VERDICT — CONFIRMED.** Wildfire smoke is a directed, wind-transported front: the upper Midwest transmits
  while the loudest state (Michigan) receives; transmitter-targeting beats the loudest-node heuristic 2.5×, with
  directedness confirmed by the null. A confirmatory win on a **new domain class (environmental disaster)**.
- **NOTABLE:** connectedness here (82%) is as high as the 2008 equity crash (81%), yet ARRO *wins* — because the
  plume is *directed*, whereas the equity crash was symmetric. This directly reinforces that **directedness, not
  connectedness magnitude, decides controllability.**
- **Honest caveats:** modest absolute reductions (a short, near-stationary episode); the total-effect (spillover)
  measure under-discriminates here (+4) because the high common-plume component spreads it, while net-connectedness
  and out-strength isolate the leaders (+10); two smoke episodes (early-June Northeast, late-June Midwest) are
  aggregated, and the dominant directional structure is the Midwest plume.

*(see `smoke23_transfer_results.json` and `smoke23_transfer.png`)*
