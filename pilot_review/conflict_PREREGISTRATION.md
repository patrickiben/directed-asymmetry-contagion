# Pre-registration — sixth out-of-domain transfer test: West-African / Sahel armed-conflict diffusion (a geopolitical / conflict domain class, Tier IV)

**Registered: 2026-06-09, BEFORE computing the connectedness network or any interdiction outcome.**
(Per `DOMAIN_EXPANSION_PROTOCOL.md`. Track record so far: COVID (confirm, epidemic), 1997 Asian FX (confirm,
financial), 2023 wildfire smoke (confirm, environmental), US flu (refine), US flight delays (FALSIFY,
infrastructure). This sixth domain is a NEW domain *class* — a **geopolitical / armed-conflict** network, the
framework's highest tier (IV, sociocognitive/geopolitical), never yet validated with a real transfer test.)

**Framing & ethics.** These are real human tragedies, not abstract cascades. The "stress" variable is monthly
conflict fatalities, and the "controller" is a stand-in for **where preventive resources have the most systemic
leverage** — early diplomatic engagement, mediation, peacekeeping, stabilization aid — *targeting the systemic
transmitter of regional violence rather than the loudest current theater*. This is retrospective diagnosis on a
calibrated surrogate (the same caveat as every other twin in the paper), the purpose for which UCDP/ACLED and
the conflict-spillover literature exist; it is not an operational targeting recommendation.

## Domain & data
- **System:** the West-African / Lake-Chad / Sahel region as a directed armed-conflict-contagion network.
  Conflict spillover ("bad neighborhoods", the conflict trap) is a well-established *directed* phenomenon: an
  insurgency that ignites in one state exports violence — fighters, weapons, tactics, refugees, cross-border
  raids — to its neighbors.
- **Data:** UCDP Georeferenced Event Dataset (GED) v24.1, public, **no key** (`ucdp.uu.se/downloads/ged/ged241-csv.zip`).
  349,733 events, 1989–2023; columns include `date_start`, `country`, `region`, `best` (best fatality estimate).
- **Window: 2012-01 .. 2023-12** (the central-Sahel insurgency era; the north-Mali takeover began 2012), monthly
  → ~144 months.
- **Units (objective, outcome-INDEPENDENT rule fixed now):** from the pre-specified West-Africa + Lake-Chad
  candidate list {Mali, Burkina Faso, Niger, Nigeria, Chad, Cameroon, Côte d'Ivoire, Benin, Togo, Ghana,
  Senegal, Guinea, Mauritania}, keep every country with **≥ 100 conflict events** in the window. (No hand-picking
  from the results; the threshold is the inclusion rule.)
- **Stress variable:** monthly total fatalities (`best`) per country (already ≥ 0).
- **Twin/pipeline:** IDENTICAL to the other transfer scripts — non-negative VAR + Diebold–Yılmaz gfevd +
  connectedness, rescaled to a supercritical cascade, run_interdiction with the SAME controllers (none, greedy,
  corr, var-out, spillover, transmitter, mpc) and the SAME symmetrization-null sweep.

## The prediction (committed before computing the network)
The central-Sahel insurgency began in **Mali** (2012: the Tuareg/jihadist takeover of the north, French
Operation Serval 2013) and spread outward — most lethally into **Burkina Faso**, which became the regional
epicenter by ~2019–2021, and into western **Niger**. Conflict therefore should propagate as a **directed** front
in which Mali (the origin) is the net-transmitter, while the loudest country by total fatalities is a *different*
state that received and escalated the violence. This is a genuine test because two confounds could break it:
(i) the region also contains a **second** insurgency — Boko Haram in the Lake-Chad basin — whose origin and
loudest theater are *both* **Nigeria**, which could dominate the network as a coincident transmitter==loudest
(a flu-style refinement); and (ii) the violence could be a near-symmetric regional common shock (a null).

- **My falsifiable bet (H-confirm):** the net-transmitter is **Mali**, and the loudest country by total
  fatalities is **not Mali** (Burkina Faso or Nigeria) → loudest ≠ transmitter, gap PRESENT →
  **transmitter-targeting ≫ loudest-node heuristic** (≥ 2× cascade reduction), with directedness confirmed by
  the symmetrization null. A confirmatory win like COVID, the Asian FX crisis and the wildfire smoke.
- **Live alternative 1 (H-refine):** directed, but the net-transmitter is *itself* the loudest country (most
  plausibly Nigeria, origin AND loudest of Boko Haram) → ARRO ≈ greedy (the influenza refinement).
- **Live alternative 2 (H-symmetric / falsify):** near-symmetric regional common shock, OR a transmitter that is
  a peripheral/unaffected state, OR transmitter ≫ greedy with no directed cause in the null → ARRO ≈ greedy / the
  prediction is falsified (the flight-delay outcome).

## Binding success criteria (fixed now)
1. **Net-transmitter = Mali** (the pre-committed origin), not a peripheral/low-violence state.
2. **If loudest ≠ transmitter:** transmitter-targeting beats the loudest-node heuristic by ≥ 2× cascade reduction.
3. **If loudest = transmitter (e.g. Nigeria), or the network is near-symmetric:** ARRO ≈ greedy (refine / null).
4. **Benchmark:** directed influence measures (net-connectedness, raw out-strength, spillover) beat undirected
   correlation centrality and the loudest-node heuristic.
5. **Symmetrization null:** the transmitter-targeting advantage collapses as the twin is symmetrized.

**Falsified iff:** the net-transmitter is a peripheral/low-violence state, or is **not** Mali while the result is
claimed as a confirmation; or transmitter-targeting ≫ greedy on a network whose advantage the symmetrization null
cannot explain (an advantage with no directed cause).

## Results (filled in by `conflict_transfer.py` AFTER the above was committed)

### RESULT (2026-06-09)

**Verdict: NULL / FALSIFY — the committed Mali-origin bet is falsified, and the network shows no exploitable
directed leverage. The law's null arm, on a sixth domain class (geopolitical / armed conflict, Tier IV).**

`conflict_transfer.py` ran cleanly first try on UCDP GED v24.1 (349,733 events; 222,839 in the 2012-01..2023-12
window; 144 months). The ≥100-event inclusion rule passed 6 of 13 candidates — Mali (1998 events), Burkina Faso
(1504), Niger (497), Nigeria (6000), Chad (142), Cameroon (1908); Côte d'Ivoire (10, UCDP spelling "Ivory
Coast"), Benin (36), Togo (29), Ghana (14), Senegal (18), Guinea (13) and Mauritania (4) fell below threshold.

- **Net-transmitter = Niger** (NET = +13.56; TO 33.65 > FROM 20.08), **not Mali**. **Criterion 1 FAILS.**
- **Mali is a net RECEIVER** (NET = −7.11; FROM 16.8 > TO 9.7). The pre-committed origin is contradicted directly:
  Mali absorbs more regional violence than it exports over the full window. `transmitter_is_mali = False`.
- **Loudest theater = Nigeria** (51,091 fatalities over the window, far above Mali's 13,646), driven by the
  Lake-Chad / Boko Haram insurgency. Loudest ≠ transmitter, but with no directed leverage the gap is moot.
- **Network is not directed:** DY total connectedness = 20.6% (low — the region behaves like six weakly-coupled
  local insurgencies, not one tightly-coupled contagion front). The symmetrization-null advantage starts at only
  **+1.2 pp** (well below the ≥5 pp directedness floor) and decays monotonically to ~0 ([1.2, 0.8, 0.3, 0.0,
  −0.1]) — there is no directed edge for a controller to exploit. `directed = False`. **Criterion 5: the
  advantage is already negligible at full directedness, so the collapse-under-symmetrization signature is absent.**
- **Interdiction confirms zero practical leverage:** every controller reduces the cascade by only 0–2% (greedy
  +0.4%, transmitter +1.6%, mpc +2.1%). The headline ×4.52 transmitter-vs-greedy margin is a **divide-by-near-zero
  artifact** (1.6% / 0.4%), NOT a directed win; with `directed = False` it is non-load-bearing and we footnote it.
- **Directed measures still beat undirected** (Criterion 4 holds in the ordering: spillover/transmitter/var-out >
  corr > greedy), but on absolute reductions so small the ordering carries no practical content.

**Why this is NOT `CONFIRMED_LAW_MISSED_ORIGIN`.** That verdict requires a *verifiably directed* win (transmitter
≳ 1.8× greedy on a network whose advantage the symmetrization null explains). Both conditions fail: the network is
not directed (adv[0] = 1.2 < 5 pp) and the margin is a small-denominator artifact. So the honest reading is the
**NULL (no-leverage) branch the law explicitly admits** — and, independently, the **Mali origin prediction is
falsified on origin** (Mali is a receiver). The law's logic survives; my specific bet about *this* domain's
structure was wrong on both falsification grounds.

**Robustness / sensitivity (honest — the transmitter identity is NOT robust):**
- Ridge sweep (0.01/0.05/0.10/0.20): transmitter STABLE = Niger, loudest = Nigeria, DY ≈ 20.6% throughout.
- Variance-stabilizing transforms (log1p / sqrt): transmitter flips Niger → **Burkina Faso** (DY ≈ 24%).
- Quarterly aggregation (T = 48): transmitter flips to **Nigeria** — which is ALSO the loudest (a flu-style
  coincident-transmitter reading, no gap); DY rises to 34.6%.
- Sub-windows: 2012–2017 → **Cameroon** (DY 22.8%); 2018–2023 → Niger (DY 14.4%); Sahel-only {Mali, BF, Niger} →
  Burkina Faso; Lake-Chad-only {Nigeria, Chad, Cameroon} → Cameroon; drop the single largest country-month
  (Nigeria 2015-02) → Burkina Faso.
- Across **every** valid cut FOUR of six countries (Niger, Burkina Faso, Cameroon, Nigeria) each win the
  transmitter slot, and the net-transmitter is **never Mali** in any non-degenerate window. (Apparent Mali-leads in
  2012–2014/2012–2015 are a NaN-argmax artifact: partner countries have all-zero fatality columns that early, so
  the GFEVD is singular and `argmax` defaults to index 0 = Mali. The 2012 Mali-onset period is therefore
  *untestable* for a directed origin — partner fatalities ramp only ~2015+.)

**Caveats (binding for the manuscript):**
- **Thin monthly T:** 144 monthly obs for an N = 6 VAR(1) is workable but thin.
- **Fatality burstiness / heavy tails:** monthly counts are dominated by single-event massacres, so a sparse
  non-negative VAR(1) on raw counts is noise-dominated and the GFEVD shares are unstable — consistent with the
  near-null and with the transmitter fragility above.
- **Two-insurgency confound (material):** the panel pools the central-Sahel insurgency (Mali → Burkina Faso →
  Niger) with the largely-decoupled Lake-Chad / Boko Haram insurgency (origin AND loudest both Nigeria). Pooling
  was pre-registered, but it is the most likely structural reason the network reads near-symmetric and splits the
  transmitter role; the monthly net-transmitter (Niger) sits on the Sahel/Lake-Chad seam.
- **Structural vs. controllable asymmetry:** the VAR kernel itself is *not* literally symmetric
  (|Φ−Φᵀ|/|Φ| = 0.739); it is near-symmetric only in the *controllable-leverage* sense — the directed edge is too
  small to exploit. We describe the result as low-leverage / weakly-directed-in-the-exploitable-sense, not as a
  structurally symmetric matrix.
- **Ethics / surrogate framing (unchanged):** these are real human tragedies; the "controller" is a stand-in for
  where *preventive* resources (early diplomacy, mediation, peacekeeping, stabilization aid) have the most systemic
  leverage — the transmitter, not the loudest theater — and this is retrospective diagnosis on a calibrated
  surrogate, NOT an operational targeting recommendation.

**Net tally after six tests:** three confirmations (COVID/epidemic, 1997 Asian FX/financial, 2023 wildfire
smoke/environmental), one refinement (US influenza — loudest == transmitter), and **two falsifications** (US
flight delays/engineered-infrastructure; this West-African/Sahel conflict case/geopolitical). Five domain classes;
the conflict case is the first geopolitical test and lands on the law's admitted null arm.
