# Pre-registration — third out-of-domain transfer test: the 1997 Asian financial crisis

**Registered: 2026-06-09, BEFORE computing the connectedness network or any interdiction outcome.**
(Per `DOMAIN_EXPANSION_PROTOCOL.md`. First test = COVID-19 (confirmed); second = US influenza (refined the law,
loudest==transmitter). This third domain is sought as a candidate *confirmatory* case — a crisis with a clear
quiet origin distinct from its loudest casualties.)

## Domain & data
- **System:** the 1997 Asian financial crisis as cross-currency FX contagion (local currency per US dollar).
- **Data:** daily exchange rates from FRED (public, no key), 1996–1998, for the available Asian currencies
  (Thailand, Korea, Malaysia, Singapore, Japan, Taiwan, Hong Kong, India, China; Indonesia/Philippines daily
  series are not in FRED). Stress = percent depreciation from the 1996 pre-crisis baseline; weekly.
- **Twin:** the same non-negative VAR + Diebold–Yılmaz pipeline, rescaled to a supercritical cascade for the
  interdiction test (as for COVID and flu).

## The prediction (committed before computing the network)
The crisis **originated in Thailand** — the baht was floated on 2 July 1997 and the devaluation then propagated
to Malaysia, Indonesia, the Philippines and Korea. So the **directed structure should make Thailand (and/or the
early-crisis ASEAN currencies) the net-transmitter(s).**

The directed-asymmetry law, in its sharpened form, predicts a large anticipatory advantage **iff** the loudest
(largest-depreciation) currency is **distinct** from the transmitter:
- **My falsifiable bet (H-confirm):** the loudest casualty will be a *larger* economy that *received* the
  contagion (the Korean won, which collapsed in November 1997), **distinct** from the quiet origin (Thailand).
  If so, loudest ≠ transmitter → **ARRO ≫ greedy** (a second confirmatory win, like COVID).
- **The live alternative (H-refine):** Thailand is *both* the origin and among the loudest depreciations, so the
  loudest-vs-transmitter gap is small → **ARRO ≈ greedy** (a refinement, like the US-influenza result).

## Binding success criteria (fixed now)
1. **Net-transmitter** = Thailand or another early-crisis ASEAN currency (not Japan/India/HK, the late/peripheral
   ones).
2. **If loudest ≠ transmitter:** ARRO (transmitter-targeting) beats greedy by ≥ 2× cascade reduction.
3. **If loudest = transmitter:** ARRO ≈ greedy (the refinement case).
4. **Benchmark:** directed influence measures beat undirected centrality and the loudest-node heuristic.
5. **Symmetrization null:** the transmitter-targeting advantage collapses as the twin is symmetrized.

**Falsified iff:** the net-transmitter is a late/peripheral currency (Japan/India), or ARRO ≫ greedy on a
network whose advantage the symmetrization null cannot explain.

## Results (filled in by `asia97_transfer.py` AFTER the above was committed)

### RESULT (2026-06-09) — CONFIRMED
- **Data:** 9 Asian currencies (FRED daily FX, 1996–1998, weekly; Indonesia/Philippines unavailable), 135 weeks.
- **Diagnosis:** DY total connectedness 54%. Net-transmitter = **Thailand** (top transmitters Thailand, Singapore,
  Taiwan); loudest by mean depreciation = **Korea** (the won; peak ~125%). The loudest-vs-transmitter GAP is
  **PRESENT** — Thailand transmits, Korea is a net *receiver* — exactly the pre-registered bet (H-confirm).
- **Interdiction:** greedy +6%, transmitter-targeting **+13% (×2.3)**, mpc +16%; directed measures (var-out +11,
  spillover +12, transmitter +13) above undirected correlation centrality (+10) and the loudest-node heuristic (+6).
- **Symmetrization null:** transmitter−greedy advantage +7 → −3 pts as the twin is symmetrized — causally directed.
- **VERDICT — CONFIRMED.** The crisis transmitter is the quiet origin (Thailand) while the loudest casualty is a
  larger economy that received the contagion (Korea); transmitter-targeting beats the loudest-node heuristic by
  2.3×, and the symmetrization null confirms directedness. A second COVID-style confirmation, on a genuinely new
  (financial) domain, with a correctly pre-registered structural prediction.
- **Honest caveats:** absolute reductions are modest (the rescaled FX twin is near-stable), and undirected
  centrality is only slightly below the directed methods here; the robust result is transmitter ≫ loudest (×2.3),
  with directedness confirmed by the symmetrization null. Indonesia/Philippines (the worst movers) were not in FRED.

*(see `asia97_transfer_results.json` and `asia97_transfer.png`)*
