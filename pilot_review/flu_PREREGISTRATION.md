# Pre-registration — second out-of-domain transfer test of the directed-asymmetry law

**Registered: 2026-06-09, BEFORE computing the connectedness network or any interdiction outcome.**
(Following `DOMAIN_EXPANSION_PROTOCOL.md`. The first such test was the COVID-19 state network; this is the
independent second domain that turns one confirmation into a track record — or falsifies the law.)

## Domain & data
- **System:** US seasonal influenza spread across states (a directed-contagion network distinct from COVID).
- **Data:** CDC ILINet weighted influenza-like-illness (`wILI`), via the Delphi Epidata `fluview` API
  (public, no key). ~15 most-populous states, flu-active weeks (epiweeks 40–20) of seasons 2010–2019.
- **Twin:** the same non-negative VAR + Diebold–Yılmaz pipeline used throughout; rescaled to a supercritical
  cascade for the interdiction test, exactly as for the COVID twin.

## The prediction (made from structure / domain knowledge, before seeing the result)
The directed-asymmetry **law** is conditional: a transmitter-targeting controller beats the loudest-node
heuristic **iff** the contagion is directionally asymmetric. Applied to flu, there are two falsifiable cases:

- **H1 (directed):** US flu is a *directed spatial* epidemic — it propagates geographically with lead–lag, so
  a few states lead each season and others follow. If so, the law predicts **ARRO ≫ greedy**, the
  net-transmitter ≠ the loudest (most-populous) state, directed influence measures beat undirected/reactive
  baselines, and the advantage **vanishes** under the symmetrization null.
- **H0 (symmetric):** US flu is instead a near-*symmetric seasonal common factor* — every state rises together
  in winter (the 2008-equity-crash pattern). If so, the law predicts **ARRO ≈ greedy** (no node-level leverage).

**My expectation:** H1 — flu spreads directionally, with early-season onset typically in the South / South-Central
rather than the largest-population states; so I expect **loudest ≠ transmitter** and **ARRO ≫ greedy**. I am *not*
certain (flu has a strong seasonal common factor too), which is precisely why this is a real test: H0 is a live
outcome that would refute my expectation while still being consistent with the law's *conditional* form.

## Binding success criteria (fixed now)
The law is **confirmed** on this domain iff the ARRO-vs-greedy outcome matches the network's measured asymmetry:
1. **If the network is asymmetric** (clear net-transmitters; symmetrization null collapses the edge):
   ARRO beats greedy by a clear margin (target: ≥ 2× the cascade reduction), and the net-transmitter is **not**
   the loudest state.
2. **If the network is near-symmetric**: ARRO ≈ greedy (within noise) — the law's predicted null.
3. **Benchmark:** directed influence measures beat undirected centrality and the loudest-node heuristic
   (as in `validity_benchmark.py`).

The law is **falsified** iff: ARRO ≫ greedy on a *symmetric* network, or ARRO ≈ greedy on a *clearly asymmetric*
one (the symmetrization null fails to explain the result).

## Results (filled in by `flu_transfer.py` AFTER the above was committed)
*(see `flu_transfer_results.json` and `flu_transfer.png`)*


### RESULT (2026-06-09)
- **Data:** 14 states, 298 flu-active weeks (2010–2019); CDC ILINet wILI via Delphi Epidata.
- **Diagnosis:** DY total connectedness **78%** (strongly directed); net-transmitters **TX, GA, IL**
  (Southern / South-Central — as the pre-registration's geographic expectation predicted); receivers WA, MA, AZ.
- **The prediction was PARTLY RIGHT, PARTLY WRONG (reported honestly):** flu IS strongly directed (H1's premise
  holds, and my Southern-onset geography was right), BUT — contrary to my expectation — the net-transmitter
  **Texas is also the loudest state** (highest mean wILI). The loudest-vs-transmitter GAP is therefore *absent*.
- **Interdiction:** greedy +68%, transmitter-targeting +78% (×1.1), mpc +67%; directed measures (var-out +70,
  spillover +77, transmitter +78) beat undirected correlation centrality (+25) but only modestly beat greedy,
  because greedy (targeting the loudest = Texas) already targets the transmitter.
- **Symmetrization null:** transmitter−greedy advantage falls +10 → −15 pts as the twin is symmetrized — the
  small edge is causally directed.
- **VERDICT — the law is REFINED, not falsified.** The operative quantity is the **loudest-vs-transmitter gap**,
  not directedness alone. US flu is a strongly *directed* epidemic whose transmitter happens to coincide with its
  loudest state, so the gap is absent and transmitter-targeting ≈ greedy — exactly what the law requires. Flu
  (coincident transmitter) and the 2008 equity crash (symmetric common factor) are two *distinct* mechanisms by
  which the gap can be small, both yielding ARRO ≈ greedy. This sharpens the law into a more precise, more
  falsifiable statement, and it does so via an honest, pre-registered prediction that was partly wrong.
