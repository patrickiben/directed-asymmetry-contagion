# PROPOSED robustness additions (candidate — not wired into the manuscript)

Two self-contained, offline, deterministic scaffolds that address the two robustness gaps a
network/econometrics referee is most likely to probe. They run on the bundled 2008-equity
network (the paper's headline case; US net-transmitter, DY total ~81%). **Nothing here edits
the manuscript** — they exist so the underlying analysis can be run and the results weighed
before deciding what, if anything, to fold in.

## 1. `PROPOSED_connectedness_bootstrap.py` — uncertainty on the connectedness numbers
Stationary block bootstrap (Politis–Romano) → refit non-negative VAR → recompute DY
connectedness. Puts a 95% CI on net connectedness per node and on the total connectedness
index, answering *"is this net-transmitter distinguishable from zero?"*.

**Result (K=500):** TCI = 80.8% [74.8, 85.9] (tight; the ~81% headline is well-determined).
The US net-transmitter NET = +21.2%, 95% CI **[+0.7, +103.7]**, P(NET>0)=0.98 — the CI
**excludes zero, but only just**; every other node's CI includes zero. So the headline
transmitter claim survives with honest, reportable uncertainty, and the rest of the ranking
is within noise. This is a strictly-strengthening addition: it pre-empts the "distinguishable
from zero?" question with a real answer instead of a bare point estimate.

## 2. `PROPOSED_second_null_directedness.py` — a structurally independent second null
The paper has one null (the per-case symmetrization null, which *shrinks* the antisymmetric
part toward the mean). This adds two nulls built by a different mechanism — a **direction-flip**
null (swap `Phi_ij ↔ Phi_ji` per pair; preserves the undirected structure exactly, randomizes
only direction) and a **weight-permutation** null — each ρ-matched to hold criticality fixed.

**Result (B=1000) — two separate questions, reported honestly:**
- **Transmitter identity is structural.** A direction-scrambling null reproduces the US as
  top transmitter only 0.008–0.011 of the time, far *below* the 0.125 chance rate. Destroy the
  directed structure and the US almost never leads → the *identity* claim is not a magnitude
  artifact, independently corroborating the origin-recovery result (P≈10⁻⁴).
- **Directedness *magnitude* is not anomalous.** Max-NET / total directed flow are *not* in the
  tail versus a reshuffle of the same weights (p≈0.41 / 0.54). The paper's claim rests on
  transmitter *identity* and the transmitter-*targeted* control advantage — not on aggregate
  directedness magnitude — so this is a **scope caveat, not a refutation**. It is consistent
  with leading on the defensible descriptive result.

**Open decision point (yours):** the apples-to-apples second null for the *headline* metric is
the same transmitter-vs-loudest interdiction advantage the symmetrization null attenuates —
feed `interdict_adv` (see `bootstrap_ci.py`) through these flip/permutation draws to test that
metric directly. Left un-wired on purpose.

## 3. `PROPOSED_robustness_suite.py` — bootstrap SE + design analysis + TOST + BY-FDR
Ports the consilience/Neuro_Atlas `robustness_suite.py` to the DY-connectedness estimand. Four checks on the
US transmitter and the per-node family.

**Results (B=3000) — sobering and honest:**
- Bootstrap: US NET +21.2, SE ≈ 30, 95% CI [+0.7, +102], one-sided p ≈ 0.012.
- **Design analysis (Gelman–Carlin Type-S / Type-M):** if the true net-transmission is modest, the single-network
  estimate is badly underpowered and inflated — at an assumed true NET of 10, power ≈ 0.06, sign-error risk ≈ 17%,
  **exaggeration ratio ≈ 7×**; even at the observed 21.2, power ≈ 0.11 and exaggeration ≈ 3.4×. The point estimate
  is very likely an overestimate unless the true effect is large.
- **TOST:** 0/7 non-US nodes can be declared *equivalent to zero* within ±5pp — underpowered for equivalence
  (same verdict the Neuro_Atlas run reached for its headline).
- **Benjamini–Yekutieli FDR:** 0/8 per-node net-transmitter tests survive dependence-robust correction; US
  (p=0.012) does not clear the BY threshold.

**Scope (important):** these are *within-one-network* diagnostics on the equity case, where T≈180 and N=8 make the
per-edge estimates noisy. They do **not** touch the paper's primary confirmatory anchor — the **cross-network
origin recovery** (5/7, Poisson–binomial P≈10⁻⁴), a different and stronger test that aggregates the transmitter
*identity* signal across seven networks. Read together, they say: do not over-interpret any single network's
connectedness magnitude; lean on the cross-network identity result. Consistent with §2 and with the descriptive
framing.

## 4. `PROPOSED_null_calibration.py` — is the directedness test calibrated?
Ports `null_calibration.py`. Generates true-null **symmetric** (zero-directedness) VAR data-generating processes,
runs each through the exact pipeline (simulate → fit → direction-flip null test), and checks the p-values.

**Result (K=200):** false-positive rate at α=0.05 is **0.03 ≤ 0.05** — the test does *not* manufacture significance
from finite-sample noise. The p-value distribution is non-uniform, but in the **conservative** direction (the flip
null preserves asymmetry magnitude, so on symmetric data the observed statistic sits mid-ensemble). For a null
used to *support* a positive directedness claim, controlling false positives is the property that matters, and it
holds.

Run:
```bash
python3 PROPOSED_connectedness_bootstrap.py       # writes .json/.pdf/.png
python3 PROPOSED_second_null_directedness.py      # writes .json
python3 PROPOSED_robustness_suite.py              # writes .json  (bootstrap/Type-S,M/TOST/BY-FDR)
python3 PROPOSED_null_calibration.py              # writes .json/.pdf/.png  (KS + FPR)
```

The independent SymPy exact oracle for the engine itself is `../tools/verify_symbolic.py` (a code-correctness
gate, wired into CI and `presubmit.sh`).
