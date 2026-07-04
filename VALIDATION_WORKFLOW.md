# Reproducible pre-submission validation workflow

Everything done from the moment *"look into methods/workflows/agents/harnesses ... to be confident before
submission"* was asked, in order, with the exact scripts and commands so it can be re-run. This is the sibling
of `~/Documents/Neuro_Atlas/VALIDATION_WORKFLOW.md` (the consilience/GigaScience run) applied to the
directed-asymmetry contagion manuscript. Environments referenced:

- **python:** `/usr/bin/python3` (numpy 2.0.2, scipy 1.13.1, pytest 8.4.2; `pip install --user hypothesis` added).
  The pinned reproduction env is `requirements.txt` (numpy==2.0.2 / pandas==2.3.3 / scipy==1.13.1 /
  matplotlib==3.9.4 / autograd==1.8.0).
- **deposit / package:** `~/Documents/directed-asymmetry-contagion/` (public at
  github.com/patrickiben/directed-asymmetry-contagion). Core lib: `pilot_cross_tier/lsa_capstone.py`.
- **manuscript:** `~/Documents/SUBMISSION_AppliedNetSci_LSAxJEPA/` (`manuscript.tex` + `supplementary_information.tex`).

---

## 0. The reusable pattern (the transferable part)

1. **Research** the current method landscape with a multi-agent workflow that *web-verifies* each tool is real
   and current, adversarially fact-checks its own findings, and tailors applicability to the specific manuscript.
2. **Tier** the results by return-on-effort (Tier 0 mechanical / Tier 1 science / Tier 2 code / Tier 3
   certificates / Tier 4 AI-panel / Tier 5 packaging).
3. **Implement** the Tier-0/1/2 items as small deterministic gates that emit a JSON/figure and a pass/fail.
4. **Verify** each against the *real* manuscript and code (run it; assert numbers; run offline from a cold clone).
5. **Integrate**: one-command gate + clean-clone CI; stage science additions as runnable PROPOSED (never edit
   the manuscript in place); record findings; deposit + push.

The single organizing idea, matching the Neuro_Atlas run's "make the one fragile number demonstrably stable":
here the fragile objects are the **fabricated-citation risk**, the **clean-clone break**, and the
**surrogate-only / single-null directedness claim**. Every gate is built to make each demonstrably handled.

---

## Phase A — Research the methods (multi-agent workflow)

**Tool:** the `Workflow` harness. Run ID **wf_3fcc2195-278**, 16 agents. Script auto-saved at
`~/.claude/projects/-Users-patrickiben-Optimized-CCR-Slides/<session>/workflows/scripts/manuscript-crossval-scan-wf_3fcc2195-278.js`.

**Structure (generalizable):**
- 7 parallel research agents, one per facet: (1) formal verification / proof assistants, (2) AI multi-agent
  referee harnesses, (3) computational reproducibility & artifact evaluation, (4) validated / rigorous numerics,
  (5) results-integrity & error-detection, (6) econometrics / time-series / network-science replication,
  (7) integrity policy & pre-submission checklists.
- Each research agent runs live `WebSearch`/`WebFetch` (2025-2026), returns a structured
  `{name, whatItIs, whatsNew, maturity, relevanceToAuthor, howToUse, sources[], caveats}`.
- A **per-facet hostile verifier** (pipeline stage, not a barrier) re-checks every named tool via the web and
  flags anything invented / mis-named / over-claimed. *This caught 5 fabricated or mis-attributed citations
  inside the research output itself* (phantom "Chen" on Diebold-Yilmaz 2211.04184; a nonexistent "spec_curve by
  Turrell" R package -> use `specr`; two mis-cited arXiv IDs). That result is the justification for making the
  citation gate deterministic, never an LLM.
- A **completeness critic**, then an **xhigh synthesis** agent that emits the tiered stack + failure-mode map.

**Output:** the verified 6-tier stack (mirrored in the reusable artifact and in this file's phases B-G).

**Re-run:** `Workflow({scriptPath: "<saved script path>"})`, or re-issue a 7-facet fan-out with the
manuscript-context string and the hostile-verifier + synthesis stages.

---

## Phase B — Tier 0: mechanical gates (deterministic, never an LLM)

Ground first: `pilot_cross_tier/lsa_capstone.py` exposes `gfevd / connectedness / fit_var_nonneg /
spectral_radius`; the symmetrization null is `symmetrize_rho_matched` in `pilot_review/directedness_null.py`;
the bibliography is 28 inline `\bibitem` entries (only 1-2 carry a DOI).

### B1. Deterministic citation-integrity gate
**Script:** `tools/check_citations.py` (stdlib only — urllib). Resolves each reference via CrossRef / OpenAlex /
arXiv and scores by **containment of the record's title within the full citation text** (robust to Nature-style
author-initial noise; a first attempt that scored an extracted title against the record over-flagged 22/28 and
was fixed). Flags fabrication, identifier-hijacking (DOI -> different paper), dead DOIs, retractions.
**Run:**
```bash
/usr/bin/python3 tools/check_citations.py \
  ~/Documents/SUBMISSION_AppliedNetSci_LSAxJEPA/manuscript.tex \
  --mailto p.iben@saeny.net --json /tmp/cite_report.json
```
**Result:** 22 OK / 6 REVIEW / 0 FLAG. The 6 are real papers CrossRef indexes poorly (NeurIPS, OpenReview, UAI,
a book) or subtitle truncations — correctly separated from "does not exist".

### B2. AI-artifact / hidden-text scrub
**Script:** `tools/scrub_ai_artifacts.py` — grep for assistant boilerplate / unfilled placeholders + `pdftotext`
hidden-text screen of the PDF.
```bash
/usr/bin/python3 tools/scrub_ai_artifacts.py \
  ~/Documents/SUBMISSION_AppliedNetSci_LSAxJEPA/manuscript.tex \
  ~/Documents/SUBMISSION_AppliedNetSci_LSAxJEPA/supplementary_information.tex \
  --pdf ~/Documents/SUBMISSION_AppliedNetSci_LSAxJEPA/manuscript.pdf
```
**Result:** clean (0 hits).

### B3. Absolute-path guard
**Script:** `tools/check_paths.sh` — fails on `/Users/`, `/home/<user>/`, `C:\`, etc. in executable code.
Normalized 24 stale `/tmp/lsa_venv` run-instruction docstrings to `python3` along the way.
```bash
bash tools/check_paths.sh          # -> "no machine-specific absolute paths in executable code"
```

---

## Phase C — Tier 2: code-correctness (property / metamorphic tests)

**Script:** `tests/test_invariants.py` (Hypothesis; degrades to a seeded pytest sweep if Hypothesis is absent).
Imports the core functions; loads `symmetrize_rho_matched` via `ast` **without executing** `directedness_null.py`
(that module runs a full analysis on import). Eight invariants over randomized `(N, seed)`:
GFEVD rows sum to 1; entries in [0,1]; GFEVD & connectedness are label-permutation equivariant; net sums to 0;
`fit_var_nonneg` off-diagonals >= 0; spectral radius scales linearly; the symmetrization null is exactly
symmetric and rho-matched.
```bash
/usr/bin/python3 -m pip install --user hypothesis
PYTHONPATH=pilot_cross_tier /usr/bin/python3 -m pytest tests/ -q     # 8 passed (150 cases each)
```
**Teeth check (demonstrates the suite is not vacuous):** an injected row-normalization bug in `gfevd` is caught
by `check_rows_sum_to_one` in 60/60 seeds and by `check_entries_unit_interval` in 45/60.
**Mutation adequacy (periodic, longer):** `setup.cfg` configures `mutmut` over `lsa_capstone.py` +
`directedness_null.py` — `pip install mutmut && mutmut run && mutmut results`.

---

## Phase D — clean-clone CI + one-command gate

### D1. CI: `.github/workflows/reproduce.yml`
A cold `ubuntu-latest` runner (no cache, non-Mac paths) reproduces a reviewer's fresh machine on every push:
install pinned `requirements.txt`, run `tools/check_paths.sh`, run `pytest tests/`, then a deterministic offline
re-derivation of the out-of-sample lead that asserts the headline numbers:
```bash
cd out_of_sample_probe && python reproduce_oos.py --offline   # asserts +7.98% equities, +5.00% EM-FX
```

### D2. One-command pre-submission gate: `tools/presubmit.sh`
Runs B1+B2+B3+C against a submission folder and prints a single verdict.
```bash
PYTHON=/usr/bin/python3 bash tools/presubmit.sh ~/Documents/SUBMISSION_AppliedNetSci_LSAxJEPA
# -> RESULT: ALL CLEAR
```
Docs: verification section added to `RUN.md`; `.gitignore` extended (`.hypothesis/`, `.pytest_cache/`,
`.mutmut-cache`, `_reports/`).

---

## Phase E — Tier 1: statistical robustness on the science (staged PROPOSED, offline)

Both run on the bundled 2008-equity network (`pilot_3p46_equity/equity_weekly_close_2007_2010.csv`) — the
headline case — deterministically and offline. **Not wired into the manuscript.**

### E1. Bootstrap CIs on the connectedness numbers
**Script:** `pilot_review/PROPOSED_connectedness_bootstrap.py` — stationary block bootstrap (Politis-Romano) ->
refit non-negative VAR -> recompute DY connectedness -> 95% CI on net connectedness per node + the TCI.
```bash
cd pilot_review && /usr/bin/python3 PROPOSED_connectedness_bootstrap.py --K 500
```
**Result:** TCI 80.8% [74.8, 85.9]; US net-transmitter +21.2%, CI **[+0.7, +103.7] excludes zero** (P=0.98,
barely); every other node's CI includes zero. (Complements the existing `bootstrap_ci.py`, which bootstraps the
transmitter *identity* and controller *advantage* but not the connectedness numbers themselves.)

### E2. A second, structurally independent null
**Script:** `pilot_review/PROPOSED_second_null_directedness.py` — two nulls that hold Sigma fixed, randomize only
the coupling, and rho-match: a **direction-flip** null (swap `Phi_ij <-> Phi_ji` per pair; preserves the
undirected structure exactly) and a **weight-permutation** null.
```bash
/usr/bin/python3 PROPOSED_second_null_directedness.py --B 1000
```
**Result (two separate questions, reported honestly):** the transmitter *identity* is structural — a
direction-scrambling null reproduces the US on top only 0.008-0.011 of the time vs 0.125 chance, corroborating
the origin-recovery claim; but the directedness *magnitude* (max NET / total flow) is *not* anomalous vs a
weight reshuffle (p≈0.41 / 0.54). A scope caveat, not a refutation. See `PROPOSED_ROBUSTNESS_README.md`.

---

## Phase F — Tier 4: upgraded adversarial referee panel (grounded)

**Script:** `tools/referee_panel.workflow.js` (a reusable `Workflow`). Run ID **wf_f28fe300-639**, 54 agents.
1. **Error-hunt** — 8 parallel lenses, each hunting ONE named error class with an explicit definition
   (surrogate-only / null-leak / OOS-nonsignificance / near-circularity / overclaim / numeric-consistency /
   citation-support / stats-validity). Each finding must copy a VERBATIM quote.
2. **Refute** — each finding faces 3 skeptics prompted to refute it; majority-refute kills it.
3. **Calibration** — the same lens is run on a numberless stub; any finding there means the panel fabricates.
```bash
# via the Workflow tool:
Workflow({scriptPath: "~/Documents/directed-asymmetry-contagion/tools/referee_panel.workflow.js",
          args: {ms: ".../manuscript.tex", si: ".../supplementary_information.tex"}})
```
**Deterministic grounding (post-run, not an LLM):** grep each surviving finding's exact quote against the source;
drop any that isn't verbatim-present.
```bash
grep -n "All five primary empirical twins are fit on time series operating in levels" \
  ~/Documents/SUBMISSION_AppliedNetSci_LSAxJEPA/supplementary_information.tex
```
**Result:** 15 candidates -> **2 survivors**; calibration clean (0 fabricated in the stub -> panel trustworthy);
the refute layer correctly killed a plausible-but-wrong "surrogate mislabeling" finding 3/3. Both survivors are
grep-grounded:
- **P1 (0/3 refuted, substance-confirmed):** the stationarity table (`tab:stationarity`) covers only the five
  transfer twins; the **2008-equity and COVID-19 networks are absent** — yet those two carry primary
  confirmatory weight, and COVID is elsewhere declared "structurally non-stationary" while its static
  full-sample GFEVD is used as a confirmatory anchor.
- **P2 (1/3 refuted):** main-text line 104 attaches the surrogate-only "+8%/+3%/+10%, nearly triples" numbers to
  "the empirical COVID-19 network" without a local "surrogate" qualifier.

---

## Phase G — consolidation

- **Reusable artifact:** a self-contained HTML checklist of the whole verified stack + failure-mode map + the
  concrete tools, for the entire submission queue.
- **Memory:** `manuscript-crossval-toolkit.md` (the toolkit) + the two grounded findings recorded against the
  active submission.

---

## Phase H — port the Neuro_Atlas statistical/formal trio (parity with the consilience run)

Three checks from `~/Documents/Neuro_Atlas/VALIDATION_WORKFLOW.md` retargeted to the DY-connectedness engine so
both manuscripts run the identical harness. All offline, deterministic.

### H1. Independent symbolic re-derivation (exact oracle) — `tools/verify_symbolic.py`
SymPy exact-rational re-derivation of the Pesaran-Shin GFEVD + connectedness on a fixed rational fixture,
asserted equal to `lsa_capstone` to < 1e-12. Proves the engine computes the *right* number (the property tests
only assert invariants). Wired into CI and `presubmit.sh`.
```bash
/usr/bin/python3 -m pip install --user sympy
/usr/bin/python3 tools/verify_symbolic.py     # theta match 1e-16; NET 1e-14 -> OK
```

### H2. Statistical robustness suite — `pilot_review/PROPOSED_robustness_suite.py`
Bootstrap SE of US NET; **Gelman-Carlin Type-S/Type-M** design analysis; **TOST** equivalence of non-US nodes;
**Benjamini-Yekutieli** FDR (dependence-robust) across per-node tests.
```bash
cd pilot_review && /usr/bin/python3 PROPOSED_robustness_suite.py --B 3000
```
**Result (honest, sobering):** US NET +21.2, SE ~30, one-sided p~0.012; design analysis shows a low-power /
high-exaggeration regime (Type-M ~3.4-7x) if the true effect is modest; 0/7 non-US nodes TOST-equivalent to zero
(underpowered); **0/8 survive BY-FDR**. Scope: these are *within-one-network* diagnostics (T~180, N=8), and do
**not** touch the cross-network origin-recovery anchor (5/7, P~1e-4). They reinforce leaning on the descriptive
identity result over any single network's magnitude.

### H3. Null calibration — `pilot_review/PROPOSED_null_calibration.py`
Generate true-null **symmetric** (zero-directedness) VAR DGPs, run the exact direction-flip test, check p-value
uniformity (KS) and FPR at 0.05.
```bash
/usr/bin/python3 PROPOSED_null_calibration.py --K 200 --Bnull 300
```
**Result:** FPR@0.05 = **0.03 <= 0.05** — the test does not manufacture significance on symmetric data
(conservative; p-values non-uniform in the safe direction).

---

## Phase I — remediate the two grounded referee findings (drafted, NOT applied in place)

The manuscript/SI are never edited in place; each fix is a separate `PROPOSED_*.md` in the submission folder with
exact anchors. Both findings changed zero headline numbers.

### I1. P1 — stationarity of the two confirmatory pillars
The SI stationarity table covered only the five transfer twins; the equity and COVID-19 networks (the two
significant anchors) were absent, and COVID was called "structurally non-stationary" while its static GFEVD is a
confirmatory anchor. Computed the SAME ADF/KPSS battery (statsmodels) on the exact series each VAR is fit on:
```bash
/usr/bin/python3 -m pip install --user statsmodels
cd pilot_review && /usr/bin/python3 PROPOSED_stationarity_equity_covid.py   # -> .json
```
**Result:** equity 8/8 and COVID 14/14 stationary in both tests -> both pillars are stationary in levels. The
apparent contradiction was a two-senses conflation: the COVID *series* are stationary (unit-root sense), while the
directed *topology* is non-stationary (the transmitter migrates 72% of windows). Draft:
`SUBMISSION_.../PROPOSED_stationarity_fix.md` (two table rows + SI prose "five->seven" + a main-text disambiguation
+ optional strengthen). Strengthens the paper.

### I2. P2 — dropped surrogate qualifier
Main-text line 104 reported the +8/+3/+10% dynamic-vs-static reductions without the "surrogate" qualifier used in
six other places (SI confirms they are surrogate-only). Draft: `SUBMISSION_.../PROPOSED_p2_surrogate_qualifier.md`
(insert one word "surrogate"; includes a COMBINED line-104 paragraph merging P1's disambiguation + P2, since both
edit that paragraph).

---

## File inventory (created in this arc)

| file | phase | purpose |
|---|---|---|
| `tools/check_citations.py` | B1 | deterministic citation gate (CrossRef/OpenAlex/arXiv) |
| `tools/scrub_ai_artifacts.py` | B2 | AI-boilerplate + hidden-text scrub |
| `tools/check_paths.sh` | B3 | machine-specific absolute-path guard |
| `tests/test_invariants.py` | C | Hypothesis property / metamorphic tests |
| `setup.cfg` | C | pytest + mutmut config |
| `.github/workflows/reproduce.yml` | D1 | clean-clone CI (path guard + tests + offline reproduction) |
| `tools/presubmit.sh` | D2 | one-command pre-submission gate |
| `pilot_review/PROPOSED_connectedness_bootstrap.py` (+.json/.pdf/.png) | E1 | bootstrap CIs on NET + TCI |
| `pilot_review/PROPOSED_second_null_directedness.py` (+.json) | E2 | direction-flip + permutation nulls |
| `pilot_review/PROPOSED_ROBUSTNESS_README.md` | E | documents E1+E2 |
| `tools/referee_panel.workflow.js` | F | upgraded adversarial referee panel |
| `tools/verify_symbolic.py` | H1 | SymPy exact oracle for gfevd/connectedness (in CI) |
| `pilot_review/PROPOSED_robustness_suite.py` (+.json) | H2 | bootstrap SE + Type-S/M + TOST + BY-FDR |
| `pilot_review/PROPOSED_null_calibration.py` (+.json/.pdf/.png) | H3 | KS + FPR calibration of the directedness null |
| `pilot_review/PROPOSED_stationarity_equity_covid.py` (+.json) | I1 | ADF/KPSS for the equity + COVID networks |
| `SUBMISSION_.../PROPOSED_stationarity_fix.md` | I1 | drafted P1 edits (exact anchors) |
| `SUBMISSION_.../PROPOSED_p2_surrogate_qualifier.md` | I2 | drafted P2 edit + combined line-104 paragraph |
| `VALIDATION_WORKFLOW.md` (this file) | G | the reproducible record |
| `RUN.md` (verification section), `.gitignore` | B-D | docs / hygiene |

Two Workflow runs: **wf_3fcc2195-278** (method research, 16 agents), **wf_f28fe300-639** (referee panel, 54 agents).
Deposit committed at **c985ea8** on `main` (sole-author, no trailer) — LOCAL, `main` ahead of `origin/main` by 1,
NOT pushed. The two `PROPOSED_*.md` fix drafts live in the submission folder and are not applied to the .tex.

---

## One-command re-run (everything computational)

```bash
D=~/Documents/directed-asymmetry-contagion
SUB=~/Documents/SUBMISSION_AppliedNetSci_LSAxJEPA
V=/usr/bin/python3
$V -m pip install --user hypothesis sympy statsmodels

cd "$D"
bash tools/presubmit.sh "$SUB"                      # B+C: path guard + property tests + symbolic oracle + citations + AI scrub -> ALL CLEAR
$V tools/verify_symbolic.py                          # H1: exact-rational GFEVD oracle (also inside presubmit/CI)
cd out_of_sample_probe && $V reproduce_oos.py --offline && cd ..   # D1: offline reproduction (+7.98% / +5.00%)
cd pilot_review
$V PROPOSED_connectedness_bootstrap.py --K 500      # E1: bootstrap CIs on connectedness
$V PROPOSED_second_null_directedness.py --B 1000    # E2: second independent null
$V PROPOSED_robustness_suite.py --B 3000            # H2: bootstrap SE + Type-S/M + TOST + BY-FDR
$V PROPOSED_null_calibration.py --K 200 --Bnull 300 # H3: KS + FPR null calibration
$V PROPOSED_stationarity_equity_covid.py            # I1: ADF/KPSS for the equity + COVID pillars
# F (referee panel): Workflow({scriptPath:"$D/tools/referee_panel.workflow.js", args:{ms,si}}), then grep-ground the survivors
```

---

## Generalizing to any applied-math manuscript (and what this run did NOT do)

Same six-tier pattern transfers; swap the domain-specific pieces. After Phase H this run has **parity** with the
sibling consilience/Neuro_Atlas run (`~/Documents/Neuro_Atlas/VALIDATION_WORKFLOW.md`) on statistical robustness
and formal verification, and goes further on the citation gate and the deterministically-grounded referee panel.

Done here:
- **Design analysis (Type-S / Type-M, Gelman-Carlin retrodesign)** — Phase H2. Quantifies the sign/magnitude
  risk on the barely-significant US transmitter.
- **Null calibration (KS + FPR)** — Phase H3. Confirms the directedness null does not manufacture significance.
- **TOST equivalence** against a SESOI — Phase H2.
- **Independent symbolic re-derivation (SymPy)** of `gfevd`/`connectedness` — Phase H1, in CI.
- **AI referee agents as a supplement, not an authority** — Phase F (calibrated + grounded).

Still portable from the Neuro_Atlas run (optional next):
- **Reporting checklist (MDAR / TOP), Binder one-click launch, archival DOI (Zenodo).**
- **Convergence-order / method-of-manufactured-solutions** — only if a numerical *solver* is added; N/A for the
  current closed-form GFEVD.

---

## Addendum (2026-07-04) — lessons ported from the false-floor run

The third sibling, `~/Documents/false-floor/VALIDATION_WORKFLOW.md` (cautionary methods paper, TMLR),
surfaced three harness lessons worth carrying here:

1. **Run CI on the Python version matrix you actually ship on; never assert cross-platform floating-point
   bit-equality.** There, an inline-consistency check using exact `==` passed on Python 3.9 but *failed* on
   3.12: CPython 3.12 made `sum()` use compensated (Neumaier) summation, so a hand-looped CDF sum and a
   `sum()` diverged by ~1 ULP. Fix = a tolerance (`< 1e-12`) plus a CI matrix (3.11 + 3.12) that exposes it.
   **Applies here:** `tests/test_invariants.py` and `tools/verify_symbolic.py` compare floats — keep the
   tolerances, and consider widening `.github/workflows/reproduce.yml` to a 3.11+3.12 matrix.
2. **Deterministic and adversarial citation checks are complementary — run both.** `check_citations.py` scores
   by title containment, so it *passes* a real paper that carries a wrong year/volume (there, `vetter2000` had
   a perfect title match but the wrong year, volume, and issue); an adversarial resolve→refute LLM Workflow
   caught that metadata error, while the deterministic gate is what catches existence / DOI-hijack / dead-DOI.
   Neither alone suffices. **Port into this repo's `check_citations.py`:** the false-floor run fixed three real
   bugs in it — (a) the `.bib` parser missed one-entry-per-line files (required the closing brace on its own
   line); now brace-matched; (b) a DOI suffix like `rspb.2001.1812` was mis-read as the year 1812; now the DOI
   is stripped before year harvest; (c) DOI-less conference venues (PMLR/ICML/MLSys/TMLR) were hard-FLAGged as
   "possible fabrication" instead of REVIEW; now widened to REVIEW.
3. **When the build injects citations from prose (pandoc/regex), assert a post-build invariant** (`\bibitem`
   count == entries cited). There, prose editing silently dropped three references (including the single most
   on-point prior work) from the bibliography while they still appeared in the text. This repo uses inline
   `\bibitem`, so the mirror failure mode is a listed-but-uncited (or cited-but-unlisted) entry — assert
   `cited == listed` either way.

### Actioned here (2026-07-04)
All three lessons were implemented and verified against the ANS manuscript (still 22 OK / 6 REVIEW / 0 FLAG,
parity 0/0):
- **Lesson 1:** `.github/workflows/reproduce.yml` now runs a **3.11 + 3.12** matrix. Our float checks already use
  tolerances (`verify_symbolic.py` `< 1e-12`; `test_invariants.py` `allclose`), so both versions pass — the matrix
  is the guard against a future exact-`==` creeping in.
- **Lesson 2:** ported the three `check_citations.py` bug fixes — (a) `.bib` parser now brace/entry-boundary
  matched (one-entry-per-line files parse); (b) the DOI/arXiv id is stripped before year harvest (a suffix like
  `rspb.2001.1812` no longer becomes the year 1812); (c) DOI-less conference venues (PMLR/ICML/NeurIPS/UAI/TMLR/…)
  are REVIEW, not a hard FLAG. Added a **deterministic year-drift check** in the query branch (a right-title /
  wrong-year citation is now downgraded to REVIEW), the deterministic complement to the adversarial resolve→refute
  pass. Verified by micro-tests (year→2001; venue→REVIEW; one-line `.bib`→2 entries).
- **Lesson 3:** `check_citations.py` now runs a `cited == listed` parity check on every `.tex` (extracts
  `\bibitem` vs the `\cite`/`\citep`/`\citet`/… family); a cited-but-unlisted key is a hard fail, listed-but-uncited
  is a note. The ANS manuscript is 0/0.
