# Anticipating Network Contagions: When Directedness Outperforms Connectedness

Analysis code, prediction-first records, and redistributable data for the paper
**"Anticipating Network Contagions: When Directedness Outperforms Connectedness"** (P. Iben, independent research).

When a shock spreads through an interfering network, is it better to reinforce the worst-hit unit (*react*) or to
support the quiet unit that *exports* the cascade (*anticipate*)? Across seven real contagion networks — epidemic,
financial, environmental, transport, and conflict — a transmitter-targeting controller beats reactive control
precisely when the network is **directionally asymmetric**. Controllability is set by the directedness of the spread,
not the magnitude of its connectedness. The pipeline is one uniform sequence: a non-negative VAR surrogate →
Diebold–Yılmaz directed connectedness (transmitter = max net) → interdiction, with a per-case symmetrization null that
isolates directedness.

## Prediction-first protocol
A central commitment of the work: for each system the predicted transmitter and the falsification criteria were
fixed **before** the network was computed. The committed records are the `*_PREREGISTRATION.md` files in
`pilot_review/` (asia97, smoke23, flu, flights, conflict). Three **further** predictions, on systems not in the
paper, are publicly pre-registered with an external timestamp at **https://osf.io/49kn7** (see
`preregistration_2026_new_tests.md`).

## Layout
- `pilot_cross_tier/lsa_capstone.py` — core library: non-negative VAR fit, generalized-FEVD / Diebold–Yılmaz
  connectedness, the interdiction simulation and controllers. (The controller labelled `learned-MPC` in the code is
  the paper's *cross-entropy MPC* — a CEM planner over a learned world-model; `oracle-MPC` is the true-dynamics
  upper bound; `greedy` is the reactive/loudest baseline.)
- `pilot_review/` — the per-network analyses and figure/robustness scripts, their cached results (`*_results.json`),
  and the prediction-first records (`*_PREREGISTRATION.md`):
  - networks: `asia97_transfer.py`, `smoke23_transfer.py`, `flu_transfer.py`, `flights_transfer.py`,
    `conflict_transfer.py`; held-out COVID benchmark/validity in `validity_benchmark.py`,
    `validity_decomposition.py`, `nonstationary_gate.py`; cross-network summary in `transfer_scorecard.py`.
  - robustness: `si_rho_sensitivity.py`, `si_horizon_stability.py`, `horizon_auc.py` (horizon-integrated
    transmitter), `bootstrap_ci.py` / `_bootstrap_check.py`, `groundtruth_gap.py`, `datalevel_null.py`,
    `directedness_null.py`, `nonnormality_predictor.py`, `review_topology.py`, `realdata_leadlag.py`
    (out-of-sample lead on real data).
  - `jhu_confirmed_US.csv` — public Johns Hopkins COVID-19 case data (the one large cached input).
- `pilot_3p46_equity/` — the 2008 equity-crash network: `equity_deep.py` (pipeline),
  `equity_weekly_close_2007_2010.csv` + `EQUITY_DATA_README.md` (redistributable processed index levels),
  `PROPOSED_verify_equity_redist.py` (one-command reproduction from the CSV → US transmitter +21%, DY total 81%).

## Data
All inputs are public. COVID is included (`pilot_review/jhu_confirmed_US.csv`); the equity network reproduces from the
included processed panel. The remaining loaders fetch live public data at run time (FRED exchange rates; EPA AirData
PM₂.₅; CDC ILINet via the Delphi Epidata API; UCDP GED). A few exploratory non-paper networks and restrictive vendor
dumps are intentionally **not** included; the corresponding loaders read locally-cached series and are documented in
the script headers. The interdiction effect sizes are properties of calibrated supercritical surrogates, not live
interventions, as stated throughout the paper.

## Reproduce
```bash
pip install -r requirements.txt
python3 pilot_3p46_equity/PROPOSED_verify_equity_redist.py   # equity network from the redistributable CSV
python3 pilot_review/horizon_auc.py                          # horizon-integrated transmitter robustness
python3 pilot_review/asia97_transfer.py                      # a prediction-first transfer test (fetches FRED)
```

Fully offline (no network): `cd out_of_sample_probe && python3 reproduce_oos.py --offline` reproduces the SI
out-of-sample real-data lead (mean +3.3%, three-network sign test p=0.50, housing unstable). See **`RUN.md`** for
which scripts require network access and a claim→script map.

## Verification & reproducibility
A deterministic pre-submission verification harness lives in `tools/` and `tests/` (nothing here uses a language
model to decide correctness or existence):

- One command: `bash tools/presubmit.sh <submission_folder>` — absolute-path guard, property/invariant tests, the
  symbolic exact oracle, citation-integrity gate, and AI-artifact scrub, with a single pass/fail verdict.
- CI (`.github/workflows/reproduce.yml`) runs the code gates + a deterministic offline reproduction on a cold
  runner every push. See the **Verification & integrity gates** table in `RUN.md` for each check individually.
- **`VALIDATION_WORKFLOW.md`** is the full, reproducible record: how the harness was built (multi-agent method
  research → tiered gates → property/metamorphic + symbolic verification → adversarial referee panel → statistical
  robustness), with the exact scripts, commands, and a single copy-paste re-run block.

## Citation & license
Please cite the paper (and this archive's DOI) when using the code. Code released under the MIT License; the included
data are public factual series attributed to their providers (see `pilot_3p46_equity/EQUITY_DATA_README.md`).
