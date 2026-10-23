# 2008 equity-crash network — data provenance

`equity_weekly_close_2007_2010.csv` — Friday weekly closing levels of eight national stock indices,
2007-01-07 to 2010-06-27 (182 weeks). These are public, factual index closing levels.

| Column | Index | Provider |
|---|---|---|
| US | S&P 500 | S&P Dow Jones Indices |
| UK | FTSE 100 | FTSE Russell |
| Germany | DAX | Deutsche Börse |
| France | CAC 40 | Euronext |
| Japan | Nikkei 225 | Nikkei Inc. |
| Hong Kong | Hang Seng | Hang Seng Indexes |
| Netherlands | AEX | Euronext |
| Brazil | Bovespa (Ibovespa) | B3 |

This processed weekly panel is deposited so the network is fully reproducible without any vendor account.
The raw daily levels are equivalently available from the index providers above and from public aggregators
(e.g. Stooq: stooq.com/q/d/?s=^spx, ^ukx, ^dax, ^cac, ^nkx, ^hsi, ^aex, ^bvp).

## Reproduce
One command, from this CSV alone (no daily/vendor data):
```bash
python3 equity_reproduce_from_csv.py --headline   # connectedness + published pair, seconds
python3 equity_reproduce_from_csv.py              # adds the ARRO/planner comparison, a few minutes
```
**The connectedness and published interdiction-pair headlines reproduce from this deposited CSV.** Verified
2026-07-25 in a clean `python 3.12` environment built from the repository's pinned `requirements.txt`:

| quantity | reproduced | printed |
|---|---|---|
| DY total connectedness | 80.8% | ~81% |
| US net-transmitter | +21.2% | ~+21% |
| transmitter-targeting | +42.0% | ~+42% |
| loudest-node (reactive) | −6.8% | ~−7% |
| advantage at α=1 | +48.7 | ~+49 |

Recipe: weekly % decline `= (-100*log(close).diff()).clip(0)`, non-negative VAR at **ridge 2e-2** (the operating
penalty for this twin — it is the best-conditioned panel in the set; every other network uses 5e-2, see SI),
Diebold–Yılmaz GFEVD for the connectedness, and `interdict_adv` for the interdiction pair.

**Which routine produces the interdiction numbers.** `interdict_adv` — canonical definition in
`pilot_review/bootstrap_ci.py` (verbatim in `nonnormality_predictor.py` and `_bootstrap_check.py`), inlined into
`equity_reproduce_from_csv.py` so that script stays self-contained. It uses the archive-wide default budget
**B = 2.0**, shared with the five transfer scripts, `bootstrap_ci`, `directedness_null`, `groundtruth_gap`,
`nonnormality_predictor`, `si_rho_sensitivity` and `validity_decomposition`. Note that `equity_deep.py`'s
`run_interdiction` is a *different* controller family (none/random/greedy/learned-MPC/oracle-MPC) and does not
produce the published pair.

**On the two budgets.** `PROPOSED_crossparadigm_baseline.py` is the one script that instead uses a
budget-proportional rule (`B = 0.2*sum(S0)` ≈ 9.02 here), so that its cross-paradigm comparison is like-for-like
within its own table; at that budget the loudest-node figure is ≈ −22% rather than ≈ −7%. Transmitter-targeting is
near budget-invariant (+42.0% at B=2.0 vs +42.3% at B=9.02); the reactive controller is not (−6.8% vs −24.2%).
Both signs hold at both budgets, so the qualitative result is budget-independent while the magnitudes are not.
The SI states this and flags that only the ordering within the cross-paradigm table is to be compared.

*(Earlier revisions of this file stated that the exact interdiction magnitudes required the non-redistributed
daily series. That was incorrect — the producing routine is `interdict_adv`, not `equity_deep.py`, and it runs on
the deposited CSV. Corrected 2026-07-25.)*

The `--headline` mode does not regenerate the full symmetrization curve or every supplementary analysis.
