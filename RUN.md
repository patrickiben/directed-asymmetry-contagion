# Running the analyses

**Setup:** `python3 -m pip install -r requirements.txt` (numpy, pandas, scipy, matplotlib, autograd).
CPU-only, fixed seeds, no GPU. Each network/robustness script writes its figure (`.pdf` and `.png`) on run.
The core pipeline — non-negative VAR → Diebold–Yılmaz GFEVD → interdiction, with the per-case symmetrization
null and the five controllers — is `pilot_cross_tier/lsa_capstone.py`.

## Offline (no network) — start here
| What it reproduces | Command |
|---|---|
| SI out-of-sample real-data lead (mean +3.3%, sign test p=0.50, housing unstable) | `cd out_of_sample_probe && python3 reproduce_oos.py --offline` |
| 2008 equity-crash network (US transmitter +21%, DY total 81%) from the bundled CSV | `python3 pilot_3p46_equity/PROPOSED_verify_equity_redist.py` |
| Cross-network scorecard, Fig. 4 (2 confirm / 1 refine / 2 falsify) from cached `*_results.json` | `python3 pilot_review/transfer_scorecard.py` |
| Held-out COVID-19 benchmark/validity (uses bundled `jhu_confirmed_US.csv`) | `python3 pilot_review/validity_benchmark.py` |

## Require network (live public data, fetched on first run)
| Script | Network | Source |
|---|---|---|
| `pilot_review/asia97_transfer.py` | 1997 Asian FX | FRED |
| `pilot_review/smoke23_transfer.py` | 2023 wildfire smoke | EPA AirData PM₂.₅ |
| `pilot_review/flu_transfer.py` | U.S. influenza | CDC Delphi Epidata (ILINet) |
| `pilot_review/flights_transfer.py` | 2013–14 flight delays | BTS On-Time Performance |
| `pilot_review/conflict_transfer.py` | Sahel conflict | UCDP GED v24.1 |

## SI robustness scripts (run the network twins first)
`si_rho_sensitivity.py` and `si_horizon_stability.py` reconstruct all five twins, so they need the cached panels the transfer scripts write to `/tmp`. Run in this order:
```bash
python3 pilot_review/flights_transfer.py      # writes /tmp/lsa_flights/flights_panel.csv
python3 pilot_review/conflict_transfer.py     # writes /tmp/lsa_conflict/conflict_panel.csv
python3 pilot_review/si_rho_sensitivity.py    # rho-sweep SI figure (all 5 twins)
python3 pilot_review/si_horizon_stability.py  # GFEVD forecast-horizon SI figure + table (all 5 twins)
```
asia97, smoke23, and flu fetch live (FRED / EPA / CDC Delphi) if their caches are absent.

## Prediction-first records
The dated pre-commitments are `pilot_review/*_PREREGISTRATION.md` (asia97, smoke23, flu, flights, conflict).
Three further out-of-sample predictions, on systems not in the paper, are pre-registered with an external
timestamp at `osf.io/49kn7` (see `preregistration_2026_new_tests.md`).
