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
python3 equity_reproduce_from_csv.py        # add --fast for a quick check
```
This reproduces, from the deposited weekly CSV:
- **the directed connectedness exactly** — weekly % decline `= (-100*log(close).diff()).clip(0)`, non-negative VAR
  (ridge 2e-2), Diebold–Yılmaz GFEVD → **US is the net-transmitter (+21%), DY total ~81%** (verified: 80.8% / +21.2%);
- **the qualitative interdiction result** — the reactive loudest-node (greedy) controller is counterproductive
  while transmitter-informed control is strongly protective.

**Note on the exact interdiction magnitudes.** The specific numbers in the paper (transmitter ~42%, loudest ~−7%,
symmetrization advantage +49→−23) are computed on the higher-frequency **daily** index series from the providers
above, which are not redistributed here. They are seed- and sampling-frequency-dependent, and the paper frames
every surrogate control magnitude as a descriptive illustration rather than a confirmatory test; on the deposited
weekly CSV the connectedness and the qualitative control ordering reproduce, while the exact magnitudes differ.
The load-bearing directed-network result (US net-transmitter, DY ~81%) reproduces exactly from this CSV.
