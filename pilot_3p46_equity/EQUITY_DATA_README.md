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

Reproduce: `python3 ../pilot_cross_tier/...`  — load this CSV, weekly % decline = (-100*log(close).diff()).clip(0),
non-negative VAR (ridge 2e-2), Diebold-Yilmaz GFEVD connectedness -> US is the net-transmitter (+21%), total ~81%.
