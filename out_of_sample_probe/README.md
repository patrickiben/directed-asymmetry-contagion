# Reproducibility bundle — out-of-sample real-data lead

Self-contained regeneration of the SI's "An out-of-sample lead on real data" result. CPU-only, <30 s.

## Run
```bash
bash run.sh                 # installs pinned deps, runs offline financial arm + live influenza
# or, directly:
python3 -m pip install -r requirements.txt
python3 reproduce_oos.py --offline   # financial/housing/FX only, fully offline
python3 reproduce_oos.py             # + live CDC influenza
```
Output should match `EXPECTED_OUTPUT.txt` exactly on the pinned environment.

## What it shows (the corrected result)
| network | reproduced lead | transmitter (train) | stable? | SI |
|---|---|---|---|---|
| 2008 equities | **+7.98%** | US / S&P 500 | yes | +8% ✓ |
| EM-FX | **+5.00%** | Turkey | yes | +5% ✓ |
| US housing | **−32.4%** | Los Angeles | **no** (migrates) | +10% ✗ |

Equities and EM-FX reproduce the SI numbers. **Housing's transmitter is non-stationary** (its
train-fold transmitter differs from the full-sample one — it migrates between the bubble and the
recovery), so by the paper's own stability criterion housing belongs in the **non-stationary arm
where the lead is absent**, not the stationary arm. The earlier SI "+10% housing" was a
miscategorisation; the corrected stationary arm is {equities +8%, EM-FX +5%, smoke −3%}: mean
**+3.3%**, 2/3 positive, one-sided sign test **p=0.50** — suggestive, not significant (exactly the
paper's framing). This does not affect the primary evidence (surrogate interdiction; ground-truth
recovery P≈1.5×10⁻³) — housing/EM-FX are probe-only networks, not part of the main seven.

## Method (verbatim, embedded — no external module)
Non-negative ridge VAR(1) (`fit_var_nonneg`, ridge=5e-2) → Diebold–Yılmaz generalized FEVD
(`gfevd`) → net connectedness (`connectedness`). Per network: 75/25 chronological split; transmitter
= argmax net on train; loudest = argmax cumulative stress on train; z-score on train; forecast the
aggregate downstream stress (mean of remaining units) one step ahead, own-lag vs +transmitter-lag vs
+loudest-lag; lead = 100·[(AR−MSE_tx)/AR − (AR−MSE_loud)/AR]. "stable" = train transmitter ==
full-sample transmitter. Stress: housing/equities = price-decline (−Δlog, clipped ≥0); EM-FX =
depreciation (Δlog USD/local, clipped ≥0); influenza = ILINet wILI.

## Data provenance (`data/`)
- `data/housing/*.csv` — FRED S&P/Case-Shiller metro indices (LVXRSA Las Vegas, PHXRSA Phoenix,
  MIXRSA Miami, TPXRSA Tampa, LXXRSA Los Angeles, SDXRSA San Diego, SFXRSA San Francisco), monthly,
  `https://fred.stlouisfed.org/series/<ID>`. Public, no account.
- `data/equity/equity_weekly_close_2007_2010.csv` — weekly closing levels of S&P 500, FTSE 100, DAX,
  CAC 40, Nikkei 225, Hang Seng, AEX, Bovespa (2008 global equity crash window), from the index
  providers.
- `data/emfx/*.json` — 8 EM currencies vs USD (TRY, ARS, ZAR, BRL, MXN, RUB, INR, IDR), daily, Yahoo
  Finance, 2017–2019.
- Influenza is fetched live from the CDC ILINet Delphi Epidata API; COVID-19 (non-stationary arm in
  the paper) uses Johns Hopkins case data (not bundled; see the main code archive).

Determinism: fixed seed; CPU-only; no stochastic step in the OOS computation. The block-bootstrap
uncertainty quantification lives in the main code archive (`REALDATA_significance_run.R`).
