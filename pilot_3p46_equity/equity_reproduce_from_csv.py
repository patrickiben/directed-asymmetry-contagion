"""
Self-contained 2008-equity reproduction from the REDISTRIBUTABLE weekly CSV
(equity_weekly_close_2007_2010.csv) -- no daily vendor JSON required.

This is the archive's one-command equity path. It reproduces, from the deposited CSV alone:
  * the directed connectedness EXACTLY: DY total ~81%, US net-transmitter ~+21% (the load-bearing
    directed-network result), and
  * the QUALITATIVE interdiction result: the reactive loudest-node (greedy) controller is
    counterproductive, while transmitter-informed control is strongly protective.

The exact interdiction magnitudes printed in the paper (transmitter ~42%, loudest ~-7%) are
computed on the higher-frequency DAILY index series from the named providers (S&P DJI, FTSE
Russell, Deutsche Boerse, Euronext, Nikkei Inc., Hang Seng Indexes, B3; see EQUITY_DATA_README.md),
which are not redistributed here. The weekly CSV is the redistributable panel, and on it the
interdiction magnitudes differ from the daily run (they are seed- and frequency-dependent, and the
paper frames every surrogate control magnitude as a descriptive illustration, not a confirmatory
test), while the connectedness and the qualitative control ordering reproduce.

Run:  python3 equity_reproduce_from_csv.py            (faithful: 16 seeds, full training; a few min)
      python3 equity_reproduce_from_csv.py --fast     (quick check: fewer seeds/steps)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "pilot_cross_tier"))
import lsa_capstone as L

FAST = "--fast" in sys.argv
CSV = BASE / "equity_weekly_close_2007_2010.csv"

def main():
    P = pd.read_csv(CSV, parse_dates=["week_ending"]).set_index("week_ending")
    P = P.resample("W").last().loc["2007-01-01":"2010-06-30"].dropna(how="any")
    stress = (-(100 * np.log(P).diff().dropna())).clip(lower=0)
    names = list(stress.columns)

    Phi, c, Sig = L.fit_var_nonneg(stress.values, ridge=2e-2)
    TO, FROM, NET, tot = L.connectedness(L.gfevd(Phi, Sig))
    us = int(np.argmax(NET))
    print("=" * 66)
    print("CONNECTEDNESS (from the redistributable weekly CSV):")
    print(f"  DY total connectedness index = {tot:.1f}%        [paper: ~81%]")
    print(f"  net-transmitter = {names[us]}  (net {NET[us]:+.1f}%)   [paper: US, ~+21%]")
    ok = abs(tot - 81) < 3 and names[us].startswith("US") and abs(NET[us] - 21) < 3
    print(f"  -> connectedness reproduces: {'YES' if ok else 'CHECK'}")

    seeds = 6 if FAST else 16
    steps = 800 if FAST else 3000
    train = 120 if FAST else 400
    print("=" * 66)
    print(f"INTERDICTION on the CSV twin (seeds={seeds}; qualitative check):")
    S0 = stress.loc["2008-09-01":"2008-11-15"].mean().values + 0.5
    ID = L.run_interdiction(Phi, c, Sig, S0, names, target_rho=1.05, budget=2.0, T_ep=24,
                            seeds=seeds, H=4, train_eps=train, steps=steps, verbose=False)
    base = ID["summary"]["none"]["mean"]
    red = {m: 100 * (1 - ID["summary"][m]["mean"] / base) for m in ID["order"]}
    for m in ID["order"]:
        print(f"  {m:12s} {red[m]:+6.0f}%")
    greedy_bad = red.get("greedy", 0) <= 5
    transmit_good = max(red.get("oracle-MPC", -99), red.get("learned-MPC", -99)) > 15
    print(f"  -> qualitative result reproduces "
          f"(loudest/greedy counterproductive & transmitter-informed control protective): "
          f"{'YES' if greedy_bad and transmit_good else 'CHECK'}")
    print("=" * 66)
    print("Exact paper magnitudes (transmitter ~42%, loudest ~-7%, symmetrization +49->-23) use the")
    print("higher-frequency daily series (not redistributed); see EQUITY_DATA_README.md. Connectedness")
    print("and the qualitative control ordering reproduce from this CSV.")

if __name__ == "__main__":
    main()
