# PROPOSED driver: P1 robustness on the REAL smoke23 twin (fetches EPA PM2.5; transmitter != loudest => confound separation).
import sys, io, zipfile, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
import PROPOSED_p1_robustness as P1
STATES = ["New York","New Jersey","Pennsylvania","Connecticut","Massachusetts","Rhode Island","Vermont",
          "New Hampshire","Maine","Ohio","Michigan","Illinois","Wisconsin","Minnesota","Indiana","Maryland","Virginia"]
print("fetching EPA daily PM2.5 2023 (~9 MB)...")
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen("https://aqs.epa.gov/aqsweb/airdata/daily_88101_2023.zip", timeout=180).read()))
csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
raw = pd.read_csv(z.open(csv_name), usecols=["State Name","Date Local","Arithmetic Mean"])
raw = raw[raw["State Name"].isin(STATES)]; raw["Date Local"] = pd.to_datetime(raw["Date Local"])
daily = raw.groupby(["State Name","Date Local"])["Arithmetic Mean"].mean().reset_index()
P = daily.pivot(index="Date Local", columns="State Name", values="Arithmetic Mean").sort_index()
P = P.loc["2023-05-01":"2023-07-31"].interpolate().dropna(axis=1, how="any")
ABBR = {"New York":"NY","New Jersey":"NJ","Pennsylvania":"PA","Connecticut":"CT","Massachusetts":"MA","Rhode Island":"RI",
        "Vermont":"VT","New Hampshire":"NH","Maine":"ME","Ohio":"OH","Michigan":"MI","Illinois":"IL","Wisconsin":"WI",
        "Minnesota":"MN","Indiana":"IN","Maryland":"MD","Virginia":"VA"}
names = [ABBR.get(s, s[:3]) for s in P.columns]
P1.run_all(P.values, names, label="smoke23", out_dir=str(BASE), K=200)
