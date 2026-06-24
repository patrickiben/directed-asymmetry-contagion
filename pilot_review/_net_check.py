import sys, json, io, zipfile, urllib.request, time
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pilot_cross_tier"))
import lsa_capstone as L
RIDGE = 5e-2

def load_asia97():
    SER = {"DEXTHUS": "Thai", "DEXKOUS": "Korea", "DEXMAUS": "Malay", "DEXSIUS": "Singa", "DEXJPUS": "Japan",
           "DEXTAUS": "Taiwan", "DEXHKUS": "HK", "DEXINUS": "India", "DEXCHUS": "China"}
    def fred(i, tries=3):
        u = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={i}&cosd=1996-01-01&coed=1998-12-31"
        for _ in range(tries):
            try:
                df = pd.read_csv(io.StringIO(urllib.request.urlopen(u, timeout=25).read().decode()))
                df.columns = ["date", i]; df["date"] = pd.to_datetime(df["date"])
                df[i] = pd.to_numeric(df[i], errors="coerce"); return df.set_index("date")[i].dropna()
            except Exception: time.sleep(2)
        return None
    cols = {}
    for i, nm in SER.items():
        s = fred(i)
        if s is not None and len(s) > 400: cols[nm] = s
    FX = pd.DataFrame(cols).resample("W").last().interpolate().dropna()
    base = FX.loc["1996-01-01":"1996-12-31"].mean(); S = (FX / base - 1.0) * 100.0
    S = S.loc["1996-06-01":"1998-12-31"]
    return S.values, np.maximum(S.values.mean(0), 0.5)
def load_smoke():
    STATES = ["New York", "New Jersey", "Pennsylvania", "Connecticut", "Massachusetts", "Rhode Island", "Vermont",
              "New Hampshire", "Maine", "Ohio", "Michigan", "Illinois", "Wisconsin", "Minnesota", "Indiana", "Maryland", "Virginia"]
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen("https://aqs.epa.gov/aqsweb/airdata/daily_88101_2023.zip", timeout=120).read()))
    csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
    raw = pd.read_csv(z.open(csv_name), usecols=["State Name", "Date Local", "Arithmetic Mean"])
    raw = raw[raw["State Name"].isin(STATES)]; raw["Date Local"] = pd.to_datetime(raw["Date Local"])
    daily = raw.groupby(["State Name", "Date Local"])["Arithmetic Mean"].mean().reset_index()
    P = daily.pivot(index="Date Local", columns="State Name", values="Arithmetic Mean").sort_index()
    P = P.loc["2023-05-01":"2023-07-31"].interpolate().dropna(axis=1, how="any")
    avail = [s for s in STATES if s in P.columns]; P = P[avail]
    return P.values, np.maximum(P.values.mean(0), 0.5)
def load_flu():
    STATES = ["ca", "tx", "fl", "ny", "pa", "il", "oh", "ga", "nc", "mi", "nj", "va", "wa", "az", "ma"]
    url = "https://api.delphi.cmu.edu/epidata/fluview/?regions=" + ",".join(STATES) + "&epiweeks=201040-201920"
    r = json.load(urllib.request.urlopen(url, timeout=60)); assert r["result"] == 1, r.get("message")
    df = pd.DataFrame(r["epidata"])[["region", "epiweek", "wili"]]
    M = df.pivot(index="epiweek", columns="region", values="wili").sort_index()
    wk = (M.index % 100); M = M[(wk >= 40) | (wk <= 20)]
    avail = [s for s in STATES if s in M.columns]; M = M[avail].interpolate().dropna()
    return M.values, np.maximum(M.values.mean(0), 0.5)

for name, loader in [("asia97", load_asia97), ("smoke", load_smoke), ("flu", load_flu)]:
    try:
        M, S0 = loader()
        Phi0, c0, Sig0 = L.fit_var_nonneg(M, ridge=RIDGE)
        # cache the panel so the bootstrap need not re-fetch
        np.save(f"/tmp/lsa_nn/panel_{name}.npy", M); np.save(f"/tmp/lsa_nn/S0_{name}.npy", S0)
        _, _, NET, _ = L.connectedness(L.gfevd(Phi0, Sig0))
        print(f"[{name:8s}] OK  T={M.shape[0]:3d} N={M.shape[1]:2d}  top_NET={int(np.argmax(NET))}  -> cached")
    except Exception as e:
        print(f"[{name:8s}] FAILED: {e}")
