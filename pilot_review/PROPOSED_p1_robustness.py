"""
PROPOSED, SEPARATE P1 robustness module (does NOT modify any committed pilot script or the manuscript).
Implements the three referee-panel P1 re-runs as drop-in functions that reuse each twin's own stress
series and the committed LSA library (lsa_capstone), so the numbers are directly comparable to the
transfer twins (same simulator, same fit_var_nonneg/gfevd/connectedness):

  (1) INFORMATION-MATCHED baselines  -> isolate direction-of-targeting from model access.
      Adds receiver-targeting (net IN-strength), in-strength (var-in) and model-predicted-damage
      controllers that ALL read the fitted Phi, so the only difference from transmitter-targeting is
      the DIRECTION of targeting, not access to the model. Also a held-out variant: targeting rule
      estimated on window 1, cascade simulated on the disjoint window 2.
  (2) CONFOUND SEPARATION  -> directed asymmetry vs out-hub centrality.
      Compares DY-transmitter targeting against raw out-strength (var-out) targeting, and reports the
      case where the out-hub node differs from the DY transmitter (the clean separation).
  (3) DATA-LEVEL DIRECTEDNESS NULL  -> time-reversal + phase-randomized Fourier surrogates, with the
      non-negative VAR kernel RE-ESTIMATED inside every surrogate draw. If the transmitter advantage
      is a real directed effect it should survive on the real series and vanish on the surrogates.

USAGE (append ~2 lines to a twin script AFTER its stress series `S` (pandas) and `names` exist; e.g. in
smoke23_transfer.py after line 49, or equity_deep.py / covid after the stress matrix is built):

    import sys; sys.path.insert(0, str(Path(__file__).parent))      # if needed
    import PROPOSED_p1_robustness as P1
    P1.run_all(S.values, names, label="smoke23", out_dir=BASE)      # writes PROPOSED_p1_<label>.json

Run e.g.:  python smoke23_transfer.py   (after appending the two lines) ; or import and call run_all
on any (T x N) stress matrix.
"""
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "pilot_cross_tier"))
import lsa_capstone as L

# ------------------------------------------------------------------ simulator (faithful to the twins)
def project(a, x, B):
    a = np.clip(a, 0, x); s = a.sum(); return a * (B / s) if s > B else a
def stepf(Phi, c, x, a, rng):
    return np.clip(Phi @ np.clip(x - a, 0, None) + c + 0.05 * rng.standard_normal(len(x)), 0, None)
def alloc(score, x, B):
    w = np.clip(score, 0, None); return project(B * w / w.sum(), x, B) if w.sum() > 0 else np.zeros_like(x)
def greedy(x, B):
    a = np.zeros_like(x); rem = B
    for i in np.argsort(-x):
        g = min(x[i], rem); a[i] = g; rem -= g
        if rem <= 1e-9: break
    return a
def rescale(Phi, rho=1.06):
    ev = max(abs(np.linalg.eigvals(Phi))); return Phi * (rho / ev) if ev > 1e-6 else Phi
def s_varout(Phi):  # OUT-strength (column sums of |off-diagonal|) = sends-to
    A = np.abs(Phi).copy(); np.fill_diagonal(A, 0); return A.sum(0)
def s_varin(Phi):   # IN-strength (row sums) = receives-from
    A = np.abs(Phi).copy(); np.fill_diagonal(A, 0); return A.sum(1)

def interdict(Phi, c, S0, scores, B=2.0, T=16, H=4, seeds=16):
    """Cascade reduction (%) vs no-action for greedy + each named score controller."""
    out = {}
    for name in ["none", "greedy"] + list(scores.keys()):
        tt = []
        for s in range(seeds):
            rng = np.random.default_rng(20 + s); x = S0.copy(); acc = 0.0
            for t in range(T):
                if name == "none": a = np.zeros_like(x)
                elif name == "greedy": a = greedy(x, B)
                else: a = alloc(scores[name], x, B)
                x = stepf(Phi, c, x, a, rng); acc += x.sum()
            tt.append(acc)
        out[name] = float(np.mean(tt))
    b = out["none"]; return {k: 100 * (1 - v / b) for k, v in out.items()}

def _fit(M):
    Phi, c, Sig = L.fit_var_nonneg(M, ridge=5e-2)
    _, _, NET, tot = L.connectedness(L.gfevd(Phi, Sig))
    return Phi, c, Sig, NET, float(tot)

def _advantage(M, seeds=16):
    """transmitter - greedy cascade-reduction advantage on a single (re-fitted) series."""
    Phi, c, Sig, NET, tot = _fit(M)
    S0 = np.maximum(M.mean(0), 0.5)
    red = interdict(rescale(Phi), c, S0, {"transmitter": np.clip(NET, 0, None)}, seeds=seeds)
    return red["transmitter"] - red["greedy"], NET

# ------------------------------------------------------------------ (1) information-matched baselines
def information_matched(M, names, seeds=16):
    Phi, c, Sig, NET, tot = _fit(M)
    S0 = np.maximum(M.mean(0), 0.5)
    scores = {
        "transmitter":      np.clip(NET, 0, None),        # directed OUT (the law's controller)
        "receiver":         np.clip(-NET, 0, None),       # directed IN  (same model, opposite direction)
        "var-out":          s_varout(Phi),                # raw out-strength (model-matched, undirected-hub)
        "var-in":           s_varin(Phi),                 # raw in-strength
        "predicted-damage": np.clip(Phi @ S0, 0, None),   # model-predicted next-step stress (model, not direction)
    }
    red = interdict(rescale(Phi), c, S0, scores, seeds=seeds)
    adv = {k: round(red[k] - red["greedy"], 1) for k in scores}   # advantage over reactive greedy
    return {"reductions": {k: round(red[k], 1) for k in red}, "advantage_over_greedy": adv,
            "direction_isolated": round(red["transmitter"] - red["receiver"], 1),
            "beyond_out_hub": round(red["transmitter"] - red["var-out"], 1)}

def heldout(M, seeds=16):
    T = len(M); h = T // 2
    if h < 6: return None
    Phi1, c1, S1, NET1, _ = _fit(M[:h])          # targeting rule estimated on window 1 (out-of-sample)
    Phi2, c2, S2, NET2, _ = _fit(M[h:])          # cascade dynamics from disjoint window 2
    S0 = np.maximum(M[h:].mean(0), 0.5)
    red = interdict(rescale(Phi2), c2, S0,
                    {"transmitter_oos": np.clip(NET1, 0, None)}, seeds=seeds)
    return {"heldout_transmitter_minus_greedy": round(red["transmitter_oos"] - red["greedy"], 1),
            "same_transmitter_both_windows": bool(np.argmax(NET1) == np.argmax(NET2))}

# ------------------------------------------------------------------ (2) confound: directed vs out-hub
def confound(M, names, seeds=16):
    Phi, c, Sig, NET, tot = _fit(M)
    t_dy = int(np.argmax(NET)); t_out = int(np.argmax(s_varout(Phi)))
    S0 = np.maximum(M.mean(0), 0.5)
    red = interdict(rescale(Phi), c, S0,
                    {"transmitter": np.clip(NET, 0, None), "var-out": s_varout(Phi)}, seeds=seeds)
    return {"dy_transmitter": names[t_dy], "out_hub": names[t_out],
            "differ": bool(t_dy != t_out),
            "transmitter_reduction": round(red["transmitter"], 1),
            "out_hub_reduction": round(red["var-out"], 1),
            "directed_minus_outhub": round(red["transmitter"] - red["var-out"], 1)}

# ------------------------------------------------------------------ (3) data-level directedness null
def _phase_randomize(M, rng):
    out = np.empty_like(M)
    for j in range(M.shape[1]):
        x = M[:, j]; F = np.fft.rfft(x); ph = rng.uniform(0, 2 * np.pi, len(F)); ph[0] = 0.0
        if len(x) % 2 == 0: ph[-1] = 0.0
        out[:, j] = np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=len(x))
    return out

def datalevel_null(M, K=200, seeds=10, seed0=0):
    real_adv, NET = _advantage(M, seeds=seeds)
    rev_adv, NET_rev = _advantage(M[::-1].copy(), seeds=seeds)     # time-reversal flips lead-lag
    rng = np.random.default_rng(seed0); surr = []
    for k in range(K):
        try: surr.append(_advantage(_phase_randomize(M, rng), seeds=max(6, seeds // 2))[0])
        except Exception: pass
    surr = np.array(surr); p = float((1 + np.sum(surr >= real_adv)) / (1 + len(surr)))
    return {"real_advantage": round(real_adv, 1),
            "time_reversed_advantage": round(rev_adv, 1),
            "phase_surrogate_mean": round(float(np.mean(surr)), 1) if len(surr) else None,
            "phase_surrogate_95pct": round(float(np.quantile(surr, 0.95)), 1) if len(surr) else None,
            "p_value_vs_phase_surrogates": round(p, 3), "K_ok": int(len(surr))}

# ------------------------------------------------------------------ driver
def run_all(M, names, label, out_dir=".", K=200):
    M = np.asarray(M, float)
    res = {"label": label, "T": int(M.shape[0]), "N": int(M.shape[1]),
           "information_matched": information_matched(M, names),
           "heldout": heldout(M),
           "confound_directed_vs_outhub": confound(M, names),
           "datalevel_null": datalevel_null(M, K=K)}
    path = Path(out_dir) / f"PROPOSED_p1_{label}.json"
    json.dump(res, open(path, "w"), indent=2)
    im = res["information_matched"]; dn = res["datalevel_null"]
    print(f"\n[P1 robustness :: {label}]  T={res['T']} N={res['N']}")
    print(f"  (1) information-matched advantage over greedy: {im['advantage_over_greedy']}")
    print(f"      transmitter - receiver (direction isolated) = {im['direction_isolated']}  | "
          f"transmitter - out-hub = {im['beyond_out_hub']}")
    if res["heldout"]: print(f"  (1b) held-out transmitter - greedy = {res['heldout']['heldout_transmitter_minus_greedy']}")
    cf = res["confound_directed_vs_outhub"]
    print(f"  (2) DY-transmitter={cf['dy_transmitter']} vs out-hub={cf['out_hub']} (differ={cf['differ']}); "
          f"directed - out-hub = {cf['directed_minus_outhub']}")
    print(f"  (3) data-level null: real adv={dn['real_advantage']}, time-reversed={dn['time_reversed_advantage']}, "
          f"phase-surrogate mean={dn['phase_surrogate_mean']} (95%={dn['phase_surrogate_95pct']}), "
          f"p={dn['p_value_vs_phase_surrogates']} (K={dn['K_ok']})")
    print(f"  wrote {path}")
    return res

if __name__ == "__main__":
    # self-test on a SYNTHETIC directed network (no external data needed): node 0 drives the rest.
    rng = np.random.default_rng(0); T, N = 200, 6
    Phi = np.zeros((N, N));
    for i in range(1, N): Phi[i, 0] = 0.35        # node 0 -> everyone (directed transmitter)
    np.fill_diagonal(Phi, 0.4)
    X = np.zeros((T, N))
    for t in range(1, T): X[t] = Phi @ X[t - 1] + rng.normal(0, 1, N) + (3.0 if t % 50 == 0 else 0)
    X = np.abs(X)
    run_all(X, [f"n{i}" for i in range(N)], label="selftest_synthetic_directed", out_dir="/tmp/qa", K=60)
