"""
PROPOSED (candidate robustness addition -- NOT wired into the manuscript).

A SECOND, structurally independent null for the directedness claim, to complement the
per-case symmetrization null already in the paper. The reviewer concern the symmetrization
null invites is that it is the ONLY null and could "collapse by construction": it linearly
attenuates the antisymmetric part toward the symmetric mean. A convergent verdict from a
null built by a DIFFERENT mechanism is the strongest available defence against that charge.

Two independent nulls are drawn (Sigma held fixed; only the coupling Phi is randomized;
each rho-matched back to the observed spectral radius so criticality is held constant --
the same control the symmetrization null uses):

  (A) DIRECTION-FLIP null:  for each unordered pair (i,j), swap Phi_ij <-> Phi_ji with
      prob 1/2. Preserves each pair's TOTAL coupling and the entire symmetric (undirected)
      structure EXACTLY; randomizes only the direction of asymmetry. This isolates
      directedness while holding connectedness magnitude fixed -- by shuffling, not by
      shrinking toward the mean. Maximally different mechanism from the symmetrization null.

  (B) WEIGHT-PERMUTATION null (configuration-style): permute all off-diagonal weights among
      off-diagonal positions. Preserves the multiset of edge weights (overall coupling
      magnitude) but destroys both structure and direction.

Directedness statistics (from DY-GFEVD net connectedness): the top net-transmitter strength
max_i NET_i, and the total directed flow  sum_i |NET_i| / 2. For each null we report an
empirical p-value and z-score for the OBSERVED statistic against the ensemble, plus how
often the null even reproduces the observed transmitter's identity (a direction-destroying
null should place it at chance, ~1/N).

Self-contained and OFFLINE (bundled 2008-equity network). Deterministic (fixed seed).

Run:  python3 PROPOSED_second_null_directedness.py [--B 1000]
Writes: PROPOSED_second_null_directedness.json
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "pilot_cross_tier"))
import lsa_capstone as L

CSV = BASE.parent / "pilot_3p46_equity" / "equity_weekly_close_2007_2010.csv"
RIDGE = 2e-2
B = int(sys.argv[sys.argv.index("--B") + 1]) if "--B" in sys.argv else 1000
SEED = 20260703

def load_fit():
    P = pd.read_csv(CSV, parse_dates=["week_ending"]).set_index("week_ending")
    P = P.resample("W").last().loc["2007-01-01":"2010-06-30"].dropna(how="any")
    ret = 100 * np.log(P).diff().dropna()
    stress = (-ret).clip(lower=0)
    Phi, c, Sig = L.fit_var_nonneg(stress.values, ridge=RIDGE)
    return Phi, Sig, list(stress.columns)

def rho_match(Phi, target):
    r = L.spectral_radius(Phi)
    return Phi * (target / r) if r > 1e-9 else Phi

def directedness(Phi, Sig):
    """Returns (max NET, total directed flow, argmax NET index)."""
    _TO, _FROM, NET, _tot = L.connectedness(L.gfevd(Phi, Sig))
    return float(np.max(NET)), float(np.abs(NET).sum() / 2.0), int(np.argmax(NET))

def flip_null(Phi, rng):
    A = Phi.copy(); N = A.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < 0.5:
                A[i, j], A[j, i] = A[j, i], A[i, j]
    return A

def perm_null(Phi, rng):
    A = Phi.copy(); N = A.shape[0]
    off = [(i, j) for i in range(N) for j in range(N) if i != j]
    vals = np.array([A[i, j] for i, j in off])
    vals = vals[rng.permutation(len(vals))]
    for (i, j), v in zip(off, vals):
        A[i, j] = v
    return A

def ensemble(Phi, Sig, draw, rng, target_rho):
    mx = np.empty(B); tf = np.empty(B); ident = np.empty(B, dtype=int)
    for b in range(B):
        A = rho_match(draw(Phi, rng), target_rho)
        mx[b], tf[b], ident[b] = directedness(A, Sig)
    return mx, tf, ident

def summarize(name, obs, null, obs_ident, null_ident, N):
    p = (np.sum(null >= obs) + 1) / (len(null) + 1)          # one-sided empirical p
    z = (obs - null.mean()) / (null.std() + 1e-12)
    id_rate = float(np.mean(null_ident == obs_ident))
    print(f"  {name:26s} obs={obs:6.1f}  null mean={null.mean():6.1f}  "
          f"z={z:+5.2f}  p={p:.4f}  transmitter-recovered={id_rate:.3f} (chance {1/N:.2f})")
    return dict(statistic=name, observed=round(obs, 2), null_mean=round(float(null.mean()), 2),
                null_sd=round(float(null.std()), 2), z=round(float(z), 2), p_value=round(float(p), 4),
                null_transmitter_recovery=round(id_rate, 3), chance=round(1 / N, 3))

def main():
    Phi, Sig, names = load_fit()
    N = len(names)
    rho0 = L.spectral_radius(Phi)
    obs_mx, obs_tf, obs_id = directedness(Phi, Sig)
    print(f"Observed 2008-equity network: N={N}, rho={rho0:.3f}")
    print(f"  top net-transmitter = {names[obs_id]}  (max NET={obs_mx:.1f}, total directed flow={obs_tf:.1f})")
    print(f"Two independent nulls, B={B} draws each, Sigma fixed, rho-matched to {rho0:.3f}")
    print("=" * 88)

    rng = np.random.default_rng(SEED)
    out = {"network": "2008 equities", "N": N, "rho": round(rho0, 3),
           "observed": {"top_transmitter": names[obs_id], "max_net": round(obs_mx, 2),
                        "total_directed_flow": round(obs_tf, 2)}, "nulls": {}}

    for nm, draw in (("(A) direction-flip", flip_null), ("(B) weight-permutation", perm_null)):
        print(nm)
        mx, tf, ident = ensemble(Phi, Sig, draw, rng, rho0)
        r1 = summarize("max net-transmitter", obs_mx, mx, obs_id, ident, N)
        r2 = summarize("total directed flow", obs_tf, tf, obs_id, ident, N)
        out["nulls"][nm] = [r1, r2]
        print()

    # ---- two DISTINCT questions; report each honestly (do not conflate) ----
    flip_id = out["nulls"]["(A) direction-flip"][0]["null_transmitter_recovery"]
    perm_id = out["nulls"]["(B) weight-permutation"][0]["null_transmitter_recovery"]
    chance = 1 / N
    flip_p = out["nulls"]["(A) direction-flip"][0]["p_value"]
    perm_p = out["nulls"]["(B) weight-permutation"][0]["p_value"]
    identity_structural = flip_id < chance and perm_id < chance   # null rarely reproduces the true transmitter
    magnitude_anomalous = flip_p < 0.05 and perm_p < 0.05          # observed directedness magnitude in the tail

    print("=" * 88)
    print("Interpretation (two SEPARATE questions):")
    print(f"  Q1  Is the transmitter IDENTITY ({names[obs_id]}) structurally determined, not a magnitude")
    print(f"      artifact?  A direction-scrambling null reproduces it only "
          f"{flip_id:.3f}/{perm_id:.3f} of the time,")
    print(f"      {'well BELOW' if identity_structural else 'near/above'} the {chance:.3f} chance rate "
          f"-> identity is {'STRUCTURAL (supports the descriptive origin-recovery claim)' if identity_structural else 'not clearly structural'}.")
    print(f"  Q2  Is the directedness MAGNITUDE (max NET / total flow) anomalously high vs a network with")
    print(f"      the same coupling weights reshuffled?  p={flip_p:.3f} (flip) / {perm_p:.3f} (perm) ->")
    print(f"      {'YES' if magnitude_anomalous else 'NO -- the magnitude is NOT unusual'}; the paper's claim rests on transmitter")
    print(f"      IDENTITY and the transmitter-TARGETED control advantage (Q1 + the symmetrization null),")
    print(f"      NOT on aggregate directedness magnitude, so this is a scope caveat, not a refutation.")
    print(f"  NOTE: this fast null uses a GFEVD magnitude statistic. The apples-to-apples second null for")
    print(f"        the paper's headline is the SAME transmitter-vs-loudest interdiction advantage the")
    print(f"        symmetrization null attenuates -- feed `interdict_adv` (see bootstrap_ci.py) through")
    print(f"        these flip/permutation draws to test that metric directly. Left as a decision point.")
    out["identity_structural"] = bool(identity_structural)
    out["magnitude_anomalous"] = bool(magnitude_anomalous)
    json.dump(out, open(BASE / "PROPOSED_second_null_directedness.json", "w"), indent=2)
    print("[wrote PROPOSED_second_null_directedness.json]")

if __name__ == "__main__":
    main()
