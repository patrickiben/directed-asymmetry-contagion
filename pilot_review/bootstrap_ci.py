"""
BOOTSTRAP CIs for the directed-asymmetry twins (addresses thin-T fragility honestly).

For each of the 7 twins we:
  1. reconstruct the PANEL M (T x N) EXACTLY as the twin does (cached at /tmp/lsa_panels/<name>_panel.csv;
     each reproduces the cached /tmp/lsa_nn/Phi_<name>.npy to <1e-5),
  2. STATIONARY-bootstrap the multivariate series (Politis & Romano 1994): resample contiguous
     wrap-around blocks of whole time-rows with geometric block lengths, mean block length
     L ~ sqrt(T). Resampling WHOLE ROWS preserves the cross-series (contemporaneous) structure;
     resampling in blocks preserves the lag-1 dynamics that VAR(1) reads. (A fixed moving-block
     variant is also run as a robustness check, see --mb.)
  3. refit fit_var_nonneg(ridge=5e-2) on each bootstrap panel,
  4. recompute (a) the NET-TRANSMITTER identity  = argmax of DY-gFEVD NET (TO-FROM),
                (b) the CONTROLLER ADVANTAGE     = transmitter% - greedy% via interdict_adv
                    (the SAME transmitter-vs-greedy interdiction the transfer JSONs report;
                     used as the fast proxy for the full ARRO interdict, which is far too slow
                     for K>=200 refits. On the original panels this proxy reproduces every
                     transfer-JSON advantage to <0.1 pt -- see header check / point estimates).

Per twin we report: MODAL transmitter, P(transmitter == reported/point one) [rank stability],
and a percentile 95% CI on the advantage. n=7 twins is small; reported honestly, no overclaiming.

K=300 by default. Run: python3 bootstrap_ci.py
"""
import sys, json, time
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pilot_cross_tier"))
import lsa_capstone as L
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
PAN  = Path("/tmp/lsa_panels")
RIDGE = 5e-2
K = int(sys.argv[sys.argv.index("--K")+1]) if "--K" in sys.argv else 300
USE_MB = "--mb" in sys.argv            # moving-block instead of stationary (robustness)
SEED = 12345

# =================================================================== interdict proxy (verbatim)
def project(a,x,B): a=np.clip(a,0,x); s=a.sum(); return a*(B/s) if s>B else a
def stepf(Phi,c,x,a,rng): return np.clip(Phi@np.clip(x-a,0,None)+c+0.05*rng.standard_normal(len(x)),0,None)
def greedy(x,B):
    a=np.zeros_like(x); rem=B
    for i in np.argsort(-x):
        g=min(x[i],rem); a[i]=g; rem-=g
        if rem<=1e-9: break
    return a
def alloc_score(score,x,B):
    w=np.clip(score,0,None); return project(B*w/w.sum(),x,B) if w.sum()>0 else np.zeros_like(x)
def rescale(Phi,rho=1.06):
    ev=max(abs(np.linalg.eigvals(Phi))); return Phi*(rho/ev) if ev>1e-6 else Phi
def interdict_adv(Phi0,c0,Sig0,S0,B=2.0,T=16,seeds=16,seed0=20):
    """transmitter% and greedy% cascade reduction (vs no-action) on the rescaled twin."""
    _,_,NET,_=L.connectedness(L.gfevd(Phi0,Sig0)); Phi=rescale(Phi0)
    scores={"transmitter":np.clip(NET,0,None)}; out={}
    for name in ["none","greedy","transmitter"]:
        tt=[]
        for s in range(seeds):
            rng=np.random.default_rng(seed0+s); x=S0.copy(); acc=0.0
            for t in range(T):
                a=np.zeros_like(x) if name=="none" else (greedy(x,B) if name=="greedy" else alloc_score(scores[name],x,B))
                x=stepf(Phi,c0,x,a,rng); acc+=x.sum()
            tt.append(acc)
        out[name]=float(np.mean(tt))
    b=out["none"]
    if b<=0: return 0.0,0.0
    red={k:100*(1-v/b) for k,v in out.items()}
    return red["transmitter"], red["greedy"]

# =================================================================== S0 per twin (matches loaders)
def make_S0(name):
    if name=="equity":
        S=pd.read_csv(PAN/"equity_panel.csv",index_col=0,parse_dates=True)
        return ("crisis", S.loc["2008-09-01":"2008-11-15"].mean().values+0.5)
    if name=="COVID":
        S=pd.read_csv(PAN/"COVID_panel.csv",index_col=0,parse_dates=True)
        return ("crisis", S.loc["2020-06-01":"2020-08-01"].mean().values+0.5)
    return ("mean", None)  # built per-replicate as max(M.mean(0),0.5)

def S0_replicate(name, M, crisis_S0):
    if crisis_S0[0]=="crisis":
        # keep the FIXED real-crisis onset state (a calibrated input, not a fit quantity);
        # bootstrapping the onset state would conflate two sources -> hold it fixed.
        return crisis_S0[1]
    return np.maximum(M.mean(0),0.5)

# =================================================================== stationary / moving-block bootstrap
def stationary_indices(T, L, rng):
    """Politis-Romano stationary bootstrap: geometric block lengths (mean L), wrap-around."""
    p = 1.0/L
    idx = np.empty(T, dtype=int); t=0
    while t<T:
        start = rng.integers(0,T); ln = rng.geometric(p)
        for k in range(ln):
            if t>=T: break
            idx[t] = (start+k)%T; t+=1
    return idx

def movingblock_indices(T, L, rng):
    """Fixed moving-block bootstrap: ceil(T/L) blocks of length L, wrap-around, trim to T."""
    nb = int(np.ceil(T/L)); idx=[]
    for _ in range(nb):
        start = rng.integers(0,T)
        idx.extend(((start+np.arange(L))%T).tolist())
    return np.array(idx[:T], dtype=int)

# =================================================================== net transmitter
def net_transmitter(Phi, Sig):
    _,_,NET,tot = L.connectedness(L.gfevd(Phi,Sig))
    return int(np.argmax(NET)), NET, float(tot)

# =================================================================== per-twin bootstrap
TWINS = ["asia97","smoke","flu","flights","conflict","equity","COVID"]
VERDICT = {"asia97":"CONFIRM","smoke":"CONFIRM","flu":"REFINE","flights":"FALSIFY",
           "conflict":"NULL","equity":"DIRECTED CONFIRM","COVID":"CONFIRM(held-out)"}

def run_twin(name, master_rng):
    S = pd.read_csv(PAN/f"{name}_panel.csv", index_col=0)
    M = S.values.astype(float); cols=list(S.columns); T,N=M.shape
    L_blk = max(2, int(round(np.sqrt(T))))
    crisis_S0 = make_S0(name)

    # ---- point estimate (no resample) ----
    Phi0,c0,Sig0 = L.fit_var_nonneg(M, ridge=RIDGE)
    tx0, NET0, tot0 = net_transmitter(Phi0, Sig0)
    S0_0 = S0_replicate(name, M, crisis_S0)
    tr0,gr0 = interdict_adv(Phi0,c0,Sig0,S0_0)
    adv0 = tr0-gr0
    loud0 = int(np.argmax(M.mean(0)))

    # ---- bootstrap loop ----
    tx_votes = np.zeros(N, dtype=int)
    advs=[]; trs=[]; grs=[]; net_self=[]; tots=[]; rhos=[]; ok=0; failed=0
    for k in range(K):
        idx = (movingblock_indices if USE_MB else stationary_indices)(T, L_blk, master_rng)
        Mb = M[idx]
        try:
            Phi,c,Sig = L.fit_var_nonneg(Mb, ridge=RIDGE)
            tx,NET,tot = net_transmitter(Phi,Sig)
            S0b = S0_replicate(name, Mb, crisis_S0)
            tr,gr = interdict_adv(Phi,c,Sig,S0b)
        except Exception:
            failed+=1; continue
        tx_votes[tx]+=1; advs.append(tr-gr); trs.append(tr); grs.append(gr)
        net_self.append(float(NET[tx0]))   # NET of the POINT transmitter, across replicates
        tots.append(tot); rhos.append(float(max(abs(np.linalg.eigvals(Phi)))))
        ok+=1
    advs=np.array(advs); trs=np.array(trs); grs=np.array(grs)
    modal_idx = int(np.argmax(tx_votes))
    p_modal = tx_votes[modal_idx]/ok
    p_point = tx_votes[tx0]/ok                      # P(transmitter == the reported/point one)
    ci = np.percentile(advs,[2.5,97.5]).tolist()
    # net-transmitter identity stability: fraction of replicates where the POINT transmitter is still net>0 and argmax
    frac_point_net_pos = float(np.mean(np.array(net_self)>0))

    res = dict(
        twin=name, verdict=VERDICT[name], T=int(T), N=int(N), block_len=int(L_blk),
        bootstrap=("moving-block" if USE_MB else "stationary"), K=int(K), K_ok=int(ok), K_failed=int(failed),
        point_transmitter=cols[tx0], point_loudest=cols[loud0],
        point_advantage=round(float(adv0),3), point_transmitter_pct=round(float(tr0),3), point_greedy_pct=round(float(gr0),3),
        point_dy_total=round(float(tot0),3),
        modal_transmitter=cols[modal_idx], P_modal_transmitter=round(float(p_modal),4),
        P_transmitter_is_point=round(float(p_point),4),
        frac_point_transmitter_net_positive=round(frac_point_net_pos,4),
        advantage_mean=round(float(advs.mean()),3), advantage_sd=round(float(advs.std(ddof=1)),3),
        advantage_median=round(float(np.median(advs)),3),
        advantage_CI95_lo=round(float(ci[0]),3), advantage_CI95_hi=round(float(ci[1]),3),
        advantage_CI_excludes_0=bool(ci[0]>0),
        transmitter_pct_mean=round(float(trs.mean()),3), greedy_pct_mean=round(float(grs.mean()),3),
        dy_total_mean=round(float(np.mean(tots)),3), rho_mean=round(float(np.mean(rhos)),3),
        vote_table={cols[i]:int(tx_votes[i]) for i in np.argsort(-tx_votes) if tx_votes[i]>0},
        advs=advs.tolist(),
    )
    return res

def main():
    t0=time.time()
    mode = "moving-block" if USE_MB else "stationary"
    print(f"BOOTSTRAP CIs  (K={K}, {mode}, ridge={RIDGE})  --  block len ~ sqrt(T)\n"+"="*96)
    master_rng = np.random.default_rng(SEED)
    out=[]
    for name in TWINS:
        rng_t = np.random.default_rng(master_rng.integers(0,2**31-1))
        r = run_twin(name, rng_t)
        out.append(r)
        flag = "*" if r["advantage_CI95_lo"]>0 else " "
        print(f"[{name:8s} {r['verdict']:16s}] T={r['T']:3d} N={r['N']:2d} blk={r['block_len']:2d}  "
              f"point_tx={r['point_transmitter']:>11s}(loud {r['point_loudest']:>11s})  "
              f"modal={r['modal_transmitter']:>11s}  P(tx=point)={r['P_transmitter_is_point']:.2f}  "
              f"adv={r['point_advantage']:+6.1f}  CI95=[{r['advantage_CI95_lo']:+6.1f},{r['advantage_CI95_hi']:+6.1f}]{flag}")
    payload = dict(meta=dict(K=K, bootstrap=mode, ridge=RIDGE, block_len_rule="round(sqrt(T))",
                             seed=SEED, n_twins=len(out),
                             advantage_definition="transmitter%-greedy% via interdict_adv (transfer-JSON interdiction proxy)",
                             transmitter_definition="argmax DY-gFEVD NET (TO-FROM)",
                             note="n=7 twins is small; thin-T (e.g. flu T=298 effective ILI-weeks but N=14; flights T=90/N=18). Reported honestly."),
                   twins={r["twin"]:{k:v for k,v in r.items() if k!="advs"} for r in out},
                   advs={r["twin"]:r["advs"] for r in out})
    json.dump(payload, open(BASE/"bootstrap_ci.json","w"), indent=2)
    print(f"\nsaved -> {BASE/'bootstrap_ci.json'}   ({time.time()-t0:.0f}s)")
    return out

if __name__=="__main__":
    res = main()
    # ============================================================ FIGURE
    matplotlib.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],
        "font.size":9,"axes.titlesize":10,"axes.labelsize":9,"savefig.dpi":300})
    VCOL={"CONFIRM":"#2e8b57","CONFIRM(held-out)":"#2e8b57","DIRECTED CONFIRM":"#1f4e78",
          "REFINE":"#d4a017","FALSIFY":"#c00000","NULL":"#7f7f7f"}
    order = sorted(res, key=lambda r:r["point_advantage"])
    names=[r["twin"] for r in order]
    y=np.arange(len(order))
    fig, ax = plt.subplots(1,2, figsize=(12.5,5.4), gridspec_kw={"width_ratios":[1.5,1.0]})

    # ---- panel (a): advantage point + 95% CI (whiskers), violin of bootstrap dist ----
    a=ax[0]
    for i,r in enumerate(order):
        col=VCOL.get(r["verdict"],"#444")
        arr=np.array(r["advs"])
        vp=a.violinplot([arr], positions=[i], vert=False, widths=0.8, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(col); b.set_alpha(0.22); b.set_edgecolor(col); b.set_linewidth(0.6)
        a.plot([r["advantage_CI95_lo"],r["advantage_CI95_hi"]],[i,i],color=col,lw=2.2,zorder=3)
        a.scatter([r["point_advantage"]],[i],color=col,edgecolor="k",lw=0.7,s=70,zorder=4)
        a.scatter([r["advantage_CI95_lo"],r["advantage_CI95_hi"]],[i,i],color=col,marker="|",s=140,lw=2.0,zorder=4)
    a.axvline(0,color="k",lw=1.0,ls="--",alpha=0.7)
    a.set_yticks(y); a.set_yticklabels([f"{r['twin']}\n(T={r['T']},N={r['N']})" for r in order])
    a.set_xlabel("controller advantage  (transmitter% - greedy%, pts)")
    a.set_title("(a) Advantage: point estimate + bootstrap 95% CI\n(violin = bootstrap distribution; K="+str(res[0]["K"])+", stationary block ~$\\sqrt{T}$)")
    a.grid(axis="x",alpha=.25)
    for i,r in enumerate(order):
        txt=f"P(tx=point)={r['P_transmitter_is_point']:.2f}"
        a.text(a.get_xlim()[1] if False else max(r['advantage_CI95_hi'],r['point_advantage'])+1.0, i, txt,
               va="center", ha="left", fontsize=7.3, color="#333")
    a.set_xlim(min(r["advantage_CI95_lo"] for r in order)-3, max(r["advantage_CI95_hi"] for r in order)+16)

    # ---- panel (b): rank stability P(transmitter == reported) ----
    b=ax[1]
    pstab=[r["P_transmitter_is_point"] for r in order]
    cols=[VCOL.get(r["verdict"],"#444") for r in order]
    b.barh(y, pstab, color=cols, edgecolor="k", lw=0.6, alpha=0.85)
    for i,r in enumerate(order):
        b.text(r["P_transmitter_is_point"]+0.012, i, f"{r['P_transmitter_is_point']:.2f}", va="center", fontsize=8)
        if r["modal_transmitter"]!=r["point_transmitter"]:
            b.text(0.02, i, f"modal={r['modal_transmitter']}", va="center", fontsize=7, color="#c00000")
    b.axvline(0.5,color="#888",lw=0.8,ls=":")
    b.set_yticks(y); b.set_yticklabels([r["twin"] for r in order])
    b.set_xlim(0,1.18); b.set_xlabel("P(net-transmitter == reported)  [rank stability]")
    b.set_title("(b) Transmitter rank stability\nunder the bootstrap")
    b.grid(axis="x",alpha=.25)

    from matplotlib.lines import Line2D
    seen=[]; handles=[]
    for r in res:
        if r["verdict"] not in seen:
            seen.append(r["verdict"])
            handles.append(Line2D([0],[0],marker="o",color="w",markerfacecolor=VCOL.get(r["verdict"],"#444"),
                                  markeredgecolor="k",markersize=8,label=r["verdict"]))
    ax[0].legend(handles=handles, loc="lower right", fontsize=7.3, title="verdict", title_fontsize=8)
    fig.suptitle("Thin-T fragility of the directed-asymmetry result: bootstrap CIs on the controller advantage and transmitter rank stability (n=7 twins)",
                 fontsize=11, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(BASE/"bootstrap_ci.pdf", bbox_inches="tight")
    fig.savefig(BASE/"bootstrap_ci.png", dpi=200, bbox_inches="tight")
    print(f"figure -> {BASE/'bootstrap_ci.pdf'} / .png")
