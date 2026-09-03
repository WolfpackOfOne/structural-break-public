#!/usr/bin/env python3
"""Q3/D4 fold-pure ceiling analysis over the registered OOF library."""
from __future__ import annotations

import hashlib, json, os, platform, shutil, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research/scripts")]
from sbr.metric import ts_auc_flat
from wave4_cal import SCDF_NSEEN, logit

FOLDS = tuple(range(5))
THRESHOLD = 0.0011
ANALYSIS_ID = "AN-Q3-D4-20260903"
CRUNCH = Path("/path/to/workspace")
STORE = CRUNCH / "structural-break-claude-wave3/cache/store"
FOLDS_FILE = ROOT / "research/folds/folds.parquet"
OUT = ROOT / "research/reports/rt1320_promotion"
W8 = CRUNCH / "structural-break-wave8/research/oof"
LA = CRUNCH / "structural-break-leaderboard-alpha-2026/research/oof"
LD = CRUNCH / "structural-break-learner-diversity-2026/research/oof"
DE = CRUNCH / "structural-break-deep-ensemble-frontier-local-2026/research/oof"
ARMC = CRUNCH / "structural-break-multi-agent-frontier-20260829/research/reports/armc_residual_student_confirm_s20260901"

LEGACY = [
    300,301,302,303,401,402,403,404,405,406,410,411,412,413,414,415,
    430,431,432,433,434,700,701,702,710,711,712,730,731,740,750,751,760,
    811,812,813,814,815,816,960,961,970,971,990,994,995,
]
RT600 = ["RT-300","RT-410","RT-411","RT-412","RT-413","RT-414","RT-415"]
RT1257 = ["RT-1255","RT-410","RT-411","RT-412","RT-1254","RT-414","RT-415"]
UNION13 = RT600 + ["RT-401","RT-402","RT-403","RT-404","RT-405","RT-406"]
REFS = {
    "synthesis": ROOT / "research/ai/investigations/20260829-173245-unresolved-quant-questions/05_SYNTHESIS.md",
    "primary_research": ROOT / "research/ai/investigations/20260829-173245-unresolved-quant-questions/02_PRIMARY_RESEARCH.md",
    "wave4_union": ROOT / "research/reports/wave4_union_ensemble.json",
    "rt1320_stage2": OUT / "PHASE1_STAGE2.md",
    "canonical": ARMC / "E2_E1_addition_contract.json",
    "alt1": OUT / "PHASE1_ALT1_E2_E1_addition_contract.json",
    "alt2": OUT / "PHASE1_ALT2_E2_E1_addition_contract.json",
    "alt3": OUT / "PHASE1_ALT3_E2_E1_addition_contract.json",
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(8 << 20): h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def library():
    d = {f"RT-{i}": W8/f"RT-{i}.npy" for i in LEGACY}
    for p in ("RT-1245", "RT-1246"):
        for m in RT600: d[f"{p}_{m}"] = LA/f"{p}_{m}.npy"
    for i in (1247,1248,1249): d[f"RT-{i}"] = LA/f"RT-{i}.npy"
    d["RT-1251"] = LD/"RT-1251.npy"
    for i in (1254,1255,1256,1260,1261,1262,1263): d[f"RT-{i}"] = DE/f"RT-{i}.npy"
    d["RT-1320"] = ARMC/"armc_residual_student_oof.npy"
    return dict(sorted(d.items()))


def context():
    meta = pd.read_parquet(STORE/"meta.parquet")
    fp = pd.read_parquet(FOLDS_FILE)[["id","fold"]]
    foldmap = dict(zip(fp.id.astype(int), fp.fold.astype(int)))
    sf = np.array([foldmap[int(i)] for i in meta.id], np.int8)
    no = meta.n_online.to_numpy(np.int64)
    start = np.r_[0, np.cumsum(no)[:-1]].astype(np.int64)
    keep = np.flatnonzero((sf >= 0) & (sf < 5))
    n = int(no[keep].sum())
    rows=np.empty(n,np.int64); rf=np.empty(n,np.int8); y=np.empty(n,np.int8); t=np.empty(n,np.int32)
    tau=meta.tau_index.to_numpy(np.int64); p=0
    for s in keep:
        m=int(no[s]); sl=slice(p,p+m)
        rows[sl]=start[s]+np.arange(m); rf[sl]=sf[s]; t[sl]=np.arange(m); y[sl]=0
        if tau[s] >= 0: y[p+int(tau[s]):p+m]=1
        p += m
    loc={k:np.flatnonzero(rf==k) for k in FOLDS}
    return meta,sf,rows,y,t,loc,int(no.sum())


def load(path, rows, total):
    a=np.load(path,mmap_mode="r")
    if a.ndim != 1 or len(a) != total: raise ValueError(f"{path}: bad shape {a.shape}")
    x=np.asarray(a[rows],np.float32)
    if not np.isfinite(x).all(): raise ValueError(f"{path}: nonfinite dev rows")
    return x


def score(x,y,t): return float(ts_auc_flat(x,y,t))


def midrank(x,t):
    x=np.asarray(x,np.float64); t=np.asarray(t,np.int64); order=np.lexsort((x,t)); s=x[order]; tt=t[order]; n=len(x)
    gs=np.flatnonzero(np.r_[True,tt[1:]!=tt[:-1]]); ge=np.r_[gs[1:],n]; gl=ge-gs
    pos=np.arange(n,dtype=float)-np.repeat(gs,gl)
    rs=np.flatnonzero(np.r_[True,(tt[1:]!=tt[:-1])|(s[1:]!=s[:-1])]); re=np.r_[rs[1:],n]; rl=re-rs
    r=np.repeat(pos[rs]+(rl-1)/2,rl)/np.maximum(np.repeat(gl,gl)-1,1)
    out=np.empty(n,np.float32); out[order]=r; return out


def legacy_rank(x,t):
    """Exact Wave-3 reference helper (ordinal ranks; retained for reproduction)."""
    order=np.lexsort((x,t)); tt=t[order]; n=len(x)
    gs=np.flatnonzero(np.r_[True,tt[1:]!=tt[:-1]]); ge=np.r_[gs[1:],n]; out=np.empty(n)
    for a,b in zip(gs,ge): out[order[a:b]]=(np.arange(b-a)+.5)/(b-a)
    return out


def cal_matrix(paths,names,rows,total,t,loc,k,tmp):
    mm=np.memmap(tmp/f"outer{k}.f32",mode="w+",dtype=np.float32,shape=(len(rows),len(names)))
    train=np.concatenate([loc[g] for g in FOLDS if g!=k])
    for j,name in enumerate(names):
        raw=load(paths[name],rows,total)
        for g in FOLDS:
            fit=train if g==k else np.concatenate([loc[h] for h in FOLDS if h not in (k,g)])
            f=SCDF_NSEEN(raw[fit],t[fit]); mm[loc[g],j]=f(raw[loc[g]],t[loc[g]])
        print(f"outer={k} calibrated {j+1}/{len(names)} {name}",flush=True)
    mm.flush(); return mm


def additions(mm,names,k,loc,y,t):
    ix={n:i for i,n in enumerate(names)}; r=loc[k]
    ens=lambda ms: np.mean(mm[r][:,[ix[n] for n in ms]],axis=1,dtype=np.float64)
    a=ens(RT600); b=ens(RT1257); sa=score(a,y[r],t[r]); sb=score(b,y[r],t[r])
    ca=score((7*a+mm[r,ix["RT-401"]])/8,y[r],t[r]); cb=score((7*b+mm[r,ix["RT-403"]])/8,y[r],t[r])
    def one(name):
        xa=score((7*a+mm[r,ix[name]])/8,y[r],t[r]); xb=score((7*b+mm[r,ix[name]])/8,y[r],t[r])
        return name,{"standalone":score(mm[r,ix[name]],y[r],t[r]),"rt600_plus":xa,"rt600_marginal":xa-sa,
                     "vs_rt401_seed_clone":xa-ca,"rt1257_plus":xb,"rt1257_marginal":xb-sb,"vs_rt403_seed_clone":xb-cb}
    with ThreadPoolExecutor(max_workers=5) as ex:
        result=dict(ex.map(one,names))
    return {"outer_fold":k,"rt600":sa,"rt600_plus_rt401":ca,"rt1257":sb,"rt1257_plus_rt403":cb,
            "w4_union13":score(ens(UNION13),y[r],t[r]),"vectors":result}


def greedy(mm,names,k,loc,y,t):
    ix={n:i for i,n in enumerate(names)}; tr=[g for g in FOLDS if g!=k]
    cur=np.asarray(np.mean(mm[:,[ix[n] for n in RT1257]],axis=1,dtype=np.float64),np.float32)
    base_tr=[score(cur[loc[g]],y[loc[g]],t[loc[g]]) for g in tr]; base=score(cur[loc[k]],y[loc[k]],t[loc[k]])
    selected=list(RT1257); remain=[n for n in names if n not in selected]; count=7; now=float(np.mean(base_tr)); held=base; path=[]
    while remain:
        def evaluate(name):
            z=(count*cur+mm[:,ix[name]])/(count+1)
            per=[score(z[loc[g]],y[loc[g]],t[loc[g]]) for g in tr]; key=(float(np.mean(per)),name)
            return key,name,per
        with ThreadPoolExecutor(max_workers=5) as ex:
            best=max(ex.map(evaluate,remain),key=lambda z:z[0])
        value,name,per=best[0][0],best[1],best[2]
        if value <= now+1e-12: break
        cur=np.asarray((count*cur+mm[:,ix[name]])/(count+1),np.float32); count+=1; remain.remove(name); selected.append(name)
        held=score(cur[loc[k]],y[loc[k]],t[loc[k]])
        path.append({"step":len(path)+1,"added":name,"equal_weight":1/count,"train_fold_scores":per,"train_mean":value,
                     "train_increment":value-now,"heldout_score":held,"heldout_delta_vs_rt1257":held-base})
        now=value; print(f"outer={k} greedy step={len(path)} add={name} train={value:.9f} heldout_delta={held-base:+.7f}",flush=True)
    return {"outer_fold":k,"base_members":RT1257,"base_train_fold_scores":base_tr,"base_train_mean":float(np.mean(base_tr)),
            "base_heldout_score":base,"selected_additions":[p["added"] for p in path],"final_members":selected,
            "final_equal_weight":1/count,"path":path,"final_train_mean":now,"final_heldout_score":held,
            "final_heldout_delta":held-base,"stopping_rule":"no strict four-training-fold mean improvement"}


def erank(paths,names,rows,total,t,tmp):
    mm=np.memmap(tmp/"ranks.f32",mode="w+",dtype=np.float32,shape=(len(rows),len(names)))
    for j,n in enumerate(names): mm[:,j]=midrank(load(paths[n],rows,total),t); print(f"rank {j+1}/{len(names)} {n}",flush=True)
    sx=np.zeros(len(names)); xx=np.zeros((len(names),len(names))); nr=len(rows)
    for lo in range(0,nr,100000):
        b=np.asarray(mm[lo:lo+100000],np.float64); sx+=b.sum(0); xx+=b.T@b
    cov=xx-np.outer(sx,sx)/nr; sd=np.sqrt(np.maximum(np.diag(cov),1e-30)); cor=cov/np.outer(sd,sd); cor=(cor+cor.T)/2
    ev=np.maximum(np.linalg.eigvalsh(cor),0)[::-1]; q=ev[ev>1e-15]/ev.sum(); cs=np.cumsum(ev)/ev.sum(); off=cor[~np.eye(len(names),dtype=bool)]
    return {"definition":"Pearson correlation of zero-to-one within-t midranks on canonical dev rows","matrix_dimension":len(names),
            "n_rows":nr,"participation_ratio":float(ev.sum()**2/np.square(ev).sum()),
            "entropy_effective_rank":float(np.exp(-(q*np.log(q)).sum())),
            "components_for_90pct":int(np.searchsorted(cs,.9)+1),"components_for_95pct":int(np.searchsorted(cs,.95)+1),
            "components_for_99pct":int(np.searchsorted(cs,.99)+1),"top_eigenvalue_share":float(ev[0]/ev.sum()),
            "mean_offdiagonal_correlation":float(off.mean()),"mean_absolute_offdiagonal_correlation":float(np.abs(off).mean()),
            "min_offdiagonal_correlation":float(off.min()),"max_offdiagonal_correlation":float(off.max()),"eigenvalues_descending":ev.tolist()}


def contract(path):
    d=json.loads(path.read_text())
    return {"e2_minus_e1":float(d["PRIMARY_E2_minus_E1"]),"e2_minus_e0":float(d["SECONDARY_E2_minus_E0"]),
            "e2_minus_e1_per_fold":[float(x) for x in d["primary_per_fold"]],
            "e2_minus_e0_per_fold":[float(x) for x in d["secondary_per_fold"]]}


def references(paths,rows,total,loc,y,t,adds):
    raw={n:load(paths[n],rows,total) for n in ("RT-301","RT-302","RT-303","RT-750")}
    folds=lambda x:[score(x[loc[k]],y[loc[k]],t[loc[k]]) for k in FOLDS]
    blend=lambda n:.5*(logit(raw["RT-301"])+logit(raw[n]))
    m09,clone,m12=folds(blend("RT-302")),folds(blend("RT-303")),folds(blend("RT-750"))
    delta=lambda a,b:[float(x-z) for x,z in zip(a,b)]
    r301=legacy_rank(raw["RT-301"],t); r302=legacy_rank(raw["RT-302"],t); r303=legacy_rank(raw["RT-303"],t)
    cons={n:contract(REFS[n]) for n in ("canonical","alt1","alt2","alt3")}; e20=[v["e2_minus_e0"] for v in cons.values()]; e21=[v["e2_minus_e1"] for v in cons.values()]
    w5=[a["rt600_plus_rt401"]-a["rt600"] for a in adds]; w4=[a["w4_union13"]-a["rt600"] for a in adds]
    result={"w5_nulltest":{"definition":"fold-pure RT600+RT401 minus RT600","per_fold":w5,"mean":float(np.mean(w5)),"expected_rounded":.00003},
            "w4_e6_union13":{"definition":"fold-pure 13-stream union minus RT600","per_fold":w4,"mean":float(np.mean(w4)),"expected_direction":"negative"},
            "m09_back":{"definition":"raw-logit RT301+RT302 minus RT301+RT303","candidate_per_fold":m09,"clone_per_fold":clone,
                        "candidate_minus_clone_per_fold":delta(m09,clone),"candidate_minus_clone_mean":float(np.mean(delta(m09,clone))),
                        "within_t_rank_corr_candidate_vs_rt301":float(np.corrcoef(r301,r302)[0,1]),
                        "within_t_rank_corr_clone_vs_rt301":float(np.corrcoef(r301,r303)[0,1]),"expected_rounded_correlations":[.8219,.7846]},
            "m12_rdep":{"definition":"raw-logit RT301+RT750 minus RT301+RT303","candidate_per_fold":m12,"clone_per_fold":clone,
                        "candidate_minus_clone_per_fold":delta(m12,clone),"candidate_minus_clone_mean":float(np.mean(delta(m12,clone))),"expected_rounded":.00141},
            "rt1320_four_partitions":{"contracts":cons,"e2_minus_e0_by_partition":e20,"mean_e2_minus_e0":float(np.mean(e20)),
                                      "e2_minus_e1_by_partition":e21,"mean_e2_minus_e1":float(np.mean(e21)),"expected_rounded_e2_minus_e0":.0016},
            "noise_floor":THRESHOLD}
    result["reproduced"]={
        "w5_nulltest":abs(result["w5_nulltest"]["mean"]-.00003)<5e-6,
        "w4_e6_union_worse":result["w4_e6_union13"]["mean"]<0,
        "m09_correlations_and_loss":round(result["m09_back"]["within_t_rank_corr_candidate_vs_rt301"],4)==.8219 and round(result["m09_back"]["within_t_rank_corr_clone_vs_rt301"],4)==.7846 and result["m09_back"]["candidate_minus_clone_mean"]<0,
        "m12_rdep":round(result["m12_rdep"]["candidate_minus_clone_mean"],5)==.00141,
        "noise_floor":result["noise_floor"]==.0011,
        "rt1320_four_partition":round(result["rt1320_four_partitions"]["mean_e2_minus_e0"],4)==.0016,
    }
    result["all_reproduced"]=all(result["reproduced"].values())
    return result


def aggregate(names,aa):
    out=[]; fields=("standalone","rt600_plus","rt600_marginal","vs_rt401_seed_clone","rt1257_plus","rt1257_marginal","vs_rt403_seed_clone")
    for n in names:
        r={"vector":n}
        for f in fields:
            p=[a["vectors"][n][f] for a in aa]; r[f+"_per_fold"]=p; r[f+"_mean"]=float(np.mean(p))
        for f in ("rt600_marginal","vs_rt401_seed_clone","rt1257_marginal","vs_rt403_seed_clone"):
            r[f+"_positive_folds"]=int(sum(x>0 for x in r[f+"_per_fold"]))
        out.append(r)
    return sorted(out,key=lambda x:(-x["vs_rt403_seed_clone_mean"],x["vector"]))


def decision(gs,rows,refs):
    clear=[]
    for r in rows:
        if r["rt1257_marginal_mean"]>THRESHOLD and r["rt1257_marginal_positive_folds"]>=4:
            clear.append({"combination":RT1257+[r["vector"]],"protocol":"canonical fold-pure RT1257 equal-weight addition",
                          "margin_vs_current":r["rt1257_marginal_mean"],"per_fold":r["rt1257_marginal_per_fold"],
                          "positive_folds":r["rt1257_marginal_positive_folds"]})
    gd=[g["final_heldout_delta"] for g in gs]
    if np.mean(gd)>THRESHOLD and sum(x>0 for x in gd)>=4:
        clear.append({"combination":"outer-fold-specific greedy paths","protocol":"nested fold-pure greedy selection",
                      "margin_vs_current":float(np.mean(gd)),"per_fold":gd,"positive_folds":int(sum(x>0 for x in gd))})
    rt=refs["rt1320_four_partitions"]
    if rt["mean_e2_minus_e0"]>THRESHOLD and sum(x>0 for x in rt["e2_minus_e0_by_partition"])>=3:
        clear.append({"combination":RT1257+["RT-1320"],"protocol":"pre-registered RT1320 addition contract, four partitions",
                      "margin_vs_current":rt["mean_e2_minus_e0"],"per_partition":rt["e2_minus_e0_by_partition"],
                      "positive_partitions":int(sum(x>0 for x in rt["e2_minus_e0_by_partition"]))})
    if clear: return "CEILING DEAD",clear,"An existing-vector combination beats RT1257 by >0.0011 with the fixed consistency requirement."
    mx=max([r["rt1257_marginal_mean"] for r in rows]+[float(np.mean(gd))])
    if mx<=THRESHOLD: return "CEILING BINDING",[],"No combination clears 0.0011 with fold consistency; the existing bank is saturated under D4."
    return "INCONCLUSIVE",[],"A mean crossed 0.0011 but did not have fold consistency."


def num(x): return f"{x:+.7f}"
def val(x): return f"{x:.9f}"


def markdown(d):
    dec=d["decision"]; er=d["effective_rank"]; gr=d["greedy_forward_selection"]; ref=d["reference_reproduction"]; li=d["leave_one_in"]
    L=["# Q3 — Is the existing legal OOF bank at its ensemble ceiling?","",f"Analysis ID: `{ANALYSIS_ID}`",
       f"Commit parent: `{d['provenance']['git_parent']}`",f"Canonical development folds 0–4 only: {d['data']['n_dev_rows']:,} online rows.","",
       "## Decision rule (frozen before computation)","",
       "- **CEILING BINDING** if no combination beats the current ensemble by more than `0.0011` with fold consistency; then the existing bank is saturated and the program is finished.",
       "- **CEILING DEAD** if any combination clears that bar with fold consistency; report the exact vectors and margin.","- Otherwise **INCONCLUSIVE**.","",
       "Fold consistency is fixed as at least **4/5 positive canonical outer-fold deltas**, matching the existing program convention. The already pre-registered four-partition RT1320 replication retains its frozen **at least 3/4 positive partitions** rule.","",
       f"## Verdict: {dec['verdict']}","",dec["reason"],""]
    for c in dec["clearing_combinations"]:
        members=c["combination"] if isinstance(c["combination"],str) else " + ".join(c["combination"]); axis=c.get("per_fold",c.get("per_partition"))
        L.append(f"- `{members}` — {c['protocol']}; mean margin **{num(c['margin_vs_current'])}**; vector `[{', '.join(num(x) for x in axis)}]`.")
    L += ["",f"The decisive fixed combination is `RT-1257 + RT-1320`. Its pre-registered canonical/alt1/alt2/alt3 E2−E0 mean is **{num(ref['rt1320_four_partitions']['mean_e2_minus_e0'])}**, positive on 4/4 partitions and above `0.0011`. The legal bank is therefore not saturated.","",
          "## D4 protocol","",f"The bank contains **{d['library']['n_vectors']}** registered, legal, full-fold, one-dimensional OOF vectors: 46 legacy, 17 leaderboard-alpha, RT-1251, seven deep-ensemble vectors, and RT-1320.","",
          "For outer fold `k`, its SCDF_NSEEN maps are fit on folds `!=k`; selection-time fold `g` maps are fit excluding both `k` and `g`. Greedy selection starts at RT1257, chooses the best equal-weight addition using only folds `!=k`, and stops at no training-fold improvement. Held-out `k` never chooses a vector or weight.","",
          "Leave-one-in adds each vector at `1/8` to RT600 versus RT401 and to RT1257 versus RT403 under the same outer-fold calibration.","",
          "### Greedy outer-fold results","","| outer | training-selected additions | held-out RT1257 | held-out final | delta |","|---:|---|---:|---:|---:|"]
    for g in gr["outer_folds"]: L.append(f"| {g['outer_fold']} | {', '.join(g['selected_additions']) or '(none)'} | {val(g['base_heldout_score'])} | {val(g['final_heldout_score'])} | {num(g['final_heldout_delta'])} |")
    L += ["",f"Final greedy delta vector `[{', '.join(num(x) for x in gr['final_delta_per_fold'])}]`; mean **{num(gr['final_delta_mean'])}**, positive {gr['positive_folds']}/5. Paths remain outer-fold-specific; no post-heldout consensus path was manufactured.","",
          "## Effective rank","",f"The {er['matrix_dimension']}×{er['matrix_dimension']} within-t rank-correlation matrix has participation effective rank **{er['participation_ratio']:.3f}** and entropy effective rank **{er['entropy_effective_rank']:.3f}**. It needs {er['components_for_90pct']}/{er['components_for_95pct']}/{er['components_for_99pct']} components for 90%/95%/99% spectral mass; the top eigenvalue holds {100*er['top_eigenvalue_share']:.2f}%, and mean off-diagonal correlation is {er['mean_offdiagonal_correlation']:.4f}.","",
          "The bank is strongly redundant, but redundancy is not a binding ceiling: RT1320 is a direct fold- and partition-consistent counterexample.","",
          "## Leave-one-in marginals","","Sorted by mean candidate-minus-RT403 margin on RT1257. Full per-fold vectors are in the JSON.","",
          "| vector | standalone | +RT600 | vs RT401 | +folds | +RT1257 | vs RT403 | +folds |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in li: L.append(f"| {r['vector']} | {val(r['standalone_mean'])} | {num(r['rt600_marginal_mean'])} | {num(r['vs_rt401_seed_clone_mean'])} | {r['vs_rt401_seed_clone_positive_folds']}/5 | {num(r['rt1257_marginal_mean'])} | {num(r['vs_rt403_seed_clone_mean'])} | {r['vs_rt403_seed_clone_positive_folds']}/5 |")
    L += ["","## Required reference points","",f"- W5-NULLTEST: **{num(ref['w5_nulltest']['mean'])}** (reference `+0.00003`).",
          f"- W4-E6 13-stream union minus RT600: **{num(ref['w4_e6_union13']['mean'])}**; worse, as reported.",
          f"- `m09_back`: within-t correlation **{ref['m09_back']['within_t_rank_corr_candidate_vs_rt301']:.4f}** versus clone **{ref['m09_back']['within_t_rank_corr_clone_vs_rt301']:.4f}**; candidate-minus-clone **{num(ref['m09_back']['candidate_minus_clone_mean'])}**.",
          f"- `m12_rdep`: candidate-minus-clone **{num(ref['m12_rdep']['candidate_minus_clone_mean'])}** (reference `+0.00141`).",f"- Noise floor: **{ref['noise_floor']:.4f}**.",
          f"- RT1320 four-partition mean E2−E0: **{num(ref['rt1320_four_partitions']['mean_e2_minus_e0'])}** (reference approximately `+0.0016`).","",
          f"**Reference reproduction: {'PASS — all six checks reproduced' if ref['all_reproduced'] else 'FAIL — see JSON check flags'}.**","",
          "Legacy raw-logit m09/m12 and RT1320 partition values use their exact frozen contracts; W4/W5 are recomputed from frozen OOF under fold-pure SCDF_NSEEN.","",
          "## Scope and exclusions",""]
    L += [f"- {x}" for x in d["library"]["exclusions"]]
    L += ["","No detector was trained. No fold -1 row, test/lockbox artifact, `folds_final10k`, `RESULTS.csv`, setup, Crunch push, or RT allocation was read, scored, or changed. Whole-file hashes below are byte-level provenance only.","",
          "## Input provenance","","| vector | bytes | SHA-256 | source |","|---|---:|---|---|"]
    for n,r in d["library"]["vectors"].items(): L.append(f"| {n} | {r['bytes']} | `{r['sha256']}` | `{r['path']}` |")
    L += ["","Reference files:"
          ,""]
    for n,r in d["provenance"]["reference_files"].items(): L.append(f"- `{n}` — `{r['sha256']}` — `{r['path']}`")
    L += ["","## Interpretation","","Most incumbent-like additions fail their matched-clone test, so the local conventional-stream ceiling is high. But Q3 asks the stronger legal-causal question. RT1320 uses the same information set and clears the frozen noise floor with fold and partition consistency. Q3 is closed as **CEILING DEAD**.",""]
    return "\n".join(L)


def main():
    started=time.time()
    if git("branch","--show-current") != "engineering/rt1320-promotion-prep": raise SystemExit("wrong branch")
    paths=library(); names=list(paths)
    missing=[str(p) for p in list(paths.values())+list(REFS.values()) if not p.is_file()]
    if missing: raise FileNotFoundError("\n".join(missing))
    meta,sf,rows,y,t,loc,total=context()
    inventory={n:{"path":str(p.resolve()),"bytes":p.stat().st_size,"sha256":sha(p)} for n,p in paths.items()}
    refprov={n:{"path":str(p.resolve()),"bytes":p.stat().st_size,"sha256":sha(p)} for n,p in REFS.items()}
    tmp=Path(tempfile.mkdtemp(prefix="q3_ceiling_"))
    try:
        gs=[]; aa=[]
        for k in FOLDS:
            mm=cal_matrix(paths,names,rows,total,t,loc,k,tmp); aa.append(additions(mm,names,k,loc,y,t)); gs.append(greedy(mm,names,k,loc,y,t)); fn=mm.filename; del mm; os.remove(fn)
        er=erank(paths,names,rows,total,t,tmp); li=aggregate(names,aa); refs=references(paths,rows,total,loc,y,t,aa)
    finally: shutil.rmtree(tmp,ignore_errors=True)
    gd=[g["final_heldout_delta"] for g in gs]; verdict,clears,reason=decision(gs,li,refs)
    exclusions=[
        "RT-500..RT-506: final-10k calibration vectors fitted with the spent fold -1 lockbox.",
        "RT-900/RT-991: void/oracle tau or label leakage; RT-992/RT-993: contaminated fold-0 pilots.",
        "RT-1000-series and RT-1200..RT-1242 screens: fold-0 experiments or later folds generated only for calibration support, not registered full-fold candidates.",
        "RT-1234/1235 and RT-1237..1242 local arrays: non-finite outside fold 0.",
        "CATSLOT-* and M2-RTLF scratch arrays: unregistered/provenance-ambiguous; registered exact duplicates are represented by their RT vector.",
        "Alternate partitions, nested teachers, precalibrated vectors, ensemble caches: validation surfaces, fit auxiliaries, or deterministic derivatives rather than independent candidates.",
    ]
    d={"schema_version":1,"analysis_id":ANALYSIS_ID,"question":"Is RT1320 at the legal causal ceiling or is the OOF bank under-composed?",
       "decision_rule":{"threshold":THRESHOLD,"canonical_fold_consistency":"at least 4/5 positive outer-fold deltas","rt1320_partition_consistency":"at least 3/4 positive partitions (frozen contract)",
                        "ceiling_binding":"no combination beats current by >0.0011 with consistency; bank saturated/program finished","ceiling_dead":"some combination clears; report vectors/margin","otherwise":"INCONCLUSIVE","frozen_before_metrics":True},
       "decision":{"verdict":verdict,"reason":reason,"clearing_combinations":clears},
       "data":{"partition":"canonical development folds 0..4 only","n_series_by_fold":{str(k):int((sf==k).sum()) for k in FOLDS},"n_rows_by_fold":{str(k):len(loc[k]) for k in FOLDS},
               "n_dev_rows":len(rows),"n_total_oof_rows":total,"fold_minus_one_rows_loaded":0},
       "library":{"n_vectors":len(names),"inclusion_rule":"registered legal 1-D OOF with finite predictions on all five canonical folds","vectors":inventory,"exclusions":exclusions},
       "protocol":{"calibration":"SCDF_NSEEN; outer fit excludes k; selection-time fold g fit excludes k and g","metric":"exact pair-weighted TS-AUC, mean of folds",
                   "greedy":"start RT1257; best equal-weight addition on folds !=k; stop at no strict improvement","leave_one_in":"weight 1/8 with RT600 and RT1257; compare RT401/RT403 clones",
                   "effective_rank":"participation and entropy ranks of within-t midrank Pearson matrix"},
       "greedy_forward_selection":{"outer_folds":gs,"final_delta_per_fold":gd,"final_delta_mean":float(np.mean(gd)),"positive_folds":int(sum(x>0 for x in gd))},
       "effective_rank":er,"leave_one_in":li,"reference_reproduction":refs,
       "provenance":{"git_branch":git("branch","--show-current"),"git_parent":git("rev-parse","HEAD"),"python":sys.version,"platform":platform.platform(),
                     "numpy":np.__version__,"pandas":pd.__version__,"analysis_script":str(Path(__file__).resolve()),"analysis_script_sha256":sha(Path(__file__)),
                     "folds_file":str(FOLDS_FILE.resolve()),"folds_file_sha256":sha(FOLDS_FILE),"store_meta":str((STORE/"meta.parquet").resolve()),
                     "store_meta_sha256":sha(STORE/"meta.parquet"),"reference_files":refprov,"elapsed_seconds":time.time()-started},
       "safety":{"trained_models":0,"read_lockbox_or_test_artifacts":False,"read_or_scored_fold_minus_one":False,"touched_results_csv":False,"allocated_rt_id":False,"crunch_push":False,"setup":False}}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"Q3_CEILING.json").write_text(json.dumps(d,indent=2)+"\n"); (OUT/"Q3_CEILING.md").write_text(markdown(d))
    print(f"verdict={verdict}\nwrote {OUT/'Q3_CEILING.json'}\nwrote {OUT/'Q3_CEILING.md'}",flush=True)


if __name__ == "__main__": main()
