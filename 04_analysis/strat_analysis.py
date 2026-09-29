#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""strat_analysis.py — strat variant analysis (based on the partial results trained so far).
   Rule:  exp1 and its by-products (results/, param/, embeddings) = reuse sizectrl
          exp2·3·4 and wrong_list / filelist = strat.
   Target claims: 1(target signal) · 2(convergence) · 4(eval distortion) · 5(feat importance).
   claim3 (embedding t-SNE) excluded since there is no strat embedding dump (needs a separate heavy step).
   Partial data allowed: if a needed exp is missing, only that (model,ds) computation is skipped."""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import csv, re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = RP.ROOT                       # 02_SCIE_NOISE
M03  = WORK / "03_model"; DSROOT = WORK / "01_dataset"; AN = RP.ANALYSIS_OUT
MODELS = ["rf","xgboost","2dcnn","etbert","netmamba","trafficformer","yatc"]
DL     = ["2dcnn","etbert","netmamba","trafficformer","yatc"]
DSS    = ["vpn16","tor16","tls1.3","cispec","ustc16","cic17","cic18","iot23"]
V = "strat"

# ── accuracy loading: exp1←sizectrl, exp2/3/4←strat (tolerant per cell) ──────────────
def _f(x):
    try: return float(x)
    except: return None
def load_acc():
    acc = {}
    def read(variant):
        d = {}
        for m in MODELS:
            p = RP.RESULT_TABLES/variant/f"{m}.csv"
            if not p.exists(): continue
            for r in csv.DictReader(open(p, encoding="utf-8-sig")):
                d[(m, r["dataset"])] = {k:_f(r.get(k,"")) for k in ("exp1","exp2","exp3","exp4")}
        return d
    strat, szc = read("strat"), read("sizectrl")
    keys = set(strat) | set(szc)
    for key in keys:
        s = strat.get(key, {}); z = szc.get(key, {})
        acc[key] = {"exp1": z.get("exp1"),                    # exp1 = sizectrl
                    "exp2": s.get("exp2"), "exp3": s.get("exp3"), "exp4": s.get("exp4")}
    return acc

# ═══ claim1 : target signal (exp1 refined vs exp3 noise) ══════════════════════
def claim1(acc):
    OUT = AN/"01_target_signal"; (OUT/V).mkdir(parents=True, exist_ok=True); (OUT/"figure"/V).mkdir(parents=True, exist_ok=True)
    summ=[]
    for ds in DSS:
        rows, deltas = [], []
        for m in MODELS:
            a = acc.get((m,ds));
            if not a or a["exp1"] is None or a["exp3"] is None: continue
            d=(a["exp1"]-a["exp3"])*100
            rows.append([m,f"{a['exp1']*100:.2f}",f"{a['exp3']*100:.2f}",f"{d:+.2f}"])
            deltas.append((m,a["exp1"]*100,a["exp3"]*100,d))
        if not rows: continue
        with open(OUT/V/f"{ds}.csv","w",newline="",encoding="utf-8-sig") as f:
            w=csv.writer(f); w.writerow(["model","exp1_refined(%)","exp3_noisy(%)","Δ(e1-e3,%p)"]); w.writerows(rows)
        ms=[x[0] for x in deltas]; e1=[x[1] for x in deltas]; e3=[x[2] for x in deltas]; xs=range(len(ms)); wb=0.38
        fig,ax=plt.subplots(figsize=(9,4.5))
        ax.bar([x-wb/2 for x in xs],e1,wb,label="exp1 denoised-train",color="#1F3864")
        ax.bar([x+wb/2 for x in xs],e3,wb,label="exp3 noisy-train",color="#C0504D")
        ax.set_xticks(list(xs)); ax.set_xticklabels(ms,rotation=20,fontsize=8)
        ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(0,100)
        avg=sum(x[3] for x in deltas)/len(deltas)
        ax.set_title(f"{ds} / strat — exp1 vs exp3 (mean Δ={avg:+.1f}%p)",fontsize=10)
        ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.3); fig.tight_layout()
        fig.savefig(OUT/"figure"/V/f"{ds}.png",dpi=130); plt.close(fig)
        summ.append([ds,f"{avg:+.2f}",sum(1 for x in deltas if x[3]>0),len(deltas)])
    with open(OUT/V/"_summary.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["dataset","mean Δ(e1-e3,%p)","n_models_refined_better","total"]); w.writerows(summ)
    if summ:
        fig,ax=plt.subplots(figsize=(9,4)); vals=[float(s[1]) for s in summ]
        ax.bar([s[0] for s in summ],vals,color=["#1F3864" if v>0 else "#C0504D" for v in vals])
        ax.axhline(0,color="k",lw=0.8); ax.set_ylabel("mean Δ exp1−exp3 (%p)")
        ax.set_title("Target-signal interference (strat) — higher = clean training wins",fontsize=10)
        ax.tick_params(axis="x",rotation=25,labelsize=8); ax.grid(axis="y",alpha=0.3); fig.tight_layout()
        fig.savefig(OUT/"figure"/f"summary_delta_{V}.png",dpi=130); plt.close(fig)
    return summ

# ═══ claim2 : convergence (exp1=sizectrl epochs vs exp3=strat epochs) ════════════
def _epochs(variant,m,ds,e):
    p=RP.MODEL_RESULTS/variant/m/ds/f"{m}_{ds}_exp{e}.csv"
    if not p.exists(): return []
    out=[]
    for r in csv.DictReader(open(p,encoding="utf-8-sig")):
        g=lambda k:_f(r.get(k,""))
        out.append((int(r["epoch"]),g("train_loss"),g("train_accuracy"),g("test_accuracy")))
    return out
def claim2():
    OUT=AN/"02_convergence"; n=0
    for ds in DSS:
        for m in DL:
            e1=_epochs("sizectrl",m,ds,1); e3=_epochs("strat",m,ds,3)
            if not e1 and not e3: continue
            (OUT/V/"csv"/ds).mkdir(parents=True,exist_ok=True); (OUT/V/"figure"/ds).mkdir(parents=True,exist_ok=True)
            d1={x[0]:x for x in e1}; d3={x[0]:x for x in e3}; eps=sorted(set(d1)|set(d3))
            with open(OUT/V/"csv"/ds/f"{m}.csv","w",newline="",encoding="utf-8-sig") as f:
                w=csv.writer(f); w.writerow(["epoch","e1_loss","e1_train_acc","e1_test_acc","e3_loss","e3_train_acc","e3_test_acc"])
                for ep in eps:
                    a=d1.get(ep,(ep,None,None,None)); b=d3.get(ep,(ep,None,None,None)); w.writerow([ep,a[1],a[2],a[3],b[1],b[2],b[3]])
            fig,ax=plt.subplots(1,2,figsize=(11,4.2)); x1=[x[0] for x in e1]; x3=[x[0] for x in e3]
            ax[0].plot(x1,[x[1] for x in e1],"-o",ms=3,label="exp1 denoised(sizectrl)",color="#1F3864")
            ax[0].plot(x3,[x[1] for x in e3],"-s",ms=3,label="exp3 noisy(strat)",color="#C0504D")
            ax[0].set_title("train loss"); ax[0].set_xlabel("epoch"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
            ax[1].plot(x1,[x[2] for x in e1],"-o",ms=3,label="exp1 train_acc",color="#1F3864")
            ax[1].plot(x3,[x[2] for x in e3],"-s",ms=3,label="exp3 train_acc",color="#C0504D")
            t1=[(x[0],x[3]) for x in e1 if x[3] is not None]; t3=[(x[0],x[3]) for x in e3 if x[3] is not None]
            if t1: ax[1].plot([p[0] for p in t1],[p[1] for p in t1],"--^",ms=4,label="exp1 test_acc",color="#2E75B6")
            if t3: ax[1].plot([p[0] for p in t3],[p[1] for p in t3],"--v",ms=4,label="exp3 test_acc",color="#E8A33D")
            ax[1].set_title("accuracy"); ax[1].set_xlabel("epoch"); ax[1].set_ylim(0,1); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7)
            fig.suptitle(f"{ds} / {m} — convergence (strat: e1=sizectrl, e3=strat)",fontsize=11); fig.tight_layout()
            fig.savefig(OUT/V/"figure"/ds/f"{m}.png",dpi=130); plt.close(fig); n+=1
    return n

# ═══ claim4 : eval distortion (exp1 refined vs exp4 unrefined) + Lift ═════════════
def _num(s): m=re.search(r"\d+",str(s)); return m.group(0) if m else ""
def _noise_keys(ds):
    fl=DSROOT/ds/"00_filelist_strat"; fn,fd=fl/"list_test_noisy.csv",fl/"list_test_denoised.csv"
    if not (fn.exists() and fd.exists()): return None
    n=pd.read_csv(fn,dtype=str,na_filter=False,encoding="utf-8-sig"); d=pd.read_csv(fd,dtype=str,na_filter=False,encoding="utf-8-sig")
    n.columns=[c.strip().lstrip("﻿") for c in n.columns]; d.columns=[c.strip().lstrip("﻿") for c in d.columns]
    nk=set(zip(n["filename"],n["session_id"])); dk=set(zip(d["filename"],d["session_id"])); noise=nk-dk
    return dict(noise=noise,noise_num=set((f,_num(s)) for f,s in noise),base=len(noise)/max(len(nk),1))
def claim4(acc):
    OUT=AN/"04_fail_to_eval"; (OUT/V).mkdir(parents=True,exist_ok=True); (OUT/"figure").mkdir(parents=True,exist_ok=True)
    summ=[]
    for ds in DSS:
        rows=[]
        for m in MODELS:
            a=acc.get((m,ds));
            if not a or a["exp1"] is None or a["exp4"] is None: continue
            e2s=f"{a['exp2']*100:.2f}" if a["exp2"] is not None else "—"
            d12=f"{(a['exp1']-a['exp2'])*100:+.2f}" if a["exp2"] is not None else "—"
            rows.append([m,f"{a['exp1']*100:.2f}",f"{a['exp4']*100:.2f}",f"{(a['exp1']-a['exp4'])*100:+.2f}",e2s,d12])
        if not rows: continue
        with open(OUT/V/f"{ds}.csv","w",newline="",encoding="utf-8-sig") as f:
            w=csv.writer(f); w.writerow(["model","exp1_refined(%)","exp4_unrefined_practice(%)","Δ(e1-e4,%p)","exp2_noisy_test(%)","Δ(e1-e2,%p)"]); w.writerows(rows)
        k=_noise_keys(ds); lift=base=wr=None
        if k:
            base=k["base"]; ratios=[]
            for m in MODELS:
                p=RP.WRONG_LIST/V/m/ds/"exp2_wrong.csv"
                if not p.exists(): continue
                wdf=pd.read_csv(p,dtype=str,na_filter=False); nn=len(wdf)
                if nn==0: continue
                hit=sum(((fn,sid) in k["noise"]) if re.search("[a-zA-Z]",sid) else ((fn,_num(sid)) in k["noise_num"])
                        for fn,sid in zip(wdf["filename"],wdf["session_id"]))
                ratios.append(hit/nn)
            if ratios and base>0: wr=sum(ratios)/len(ratios); lift=wr/base
        pool=[(m,ds) for m in MODELS if acc.get((m,ds)) and acc[(m,ds)]["exp1"] is not None and acc[(m,ds)]["exp4"] is not None]
        d14=sum(acc[k2]["exp1"]-acc[k2]["exp4"] for k2 in pool)/max(1,len(pool))*100
        if lift is not None:      verdict = "⚠️Lift<1" if lift<1 else "holds"
        elif base is None:        verdict = "not_counted(no filelist)"
        elif base==0:             verdict = "control(clean)"
        else:                     verdict = "not_counted(no wrong_list)"
        summ.append([ds,f"{d14:+.2f}",f"{base*100:.1f}" if base is not None else "—",
                     f"{wr*100:.1f}" if wr is not None else "—",f"{lift:.2f}" if lift is not None else "—", verdict])
    with open(OUT/V/"_summary.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["dataset","exp1-exp4(%p)","base_noise_rate(%)","wrong_noise_rate(%)","Lift","verdict"]); w.writerows(summ)
    lv=[(s[0],float(s[4])) for s in summ if s[4] not in ("—","0.00")]
    if lv:
        fig,ax=plt.subplots(figsize=(9,4)); cols=["#1F3864" if v>=1.3 else ("#7F9DB9" if v>=1 else "#C0504D") for _,v in lv]
        ax.bar([x[0] for x in lv],[x[1] for x in lv],color=cols); ax.axhline(1,color="k",lw=0.8,ls="--")
        ax.set_ylabel("Lift (wrong-noise / base)")
        ax.set_title("Evaluation distortion Lift (strat) — >1 = noise over-represented in errors",fontsize=10)
        ax.tick_params(axis="x",rotation=25,labelsize=8); ax.grid(axis="y",alpha=0.3); fig.tight_layout()
        fig.savefig(OUT/"figure"/f"lift_{V}.png",dpi=130); plt.close(fig)
    return summ

# ═══ claim5 : feat importance (rf exp1=sizectrl vs exp3=strat) ══════════════
ART=re.compile(r"(?:ip_id|checksum|ip_ttl|dsfield|ip_flags|_frag|port|addr|src|dst|ip_ver|ip_total_len)",re.I)
def _imp(variant,ds,exp):
    p=RP.PARAM/variant/"rf"/ds/f"exp{exp}"/"feature_importance.csv"
    if not p.exists(): return None
    df=pd.read_csv(p); col="importance" if "importance" in df.columns else "gain"
    s=df.set_index(df.columns[0])[col]; return s/s.sum() if s.sum()>0 else s
def claim5():
    OUT=AN/"05_feat_importance"; (OUT/V).mkdir(parents=True,exist_ok=True); (OUT/"figure").mkdir(parents=True,exist_ok=True)
    summ=[]
    for ds in DSS:
        a=_imp("sizectrl",ds,1); b=_imp("strat",ds,3)     # e1=sizectrl, e3=strat
        if a is None or b is None: continue
        idx=a.index.union(b.index); a=a.reindex(idx,fill_value=0); b=b.reindex(idx,fill_value=0)
        rho=a.rank().corr(b.rank())
        art1=a[a.index.to_series().str.contains(ART,regex=True)].sum(); art3=b[b.index.to_series().str.contains(ART,regex=True)].sum()
        out=pd.DataFrame({"exp1_refined":a,"exp3_noisy":b}); out["delta"]=out["exp3_noisy"]-out["exp1_refined"]
        out.sort_values("exp3_noisy",ascending=False).head(25).round(6).to_csv(OUT/V/f"{ds}_rf_top25.csv",encoding="utf-8-sig")
        summ.append([ds,f"{rho:.3f}",f"{art1*100:.1f}",f"{art3*100:.1f}",f"{(art3-art1)*100:+.1f}",
                     a.idxmax(),b.idxmax(),("⚠️reversed" if (art3-art1)<0 else ("strong_distortion" if (art3-art1)*100>5 else "weak"))])
    with open(OUT/V/"_summary.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["dataset","rankρ(e1,e3)","artifact_mass_e1(%)","_e3(%)","Δ(%p)","top1_e1","top1_e3","verdict"]); w.writerows(summ)
    if summ:
        fig,ax=plt.subplots(figsize=(9,4)); vals=[float(s[4]) for s in summ]
        ax.bar([s[0] for s in summ],vals,color=["#C0504D" if v<0 else "#1F3864" for v in vals]); ax.axhline(0,color="k",lw=0.8)
        ax.set_ylabel("artifact-mass Δ exp3−exp1 (%p)")
        ax.set_title("Importance distortion (strat) — noise shifts mass to artifacts",fontsize=10)
        ax.tick_params(axis="x",rotation=25,labelsize=8); ax.grid(axis="y",alpha=0.3); fig.tight_layout()
        fig.savefig(OUT/"figure"/f"artifact_delta_{V}.png",dpi=130); plt.close(fig)
    return summ

if __name__=="__main__":
    acc=load_acc()
    s1=claim1(acc); n2=claim2(); s4=claim4(acc); s5=claim5()
    def pr(title,summ,cols):
        print(f"\n########## {title} ##########"); print("  "+" | ".join(cols))
        for r in summ: print("  "+" | ".join(str(x) for x in r))
    pr("claim1 target-signal (exp1−exp3, + means refined better)",s1,["ds","meanΔ%p","refined_better","total"])
    print(f"\n########## claim2 convergence figure {n2} files (e1=sizectrl / e3=strat) ##########")
    pr("claim4 eval-distortion (exp1−exp4 / Lift)",s4,["ds","e1-e4%p","base%","wrong%","Lift","verdict"])
    pr("claim5 feat-importance (rankρ / artifact massΔ)",s5,["ds","rankρ","art_e1%","art_e3%","Δ%p","verdict7"])
    print("\n[strat_analysis done]  claim3(t-SNE)=excluded (no strat embedding dump)")
