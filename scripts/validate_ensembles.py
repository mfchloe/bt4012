from pathlib import Path
exec(open(Path(__file__).with_name("validate_graph_features.py")).read().split("run(\"base 165\")")[0])
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from scipy.stats import rankdata
EASY=(24,26,28,30); HARD=(20,22); SEEDS=(42,1,2)

def build(fit, val, nb, iters=1, hop2=False):
    F=list(FEATURE_COLS)+STRUCT
    if nb:
        feats=list(FEATURE_COLS)+STRUCT
        for it in range(iters):
            s=stage1_scores(fit,val,feats)
            N=neighbour_score_features(ke,s,prefix=f"nb{it}")
            if hop2:
                N2=neighbour_score_features(ke,N[f"nb{it}_all_mean"].reindex(s.index),prefix=f"h2_{it}")
                N=N.join(N2[[f"h2_{it}_all_mean",f"h2_{it}_all_max"]])
            fit=fit.drop(columns=[c for c in N.columns if c in fit],errors="ignore").join(N,on="txId")
            val=val.drop(columns=[c for c in N.columns if c in val],errors="ignore").join(N,on="txId")
            feats=list(FEATURE_COLS)+STRUCT+list(N.columns)
        F=feats
    return fit,val,F

def preds(fit,val,F,kind,decay=None,end=None):
    w=None if not decay else decay**(end-fit.time_step.values)
    out=[]
    for sd in SEEDS:
        if kind=="lgb": out.append(fitpred(fit[F],fit.label,val[F],seed=sd,w=w))
        elif kind=="et": out.append(ExtraTreesClassifier(500,max_features=0.3,min_samples_leaf=2,class_weight="balanced",n_jobs=8,random_state=sd).fit(fit[F].fillna(-1),fit.label,sample_weight=w).predict_proba(val[F].fillna(-1))[:,1])
        elif kind=="rf": out.append(RandomForestClassifier(500,max_features=0.3,class_weight="balanced",n_jobs=8,random_state=sd).fit(fit[F].fillna(-1),fit.label,sample_weight=w).predict_proba(val[F].fillna(-1))[:,1])
        if kind!="lgb": break
    return np.mean(out,0)

cache={}
def go(name, nb=False, iters=1, hop2=False, kinds=("lgb",), decay=None, wts=None):
    res=[]
    for end in HARD+EASY:
        fit=L[L.time_step<=end]; val=L[L.time_step>end]
        key=(end,nb,iters,hop2)
        if key not in cache: cache[key]=build(fit,val,nb,iters,hop2)
        f,v,F=cache[key]
        ps=[rankdata(preds(f,v,F,k,decay,end)) for k in kinds]
        w=wts or [1]*len(ps)
        res.append(roc_auc_score(v.label,np.average(ps,axis=0,weights=w)))
    h=np.mean(res[:2]); ez=np.mean(res[2:])
    print(f"{name:38s} easy {ez:.4f} hard {h:.4f} | "+" ".join(f"{a:.4f}" for a in res),flush=True)

go("lgb struct")
go("lgb struct+nb", nb=True)
go("lgb struct+nb decay.95", nb=True, decay=0.95)
go("lgb struct+nb x2 iter", nb=True, iters=2)
go("lgb struct+nb hop2", nb=True, hop2=True)
go("et struct+nb", nb=True, kinds=("et",))
go("rf struct+nb", nb=True, kinds=("rf",))
go("lgb+et struct+nb", nb=True, kinds=("lgb","et"))
go("lgb+rf struct+nb", nb=True, kinds=("lgb","rf"))
go("lgb+et+rf struct+nb", nb=True, kinds=("lgb","et","rf"))
