"""Standalone V12: full economics plus momentum, binary direction, quarterly expanding."""
from pathlib import Path
import os,sys,json,hashlib,time,warnings,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(os.environ['DFMBA_RUNTIME'])) if os.environ.get('DFMBA_RUNTIME') else str(ROOT/'runtime_not_bundled'))
sys.path.insert(0,str(ROOT/'ml'))
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,ExtraTreesRegressor,ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBRegressor,XGBClassifier
from features_TH_v12 import engineer,history,select,matrix,addmonths
OUT=ROOT/'analysis/output_TH_v12'
TASKS=['Y1 Growth','Y2 Direction','Y3 Surge','Y4 Drop']
FAMILIES=['Linear','Random Forest','Extra Trees','XGBoost']
YEARS=list(range(2018,2026))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def labels(d,th,task):
    if task==TASKS[0]:return d.g.to_numpy()
    if task==TASKS[1]:
        assert d.g.ne(0).all()
        return (d.g>0).astype(int).to_numpy()
    return (d.U>th['up']).astype(int).to_numpy() if task==TASKS[2] else (d.D<th['dn']).astype(int).to_numpy()

def make_model(family,reg,y):
    if family=='Linear':
        return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),
            Ridge(alpha=100) if reg else LogisticRegression(C=.1,max_iter=2500,random_state=42,class_weight='balanced'))
    if family in ['Random Forest','Extra Trees']:
        cls=(RandomForestRegressor if reg else RandomForestClassifier) if family=='Random Forest' else (ExtraTreesRegressor if reg else ExtraTreesClassifier)
        kw=dict(n_estimators=128,min_samples_leaf=8,max_features=.5,max_depth=10,n_jobs=2,random_state=42)
        if not reg:kw['class_weight']='balanced'
        return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),cls(**kw))
    cls=XGBRegressor if reg else XGBClassifier
    kw=dict(n_estimators=128,max_depth=2,learning_rate=.06,min_child_weight=3,reg_lambda=5,
            subsample=.8,colsample_bytree=.8,n_jobs=2,random_state=42)
    if not reg:kw['scale_pos_weight']=float((y==0).sum()/max(1,(y==1).sum()))
    return cls(**kw)

def fit(family,task,tr,te,cols,regions,th):
    reg=task==TASKS[0];y=labels(tr,th,task);classes=None if reg else np.unique(y)
    est=make_model(family,reg,y)
    if not reg and len(classes)==1:est=DummyClassifier(strategy='prior')
    target=y if reg else np.searchsorted(classes,y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');est.fit(matrix(tr,cols,regions),target)
    pred=est.predict(matrix(te,cols,regions));scores=None
    if not reg:
        pred=classes[pred.astype(int)];scores=np.zeros((len(te),2));prob=est.predict_proba(matrix(te,cols,regions))
        for j,c in enumerate(est.classes_):scores[:,int(classes[int(c)])]=prob[:,j]
    return est,pred,scores,classes,[str(w.message) for w in caught]

def rows(te,meta,actual,pred,scores=None):
    r=te[['region','month','target_end','label_available']].copy()
    for k,v in meta.items():r[k]=v
    r['actual']=actual;r['prediction']=pred
    if scores is not None:
        r['auc_score_0']=scores[:,0];r['auc_score_1']=scores[:,1]
    return r

def run():
    for name in ['parts_TH_v12','models_TH_v12']: (OUT/name).mkdir(parents=True,exist_ok=True)
    source_hashes={p.name:sha(p) for p in sorted((ROOT/'input_TH_v12').glob('*.csv'))}
    code_hashes={p.name:sha(p) for p in sorted((ROOT/'ml').glob('*.py'))}
    spec=dict(version='TH_v12',years=YEARS,models=FAMILIES,case='E_M',seed=42,
        policy='Quarterly expanding; known labels only; unchanged economics plus 1m/6m causal own-rent momentum only; no Seoul pruning',
        momentum_horizons=[1,6],
        y2='g>0 up=1; g<0 down=0; exact g=0 excluded for Y2 train and test only',
        y3_y4='Annual January training-only path upper85/lower15, frozen for year; no acceleration target',
        evaluation='Primary pooled period; annual class support shown; no operational 50% alarms; no test-based tuning',
        input_hashes=source_hashes,code_hashes=code_hashes)
    specpath=OUT/'prespecified_design_TH_v12.json'
    if specpath.exists():assert json.loads(specpath.read_text(encoding='utf-8'))==spec,'Checkpoint design differs'
    else:specpath.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    began=time.time();warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
    for panel in ['provinces','seoul']:
        raw=pd.read_csv(ROOT/'input_TH_v12'/f'{panel}_aligned_candidates_TH_v12.csv',float_precision='round_trip',low_memory=False)
        d,econ,mom=engineer(raw,panel);d=d[d[['g','U','D']].notna().all(axis=1)].reset_index(drop=True)
        d.to_csv(OUT/f'{panel}_model_ready_TH_v12.csv',index=False,encoding='utf-8-sig')
        for year in YEARS:
            part=OUT/'parts_TH_v12'/f'{panel}_{year}_TH_v12.joblib'
            if part.exists():print(panel,year,'RESUME',flush=True);continue
            annual=history(d,year*100+1)
            # Preserve V10's exact q / (1-q) convention at floating-point boundaries.
            th=dict(up=float(max(0,annual.U.quantile(.85))),dn=float(min(0,annual.D.quantile(1-.85))))
            records=[];folds=[];features=[]
            for origin in [year*100+m for m in [1,4,7,10]]:
                tr0=history(d,origin);te0=d[d.month.between(origin,addmonths(origin,2))].copy()
                assert tr0.label_available.max()<=origin and tr0.target_end.max()<te0.month.min()
                for task in TASKS:
                    tr=tr0[tr0.g.ne(0)].copy() if task==TASKS[1] else tr0
                    te=te0[te0.g.ne(0)].copy() if task==TASKS[1] else te0
                    # Keep the same causal feature screen across targets, including zero-growth origins.
                    ec,en=select(tr0,econ);mc,mn=select(tr0,mom);cols=ec+mc;regions=sorted(tr0.region.unique())
                    if task==TASKS[0]:features.extend(dict(panel=panel,year=year,origin=origin,**n) for n in en+mn)
                    for family in FAMILIES:
                        est,pred,sc,classes,msg=fit(family,task,tr,te,cols,regions,th)
                        meta=dict(panel=panel,year=year,origin=origin,experiment='E_M',model=family,task=task)
                        records.append(rows(te,meta,labels(te,th,task),pred,sc))
                        fn=''
                        if origin in [201801,202510]:
                            fn=f'{panel}_{origin}_{task.split()[0]}_{family.replace(" ","_")}_TH_v12.joblib'
                            joblib.dump(dict(estimator=est,classes=classes,features=cols,regions=regions,thresholds=th,metadata=meta),OUT/'models_TH_v12'/fn,compress=3)
                        last=est.steps[-1][1] if hasattr(est,'steps') else est
                        folds.append(dict(**meta,train_start=int(tr.month.min()),train_end=int(tr.month.max()),
                            train_target_end=int(tr.target_end.max()),train_label_available=int(tr.label_available.max()),
                            test_start=int(te.month.min()),test_end=int(te.month.max()),ntrain=len(tr),ntest=len(te),
                            excluded_zero_train=len(tr0)-len(tr),excluded_zero_test=len(te0)-len(te),
                            nfeatures=len(cols),momentum_features=len(mc),model_file=fn,
                            parameters=json.dumps(last.get_params(),default=str),warnings=json.dumps(msg,ensure_ascii=False),**th))
                        if task==TASKS[0]:
                            valid=te.g.ne(0);sign=np.where(pred[valid]>0,1,np.where(pred[valid]<0,0,-1))
                            records.append(rows(te[valid],dict(**meta,**{} )|dict(experiment='Regression_sign',task=TASKS[1]),(te.loc[valid,'g']>0).astype(int),sign))
                    yt=labels(tr,th,task);actual=labels(te,th,task)
                    if task==TASKS[0]:
                        bench=[('Zero growth',np.zeros(len(te))),('Training median',np.repeat(np.median(yt),len(te))),('Past6 persistence',te.M10_rent_growth_6m.to_numpy())]
                    else:
                        majority=int(np.argmax(np.bincount(yt,minlength=2)))
                        bench=[('Training majority',np.repeat(majority,len(te))),('Always up' if task==TASKS[1] else 'Always event',np.ones(len(te))),('Always down' if task==TASKS[1] else 'Always no event',np.zeros(len(te)))]
                    for name,pred in bench:
                        meta=dict(panel=panel,year=year,origin=origin,experiment='Benchmark',model=name,task=task)
                        score=None if task==TASKS[0] else np.tile([1-np.mean(yt),np.mean(yt)],(len(te),1))
                        records.append(rows(te,meta,actual,pred,score))
                print(panel,origin,'DONE',round(time.time()-began),'seconds',flush=True)
            pd.concat(records,ignore_index=True).to_csv(part.with_suffix('.csv'),index=False,encoding='utf-8-sig')
            joblib.dump(dict(folds=folds,features=features),part,compress=3)
    stores=[joblib.load(p) for p in sorted((OUT/'parts_TH_v12').glob('*.joblib'))]
    for key in ['folds','features']:pd.DataFrame([r for s in stores for r in s[key]]).to_csv(OUT/f'{key}_TH_v12.csv',index=False,encoding='utf-8-sig')
    pred=pd.concat([pd.read_csv(p,float_precision='round_trip',low_memory=False) for p in sorted((OUT/'parts_TH_v12').glob('*.csv'))],ignore_index=True)
    pred.to_csv(OUT/'predictions_TH_v12.csv',index=False,encoding='utf-8-sig')
    assert source_hashes=={p.name:sha(p) for p in sorted((ROOT/'input_TH_v12').glob('*.csv'))}
    manifest=dict(**spec,fit_count=sum(len(s['folds']) for s in stores),prediction_rows=len(pred),seconds=time.time()-began,
        versions={m:__import__(m).__version__ for m in ['numpy','pandas','sklearn','xgboost','joblib']},
        model_hashes={p.name:sha(p) for p in sorted((OUT/'models_TH_v12').glob('*.joblib'))})
    (OUT/'run_manifest_TH_v12.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMPLETE',manifest['fit_count'],'fits',len(pred),'rows',flush=True)

if __name__=='__main__':run()
