"""Independent target/causality/replay checks, transparent binary metrics and charts."""
from pathlib import Path
import os,sys,json,warnings
ROOT=Path(__file__).resolve().parents[1]
if os.environ.get('DFMBA_RUNTIME'):sys.path.insert(0,os.environ['DFMBA_RUNTIME'])
sys.path[:0]=[str(ROOT/'ml')]
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score,average_precision_score
from train_simple_TH_v12 import OUT,TASKS,FAMILIES,engineer,history,matrix,sha,select,labels

def divide(a,b):return float(a/b) if b else np.nan
def binary_metrics(g):
    y=g.actual.to_numpy();p=g.prediction.to_numpy()
    tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum())
    fn=int(((y==1)&(p!=1)).sum());tn=int(((y==0)&(p==0)).sum())
    down_fn=int(((y==0)&(p!=0)).sum());down_fp=int(((y==1)&(p==0)).sum())
    fup=divide(2*tp,2*tp+fp+fn);fdown=divide(2*tn,2*tn+down_fp+down_fn)
    rup=divide(tp,int((y==1).sum()));rdown=divide(tn,int((y==0).sum()))
    result=dict(n=len(g),Accuracy=float(np.mean(y==p)),BalancedAccuracy=float(np.nanmean([rup,rdown])),
        Macro_F1=float(np.nanmean([fup,fdown])),Macro_F1_fixed2=float(np.nan_to_num([fup,fdown]).mean()),
        TP=tp,FP=fp,FN=fn,TN=tn,ActualEvents=int((y==1).sum()),ActualDown=int((y==0).sum()),
        PredictedEvents=int((p==1).sum()),Abstentions=int((p==-1).sum()),
        Precision=divide(tp,tp+fp),Recall=rup,Event_F1=fup,
        DownPrecision=divide(tn,tn+down_fp),DownRecall=rdown,Down_F1=fdown,
        ActualClasses=len(np.unique(y)),UpShare=float(np.mean(y==1)),
        AUC_within_month=np.nan,valid_auc_months=0,valid_month_pairs=0,AUC_pooled_secondary=np.nan,PR_AUC_pooled_secondary=np.nan)
    if 'auc_score_1' in g and g.auc_score_1.notna().all():
        if len(np.unique(y))==2:
            result['AUC_pooled_secondary']=roc_auc_score(y,g.auc_score_1)
            result['PR_AUC_pooled_secondary']=average_precision_score(y,g.auc_score_1)
        pairs=0;weighted=0;months=0
        for _,h in g.groupby('month'):
            n1=int(h.actual.sum());n0=len(h)-n1
            if n1 and n0:
                weight=n1*n0;pairs+=weight;months+=1
                weighted+=roc_auc_score(h.actual,h.auc_score_1)*weight
        result.update(AUC_within_month=divide(weighted,pairs),valid_auc_months=months,valid_month_pairs=pairs)
    return result

def metrics(g):
    if g.task.iloc[0]==TASKS[0]:
        valid=g[['actual','prediction']].notna().all(axis=1);z=g[valid];y=z.actual.to_numpy();p=z.prediction.to_numpy()
        den=np.sum((y-y.mean())**2);error=y-p
        return dict(n=len(z),MissingPredictions=int((~valid).sum()),MAE=float(np.mean(abs(error))),
            RMSE=float(np.sqrt(np.mean(error**2))),R2=1-float(np.sum(error**2))/den if den else np.nan)
    return binary_metrics(g)

def verify(pred,folds):
    checks=[]
    def check(name,ok):checks.append(dict(check=name,passed=bool(ok)))
    keys=['panel','experiment','model','task','region','month']
    check('Unique keyed predictions',not pred.duplicated(keys).any())
    check('Fit count 1024',len(folds)==1024)
    check('Saved boundary models 64',len(list((OUT/'models_TH_v12').glob('*.joblib')))==64)
    for r in folds.itertuples():
        check(f'{r.panel} {r.origin} {r.model} {r.task} known training labels',r.train_label_available<=r.origin and r.train_target_end<r.test_start)
    for panel in ['provinces','seoul']:
        raw=pd.read_csv(ROOT/'input_TH_v12'/f'{panel}_aligned_candidates_TH_v12.csv',float_precision='round_trip',low_memory=False).sort_values(['region','month']).reset_index(drop=True)
        d=pd.read_csv(OUT/f'{panel}_model_ready_TH_v12.csv',float_precision='round_trip',low_memory=False)
        # Independent keyed direct target paths, not the shift-based implementation.
        ycol=next(c for c in raw if c.startswith('Y_V001'))
        observed={(region,int(month)):value if flag==1 else np.nan for region,month,value,flag in raw[['region','month',ycol,'Y_평가사용가능']].itertuples(index=False,name=None)}
        from features_TH_v12 import addmonths
        for region,h in d.groupby('region'):
            future=[]
            for m in h.month:
                base=observed.get((region,addmonths(m,-1)),np.nan)
                future.append([100*(observed.get((region,addmonths(m,k)),np.nan)/base-1) for k in range(6)])
            future=np.asarray(future)
            for name,value in [('g',future[:,-1]),('U',future.max(1)),('D',future.min(1))]:
                check(panel+' '+region+' independent '+name,np.allclose(h[name],value,atol=1e-10,rtol=1e-10,equal_nan=True))
        for origin in [201801,202101,202301,202510]:
            cut=raw[raw.month<=origin].copy();short,ec,mo=engineer(cut,panel)
            shared=short.merge(d,on=['region','month'],suffixes=('_short','_full'))
            for c in ec+mo:
                check(f'{panel} {origin} causal prefix {c}',np.allclose(shared[c+'_short'],shared[c+'_full'],equal_nan=True,atol=1e-10,rtol=1e-10))
        subset=pred[(pred.panel==panel)&(pred.experiment=='E_M')]
        for (year,task),g in subset.groupby(['year','task']):
            actual=g.drop_duplicates(['region','month']).merge(d[['region','month','g','U','D']],on=['region','month'],validate='one_to_one')
            train=history(d,year*100+1);th=dict(up=max(0,train.U.quantile(.85)),dn=min(0,train.D.quantile(1-.85)))
            check(f'{panel} {year} {task} independent class labels',np.allclose(actual.actual,labels(actual,th,task),atol=1e-12))
        for file in sorted((OUT/'models_TH_v12').glob(panel+'*.joblib')):
            saved=joblib.load(file);meta=saved['metadata'];g=subset[(subset.origin==meta['origin'])&(subset.model==meta['model'])&(subset.task==meta['task'])]
            te=g[['region','month']].merge(d,on=['region','month'],validate='one_to_one')
            estimate=saved['estimator'].predict(matrix(te,saved['features'],saved['regions']))
            if saved['classes'] is not None:estimate=saved['classes'][estimate.astype(int)]
            check(file.name+' replay',np.allclose(estimate,g.prediction,atol=1e-10,rtol=1e-10))
        feature_notes=pd.read_csv(OUT/'features_TH_v12.csv');feature_notes=feature_notes[feature_notes.panel==panel]
        for origin,h in feature_notes.groupby('origin'):
            candidates=h.feature.tolist();chosen,notes=select(history(d,origin),candidates)
            check(f'{panel} {origin} train-only selection',set(chosen)==set(h[h.selected].feature))
    q=pd.DataFrame(checks);q.to_csv(OUT/'verification_TH_v12.csv',index=False,encoding='utf-8-sig')
    summary=dict(checks=len(q),failed=int((~q.passed).sum()))
    (OUT/'verification_summary_TH_v12.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    assert summary['failed']==0,q[~q.passed].to_string()
    return summary

def comparisons(pred):
    annual=[];summary=[]
    for key,g in pred.groupby(['panel','experiment','model','task','year']):
        annual.append(dict(zip(['panel','experiment','model','task','year'],key))|metrics(g))
    for period,start in [('all_2018_2025',2018),('recent_2021_2025',2021)]:
        for key,g in pred[pred.year>=start].groupby(['panel','experiment','model','task']):
            summary.append(dict(zip(['panel','experiment','model','task'],key))|dict(period=period)|metrics(g))
    a=pd.DataFrame(annual);c=pd.DataFrame(summary)
    a.to_csv(OUT/'annual_metrics_TH_v12.csv',index=False,encoding='utf-8-sig')
    c.to_csv(OUT/'comparison_TH_v12.csv',index=False,encoding='utf-8-sig')
    # Audit pooled metric formulas independently using a confusion matrix / direct residuals.
    audit=[]
    for r in c.itertuples():
        z=pred[(pred.panel==r.panel)&(pred.experiment==r.experiment)&(pred.model==r.model)&(pred.task==r.task)&(pred.year>=(2021 if r.period.startswith('recent') else 2018))]
        if r.task==TASKS[0]:
            z=z.dropna(subset=['actual','prediction']);error=z.actual-z.prediction
            value=float(error.abs().sum()/len(z));ok=np.isclose(value,r.MAE)
            ok=ok and np.isclose(np.sqrt(float((error**2).sum()/len(z))),r.RMSE)
            den=float(((z.actual-z.actual.mean())**2).sum());expected=1-float((error**2).sum())/den if den else np.nan
            ok=ok and (np.isclose(expected,r.R2) if np.isfinite(expected) else pd.isna(r.R2))
        else:
            cm=pd.crosstab(z.actual,z.prediction).reindex(index=[0,1],columns=[-1,0,1],fill_value=0).to_numpy()
            diagonal=cm[0,1]+cm[1,2];ok=np.isclose(diagonal/cm.sum(),r.Accuracy)
            fps=cm[0,2];fns=cm[1,0]+cm[1,1];tp=cm[1,2]
            calculated=divide(2*tp,2*tp+fps+fns)
            ok=ok and (np.isclose(calculated,r.Event_F1) if np.isfinite(calculated) else pd.isna(r.Event_F1))
            tn=cm[0,1];down_fp=cm[1,1];down_fn=cm[0,0]+cm[0,2]
            fd=divide(2*tn,2*tn+down_fp+down_fn)
            macro=float(np.nanmean([calculated,fd]));ok=ok and np.isclose(macro,r.Macro_F1)
            balanced=float(np.nanmean([divide(tp,int(cm[1].sum())),divide(tn,int(cm[0].sum()))]))
            ok=ok and np.isclose(balanced,r.BalancedAccuracy)
        audit.append(dict(panel=r.panel,experiment=r.experiment,model=r.model,task=r.task,period=r.period,passed=bool(ok)))
    q=pd.DataFrame(audit);q.to_csv(OUT/'metric_audit_TH_v12.csv',index=False,encoding='utf-8-sig');assert q.passed.all()
    return a,c,len(q)

def charts(c,a):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Malgun Gothic','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#475569'})
    folder=OUT/'charts_TH_v12';folder.mkdir(exist_ok=True)
    names=['Ridge / Logistic','Random Forest','Extra Trees','XGBoost'];colors=['#64748B','#2A9D8F','#D97706','#163A5F']
    recent=c[(c.period=='recent_2021_2025')&(c.experiment=='E_M')]
    for panel,title in [('provinces','17 Provinces'),('seoul','Seoul 25 Districts')]:
        fig,axes=plt.subplots(2,2,figsize=(11,7.2));fig.subplots_adjust(left=.09,right=.98,bottom=.10,top=.86,wspace=.28,hspace=.55)
        fig.suptitle(title+' | V12 Economics + Rent Momentum',x=.09,ha='left',fontsize=15,fontweight='bold',color='#163A5F')
        fig.text(.09,.905,'2021-2025 out-of-sample | Binary direction | Quarterly expanding refits',color='#64748B',fontsize=10)
        for ax,task,col,subtitle in zip(axes.flat,TASKS,['MAE','Macro_F1','Event_F1','Event_F1'],['Y1 Growth: MAE (pp), lower is better','Y2 Direction: Pooled Macro F1','Y3 Surge: Event F1','Y4 Drop: Event F1']):
            z=recent[(recent.panel==panel)&(recent.task==task)].set_index('model');values=[z.loc[m,col] for m in FAMILIES]
            ax.bar(range(4),values,color=colors,width=.6);ax.set_xticks(range(4),['Linear','RF','ET','XGB']);ax.set_title(subtitle,loc='left',fontsize=10,pad=12)
            top=max(values)*1.25 if col=='MAE' else 1.;ax.set_ylim(0,top);ax.yaxis.grid(True,color='#E5E7EB',linewidth=.7);ax.set_axisbelow(True)
            for i,v in enumerate(values):ax.text(i,v+top*.025,f'{v:.3f}',ha='center',fontsize=10)
        fig.savefig(folder/f'{panel}_overview_TH_v12.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10.8,4.3));fig.subplots_adjust(left=.08,right=.98,bottom=.16,top=.8)
        z=a[(a.panel==panel)&(a.task==TASKS[1])&(a.year>=2021)]
        for model,color in zip(FAMILIES,colors):
            h=z[(z.experiment=='E_M')&(z.model==model)].sort_values('year');ax.plot(h.year,h.Accuracy,label=model,color=color,marker='o',linewidth=2)
        h=z[(z.experiment=='Benchmark')&(z.model=='Always up')].sort_values('year');ax.plot(h.year,h.Accuracy,label='Always up',color='#94A3B8',linestyle='--',linewidth=2)
        ax.set_ylim(0,1.05);ax.set_xticks(range(2021,2026));ax.set_ylabel('Direction accuracy');ax.grid(axis='y',color='#E5E7EB');ax.legend(ncol=3,frameon=False,loc='upper left',bbox_to_anchor=(0,1.20))
        fig.suptitle(title+' | Annual Direction Accuracy',x=.08,ha='left',fontsize=13,fontweight='bold',color='#163A5F')
        fig.savefig(folder/f'{panel}_annual_direction_TH_v12.png',dpi=180);plt.close(fig)

def prior_compare(pred,c):
    path=ROOT.parent/'DFMBA_작업본_TH_v11/analysis/output_TH_v11/predictions_TH_v11.csv'
    if not path.exists():return {'status':'V11 optional reference unavailable'}
    old=pd.read_csv(path,float_precision='round_trip',low_memory=False)
    key=['panel','experiment','model','task','region','month']
    pair=pred.merge(old[key+['actual','prediction']],on=key,suffixes=('_v12','_v11'),validate='one_to_one')
    assert len(pair)==len(pred)==len(old),'Evaluation keys changed'
    assert np.allclose(pair.actual_v12,pair.actual_v11,atol=1e-10),'Ground truth changed'
    bench=pair[pair.experiment=='Benchmark']
    assert np.allclose(bench.prediction_v12,bench.prediction_v11,equal_nan=True),'Benchmark changed'
    oldfold=pd.read_csv(path.parent/'folds_TH_v11.csv')
    newfold=pd.read_csv(OUT/'folds_TH_v12.csv')
    foldkey=['panel','experiment','model','task','origin']
    samecols=['train_start','train_end','train_target_end','train_label_available','test_start','test_end','ntrain','ntest','up','dn','parameters']
    matched=newfold.merge(oldfold,on=foldkey,suffixes=('_v12','_v11'),validate='one_to_one')
    assert len(matched)==len(newfold)==len(oldfold)
    for col in samecols:
        assert matched[col+'_v12'].equals(matched[col+'_v11']),(col,'Design changed')
    assert newfold.momentum_features.le(2).all()
    oldnote=pd.read_csv(path.parent/'features_TH_v11.csv')
    newnote=pd.read_csv(OUT/'features_TH_v12.csv')
    oldecon=oldnote[~oldnote.feature.str.startswith('M10_')].reset_index(drop=True)
    newecon=newnote[~newnote.feature.str.startswith('M10_')].reset_index(drop=True)
    pd.testing.assert_frame_equal(oldecon,newecon)
    for panel in ['provinces','seoul']:
        copied=ROOT/'input_TH_v12'/f'{panel}_aligned_candidates_TH_v12.csv'
        source=ROOT.parent/'DFMBA_작업본_TH_v11/input_TH_v11'/f'{panel}_aligned_candidates_TH_v11.csv'
        assert sha(copied)==sha(source)
        newready=pd.read_csv(OUT/f'{panel}_model_ready_TH_v12.csv',float_precision='round_trip',low_memory=False)
        oldready=pd.read_csv(path.parent/f'{panel}_model_ready_TH_v11.csv',float_precision='round_trip',low_memory=False)
        assert [x for x in newready if x.startswith('M10_')]==['M10_rent_growth_1m','M10_rent_growth_6m']
        pd.testing.assert_frame_equal(newready,oldready[newready.columns])
    oldc=pd.read_csv(path.parent/'comparison_TH_v11.csv')
    metriccols=[x for x in c.columns if x not in ['panel','experiment','model','task','period']]
    compkey=['panel','experiment','model','task','period']
    delta=c.merge(oldc,on=compkey,suffixes=('_v12','_v11'),validate='one_to_one')
    assert len(delta)==len(c)==len(oldc)
    for col in metriccols:
        delta[col+'_change']=delta[col+'_v12']-delta[col+'_v11']
    delta.to_csv(OUT/'v11_vs_v12_metrics_TH_v12.csv',index=False,encoding='utf-8-sig')
    return dict(status='Only momentum reduced; inputs targets economics parameters splits and benchmarks matched V11',
        pairs=len(pair),matched_fits=len(matched),v11_sha=sha(path),momentum_features_range=[int(newfold.momentum_features.min()),int(newfold.momentum_features.max())])

def run():
    warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
    p=pd.read_csv(OUT/'predictions_TH_v12.csv',float_precision='round_trip',low_memory=False)
    f=pd.read_csv(OUT/'folds_TH_v12.csv');q=verify(p,f);a,c,audits=comparisons(p)
    comparison=prior_compare(p,c);charts(c,a)
    result=dict(**q,metric_audits=audits,metric_audit_failed=0,v11_comparison=comparison,
        artifact_hashes={p.name:sha(p) for p in OUT.glob('*.csv')})
    (OUT/'review_manifest_TH_v12.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    recent=c[(c.period=='recent_2021_2025')&(c.experiment=='E_M')]
    print(recent[['panel','model','task','n','MAE','R2','Accuracy','Macro_F1','Precision','Recall','Event_F1','DownPrecision','DownRecall','Down_F1','AUC_within_month']].to_string(index=False))
    print('VERIFIED',q,'metric audits',audits,'V11 comparison',comparison['status'],flush=True)

if __name__=='__main__':run()
