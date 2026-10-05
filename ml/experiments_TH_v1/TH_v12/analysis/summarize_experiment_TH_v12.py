"""Generate readable V11/V12 comparison from verified, identically pooled metrics."""
from pathlib import Path
import os,sys,json
ROOT=Path(__file__).resolve().parents[1]
if os.environ.get('DFMBA_RUNTIME'):sys.path.insert(0,os.environ['DFMBA_RUNTIME'])
import numpy as np
import pandas as pd
OUT=ROOT/'analysis/output_TH_v12'
OLD=ROOT/'reference_v11_TH_v12'
MODELS=['Linear','Random Forest','Extra Trees','XGBoost']
TASKS=['Y1 Growth','Y2 Direction','Y3 Surge','Y4 Drop']
def run():
    current=pd.read_csv(OUT/'comparison_TH_v12.csv')
    reference=pd.read_csv(OLD/'comparison_v11_reference_TH_v12.csv')
    z=current.merge(reference,on=['panel','experiment','model','task','period'],suffixes=('_v12','_v11'),validate='one_to_one')
    assert len(z)==len(current)==len(reference)
    z=z[(z.experiment=='E_M')&(z.period=='recent_2021_2025')]
    text=['모멘텀 축소 실험 분석 TH v12','2026-10-04','',
        '1. 요약과 비교 조건',
        'V11의 시도 5개·서울 6개 모멘텀을 최근 1개월·6개월 변화율 두 개로 줄였습니다.',
        '동일한 2021~2025년 예측 관측을 통합하여 비교합니다. 경제변수와 정답·기간·가중치·모형 설정은 유지하였습니다.',
        '시험기간을 여러 실험에서 이미 확인하였으므로 본 비교는 사후 설계 비교입니다. 새로운 미관측 최종 검증은 아닙니다.',
        'Y1 MAE는 %p이며 낮을수록 좋습니다. Y2는 상승·하락 Macro F1, Y3·Y4는 해당 사건 F1이며 높을수록 좋습니다.',
        '아래 각 칸은 V11 → V12 순서입니다.','']
    compact=[]
    for panel,title in [('provinces','전국 17개 시도'),('seoul','서울 25개 구')]:
        text+=['2.1 '+title if panel=='provinces' else '2.2 '+title,'모형 | Y1 MAE(%p) | Y2 Macro F1 | Y3 급등 F1 | Y4 급락 F1']
        for model in MODELS:
            h=z[(z.panel==panel)&(z.model==model)].set_index('task')
            vals=[]
            for task,col in zip(TASKS,['MAE','Macro_F1','Event_F1','Event_F1']):
                row=h.loc[task];old=row[col+'_v11'];new=row[col+'_v12'];change=new-old
                vals.append(f'{old:.3f} → {new:.3f}')
                compact.append(dict(panel=panel,model=model,task=task,metric=col,v11=old,v12=new,
                    change=change,improved=bool(change<0 if col=='MAE' else change>0)))
            text.append(model+' | '+' | '.join(vals))
        text+=['']
    compact=pd.DataFrame(compact)
    compact.to_csv(OUT/'핵심비교표_TH_v12.csv',index=False,encoding='utf-8-sig')
    text+=['3. 목표별 변화 해석']
    text+=['서울 Logistic의 방향·급락 F1은 개선되었으나 서울 Y1 MAE는 네 모형 모두 커졌고 시도 급등 F1은 네 모형 모두 낮아졌습니다.',
        '두 개 축소안이 전반적으로 더 낫다는 근거는 없습니다. V11을 대체하기보다는 V12를 복잡도 축소에 대한 비교·민감도 실험으로 보존하는 것이 타당합니다.']
    for panel,title in [('provinces','시도'),('seoul','서울')]:
        for task in TASKS:
            h=compact[(compact.panel==panel)&(compact.task==task)]
            best=h.sort_values('v12',ascending=task==TASKS[0]).iloc[0]
            text.append(f'{title} {task}: 네 모형 중 개선 {int(h.improved.sum())}개. V12 최고 {best.model}, {best.metric} {best.v12:.3f} (동일 모형 V11 {best.v11:.3f}).')
    text+=['',
        '복잡도 축소의 효과는 목표·지역·모형별로 다르게 판단해야 합니다. 목표마다 시험 최고 모형을 골라 하나의 개선된 운영모형으로 해석하지 않습니다.',
        '모멘텀 두 개를 남겨도 경제변수의 개수가 줄어든 것은 아닙니다. 성능 변화만으로 경제변수의 인과 기여가 커졌다고 주장할 수 없습니다.',
        'Random Forest/Extra Trees의 max_features 비율 및 XGBoost의 colsample 비율은 동일하지만, 전체 후보 수가 감소하면 실제 추출되는 특징 수와 난수 경로가 달라질 수 있습니다.',
        '동일 입력·난수·설정으로 반복 가능하지만, 점수 차이의 통계적 유의성이나 미래 안정성이 증명된 것은 아닙니다.','',
        '4. Y2 하락 포착의 정밀도·재현율 (V11 → V12)',
        '지역 | 모형 | 하락 정밀도 | 하락 재현율 | 하락 F1']
    for row in z[z.task==TASKS[1]].itertuples():
        pairs=[f'{getattr(row,col+"_v11"):.3f} → {getattr(row,col+"_v12"):.3f}' for col in ['DownPrecision','DownRecall','Down_F1']]
        text.append(row.panel+' | '+row.model+' | '+' | '.join(pairs))
    a=pd.read_csv(OUT/'annual_metrics_TH_v12.csv')
    b=pd.read_csv(OLD/'annual_v11_reference_TH_v12.csv')
    keys=['panel','experiment','model','task','year']
    annual=a.merge(b,on=keys,suffixes=('_v12','_v11'),validate='one_to_one')
    assert len(annual)==len(a)==len(b)
    for col in ['Accuracy','Macro_F1','Event_F1','MAE','R2']:
        annual[col+'_change']=annual[col+'_v12']-annual[col+'_v11']
    annual.to_csv(OUT/'v11_vs_v12_annual_TH_v12.csv',index=False,encoding='utf-8-sig')
    text+=['','5. 서울 2023년 방향 전환 점검','모형 | 정확도 V11 → V12 | Macro F1 V11 → V12']
    for row in annual[(annual.panel=='seoul')&(annual.year==2023)&(annual.task==TASKS[1])&(annual.experiment=='E_M')].itertuples():
        text.append(f'{row.model} | {row.Accuracy_v11:.3f} → {row.Accuracy_v12:.3f} | {row.Macro_F1_v11:.3f} → {row.Macro_F1_v12:.3f}')
    text+=['','6. Y1 예측 부호로 얻은 방향과 별도 Y2 분류 비교','지역 | 모형 | Y1 부호 Macro F1 | 별도 Y2 Macro F1']
    c=pd.read_csv(OUT/'comparison_TH_v12.csv')
    recent=c[c.period=='recent_2021_2025']
    for panel in ['provinces','seoul']:
        for model in MODELS:
            sign=recent[(recent.panel==panel)&(recent.model==model)&(recent.task==TASKS[1])&(recent.experiment=='Regression_sign')].iloc[0]
            separate=recent[(recent.panel==panel)&(recent.model==model)&(recent.task==TASKS[1])&(recent.experiment=='E_M')].iloc[0]
            text.append(f'{panel} | {model} | {sign.Macro_F1:.3f} | {separate.Macro_F1:.3f}')
    text+=['','7. 검증 범위와 남은 한계']
    review=json.loads((OUT/'review_manifest_TH_v12.json').read_text(encoding='utf-8'))
    text.append(f'학습 1,024회. 구현 검증 {review["checks"]}건 중 실패 {review["failed"]}건. 지표 산식 {review["metric_audits"]}건 중 실패 {review["metric_audit_failed"]}건.')
    text.append('V11 대비 입력 해시·평가 키·모든 실제 정답·경제변수 값 및 선택·학습구간·모형 설정·사건 경계·비교모형 일치를 확인하였습니다.')
    text+=['모멘텀만 변경하였으므로 기존 공표 빈티지·연결자료·지역 독립성·계절성 한계는 남아 있습니다.',
        '2018~2025년 전체 기간과 연도별 결과도 CSV로 제공합니다. 주 비교 기간을 결과에 맞춰 변경하지 않았습니다.',
        '표의 F1 차이는 기술적 비교입니다. 이 실험을 근거로 경계값·가중치·하이퍼파라미터를 추가로 조정하지 않았습니다.',
        '원자료·V11·팀 Git 저장소는 보존하였으며 .env나 API 키를 읽거나 공유하지 않습니다.']
    (ROOT/'실험결과_모멘텀축소_TH_v12.txt').write_text('\n'.join(text),encoding='utf-8-sig')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Malgun Gothic','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for panel,title in [('provinces','17 Provinces'),('seoul','Seoul 25 Districts')]:
        fig,axes=plt.subplots(2,2,figsize=(11,7.2))
        fig.subplots_adjust(left=.09,right=.98,bottom=.10,top=.84,wspace=.28,hspace=.55)
        fig.suptitle(title+' | Momentum Reduction',x=.09,ha='left',fontsize=15,fontweight='bold',color='#163A5F')
        fig.text(.09,.89,'2021-2025 matched observations | V11: 5/6 features | V12: 1m + 6m only',fontsize=10,color='#64748B')
        for ax,task,label in zip(axes.flat,TASKS,['Y1 MAE (pp): lower is better','Y2 Direction: Macro F1','Y3 Surge: Event F1','Y4 Drop: Event F1']):
            h=compact[(compact.panel==panel)&(compact.task==task)].set_index('model').loc[MODELS]
            x=np.arange(4);width=.34
            ax.bar(x-width/2,h.v11,width,label='V11',color='#94A3B8')
            ax.bar(x+width/2,h.v12,width,label='V12',color='#163A5F')
            ax.set_xticks(x,['Linear','RF','ET','XGB']);ax.set_title(label,loc='left',fontsize=10,pad=10)
            top=max(h.v11.max(),h.v12.max())*1.3 if task==TASKS[0] else 1
            ax.set_ylim(0,top);ax.grid(axis='y',color='#E5E7EB',linewidth=.7);ax.set_axisbelow(True)
            for offset,col in [(-width/2,'v11'),(width/2,'v12')]:
                for i,v in enumerate(h[col]):ax.text(i+offset,v+.02*top,f'{v:.3f}',ha='center',fontsize=8)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,frameon=False,fontsize=10,ncol=2,loc='upper right',bbox_to_anchor=(.98,.945))
        fig.savefig(OUT/'charts_TH_v12'/f'{panel}_v11_vs_v12_TH_v12.png',dpi=180)
        plt.close(fig)
    print(compact.to_string(index=False))
if __name__=='__main__':run()
