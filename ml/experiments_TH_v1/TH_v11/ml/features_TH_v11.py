"""Frozen V8 economic transformations and V10 causal own-rent momentum."""
import numpy as np
import pandas as pd
import re

RATE_TOKENS = ('%', '지수', '동향', 'CSI', '÷', '천세대당', '전국대비', '비중', '순환변동치', '심리', '=100', '고용률', '태도')
FLOW_IDS = ('V021', 'V022', 'V023', 'V035', 'V036')

def addmonths(m, n):
    p = pd.Period(str(int(m)), freq='M') + n
    return p.year * 100 + p.month

def engineer(raw, panel):
    df = raw.sort_values(['region', 'month']).reset_index(drop=True)
    assert not df.duplicated(['region', 'month']).any()
    for _, g in df.groupby('region'):
        assert np.all(np.diff([pd.Period(str(int(m)),freq='M').ordinal for m in g.month])==1)
    ycol = next(c for c in df if c.startswith('Y_V001'))
    rent = df[ycol].where(df['Y_평가사용가능']==1)
    rg = rent.groupby(df.region)
    baseline = rg.shift(1)
    path = pd.concat([100*(rg.shift(1-h)/baseline-1) for h in range(1,7)],axis=1)
    out = df[['region','month']].copy()
    out['g'] = path.iloc[:,-1]
    out['U'] = path.max(axis=1,skipna=False)
    out['D'] = path.min(axis=1,skipna=False)
    economic = []
    pop = [c for c in df if c.startswith('V009_')]
    for c in df:
        if c in ['panel','region','region_code','month',ycol] or c.startswith(('Y_','D_','V006_','V007_','V002_','V026_')):
            continue
        x = pd.to_numeric(df[c],errors='coerce'); xg=x.groupby(df.region)
        v=(c[3:] if c.startswith('서울_') else c)[:4]
        if v=='V012' and pop:
            name=c+'|천명당12개월'
            out[name]=1000*xg.transform(lambda s:s.rolling(12,min_periods=12).sum())/df[pop[0]]
            economic.append(name)
        elif v in FLOW_IDS:
            val=np.log1p(xg.transform(lambda s:s.rolling(12,min_periods=12).sum()))
            names=[c+'|12개월누적log',c+'|12개월누적log_12m차']
            out[names[0]]=val;out[names[1]]=val-val.groupby(df.region).shift(12);economic+=names
        elif v=='V038':
            val=np.log1p(x.clip(lower=0));names=[c+'|log',c+'|log_12m차']
            out[names[0]]=val;out[names[1]]=val-val.groupby(df.region).shift(12);economic+=names
        elif any(t in c for t in RATE_TOKENS):
            names=[c+'|수준',c+'|12m차'];out[names[0]]=x;out[names[1]]=x-xg.shift(12);economic+=names
        else:
            val=np.log(x.where(x>0));name=c+'|12m로그변화'
            out[name]=100*(val-val.groupby(df.region).shift(12));economic.append(name)
    for c in df:
        if c.startswith('D_'):out[c]=df[c];economic.append(c)
    out['D_Wage_10th']=((df.month>=202003)&(df.month<202603)).astype(int)
    out['D_Wage_11th']=(df.month>=202603).astype(int)
    economic+=['D_Wage_10th','D_Wage_11th']
    momentum=[]
    for h in [1,3,6,12]:
        name=f'M10_rent_growth_{h}m';out[name]=100*(rg.shift(1)/rg.shift(h+1)-1);momentum.append(name)
    out['M10_rent_acceleration_3m']=100*(rg.shift(1)/rg.shift(4)-rg.shift(4)/rg.shift(7));momentum.append('M10_rent_acceleration_3m')
    if panel=='seoul':
        val=out['M10_rent_growth_6m'];out['M10_relative_rent_growth_6m']=val-val.groupby(df.month).transform('mean')
        momentum.append('M10_relative_rent_growth_6m')
    out['target_end']=df.month.map(lambda m:addmonths(m,5))
    out['label_available']=df.month.map(lambda m:addmonths(m,6))
    return out,economic,momentum

def history(d,origin):
    return d[(d.month<=addmonths(origin,-6))&(d.label_available<=origin)].copy()

def select(tr,candidates):
    z=tr[candidates].replace([np.inf,-np.inf],np.nan);coverage=z.notna().mean();unique=z.nunique()
    notes=[dict(feature=c,coverage=float(coverage[c]),unique=int(unique[c]),selected=bool(coverage[c]>=.7 and unique[c]>1)) for c in candidates]
    return [n['feature'] for n in notes if n['selected']],notes

def matrix(d,cols,regions):
    x=d[cols].replace([np.inf,-np.inf],np.nan).to_numpy(float)
    return np.c_[x,np.array([[int(r==c) for c in regions] for r in d.region])]
