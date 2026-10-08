"""V18 평가: 클래스 가중치 처리 세 방식 비교 (Y3·Y4, 전체·모멘텀 모형).

A  가중치 없음        : TH_v18 재학습 확률 그대로
B  가중치 + 공식 되돌리기: TH_v16 확률 p_w 를 odds 로 바꿔 그 분기 가중치 w = (1−r)/r 로 나눔 (r = 학습 사건비율).
                         로지스틱은 정확한 사전확률 보정, 트리·XGB는 근사.
C  가중치 + Platt 보정   : TH_v17 보정 확률
(참고) 가중치 원래 확률 : TH_v16 확률 그대로

평가 1 (확률·경보): 2021~2025 중 C 까지 모두 값이 있는 같은 지역·월만.
  Brier, BSS = 1 − Brier/Brier(학습 사건비율 예측), 보정 곡선, 30%·50% 경보의 적중률·포착률·F1(vs 항상 사건).
평가 2 (순위 실력): 2021~2025 전체에서 PR-AUC, A vs 가중치 원래 확률 (B·C 는 순위를 바꾸지 않으므로 원래 확률과 같다).
신뢰구간: 월 단위 6개월 블록 부트스트랩 1,000회.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
sys.path.insert(0, str(EXP / 'TH_v16' / 'analysis'))
from eval_TH_v16 import W_of, ci, verdict  # noqa: E402

OUT = HERE / 'eval_TH_v18'; OUT.mkdir(exist_ok=True)
SEED = 42
BINS = [0, .1, .3, .5, .7, 1.0001]
KEY = ['panel', 'task', 'name', 'region', 'month']


def assemble():
    v16 = pd.concat([pd.read_csv(EXP / f'TH_v16/analysis/output_TH_v16_{t}/predictions_TH_v16.csv', low_memory=False) for t in ['full', 'momentum']])
    v16 = v16[v16.task.isin(['Y3 Surge', 'Y4 Drop']) & (v16.experiment == 'Model')].copy()
    v16['name'] = np.where(v16.features == 'full', '전체 ', '모멘텀 ') + v16.model
    a = pd.concat([pd.read_csv(HERE / f'output_TH_v18_{t}/predictions_TH_v18.csv') for t in ['full', 'momentum']])
    a['name'] = np.where(a.features == 'full', '전체 ', '모멘텀 ') + a.model
    d = v16[KEY + ['origin', 'year', 'actual', 'auc_score_1']].rename(columns={'auc_score_1': 'p_weighted'}).merge(
        a[KEY + ['p_noweight', 'train_rate', 'actual']], on=KEY, suffixes=('', '_a'), validate='one_to_one')
    assert (d.actual == d.actual_a).all()
    pw = d.p_weighted.clip(1e-6, 1 - 1e-6); w = (1 - d.train_rate) / d.train_rate
    odds = pw / (1 - pw) / w; d['p_B'] = odds / (1 + odds)
    c = pd.read_csv(EXP / 'TH_v17/analysis/eval_TH_v17/보정확률_TH_v17.csv')[KEY + ['p_cal']]
    d = d.merge(c, on=KEY, how='left')
    return d.rename(columns={'p_noweight': 'p_A', 'p_cal': 'p_C'})


METHODS = {'A 가중치 없음': 'p_A', 'B 가중치+공식': 'p_B', 'C 가중치+Platt': 'p_C', '(참고) 가중치 원래확률': 'p_weighted'}


def f1(M):
    with np.errstate(invalid='ignore', divide='ignore'): return np.nan_to_num(2 * M[..., 0] / (2 * M[..., 0] + M[..., 1] + M[..., 2]))


def evaluate_prob(d):
    rng = np.random.default_rng(SEED); rows, curves = [], []
    d = d[(d.year >= 2021) & (d.year <= 2025)]
    for (panel, task), g in d.groupby(['panel', 'task']):
        ok = g.groupby(['region', 'month']).p_C.apply(lambda s: s.notna().all()); keys = set(ok[ok].index)
        g = g[[k in keys for k in zip(g.region, g.month)]]
        if g.empty: continue
        months = np.sort(g.month.unique()); W = W_of(len(months), rng)
        pm = lambda df, col: df.groupby('month')[col].sum().reindex(months).fillna(0).to_numpy(float)
        first = g[g.name == g.name.iloc[0]]
        Ma = np.c_[pm(first, 'actual'), pm(first.assign(z=1 - first.actual), 'z'), np.zeros(len(months))]; ba = f1(W @ Ma)
        for name, h in g.groupby('name'):
            S0 = pm(h.assign(z=(h.train_rate - h.actual) ** 2), 'z'); N = pm(h.assign(z=1), 'z')
            for meth, col in METHODS.items():
                S = pm(h.assign(z=(h[col] - h.actual) ** 2), 'z'); bss = 1 - (W @ S) / (W @ S0); lo, hi = ci(bss)
                row = dict(panel=panel, task=task, model=name, 방식=meth, 평가월수=len(months), 첫달=int(months.min()),
                           사건비율=float(h.actual.mean()), 평균확률=float(h[col].mean()), Brier=S.sum() / N.sum(),
                           BSS=1 - S.sum() / S0.sum(), BSS_lo=lo, BSS_hi=hi, BSS_판정=verdict(lo, hi), F1_항상사건=float(f1(Ma.sum(0))))
                for a in (0.3, 0.5):
                    fl = (h[col] >= a).astype(int); t = f'{int(a * 100)}%'
                    M = np.c_[pm(h.assign(z=fl * h.actual), 'z'), pm(h.assign(z=fl * (1 - h.actual)), 'z'), pm(h.assign(z=(1 - fl) * h.actual), 'z')]
                    tp, fp, fn = M.sum(0); q = h.assign(f=fl).groupby('origin').f.mean(); flo, fhi = ci(f1(W @ M) - ba)
                    row.update({f'경보빈도_{t}': float(fl.mean()), f'분기최대_{t}': float(q.max()),
                                f'적중률_{t}': tp / (tp + fp) if tp + fp else np.nan, f'포착률_{t}': tp / (tp + fn) if tp + fn else np.nan,
                                f'F1_{t}': float(f1(M.sum(0))), f'F1판정_{t}': verdict(flo, fhi)})
                rows.append(row)
                cb = h.assign(구간=pd.cut(h[col], BINS, right=False)).groupby('구간', observed=True).agg(
                    건수=('actual', 'size'), 평균확률=(col, 'mean'), 실제비율=('actual', 'mean')).reset_index()
                curves.append(cb.assign(panel=panel, task=task, model=name, 방식=meth))
    return pd.DataFrame(rows), pd.concat(curves)


def evaluate_rank(d):
    rng = np.random.default_rng(SEED); rows = []
    d = d[(d.year >= 2021) & (d.year <= 2025)]
    for (panel, task), g in d.groupby(['panel', 'task']):
        months = np.sort(g.month.unique()); midx = {m: i for i, m in enumerate(months)}; W = W_of(len(months), rng)
        for name, h in g.groupby('name'):
            y = h.actual.to_numpy(); mi = h.month.map(midx).to_numpy(); res = {}
            for col in ['p_A', 'p_weighted']:
                s = h[col].to_numpy()
                res[col] = (average_precision_score(y, s),
                            np.array([average_precision_score(y, s, sample_weight=W[b, mi]) if W[b, mi][y == 1].sum() > 0 else np.nan for b in range(len(W))]))
            lo, hi = ci(res['p_A'][1] - res['p_weighted'][1])
            rows.append(dict(panel=panel, task=task, model=name, 사건비율=float(y.mean()), PR_AUC_가중치없음=res['p_A'][0],
                             PR_AUC_가중치있음=res['p_weighted'][0], 차이=res['p_A'][0] - res['p_weighted'][0], 판정=verdict(lo, hi)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    d = assemble()
    d.to_csv(OUT / '확률_세방식_TH_v18.csv', index=False, encoding='utf-8-sig')
    r, c = evaluate_prob(d)
    r.to_csv(OUT / 'Y3Y4_확률경보_세방식_TH_v18.csv', index=False, encoding='utf-8-sig')
    c.to_csv(OUT / 'Y3Y4_보정곡선_세방식_TH_v18.csv', index=False, encoding='utf-8-sig')
    evaluate_rank(d).to_csv(OUT / 'Y3Y4_PRAUC_가중치유무_TH_v18.csv', index=False, encoding='utf-8-sig')
    print('done')
