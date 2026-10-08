"""V19 평가: V18(기준) vs V19a(+3·6개월 변화) vs V19b(+3·6개월 변화, 금리 수준 제외). 같은 알고리즘끼리 비교.

V18 기준 = TH_v16 전체 모형 Y1 + TH_v18 가중치 없음 Y3/Y4 (모두 V15 변수, 가중치 없음).
기간: 주 2021~2025, 보조 2018~2025. 신뢰구간: 월 단위 6개월 블록 부트스트랩 1,000회.
Y1      MAE, 방향 Macro F1. 차이(변형 − V18)의 95% 구간으로 판정.
Y3/Y4   PR-AUC, BSS(기준 = 그 분기 학습 사건비율), 30% 경보 F1. '항상 사건' 대비, V18 대비 판정.
하락기 진단 (시도 Y4, 30% 경보)
  2018~19 하락기(2018.07~2020.06), 2022~23 하락기(2022.01~2023.12): 월별 실제 급락 시도 수와 경보 시도 수,
  구간별 적중률·포착률·헛경보 수, 그리고 '하락기 후반 헛경보'(실제 급락 시도 수가 정점 이후 2곳 이하로 내려온 달들의 헛경보).
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

OUT = HERE / 'eval_TH_v19'; OUT.mkdir(exist_ok=True)
SEED = 42
VERS = ['V18', 'V19a', 'V19b']
PERIODS = {'2021~2025': (2021, 2025), '2018~2025': (2018, 2025)}
KEY = ['panel', 'task', 'model', 'region', 'month']


def load():
    v16 = pd.read_csv(EXP / 'TH_v16/analysis/output_TH_v16_full/predictions_TH_v16.csv', low_memory=False)
    y1 = v16[(v16.task == 'Y1 Growth') & (v16.experiment == 'Model')][KEY + ['year', 'origin', 'actual', 'prediction']].assign(train_rate=np.nan)
    v18 = pd.read_csv(EXP / 'TH_v18/analysis/output_TH_v18_full/predictions_TH_v18.csv').rename(columns={'p_noweight': 'prediction'})
    base = pd.concat([y1, v18[KEY + ['year', 'origin', 'actual', 'prediction', 'train_rate']]]).assign(version='V18')
    parts = [base]
    for tag, v in [('short', 'V19a'), ('short_nolevel', 'V19b')]:
        p = pd.read_csv(HERE / f'output_TH_v19_{tag}/predictions_TH_v19.csv')
        parts.append(p[KEY + ['year', 'origin', 'actual', 'prediction', 'train_rate']].assign(version=v))
    d = pd.concat(parts, ignore_index=True)
    mom = v16[(v16.model == 'Past1 momentum')][KEY + ['year', 'origin', 'actual', 'auc_score_1']].rename(columns={'auc_score_1': 'prediction'})
    return d, mom


def pm(df, col, months): return df.groupby('month')[col].sum().reindex(months).fillna(0).to_numpy(float)


def f1(M):
    with np.errstate(invalid='ignore', divide='ignore'): return np.nan_to_num(2 * M[..., 0] / (2 * M[..., 0] + M[..., 1] + M[..., 2]))


def y1_eval(d):
    rng = np.random.default_rng(SEED); rows = []
    g0 = d[d.task == 'Y1 Growth']
    for (panel, pname), _ in [((p, n), None) for p in g0.panel.unique() for n in PERIODS]:
        y0, y1_ = PERIODS[pname]; g = g0[(g0.panel == panel) & (g0.year >= y0) & (g0.year <= y1_)]
        months = np.sort(g.month.unique()); W = W_of(len(months), rng)
        for model, h in g.groupby('model'):
            res = {}
            for ver, x in h.groupby('version'):
                e = pm(x.assign(z=(x.actual - x.prediction).abs()), 'z', months); n = pm(x.assign(z=1), 'z', months)
                v = x[x.actual != 0]; yy = (v.actual > 0).to_numpy(); pp = (v.prediction > 0).to_numpy()
                D = np.c_[pm(v.assign(z=yy & pp), 'z', months), pm(v.assign(z=~yy & pp), 'z', months), pm(v.assign(z=yy & ~pp), 'z', months), pm(v.assign(z=~yy & ~pp), 'z', months)]

                def mf(M):
                    a, b, c, dd = (M[..., i] for i in range(4))
                    with np.errstate(invalid='ignore', divide='ignore'):
                        return np.nanmean(np.stack([2 * a / (2 * a + b + c), 2 * dd / (2 * dd + b + c)]), axis=0)
                res[ver] = dict(mae=e.sum() / n.sum(), bmae=(W @ e) / (W @ n), dir=float(mf(D.sum(0))), bdir=mf(W @ D))
            for ver in VERS:
                r = res[ver]; row = dict(panel=panel, period=pname, model=model, version=ver, MAE=r['mae'], 방향F1=r['dir'])
                if ver != 'V18':
                    lo, hi = ci(r['bmae'] - res['V18']['bmae']); row.update(MAE_차이=r['mae'] - res['V18']['mae'], MAE_판정=verdict(lo, hi, True))
                    lo, hi = ci(r['bdir'] - res['V18']['bdir']); row.update(방향_차이=r['dir'] - res['V18']['dir'], 방향_판정=verdict(lo, hi))
                rows.append(row)
    return pd.DataFrame(rows)


def ev_eval(d, mom):
    rng = np.random.default_rng(SEED); rows = []
    ev = d[d.task.isin(['Y3 Surge', 'Y4 Drop'])]
    for (panel, task), g0 in ev.groupby(['panel', 'task']):
        for pname, (y0, y1_) in PERIODS.items():
            g = g0[(g0.year >= y0) & (g0.year <= y1_)]
            if g.empty: continue
            months = np.sort(g.month.unique()); midx = {m: i for i, m in enumerate(months)}; W = W_of(len(months), rng)
            first = g[(g.version == 'V18') & (g.model == 'Linear')]
            Ma = np.c_[pm(first, 'actual', months), pm(first.assign(z=1 - first.actual), 'z', months), np.zeros(len(months))]; ba = f1(W @ Ma)
            for model, h in g.groupby('model'):
                res = {}
                for ver, x in h.groupby('version'):
                    y = x.actual.to_numpy(); s = x.prediction.to_numpy(); mi = x.month.map(midx).to_numpy()
                    bpr = np.array([average_precision_score(y, s, sample_weight=W[b, mi]) if W[b, mi][y == 1].sum() > 0 else np.nan for b in range(len(W))])
                    S = pm(x.assign(z=(x.prediction - x.actual) ** 2), 'z', months); S0 = pm(x.assign(z=(x.train_rate - x.actual) ** 2), 'z', months)
                    fl = (x.prediction >= 0.3).astype(int)
                    M = np.c_[pm(x.assign(z=fl * x.actual), 'z', months), pm(x.assign(z=fl * (1 - x.actual)), 'z', months), pm(x.assign(z=(1 - fl) * x.actual), 'z', months)]
                    tp, fp, fn = M.sum(0)
                    res[ver] = dict(pr=average_precision_score(y, s), bpr=bpr, bss=1 - S.sum() / S0.sum(), bbss=1 - (W @ S) / (W @ S0),
                                    f1=float(f1(M.sum(0))), bf1=f1(W @ M), freq=float(fl.mean()), hit=tp / (tp + fp) if tp + fp else np.nan,
                                    cap=tp / (tp + fn) if tp + fn else np.nan)
                for ver in VERS:
                    r = res[ver]; lo, hi = ci(r['bf1'] - ba)
                    row = dict(panel=panel, task=task, period=pname, model=model, version=ver, 사건비율=float(first.actual.mean()),
                               PR_AUC=r['pr'], BSS=r['bss'], 경보빈도_30=r['freq'], 적중률_30=r['hit'], 포착률_30=r['cap'], F1_30=r['f1'],
                               F1_항상사건=float(f1(Ma.sum(0))), F1_vs_항상사건=verdict(lo, hi))
                    if ver != 'V18':
                        b = res['V18']
                        for k, nm in [('pr', 'PR_AUC'), ('bss', 'BSS'), ('f1', 'F1_30')]:
                            lo, hi = ci(r['b' + k] - b['b' + k]); row[f'{nm}_차이'] = r[k] - b[k]; row[f'{nm}_vs_V18'] = verdict(lo, hi)
                    rows.append(row)
    return pd.DataFrame(rows)


def downturn(d):
    """시도 Y4, 30% 경보: 두 하락기의 월별 표와 요약."""
    g = d[(d.panel == 'provinces') & (d.task == 'Y4 Drop')].assign(alert=lambda x: (x.prediction >= 0.3).astype(int))
    monthly = g.groupby(['version', 'model', 'month']).agg(실제급락=('actual', 'sum'), 경보=('alert', 'sum'),
                                                         헛경보=('alert', lambda s: int((s.values * (1 - g.loc[s.index, 'actual'].values)).sum()))).reset_index()
    summ = []
    for name, (a, b) in {'2018~19 하락기 (2018.07~2020.06)': (201807, 202006), '2022~23 하락기 (2022.01~2023.12)': (202201, 202312)}.items():
        w = monthly[(monthly.month >= a) & (monthly.month <= b)]
        act = w[(w.version == 'V18') & (w.model == 'Linear')].set_index('month').실제급락
        peak = act.idxmax(); tail = act[(act.index > peak) & (act <= 2)].index
        for (ver, model), x in w.groupby(['version', 'model']):
            gg = g[(g.version == ver) & (g.model == model) & (g.month >= a) & (g.month <= b)]
            tp = int((gg.alert * gg.actual).sum()); fp = int((gg.alert * (1 - gg.actual)).sum()); fn = int(((1 - gg.alert) * gg.actual).sum())
            summ.append(dict(구간=name, version=ver, model=model, 실제급락=int(gg.actual.sum()), 경보=int(gg.alert.sum()),
                             적중률=tp / (tp + fp) if tp + fp else np.nan, 포착률=tp / (tp + fn) if tp + fn else np.nan, 헛경보=fp,
                             후반헛경보=int(x[x.month.isin(tail)].헛경보.sum()), 후반월수=len(tail)))
    return monthly, pd.DataFrame(summ)


if __name__ == '__main__':
    d, mom = load()
    y1_eval(d).to_csv(OUT / 'Y1_V18vsV19_TH_v19.csv', index=False, encoding='utf-8-sig')
    ev_eval(d, mom).to_csv(OUT / 'Y3Y4_V18vsV19_TH_v19.csv', index=False, encoding='utf-8-sig')
    mth, sm = downturn(d)
    mth.to_csv(OUT / '하락기_월별경보_TH_v19.csv', index=False, encoding='utf-8-sig')
    sm.to_csv(OUT / '하락기_요약_TH_v19.csv', index=False, encoding='utf-8-sig')
    print('done')
