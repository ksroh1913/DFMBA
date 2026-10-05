"""Non-destructive TH-to-team result-table schema bridge.

Preserve metrics and target meanings. Do not fabricate SHAP, neutral classes,
calibration, legacy AUC definitions, or API collection provenance.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import sys

BASE = Path(__file__).resolve().parents[1]
HOME = BASE / 'ml' / 'experiments_TH_v1'
PANELS = {'provinces': '17시도', 'seoul': '서울25구'}
TASKS = {'Y1 Growth': '변화율', 'Y2 Direction': '방향', 'Y3 Surge': '급등', 'Y4 Drop': '급락'}
PERIODS = {'all_2018_2025': '2018~2025', 'recent_2021_2025': '2021~2025'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(version):
    import numpy as np
    import pandas as pd

    assert version in ['v11', 'v12']
    core = HOME / f'TH_{version}'
    source = core / f'analysis/output_TH_{version}'
    destination = BASE / 'ml' / 'output' / f'TH_{version}'
    destination.mkdir(exist_ok=True)

    def read(name):
        return pd.read_csv(source / f'{name}_TH_{version}.csv', float_precision='round_trip', low_memory=False)

    def metadata(frame):
        out = pd.DataFrame(index=frame.index)
        out['패널'] = frame.panel.map(PANELS)
        out['과제'] = frame.task.map(TASKS)
        out['변수'] = frame.experiment
        out['모형'] = frame.model
        linear = frame.model.eq('Linear')
        out.loc[linear & frame.task.eq('Y1 Growth'), '모형'] = 'Ridge'
        out.loc[linear & ~frame.task.eq('Y1 Growth'), '모형'] = 'Logistic'
        out['목표ID'] = frame.task.str[:2]
        out['실험버전'] = f'TH_{version}'
        out['정답정의'] = frame.task.map({
            'Y1 Growth': '직전월 대비 향후6번째월 변화율(%p)',
            'Y2 Direction': '양수 상승/음수 하락; 실제0%는 Y2만 제외',
            'Y3 Surge': '향후6개월 U>연초학습85%분위(0이상)',
            'Y4 Drop': '향후6개월 D<연초학습15%분위(0이하)',
        })
        assert out['패널'].notna().all() and out['과제'].notna().all()
        return out

    def metrics(frame):
        out = metadata(frame)
        if 'period' in frame:
            out['시험구간'] = frame.period.map(PERIODS)
        if 'year' in frame:
            out['시험연도'] = frame.year
        columns = {'n': '시험행수', 'Accuracy': '정확도', 'Precision': '정밀도', 'Recall': '재현율',
                   'Event_F1': '양성사건_F1', 'Macro_F1': 'Macro_F1', 'BalancedAccuracy': '균형정확도',
                   'DownPrecision': '음성정밀도', 'DownRecall': '음성재현율', 'Down_F1': '음성_F1',
                   'MAE': 'MAE', 'RMSE': 'RMSE', 'R2': 'R2', 'ActualEvents': '양성정답수',
                   'ActualDown': '음성정답수', 'PredictedEvents': '양성예측수', 'TP': 'TP', 'FP': 'FP',
                   'FN': 'FN', 'TN': 'TN', 'Abstentions': '미결정수',
                   'AUC_within_month': 'AUC_동월지역순위', 'valid_auc_months': 'AUC_유효월수',
                   'AUC_pooled_secondary': 'AUC_전체보조', 'PR_AUC_pooled_secondary': 'AP_전체보조'}
        for old, new in columns.items():
            if old in frame:
                out[new] = frame[old]
        # Negative class = down for Y2, non-event for Y3/Y4; do not label all as 'drop'.
        out['양성범주'] = frame.task.map({'Y2 Direction': '상승', 'Y3 Surge': '급등', 'Y4 Drop': '급락'})
        out['음성범주'] = frame.task.map({'Y2 Direction': '하락', 'Y3 Surge': '비급등', 'Y4 Drop': '비급락'})
        out['평가집계'] = '해당기간 관측 통합; AUC는 동월 지역쌍 가중'
        assert out['시험행수'].equals(frame.n)
        for old, new in columns.items():
            if old in frame:
                assert np.allclose(out[new], frame[old], equal_nan=True)
        return out

    outputs = {}
    def save(name, frame):
        path = destination / f'{name}_TH_{version}.csv'
        frame.to_csv(path, index=False, encoding='utf-8-sig')
        outputs[path.name] = dict(rows=len(frame), columns=frame.columns.tolist(), sha256=sha(path))

    comparison = read('comparison')
    save('결과_요약', metrics(comparison))
    save('결과_fold별', metrics(read('annual_metrics')))
    fold = read('folds')
    meta = metadata(fold)
    for old, new in {'origin': '학습기준월', 'train_start': '학습시작월', 'train_end': '학습끝월',
                     'train_target_end': '학습정답경로끝월', 'train_label_available': '학습정답이용가능월',
                     'test_start': '시험시작월', 'test_end': '시험끝월', 'ntrain': '학습행수', 'ntest': '시험행수',
                     'nfeatures': '선택특징수', 'momentum_features': '모멘텀특징수', 'parameters': '모형설정'}.items():
        meta[new] = fold[old]
    save('학습_fold별', meta)
    th = fold[['panel', 'year', 'up', 'dn']].drop_duplicates()
    assert not th.duplicated(['panel', 'year']).any(), 'Annual thresholds are not fixed'
    threshold = pd.DataFrame({'패널': th.panel.map(PANELS), '시험연도': th.year,
                              'eps': 0.0, 'up': th.up, 'dn': -th.dn,
                              '급락하한_D': th.dn, '보합사용': False, '꼬리비중': .15,
                              '정답정의': 'Y2 2분류; Y3/Y4 연초학습 기준 연내고정'})
    save('임계값', threshold)
    features = read('features')
    used = pd.DataFrame({'패널': features.panel.map(PANELS), '학습기준월': features.origin,
                         '변수': features.feature, '관측률': features.coverage,
                         '고유값수': features.unique, '사용여부': features.selected})
    used['Master_ID'] = used['변수'].str.extract(r'(V\d{3})', expand=False)
    save('변수선택', used)

    pred = read('predictions')
    out = metadata(pred)
    out['시험연도'] = pred.year
    out['region'] = pred.region
    out['month'] = pred.month
    out['학습기준월'] = pred.origin
    out['정답경로끝월'] = pred.target_end
    out['정답이용가능월'] = pred.label_available
    out['정답'] = pred.actual.astype(object)
    out['예측'] = pred.prediction.astype(object)
    direction = pred.task.eq('Y2 Direction')
    out.loc[direction, '정답'] = pred.loc[direction, 'actual'].map({0: '하락', 1: '상승'})
    out.loc[direction, '예측'] = pred.loc[direction, 'prediction'].map({0: '하락', 1: '상승', -1: '미결정'})
    assert out.loc[direction, '정답'].notna().all() and out.loc[direction, '예측'].notna().all()
    for old, new in [('auc_score_0', 'p_하락'), ('auc_score_1', 'p_상승')]:
        out[new] = pred[old].where(direction)
    event = pred.task.isin(['Y3 Surge', 'Y4 Drop'])
    out['p_0'] = pred.auc_score_0.where(event)
    out['p_1'] = pred.auc_score_1.where(event)
    out['점수용도'] = '미보정 모델점수; AUC 진단용; 별도50%운영경보없음'
    assert 'p_보합' not in out
    assert not out.duplicated(['패널', '변수', '모형', '목표ID', 'region', 'month']).any()
    reverse = out.loc[direction, '정답'].map({'하락': 0, '상승': 1})
    assert np.array_equal(reverse.to_numpy(), pred.loc[direction, 'actual'].to_numpy())
    save('예측값', out)

    inputs = {}
    for panel, count in [('provinces', 17), ('seoul', 25)]:
        path = core / f'input_TH_{version}' / f'{panel}_aligned_candidates_TH_{version}.csv'
        raw = pd.read_csv(path, float_precision='round_trip', low_memory=False)
        required = ['panel', 'region', 'region_code', 'month', 'Y_평가사용가능']
        assert all(c in raw for c in required)
        assert raw.region.nunique() == count and not raw.duplicated(['region', 'month']).any()
        target = [c for c in raw if c.startswith('Y_V001')]
        assert len(target) == 1 and raw['Y_평가사용가능'].isin([0, 1]).all()
        inputs[panel] = dict(rows=len(raw), regions=count, first_month=int(raw.month.min()),
                             last_month=int(raw.month.max()), target_column=target[0],
                             variable_ids=sorted(set(re.findall(r'V\d{3}', ' '.join(raw.columns)))), sha256=sha(path))
    contract = dict(version=f'TH_{version}', inputs=inputs, outputs=outputs,
                    legacy_pipeline_replaced=False, legacy_report_builder_compatible=False,
                    differences=['CSV fixed snapshot instead of automatically latest Excel', 'Y1 regression added',
                                 'Y2 binary without neutral probability', '15% instead of original team 20% tails',
                                 'quarterly instead of annual refits', '2018-2025 full test and 2021-2025 primary',
                                 'pooled metrics and within-month AUC, not legacy annual metric averages',
                                 'no new SHAP; do not supply synthetic SHAP tables'])
    (destination / f'양식점검_TH_{version}.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SCHEMA EXPORT PASS', version, len(outputs), 'tables;', len(pred), 'predictions unchanged')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=['v11', 'v12', 'both'], default='both')
    parser.add_argument('--runtime', type=Path)
    args = parser.parse_args()
    if args.runtime:
        sys.path.insert(0, str(args.runtime.resolve(strict=True)))
    elif os.environ.get('DFMBA_RUNTIME'):
        sys.path.insert(0, os.environ['DFMBA_RUNTIME'])
    for version in ['v11', 'v12'] if args.version == 'both' else [args.version]:
        export(version)


if __name__ == '__main__':
    main()
