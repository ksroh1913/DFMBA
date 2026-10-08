"""V18-A: Y3/Y4 retrained WITHOUT class weighting (everything else identical to TH_v16).

Only change from TH_v16 (train_TH_v16.py):
  LogisticRegression / RandomForestClassifier / ExtraTreesClassifier: no class_weight='balanced'
  XGBClassifier: no scale_pos_weight
Feature sets 'full' and 'momentum', Y3 Surge (U > +1.0) and Y4 Drop (D < -1.0), quarterly expanding retraining
2018-2025, eligibility 30 events in 3 regions, inputs TH_v11/input_TH_v11 (hash-checked), seed 42.
Y1 and all benchmarks are unchanged and are taken from TH_v16.
"""
from pathlib import Path
import os, sys, json, hashlib, time, warnings, argparse
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT.parent / 'TH_v11' / 'input_TH_v11'
sys.path.insert(0, str(ROOT / 'ml'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from features_TH_v18 import engineer, history, select, matrix, addmonths

TASKS = ['Y3 Surge', 'Y4 Drop']
FAMILIES = ['Linear', 'Random Forest', 'Extra Trees', 'XGBoost']
YEARS = list(range(2018, 2026))
UP, DN = 1.0, -1.0
MIN_EVENTS, MIN_REGIONS = 30, 3
V11_INPUT_HASHES = {
    'provinces_aligned_candidates_TH_v11.csv': 'fe41218257559178b817b3595a67ac3c4fc0bad75c18c504094565e54129b249',
    'seoul_aligned_candidates_TH_v11.csv': '50a0ceea3a8619cfbb28ba6e8b9b034412b62798ea2a44ce78374488f2523a48'}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def labels(d, task): return (d.U > UP).astype(int).to_numpy() if task == 'Y3 Surge' else (d.D < DN).astype(int).to_numpy()


def make_model(family):
    if family == 'Linear':
        return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                             LogisticRegression(C=.1, max_iter=2500, random_state=42))
    if family in ['Random Forest', 'Extra Trees']:
        cls = RandomForestClassifier if family == 'Random Forest' else ExtraTreesClassifier
        return make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                             cls(n_estimators=128, min_samples_leaf=8, max_features=.5, max_depth=10, n_jobs=2, random_state=42))
    return XGBClassifier(n_estimators=128, max_depth=2, learning_rate=.06, min_child_weight=3, reg_lambda=5,
                         subsample=.8, colsample_bytree=.8, n_jobs=2, random_state=42)


def run(feature_set):
    out = ROOT / 'analysis' / f'output_TH_v18_{feature_set}'; (out / 'parts').mkdir(parents=True, exist_ok=True)
    hashes = {p.name: sha(p) for p in sorted(INPUT.glob('*.csv'))}; assert hashes == V11_INPUT_HASHES
    began = time.time(); warnings.filterwarnings('ignore')
    for panel in ['provinces', 'seoul']:
        raw = pd.read_csv(INPUT / f'{panel}_aligned_candidates_TH_v11.csv', float_precision='round_trip', low_memory=False)
        d, econ, mom = engineer(raw, panel); d = d[d[['g', 'U', 'D']].notna().all(axis=1)].reset_index(drop=True)
        for year in YEARS:
            part = out / 'parts' / f'{panel}_{year}.csv'
            if part.exists(): continue
            recs = []
            for origin in [year * 100 + m for m in [1, 4, 7, 10]]:
                tr = history(d, origin); te = d[d.month.between(origin, addmonths(origin, 2))].copy()
                assert tr.label_available.max() <= origin and tr.target_end.max() < te.month.min()
                ec, _ = select(tr, econ); mc, _ = select(tr, mom); cols = (ec + mc) if feature_set == 'full' else mc
                regions = sorted(tr.region.unique())
                for task in TASKS:
                    y = labels(tr, task)
                    if not (y.sum() >= MIN_EVENTS and tr.region[y == 1].nunique() >= MIN_REGIONS): continue
                    for family in FAMILIES:
                        classes = np.unique(y)
                        est = DummyClassifier(strategy='prior') if len(classes) == 1 else make_model(family)
                        est.fit(matrix(tr, cols, regions), np.searchsorted(classes, y))
                        p = est.predict_proba(matrix(te, cols, regions)); s = np.zeros(len(te))
                        for j, c in enumerate(est.classes_):
                            if classes[int(c)] == 1: s = p[:, j]
                        r = te[['region', 'month', 'target_end', 'label_available']].copy()
                        r = r.assign(panel=panel, year=year, origin=origin, features=feature_set, model=family, task=task,
                                     actual=labels(te, task), p_noweight=s, train_rate=float(y.mean()))
                        recs.append(r)
                print(feature_set, panel, origin, 'DONE', round(time.time() - began), 's', flush=True)
            # 평가 가능한 과제가 없는 해(예: 서울 2018)는 빈 파일로 표시만 한다.
            (pd.concat(recs) if recs else pd.DataFrame()).to_csv(part, index=False, encoding='utf-8-sig')
    pred = pd.concat([pd.read_csv(p) for p in sorted((out / 'parts').glob('*.csv')) if p.stat().st_size > 10], ignore_index=True)
    pred.to_csv(out / 'predictions_TH_v18.csv', index=False, encoding='utf-8-sig')
    (out / 'run_manifest_TH_v18.json').write_text(json.dumps(dict(version='TH_v18-A', features=feature_set, change='no class weighting',
        input_hashes=hashes, code_hashes={p.name: sha(p) for p in sorted((ROOT / 'ml').glob('*.py'))}, rows=len(pred),
        seconds=time.time() - began, versions={m: __import__(m).__version__ for m in ['numpy', 'pandas', 'sklearn', 'xgboost']}),
        ensure_ascii=False, indent=2), encoding='utf-8')
    print('COMPLETE', feature_set, len(pred), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--features', choices=['full', 'momentum'], default='full')
    run(ap.parse_args().features)
