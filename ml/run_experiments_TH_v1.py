"""Repository entry point for preserved TH V11/V12 research experiments.

Default: read-only snapshot checks. API/.env/Git mutations are never performed.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import warnings

BASE = Path(__file__).resolve().parents[1]
HOME = BASE / 'ml' / 'experiments_TH_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(version):
    root = HOME / f'TH_{version}'
    manifest = json.loads((root / f'analysis/output_TH_{version}/run_manifest_TH_{version}.json').read_text(encoding='utf-8'))
    for name, digest in manifest['input_hashes'].items():
        assert sha(root / f'input_TH_{version}' / name) == digest, ('Input changed', name)
    for name, digest in manifest['code_hashes'].items():
        assert sha(root / 'ml' / name) == digest, ('Training code changed', name)
    assert manifest['fit_count'] == 1024 and manifest['prediction_rows'] == 129024
    print('SNAPSHOT CHECK PASS', version, 'input/training-code hashes match completed experiment', flush=True)


def smoke(version):
    import numpy as np
    import pandas as pd
    sys.dont_write_bytecode = True
    root = HOME / f'TH_{version}'
    sys.path.insert(0, str(root / 'ml'))
    module = importlib.import_module(f'train_simple_TH_{version}')
    assert module.ROOT.resolve() == root.resolve()
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    reference = pd.read_csv(root / f'analysis/output_TH_{version}/predictions_TH_{version}.csv',
                            float_precision='round_trip', low_memory=False)
    records = []
    for panel in ['provinces', 'seoul']:
        raw = pd.read_csv(root / f'input_TH_{version}/{panel}_aligned_candidates_TH_{version}.csv',
                          float_precision='round_trip', low_memory=False)
        data, economics, momentum = module.engineer(raw, panel)
        data = data[data[['g', 'U', 'D']].notna().all(axis=1)].reset_index(drop=True)
        annual = module.history(data, 202501)
        threshold = dict(up=float(max(0, annual.U.quantile(.85))), dn=float(min(0, annual.D.quantile(1-.85))))
        tr0 = module.history(data, 202510)
        te0 = data[data.month.between(202510, 202512)].copy()
        econ, _ = module.select(tr0, economics)
        mom, _ = module.select(tr0, momentum)
        regions = sorted(tr0.region.unique())
        for task in module.TASKS:
            train = tr0[tr0.g.ne(0)].copy() if task == module.TASKS[1] else tr0
            test = te0[te0.g.ne(0)].copy() if task == module.TASKS[1] else te0
            for family in module.FAMILIES:
                _, prediction, score, _, _ = module.fit(family, task, train, test, econ+mom, regions, threshold)
                ref = reference[(reference.panel == panel) & (reference.origin == 202510) &
                                (reference.task == task) & (reference.model == family) & (reference.experiment == 'E_M')]
                observed = test[['region', 'month']].copy()
                observed['new_prediction'] = prediction
                observed['new_actual'] = module.labels(test, threshold, task)
                if score is not None:
                    observed['new_score'] = score[:, 1]
                joined = observed.merge(ref, on=['region', 'month'], validate='one_to_one')
                assert len(joined) == len(ref) == len(test)
                error = float(np.max(np.abs(joined.new_prediction-joined.prediction)))
                assert error < 1e-8 and np.allclose(joined.new_actual, joined.actual, atol=1e-10)
                if score is not None:
                    assert np.max(np.abs(joined.new_score-joined.auc_score_1)) < 1e-8
                records.append(dict(panel=panel, task=task, model=family, rows=len(test), max_error=error, passed=True))
        print('REPOSITORY REFIT MATCH', version, panel, flush=True)
    folder = BASE / 'ml/output' / f'TH_{version}'
    folder.mkdir(exist_ok=True)
    (folder / f'실행점검_TH_{version}.json').write_text(
        json.dumps(dict(refit_checks=len(records), failed=0, checks=records), ensure_ascii=False, indent=2), encoding='utf-8')
    print('REPOSITORY SMOKE PASS', version, len(records), 'independent refits', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=['v11', 'v12', 'both'], default='both')
    parser.add_argument('--stage', choices=['check', 'smoke', 'train', 'review', 'export', 'all'], default='check')
    parser.add_argument('--runtime', type=Path, help='Optional Python library directory, never an .env file')
    args = parser.parse_args()
    environment = os.environ.copy()
    if args.runtime:
        runtime = args.runtime.resolve(strict=True)
        assert runtime.is_dir(), '--runtime must be a directory'
        environment['DFMBA_RUNTIME'] = str(runtime)
        sys.path.insert(0, str(runtime))
        os.environ['DFMBA_RUNTIME'] = str(runtime)
    elif environment.get('DFMBA_RUNTIME'):
        sys.path.insert(0, environment['DFMBA_RUNTIME'])

    def execute(path):
        subprocess.run([sys.executable, '-B', '-X', 'utf8', str(path)], cwd=BASE, env=environment, check=True)

    for version in ['v11', 'v12'] if args.version == 'both' else [args.version]:
        root = HOME / f'TH_{version}'
        if args.stage == 'check':
            check(version)
        elif args.stage == 'smoke':
            check(version)
            smoke(version)
        else:
            if args.stage in ['train', 'all']:
                execute(root / f'ml/train_simple_TH_{version}.py')
            if args.stage in ['review', 'all']:
                execute(root / f'analysis/review_simple_TH_{version}.py')
                if version == 'v12':
                    execute(root / 'analysis/summarize_experiment_TH_v12.py')
            if args.stage in ['export', 'all']:
                from export_results_TH_v1 import export
                export(version)


if __name__ == '__main__':
    main()
