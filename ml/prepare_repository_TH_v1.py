"""Copy approved V11/V12 snapshots into a non-destructive repository namespace.

No API, environment-file loading, Git add/commit/push, or source deletion.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

BASE = Path(__file__).resolve().parents[1]
HOME = BASE / 'ml' / 'experiments_TH_v1'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tracked_snapshot():
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=BASE).decode('utf-8').split('\0')
    result = {}
    for name in names:
        if not name:
            continue
        path = BASE / name
        if path.name.lower().startswith('.env'):
            raise RuntimeError('An environment file is tracked; stop without reading it')
        if path.is_file():
            result[name] = sha(path)
    return result


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    args = parser.parse_args()
    source_root = args.source_root.resolve(strict=True)
    assert not HOME.exists(), 'Do not overwrite an existing experiment migration'
    before = tracked_snapshot()
    HOME.mkdir()
    records = []
    for version in ['v11', 'v12']:
        source = source_root / f'DFMBA_작업본_TH_{version}'
        assert source.is_dir(), f'Missing approved source {version}'
        destination = HOME / f'TH_{version}'
        destination.mkdir()
        manifest = json.loads((source / f'analysis/output_TH_{version}/run_manifest_TH_{version}.json').read_text(encoding='utf-8'))
        for name, digest in manifest['input_hashes'].items():
            assert sha(source / f'input_TH_{version}' / name) == digest
        for name, digest in manifest['code_hashes'].items():
            assert sha(source / 'ml' / name) == digest
        patterns = ['ml/*.py', 'analysis/*.py', f'input_TH_{version}/*.csv',
                    f'analysis/output_TH_{version}/*.csv', f'analysis/output_TH_{version}/*.json',
                    f'analysis/output_TH_{version}/charts_TH_{version}/*.png',
                    f'analysis/output_TH_{version}/models_TH_{version}/*.joblib',
                    f'reference_v11_TH_{version}/*.csv', '*.txt']
        for pattern in patterns:
            for path in sorted(source.glob(pattern)):
                # Packaging utilities target an old sharing layout, not the repository entry point.
                if path.name.startswith(('package_', 'share_smoke_')):
                    continue
                assert '.env' not in path.name.lower()
                relative = path.relative_to(source)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                assert not target.exists()
                shutil.copy2(path, target)
                assert sha(path) == sha(target)
                records.append(dict(version=f'TH_{version}', source_file=str(relative).replace('\\', '/'),
                                    repository_file=str(target.relative_to(BASE)).replace('\\', '/'), sha256=sha(target)))
        document_folder = BASE / 'docs' / f'TH_{version}'
        assert not document_folder.exists(), 'Do not overwrite existing documents'
        document_folder.mkdir()
        for path in sorted((source / f'reports_TH_{version}').glob('*.docx')):
            target = document_folder / path.name
            shutil.copy2(path, target)
            records.append(dict(version=f'TH_{version}', source_file=str(path.relative_to(source)).replace('\\', '/'),
                                repository_file=str(target.relative_to(BASE)).replace('\\', '/'), sha256=sha(target)))
        for path in sorted(source.glob('*.txt')):
            target = document_folder / path.name
            shutil.copy2(path, target)
            records.append(dict(version=f'TH_{version}', source_file=path.name,
                                repository_file=str(target.relative_to(BASE)).replace('\\', '/'), sha256=sha(target)))
    after = tracked_snapshot()
    assert before == after, 'Existing tracked files changed during migration'
    report = dict(date='2026-10-05', source_policy='Approved OneDrive V11/V12 snapshots; sources retained',
                  copied_files=len(records), existing_tracked_files_verified=len(before),
                  existing_tracked_sha256=before, files=records,
                  branch_created=False, staged=False, committed=False, pushed=False,
                  environment_files_read_or_copied=False)
    (HOME / '이관기록_TH_v1.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('MIGRATION PASS', len(records), 'copied files;', len(before), 'existing tracked files unchanged')


if __name__ == '__main__':
    run()
