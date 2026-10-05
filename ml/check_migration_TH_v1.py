"""Read-only integrity/Git diagnostics, with a TH-only JSON audit artifact.

Never read environment files, modify the index, or publish to GitHub.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

BASE = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=BASE).decode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, help='Optionally verify preserved source copies too')
    args = parser.parse_args()
    home = BASE / 'ml/experiments_TH_v1'
    manifest = json.loads((home / '이관기록_TH_v1.json').read_text(encoding='utf-8'))
    originals = manifest['existing_tracked_sha256']
    for name, digest in originals.items():
        if name != '.gitignore':
            assert sha(BASE / name) == digest, ('Original tracked file changed', name)
    assert git('diff', '--name-only').splitlines() == ['.gitignore']
    assert not git('diff', '--cached', '--name-only').strip(), 'Unexpected staged changes'
    diff = git('diff', '--', '.gitignore').splitlines()
    assert not any(line.startswith('-') and not line.startswith('---') for line in diff)
    for row in manifest['files']:
        path = BASE / row['repository_file']
        assert not path.name.lower().startswith('.env')
        assert sha(path) == row['sha256'], ('Copied content changed', row['repository_file'])
        if args.source_root:
            source = args.source_root / f"DFMBA_작업본_{row['version']}" / row['source_file']
            assert sha(source) == row['sha256'], ('Source changed', row['source_file'])
    candidate = [n for n in git('ls-files', '--others', '--exclude-standard', '-z').split('\0') if n]
    candidates = []
    for name in candidate:
        path = BASE / name
        assert not path.name.lower().startswith('.env'), 'Environment file exposed; not opened'
        assert path.suffix.lower() not in ['.pem', '.key', '.joblib', '.zip', '.pyc']
        assert path.stat().st_size < 100_000_000, ('Oversized Git file', name)
        if path.suffix.lower() in ['.py', '.txt', '.json', '.csv']:
            content = path.read_text(encoding='utf-8-sig')
            assert not re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', content)
            assert not re.search(r'gh[pousr]_[A-Za-z0-9]{30,}', content)
        candidates.append(dict(file=name, bytes=path.stat().st_size))
    ignored = ['.env']
    for version in ['v11', 'v12']:
        paths = list((home / f'TH_{version}').rglob('*.joblib'))
        paths += list((home / f'TH_{version}').rglob('*_model_ready_TH_v*.csv'))
        paths += [BASE / f'ml/output/TH_{version}/예측값_TH_{version}.csv']
        for path in paths:
            relative = str(path.relative_to(BASE)).replace('\\', '/')
            assert subprocess.run(['git', 'check-ignore', '-q', relative], cwd=BASE).returncode == 0
            ignored.append(relative)
    assert subprocess.run(['git', 'check-ignore', '-q', '.env'], cwd=BASE).returncode == 0
    report = dict(date='2026-10-05', branch=git('branch', '--show-current').strip(),
                  existing_files_checked=len(originals), original_exception='.gitignore: append-only TH rules',
                  copied_files_checked=len(manifest['files']), sources_checked=bool(args.source_root),
                  ignored_files_checked=len(ignored), untracked_candidates=len(candidates),
                  candidate_total_bytes=sum(r['bytes'] for r in candidates),
                  private_key_and_github_token_pattern_scan='passed; not a comprehensive secret audit',
                  env_contents_read=False, staged_changes=False, git_mutations_performed=False,
                  candidates=candidates)
    target = BASE / 'docs/이관검증_TH_v1.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('MIGRATION AUDIT PASS', report['copied_files_checked'], 'copies;', len(originals),
          'existing files;', len(candidates), 'Git candidates;', round(report['candidate_total_bytes']/1e6, 2), 'MB')


if __name__ == '__main__':
    main()
