# -*- coding: utf-8 -*-
"""설정 파일 읽기. 모든 실행 스크립트는 여기서 설정과 입력 파일 해시를 받는다."""

import hashlib
import os
import subprocess

import pandas as pd
import yaml

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(BASE, "config", "plan_v8_settings.yaml")


def sha256(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


def load(path=DEFAULT):
    with open(path, encoding="utf-8") as f:
        s = yaml.safe_load(f)
    s["_path"] = os.path.relpath(path, BASE)
    s["_sha"] = sha256(path)
    return s


def path(s, key):
    """inputs 항목의 절대 경로"""
    return os.path.join(BASE, s["inputs"][key])


def per(v):
    """2016-01 / 201601 / Period -> Period('M')"""
    if isinstance(v, pd.Period):
        return v
    if isinstance(v, (int, float)):
        v = int(v)
        return pd.Period(f"{v // 100}-{v % 100:02d}", "M")
    return pd.Period(str(v), "M")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=BASE, text=True).strip()
    except Exception:
        return "unknown"


def manifest(s, extra=None):
    """실행 기록: 설정 해시, 입력 파일 해시, 커밋, 패키지 버전, 시드"""
    import numpy
    import sklearn
    rows = [("settings", s["_path"], s["_sha"]), ("commit", git_commit(), "")]
    for k in ("preprocessed_xlsx", "merged_xlsx", "raw_rent", "raw_sale", "raw_jeonse", "raw_old_rent"):
        p = path(s, k)
        rows.append(("input", s["inputs"][k], sha256(p) if os.path.exists(p) else "missing"))
    rows += [("package", "pandas", pd.__version__), ("package", "numpy", numpy.__version__),
             ("package", "scikit-learn", sklearn.__version__), ("seed", str(s["meta"]["random_seed"]), "")]
    for k, v in (extra or {}).items():
        rows.append(("run", k, str(v)))
    return pd.DataFrame(rows, columns=["구분", "항목", "값"])
