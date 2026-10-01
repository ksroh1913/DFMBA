# -*- coding: utf-8 -*-
"""
수집/전처리 스크립트가 공유하는 공통 유틸.

디렉터리 규약:
  raw/<기관>/<통계표>.csv    - API 응답을 가공 없이 그대로 저장 (수집 단계 산출물)
  processed/<통계>.csv       - 전처리 결과 (병합 단계 입력). 데이터사전 변수만 둔다
  analysis/output/           - 데이터사전 밖의 분석 산출물 (품질조정 지수 등)
  variables.py               - 데이터사전 Master ID ↔ raw·전처리·processed 대응표

raw는 원본 필드를 그대로 보존하므로, 전처리 기준이 바뀌어도 API를 다시 호출할
필요 없이 전처리만 다시 돌리면 된다.
"""

import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

# 취합 대상 광역지자체 (전국/수도권 등 집계행 제외)
REGIONS = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
]
REGION_SET = set(REGIONS)

SEOUL_GU = {
    "11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구",
    "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구",
    "11305": "강북구", "11320": "도봉구", "11350": "노원구", "11380": "은평구",
    "11410": "서대문구", "11440": "마포구", "11470": "양천구", "11500": "강서구",
    "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
    "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구",
    "11740": "강동구",
}


# 시도 표기 -> 약칭. 표마다 정식명칭·구명칭·약칭이 섞여 있어 이름으로 맞춘다.
_SIDO_ALIASES = {
    "서울": ["서울특별시", "서울시"], "부산": ["부산광역시"], "대구": ["대구광역시"],
    "인천": ["인천광역시"], "광주": ["광주광역시"], "대전": ["대전광역시"],
    "울산": ["울산광역시"], "세종": ["세종특별자치시", "세종시"], "경기": ["경기도"],
    "강원": ["강원도", "강원특별자치도"], "충북": ["충청북도"], "충남": ["충청남도"],
    "전북": ["전라북도", "전북특별자치도"], "전남": ["전라남도"], "경북": ["경상북도"],
    "경남": ["경상남도"], "제주": ["제주도", "제주특별자치도"],
}
SIDO_SHORT = {alias: short for short, names in _SIDO_ALIASES.items() for alias in names + [short]}


def sido_short(name):
    """'서울특별시'·'강원특별자치도'·'전라북도' 등 -> '서울'·'강원'·'전북'. 대상 밖이면 None."""
    return SIDO_SHORT.get((name or "").strip())


# 행정표준코드 2자리 -> 약칭 (주민등록·인구이동 등 행안부 계열 표)
# 강원 42->51(2023.6), 전북 45->52(2024.1)로 코드가 바뀌었으므로 양쪽 모두 받는다.
ADMIN_SIDO = {
    "11": "서울", "26": "부산", "27": "대구", "28": "인천", "29": "광주", "30": "대전",
    "31": "울산", "36": "세종", "41": "경기", "42": "강원", "51": "강원", "43": "충북",
    "44": "충남", "45": "전북", "52": "전북", "46": "전남", "47": "경북", "48": "경남",
    "50": "제주",
}
# 2026-07 광주·전남 통합 이후 코드. 통합 시도(12) 아래 광주 5개 구가 남아 있다.
MERGED_GWANGJU_JEONNAM = "12"
GWANGJU_GU_2026 = ("12210", "12240", "12270", "12300", "12330")


def restore_gwangju_jeonnam(values_by_code):
    """
    한 시점·한 항목의 {행정코드: 값}에서 통합(12)만 있고 광주(29)·전남(46)이 없으면
    광주 = 5개 구 합계, 전남 = 통합 - 광주 로 복원한다.
    합산이 성립하는 저량(인구·세대)과 순이동에만 써야 한다. 전입·전출은 광주-전남 간
    이동이 통합 내부 이동으로 바뀌므로 복원할 수 없다.
    반환: {'광주': v, '전남': v} 또는 {}
    """
    if "29" in values_by_code or "46" in values_by_code:
        return {}
    total = values_by_code.get(MERGED_GWANGJU_JEONNAM)
    parts = [values_by_code.get(c) for c in GWANGJU_GU_2026]
    if total is None or any(p is None for p in parts):
        return {}
    gwangju = sum(parts)
    return {"광주": gwangju, "전남": total - gwangju}


# ---------------------------------------------------------------------------
# 환경변수
# ---------------------------------------------------------------------------
def load_env(path=ENV_PATH):
    env = {}
    if not os.path.exists(path):
        return env
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$", line)
            if m:
                env[m.group(1)] = m.group(2).strip()
    return env


def require_key(name):
    env = load_env()
    key = env.get(name) or os.environ.get(name)
    if not key:
        raise SystemExit(f"{name}가 .env에 없습니다.")
    return key


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def http_get(url, timeout=30, retries=3, retry_wait=2):
    last_err = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as res:
                return res.read().decode("utf-8")
        except Exception as e:
            last_err = e
            time.sleep(retry_wait)
    raise last_err


def http_get_json(base_url, params, **kwargs):
    url = base_url + "?" + urllib.parse.urlencode(params, safe="+")
    return json.loads(http_get(url, **kwargs))


# ---------------------------------------------------------------------------
# raw 저장/로드 (키 기준 병합 -> 재실행 시 갱신분만 덮어씀)
# ---------------------------------------------------------------------------
def raw_path(source, name):
    """(출처, 원래 이름) -> raw/<1차 동인 폴더>/<ID_데이터명>.csv. 위치 표는 raw_layout.py"""
    from raw_layout import locate
    path = os.path.join(RAW_DIR, *locate(source, name))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def load_raw(source, name):
    """raw CSV를 dict 리스트로 읽는다. 없으면 빈 리스트."""
    path = raw_path(source, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _merge_key(row, key_fields):
    """
    병합 키. API가 주는 None과 CSV에서 읽은 빈 문자열은 같은 값으로 취급해야 한다.
    (그렇지 않으면 재실행 때 'None' != '' 이 되어 같은 행이 중복 누적된다)
    """
    parts = []
    for k in key_fields:
        v = row.get(k)
        parts.append("" if v is None else str(v))
    return tuple(parts)


def save_raw(source, name, rows, key_fields):
    """
    새로 받은 rows를 기존 raw에 병합해 저장한다.
    key_fields 조합이 같으면 새 값으로 교체(확정치 갱신 반영).
    """
    if not rows and not os.path.exists(raw_path(source, name)):
        return 0

    existing = load_raw(source, name)
    merged = {}
    for r in existing:
        merged[_merge_key(r, key_fields)] = r
    for r in rows:
        merged[_merge_key(r, key_fields)] = r

    fieldnames = []
    for r in merged.values():
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    path = raw_path(source, name)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for _, r in sorted(merged.items()):
            w.writerow(r)
    return len(merged)


def raw_max_value(source, name, field):
    """기존 raw에서 특정 필드의 최대값(예: 최신 시점)을 반환. 증분 수집 기준점."""
    rows = load_raw(source, name)
    vals = [r.get(field, "") for r in rows if r.get(field)]
    return max(vals) if vals else None


# ---------------------------------------------------------------------------
# processed 저장
# ---------------------------------------------------------------------------
def write_processed(name, header, rows):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, f"{name}.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


ANALYSIS_OUTPUT_DIR = os.path.join(BASE_DIR, "analysis", "output")


def write_analysis(name, header, rows):
    """데이터사전 변수가 아닌 분석 산출물은 processed/ 대신 analysis/output/ 에 둔다."""
    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(ANALYSIS_OUTPUT_DIR, f"{name}.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def check_unique_keys(rows, key_idx, label=""):
    """(지역,연월,통계명,...) 키 유일성 검증. 중복 시 경고 출력."""
    seen = {}
    dups = []
    for r in rows:
        key = tuple(r[i] for i in key_idx)
        val = r[-2] if len(r) >= 2 else None
        if key in seen and seen[key] != val:
            dups.append((key, seen[key], val))
        seen[key] = val
    if dups:
        print(f"!! {label} 키 중복 {len(dups)}건:", file=sys.stderr)
        for key, v1, v2 in dups[:5]:
            print(f"   {key}: {v1} vs {v2}", file=sys.stderr)
    return dups


# ---------------------------------------------------------------------------
# R-ONE 지역 계층 파싱 (여러 전처리 스크립트가 공유)
# ---------------------------------------------------------------------------
def parse_rone_region(cls_fullnm):
    """
    R-ONE의 CLS_FULLNM을 (광역지자체, 자치구)로 변환. 대상 밖이면 None.

    계층 깊이가 통계표마다 달라서 마지막 조각으로 판정한다.
      '경기'                          -> ('경기', '')
      '전남광주>광주'                  -> ('광주', '')      통합코드 하위의 기존 시도
      '전남광주>(구)광주'              -> ('광주', '')      통합 이전 명칭을 쓰는 표
      '경기>계'                        -> ('경기', '')      거래현황/미분양 계열의 시도 합계행
      '서울>강남구'                    -> ('서울', '강남구')
      '서울>강북지역>도심권>종로구'     -> ('서울', '종로구')  가격지수 계열의 4단 계층
      '서울>강남지역'                  -> None              권역 중간 집계행은 제외
    """
    if not cls_fullnm:
        return None
    parts = cls_fullnm.split(">")
    last, first = parts[-1], parts[0]
    # 전남·광주 통합 이후 일부 표는 통합 이전 시도를 '(구)광주'처럼 표기한다.
    # 값이 관할 시군구 합계와 일치함을 확인했으므로 해당 시도 합계로 취급한다.
    if last.startswith("(구)") and last[3:] in REGION_SET:
        return last[3:], ""
    if last in REGION_SET:
        return last, ""
    if last == "계" and first in REGION_SET and len(parts) == 2:
        return first, ""
    if first == "서울" and last.endswith(("구", "군")) and len(parts) >= 2:
        return "서울", last
    return None


# ---------------------------------------------------------------------------
# 기간 유틸
# ---------------------------------------------------------------------------
def month_range(start_ym, end_ym):
    """'YYYYMM' 구간의 월 목록"""
    y, m = int(start_ym[:4]), int(start_ym[4:6])
    ey, em = int(end_ym[:4]), int(end_ym[4:6])
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def shift_ym(ym, months):
    """'YYYYMM' 또는 'YYYY-MM'에서 months만큼 이동한 'YYYYMM'"""
    ym = ym.replace("-", "")
    y, m = int(ym[:4]), int(ym[4:6])
    total = y * 12 + (m - 1) + months
    ny, nm = divmod(total, 12)
    return f"{ny:04d}{nm + 1:02d}"


def year_chunks(start_year, end_year, years_per_chunk=3):
    chunks = []
    y = start_year
    while y <= end_year:
        chunks.append((y, min(y + years_per_chunk - 1, end_year)))
        y += years_per_chunk
    return chunks


def to_ym(value):
    """'YYYYMM' -> 'YYYY-MM'"""
    v = str(value).replace("-", "")
    return f"{v[:4]}-{v[4:6]}"
