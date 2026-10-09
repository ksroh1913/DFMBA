# -*- coding: utf-8 -*-
"""
[10차 준비] 변수 카드 생성기 — raw/ 의 API 원자료만 보고 변수 하나의 전처리를 결정하기 위한 한 페이지 html.

processed/·취합본(전처리본)은 읽지 않는다. raw 파일을 직접 시도×기간 표로 만들어 결측·후보 변환·정상성·타깃 관계를 계산한다.
타깃도 raw 의 V001(R-ONE 월세통합가격지수)에서 직접 만든다.

카드 구성: 1 사전 정보(데이터사전·variables.py 공표시차) 2 raw 파일별 요약·표본 3 raw → 시도×기간 표(항목·기간·지역·결측률, 그래프)
4 결측 진단·처리 제안 5 후보 변환별 정상성·분포 6 타깃 관계(결정월 ≤ 2017-12 탐색창만) [V001: 타깃 절] 7 자동 권고(초안) 8 결정란

결정 로그 docs/전처리결정로그_10차.csv: 카드를 만들면 '초안' 행을 넣고(이미 '결정'이면 건드리지 않음), --decide 로 결정을 적는다.
사용:
  PYTHONUTF8=1 python analysis/variable_card.py V001 V009
  PYTHONUTF8=1 python analysis/variable_card.py --decide V009 변환=100·로그Δ12 역할=취약성 결측처리="..." 비고="..."
  PYTHONUTF8=1 python analysis/variable_card.py --refresh-cache       # 타깃·세대수·인구 캐시 다시 만들기
출력: analysis/output/변수카드/<ID>_<변수명>.html, 캐시 analysis/output/변수카드/_cache.pkl (git 제외)
"""

import argparse
import base64
import datetime as dt
import glob
import io
import os
import pickle
import re
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "analysis"))
from common import ADMIN_SIDO, GWANGJU_GU_2026, REGIONS, SEOUL_GU, parse_rone_region, sido_short  # noqa: E402

GU_LIST = list(SEOUL_GU.values())      # 서울 25개 구 (행정코드 순)
from review_model_inputs import test_series  # noqa: E402
from variables import BY_ID, RELEASE, RELEASE_MONTHLY  # noqa: E402

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(BASE, "analysis", "output", "변수카드")
CACHE = os.path.join(OUT, "_cache.pkl")
LOG = os.path.join(BASE, "docs", "전처리결정로그_10차.csv")
DICT = os.path.join(BASE, "docs", "260928_데이터사전_취합_수정.xlsx")
RAW = os.path.join(BASE, "raw")
EXPLORE_END = pd.Period("2017-12", "M")          # 타깃 관계는 정답 확인 시점이 이 달 이전인 결정월만
WIN = (pd.Period("2011-01", "M"), pd.Period("2025-12", "M"))
ZWIN = (pd.Period("2016-01", "M"), pd.Period("2017-12", "M"))   # 외삽 진단용 기준 창
LOG_COLS = ["ID", "변수명", "동인", "역할", "상태", "변환", "변환식", "공표시차_개월", "추가시차", "지역단위", "저빈도→월변환", "중복처리", "결측처리",
            "정상성판정", "타깃상관(탐색창,G6)", "결정일", "비고"]
OLD_COLS = {"저빈도처리": "저빈도→월변환", "보완셀처리": "결측처리", "탐색창ρ_G6": "타깃상관(탐색창,G6)"}
DRIVER_IDS = {1: ["V009", "V010", "V011", "V012", "V013", "V015", "V017"],
              2: ["V021", "V022", "V023", "V024", "V038", "V072", "V076", "V077"],
              3: ["V002", "V003", "V004", "V007", "V026", "V031", "V036", "V050", "V056", "V057", "V059", "V060", "V069", "V071"],
              4: ["V005", "V006", "V032", "V033", "V034", "V035"],
              5: ["V001", "V037", "V061", "V062", "V063"],
              6: ["V043", "V044", "V045", "V046", "V047", "V048", "V049", "V064", "V066", "V067", "V070", "V074", "V078"]}
FLOW, STOCK, DUMMY = {"V021", "V022", "V023", "V035", "V036", "V006", "V007"}, {"V038"}, {"V078"}
CUMULATIVE = {"V021"}                     # raw 가 연초 누계(월별 누적)인 표: 인허가(DT_MLTM_1948). 착공·준공(5387·5373)은 월값 (총계가 달마다 오르내리고 12월값 ≠ 연합)
ZERO_GAPS = {"V021", "V022", "V023"}      # 주택건설실적: KOSIS 가 값 0인 행을 생략 → 첫~마지막 관측 사이 빈 셀 = 0. 보완 소분류 파일의 합계 항등식으로 검증(zero_check)
DIFFUSION = {"V005", "V031", "V032", "V033", "V034", "V037", "V060", "V061", "V062", "V063", "V067"}
RATE = {"V003", "V004", "V013", "V043", "V044", "V045", "V056", "V057", "V059", "V072"}
INTEREST = {"V043", "V044", "V045", "V056", "V057", "V059"}
TRIGGER_IDS = INTEREST | DIFFUSION | {"V035", "V036", "V006", "V007", "V064", "V078"}
NATIONAL_IDS = INTEREST | {"V046", "V047", "V048", "V049", "V060", "V064", "V066", "V067", "V069", "V070", "V071", "V074", "V078", "V061", "V062", "V063"}
V078_PATH = os.path.join(RAW, "6_거시경기금융시장여건", "V078_Korea_LTV_DSTI_monthly_dummies_1990_2026.csv")
ASSUMPTIONS = [
    ("가정_1", "분석 단위", "17개 시도 × 월 패널, 결정월 = 월말. 서울 25개 구는 보조(카드에 '구 가용'만 표시)"),
    ("가정_2", "정보 규칙", "결정월 말에 공표된 값만 (variables.RELEASE/RELEASE_MONTHLY 시차; 추가 시차는 변수마다 결정)"),
    ("가정_3", "타깃 형태", "V001 의 앞으로 h개월 변화율 G_h = 100·[R(t−1+h)/R(t−1) − 1]. 주 지평 h=6, 보조 h=3"),
    ("가정_4", "변환 기준", "모형 독립: 변수당 변환 1개, 정상(I(0)) 또는 유계. 취약성(느림→Δ12·수준) / 트리거(빠름→Δ3·변동성) 구분"),
    ("가정_5", "탐색창", "타깃 상관은 정답 확인 시점 ≤ 2017-12 인 결정월만. 2018~ 평가 구간의 타깃은 보지 않음"),
    ("가정_6", "미결", "사건 경계·모형·검증 방식은 이 단계에서 정하지 않음 (타깃 카드에 경계 후보별 사건 수만 수록)"),
]


# ============================================================ raw 파싱 → 시도×기간 표
def _period(v, kind):
    v = str(v).strip()
    if kind == "M":
        return pd.Period(f"{v[:4]}-{v[4:6]}", "M") if re.fullmatch(r"\d{6}", v) else (pd.Period(v, "M") if re.fullmatch(r"\d{4}-\d{2}", v) else None)
    if kind == "Q":
        if re.fullmatch(r"\d{4}Q\d", v):
            return pd.Period(v, "Q")
        if re.fullmatch(r"\d{6}", v):
            return pd.Period(f"{v[:4]}Q{int(v[4:6])}", "Q")
    if kind == "H":
        return pd.Period(f"{v[:4]}-{'06' if v[4:6] == '01' else '12'}", "M").asfreq("Q").asfreq("6M") if False else f"{v[:4]}H{int(v[4:6])}"
    if kind == "Y":
        return pd.Period(v[:4], "Y")
    return None


def _kosis_region(row):
    """KOSIS 행 → (지역, 수준). 표마다 코드 체계가 달라(행안부 코드 vs 통계청 코드) 이름을 먼저 보고, 코드는 이름으로 못 정할 때만 쓴다."""
    c1, nm = str(row.get("C1", "")).strip(), str(row.get("C1_NM", "")).strip()
    s = sido_short(nm)
    if s:
        return s, "sido"
    if nm in ("전국", "계", "전체", "총계", "합계") or c1 in ("00", "0"):
        return "전국", "nat"
    if nm.startswith("서울 ") and nm.endswith("구"):
        return nm[3:], "gu"
    if len(c1) == 5 and c1.isdigit() and c1[:2] == "11":
        return nm, "gu"
    if nm in GU_LIST and re.fullmatch(r"10\d\d", c1):       # 서울통계(orgId 201) 표: 구 코드 1001~1025, 이름 '종로구' 등
        return nm, "gu"
    if c1 == "12" or nm in ("전남광주", "광주전남"):
        return "광주전남(통합)", "merged"
    if c1 in GWANGJU_GU_2026:
        return nm, "gj_gu"
    if c1 in ADMIN_SIDO and len(c1) == 2 and not nm:
        return ADMIN_SIDO[c1], "sido"
    return None, None


def parse_raw_file(path, vid):
    """raw 파일 하나 → dict(info, sample, kind, items{item: wide(시도×기간)}, gu_count, notes)"""
    df = pd.read_csv(path, low_memory=False)
    cols = set(df.columns)
    info = {"파일": os.path.relpath(path, BASE), "행수": len(df)}
    notes, items_w, gu = [], {}, set()
    if "PRD_DE" in cols:                                   # KOSIS
        fmt = "KOSIS"
        se = str(df["PRD_SE"].iloc[0]).strip() if "PRD_SE" in cols else ("M" if re.fullmatch(r"\d{6}", str(df["PRD_DE"].iloc[0])) else "Y")
        kind = {"M": "M", "Q": "Q", "S": "H", "A": "Y", "Y": "Y"}.get(se, "M")
        icols = [c for c in ("ITM_NM", "C2_NM", "C3_NM", "C4_NM") if c in cols and df[c].nunique() > 1]
        if not icols:
            icols = ["ITM_NM"] if "ITM_NM" in cols else []
        reg = df.apply(_kosis_region, axis=1, result_type="expand")
        df["_reg"], df["_lv"] = reg[0], reg[1]
        gu = set(df.loc[df["_lv"] == "gu", "_reg"].dropna())
        df["_item"] = df[icols].astype(str).agg(" | ".join, axis=1) if icols else "(단일)"
        df["_P"] = df["PRD_DE"].map(lambda v: _period(v, kind))
        df["_val"] = pd.to_numeric(df["DT"], errors="coerce")
        if (df["_lv"] == "merged").any():
            # 2026-07~ 광주·전남 통합코드(12). 저량·순이동은 광주 = 통합 아래 광주 5개 구 합, 전남 = 통합 − 광주 로 복원 (common.restore_gwangju_jeonnam 과 같은 규칙)
            add, n_rest = [], 0
            for (item, P), g in df[df["_lv"].isin(["merged", "gj_gu", "sido"])].groupby(["_item", "_P"]):
                if P is None or (g["_lv"] == "sido").any() and g.loc[g["_lv"] == "sido", "_reg"].isin(["광주", "전남"]).any():
                    continue
                tot = g.loc[g["_lv"] == "merged", "_val"]
                parts = g.loc[g["_lv"] == "gj_gu", "_val"]
                if len(tot) == 1 and len(parts) == len(GWANGJU_GU_2026) and parts.notna().all() and pd.notna(tot.iloc[0]):
                    gj = float(parts.sum())
                    add += [{"_reg": "광주", "_lv": "sido", "_item": item, "_P": P, "_val": gj}, {"_reg": "전남", "_lv": "sido", "_item": item, "_P": P, "_val": float(tot.iloc[0]) - gj}]
                    n_rest += 1
            if add:
                df = pd.concat([df, pd.DataFrame(add)], ignore_index=True)
                notes.append(f"행정코드 12(광주·전남 통합, 2026-07~) {n_rest}개 (항목, 달) 복원: 광주 = 5개 구 합, 전남 = 통합 − 광주. 저량·순이동에만 타당")
            else:
                notes.append("행정코드 12(광주·전남 통합) 행 존재하나 복원 조건(통합값 + 광주 5개 구) 미충족 → 2026-07~ 광주·전남 결측")
        info.update({"형식": "KOSIS", "주기": {"M": "월", "Q": "분기", "H": "반기", "Y": "연"}[kind], "단위": ", ".join(map(str, df["UNIT_NM"].dropna().unique()[:4])) if "UNIT_NM" in cols else "",
                     "최종갱신(LST_CHN_DE)": str(df["LST_CHN_DE"].dropna().astype(str).max()) if "LST_CHN_DE" in cols else "", "지역 구분값 예": ", ".join(map(str, df["C1_NM"].dropna().unique()[:10])) if "C1_NM" in cols else ""})
        show = [c for c in ["PRD_DE", "C1", "C1_NM"] + icols + ["DT", "UNIT_NM"] if c in cols]
    elif "WRTTIME_IDTFR_ID" in cols:                       # R-ONE
        fmt, kind = "R-ONE", "M"
        def rreg(f):
            f = str(f) if pd.notna(f) else ""
            if f == "전국" or f.startswith("전국>"):
                return ("전국", "nat")
            r = parse_rone_region(f)
            if r is None:
                return (None, None)
            return (r[0], "sido") if r[1] == "" else (r[1], "gu")
        reg = df["CLS_FULLNM"].map(rreg)
        df["_reg"], df["_lv"] = [r[0] for r in reg], [r[1] for r in reg]
        gu = set(df.loc[df["_lv"] == "gu", "_reg"].dropna())
        df["_item"] = df["ITM_NM"].astype(str)
        df["_P"] = df["WRTTIME_IDTFR_ID"].map(lambda v: _period(v, "M"))
        df["_val"] = pd.to_numeric(df["DTA_VAL"], errors="coerce")
        info.update({"형식": "R-ONE", "주기": "월", "단위": ", ".join(map(str, df["UI_NM"].dropna().unique()[:4])) if "UI_NM" in cols else "",
                     "지역 구분값 예": ", ".join(map(str, df["CLS_FULLNM"].dropna().unique()[:10]))})
        show = [c for c in ["WRTTIME_IDTFR_ID", "CLS_FULLNM", "ITM_NM", "DTA_VAL", "UI_NM"] if c in cols]
    elif "DATA_VALUE" in cols:                             # ECOS
        fmt = "ECOS"
        t0 = str(df["TIME"].iloc[0])
        kind = "Q" if "Q" in t0 else ("M" if len(t0) == 6 else "Y")
        reg_col = None
        for c in ("ITEM_NAME2", "ITEM_NAME3"):
            if c in cols and df[c].map(lambda x: sido_short(str(x)) is not None).mean() > 0.5:
                reg_col = c
        df["_reg"] = df[reg_col].map(lambda x: sido_short(str(x))) if reg_col else "전국"
        df["_lv"] = np.where(df["_reg"] == "전국", "nat", "sido")
        icols = [c for c in ("ITEM_NAME1", "ITEM_NAME2", "ITEM_NAME3", "ITEM_NAME4") if c in cols and df[c].notna().any() and c != reg_col]
        df["_item"] = df[icols].astype(str).agg(" | ".join, axis=1) if icols else "(단일)"
        df["_P"] = df["TIME"].map(lambda v: _period(v, kind))
        df["_val"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
        info.update({"형식": "ECOS", "주기": {"M": "월", "Q": "분기", "Y": "연"}[kind], "단위": ", ".join(map(str, df["UNIT_NAME"].dropna().unique()[:4])) if "UNIT_NAME" in cols else "",
                     "통계표": ", ".join(map(str, df["STAT_NAME"].dropna().unique()[:2])) if "STAT_NAME" in cols else ""})
        show = [c for c in ["TIME"] + icols + ([reg_col] if reg_col else []) + ["DATA_VALUE", "UNIT_NAME"] if c in cols]
    elif "Month" in cols:                                  # 사용자 CSV (V078)
        fmt, kind = "사용자 CSV", "M"
        P = pd.to_datetime(df["Month"], format="%b-%y", errors="coerce").dt.to_period("M")
        long = []
        for c in [c for c in df.columns if c != "Month"]:
            long.append(pd.DataFrame({"_reg": "전국", "_lv": "nat", "_item": c, "_P": P, "_val": pd.to_numeric(df[c], errors="coerce")}))
        num = df.drop(columns=["Month"])
        info.update({"형식": "사용자 CSV", "주기": "월", "값 분포": "; ".join(f"{c}: {num[c].value_counts().to_dict()}" for c in num.columns),
                     "비영(0 아닌) 월수": "; ".join(f"{c}: {int((num[c] != 0).sum())}" for c in num.columns)})
        show = list(df.columns)
        sample = df.head(5)
        df = pd.concat(long, ignore_index=True)
    else:
        return {"info": {**info, "형식": "알 수 없음"}, "sample": df.head(5), "kind": "M", "items": {}, "gu": set(), "notes": ["형식을 알 수 없어 파싱하지 않음"]}
    if fmt != "사용자 CSV":
        sample = df[show].head(5)
    ok = df["_P"].notna() & df["_val"].notna()
    vc = df.loc[ok, "_item"].value_counts()
    info["항목(행수)"] = "; ".join(f"{k} ({v})" for k, v in vc.head(10).items()) + (f" … 총 {len(vc)}개" if len(vc) > 10 else "")
    pr = df.loc[ok, "_P"]
    info["기간"] = f"{min(pr, key=str)} ~ {max(pr, key=str)} ({pr.astype(str).nunique()}개)" if len(pr) else ""
    sido_rows = df[ok & df["_lv"].isin(["sido", "nat"])]
    info["시도 수(전국 제외)"] = int(sido_rows.loc[sido_rows["_lv"] == "sido", "_reg"].nunique())
    info["서울 구 수"] = len(gu)

    def pivot(g, keep):
        w = g.pivot_table(index="_P", columns="_reg", values="_val", aggfunc="first")
        if kind in ("M", "Q", "Y"):
            w.index = pd.PeriodIndex(w.index, freq={"M": "M", "Q": "Q", "Y": "Y"}[kind])
        w = w.sort_index()
        w = w[[c for c in keep if c in w.columns]]
        w.attrs["kind"] = kind                                # 표마다 주기가 달라 표 자체에 기록
        w.attrs["file"] = os.path.basename(path)
        return w
    for item, g in sido_rows.groupby("_item"):
        items_w[item] = pivot(g, REGIONS + ["전국"])
    items_gu = {}
    gu_rows = df[ok & (df["_lv"] == "gu") & df["_reg"].isin(GU_LIST)]
    for item, g in gu_rows.groupby("_item"):
        items_gu[item] = pivot(g, GU_LIST)
    return {"info": info, "sample": sample, "kind": kind, "items": items_w, "items_gu": items_gu, "gu": gu, "notes": notes}


def find_raw_files(vid):
    files = []
    for f in glob.glob(os.path.join(RAW, "*", "*.csv")):
        b = os.path.basename(f)[:-4]
        if re.search(rf"(^|_){vid}(_|$)", b) or (vid in ("V061", "V062", "V063") and "V061-V063" in b):
            files.append(f)
    return sorted(files)


def select_items(vid, parsed_list, level="sido"):
    """variables.py 의 items/stat 에 맞는 항목을 고른다(없으면 전부). 반환 {항목: wide} (주 항목이 먼저). level='gu' 면 서울 구 표"""
    want = [str(i) for i in (BY_ID.get(vid, {}).get("items") or [])]
    src = "items" if level == "sido" else "items_gu"
    out = {}
    for p in parsed_list:
        if "보완_" in p["info"]["파일"]:
            continue
        for item, w in p.get(src, {}).items():
            if want and not any(x in item for x in want):
                continue
            key = item if item not in out else f"{item} [{os.path.basename(p['info']['파일'])[:25]}]"
            out[key] = w
    if not out:
        for p in parsed_list:
            if "보완_" in p["info"]["파일"]:
                continue
            out.update(p.get(src, {}))
    if want:
        out = dict(sorted(out.items(), key=lambda kv: next((i for i, x in enumerate(want) if x in kv[0]), 99)))
    return out


# ============================================================ raw 항목을 묶어 만드는 파생 원값 (결정 로그의 집계 규칙)
def _v011_share(parsed_list, level="sido"):
    """V011: raw 1세 단위(20세~39세) 합 ÷ 계 × 100 = 20-39세 비중(%). 반환 wide(달력월 × 지역)"""
    src = "items" if level == "sido" else "items_gu"
    items = {}
    for p in parsed_list:
        if "보완_" not in p["info"]["파일"]:
            items.update(p.get(src, {}))
    ages = [items[f"{a}세"] for a in range(20, 40) if f"{a}세" in items]
    if len(ages) < 20 or "계" not in items:
        return pd.DataFrame()
    tot = items["계"]
    young = sum(a.reindex(tot.index) for a in ages)
    return 100 * young / tot


def _v077_link(parsed_list, level="sido"):
    """V077: DT_MLTM_5560 총계(2012~2019) 뒤에 V076 DT_MLTM_6827(민간임대 등록 공급, 2020~) 의 전 구분(개인·법인 × 단기·장기일반·공공지원 × 건설·매입 = 12)
    × 주택유형(아파트·다세대·다가구·단독·오피스텔·연립·도시형생활주택·기타 = 8) 합을 이어 한 계열(연, 호)로 만든다 = 시도별 연간 임대주택 공급량.
    6827 에는 전국 행이 없어 17개 시도 합을 전국으로 둔다. 반환 wide(연 × 지역), attrs kind='Y'"""
    if level != "sido":
        return pd.DataFrame()
    base = select_items("V077", parsed_list, "sido")
    if not base:
        return pd.DataFrame()
    old = base[list(base)[0]]
    old = old[old.index <= pd.Period("2019", "Y")]
    new = None
    for f in find_raw_files("V076"):
        if "6827" not in os.path.basename(f):
            continue
        ws = list(parse_raw_file(f, "V076")["items"].values())
        if ws:
            new = pd.concat(ws, keys=range(len(ws))).groupby(level=1).sum(min_count=1)     # 구분 × 유형 전 항목 합 (빈 항목은 무시, 전부 비면 결측)
    if new is None or new.empty:
        return pd.DataFrame()
    regs = [c for c in new.columns if c in REGIONS]
    new["전국"] = new[regs].sum(axis=1, min_count=len(regs))
    new = new[new.index >= pd.Period("2020", "Y")]
    w = pd.concat([old, new]).sort_index()
    w = w[[c for c in REGIONS + ["전국"] if c in w.columns]]
    w.index = pd.PeriodIndex(w.index, freq="Y")
    w.attrs["kind"] = "Y"
    w.attrs["file"] = "DT_MLTM_5560 + DT_MLTM_6827"
    return w


DERIVED_RAW = {"V011": {"fn": _v011_share, "label": "20-39세비중(%)", "desc": "raw 1세 단위 항목 20세~39세 합 ÷ 계 × 100 (지역·달마다 21개 항목 합산)"},
               "V077": {"fn": _v077_link, "label": "연결(5560+6827)",
                        "desc": "DT_MLTM_5560 총계(2012~2019) 뒤에 V076 DT_MLTM_6827 민간임대 등록 공급의 전 구분(12) × 주택유형(8) 합(2020~)을 이어 한 계열 = 시도별 연간 임대주택 공급량(호). "
                                "6827 전국 = 17개 시도 합(전국 행 없음). 2019→2020 연결점은 5560(공공 포함 추정)→6827(민간만) 정의 차이로 하향 단절 포함 — 학습 전 재검토"}}


def derived_raw(vid, parsed_list, level="sido"):
    """vid 에 파생 집계 규칙이 있으면 (wide, 표시이름, 설명) 을, 없으면 None"""
    d = DERIVED_RAW.get(vid)
    if not d:
        return None
    w = d["fn"](parsed_list, level)
    return (w, d["label"], d["desc"]) if len(w) else None


# ============================================================ 캐시: 타깃(raw V001)·세대수(raw V010)·인구(raw V009)
def _kosis_wide(vid, item_contains, level="sido"):
    for f in find_raw_files(vid):
        p = parse_raw_file(f, vid)
        for item, w in p["items" if level == "sido" else "items_gu"].items():
            if item_contains in item:
                return w
    return pd.DataFrame()


def _panel_targets(Rs, full):
    """Rs: 달력월 × 지역 지수 → 결정월 기준 G·r (모든 지역이 있는 달만 평균)"""
    Rs = Rs.reindex(full)
    out = {"R_panel": Rs}
    for h in (1, 3, 6):
        Gp = 100 * (Rs.shift(1 - h) / Rs.shift(1) - 1)
        out[f"G{h}_panel"] = Gp
        out[f"r{h}_panel"] = Gp.sub(Gp.mean(axis=1).where(Gp.notna().all(axis=1)), axis=0)
    return out


def load_context(refresh=False):
    if os.path.exists(CACHE) and not refresh:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    vm = pd.read_excel(DICT, sheet_name="Variable_Master")
    p = parse_raw_file(find_raw_files("V001")[0], "V001")
    R = p["items"]["지수"]                                 # 시도×월 + 전국, 2015-06~
    full = pd.period_range(R.index.min(), R.index.max() + 6, freq="M")
    Rn = R["전국"].reindex(full)
    targets = _panel_targets(R[[c for c in REGIONS if c in R.columns]], full)
    for h in (1, 3, 6):
        targets[f"G{h}_nat"] = 100 * (Rn.shift(1 - h) / Rn.shift(1) - 1)   # 결정월 t: R(t−1+h)/R(t−1)
    targets["R_last_nat"] = Rn.shift(1)
    Rg = p["items_gu"].get("지수", pd.DataFrame())
    targets_gu = _panel_targets(Rg[[c for c in GU_LIST if c in Rg.columns]], full) if len(Rg) else {}
    ctx = dict(vm=vm, targets=targets, targets_gu=targets_gu, hh=_kosis_wide("V010", "세대"), pop=_kosis_wide("V009", "총인구"),
               hh_gu=_kosis_wide("V010", "세대", "gu"), pop_gu=_kosis_wide("V009", "총인구", "gu"))
    os.makedirs(OUT, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(ctx, f)
    return ctx


# ============================================================ 1 사전 정보
def dict_info(vid, ctx):
    vm = ctx["vm"]
    row = vm[vm["Master ID"] == vid]
    if row.empty:
        return {"Master ID": vid, "변수명": BY_ID.get(vid, {}).get("name", vid), "비고": "데이터사전에 없음 (사용자 추가)"}
    r = row.iloc[0]
    keep = ["Master ID", "변수명", "1차 동인", "2차 동인", "변수 역할", "정보유형", "예상 부호", "기대 선행성", "Leakage Risk", "권장 가공/파생변수",
            "수록기간", "빈도", "발표·가용 시점", "지역 범위", "서울25구", "17개시도", "출처기관", "표ID/CODE", "원본 주의사항", "중복/유사 표시", "분류 비고", "분류 근거"]
    out = {k: ("" if pd.isna(r.get(k, "")) else r.get(k, "")) for k in keep if k in vm.columns}
    out["발표·가용 시점"] = release_text(vid)                  # 사전 원문 대신 9차에서 확인한 공표시차(variables.py RELEASE) 를 쓴다
    return out


def release_text(vid):
    """variables.py 의 공표시차 기록을 한 줄로: 'N개월 — 근거 (판정)'"""
    rel = RELEASE_MONTHLY.get(vid) or RELEASE.get(vid) or {}
    if not rel:
        return "공표시차 기록 없음 (variables.py 에 없는 변수)"
    lag = rel.get("lag", "")
    txt = f"{lag}개월" + (f" (공표 {rel['day']}일 전후)" if rel.get("day") else "")
    if rel.get("evidence"):
        txt += f" — {rel['evidence']}"
    txt += f" [판정: {rel.get('verdict', '')}]"
    if rel.get("known_delay"):
        txt += " · 수집원(R-ONE·ECOS) 게시가 공식 공표보다 늦음"
    if rel.get("ended"):
        txt += " · 종료된 통계"
    return txt


def code_info(vid):
    v = BY_ID.get(vid, {})
    rel = RELEASE_MONTHLY.get(vid) or RELEASE.get(vid) or {}
    return {"variables.py 이름": v.get("name", ""), "빈도": v.get("freq", ""), "지역단위": v.get("level", ""), "출처": v.get("source", ""), "표ID": v.get("table", ""),
            "항목(variables.py)": ", ".join(v.get("items", []) or []), "상태": v.get("status", ""), "같은 변수": v.get("same_as", "") or "", "주석": v.get("note", ""),
            "공표시차(개월)": rel.get("lag", ""), "공표시차 근거(9차 확인)": rel.get("evidence", ""), "근거 출처": rel.get("source", ""),
            "사전에 적힌 시점(참고)": rel.get("dict", ""), "판정(사전 대비)": rel.get("verdict", ""), "known_delay": rel.get("known_delay", "")}


# ============================================================ 4 결측 진단
def missing_section(vid, items, kind, parsed_list):
    key0 = list(items)[0]
    w = items[key0]
    rows = []
    for c in w.columns:
        x = w[c]
        if not x.notna().any():
            rows.append({"지역": c, "첫 관측": "", "마지막 관측": "", "첫~마지막 사이 결측": "", "결측 시점(최대 8개)": ""})
            continue
        f, l = x.first_valid_index(), x.last_valid_index()
        inner = x.loc[f:l]
        miss = [str(p) for p in inner.index[inner.isna()]]
        rows.append({"지역": c, "첫 관측": str(f), "마지막 관측": str(l), "첫~마지막 사이 결측": len(miss), "결측 시점(최대 8개)": ", ".join(miss[:8]) + (" …" if len(miss) > 8 else "")})
    tab = pd.DataFrame(rows)
    gaps = pd.to_numeric(tab["첫~마지막 사이 결측"], errors="coerce").fillna(0)
    firsts = tab["첫 관측"].replace("", np.nan).dropna()
    late = tab.loc[(tab["첫 관측"] != "") & (tab["첫 관측"] != firsts.min()), ["지역", "첫 관측"]] if len(firsts) else pd.DataFrame()
    lasts = tab["마지막 관측"].replace("", np.nan).dropna()
    early_end = tab.loc[(tab["마지막 관측"] != "") & (tab["마지막 관측"] != lasts.max()), ["지역", "마지막 관측"]] if len(lasts) else pd.DataFrame()
    is_gu = any(c in GU_LIST for c in w.columns)
    missing_regions = [r for r in (GU_LIST if is_gu else REGIONS) if r not in w.columns]
    notes = list(dict.fromkeys(n for p in parsed_list for n in p["notes"]))      # 주 파일·보완 파일이 같은 주석을 내면 한 번만
    prop, extra_html = [], ""
    if missing_regions:
        prop.append(f"raw 에 없는 시도: {', '.join(missing_regions)} → 결측 유지(대체 자료 여부 결정)")
    if len(late):
        prop.append("시작이 늦은 시도(" + ", ".join(f"{r['지역']} {r['첫 관측']}" for _, r in late.iterrows()) + ")는 그 전 구간을 채우지 않고 결측으로 둠")
    if len(early_end):
        prop.append("끝이 이른 시도(" + ", ".join(f"{r['지역']} {r['마지막 관측']}" for _, r in early_end.iterrows()) + "): 공표 중단·통합 여부 확인")
    if vid in ZERO_GAPS:
        zc = zero_check(vid, w)
        prop.append("합계 대조: " + zc["요약"] + (" → 첫~마지막 관측 사이 빈 셀은 0 으로 채움 (결정 시 결측처리에 '빈 셀 0' 기재)" if zc["ok"] else " → 0 으로 확인되지 않은 빈 셀은 결측 유지"))
        if zc["표"] is not None:
            extra_html = "<p class='note'><b>보완 소분류 파일 합계 대조</b> (" + zc["파일"] + ")</p>" + zc["표"].to_html(index=False)
    elif (gaps > 0).any():
        prop.append(f"중간 결측 {int(gaps.sum())}셀: KOSIS 는 값이 0인 행을 생략하기도 함 → 상위 합계와 대조해 0 이 확인되면 0, 아니면 직전 공표값 유지 또는 결측 유지. 전후 보간(미래 값)은 금지")
    prop += notes
    if not prop:
        prop.append("결측 없음 — 처리 불필요")
    html = tab.to_html(index=False) + extra_html + "<p class='note'><b>처리 제안(초안):</b> " + " / ".join(prop) + "</p>"
    return html, " / ".join(prop)


# ============================================================ 5 후보 변환·정상성
def vtype_of(vid):
    if vid in DUMMY:
        return "더미"
    if vid == "V012":
        return "순이동"
    if vid in FLOW:
        return "흐름"
    if vid in STOCK:
        return "재고"
    if vid in DIFFUSION:
        return "확산지수"
    if vid in RATE:
        return "비율·금리"
    return "지수·금액·수량"


_ZERO_CACHE = {}


def zero_check(vid, wide):
    """주택건설실적(ZERO_GAPS) 주 파일의 빈 셀이 '아파트 = 0' 인지 보완 소분류 파일로 확인.
    항등식 계 = 단독 + 다가구(동수) + 아파트 + 다세대 + 연립 (KOSIS 합계가 다가구를 동수로 세므로) 이 성립하면,
    아파트가 빈 (시도, 달) 에서 계 − 나머지 == 0 ⇔ 아파트 0. 반환: ok(첫~마지막 관측 사이 빈 셀이 전부 0 확인), 요약, 표, 파일"""
    if vid in _ZERO_CACHE:
        return _ZERO_CACHE[vid]
    res = {"ok": False, "요약": "보완 소분류 파일 없음 → 합계 대조 불가", "표": None, "파일": ""}
    files = [f for f in find_raw_files(vid) if "보완_" in os.path.basename(f) and "소분류" in os.path.basename(f)]
    if not files:
        _ZERO_CACHE[vid] = res
        return res
    b = pd.read_csv(files[0], low_memory=False, dtype={"PRD_DE": str})
    b["DT"] = pd.to_numeric(b["DT"], errors="coerce")
    ccols = [c for c in ("C2_NM", "C3_NM") if c in b.columns]

    def cat(r):
        a = [str(r[c]).strip() for c in ccols]
        if a[0].startswith(("계", "합계")):
            return "계"
        if a[0] == "단독" and len(a) > 1 and a[1] == "다가구":
            return "다가구동수"
        return a[0]
    b["_cat"] = b.apply(cat, axis=1)
    reg = b.apply(_kosis_region, axis=1, result_type="expand")
    b["_reg"], b["_lv"] = reg[0], reg[1]
    b = b[b["_lv"] == "sido"].copy()
    b["_P"] = b["PRD_DE"].map(lambda v: _period(v, "M"))
    p = b.pivot_table(index=["_reg", "_P"], columns="_cat", values="DT", aggfunc="first")
    res["파일"] = os.path.relpath(files[0], BASE)
    if "계" not in p.columns or "아파트" not in p.columns:
        res["요약"] = f"보완 파일에 계·아파트 항목이 없음({list(p.columns)}) → 대조 불가"
        _ZERO_CACHE[vid] = res
        return res
    parts = [c for c in p.columns if c != "계"]
    full = p.dropna(subset=["계"] + parts)
    ident = float(((full["계"] - full[parts].sum(axis=1)) == 0).mean()) if len(full) else np.nan
    others = [c for c in parts if c != "아파트"]
    regs = [c for c in wide.columns if c in REGIONS]
    inner_gaps = inner_ok = outer_ok = jan_ok = 0
    unexplained, no_row = [], 0
    for c in regs:
        x = wide[c]
        if not x.notna().any():
            continue
        f, l = x.first_valid_index(), x.last_valid_index()
        inner_gaps += int(x.loc[f:l].isna().sum())
        for P in x.index[x.isna()]:
            if (c, P) not in p.index or pd.isna(p.loc[(c, P), "계"]):
                no_row += int(f <= P <= l)
                continue
            r = p.loc[(c, P)]
            zero = float(r["계"] - r[others].fillna(0).sum()) == 0.0
            inside = f <= P <= l
            if zero and inside:
                inner_ok += 1
                jan_ok += int(P.month == 1)
            elif zero:
                outer_ok += 1
            elif inside:
                unexplained.append(f"{c} {P}")
    ok = inner_gaps > 0 and inner_ok == inner_gaps and not unexplained
    res["ok"] = ok
    res["요약"] = (f"항등식 계 = {' + '.join(parts)} 성립 {ident:.1%} (시도, 항목 모두 있는 {len(full)}행); "
                 f"첫~마지막 관측 사이 빈 셀 {inner_gaps} 중 아파트 0 확인 {inner_ok}(1월 {jan_ok}, 그 외 {inner_ok - jan_ok})"
                 + (f", 미확인 {len(unexplained)}({', '.join(unexplained[:6])})" if unexplained else "")
                 + (f", 보완에도 행 없음 {no_row}" if no_row else "")
                 + (f"; 관측 범위 밖 0 확인 {outer_ok}(채우지 않음)" if outer_ok else ""))
    res["표"] = pd.DataFrame([{"항등식 성립 비율": round(ident, 4), "검사 행수": len(full), "첫~마지막 사이 빈 셀": inner_gaps, "아파트 0 확인": inner_ok,
                              "그중 1월": jan_ok, "미확인": len(unexplained), "보완에도 행 없음": no_row, "범위 밖 0 확인": outer_ok, "판정": "빈 셀 = 0 채움" if ok else "결측 유지"}])
    _ZERO_CACHE[vid] = res
    return res


def fill_gaps_zero(w):
    """각 열(지역)의 첫~마지막 관측 사이 빈 셀을 0 으로. 범위 밖(시작 전·종료 후)은 결측 유지"""
    out = w.copy()
    for c in out.columns:
        x = out[c]
        if x.notna().any():
            f, l = x.first_valid_index(), x.last_valid_index()
            out.loc[f:l, c] = x.loc[f:l].fillna(0.0)
    return out


def monthly_flow_from_cumulative(w):
    """KOSIS 연초누계(월별 누계) → 월값: 1월은 누계 그대로, 그 뒤는 전월 누계와의 차 (빈 셀은 fill_gaps_zero 로 먼저 0 처리)"""
    w = w.copy()
    out = w.copy()
    prev = w.shift(1)
    same_year = pd.Series(w.index.year, index=w.index) == pd.Series(w.index.year, index=w.index).shift(1)
    out[same_year.values] = (w - prev)[same_year.values]
    return out


# 변환 선호 순서(작을수록 원형에 가까움). 정상성을 통과한 후보 중 이 값이 가장 작은 것을 권고한다
PREF = {"수준": 0, "수준(저빈도 그대로)": 0, "수준−100": 0, "월값(누계 차분)": 0, "수준(0/±1)": 0, "수준_천명당": 0, "수준_천세대당": 0,
        "Δ12": 1, "100·로그Δ12": 1, "12개월합_천세대당": 1, "log1p(12개월합_천세대당)": 1, "12개월합 전년비(%)": 1, "log1p(천세대당 잔량)": 1, "12개월합_천명당": 1, "전기 대비 차분": 1, "전기 대비 로그변화(%)": 1,
        "Δ12_천세대당": 1, "Δ3": 2, "Δ3_천세대당": 2, "100·로그Δ3": 2, "3개월합_천명당": 2, "3개월합_천세대당": 2, "최근12개월 내 변경(상태화)": 2, "월값_천세대당": 3,
        "Δ1": 3, "100·로그Δ1": 3, "6개월 이동SD(월변화)": 3,
        "전국대비": 4, "시도평균대비": 4, "전국대비(저빈도)": 4,
        "추세제거(과거36개월 평균 대비)": 5, "가속도(12개월 변화의 12개월 차)": 6, "12개월 변화의 1개월 차": 6, "z(과거36개월 표준화)": 7}
PREF_DESC = {0: "원형", 1: "12개월 변화", 2: "3개월 변화", 3: "1개월 변화", 4: "횡단면 상대화", 5: "추세 제거", 6: "2차 차분", 7: "표준화"}


def _derived(out, base_name, g, nat_g=None):
    """기본 변화율 g(달력월 × 지역) 에서 2차 후보를 만든다: 상대화·추세 제거·가속도·표준화"""
    regs = [c for c in g.columns if c in REGIONS]
    if nat_g is not None:
        out[f"전국대비"] = g[regs].sub(nat_g, axis=0) if regs else g.sub(nat_g, axis=0)
    elif len(regs) > 1:
        out[f"시도평균대비"] = g[regs].sub(g[regs].mean(axis=1).where(g[regs].notna().all(axis=1)), axis=0)
    mean36 = g.rolling(36, min_periods=24).mean().shift(1)
    sd36 = g.rolling(36, min_periods=24).std().shift(1)
    out["추세제거(과거36개월 평균 대비)"] = g - mean36
    out["가속도(12개월 변화의 12개월 차)"] = g - g.shift(12)
    out["12개월 변화의 1개월 차"] = g - g.shift(1)
    out["z(과거36개월 표준화)"] = (g - mean36) / sd36
    out["_기준변화율"] = base_name


def candidates(vid, wide, kind, vtype, ctx):
    """후보 변환을 넓게 만든다. 유형별 기본 후보 + 기본 변화율에서 파생한 2차 후보(상대화·추세제거·가속도·표준화)"""
    out = {}
    if kind != "M":
        out["수준(저빈도 그대로)"] = wide
        out["전기 대비 차분"] = wide.diff()
        positive = bool(((wide > 0) | wide.isna()).all().all()) and wide.notna().any().any()
        if positive:
            out["전기 대비 로그변화(%)"] = 100 * (np.log(wide) - np.log(wide).shift(1))
        regs = [c for c in wide.columns if c in REGIONS]
        if "전국" in wide.columns and regs:
            out["전국대비(저빈도)"] = wide[regs].sub(wide["전국"], axis=0)
        return out
    x = wide
    nat = x["전국"] if "전국" in x.columns else None
    regs = [c for c in x.columns if c in REGIONS]
    if vid in ZERO_GAPS and zero_check(vid, x)["ok"]:
        x = fill_gaps_zero(x)                                   # KOSIS 0행 생략 → 보완 합계로 0 이 확인된 빈 셀을 0 으로
    if vid in CUMULATIVE:
        out["누계(raw 그대로)"] = x
        x = monthly_flow_from_cumulative(x)
        out["월값(누계 차분)"] = x
        nat = x["전국"] if "전국" in x.columns else None
    elif vtype == "확산지수":
        out["수준"] = x
        out["수준−100"] = x - 100
    else:
        out["수준"] = x
    out["Δ1"], out["Δ3"], out["Δ12"] = x - x.shift(1), x - x.shift(3), x - x.shift(12)
    positive = bool(((x > 0) | x.isna()).all().all()) and x.notna().any().any()
    base, base_name = out["Δ12"], "Δ12"
    if positive and vtype in ("지수·금액·수량", "재고", "흐름", "순이동"):
        lx = np.log(x.where(x > 0))
        out["100·로그Δ1"], out["100·로그Δ3"], out["100·로그Δ12"] = 100 * (lx - lx.shift(1)), 100 * (lx - lx.shift(3)), 100 * (lx - lx.shift(12))
        base, base_name = out["100·로그Δ12"], "100·로그Δ12"
    hh = ctx["hh"].reindex(x.index) if len(ctx["hh"]) else pd.DataFrame(index=x.index)
    ch = [c for c in x.columns if c in hh.columns]
    if vtype == "흐름" and ch:
        out["월값_천세대당"] = 1000 * x[ch] / hh[ch]
        out["3개월합_천세대당"] = 1000 * x[ch].rolling(3, min_periods=3).sum() / hh[ch]
        out["12개월합_천세대당"] = 1000 * x[ch].rolling(12, min_periods=12).sum() / hh[ch]
        out["log1p(12개월합_천세대당)"] = np.log1p(out["12개월합_천세대당"])      # 대형 사업 달의 극단값을 눌러 분포를 대칭에 가깝게
        s12 = x[ch].rolling(12, min_periods=12).sum()
        out["12개월합 전년비(%)"] = 100 * (s12 / s12.shift(12).where(s12.shift(12) > 0) - 1)   # YoY: 최근 12개월 합 ÷ 1년 전 12개월 합 − 1 (월값 Δ12 보다 매끈)
        base, base_name = out["12개월합_천세대당"], "12개월합_천세대당"
    if vtype == "재고" and ch:
        out["log1p(천세대당 잔량)"] = np.log1p(1000 * x[ch].clip(lower=0) / hh[ch])
        per = 1000 * x[ch] / hh[ch]
        out["수준_천세대당"] = per
        out["Δ3_천세대당"], out["Δ12_천세대당"] = per - per.shift(3), per - per.shift(12)     # 호 단위 차분은 지역 규모에 좌우되므로 세대당으로
        base, base_name = out["log1p(천세대당 잔량)"], "log1p(천세대당 잔량)"
    if vtype == "순이동" and len(ctx["pop"]):
        pop = ctx["pop"].reindex(x.index)
        cp = [c for c in x.columns if c in pop.columns]
        out["수준_천명당"] = 1000 * x[cp] / pop[cp]                       # 월 순이동률: 원값을 인구로만 나눔
        out["12개월합_천명당"] = 1000 * x[cp].rolling(12, min_periods=12).sum() / pop[cp]
        out["3개월합_천명당"] = 1000 * x[cp].rolling(3, min_periods=3).sum() / pop[cp]
        base, base_name = out["12개월합_천명당"], "12개월합_천명당"
        for k in ("100·로그Δ1", "100·로그Δ3", "100·로그Δ12"):
            out.pop(k, None)
    if vtype in ("비율·금리", "확산지수"):
        out["6개월 이동SD(월변화)"] = x.diff().rolling(6, min_periods=6).std()
    if vtype == "더미":
        out = {"수준(0/±1)": x, "최근12개월 내 변경(상태화)": x.rolling(12, min_periods=1).sum().clip(-1, 1)}
        return out
    nat_base = None
    if nat is not None and base_name in ("Δ12", "100·로그Δ12") and "전국" in base.columns:
        nat_base = base["전국"]
    _derived(out, base_name, base, nat_base)
    return out


def rank_candidates(st, tr=None):
    """정상성 표에 선호 순위·통과 여부·점수를 붙이고 순위대로 정렬. 통과 = 시도 다수 판정 I(0)"""
    st = st.copy()
    st["선호"] = st["변환"].map(lambda k: PREF.get(k, 8))
    st["변환 성격"] = st["선호"].map(PREF_DESC).fillna("기타")
    passed = st["다수판정"].astype(str).str.startswith("I(0)") & (st["변환"] != "누계(raw 그대로)")   # 연초 누계 톱니는 참고용이라 권고 대상에서 제외
    ar1 = pd.to_numeric(st["AR1중앙"], errors="coerce").fillna(0)
    st["정상성 통과"] = passed & (ar1 >= 0.3)                   # 통과했어도 AR(1) < 0.3 이면 잡음에 가까워 '유효 통과'로 보지 않음
    st["잡음 주의"] = passed & (ar1 < 0.3)
    adf = pd.to_numeric(st["ADF기각비율"], errors="coerce").fillna(0)
    kp = pd.to_numeric(st["KPSS기각비율"], errors="coerce").fillna(1)
    st["정상성 점수"] = (adf - kp).round(2)                      # 클수록 정상에 가까움 (−1 ~ +1)
    st = st.sort_values(["정상성 통과", "정상성 점수", "선호"], ascending=[False, False, True]).reset_index(drop=True)
    st.insert(0, "순위", range(1, len(st) + 1))
    return st


MIN_REGIONS = 15      # 시도 평균·월내 편차를 계산할 때 필요한 최소 시도 수 (세종 등 늦게 시작한 시도 때문에 17 전부를 요구하지 않음)


def _agg_series(w):
    """공통 성분: 전국 열이 있으면 전국, 아니면 15개 이상 관측된 달의 시도 평균"""
    if "전국" in w.columns:
        return w["전국"]
    if w.shape[1] == 1:
        return w.iloc[:, 0]
    return w.mean(axis=1).where(w.notna().sum(axis=1) >= min(MIN_REGIONS, w.shape[1]))


def _nanmedian(vals):
    """NaN 을 뺀 중앙값. 전부 NaN(또는 빈 목록)이면 NaN — np.nanmedian 의 All-NaN 경고를 피한다"""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    return round(float(np.median(v)), 3) if v.size else np.nan


def stationarity(cands, kind):
    rows = []
    for name, w in cands.items():
        if name.startswith("_") or not isinstance(w, pd.DataFrame):
            continue
        ws = w[[c for c in w.columns if c != "전국"]] if (w.shape[1] > 1 and "전국" in w.columns) else w
        if kind == "M":
            ws = ws[(ws.index >= WIN[0]) & (ws.index <= WIN[1])]
        res = [test_series(ws[c], za=False) for c in ws.columns]
        tested = [r for r in res if r["판정"] not in ("", "검정 생략")]
        agg = _agg_series(w).dropna()
        rec = {"변환": name, "검정 지역수": len(res), "검정 가능": len(tested),
               "ADF기각비율": round(float(np.mean([r["ADF_c_p"] < 0.05 for r in tested])), 2) if tested else np.nan,
               "KPSS기각비율": round(float(np.mean([r["KPSS_c_기각"] for r in tested])), 2) if tested else np.nan,
               "다수판정": (pd.Series([r["판정"] for r in tested]).mode().iloc[0] if tested else "생략: " + (res[0]["처리"][:28] if res else "")),
               "AR1중앙": _nanmedian([r.get("AR1", np.nan) for r in res]),      # 전부 결측인 지역은 AR1 키가 없음
               "평균": round(float(agg.mean()), 3) if len(agg) else np.nan, "SD": round(float(agg.std()), 3) if len(agg) > 1 else np.nan,
               "왜도": round(float(stats.skew(agg)), 2) if len(agg) > 3 else np.nan}
        if kind == "M" and len(agg) > 24:
            fw = agg[(agg.index >= ZWIN[0]) & (agg.index <= ZWIN[1])]
            if len(fw) > 6 and fw.std(ddof=0) > 0:
                z = (agg - fw.mean()) / fw.std(ddof=0)
                rec["|z|최대(2016~17 기준)"], rec["|z|최대 시점"], rec["|z|>5 비율"] = round(float(z.abs().max()), 1), str(z.abs().idxmax()), round(float((z.abs() > 5).mean()), 2)
        rows.append(rec)
    return pd.DataFrame(rows)


# ============================================================ 6 타깃 관계 (탐색창)
def _sp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return (float(stats.spearmanr(a[m], b[m])[0]), int(m.sum())) if m.sum() >= 12 else (np.nan, int(m.sum()))


def target_relation(cands, kind, lag, ctx):
    if kind != "M":
        return pd.DataFrame([{"비고": "저빈도 계열: 월로 펼치는 규칙(저빈도→월변환)을 정한 뒤 모형 단계에서 본다"}])
    tg = ctx["targets"]
    rows = []
    for name, w in cands.items():
        if name.startswith("_") or not isinstance(w, pd.DataFrame):
            continue
        known = w.shift(int(lag))                                  # 결정월 t 에 아는 값 = x(t − 시차)
        common = _agg_series(known)
        rec = {"변환": name}
        for h in (3, 6):
            g = tg[f"G{h}_nat"].reindex(common.index)
            ok = (common.index + h <= EXPLORE_END) & (common.index >= WIN[0])
            rho, n = _sp(common[ok].values, g[ok].values)
            rec[f"ρ공통_G{h}"], rec[f"n{h}"] = (round(rho, 2) if np.isfinite(rho) else np.nan), n
        g6 = tg["G6_nat"].reindex(common.index)
        ok6 = (common.index + 6 <= EXPLORE_END) & (common.index >= WIN[0])
        for k in (3, 6, 12):
            rho, _ = _sp(common.shift(k)[ok6].values, g6[ok6].values)
            rec[f"선행{k}개월 ρ_G6"] = round(rho, 2) if np.isfinite(rho) else np.nan
        regs = [c for c in known.columns if c in REGIONS or c in GU_LIST]
        if len(regs) > 1 and "r6_panel" in tg:
            dev = known[regs].sub(known[regs].mean(axis=1).where(known[regs].notna().sum(axis=1) >= min(MIN_REGIONS, len(regs))), axis=0)
            r6 = tg["r6_panel"].reindex(dev.index)
            a = dev.stack(future_stack=True).rename("x").reset_index()
            a.columns = ["P", "region", "x"]
            b = r6.stack(future_stack=True).rename("r6").reset_index()
            b.columns = ["P", "region", "r6"]
            m = a.merge(b, on=["P", "region"])
            m = m[(m["P"] + 6 <= EXPLORE_END) & (m["P"] >= WIN[0])]
            rho, n = _sp(m["x"].values, m["r6"].values)
            rec["ρ지역_r6(풀링)"], rec["n지역"] = (round(rho, 2) if np.isfinite(rho) else np.nan), n
            wm = []
            for _, g in m.groupby("P"):
                mm = g["x"].notna() & g["r6"].notna()
                if mm.sum() >= 10:
                    wm.append(stats.spearmanr(g.loc[mm, "x"], g.loc[mm, "r6"])[0])
            rec["월내순위ρ_r6 평균"] = round(float(np.nanmean(wm)), 2) if wm else np.nan
            rec["월내 양수비율"] = round(float(np.mean(np.array(wm) > 0)), 2) if wm else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# ============================================================ 7 권고
def recommend(vid, vtype, kind, st, tr, dinfo, cinfo, nat_only):
    """st 는 rank_candidates 를 거친 표. 권고 = 정상성을 통과한 후보 중 변환이 가장 적은 것(선호 순위). 통과 후보가 없으면 가장 정상에 가까운 것"""
    notes = []
    lag = cinfo.get("공표시차(개월)", "")
    role = "타깃" if vid == "V001" else ("가격추세" if vid in ("V002", "V026") else ("트리거" if vid in TRIGGER_IDS else "취약성"))
    if vid == "V001":
        rec = "타깃: G_h = 100·[R(t−1+h)/R(t−1) − 1]"
        why = ""
    elif kind != "M":
        rec = "수준(저빈도 그대로) — 공표 시 갱신하는 계단으로 월에 펼침"
        why = {"Y": "연간", "Q": "분기", "H": "반기"}.get(kind, "저빈도") + " 계열은 관측이 적어 월 단위 정상성 검정이 무의미. 변화율을 만들지 않음(제외 후보)"
    elif len(st):
        usable = st[~st["잡음 주의"]] if "잡음 주의" in st.columns else st
        passed = usable[usable["정상성 통과"]]
        if len(passed):
            top = passed.sort_values(["선호", "정상성 점수"], ascending=[True, False]).iloc[0]
            rec = top["변환"]
            why = (f"정상성 통과 {len(passed)}개 중 변환이 가장 적은 것 (성격: {top['변환 성격']}, ADF 기각 {top['ADF기각비율']}, KPSS 기각 {top['KPSS기각비율']})"
                   + (". 통과한 다른 후보: " + ", ".join(p for p in passed["변환"] if p != rec)[:120] if len(passed) > 1 else ""))
        else:
            top = usable.sort_values(["정상성 점수", "선호"], ascending=[False, True]).iloc[0] if len(usable) else st.iloc[0]
            rec = top["변환"]
            why = f"정상성 통과 후보 없음 → 가장 정상에 가까운 것 (점수 {top['정상성 점수']}, AR1 {top['AR1중앙']}). 느린 구조 변수로 두거나(비정상 기록) 제외를 검토"
        noisy = st[st["잡음 주의"]]["변환"].tolist() if "잡음 주의" in st.columns else []
        if noisy:
            notes.append("통과했으나 잡음(AR1<0.3)이라 제외: " + ", ".join(noisy)[:100])
        if vid in CUMULATIVE:
            notes.append("raw 는 연초 누계 → 빈 셀 0(보완 합계로 확인) → 월값(누계 차분) 뒤에 변환")
        elif vid in ZERO_GAPS:
            notes.append("raw 는 월값(누계 아님) → 빈 셀 0(보완 합계로 확인) 뒤에 변환")
    else:
        rec, why = "", ""
    notes.insert(0, why) if why else None
    dup = {"V033": "V005 와 같은 표(중복) → V005 로 대표", "V044": "V043 기준금리와 거의 같은 움직임 → 하나만", "V043": "V044 CD금리와 거의 같은 움직임 → 하나만", "V045": "V043·V044 와 같은 금리 묶음",
           "V069": "가계대출·주택관련대출 2계열 → 하나만", "V035": "V036 매매거래와 겹침(전체 거래 ⊃ 매매)", "V036": "V007 국토부 매매건수와 같은 현상",
           "V011": "하위 항목(20-29·30-39·20-39·비중) → 1개로", "V032": "V005 와 매우 유사", "V005": "V032 와 매우 유사",
           "V015": "V017 과 같은 표·같은 움직임", "V017": "V015 와 같은 표", "V061": "V062·V063 과 권역 3값", "V062": "권역 3값", "V063": "권역 3값"}
    if vid in dup:
        notes.append("중복: " + dup[vid])
    ended = {"V034": "2015.06 종료", "V077": "5560(2012~2019) 뒤에 6827 민간 등록 공급 전 유형 합(2020~)을 이어 붙인 연결 계열. 2019→2020 은 공공 포함→민간만 정의 차이로 하향 단절(전국 405,377→280,853)", "V074": "구계열 2019Q4 종료(신계열 보완 파일 별도)", "V024": "서울 구 전용(시도 없음)", "V076": "6827 전 유형 합이 V077 의 2020~ 구간으로 흡수됨(한 계열). 7174 는 2024 만(대조용: 민간 계 51,956 = 6827 합)"}
    if vid in ended:
        notes.append("기간·범위: " + ended[vid])
    if nat_only:
        notes.append("전국 값 하나 → 17개 시도에 복제됨(시도 간 변동 없음). 시점 효과와 구분 안 됨")
    if lag != "":
        notes.append(f"공표시차 {lag}개월 반영 필요")
    best = ""
    if len(tr) and "ρ공통_G6" in tr.columns:
        t = tr.dropna(subset=["ρ공통_G6"])
        if len(t):
            b = t.iloc[int(t["ρ공통_G6"].abs().argmax())]
            best = f"탐색창 |ρ| 최대 변환 {b['변환']} (ρ_G6 {b['ρ공통_G6']}, n {b['n6']})"
    return {"권고 변환(초안)": rec, "역할(초안)": role, "유형": vtype, "사전 권장 가공": dinfo.get("권장 가공/파생변수", ""), "사전 예상 부호": dinfo.get("예상 부호", ""),
            "탐색창 참고": best, "주의": " / ".join(notes)}


# ============================================================ 타깃(V001) 전용 절
def target_sections(ctx, parsed_list):
    tg = ctx["targets"]
    secs = []
    Rs = tg["R_panel"]
    t1 = pd.DataFrame({"첫 관측": [str(Rs[c].first_valid_index()) for c in Rs.columns], "마지막 관측": [str(Rs[c].last_valid_index()) for c in Rs.columns]}, index=Rs.columns)
    secs.append(("타깃 계열 범위 (raw V001, 시도별)", t1.reset_index().rename(columns={"index": "지역"}).to_html(index=False)))
    old = [p for p in parsed_list if "보완_V001" in p["info"]["파일"]]
    if old:
        w = list(old[0]["items"].values())[0]
        t2 = pd.DataFrame({"첫 관측": [str(w[c].first_valid_index()) for c in w.columns], "마지막 관측": [str(w[c].last_valid_index()) for c in w.columns]}, index=w.columns)
        secs.append(("참고 — 옛 월세가격지수(보완 파일, 2015-06 이전) 범위: 결정에 따라 사용/미사용", t2.reset_index().rename(columns={"index": "지역"}).to_html(index=False)))
    cands = {"R_last(전국 수준)": tg["R_last_nat"].to_frame("전국"), "G1(전국)": tg["G1_nat"].to_frame("전국"), "G3(전국)": tg["G3_nat"].to_frame("전국"), "G6(전국)": tg["G6_nat"].to_frame("전국")}
    for h in (3, 6):
        cands[f"G{h}(17시도)"] = tg[f"G{h}_panel"]
        cands[f"r{h}(17시도)"] = tg[f"r{h}_panel"]
    st = stationarity({k: v[(v.index >= pd.Period("2016-01", "M")) & (v.index <= pd.Period("2025-12", "M"))] for k, v in cands.items()}, "M")
    secs.append(("타깃 계열 정상성 (결정월 2016-01~2025-12; 전국은 1개 검정, 17시도는 시도별 기각 비율)", st.to_html(index=False)))
    rows = []
    for h in (3, 6):
        G = tg[f"G{h}_panel"]
        G = G[(G.index >= pd.Period("2016-01", "M")) & (G.index <= pd.Period("2025-12", "M"))]
        g = G.stack(future_stack=True).dropna()
        for a in (-0.02, -0.03, -0.04):
            thr = 100 * ((1 + a) ** (h / 12) - 1)
            hit = g <= thr
            rows.append({"h": h, "경계": f"급락 연율 {a:.0%} → G{h} ≤ {thr:.3f}%", "사건 행": int(hit.sum()), "비율": round(float(hit.mean()), 3), "사건 시도 수": int(g[hit].index.get_level_values(1).nunique())})
        for a in (0.03, 0.04):
            thr = 100 * ((1 + a) ** (h / 12) - 1)
            hit = g >= thr
            rows.append({"h": h, "경계": f"급등 연율 +{a:.0%} → G{h} ≥ {thr:.3f}%", "사건 행": int(hit.sum()), "비율": round(float(hit.mean()), 3), "사건 시도 수": int(g[hit].index.get_level_values(1).nunique())})
        hit = g <= g.mean() - g.std()
        rows.append({"h": h, "경계": f"G{h} ≤ 평균 − 1σ ({g.mean() - g.std():.3f}%)", "사건 행": int(hit.sum()), "비율": round(float(hit.mean()), 3), "사건 시도 수": int(g[hit].index.get_level_values(1).nunique())})
    path = pd.concat({h: 100 * (Rs.shift(1 - h) / Rs.shift(1) - 1) for h in range(1, 7)}, axis=1)
    U = path.T.groupby(level=1).max().T.where(path.notna().T.groupby(level=1).all().T)
    Dn = path.T.groupby(level=1).min().T.where(path.notna().T.groupby(level=1).all().T)
    for name, M, cond in (("급등: 앞으로 6개월 중 어느 달이든 R(t−1) 대비 +1%p 초과", U, lambda v: v > 1), ("급락: 앞으로 6개월 중 어느 달이든 R(t−1) 대비 −1%p 미만", Dn, lambda v: v < -1)):
        Mw = M[(M.index >= pd.Period("2016-01", "M")) & (M.index <= pd.Period("2025-12", "M"))].stack(future_stack=True).dropna()
        hit = cond(Mw)
        rows.append({"h": 6, "경계": name, "사건 행": int(hit.sum()), "비율": round(float(hit.mean()), 3), "사건 시도 수": int(Mw[hit].index.get_level_values(1).nunique())})
    secs.append(("경계 후보별 사건 수 (결정월 2016-01~2025-12, 17시도 패널 행 기준; 결정 아님, 참고)", pd.DataFrame(rows).to_html(index=False)))
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for c in Rs.columns:
        ax[0].plot(Rs.index.to_timestamp(), Rs[c], lw=0.6, alpha=0.5)
    ax[0].plot(Rs.index.to_timestamp(), tg["R_last_nat"].shift(-1), lw=2, color="k", label="전국")
    ax[0].set_title("V001 월세통합가격지수 수준 (raw; 17시도 얇은 선, 전국 굵은 선; 기준 2026-06=100)")
    ax[0].legend()
    g6 = tg["G6_nat"].dropna()
    ax[1].plot(g6.index.to_timestamp(), g6.values, color="k")
    ax[1].axhline(0, color="gray", lw=0.5)
    ax[1].axvspan(pd.Timestamp("2018-01-01"), pd.Timestamp("2025-12-31"), color="orange", alpha=0.08)
    ax[1].set_title("전국 G6 = 6개월 앞 변화율(%) (음영: 2018~2025)")
    secs.append(("그래프", _img(fig)))
    return secs


# ============================================================ html
def _img(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" style="max-width:100%">'


def _kv(d):
    return "<table class='kv'>" + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in d.items()) + "</table>"


def plot_levels(vid, items, kind):
    figs = []
    for key, w in list(items.items())[:3]:
        fig, ax = plt.subplots(figsize=(10, 3.4))
        xs = w.index.to_timestamp() if kind in ("M", "Q", "Y") else range(len(w))
        regs = [c for c in w.columns if c != "전국"]
        for c in regs:
            ax.plot(xs, w[c].values, lw=0.7 if len(regs) > 1 else 1.6, alpha=0.6 if len(regs) > 1 else 1, label=c if len(regs) <= 3 else None, drawstyle="steps-post" if kind != "M" else "default")
        if "전국" in w.columns:
            ax.plot(xs, w["전국"].values, lw=2, color="k", label="전국", drawstyle="steps-post" if kind != "M" else "default")
        elif len(regs) > 1:
            ax.plot(xs, w[regs].mean(axis=1).values, lw=2, color="k", label="시도 평균")
        ax.set_title(f"{vid} {key} — raw 수준 ({len(regs)}개 시도{' + 전국' if '전국' in w.columns else ''})")
        ax.legend(loc="best", fontsize=8)
        figs.append(_img(fig))
    return "".join(figs)


def plot_candidates(cands, kind):
    names = [k for k, v in cands.items() if not k.startswith("_") and isinstance(v, pd.DataFrame)]
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.2 * len(names)), sharex=(kind == "M"))
    for ax, name in zip(np.atleast_1d(axes), names):
        agg = _agg_series(cands[name])
        if kind == "M":
            agg = agg[(agg.index >= WIN[0]) & (agg.index <= WIN[1])]
            xs = agg.index.to_timestamp()
            ax.axvspan(pd.Timestamp("2016-01-01"), pd.Timestamp("2017-12-31"), color="green", alpha=0.08)
        else:
            xs = agg.index.to_timestamp() if hasattr(agg.index, "to_timestamp") else range(len(agg))
        ax.plot(xs, agg.values, color="k", lw=1.2, drawstyle="steps-post" if kind != "M" else "default")
        ax.axhline(0, color="gray", lw=0.4)
        ax.set_title(f"{name} (공통 성분: 전국 또는 시도 평균)", fontsize=9)
    return _img(fig)


def render(vid, parts):
    css = ("<style>body{font-family:'Malgun Gothic',sans-serif;font-size:13px;margin:24px;max-width:1200px}h1{font-size:20px}h2{font-size:15px;border-bottom:1px solid #999;margin-top:28px}"
           "table{border-collapse:collapse;margin:6px 0}td,th{border:1px solid #ccc;padding:3px 6px;font-size:12px;vertical-align:top}table.kv th{text-align:left;background:#f3f3f3;width:180px}"
           ".note{background:#fff8e1;padding:8px;border:1px solid #e0c060}</style>")
    keys = [t.split(" ")[0] for t, _ in parts]
    numbered, n, prev, k = [], 0, None, 0
    for (t, h), key in zip(parts, keys):
        base = re.sub(r"^[①-⑩]\s*", "", t)
        if key != prev:
            n, k, prev = n + 1, 0, key
        else:
            k += 1
        numbered.append((f"{n}{chr(ord('a') + k) if keys.count(key) > 1 else ''}. {base}", h))
    body = "".join(f"<h2>{t}</h2>{h}" for t, h in numbered)
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{vid} 변수 카드</title>{css}</head><body><h1>{vid} 변수 카드 (raw 원자료 기준)</h1>{body}</body></html>"


# ============================================================ 결정 로그
def log_load():
    if os.path.exists(LOG):
        df = pd.read_csv(LOG, encoding="utf-8-sig", dtype=str).fillna("")
        if set(df.columns) != set(LOG_COLS):
            df = df.rename(columns=OLD_COLS)
            for c in LOG_COLS:
                if c not in df.columns:
                    df[c] = ""
            df = df[LOG_COLS]
            df.to_csv(LOG, index=False, encoding="utf-8-sig")
        return df
    rows = [{"ID": aid, "변수명": name, "역할": "가정", "상태": "가정", "결정일": dt.date.today().isoformat(), "비고": txt} for aid, name, txt in ASSUMPTIONS]
    df = pd.DataFrame(rows).reindex(columns=LOG_COLS).fillna("")
    df.to_csv(LOG, index=False, encoding="utf-8-sig")
    return df


def log_upsert(row, decide=False):
    df = log_load()
    i = df.index[df["ID"] == row["ID"]]
    if len(i):
        i = i[0]
        if df.at[i, "상태"] not in ("", "초안") and not decide:      # 결정·보류·흡수 등 사람이 정한 상태는 카드 재생성이 덮어쓰지 않음
            return df
        for k, v in row.items():
            if v != "" or decide:
                df.at[i, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([{c: row.get(c, "") for c in LOG_COLS}])], ignore_index=True)
    df.to_csv(LOG, index=False, encoding="utf-8-sig")
    return df


# ============================================================ 모형 입력표 (xlsx, 수식 포함; analysis/model_sheet.py)
from model_sheet import SHEET_PATH, write_workbook  # noqa: E402


def apply_decision(vid, ctx):
    """결정 로그의 '결정' 행 전부를 raw 에 적용해 모형 입력표(xlsx)를 다시 쓴다. 반환: 요약 dict"""
    return write_workbook(ctx, log_load(), sys.modules[__name__])


# ============================================================ 카드 한 장
def build_card(vid, ctx):
    dinfo = dict_info(vid, ctx)
    cinfo = code_info(vid)
    name = dinfo.get("변수명") or cinfo.get("variables.py 이름") or vid
    parts = [("① 데이터사전", _kv(dinfo)), ("① variables.py · 공표시차", _kv(cinfo))]
    files = find_raw_files(vid)
    if vid in ("V006", "V007"):
        parts.append(("② raw 원자료", "<p class='note'>국토부 실거래 건별 원자료는 git 제외(수 GB)라 이 작업 폴더에 없음. 재수집: collect/collect_molit.py (API 키 필요)</p>"))
        return name, render(vid, parts), None
    if not files:
        parts.append(("② raw 원자료", "<p class='note'>raw 파일을 찾지 못함</p>"))
        return name, render(vid, parts), None
    parsed = [parse_raw_file(f, vid) for f in files]
    for p in parsed:
        parts.append((f"② raw 원자료 — {os.path.basename(p['info']['파일'])}", _kv(p["info"]) + p["sample"].to_html(index=False) + ("".join(f"<p class='note'>{n}</p>" for n in p["notes"]))))
    der = derived_raw(vid, parsed)
    if der is not None:                                      # raw 항목을 묶어 만드는 파생 원값(V011 비중, V077 연결 계열): 카드도 입력표와 같은 계열을 본다
        dw, dlabel, ddesc = der
        items = {dlabel: dw}
        parts.append(("③ 파생 원값 — 집계 규칙 (입력표도 같은 규칙으로 만듦)", f"<p class='note'>{ddesc}</p>"))
    else:
        items = select_items(vid, parsed)
    items_gu = select_items(vid, parsed, "gu")
    level = "sido"
    if items_gu and (not items or all(set(w.columns) <= {"서울", "전국"} for w in items.values())):
        # 서울 구 전용 변수(시도 자료 없음): 25개 구 패널로 본다. 타깃·세대수·인구도 구 패널로 바꿔 끼움
        items, level = items_gu, "gu"
        ctx = dict(ctx, targets={**ctx["targets"], **(ctx.get("targets_gu") or {})},      # G_nat 는 전국 그대로, 패널(G·r)은 구 기준
                   hh=ctx.get("hh_gu", ctx["hh"]), pop=ctx.get("pop_gu", ctx["pop"]))
        parts.append(("③ 지역 단위", "<p class='note'>시도 자료가 없는 서울 구 전용 변수 → 아래는 서울 25개 구 패널 기준 (시도 입력표에는 서울 행에만 값이 들어감)</p>"))
    if not items:
        parts.append(("③ raw → 시도×기간 표", "<p class='note'>시도 단위 계열을 만들지 못함</p>"))
        return name, render(vid, parts), None
    kind = items[list(items)[0]].attrs.get("kind", "M")      # 주 항목이 든 표의 주기
    cov = []
    for key, w in items.items():
        regs = [c for c in w.columns if c != "전국"]
        ww = w[(w.index >= WIN[0]) & (w.index <= WIN[1])] if kind == "M" else w
        cov.append({"항목": key, "기간": f"{w.index.min()}~{w.index.max()}", "시도 수": len(regs), "전국 열": "예" if "전국" in w.columns else "",
                    "결측률(2011-01~2025-12, 시도)": round(float(ww[regs].isna().mean().mean()), 3) if regs and len(ww) else np.nan,
                    "서울 구 수(raw)": len(set().union(*[p["gu"] for p in parsed]))})
    parts.append(("③ raw → 시도×기간 표 (전국·수도권 등 집계행 제외, 시도 17 + 전국)", pd.DataFrame(cov).to_html(index=False) + plot_levels(vid, items, kind)))
    miss_html, miss_prop = missing_section(vid, items, kind, parsed)
    parts.append(("④ 결측 진단·처리 제안 (raw 기준, 주 항목)", miss_html))
    vtype = vtype_of(vid)
    key0 = list(items)[0]
    wide = items[key0]
    nat_only = (wide.shape[1] == 1 and "전국" in wide.columns)
    cands = candidates(vid, wide, kind, vtype, ctx)
    st = rank_candidates(stationarity(cands, kind))
    title5 = ("⑤ 참고 — 타깃 자신의 과거 변화율(모멘텀) 정상성 (설명변수 past1·3·6 에 해당)" if vid == "V001" else
              f"⑤ 후보 변환별 정상성·분포 — 순위 = 정상성 통과 여부 → 변환이 적은 순 (항목 '{key0}', 유형 {vtype}; 2011-01~2025-12 시도별 ADF·KPSS; |z| 는 2016-01~2017-12 평균·SD 기준)")
    parts.append((title5, st.to_html(index=False) + plot_candidates(cands, kind)))
    lag = cinfo.get("공표시차(개월)", "")
    lag_n = int(lag) if str(lag).lstrip("-").isdigit() else 0
    tr = target_relation(cands, kind, max(lag_n, 0), ctx) if vid != "V001" else pd.DataFrame()
    if vid != "V001":
        parts.append((f"⑥ 타깃 관계 — 탐색창(정답 확인 ≤ 2017-12)만, 공표시차 {lag_n}개월 반영한 '결정월에 아는 값' 기준 (타깃은 raw V001 에서 계산)", tr.to_html(index=False)))
    if vid == "V001":
        parts += [("⑥ 타깃 — " + t, h) for t, h in target_sections(ctx, parsed)]
    rec = recommend(vid, vtype, kind, st, tr, dinfo, cinfo, nat_only)
    rec["결측 처리 제안(초안)"] = miss_prop
    parts.append(("⑦ 자동 권고 (규칙 초안 — 결정은 대화에서)", _kv(rec)))
    parts.append(("⑧ 결정란 (docs/전처리결정로그_10차.csv 의 행; --decide 로 갱신)", _kv({c: "" for c in LOG_COLS if c not in ("ID", "변수명", "동인")})))
    html = render(vid, parts)
    first = rec["권고 변환(초안)"]
    if first not in set(st["변환"]):
        first = first.split(" ")[0]
    sel = st[st["변환"] == first]
    rho = ""
    if len(tr) and "ρ공통_G6" in tr.columns:
        r0 = tr[tr["변환"] == first]
        rho = str(r0["ρ공통_G6"].iloc[0]) if len(r0) else ""
    row = {"ID": vid, "변수명": name, "동인": dinfo.get("1차 동인", ""), "역할": rec["역할(초안)"], "상태": "초안", "변환": rec["권고 변환(초안)"], "변환식": "",
           "공표시차_개월": str(lag), "추가시차": "", "지역단위": ("전국(시도 복제)" if nat_only else cinfo.get("지역단위", "")),
           "저빈도→월변환": ("계단(공표 시 갱신)" if kind != "M" else ""), "중복처리": "", "결측처리": miss_prop,
           "정상성판정": (str(sel["다수판정"].iloc[0]) if len(sel) else ""), "타깃상관(탐색창,G6)": rho, "결정일": "", "비고": rec["주의"]}
    return name, html, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--driver", type=int, default=None)
    ap.add_argument("--decide", default=None, help="ID 뒤에 키=값 … (변환·역할·변환식·결측처리·중복처리·저빈도→월변환·비고 등). 기록 뒤 입력표에 열을 넣는다")
    ap.add_argument("--apply", default=None, help="결정된 ID 의 열을 입력표에 (다시) 넣는다")
    ap.add_argument("--refresh-cache", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.decide:
        kv = {}
        for a in args.ids:
            k, _, v = a.partition("=")
            kv[k] = v
        kv.update({"ID": args.decide, "결정일": dt.date.today().isoformat()})
        kv.setdefault("상태", "결정")                          # '상태=보류' 를 주면 로그에만 남기고 입력표에는 넣지 않음
        df = log_upsert(kv, decide=True)
        print(df[df["ID"] == args.decide].T.to_string())
        args.apply = args.decide
    if args.apply:
        ctx = load_context(refresh=args.refresh_cache)
        summ = apply_decision(args.apply, ctx)
        print(f"{os.path.relpath(SHEET_PATH, BASE)} 갱신: {summ['행']}행, 결정월 {summ['결정월']} | 타깃 열 {summ['타깃 열']} | 설명변수 열 {summ['설명변수 열']}")
        return
    ctx = load_context(refresh=args.refresh_cache)
    ids = list(args.ids) + (DRIVER_IDS[args.driver] if args.driver else [])
    log_load()
    for vid in ids:
        name, html, row = build_card(vid, ctx)
        safe = re.sub(r"[\\/:*?\"<>|]", "_", str(name))[:40]
        path = os.path.join(OUT, f"{vid}_{safe}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        if row:
            log_upsert(row)
            print(f"{vid} {name}: 권고 {row['변환']} / {row['역할']} / 판정 {row['정상성판정']} / 타깃상관 {row['타깃상관(탐색창,G6)']} → {os.path.relpath(path, BASE)}")
        else:
            print(f"{vid} {name}: raw 없음 → {os.path.relpath(path, BASE)}")


if __name__ == "__main__":
    main()
