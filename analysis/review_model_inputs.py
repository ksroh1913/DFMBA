# -*- coding: utf-8 -*-
"""
[리뷰] 모형에 실제로 들어간 최종 데이터 표와 정상성 검정. docs/리뷰_9차_모형입력_학습방법.md 의 근거 산출물.

자료는 rmpi/run_v9_stage6.py 와 같은 순서로 만든다: data.build(마스킹 → 공표 시차 → 변환 → 기준 지역 분해) → targets → frames.
그 두 표(전국 nf, 패널 pf)가 파이프라인에 들어가는 '모형 입력표' 이고, 파이프라인 features 단계(RMPI 변환기)가 훈련행으로 만든
동인 지수가 Ridge 가 실제로 보는 '설계행렬' 이다. 둘 다 값 그대로 저장하고, 열마다 ADF·KPSS 로 정상성을 본다. 예측 성능은 계산하지 않는다.

출력 (analysis/output/)
  review_모형입력자료.xlsx
    1_열목록         nf·pf 의 모든 열: 역할, 쓰는 모형, 원변수·출처·빈도·공표시차·변환식, 관측 범위, 정상성 판정
    2_전국_입력표    nf 값 (결정월 2016-01 ~ 사후 확인 끝)
    3_패널_입력표    pf 값 (17개 시도 x 결정월)
    4a_설계행렬_전국 전국 B 의 features 단계 출력 (h=1·6, 적합 시점 2018-01·2025-12)
    4b_설계행렬_패널 패널 B(통합 G, h=6, 적합 시점 2025-12) 의 features 단계 출력
    4c_구성명세      위 적합들의 RMPI 구성 (유지·부호·가중)
    5_정상성_전국    nf 수치 열과 전국 동인 지수의 ADF·KPSS (모형창 2016-01~2025-12 와 전체 구간)
    6_정상성_패널    pf 지역 열의 시도별 ADF·KPSS 요약 (기각 비율, Fisher·Stouffer 결합)
    7_요약           판정 집계, 수준형 입력 플래그, 열 집계, 전처리 규칙, 자가 점검, manifest
  review_전국_입력표.csv, review_패널_입력표.csv   (2·3 시트의 CSV 사본)
사용: PYTHONUTF8=1 python analysis/review_model_inputs.py [--cache <pickle 경로>]   (cache 는 개발용: 세 시트 읽기를 건너뛴다)
"""

import argparse
import os
import pickle
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from rmpi import data as D  # noqa: E402
from rmpi import frames as F  # noqa: E402
from rmpi import models as M  # noqa: E402
from rmpi import settings as S  # noqa: E402
from rmpi import targets as T  # noqa: E402
from rmpi.run_stage2_data import EXCLUDE_REASON  # noqa: E402
from variables import BY_ID, RELEASE, RELEASE_MONTHLY  # noqa: E402

warnings.simplefilter("ignore", InterpolationWarning)
warnings.simplefilter("ignore", RuntimeWarning)
OUT = os.path.join(BASE, "analysis", "output")
XLSX = os.path.join(OUT, "review_모형입력자료.xlsx")
ALPHA = 0.05
MIN_N = 40

# 입력 ID -> 데이터사전 Master ID (variables.BY_ID 키)
ALIAS = {"V011a": "V011", "V011b": "V011", "V011c": "V011", "CSI": "V061", "RT_mix": "V006", "RT_act": "V006"}
# 변환 출력 접미사 -> 식 (data.transform 의 계산 그대로)
FORMULA = {"수준": "x", "Δ12": "x − x[−12]", "로그Δ12": "100·(ln x − ln x[−12])",
           "12개월합_천세대당": "1000·Σ(최근 12개월 x) / 세대수(V010)", "잔량_천세대당_log1p": "ln(1 + 1000·x / 세대수(V010))",
           "12개월합_천명당": "1000·Σ(최근 12개월 x) / 인구(V009)", "6개월%": "100·(x / x[−6] − 1)"}
RT_FORMULA = {"RT_mix|Δ12": "100·Δ12[ ln(월세 3개월합 / (전체 3개월합 − 월세 3개월합)) ]  (V006, 신고제 경계 14행 결측)",
              "RT_act|Δ12": "100·Δ12[ ln(1000·전체 3개월합 / 세대수(V010)) ]  (V006, 신고제 경계 14행 결측)"}
# 읽기용 열 이름 'ID_변수명_전처리' 에 쓰는 전처리 이름과 변수명 보정
TR_KO = {"수준": "수준", "Δ12": "12개월차분", "로그Δ12": "12개월로그변화율(%)", "12개월합_천세대당": "12개월합_천세대당",
         "잔량_천세대당_log1p": "천세대당잔량_log1p", "12개월합_천명당": "12개월순이동_천명당", "6개월%": "6개월변화율(%)"}
NAME_FIX = {"V011a": "주민등록인구 20-29세", "V011b": "주민등록인구 30-39세", "V011c": "20-39세 인구비중", "CSI": "주택가격전망CSI(지역유형별)",
            "RT_mix": "전월세 실거래 월세비중로짓 3개월합", "RT_act": "전월세 실거래 천세대당건수로그 3개월합"}
ID_TXT = {"CSI": "V061-63", "RT_mix": "V006", "RT_act": "V006"}


def readable_name(col, table, sp, s):
    """열 이름 -> 'ID_변수명_전처리 (원래 열 이름)'. 패널 표의 C|·D| 접두사는 '공통_'·'지역편차_' 로 옮긴다."""
    pre, base = "", col
    for p in ("C|", "D|"):
        if col.startswith(p):
            pre, base = p, col[len(p):]
    rent = "V001_아파트 월세통합가격지수"
    fixed = {"region": "시도", "P": "결정월", "R_last": f"{rent}_수준 R(t-1)"}
    if col in fixed:
        return f"{fixed[col]} ({col})"
    m = re.match(r"^(G|Gbar|Gnat|r|g|down|up|state_down|state_up|label_month)(\d)$", col)
    if m:
        k, h = m.group(1), m.group(2)
        txt = {"G": f"{h}개월앞변화율(%) 타깃", "Gbar": f"{h}개월앞변화율 17시도평균", "Gnat": f"{h}개월앞변화율 전국(부호결정용)",
               "r": f"{h}개월앞상대변화율 타깃", "g": f"{h}개월앞변화율 R(t)기준(민감도)", "down": f"{h}개월 급락여부(0/1)",
               "up": f"{h}개월 급등여부(0/1)", "state_down": f"기존급락상태 {h}개월", "state_up": f"기존급등상태 {h}개월"}.get(k)
        return f"정답확인월 h={h} ({col})" if k == "label_month" else f"{rent}_{txt} ({col})"
    m = re.match(r"^(rel_)?past(\d)$", col)
    if m:
        return f"{rent}_과거{m.group(2)}개월변화율(%){'_17시도평균대비' if m.group(1) else ''} ({col})"
    if base in sp.index:
        r = sp.loc[base]
        vid, blk, suf = r["ID"], r["block"], base.split("|", 1)[1]
        mid = ALIAS.get(vid, vid)
        nm = NAME_FIX.get(vid) or BY_ID.get(mid, {}).get("name", vid)
        core = f"{ID_TXT.get(vid, vid)}_{nm}_{TR_KO.get(suf, suf)}"
        if blk == "price_trend":
            return f"{core} ({col})"
        if pre == "D|":
            return f"지역편차_{core}_시도-기준지역평균 ({col})" if blk == "regional" else f"지역편차_{core} ({col})"
        tail = "_기준지역평균" if blk == "regional" else "_전국"
        return f"{'공통_' if pre == 'C|' else ''}{core}{tail} ({col})"
    if base.endswith("|월내편차"):
        inner = base[:-len("|월내편차")]
        return readable_name(inner, table, sp, s).replace(f" ({inner})", "") + f"_17시도평균대비 ({col})"
    m = re.match(r"^(RMPI공통|RMPI지역)\|([①-⑥])(\|결측)?$", col)
    if m:
        dn = {v.split(" ")[0]: v.replace(" ", "") for v in s["driver_names"].values()}.get(m.group(2), m.group(2))
        return f"{m.group(1)}_{dn}_동인지수{'_결측표시' if m.group(3) else ''} ({col})"
    return col


def rename_cols(df, table, sp, s):
    return df.rename(columns={c: readable_name(c, table, sp, s) for c in df.columns})


# ============================================================ 자료 (run_v9_stage6.py 와 같은 순서)
def prepare(s, cache=None):
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    b = D.build(s, "sido")
    pt = T.regional(s, b["ml"], "sido")
    nat = T.national(s)
    nf = F.national_frame(s, b, nat).join(T.gbar_frame(pt, s), how="left")
    pf = F.panel_frame(s, b, pt, nat)
    out = dict(b=b, pt=pt, nat=nat, nf=nf, pf=pf)
    if cache:
        with open(cache, "wb") as f:
            pickle.dump(out, f)
    return out


# ============================================================ 열 역할·메타데이터 (1_열목록)
def _release(vid):
    r = RELEASE_MONTHLY.get(vid) or RELEASE.get(vid) or {}
    return r.get("lag", np.nan), r.get("dict", ""), r.get("verdict", "")


def meta_of(vid):
    """입력 ID -> 데이터사전 메타데이터 (variables.py)"""
    mid = ALIAS.get(vid, vid)
    v = BY_ID.get(mid, {})
    lag, dict_txt, verdict = _release(mid)
    return {"MasterID": mid, "변수명": v.get("name", ""), "빈도": v.get("freq", ""), "지역단위": v.get("level", ""),
            "출처기관": v.get("source", ""), "API표ID": v.get("table", ""), "전처리스크립트": v.get("prep", ""),
            "공표시차_개월": lag, "공표근거": f"{dict_txt} ({verdict})" if dict_txt else ""}


def _wb_sheet(path, prefix):
    """전처리 워크북에서 이름이 prefix 로 시작하는 시트 (2_데이터목록, 4_공표시점규칙)"""
    x = pd.ExcelFile(path)
    name = next((n for n in x.sheet_names if n.startswith(prefix)), None)
    return x.parse(name) if name else pd.DataFrame()


def _col(df, *subs):
    for c in df.columns:
        if all(s_ in str(c) for s_ in subs):
            return c
    return None


def column_table(s, spec, nf, pf, lo, hi):
    """nf·pf 의 모든 열에 역할·사용 모형·원변수·출처·변환식·관측 범위를 붙인다."""
    sp = spec.set_index("입력")
    incommon = dict(zip(spec["입력"], M._in_common(spec)))
    trend_ex = set(s["rmpi"].get("trend_exclusion", {}).get("applied", []))
    lag_adj = s.get("lag_adjustments", {})
    rev = set(s["revision_risk"]["columns"])
    mask_cols = set(s["masking"]["columns"])
    hs = s["timing"]["horizons"]
    wb = S.path(s, "preprocessed_xlsx")
    lst = _wb_sheet(wb, "2_")       # 2_데이터목록: 컬럼명 기준
    rule = _wb_sheet(wb, "4_")      # 4_공표시점규칙: Master ID 기준
    lst_col = _col(lst, "컬럼") or _col(lst, "열")
    lst = lst.set_index(lst_col) if lst_col else lst
    rule_id = _col(rule, "Master")
    rule = rule.set_index(rule_id) if rule_id else rule

    def from_list(raw, *subs):
        if raw in lst.index:
            c = _col(lst, *subs)
            if c:
                return lst.loc[raw, c] if not isinstance(lst.loc[raw], pd.DataFrame) else lst.loc[raw, c].iloc[0]
        return ""

    def from_rule(mid, *subs):
        if mid in rule.index:
            c = _col(rule, *subs)
            if c:
                return rule.loc[mid, c] if not isinstance(rule.loc[mid], pd.DataFrame) else rule.loc[mid, c].iloc[0]
        return ""

    def spec_input(name):
        """'C|V003|Δ12' / 'D|...' / 'V003|Δ12' -> (접두, spec 입력명)"""
        for p in ("C|", "D|"):
            if name.startswith(p):
                return p, name[len(p):]
        return "", name

    def describe(col, table):
        rec = {"열이름": col, "표": table}
        pre, base = spec_input(col)
        m = re.match(r"^(G|Gbar|Gnat|r|g|down|up|state_down|state_up|label_month)(\d)$", col)
        if col in ("region", "P", "panel", "region_code"):
            rec.update(역할="식별", 모형사용="—", 원변수ID="")
        elif col == "R_last":
            rec.update(역할="타깃 원계열", 모형사용="직접 사용 안 함 (G·past 계산용 R(t−1))", 원변수ID="V001", 변환식="R(t−1): 결정월 말에 관측된 마지막 월세지수")
        elif m:
            k, h = m.group(1), int(m.group(2))
            if k == "G":
                rec.update(역할="타깃", 원변수ID="V001", 변환식=f"100·[R(t−1+{h}) / R(t−1) − 1]",
                           모형사용=("전국 타깃 (A·B·C Ridge; M0~M3 는 G − mom1 학습)" if table == "전국" else "통합 패널 타깃 (M0~M3·패널 B); r·사건 계산 원천"))
            elif k == "Gbar":
                rec.update(역할="타깃 보조", 원변수ID="V001", 변환식=f"17개 시도 G{h} 단순평균 (모두 있는 달만)", 모형사용="8차 결합 예측용. 9차 주 분석 미사용")
            elif k == "Gnat":
                rec.update(역할="부호 결정용", 원변수ID="V001", 변환식=f"전국 G{h} (R-ONE 전국 지수)", 모형사용="공통 RMPI ± 부호 결정 (SIGN_C). 출력에서 제외")
            elif k == "r":
                rec.update(역할="타깃", 원변수ID="V001", 변환식=f"G{h} − Gbar{h}", 모형사용="패널 r 타깃 (RQ2) + 지역 RMPI ± 부호 결정 (SIGN_R)")
            elif k == "g":
                rec.update(역할="타깃(민감도)", 원변수ID="V001", 변환식=f"100·[R(t+{h}) / R(t) − 1]", 모형사용="민감도 타깃. 주 분석 미사용")
            elif k == "down":
                rec.update(역할="타깃(사건)", 원변수ID="V001", 변환식=f"G{h} ≤ 100·[0.98^({h}/12) − 1]", 모형사용=("전국 급락 표시 (참고)" if table == "전국" else "급락 로지스틱 타깃 (RQ3·경보)"))
            elif k == "up":
                rec.update(역할="타깃(사건)", 원변수ID="V001", 변환식=f"G{h} ≥ 100·[1.03^({h}/12) − 1]", 모형사용="급등 보조 과제 타깃 (run_v9_up)")
            elif k.startswith("state"):
                rec.update(역할="상태", 원변수ID="V001", 변환식=f"직전 {h}개월 변화가 사건 경계 밖/안", 모형사용="경보 모집단 판정 (기존 상태 행 제외)")
            else:
                rec.update(역할="정답 시점", 원변수ID="", 변환식=f"t + {h}", 모형사용="식별")
        elif re.match(r"^past\d$", col):
            k = int(col[-1])
            rec.update(역할="A 가격추세", 원변수ID="V001", 변환식=f"100·[R(t−1) / R(t−1−{k}) − 1]",
                       모형사용=("A 정보군 (A·B·C 공통); mom1 = h × past1 (M0~M3 기준)" if table == "전국" else "패널 G·down A_REG; 통합 패널 mom1 = h × past1"))
        elif re.match(r"^rel_past\d$", col):
            rec.update(역할="A 가격추세(상대)", 원변수ID="V001", 변환식=f"{col[4:]} − 17개 시도 평균 (모두 있는 달만)", 모형사용="패널 r A_REL (RQ2)")
        elif base in ("V002|6개월%", "V026|6개월%") or base.endswith("|월내편차"):
            vid = base.split("|")[0]
            rec.update(역할="A 가격추세", 원변수ID=vid, 변환식=("100·(x / x[−6] − 1)" + (" − 17개 시도 평균" if base.endswith("월내편차") else "")),
                       모형사용=("패널 r A_REL" if base.endswith("월내편차") else ("A 정보군; M2·M3 가격추세 입력" if table == "전국" else "패널 G·down A_REG; 통합 패널 M2·M3")))
        elif base in sp.index:
            r = sp.loc[base]
            vid, blk = r["ID"], r["block"]
            suf = base.split("|", 1)[1]
            f0 = RT_FORMULA.get(base, FORMULA.get(suf, r["변환"]))
            ic = bool(incommon.get(base, True))
            dmark = {1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤", 6: "⑥"}.get(int(r["동인"]), "")
            if table == "전국" or pre == "C|":
                if blk == "common":
                    rec.update(역할="공통 블록(전국 공통)", 변환식=f0 + " (전국 값, 지역 공통)")
                else:
                    rec.update(역할="공통 블록(기준지역 평균)" if ic else "공통 블록(모형 미사용)", 변환식=f0 + " → 기준지역 월별 평균")
                if not ic:
                    rec["모형사용"] = "모형 미사용 (표에만 존재; in_common=false, 지역 블록 전용 출력)"
                elif table == "전국":
                    rec["모형사용"] = f"B: RMPI공통 동인 {dmark}; C: 개별 입력; M2·M3: RMPI공통"
                else:
                    rec["모형사용"] = f"패널 G·down B·Bc: RMPI공통 동인 {dmark}; C: 개별; 통합 패널 M2·M3: RMPI공통. r 에는 미사용"
            else:   # D|
                if blk == "extra":
                    rec.update(역할="별도 입력(실거래)", 변환식=f0 + " → 월내 편차", 모형사용="M3; 패널 B+실거래 (r·down); 급등 B+실거래. RMPI 밖")
                else:
                    rec.update(역할="지역 블록(월내 편차)", 변환식=f0 + " → 시도값 − 기준지역 평균",
                               모형사용=f"패널 r·G·down B: RMPI지역 동인 {dmark}; C: 개별; 통합 패널 M2·M3: RMPI지역")
            rec.update(원변수ID=vid, 원열=r["원열"], 변환=r["변환"], block=blk, 동인=f"{dmark} {s['driver_names'][int(r['동인'])]}",
                       하위묶음=r["하위묶음"], concept=r["concept"], 예상부호=r["예상부호"], in_common=ic,
                       추세제외=("예" if base in trend_ex else ""), 개정위험=("예" if vid in rev else ""), 보완셀마스킹=("예" if vid in mask_cols else ""),
                       추가시차=lag_adj.get(vid, ""))
            raw = r["원열"]
            rec.update(주기_워크북=from_list(raw, "주기"), 결측보완_워크북=from_list(raw, "결측"), 보완셀수_워크북=from_list(raw, "보완 셀"),
                       공표시기_워크북=from_list(raw, "공표"), 이차시트규칙_워크북=from_list(raw, "2차"))
            mid = ALIAS.get(vid, vid)
            rec.update(공표시점규칙_개월=from_rule(mid, "개월"), 공표시점규칙_적용=from_rule(mid, "적용"))
        else:
            rec.update(역할="기타", 모형사용="", 원변수ID="")
        vid = rec.get("원변수ID", "")
        if vid:
            rec.update(meta_of(vid))
        return rec

    rows = []
    for c in nf.columns:
        rows.append(describe(c, "전국"))
    for c in pf.columns:
        rows.append(describe(c, "패널"))
    tab = pd.DataFrame(rows)
    # 관측 범위·결측률 (모형창)
    nf_w = nf.loc[lo:hi]
    pf_w = pf[(pf["P"] >= lo) & (pf["P"] <= hi)]
    first, miss = [], []
    for _, r in tab.iterrows():
        src = nf_w if r["표"] == "전국" else pf_w
        col = r["열이름"]
        x = src[col] if col in src.columns else pd.Series(dtype=float)
        if r["표"] == "전국":
            fv = nf.loc[nf[col].notna()].index.min() if col in nf.columns and nf[col].notna().any() else None
        else:
            fv = pf.loc[pf[col].notna(), "P"].min() if col in pf.columns and pf[col].notna().any() else None
        first.append(str(fv) if fv is not None and not pd.isna(fv) else "")
        miss.append(round(float(x.isna().mean()), 3) if len(x) and pd.api.types.is_numeric_dtype(x) else np.nan)
    tab["첫관측_결정월"] = first
    tab["결측률_모형창"] = miss
    return tab


# ============================================================ 설계행렬 (파이프라인 features 단계 출력)
def design_cases(s, spec, nf, pf):
    """engine.national_run / panel_run 과 같은 훈련행·변환기로 적합한 뒤 features 단계 출력을 돌려준다."""
    first = S.per(s["timing"]["official_first_decision"])
    sp = spec
    ccols = [c for c in sp.loc[sp.block.isin(["regional", "common"]) & M._in_common(sp), "입력"] if c in nf.columns]
    nat_rows, comps = [], []
    for h in (1, 6):
        for Tm in ("2018-01", "2025-12"):
            Tp = pd.Period(Tm, "M")
            ycol = f"G{h}"
            frame = nf.copy()
            frame[F.SIGN_C] = frame[ycol]
            A = list(F.A_NAT)
            tr_mask = (frame.index >= first) & (frame.index <= Tp - h) & frame[ycol].notna() & frame[A].notna().all(axis=1)   # engine.national_run L95
            tr = frame[tr_mask]
            Xtr, ytr = tr.drop(columns=[ycol]), tr[ycol]
            pipe = M.ridge(s, M.features(s, sp, A, common_cols=ccols), "national")
            fitted = clone(pipe).fit(Xtr, ytr)
            Z = fitted.named_steps["features"].transform(Xtr)
            Z.insert(0, "y(G)", ytr.values)
            Z.insert(0, "P", [str(p) for p in Xtr.index])
            Z.insert(0, "사례", f"전국 B h={h} 적합시점 {Tm} (훈련행 {len(tr)})")
            nat_rows.append(Z)
            c = M.compositions(fitted)
            c.insert(0, "사례", f"전국 B h={h} {Tm}")
            comps.append(c)
    Znat = pd.concat(nat_rows, ignore_index=True)
    # 패널 B, 통합 패널 G, h=6, 2025-12 (engine.panel_run L142~181)
    h, Tp = 6, pd.Period("2025-12", "M")
    dcols = [f"D|{c}" for c in sp.loc[sp.block == "regional", "입력"] if f"D|{c}" in pf.columns]
    pcc = [f"C|{c}" for c in sp.loc[sp.block.isin(["regional", "common"]) & M._in_common(sp), "입력"] if f"C|{c}" in pf.columns]
    frame = pf.copy()
    frame[F.SIGN_R] = frame[f"r{h}"]
    frame[F.SIGN_C] = frame[f"Gnat{h}"]
    ycol = f"G{h}"
    A = list(F.A_REG)
    tr = frame[(frame.P >= first) & (frame.P <= Tp - h) & frame[ycol].notna() & frame[A].notna().all(axis=1)]
    Xtr, ytr = tr.drop(columns=[ycol]), tr[ycol]
    pipe = M.ridge(s, M.features(s, sp, A, common_cols=pcc, regional_cols=dcols, prefix_c="C|", prefix_d="D|"), "panel")
    fitted = clone(pipe).fit(Xtr, ytr)
    Zp = fitted.named_steps["features"].transform(Xtr)
    Zp.insert(0, "y(G)", ytr.values)
    Zp.insert(0, "P", [str(p) for p in Xtr["P"]])
    Zp.insert(0, "region", Xtr["region"].values)
    Zp.insert(0, "사례", f"패널 B 통합 G h={h} 적합시점 2025-12 (훈련행 {len(tr)})")
    c = M.compositions(fitted)
    c.insert(0, "사례", "패널 B h=6 2025-12")
    comps.append(c)
    return Znat, Zp, pd.concat(comps, ignore_index=True)


# ============================================================ 정상성 검정
def _longest_run(x):
    """가장 긴 연속 비결측 구간"""
    x = pd.Series(x).astype(float)
    ok = x.notna().values
    best, cur, start, bstart = 0, 0, 0, 0
    for i, o in enumerate(ok):
        if o:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, bstart = cur, start
        else:
            cur = 0
    return x.iloc[bstart:bstart + best] if best else x.iloc[0:0]


def _step(x):
    """연속 동일값 최빈 런 길이 k 와 Δ²x = 0 비율 (계단 계열 탐지)"""
    v = x.values
    runs, cur = [], 1
    for i in range(1, len(v)):
        if v[i] == v[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    k = int(pd.Series(runs).mode().iloc[0]) if runs else 1
    d2 = np.diff(v, n=2)
    return k, (float(np.mean(np.isclose(d2, 0))) if len(d2) else 0.0)


def _ar1(v):
    if len(v) < 3 or np.std(v) == 0:
        return np.nan, np.nan
    rho = float(np.corrcoef(v[1:], v[:-1])[0, 1])
    hl = float(np.log(0.5) / np.log(rho)) if 0 < rho < 1 else np.nan
    return rho, hl


def _verdict(adf_p, kpss_rej):
    adf_rej = bool(adf_p < ALPHA)
    if adf_rej and not kpss_rej:
        return "I(0)"
    if (not adf_rej) and kpss_rej:
        return "I(1)"
    if adf_rej and kpss_rej:
        return "불확정(구조변화·강한 지속성)"
    return "불확정(검정력 부족)"


def test_series(x, za=True, maxlag=12):
    """주 사양: ADF(c, maxlag 12, AIC) + KPSS(c, auto, 5% 임계값).  강건성: ct, KPSS legacy, Zivot-Andrews(주 사양 I(0) 아닐 때).
    건너뛰기: 상수 / 계단(12개월) / 표본 부족(n<40). 분기 계단(k=3)은 분기 부표본(maxlag 4, n>=40)."""
    r = {"n": 0, "시작": "", "끝": "", "처리": "", "판정": ""}
    x = _longest_run(x)
    r["n"] = int(len(x))
    if len(x) == 0:
        r.update(처리="전부 결측", 판정="검정 생략")
        return r
    r["시작"], r["끝"] = str(x.index[0]), str(x.index[-1])
    v = x.values.astype(float)
    r["평균"], r["표준편차"] = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
    r["AR1"], r["반감기_개월"] = _ar1(v)
    if x.nunique() <= 1:
        r.update(처리="상수", 판정="검정 생략")
        return r
    k, share0 = _step(x)
    r["연속동일값_런"], r["Δ²x=0 비율"] = k, round(share0, 2)
    if k >= 3 and share0 > 0.5:
        if k >= 12:
            r.update(처리=f"계단({k}개월) — 월별 검정 생략(연간 ~{len(v) // k}개 관측)", 판정="검정 생략")
            return r
        v = x[x != x.shift()].values.astype(float)       # 분기 부표본 (값이 바뀌는 시점만)
        r["처리"] = f"계단({k}개월) — 분기 부표본 n={len(v)}, maxlag 4"
        maxlag = 4
    if len(v) < MIN_N:
        r.update(처리=(r["처리"] + "; " if r["처리"] else "") + f"표본 부족(n={len(v)}<{MIN_N}) — AR(1)만", 판정="검정 생략")
        return r
    ml_ = min(maxlag, max(1, len(v) // 5))
    try:
        a = adfuller(v, maxlag=ml_, regression="c", autolag="AIC")
        r.update(ADF_c_stat=float(a[0]), ADF_c_p=float(a[1]), ADF_lag=int(a[2]))
        kp = kpss(v, regression="c", nlags="auto")
        r.update(KPSS_c_stat=float(kp[0]), KPSS_c_cv5=float(kp[3]["5%"]), KPSS_c_기각=bool(kp[0] > kp[3]["5%"]), KPSS_lag=int(kp[2]))
        r["판정"] = _verdict(r["ADF_c_p"], r["KPSS_c_기각"])
        # 강건성
        a2 = adfuller(v, maxlag=ml_, regression="ct", autolag="AIC")
        kp2 = kpss(v, regression="ct", nlags="auto")
        kp3 = kpss(v, regression="c", nlags="legacy")
        r.update(ADF_ct_p=float(a2[1]), KPSS_ct_기각=bool(kp2[0] > kp2[3]["5%"]), KPSS_c_legacy_기각=bool(kp3[0] > kp3[3]["5%"]),
                 판정_ct=_verdict(float(a2[1]), bool(kp2[0] > kp2[3]["5%"])), 판정_legacy=_verdict(r["ADF_c_p"], bool(kp3[0] > kp3[3]["5%"])))
        if za and r["판정"] != "I(0)":
            try:
                z = zivot_andrews(v, trim=0.15, maxlag=ml_, regression="c", autolag="AIC")
                bp = int(z[4])
                r.update(ZA_stat=float(z[0]), ZA_p=float(z[1]), ZA_구조변화=str(x.index[bp]) if len(x) == len(v) and bp < len(x) else str(bp))
            except Exception as e:   # 상수·짧은 계열에서 예외
                r["ZA_구조변화"] = f"계산 불가({type(e).__name__})"
    except Exception as e:
        r.update(처리=f"검정 오류({type(e).__name__}: {e})", 판정="검정 생략")
    return r


def national_tests(nf, cols, windows, roles):
    rows = []
    for c in cols:
        for wname, (lo, hi) in windows.items():
            x = nf.loc[lo:hi, c] if (lo is not None) else nf[c]
            rec = {"열이름": c, "역할": roles.get(c, ""), "창": wname}
            rec.update(test_series(x, za=True))
            rows.append(rec)
    return pd.DataFrame(rows)


def panel_tests(pf, cols, lo, hi, roles):
    """시도별 검정 요약: 기각 비율, Fisher(Maddala-Wu)·Stouffer 결합 p, 다수 판정, ΔD 지역간 평균상관"""
    sub = pf[(pf["P"] >= lo) & (pf["P"] <= hi)]
    rows, detail = [], []
    for c in cols:
        wide = sub.pivot(index="P", columns="region", values=c)
        res = {}
        for reg in wide.columns:
            r = test_series(wide[reg], za=False)
            r["region"], r["열이름"] = reg, c
            detail.append(r)
            res[reg] = r
        tested = [r for r in res.values() if r["판정"] not in ("", "검정 생략")]
        skipped = [r for r in res.values() if r["판정"] in ("", "검정 생략")]
        rec = {"열이름": c, "역할": roles.get(c, ""), "시도수": int(len(res)), "검정시도수": int(len(tested)),
               "생략사유": "; ".join(sorted({r["처리"] for r in skipped})) if skipped else ""}
        if tested:
            ps = np.clip([r["ADF_c_p"] for r in tested], 1e-12, 1 - 1e-12)
            rec["ADF기각비율"] = round(float(np.mean(np.array(ps) < ALPHA)), 2)
            rec["KPSS기각비율"] = round(float(np.mean([r["KPSS_c_기각"] for r in tested])), 2)
            rec["Fisher_p(ADF)"] = float(stats.combine_pvalues(ps, method="fisher")[1])
            rec["Stouffer_p(ADF)"] = float(stats.combine_pvalues(ps, method="stouffer")[1])
            vs = pd.Series([r["판정"] for r in tested])
            rec["다수판정"] = f"{vs.mode().iloc[0]} ({int((vs == vs.mode().iloc[0]).sum())}/{len(vs)})"
            rec["AR1_중앙값"] = round(float(np.nanmedian([r["AR1"] for r in tested])), 3)
            rec["판정_ct_다수"] = pd.Series([r.get("판정_ct", "") for r in tested]).mode().iloc[0]
        dcorr = wide.diff().corr()
        n = dcorr.shape[0]
        rec["ΔD_지역간평균상관"] = round(float((dcorr.values[np.triu_indices(n, 1)]).mean()), 3) if n > 1 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(detail)


def self_test():
    rng = np.random.default_rng(20261007)
    idx = pd.period_range("2016-01", periods=120, freq="M")
    rw = pd.Series(np.cumsum(rng.normal(size=120)), index=idx)
    wn = pd.Series(rng.normal(size=120), index=idx)
    r1, r2 = test_series(rw, za=False), test_series(wn, za=False)
    assert r1["판정"] == "I(1)", r1
    assert r2["판정"] == "I(0)", r2
    return pd.DataFrame([{"항목": "랜덤워크 n=120 (기대 I(1))", "ADF_p": round(r1["ADF_c_p"], 3), "KPSS_기각": r1["KPSS_c_기각"], "판정": r1["판정"]},
                         {"항목": "백색잡음 n=120 (기대 I(0))", "ADF_p": round(r2["ADF_c_p"], 3), "KPSS_기각": r2["KPSS_c_기각"], "판정": r2["판정"]}])


# ============================================================ 요약·규칙
def rules_table(s):
    r = s["rmpi"]
    mk = s["masking"]
    rows = [
        ("① 보완 셀 마스킹", f"대상 {mk['columns']}, 방법 {mk['future_info_methods']}, 시도 셀 수 {mk['expected_cells']['sido']} (data.future_mask: 보완 셀을 공표 시차만큼 민 뒤 2차 값에서 지움)"),
        ("② 공표 시차", f"2차 시트(공표시점반영) 값 사용 + 추가 시차 {s.get('lag_adjustments', {})} (data.lag_adjust). 타깃 Y 는 밀지 않음"),
        ("③ 변환", "; ".join(f"{k}: {v}" for k, v in FORMULA.items()) + "; 계산 창 안에 결측이 하나라도 있으면 결측 (data.transform)"),
        ("③ 실거래 별도 입력", "; ".join(RT_FORMULA.values()) + f"; 경계 {mk['rt_break_first_row']} 부터 {mk['rt_mask_rows']['sum3_d12']}행 결측, 최신 {mk['rt_truncate_last_months']}개월 결측 (data.rt_inputs)"),
        ("④ 기준 지역 분해", f"지역 변수 → 공통 C = 기준 지역 월별 평균(한 지역이라도 빠진 달은 결측), 지역 D = 시도값 − 평균. 기준 지역 예외 {r['reference_regions']['exceptions']} (data.decompose)"),
        ("저빈도 → 월", "분기·반기·연간 계열은 공표 시점 이후 최신 공표값을 유지(계단 채움, impute/build_preprocessed.py stepfill). 보간 아님 — yaml 의 V017 '연간 보간' 주석은 코드와 다름"),
        ("CSI", f"지역 유형별 1열 {s['csi_region_type']} 로 매핑 뒤 지역 변수처럼 분해 (3개 값이 17개 시도에 복제)"),
        ("⑤ 관측률 필터", f"훈련 구간 관측률 ≥ {r['obs_rate_min']}, 상수 아님 (index.RMPI.fit)"),
        ("⑥ 표준화", f"공통: {r['standardization']['common']}; 지역: {r['standardization']['regional']}; 절삭 winsor_sd = {r['winsor_sd']}"),
        ("⑦ 부호", f"사전 부호(+/−) 또는 ± 는 훈련기간 목표({r['direction_target']})와의 Spearman 부호 (관측 < 8 이면 +) (index._sign_of)"),
        ("⑧ 집계", f"묶음 안 동일 개념(concept) 평균 → 개념 간 동일 가중(남은 개념 ≥ {r['group_min_share']}) → 묶음 간 동일 가중(남은 묶음 ≥ {r['driver_min_share']}) (index.RMPI._indices)"),
        ("⑨ 결측 대체", f"{r['impute']} (동인 지수 단위) → 파이프라인 중앙값 대체 → StandardScaler → Ridge/Logistic (models.ridge/logistic)"),
        ("S1 추세 제외", f"{r.get('trend_exclusion', {}).get('rule', '')}: {r.get('trend_exclusion', {}).get('applied', [])}"),
        ("훈련행", "결정월 ≥ 2016-01 이고 ≤ T − h, 타깃·A 입력 비결측 (engine.national_run L95 / panel_run L181 / correction_run L378). 최초 평가월 2018-01 의 훈련행: h=1/3/6 → 24/22/19 결정월 중 G·past6 가 있는 행"),
        ("alpha", f"전국·보정 모형: {s['models']['ridge_national']['nested']['rule']} (1단계 격자 10^-2~10^4 25점 → 2단계 ±0.5 로그 0.125 간격, 평평하면 10 에 가까운 것, 폴드 < {s['models']['ridge_national']['nested']['min_folds']} 이면 {s['models']['ridge_national']['nested']['fallback_alpha']}), 매년 {s['models']['ridge_national']['nested']['refit_month']}월 재선택; 패널 Ridge alpha {s['models']['ridge_panel']['alpha']}, 로지스틱 C {s['models']['logistic_panel']['C']} 고정"),
    ]
    return pd.DataFrame(rows, columns=["항목", "규칙"])


def write_block(ws_writer, sheet, tables, titles):
    """한 시트에 여러 표를 제목 행과 함께 세로로 쌓는다."""
    row = 0
    for t, title in zip(tables, titles):
        pd.DataFrame({title: []}).to_excel(ws_writer, sheet_name=sheet, startrow=row, index=False)
        row += 1
        t.to_excel(ws_writer, sheet_name=sheet, startrow=row, index=False)
        row += len(t) + 3


def _str_periods(df):
    df = df.copy()
    for c in df.columns:
        if isinstance(df[c].dtype, pd.PeriodDtype):
            df[c] = df[c].astype(str)
        elif df[c].dtype == object and len(df[c]) and isinstance(df[c].dropna().iloc[0] if df[c].notna().any() else None, pd.Period):
            df[c] = df[c].astype(str)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    s = S.load()
    os.makedirs(OUT, exist_ok=True)
    lo, hi = S.per(s["timing"]["official_first_decision"]), S.per(s["timing"]["eval_end"])
    post = S.per(s["timing"]["posthoc_label_end"])
    selftest = self_test()
    print("자가 점검 통과:", selftest["판정"].tolist(), flush=True)

    d = prepare(s, args.cache)
    b, nf, pf = d["b"], d["nf"], d["pf"]
    spec = b["spec"]
    print(f"자료 준비 완료: nf {nf.shape}, pf {pf.shape}, 설정 {s['meta']['settings_version']}", flush=True)

    # ---- 1 열목록
    tab = column_table(s, spec, nf, pf, lo, hi)
    roles_nat = dict(zip(tab.loc[tab["표"] == "전국", "열이름"], tab.loc[tab["표"] == "전국", "역할"]))
    roles_pan = dict(zip(tab.loc[tab["표"] == "패널", "열이름"], tab.loc[tab["표"] == "패널", "역할"]))

    # ---- 읽기용 열 이름 'ID_변수명_전처리 (원래 이름)'
    spi = spec.set_index("입력")
    tab.insert(1, "읽기용 열이름", [readable_name(c, t, spi, s) for c, t in zip(tab["열이름"], tab["표"])])

    # ---- 2·3 입력표 (값 그대로, 열 이름은 읽기용)
    nf_out = nf.loc[lo:post].copy()
    nf_out.insert(0, "P", [str(p) for p in nf_out.index])
    nf_out = rename_cols(_str_periods(nf_out.reset_index(drop=True)), "전국", spi, s)
    pf_out = rename_cols(_str_periods(pf[(pf["P"] >= lo) & (pf["P"] <= post)].reset_index(drop=True)), "패널", spi, s)
    nf_out.to_csv(os.path.join(OUT, "review_전국_입력표.csv"), index=False, encoding="utf-8-sig")
    pf_out.to_csv(os.path.join(OUT, "review_패널_입력표.csv"), index=False, encoding="utf-8-sig")

    # ---- 4 설계행렬
    Znat, Zpan, comps = design_cases(s, spec, nf, pf)
    print(f"설계행렬: 전국 {Znat.shape}, 패널 {Zpan.shape}, 구성명세 {comps.shape}", flush=True)
    Znat_out, Zpan_out = rename_cols(Znat, "전국", spi, s), rename_cols(Zpan, "패널", spi, s)

    # ---- 5 정상성 전국: nf 수치 열(식별·정답시점·0/1 제외) + 전국 동인 지수(h=1, 2025-12 적합)
    skip = {"식별", "정답 시점", "상태", "타깃(사건)"}
    nat_cols = [c for c in nf.columns if roles_nat.get(c) not in skip and pd.api.types.is_numeric_dtype(nf[c])]
    windows = {"모형창 2016-01~2025-12": (lo, hi), "전체 가용": (None, None)}
    st_nat = national_tests(nf, nat_cols, windows, roles_nat)
    zsel = Znat[Znat["사례"].str.startswith("전국 B h=1 적합시점 2025-12")].set_index(pd.PeriodIndex(Znat.loc[Znat["사례"].str.startswith("전국 B h=1 적합시점 2025-12"), "P"], freq="M"))
    dcols = [c for c in zsel.columns if c.startswith("RMPI공통|") and not c.endswith("|결측")]
    st_idx = national_tests(zsel, dcols, {"설계행렬 h=1 2025-12 적합(훈련행)": (None, None)}, {c: "동인 지수(설계행렬)" for c in dcols})
    st_nat = pd.concat([st_nat, st_idx], ignore_index=True)

    # ---- 6 정상성 패널: D| 열, 타깃 G·r, A 열 + 패널 동인 지수
    pan_cols = [c for c in pf.columns if c.startswith("D|")] + [f"G{h}" for h in s["timing"]["horizons"]] + [f"r{h}" for h in s["timing"]["horizons"]] + \
               ["past1", "past3", "past6", "rel_past1", "rel_past6", "V002|6개월%", "V026|6개월%", "V002|6개월%|월내편차", "V026|6개월%|월내편차"]
    pan_cols = [c for c in pan_cols if c in pf.columns]
    st_pan, st_pan_detail = panel_tests(pf, pan_cols, lo, hi, roles_pan)
    zp = Zpan.copy()
    zp["P"] = pd.PeriodIndex(zp["P"], freq="M")
    pdcols = [c for c in zp.columns if c.startswith("RMPI지역|") and not c.endswith("|결측")]
    st_pidx, _ = panel_tests(zp, pdcols, lo, hi, {c: "지역 동인 지수(설계행렬)" for c in pdcols})
    st_pan = pd.concat([st_pan, st_pidx], ignore_index=True)

    # ---- 열목록에 판정 조인
    v_nat = st_nat[st_nat["창"].str.startswith("모형창")].set_index("열이름")["판정"]
    v_pan = st_pan.set_index("열이름")
    tab["정상성판정_모형창"] = [v_nat.get(c, "") if t == "전국" else
                         (v_pan.loc[c, "다수판정"] if c in v_pan.index else v_nat.get(c[2:], "") if c.startswith("C|") else "")   # 패널 C| 열은 전국 열과 같은 계열
                         for c, t in zip(tab["열이름"], tab["표"])]
    tab["ADF기각비율_시도"] = [v_pan.loc[c, "ADF기각비율"] if (t == "패널" and c in v_pan.index and "ADF기각비율" in v_pan.columns) else np.nan for c, t in zip(tab["열이름"], tab["표"])]

    for df_, t_ in ((st_nat, "전국"), (st_pan, "패널"), (st_pan_detail, "패널")):
        df_.insert(1, "읽기용 열이름", [readable_name(c, t_, spi, s) for c in df_["열이름"]])

    # ---- 7 요약
    nat_model = st_nat[st_nat["창"].str.startswith("모형창")]
    cnt = nat_model.groupby(["역할", "판정"]).size().rename("열수").reset_index()
    flag_nat = nat_model[nat_model["역할"].str.startswith("공통 블록") & nat_model["열이름"].str.endswith("|수준") & ~nat_model["판정"].isin(["I(0)"])].reindex(
        columns=["열이름", "역할", "판정", "ADF_c_p", "KPSS_c_기각", "AR1", "처리", "ZA_구조변화"])
    flag_pan = st_pan[st_pan["열이름"].str.endswith("|수준") & (st_pan.get("ADF기각비율", pd.Series(1.0, index=st_pan.index)).fillna(1.0) < 0.5)].reindex(
        columns=["열이름", "역할", "ADF기각비율", "KPSS기각비율", "다수판정", "AR1_중앙값"])
    col_cnt = tab.groupby(["표", "역할"]).size().rename("열수").reset_index()
    used = tab[~tab["모형사용"].fillna("").str.contains("미사용|—|식별", regex=True)].groupby("표").size().rename("모형이 쓰는 열수").reset_index()
    excl = pd.DataFrame([{"열": c, "이유": EXCLUDE_REASON.get(next((k for k in EXCLUDE_REASON if c.startswith(k)), ""), "")} for c in s["excluded_columns"]])
    manifest = S.manifest(s, {"stage": "review", "nf열": nf.shape[1], "pf열": pf.shape[1]})

    with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
        tab.to_excel(w, sheet_name="1_열목록", index=False)
        nf_out.to_excel(w, sheet_name="2_전국_입력표", index=False)
        pf_out.to_excel(w, sheet_name="3_패널_입력표", index=False)
        Znat_out.to_excel(w, sheet_name="4a_설계행렬_전국", index=False)
        Zpan_out.to_excel(w, sheet_name="4b_설계행렬_패널", index=False)
        comps.to_excel(w, sheet_name="4c_구성명세", index=False)
        st_nat.to_excel(w, sheet_name="5_정상성_전국", index=False)
        st_pan.to_excel(w, sheet_name="6_정상성_패널", index=False)
        st_pan_detail.to_excel(w, sheet_name="6b_정상성_패널_시도별", index=False)
        write_block(w, "7_요약",
                    [cnt, flag_nat, flag_pan, col_cnt, used, excl, rules_table(s), selftest, manifest],
                    ["전국 모형창 판정 집계 (역할 x 판정)", "플래그: 공통 블록 수준형 입력 중 I(0) 아님", "플래그: 지역 블록 수준형 입력 중 시도 ADF 기각 비율 < 0.5",
                     "열 집계 (표 x 역할)", "모형이 실제로 쓰는 열 수", "제외 열 (settings.excluded_columns)", "전처리·학습 규칙 (설정·코드 기준)",
                     "자가 점검 (검정 래퍼)", "manifest"])
    print(f"저장: {XLSX}")
    print(f"열목록 {len(tab)}행 (전국 {nf.shape[1]} + 패널 {pf.shape[1]}), 정상성 전국 {len(st_nat)}행, 패널 {len(st_pan)}행")
    print(cnt.to_string(index=False))
    print("플래그(공통 수준형):", flag_nat["열이름"].tolist())
    print("플래그(지역 수준형):", flag_pan["열이름"].tolist())


if __name__ == "__main__":
    main()
