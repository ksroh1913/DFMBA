# -*- coding: utf-8 -*-
"""
※ 데이터사전 변수가 아닌 분석용 산출물 (V006·V007 실거래 원본을 가공). 결과는 analysis/output/ 에 저장.
[전처리] 실거래 기반 '품질조정' 월세지수 - 단지x평형 고정효과 헤도닉

입력: raw:molit/apt_rent_<시군구코드>.csv          (건별 원자료)
      processed/부동산원_전월세전환율_아파트.csv     (자치구별 월별 전환율, 연 %)
출력: analysis/output/서울_아파트_월세지수_품질조정.csv      (자치구별 / 유형별 지수)
      analysis/output/서울_아파트_월세지수_준공연도별.csv    (준공연도대 코호트별 지수)
      analysis/output/서울_아파트_월세_연식프리미엄.csv      (연식 구간별 가격 프리미엄)

--------------------------------------------------------------------------------
단순 구별 평균이 왜 틀리는가
--------------------------------------------------------------------------------
"강남구 평균 월세가 올랐다"는 두 가지 이유로 생긴다.
  (1) 같은 집의 임대료가 올랐다              <- 우리가 알고 싶은 것
  (2) 이번 달에 비싼 단지/큰 평형이 더 많이 거래됐다  <- 구성효과(composition), 잡음
표본이 매달 뒤바뀌는 실거래 자료에서는 (2)가 (1)보다 클 수 있다. 신축 입주장이
열리면 그 달 평균이 튀고, 재건축 이주로 노후 단지 거래가 몰리면 평균이 꺼진다.

--------------------------------------------------------------------------------
그래서 이렇게 푼다 (matched-model / 고정효과 헤도닉)
--------------------------------------------------------------------------------
셀(cell) = 단지(aptSeq) x 전용면적          <- '같은 집'의 대리변수. 동일 단지 동일 평형.

  log(환산월세) = a_cell + b1*층 + b2*층^2 + t_연월 + e

a_cell 이 단지의 위치·브랜드·평형·기본 품질을 통째로 흡수하므로, 남은 t_연월 은
"동일한 셀 안에서 시간에 따라 변한 임대료"만 나타낸다. 지수 = exp(t_연월),
기준월 = 100. 이것이 사용자가 말한 '단지 가격 흐름을 먼저 산출하고 그걸 모으는 것'의
회귀 버전이다. 인접 월만 이어붙이는 연쇄지수와 달리 멀리 떨어진 거래쌍도 모두 쓴다.

추정은 within 변환(셀별 평균 차감) 후 정규방정식으로 푼다. 셀이 수만 개라 더미를
만들 수 없으므로, X'MX = (G'X)' diag(1/n_c) (G'X) 항등식으로 희소행렬 곱만 쓴다.
(G = 셀 지시행렬) 셀 안에 거래가 1건뿐이면 정보가 없으므로 제외한다.

--------------------------------------------------------------------------------
한계 (반드시 같이 읽을 것)
--------------------------------------------------------------------------------
  - 연식 감가는 지수에 남는다. 셀 고정효과를 넣으면 연식 = 연월 - 준공연도 이고
    준공연도는 셀 안에서 상수라, 연식과 연월더미가 완전공선이다(age-period 문제).
    모든 반복거래형 지수(부동산원 실거래지수, Case-Shiller 포함)가 같은 한계를 갖는다.
    연식 효과는 별도로 횡단면 추정해 '연식프리미엄' 파일로 낸다.
  - 리모델링·수선으로 품질이 바뀐 셀은 그 변화가 가격상승으로 잡힌다.
  - 전세는 보증금 전액을 전환율로 월세화한 값이라, 지수에 전환율 변동이 섞인다.
"""

import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import PROCESSED_DIR, SEOUL_GU, raw_path, write_analysis  # noqa: E402

CONV_CSV = os.path.join(PROCESSED_DIR, "부동산원_전월세전환율_아파트.csv")
BASE_YM = "2015-06"          # 지수 기준월 (부동산원 월세통합지수 시작월에 맞춤)
MIN_CELL_OBS = 2             # 고정효과가 정보를 갖는 최소 거래건수
TYPES = ["전세", "월세", "준월세", "준전세"]


# ------------------------------------------------------------------ 데이터 적재
def load_rates():
    df = pd.read_csv(CONV_CSV, encoding="utf-8-sig")
    df = df[df["자치구"].notna() & (df["자치구"] != "")]
    return df.set_index(["자치구", "연월"])["값"].astype(float)


def load_contracts():
    frames = []
    cols = ["aptSeq", "buildYear", "contractType", "dealMonth", "dealYear",
            "deposit", "excluUseAr", "floor", "monthlyRent", "_dealYmd", "_lastSeen"]
    for lawd_cd, gu in SEOUL_GU.items():
        path = raw_path("molit", f"apt_rent_{lawd_cd}")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path, encoding="utf-8-sig", usecols=cols, dtype=str,
                        thousands=",", low_memory=False)
        d["자치구"] = gu
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # 월별 최신 확인분만 = 현재 유효한 스냅샷 (사라진 행 = 취소 추정)
    newest = df.groupby("_dealYmd")["_lastSeen"].transform("max")
    df = df[df["_lastSeen"] == newest]
    # 갱신계약은 직전 계약 조건에 묶여 시세를 반영하지 않는다
    df = df[df["contractType"].fillna("").str.strip() != "갱신"]

    for c in ("deposit", "monthlyRent", "excluUseAr", "floor", "buildYear",
              "dealYear", "dealMonth"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    df["monthlyRent"] = df["monthlyRent"].fillna(0)

    df = df[(df["excluUseAr"] > 0) & df["deposit"].notna() & df["dealYear"].notna()]
    df["연월"] = (df["dealYear"].astype(int).astype(str) + "-"
                 + df["dealMonth"].astype(int).astype(str).str.zfill(2))

    ratio = np.where(df["monthlyRent"] > 0,
                     df["deposit"] / df["monthlyRent"].replace(0, np.nan), np.inf)
    df["계약유형"] = np.select(
        [df["monthlyRent"] <= 0, ratio <= 12, ratio <= 240],
        ["전세", "월세", "준월세"], default="준전세")

    rates = load_rates()
    df["전환율"] = rates.reindex(
        pd.MultiIndex.from_arrays([df["자치구"], df["연월"]])).to_numpy()
    df = df[df["전환율"].notna()]

    df["환산월세"] = df["monthlyRent"] + df["deposit"] * (df["전환율"] / 100) / 12
    df = df[df["환산월세"] > 0]
    # 셀 = 동일 단지 x 동일 전용면적 ('같은 집'의 대리변수)
    df["cell"] = df["aptSeq"] + "|" + df["excluUseAr"].round(2).astype(str)
    df["층"] = df["floor"].fillna(0).clip(-2, 70)
    df["연식"] = df["dealYear"] - df["buildYear"]
    return df


# --------------------------------------------------------- 고정효과 헤도닉 추정
def fe_index(sub, base_ym=BASE_YM):
    """셀 고정효과를 흡수하고 연월더미를 추정해 지수(기준월=100)를 돌려준다."""
    sub = sub[sub.groupby("cell")["환산월세"].transform("size") >= MIN_CELL_OBS]
    if len(sub) < 200:
        return None

    months = np.sort(sub["연월"].unique())
    if len(months) < 12:
        return None
    # 기준월이 표본에 없으면 가장 이른 달을 기준으로 잡고 나중에 재기준화
    anchor = base_ym if base_ym in months else months[0]
    others = [m for m in months if m != anchor]
    midx = {m: i for i, m in enumerate(others)}

    n = len(sub)
    rows = np.arange(n)
    mcol = sub["연월"].map(midx).to_numpy()          # 기준월 행은 NaN
    keep = ~np.isnan(mcol)
    T = len(others)

    # X = [연월더미(기준월 제외) | 층 | 층^2]
    fl = sub["층"].to_numpy(dtype=float)
    dummies = sp.csr_matrix(
        (np.ones(keep.sum()), (rows[keep], mcol[keep].astype(int))), shape=(n, T))
    ctrl = sp.csr_matrix(np.column_stack([fl / 10.0, (fl / 10.0) ** 2]))
    X = sp.hstack([dummies, ctrl]).tocsr()
    y = np.log(sub["환산월세"].to_numpy(dtype=float))

    # 셀 지시행렬 G (n x C) 로 within 변환을 정규방정식에서 직접 처리
    codes, _ = pd.factorize(sub["cell"])
    C = codes.max() + 1
    G = sp.csr_matrix((np.ones(n), (rows, codes)), shape=(n, C))
    inv_nc = sp.diags(1.0 / np.asarray(G.sum(axis=0)).ravel())

    GX = (G.T @ X).tocsr()                            # 셀별 X 합
    Gy = G.T @ y                                      # 셀별 y 합
    XtX = (X.T @ X).toarray() - (GX.T @ inv_nc @ GX).toarray()
    Xty = (X.T @ y) - GX.T @ (inv_nc @ Gy)

    beta = np.linalg.lstsq(XtX, Xty, rcond=1e-10)[0]
    tau = dict(zip(others, beta[:T]))
    tau[anchor] = 0.0

    counts = sub["연월"].value_counts()
    cells = sub.groupby("연월")["cell"].nunique()
    out = pd.DataFrame({
        "연월": months,
        "지수": [np.exp(tau[m]) * 100 for m in months],
        "표본수": [int(counts.get(m, 0)) for m in months],
        "셀수": [int(cells.get(m, 0)) for m in months],
    })
    if base_ym in out["연월"].values:                 # 항상 기준월=100 으로 재기준화
        out["지수"] = out["지수"] / float(out.loc[out["연월"] == base_ym, "지수"].iloc[0]) * 100
    return out


def simple_mean_index(sub, base_ym=BASE_YM):
    """비교용: 품질조정 없는 단순 ㎡당 평균 지수"""
    g = (sub.assign(unit=sub["환산월세"] / sub["excluUseAr"])
            .groupby("연월")["unit"].mean())
    if base_ym not in g.index:
        return None
    return g / g[base_ym] * 100


# ---------------------------------------------------------------------- 산출물
def build_index_table(df):
    rows = []
    jobs = [("서울", "", t, df[df["계약유형"] == t]) for t in TYPES]
    jobs.append(("서울", "", "월세전체", df[df["계약유형"] != "전세"]))
    jobs.append(("서울", "", "전체", df))
    for gu in sorted(df["자치구"].unique()):
        d = df[df["자치구"] == gu]
        jobs.append(("서울", gu, "월세전체", d[d["계약유형"] != "전세"]))
        jobs.append(("서울", gu, "전세", d[d["계약유형"] == "전세"]))

    for sido, gu, label, sub in jobs:
        idx = fe_index(sub)
        if idx is None:
            print(f"  ! 표본부족: {gu or '서울'} / {label}")
            continue
        simple = simple_mean_index(sub)
        for _, r in idx.iterrows():
            rows.append([sido, gu, r["연월"], label,
                         round(r["지수"], 2),
                         "" if simple is None else round(float(simple.get(r["연월"], np.nan)), 2),
                         r["표본수"], r["셀수"]])
        print(f"  - {gu or '서울'} / {label}: {len(sub):,}건, {idx['셀수'].max():,}셀", flush=True)

    rows.sort(key=lambda x: (x[1], x[3], x[2]))
    return write_analysis(
        "서울_아파트_월세지수_품질조정",
        ["광역지자체", "자치구", "연월", "구분", "품질조정지수", "단순평균지수", "표본수", "셀수"],
        rows)


def build_cohort_table(df):
    """준공연도대는 셀 안에서 불변이라 코호트를 나눠도 고정효과 추정이 성립한다."""
    w = df[df["계약유형"] != "전세"].copy()
    bins = [0, 1990, 2000, 2010, 2020, 9999]
    labels = ["1980년대이전", "1990년대", "2000년대", "2010년대", "2020년대"]
    w["준공연대"] = pd.cut(w["buildYear"], bins=bins, labels=labels, right=False)

    rows = []
    for coh in labels:
        sub = w[w["준공연대"] == coh]
        idx = fe_index(sub)
        if idx is None:
            print(f"  ! 표본부족: {coh}")
            continue
        for _, r in idx.iterrows():
            rows.append(["서울", "", r["연월"], coh, round(r["지수"], 2),
                         r["표본수"], r["셀수"]])
        print(f"  - {coh}: {len(sub):,}건", flush=True)
    rows.sort(key=lambda x: (x[3], x[2]))
    return write_analysis(
        "서울_아파트_월세지수_준공연도별",
        ["광역지자체", "자치구", "연월", "준공연대", "품질조정지수", "표본수", "셀수"], rows)


def build_age_premium(df):
    """연식 프리미엄: (자치구 x 연월) 고정효과를 흡수한 횡단면 헤도닉.

    같은 구·같은 달에 거래된 아파트끼리만 비교하므로 지역·시황이 통제된다.
      log(㎡당 환산월세) = a_(구x월) + b*log(면적) + c*층 + d_연식구간 + e
    연식 0~5년을 기준(0%)으로 한 상대 가격차를 낸다.
    """
    w = df[(df["계약유형"] != "전세") & df["연식"].notna()
           & (df["연식"] >= 0) & (df["연식"] <= 60)].copy()
    bands = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 61)]
    names = ["0-4년", "5-9년", "10-14년", "15-19년", "20-24년", "25-29년", "30년+"]
    w["연식구간"] = pd.cut(w["연식"], bins=[b[0] for b in bands] + [61],
                          labels=names, right=False)

    rows = []
    for period, sub in [("전체(2011-2026)", w),
                        ("2011-2015", w[w["연월"] < "2016-01"]),
                        ("2016-2020", w[(w["연월"] >= "2016-01") & (w["연월"] < "2021-01")]),
                        ("2021-2026", w[w["연월"] >= "2021-01"])]:
        if len(sub) < 1000:
            continue
        n = len(sub)
        r_ = np.arange(n)
        band = pd.Categorical(sub["연식구간"], categories=names, ordered=True)
        bcode = band.codes
        ok = bcode > 0                                  # 0-4년 = 기준
        D = sp.csr_matrix((np.ones(ok.sum()), (r_[ok], bcode[ok] - 1)),
                          shape=(n, len(names) - 1))
        fl = sub["층"].to_numpy(dtype=float) / 10.0
        ctrl = sp.csr_matrix(np.column_stack([
            np.log(sub["excluUseAr"].to_numpy(dtype=float)), fl, fl ** 2]))
        X = sp.hstack([D, ctrl]).tocsr()
        y = np.log((sub["환산월세"] / sub["excluUseAr"]).to_numpy(dtype=float))

        codes, _ = pd.factorize(sub["자치구"] + "|" + sub["연월"])
        G = sp.csr_matrix((np.ones(n), (r_, codes)), shape=(n, codes.max() + 1))
        inv_nc = sp.diags(1.0 / np.asarray(G.sum(axis=0)).ravel())
        GX = (G.T @ X).tocsr()
        Gy = G.T @ y
        beta = np.linalg.lstsq((X.T @ X).toarray() - (GX.T @ inv_nc @ GX).toarray(),
                               (X.T @ y) - GX.T @ (inv_nc @ Gy), rcond=1e-10)[0]

        cnt = sub["연식구간"].value_counts()
        rows.append(["서울", "", period, "0-4년", 0.0, int(cnt.get("0-4년", 0))])
        for i, nm in enumerate(names[1:]):
            rows.append(["서울", "", period, nm,
                         round((np.exp(beta[i]) - 1) * 100, 2), int(cnt.get(nm, 0))])
        print(f"  - {period}: {n:,}건", flush=True)

    return write_analysis(
        "서울_아파트_월세_연식프리미엄",
        ["광역지자체", "자치구", "기간", "연식구간", "㎡당환산월세_프리미엄(%)", "표본수"], rows)


def main():
    print("[1/4] 건별 자료 적재")
    df = load_contracts()
    print(f"  유효 {len(df):,}건 / 셀 {df['cell'].nunique():,}개 / "
          f"{df['연월'].min()}~{df['연월'].max()}")

    print("[2/4] 자치구·유형별 품질조정 지수")
    p1 = build_index_table(df)
    print("[3/4] 준공연도대별 지수")
    p2 = build_cohort_table(df)
    print("[4/4] 연식 프리미엄")
    p3 = build_age_premium(df)
    for p in (p1, p2, p3):
        print(f"완료: {p}")


if __name__ == "__main__":
    main()
