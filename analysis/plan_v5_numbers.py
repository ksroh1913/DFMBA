# -*- coding: utf-8 -*-
"""
※ 데이터사전 변수가 아닌 분석용 점검 스크립트. 연구계획 5차 수정안에 인용한 기술 통계를 다시 계산한다.

입력: 데이터취합_전처리_20261004.xlsx (17시도·서울25구 1차/2차 시트)
      raw/5_시장과열단기트리거/V001_월세통합가격지수_A_2024_00054.csv   (월세통합, 공식 2015.06~)
      raw/3_금융여건상대가격/V002_매매가격지수_A_2024_00045.csv         (매매)
      raw/3_금융여건상대가격/V026_전세가격지수_A_2024_00050.csv         (전세)
      raw/7_보완용_사전외/보완_V001_월세가격지수(구)_A_2024_00164.csv  (옛 월세가격지수, 2010.06~2015.06)
출력: analysis/output/plan_v5_numbers.csv  (구분, 항목, 값, 설명) + 입력 파일 SHA256

정의
  G(i,t,h) = 100 x [R(i,t-1+h) / R(i,t-1) - 1]   마지막 공표 수준 기준 다음 h회 공표분 누적 변화
  개발 성격 구간 = 결정월 2016.01~, 정답월(t-1+h) <= 2020.12
  월 공통 비중 = 월 고정효과의 제곱합 / 총제곱합 (공식 지수 행만)
  모든 상관은 기술 통계이다. 6개월 변화는 이웃한 달끼리 겹치므로 유의성 검정을 하지 않는다.
"""

import glob
import hashlib
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "analysis", "output")
XLSX = os.path.join(BASE, "데이터취합_전처리_20261004.xlsx")
RAW_RENT = os.path.join(BASE, "raw/5_시장과열단기트리거/V001_월세통합가격지수_A_2024_00054.csv")
RAW_SALE = glob.glob(os.path.join(BASE, "raw/3_금융여건상대가격/V002_*.csv"))[0]
RAW_JEON = glob.glob(os.path.join(BASE, "raw/3_금융여건상대가격/V026_*.csv"))[0]
RAW_OLD = glob.glob(os.path.join(BASE, "raw/7_보완용_사전외/보완_V001_*.csv"))[0]
PANELS = {
    "17시도": ("17시도_1차_결측보완", "17시도_2차_공표시점반영(ML용)", "Y_V001_월세통합가격지수(구지수연결)"),
    "서울25구": ("서울25구_1차_결측보완", "서울25구_2차_공표시점반영(ML용)", "Y_V001_월세통합가격지수(권역역산_학습용)"),
}
DEV_END = pd.Period("2020-12", "M")
rows = []


def put(sec, item, val, note=""):
    rows.append({"구분": sec, "항목": item, "값": val, "설명": note})


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rone(path, name):
    df = pd.read_csv(path)
    s = df[df.CLS_NM == name].groupby("WRTTIME_IDTFR_ID").DTA_VAL.mean()
    s.index = s.index.astype(int)
    return s.sort_index()


def month_share(df, col):
    mu = df[col].mean()
    tot = ((df[col] - mu) ** 2).sum()
    mfe = df.groupby("m")[col].transform("mean")
    return ((mfe - mu) ** 2).sum() / tot


def target_frame(d, ycol, h):
    R = d[ycol].where(d["Y_평가사용가능"] == 1)
    g = R.groupby(d["region"])
    out = pd.DataFrame({"m": d["month"], "r": d["region"],
                        "G": 100 * (g.shift(1 - h) / g.shift(1) - 1),
                        "past": 100 * (g.shift(1) / g.shift(1 + h) - 1)})
    out["lab"] = pd.PeriodIndex(d["month"].astype(str), freq="M") + (h - 1)
    return out


# 0. 입력 파일
for p in (XLSX, RAW_RENT, RAW_SALE, RAW_JEON, RAW_OLD):
    put("입력", os.path.relpath(p, BASE), sha(p)[:16], "SHA256 앞 16자리")

# 1. 2차(ML용) 시트의 열 구조
for name, (s1, s2, ycol) in PANELS.items():
    d2 = pd.read_excel(XLSX, sheet_name=s2)
    off = d2[d2["month"] >= 201506]
    xcols = [c for c in d2.columns if c not in ("panel", "region", "region_code", "month")
             and not c.startswith(("Y_", "D_"))]
    vary = {c: (off.groupby("month")[c].nunique() > 1).mean() for c in xcols}
    miss = off[xcols].isna().mean()
    if name == "서울25구":
        n_var = sum(v > 0 for v in vary.values())
        put("열 구조", f"{name} 설명변수(더미 제외)", len(xcols))
        put("열 구조", f"{name} 구마다 다른 열", n_var, "V013 서울구는 2021.08 이후만")
        put("열 구조", f"{name} 같은 달 공통 열", len(xcols) - n_var)
    else:
        regional = [c for c in xcols if vary[c] > 0.5 and miss[c] < 0.4]
        common = [c for c in xcols if vary[c] == 0 and miss[c] < 0.1]
        put("열 구조", f"{name} 설명변수(더미 제외)", len(xcols))
        put("열 구조", f"{name} 지역(시도별) 사용 열", len(regional))
        put("열 구조", f"{name} 전국 공통 열", len(common))
        put("열 구조", f"{name} 부분기간·종료·미수집 열", len(xcols) - len(regional) - len(common))

# 2. 공통 변동과 지역 변동 (G 기준, 공식 지수 행만)
for name, (s1, s2, ycol) in PANELS.items():
    d = pd.read_excel(XLSX, sheet_name=s1, usecols=["region", "month", ycol, "Y_평가사용가능"])
    d = d.sort_values(["region", "month"]).reset_index(drop=True)
    for h in (1, 3, 6):
        tf = target_frame(d, ycol, h).dropna(subset=["G", "past"])
        tf = tf[tf.m >= 201601]
        dev = tf[tf.lab <= DEV_END].copy()
        put("분산 구조", f"{name} h={h} 월 공통 비중(전체)", round(month_share(tf, "G"), 3))
        put("분산 구조", f"{name} h={h} 월 공통 비중(개발)", round(month_share(dev, "G"), 3))
        dev["Gr"] = dev.G - dev.groupby("m").G.transform("mean")
        dev["pr"] = dev.past - dev.groupby("m").past.transform("mean")
        sp = np.mean([spearmanr(x.pr, x.Gr)[0] for _, x in dev.groupby("m")])
        put("분산 구조", f"{name} h={h} 과거 상대변화→다음 상대순위 월내 Spearman(개발)", round(sp, 2))
        if h == 6:
            put("분산 구조", f"{name} h=6 월내 지역 간 표준편차(개발)", round(dev.groupby("m").G.std().mean(), 2))
            pv = tf.pivot(index="m", columns="r", values="G")
            c = pv.corr().values
            put("분산 구조", f"{name} h=6 지역 간 상관 평균(전체)", round(c[np.triu_indices(c.shape[0], 1)].mean(), 2))

# 3. 전국 월세·매매·전세
nat = pd.DataFrame({"월세": rone(RAW_RENT, "전국"), "매매": rone(RAW_SALE, "전국"),
                    "전세": rone(RAW_JEON, "전국")}).loc[201506:]
c6 = 100 * (nat / nat.shift(6) - 1)
cc = c6.corr()
put("전국 비교", "6개월 변화 상관 월세-매매", round(cc.loc["월세", "매매"], 2))
put("전국 비교", "6개월 변화 상관 월세-전세", round(cc.loc["월세", "전세"], 2))
put("전국 비교", "6개월 변화 상관 매매-전세", round(cc.loc["매매", "전세"], 2))
yoy = 100 * (nat / nat.shift(12) - 1)
for ym, r in yoy[yoy.index % 100 == 12].dropna().iterrows():
    put("전국 비교", f"{ym // 100} 연말 전년비 월세/매매/전세", f"{r['월세']:+.1f}/{r['매매']:+.1f}/{r['전세']:+.1f}")
fut = 100 * (nat["월세"].shift(-6) / nat["월세"] - 1)
for x in ("매매", "전세"):
    put("전국 비교", f"{x} 최근6개월→월세 향후6개월 상관(시차 0/6/12)",
        "/".join(f"{fut.corr(c6[x].shift(k)):.2f}" for k in (0, 6, 12)))

# 4. 권역 집계와 옛 지수
agg = pd.DataFrame({"전국": rone(RAW_RENT, "전국"), "수도권": rone(RAW_RENT, "수도권"),
                    "지방권": rone(RAW_RENT, "지방권")})
g6 = 100 * (agg.shift(-5) / agg.shift(1) - 1)
for lo, hi in ((201506, 202608), (201506, 202012), (202101, 202608)):
    x = g6.loc[lo:hi].dropna()
    put("권역", f"6개월 변화 상관 {lo}~{hi} 전국-수도권/전국-지방권/수도권-지방권",
        f"{x['전국'].corr(x['수도권']):.2f}/{x['전국'].corr(x['지방권']):.2f}/{x['수도권'].corr(x['지방권']):.2f}")
old_cap = rone(RAW_OLD, "수도권")
put("옛 지수", "월변화 표준편차 옛 수도권(2012.06~2015.06) / 현 수도권(2015.07~2020.12)",
    f"{(100 * old_cap.pct_change()).loc[201206:].std():.3f}/{(100 * agg['수도권'].pct_change()).loc[201507:202012].std():.3f}")
old = pd.DataFrame({"수도권": old_cap, "5대광역시": rone(RAW_OLD, "5대광역시"), "8개시도": rone(RAW_OLD, "8개시도")})
og = (100 * (old.shift(-5) / old.shift(1) - 1)).dropna()
put("옛 지수", "옛 집계 6개월 변화 상관 수도권-5대광역시/수도권-8개시도",
    f"{og['수도권'].corr(og['5대광역시']):.2f}/{og['수도권'].corr(og['8개시도']):.2f}")
oy = 100 * (old_cap / old_cap.shift(12) - 1)
put("옛 지수", "옛 수도권 연말 전년비 2011~2014",
    "/".join(f"{oy.loc[y * 100 + 12]:+.1f}" for y in (2011, 2012, 2013, 2014)))
s1, _, ycol = PANELS["17시도"]
d = pd.read_excel(XLSX, sheet_name=s1, usecols=["region", "month", ycol, "Y_평가사용가능"]).sort_values(["region", "month"])
d["G6"] = 100 * (d.groupby("region")[ycol].shift(-5) / d.groupby("region")[ycol].shift(1) - 1)
p3 = d[d.region.isin(["서울", "경기", "인천"])].pivot(index="month", columns="region", values="G6")
for lo, hi, lab in ((201101, 201505, "옛 연결 구간"), (201506, 202608, "공식 구간")):
    c = p3.loc[lo:hi].dropna().corr().values
    put("옛 지수", f"서울·경기·인천 6개월 변화 상관 평균({lab})", round(c[np.triu_indices(3, 1)].mean(), 2))

# 5. 최근 흐름 (2026.08 기준 12개월 변화)
for name, (s1, s2, ycol) in PANELS.items():
    d = pd.read_excel(XLSX, sheet_name=s1, usecols=["region", "month", ycol])
    pv = d.pivot(index="month", columns="region", values=ycol)
    ch = (100 * (pv.loc[202608] / pv.loc[202508] - 1)).sort_values()
    put("최근 흐름", f"{name} 12개월 상승률 최고", f"{ch.index[-1]} {ch.iloc[-1]:.1f}%")
    put("최근 흐름", f"{name} 12개월 상승률 최저", f"{ch.index[0]} {ch.iloc[0]:.1f}%")

# 6. 사건 수 (17시도, 개발 성격 구간, 복리 경계)
s1, _, ycol = PANELS["17시도"]
d = pd.read_excel(XLSX, sheet_name=s1, usecols=["region", "month", ycol, "Y_평가사용가능"])
d = d.sort_values(["region", "month"]).reset_index(drop=True)
for h in (1, 3, 6):
    tf = target_frame(d, ycol, h).dropna(subset=["G"])
    dev = tf[(tf.m >= 201601) & (tf.lab <= DEV_END)].copy()
    up, dn = 100 * (1.03 ** (h / 12) - 1), 100 * (0.98 ** (h / 12) - 1)
    dev["dn"], dev["up"] = dev.G <= dn, dev.G >= up
    runs = 0
    for _, x in dev.sort_values("m").groupby("r"):
        e = x.dn.astype(int)
        runs += int((e.diff() == 1).sum() + e.iloc[0])
    put("사건 수", f"h={h} 급락(연율 -2%, 경계 {dn:.3f}%) 비율/행/월/시도/지역별 국면",
        f"{dev.dn.mean():.3f}/{int(dev.dn.sum())}/{dev.m[dev.dn].nunique()}/{dev.r[dev.dn].nunique()}/{runs}")
    put("사건 수", f"h={h} 급등(연율 +3%, 경계 {up:.3f}%) 비율/행/월/2020년 행",
        f"{dev.up.mean():.3f}/{int(dev.up.sum())}/{dev.m[dev.up].nunique()}/{int(dev.up[dev.m // 100 == 2020].sum())}")

os.makedirs(OUT, exist_ok=True)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "plan_v5_numbers.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)
print(res.to_string(index=False))
