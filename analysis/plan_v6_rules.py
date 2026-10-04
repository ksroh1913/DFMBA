# -*- coding: utf-8 -*-
"""
※ 데이터사전 변수가 아닌 분석용 점검 스크립트. 연구계획 6차 수정안의 계산 규칙에 쓴 수치를 다시 계산한다.
  (5차 수정안의 기술 통계는 analysis/plan_v5_numbers.py가 그대로 담당한다.)

입력: 데이터취합_20261004.xlsx            (로데이터 취합본, 빈칸 그대로)
      데이터취합_전처리_20261004.xlsx      (1차 = 보완 후 기준기간, 2차 = 공표시점 반영, 설명서 '3_전처리방법')
      raw/5_시장과열단기트리거/V001_월세통합가격지수_A_2024_00054.csv   (공식 월세통합, 2015.06~)
      raw/7_보완용_사전외/보완_V001_월세가격지수(구)_A_2024_00164.csv  (옛 월세가격지수, 2010.06~2015.06)
출력: analysis/output/plan_v6_rules.csv  (구분, 항목, 값, 설명) + 입력 파일 SHA256

결정월 t 의 정보: 2차 시트 행 t 의 입력(그 달 말 가용) + 월세 수준 R(t-1)까지.
목표 구간 [t-1, t-1+h], 과거 월세 입력(A) 구간 [t-7, t-1]. 정답은 t+h 월 말에 확인되므로 시점 T의 훈련 행은 t <= T-h.

1. 보완 셀 마스킹
   취합본에서 빈칸이고 1차 시트에서 채워진 셀 = 보완 셀. 설명서의 처리 방법으로 미래정보 사용 여부를 나눈다.
   미래정보 셀(역산_지수변화 V003, 대용_금리차 V057·V059)을 기준기간에서 지우고, 공표 시차만큼 밀어 2차 시트 행에 대응시킨 뒤
   변환한다. 반대 순서(변환 후 해당 행만 지움)로 하면 12개월 차분의 과거 기준값에 보완값이 남는 행이 생긴다.
   교란 시험: 마스킹 셀을 난수로 바꿔도 마스킹-먼저 순서의 파생값은 변하지 않아야 한다.
2. 접합 규칙
   2015.06은 두 체계의 공통 기준점(연결 비율로 수준이 같음). 구간 [a,b]는 b<=2015.06이면 옛 체계, a>=2015.06이면 공식 체계,
   그 밖은 접합을 가로지름. 한 행의 목표 구간과 A 구간이 모두 같은 체계일 때만 쓴다.
3. 선택형 단순기준
   시점 T의 평가창 W(T) = 정답이 확인된 결정월 중 최근 최대 24개월(공식 구간). 후보는 W(T)의 모든 달에 그 당시 자료로
   만든 예측이 있어야 비교 대상이 된다. |W(T)| < 12 이면 사전 지정 기본 규칙(변화율 0).
4. 17개 시도 완전 패널: 공식 V001이 17개 시도 모두 있는 달.
5. 광주·전남: 공식 V001 원자료에 두 시도가 따로 공표되는지.
"""

import glob
import hashlib
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from variables import RELEASE_MONTHLY  # noqa: E402

OUT = os.path.join(BASE, "analysis", "output")
XLSX = os.path.join(BASE, "데이터취합_전처리_20261004.xlsx")
MERGED = os.path.join(BASE, "데이터취합_20261004.xlsx")
RAW_RENT = os.path.join(BASE, "raw/5_시장과열단기트리거/V001_월세통합가격지수_A_2024_00054.csv")
RAW_OLD = glob.glob(os.path.join(BASE, "raw/7_보완용_사전외/보완_V001_*.csv"))[0]
S1, S2, RAW_SHEET = "17시도_1차_결측보완", "17시도_2차_공표시점반영(ML용)", "17시도_월별"
YCOL = "Y_V001_월세통합가격지수(구지수연결)"
J = pd.Period("2015-06", "M")          # 접합 기준점
OFF0 = pd.Period("2016-01", "M")       # 공식 구간 첫 결정월 (R(t-7) >= 2015.06)
EXT0 = pd.Period("2012-01", "M")       # 확장 실험 첫 결정월 (고정)
EVAL = pd.period_range("2018-01", "2025-12", freq="M")
HS = (1, 3, 6)
FUTURE = {"역산_지수변화": "첫 공표월 전세가율(미래 값)로 역산",
          "대용_금리차": "공표 시작 뒤 2년 평균 금리차(미래 값)로 대용"}
REALTIME = {"0채움_합계대조": "같은 달 합계 항등식", "차분복원_누계0확인": "같은 달 누계 항등식",
            "전신계열": "당시 공표된 전신 통계(문항 차이는 더미)",
            "연결_구계열": "목표값. 2015.06 비율은 수준만 바꾸고 같은 체계 안 변화율은 불변"}
rows = []


def put(sec, item, val, note=""):
    rows.append({"구분": sec, "항목": item, "값": val, "설명": note})


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def per(m):
    return pd.Period(f"{int(m) // 100}-{int(m) % 100:02d}", "M")


def span(ps):
    ps = sorted(set(ps))
    return f"{ps[0]}~{ps[-1]} ({len(ps)}개월)" if ps else "없음"


for p in (MERGED, XLSX, RAW_RENT, RAW_OLD):
    put("입력", os.path.relpath(p, BASE), sha(p)[:16], "SHA256 앞 16자리")

# ============================================================ 1. 보완 셀 마스킹
s1 = pd.read_excel(XLSX, sheet_name=S1).sort_values(["region", "month"]).reset_index(drop=True)
s2 = pd.read_excel(XLSX, sheet_name=S2).sort_values(["region", "month"]).reset_index(drop=True)
raw = pd.read_excel(MERGED, sheet_name=RAW_SHEET).sort_values(["region", "month"]).reset_index(drop=True)
assert (s1[["region", "month"]].values == raw[["region", "month"]].values).all()
assert (s1[["region", "month"]].values == s2[["region", "month"]].values).all()
s1["P"] = s1["month"].map(per)
s2["P"] = s2["month"].map(per)

doc = pd.read_excel(XLSX, sheet_name="3_전처리방법", header=None)
hdr = doc.index[doc[0].astype(str).eq("시트")][0]
doc = doc.iloc[hdr + 1:, :8].set_axis(["시트", "컬럼", "처리", "상세", "지역", "기간", "셀수", "근거"], axis=1)
doc = doc[doc["시트"].eq(S1)]
methods = doc.groupby("컬럼")["처리"].agg(lambda s: "/".join(sorted(set(s))))
cells = doc.groupby("컬럼")["셀수"].sum()

common = [c for c in s1.columns if c in raw.columns and c not in ("panel", "region", "region_code", "month")]
filled = {c: s1[c].notna() & raw[c].isna() for c in common}
for c in sorted(common):
    n = int(filled[c].sum())
    if not n:
        continue
    how = methods.get(c, "설명서에 없음")
    parts = how.split("/")
    kind = ("미래정보 → 마스킹" if any(p in FUTURE for p in parts)
            else "실시간 정보 → 유지" if all(p in REALTIME for p in parts) else "확인 필요")
    put("보완 셀 목록", c, n, f"{how}: 설명서 {int(cells.get(c, 0))}셀, {kind}")

mask_cols = [c for c in common if any(p in FUTURE for p in str(methods.get(c, "")).split("/"))]
mask_d12 = {}
rng = np.random.default_rng(0)
off_rows = s2["P"] >= OFF0
ext_rows = (s2["P"] >= EXT0) & (s2["P"] <= J)
for c in mask_cols:
    vid = c[:4]
    lag = RELEASE_MONTHLY[vid]["lag"] if vid in RELEASE_MONTHLY else 0
    m1 = filled[c]
    m2 = m1.groupby(s1["region"]).shift(lag, fill_value=False).astype(bool)
    # 2차 값 = 1차 값을 시차만큼 민 것인지 확인
    chk = s1[c].groupby(s1["region"]).shift(lag)
    assert np.allclose(s2[c], chk, equal_nan=True), c
    x = s2[c].astype(float)
    g = s2["region"]

    def derive(v, first):
        if first:   # 마스킹 먼저 → 변환
            vm = v.mask(m2)
            return vm, vm - vm.groupby(g).shift(12)
        lv, d = v.copy(), v - v.groupby(g).shift(12)   # 변환 먼저 → 해당 행만 마스킹 (5차 순서)
        return lv.mask(m2), d.mask(m2)

    lv_ok, d_ok = derive(x, True)
    lv_bad, d_bad = derive(x, False)
    xp = x.where(~m2, x + rng.normal(0, 1, len(x)))
    lvp_ok, dp_ok = derive(xp, True)
    lvp_bad, dp_bad = derive(xp, False)
    same_ok = np.allclose(d_ok, dp_ok, equal_nan=True) and np.allclose(lv_ok, lvp_ok, equal_nan=True)
    leak = d_bad.notna() & ~np.isclose(d_bad, dp_bad, equal_nan=True)
    regions = s2.loc[m2, "region"].nunique()
    put("마스킹", f"{c} 기준기간 마스킹 셀", f"{int(m1.sum())}셀, {regions}개 지역",
        f"{span(s1.loc[m1, 'P'])}, 공표 시차 {lag}개월")
    put("마스킹", f"{c} 2차 시트 마스킹 행(결정월)", span(s2.loc[m2, "P"]), "수준 입력이 결측이 되는 결정월")
    put("마스킹", f"{c} 12개월 차분 결측 결정월(마스킹 먼저)", span(s2.loc[d_ok.isna() & d_bad.notna() | m2 & x.notna(), "P"]),
        "마스킹 행과 그 12개월 뒤 행")
    put("마스킹", f"{c} 5차 순서에서 보완값이 남는 결정월", span(s2.loc[leak, "P"]),
        f"{int(leak.sum())}행. 12개월 차분의 과거 기준값이 보완값")
    put("마스킹", f"{c} 교란 시험(마스킹 먼저)", "통과" if same_ok else "실패", "마스킹 셀을 난수로 바꿔도 파생값 불변")
    put("마스킹", f"{c} 주 분석(결정월 2016.01~) 영향 행", int(((d_ok.isna() & x.notna()) & off_rows).sum()),
        f"결정월 {span(s2.loc[(d_ok.isna() & x.notna()) & off_rows, 'P'])}의 12개월 차분 결측")
    put("마스킹", f"{c} 확장 실험(2012.01~2015.06) 영향 행", int(((d_ok.isna() & x.notna()) & ext_rows).sum()),
        "12개월 차분 결측 행")
    mask_d12[c] = (lv_ok, d_ok)

# 전신계열 CSI: 12개월 차분이 전신·현행 경계(2012.12|2013.01)를 가로지르는 결정월
for c in [c for c in common if methods.get(c) == "전신계열"][:1]:
    vid = c[:4]
    lag = RELEASE_MONTHLY[vid]["lag"] if vid in RELEASE_MONTHLY else 0
    pre = filled[c].groupby(s1["region"]).shift(lag, fill_value=False).astype(bool)
    prev = pre.groupby(s2["region"]).shift(12, fill_value=False).astype(bool)
    cross = prev & ~pre
    put("마스킹", "V061~V063 CSI 12개월 차분이 전신·현행 경계를 가로지르는 결정월", span(s2.loc[cross, "P"]),
        f"공표 시차 {lag}개월. 수준 입력은 전신계열 더미와 함께 유지")

# ============================================================ 2. 접합 규칙
def system(a, b):
    return "old" if b <= J else ("new" if a >= J else "cross")


old_cap = pd.read_csv(RAW_OLD)
old_cap = old_cap[old_cap.CLS_NM == "수도권"].groupby("WRTTIME_IDTFR_ID").DTA_VAL.mean()
old_months = sorted(per(m) for m in old_cap.dropna().index)
put("접합", "옛 수도권 집계 지수 기간", f"{old_months[0]}~{old_months[-1]}", f"{len(old_months)}개월")
T0 = EVAL[0]
for h in HS:
    rec = []
    for t in pd.period_range("2011-01", T0 - h, freq="M"):
        tgt = system(t - 1, t - 1 + h)
        mom = system(t - 7, t - 1)
        if tgt == mom == "new" and t >= OFF0:
            rec.append((t, "공식"))
        elif tgt == mom == "old" and t >= EXT0 and t - 7 >= old_months[0]:
            rec.append((t, "옛"))
        elif t >= EXT0:
            rec.append((t, "제외"))
    r = pd.DataFrame(rec, columns=["t", "k"])
    n_new, n_old = (r.k == "공식").sum(), (r.k == "옛").sum()
    put("접합", f"h={h} 첫 평가 시점(2018.01) 훈련 결정월",
        f"공식 {n_new} / 옛 {n_old} / 합계 {n_new + n_old}",
        f"공식 {span(r.t[r.k == '공식'])}, 옛 {span(r.t[r.k == '옛'])}")
    put("접합", f"h={h} 제외 결정월", span(r.t[r.k == "제외"]), "목표·A 구간이 접합을 가로지르거나 체계가 섞인 행")
    put("접합", f"h={h} 확장에 남는 2015년 결정월", span(r.t[(r.k == "옛") & (r.t.dt.year == 2015)]),
        "목표 구간이 2015.06에서 끝나는 행까지 옛 체계")
    use = set(r.t[r.k != "제외"])
    sel = s2["P"].isin(use)
    for c, (lv, d) in mask_d12.items():
        put("접합", f"h={h} 확장 훈련행의 {c[:4]} 관측률(수준/12개월 차분)",
            f"{lv[sel].notna().mean():.0%} / {d[sel].notna().mean():.0%}", "70% 미만이면 그 시점 학습에서 제외")

# ============================================================ 3. 선택형 단순기준
MIN_EVAL, WIN, MIN_TRAIN = 12, 24, 12
for h in HS:
    first_med = OFF0 + (MIN_TRAIN - 1) + h      # 중앙값 예측: 공식 훈련 정답 12개 이상
    sizes, med_ok, dflt = [], None, 0
    for T in EVAL:
        hi = T - h
        lo = max(OFF0, hi - (WIN - 1))
        n = (hi - lo).n + 1
        sizes.append(n)
        if n < MIN_EVAL:
            dflt += 1
        if med_ok is None and lo >= first_med:
            med_ok = T
    put("단순기준", f"h={h} 평가창 크기(첫 시점/최소/최대)", f"{sizes[0]}/{min(sizes)}/{max(sizes)}",
        f"기본 규칙 발동 시점 {dflt}개")
    put("단순기준", f"h={h} 학습 중앙값 후보의 첫 예측 결정월 / 비교 대상이 되는 첫 평가 시점",
        f"{first_med} / {med_ok}", "중앙값은 각 결정월에 그때 확인된 정답(12개 이상)만으로 계산")

# ============================================================ 4. 17개 시도 완전 패널
off = s2[(s2["Y_평가사용가능"] == 1) & s2[YCOL].notna()]
cnt = off.groupby("P")["region"].nunique()
put("패널", "공식 V001 기간", f"{cnt.index.min()}~{cnt.index.max()}", f"{len(cnt)}개월")
put("패널", "17개 시도가 모두 있는 달 / 빠진 달", f"{int((cnt == 17).sum())} / {int((cnt < 17).sum())}",
    "빠진 달이 0이면 평가 행 전체가 공통 평가행")
for h in HS:
    R = s2[YCOL].where(s2["Y_평가사용가능"] == 1)
    G = 100 * (R.groupby(s2["region"]).shift(1 - h) / R.groupby(s2["region"]).shift(1) - 1)
    ev = s2.loc[G.notna() & s2["P"].isin(EVAL)].groupby("P")["region"].nunique()
    put("패널", f"h={h} 평가 결정월(2018~2025) 중 17개 시도 정답이 모두 있는 달", f"{int((ev == 17).sum())}/{len(EVAL)}")

# ============================================================ 5. 광주·전남
v1 = pd.read_csv(RAW_RENT)
for name in ("광주", "전남", "광주광역시", "전라남도"):
    s = v1[v1.CLS_NM == name].groupby("WRTTIME_IDTFR_ID").DTA_VAL.mean().dropna()
    if len(s):
        put("광주·전남", f"공식 V001 원자료 '{name}'", f"{per(s.index.min())}~{per(s.index.max())}", f"{len(s)}개월")
gj = v1[v1.CLS_NM == "광주"].groupby("WRTTIME_IDTFR_ID").DTA_VAL.mean()
jn = v1[v1.CLS_NM == "전남"].groupby("WRTTIME_IDTFR_ID").DTA_VAL.mean()
both = pd.concat([gj, jn], axis=1, keys=["광주", "전남"]).dropna()
if len(both):
    tail = both.loc[202601:]
    put("광주·전남", "2026년 월별 값(광주/전남)",
        ", ".join(f"{per(i)} {a:.2f}/{b:.2f}" for i, (a, b) in tail.iterrows()), "두 시도 값이 다르면 따로 공표")

os.makedirs(OUT, exist_ok=True)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "plan_v6_rules.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 120)
print(out.to_string(index=False))
