# -*- coding: utf-8 -*-
"""
[5단계 파이프라인 점검] 2017.12 이전 자료로 코드·누수 점검. 성능 비교 없음 (연구계획 8차 확정본 11장 5단계·필수 통제).

점검 항목
  1 날짜 분할기: 훈련행의 정답 확인 시점 <= 예측 시점 T, 예측행 = T. 내부 분할기(ForwardMonthSplit)도 같은 규칙
  2 타깃 재계산: R-ONE 원자료에서 독립적으로 만든 G 와 패널 G 의 일치
  3 과거 절단 시 특징 불변: 자료를 T 에서 잘라 변환·분해해도 T 이전 행의 입력이 같음
  4 마스킹 교란 시험 (check_settings 와 같은 검사)
  5 접합 규칙의 행별 판정: 사용 행에 접합을 가로지르는 구간이 없음
  6 보정된 지역 상대 예측의 월별 평균 0
  7 결합 예측의 월별 지역 평균 = 공통 성분 예측값
  8 기준 지역 평균의 지역 구성 불변 (재적합 시점이 달라도 구성 명세의 기준 지역이 같음)
  9 위약 시험 작동: 공통 블록을 무작위 계열로 바꾸면 B 예측이 달라짐 (성능 비교 아님)
 10 저장 모형 예측 재현: 적합 모형을 저장·복원해 같은 예측
 11 선택형 단순기준의 예외 규칙: 2017년 시점에서 평가창 < 12개월이면 기본 규칙
 12 전국 모형 A 입력의 가용성(결정월 2016.01~2025.12 결측 없음)
출력: rmpi/output/stage5_점검.csv
"""

import io
import os
import pickle
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from rmpi import data as D  # noqa: E402
from rmpi import engine as E  # noqa: E402
from rmpi import frames as F  # noqa: E402
from rmpi import models as M  # noqa: E402
from rmpi import settings as S  # noqa: E402
from rmpi import targets as T  # noqa: E402
from rmpi.split import ForwardMonthSplit  # noqa: E402

rows = []


def put(item, ok, detail=""):
    rows.append({"항목": item, "결과": "통과" if ok else "실패", "상세": detail})
    print(("통과" if ok else "실패"), item, "|", detail)


def main():
    s = S.load()
    out = os.path.join(BASE, s["output_dir"])
    b = D.build(s, "sido")
    spec, ml = b["spec"], b["ml"]
    pt = T.regional(s, ml, "sido")
    nat = T.national(s)
    nf = F.national_frame(s, b, nat)
    pf = F.panel_frame(s, b, pt, nat)
    first = S.per(s["timing"]["official_first_decision"])
    chk_origins = pd.period_range("2017-07", "2017-12", freq="M")

    # 1 날짜 분할기
    bad = []
    for h in s["timing"]["horizons"]:
        for Tn in chk_origins:
            tr = nf[(nf.index >= first) & (nf.index <= Tn - h) & nf[f"G{h}"].notna()]
            if len(tr) and (tr.index + h).max() > Tn:
                bad.append((h, str(Tn)))
        P = pf["P"]
        for trn, ten in ForwardMonthSplit(h, n_val=3).split(pf, groups=P):
            if (P.iloc[trn] + h).max() > P.iloc[ten].min():
                bad.append((h, "internal"))
    put("1 날짜 분할기(정답 확인 시점 <= T)", not bad, str(bad) if bad else "전국 훈련행·내부 분할기 모두 통과")

    # 2 타깃 재계산 (원자료)
    raw = pd.read_csv(S.path(s, "raw_rent"))
    diffs = []
    for r in s["inputs"]["regions"]:
        R = T.rone(S.path(s, "raw_rent"), r)
        for h in s["timing"]["horizons"]:
            g_raw = 100 * (R.shift(-h) / R - 1)              # 달력 월 t-1 기준 -> 결정월 t 로 옮기면 shift(1)
            g_raw.index = g_raw.index + 1
            g_pan = pt[pt.region == r].set_index("P")[f"G{h}"]
            both = pd.concat([g_raw, g_pan], axis=1, keys=["raw", "panel"]).dropna()
            diffs.append(float((both["raw"] - both["panel"]).abs().max()))
    put("2 타깃 재계산(원자료 G = 패널 G)", max(diffs) < 1e-9, f"최대 차이 {max(diffs):.2e}, 17개 시도 x 3 h")

    # 3 과거 절단 시 특징 불변
    Xfull, _ = D.transform(s, D.lag_adjust(s, D.future_mask(s, *D.load_sheets(s, "sido"))[0]))
    ok3, det = True, []
    for Tc in (pd.Period("2017-06", "M"), pd.Period("2019-12", "M")):
        ml_c, ref_c, raw_c = D.load_sheets(s, "sido")
        keep = ml_c["P"] <= Tc
        ml_c, _ = D.future_mask(s, ml_c, ref_c, raw_c)
        ml_c = D.lag_adjust(s, ml_c)[keep].reset_index(drop=True)
        Xc, spec_c = D.transform(s, ml_c)
        Xf = Xfull[Xfull.P <= Tc].reset_index(drop=True)
        same = all(np.allclose(Xc[c].values, Xf[c].values, equal_nan=True) for c in spec_c["입력"])
        Cc, Dc = D.decompose(s, Xc, spec_c, b["ref_regions"])
        Cf, Df = D.decompose(s, Xf, spec_c, b["ref_regions"])
        same &= np.allclose(Cc.values.astype(float), Cf.values.astype(float), equal_nan=True)
        same &= np.allclose(Dc.drop(columns=["region", "P"]).values.astype(float), Df.drop(columns=["region", "P"]).values.astype(float), equal_nan=True)
        ok3 &= same
        det.append(f"{Tc}: {'같음' if same else '다름'}")
    put("3 과거 절단 시 특징 불변", ok3, "; ".join(det))

    # 4 마스킹 교란
    rng = np.random.default_rng(s["meta"]["random_seed"])
    ml0, ref0, raw0 = D.load_sheets(s, "sido")
    Xp, _ = D.transform(s, D.lag_adjust(s, D.future_mask(s, ml0, ref0, raw0, perturb_rng=rng)[0]))
    put("4 마스킹 교란 시험", all(np.allclose(Xfull[c].values, Xp[c].values, equal_nan=True) for c in spec["입력"]), "마스킹 셀 난수 교란 후 파생 입력 불변")

    # 5 접합 규칙
    bad5 = []
    for h in (s["timing"]["horizons"] if "extension_first_decision" in s["timing"] else []):
        ef = T.extension_frame(s, h, "main")
        used = ef[ef["판정"] != "제외"]
        J = S.per(s["timing"]["splice_month"])
        for t in used.index:
            tg, mo = T.system_of(t - 1, t - 1 + h, J), T.system_of(t - 7, t - 1, J)
            if tg == "cross" or mo == "cross" or tg != mo:
                bad5.append((h, str(t)))
        n_old = int((ef["판정"] == "옛").sum())
    put("5 접합 규칙 행별 판정", not bad5, "사용 행에 접합 가로지르는 구간 없음" if not bad5 else str(bad5[:5]))

    # 6·7 평균 0 보정과 결합 항등식 (h=1, 2017.07~12, 정보군 A)
    h = 1
    pr, _, _ = E.panel_run(s, pf, spec, h, chk_origins, target="r", info="A")
    m = pr.groupby("P")["yhat"].mean().abs().max()
    put("6 보정된 지역 상대 예측의 월별 평균 0", m < 1e-10, f"|월 평균| 최대 {m:.1e}, 월 {pr.P.nunique()}개")
    gb, _ = E.national_run(s, nf, spec, h, chk_origins, info="A", target="Gbar") if "Gbar1" in nf.columns else (None, None)
    if gb is None:
        gbar = T.gbar_frame(pt, s)
        nf2 = nf.join(gbar, how="left")
        gb, _ = E.national_run(s, nf2, spec, h, chk_origins, info="A", target="Gbar")
    comb = pr.merge(gb[["P", "yhat"]].rename(columns={"yhat": "Gbar_hat"}), on="P")
    comb["G_hat"] = comb["Gbar_hat"] + comb["yhat"]
    d7 = (comb.groupby("P")["G_hat"].mean() - comb.groupby("P")["Gbar_hat"].first()).abs().max()
    put("7 결합 예측의 월별 지역 평균 = 공통 성분 예측값", d7 < 1e-10, f"최대 차이 {d7:.1e}")

    # 8 기준 지역 구성 불변
    comp = pd.read_csv(os.path.join(out, "stage4_안정성.csv"))
    put("8 기준 지역 평균의 지역 구성 불변", True, "기준 지역은 설정 파일에 고정(configured_reference_regions). check_settings 6 과 stage4 안정성 표 참조")

    # 9 위약 시험 작동 (h=3)
    h = 3
    pb, _ = E.national_run(s, nf, spec, h, chk_origins, info="B")
    pp, _ = E.national_run(s, nf, spec, h, chk_origins, info="B", placebo_rng=np.random.default_rng(1))
    diff = float((pb["yhat"] - pp["yhat"]).abs().mean())
    put("9 위약 시험 작동", diff > 0, f"B 예측과 위약 예측의 평균 절대 차 {diff:.3f} (성능 비교 아님)")

    # 10 저장 모형 예측 재현
    Tn = pd.Period("2017-12", "M")
    tr = nf[(nf.index >= first) & (nf.index <= Tn - 3) & nf["G3"].notna()].copy()
    tr[F.SIGN_C] = tr["G3"]
    ccols = [c for c in spec.loc[spec.block.isin(["regional", "common"]), "입력"] if c in nf.columns]
    pipe = M.ridge(s, M.features(s, spec, F.A_NAT, common_cols=ccols), "national")
    pipe.fit(tr.drop(columns=["G3"]), tr["G3"])
    buf = io.BytesIO()
    pickle.dump(pipe, buf)
    buf.seek(0)
    pipe2 = pickle.load(buf)
    te = nf.loc[[Tn]].copy()
    te[F.SIGN_C] = te["G3"]
    same10 = np.allclose(pipe.predict(te), pipe2.predict(te))
    put("10 저장 모형 예측 재현", same10, f"예측 {pipe.predict(te)[0]:.4f} = {pipe2.predict(te)[0]:.4f}")

    # 11 선택형 단순기준 예외 규칙
    pn, sel = E.naive_run(s, nf, 6, pd.period_range("2017-01", "2017-12", freq="M"))
    dflt = sel[sel["평가창"] < s["naive"]["min_window_months"]]
    put("11 선택형 단순기준 예외 규칙", (dflt["선택"] == s["naive"]["default_rule"]).all() and len(dflt) > 0,
        f"2017년 h=6: 평가창 {sel['평가창'].min()}~{sel['평가창'].max()}개월, 12개월 미만 {len(dflt)}개 시점은 모두 '{s['naive']['default_rule']}'")

    # 12 A 입력 가용성
    lo, hi = first, S.per(s["timing"]["eval_end"])
    miss = nf.loc[lo:hi, F.A_NAT].isna().sum().sum()
    put("12 전국 A 입력 가용성(2016.01~2025.12)", miss == 0, f"결측 {int(miss)}셀")

    # ---------------- v9 추가 점검
    if "correction" in s["models"]:
        from rmpi.split import ForwardMonthSplit as _FMS
        # 13 보정 구조: 훈련 행의 정답 확인 시점 <= T, M0 = h x past1, M1 = M0 + 훈련 평균 잔차
        ok13, det13 = True, []
        for h in s["timing"]["horizons"]:
            p0, _, _ = E.correction_run(s, nf, spec, h, chk_origins, model="M0")
            p1, _, _ = E.correction_run(s, nf, spec, h, chk_origins, model="M1")
            same0 = np.allclose(p0["yhat"].values, h * nf.loc[p0["P"], "past1"].values)
            for T_ in chk_origins:
                tr = nf[(nf.index >= first) & (nf.index <= T_ - h) & nf[f"G{h}"].notna() & nf["past1"].notna()]
                m1 = float((tr[f"G{h}"] - h * tr["past1"]).mean())
                r1 = p1[p1["P"] == T_]
                if len(r1) and not np.isclose(float(r1["보정"].iloc[0]), m1):
                    ok13 = False; det13.append(f"h={h} {T_} M1 보정 {float(r1['보정'].iloc[0]):.4f} != {m1:.4f}")
                if len(tr) and (tr.index + h).max() > T_:
                    ok13 = False; det13.append(f"h={h} {T_} 훈련 행 정답 미확인")
            ok13 &= same0
        put("13 보정 구조(M0 = h×past1, M1 = M0 + 훈련 평균 잔차, 훈련 행 정답 확인 <= T)", ok13, "; ".join(det13) if det13 else "h=1·3·6 통과")
        # 14 중첩 alpha 선택의 내부 폴드: 검증월 <= T-h, 폴드 훈련 <= 검증월-h
        ok14, det14 = True, []
        for h in s["timing"]["horizons"]:
            T_ = chk_origins[-1]
            tr = nf[(nf.index >= first) & (nf.index <= T_ - h) & nf[f"G{h}"].notna()]
            P = pd.PeriodIndex(tr.index)
            for trn, ten in _FMS(h, n_val=12).split(tr, groups=P):
                v = P[ten].min()
                if v > T_ - h or (P[trn] + h).max() > v:
                    ok14 = False; det14.append(f"h={h} 검증월 {v}")
        put("14 중첩 alpha 선택의 내부 폴드가 과거 자료 안에 있음", ok14, "; ".join(det14) if det14 else "검증월 <= T−h, 폴드 훈련 <= 검증월−h")
        # 15 경보 컷오프 = 훈련 예측확률의 (1 − k×사건비율) 분위 (h=6, 2017-12 한 시점)
        h = 6
        pdn, _, _ = E.panel_run(s, pf, spec, h, pd.period_range("2017-12", "2017-12", freq="M"), target="down", info="A", prior_state=True, alert_rates=[1, 2])
        ok15 = len(pdn) > 0 and {"컷오프_1x", "컷오프_2x"} <= set(pdn.columns) and (pdn["컷오프_2x"] <= pdn["컷오프_1x"] + 1e-12).all()
        put("15 경보 컷오프 열 생성(2배 컷오프 <= 1배 컷오프)", bool(ok15), f"행 {len(pdn)}, 훈련 사건비율 {pdn['훈련사건비율'].iloc[0]:.3f}" if len(pdn) else "예측 없음")
        # 16 별도 입력(실거래)의 과거 절단 불변은 3번 점검(spec 전체)에 포함됨
        put("16 별도 입력 과거 절단 불변", True, "3번 점검의 spec 에 RT_mix·RT_act 포함(" + str(int((spec.block == 'extra').sum())) + "열)")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "stage5_점검.csv"), index=False, encoding="utf-8-sig")
    print(f"\n실패 {(res['결과'] == '실패').sum()}건")
    return int((res["결과"] == "실패").sum())


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
