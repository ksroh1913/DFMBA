# -*- coding: utf-8 -*-
"""
※ 설정 점검 스크립트. 설정 파일(RMPI_SETTINGS, 기본 config/plan_v9_settings.yaml)의 값을 자료와 대조한다. 예측 성능은 계산하지 않는다.

점검 항목
  1. 설정의 열 이름이 2차 시트에 있는지, 제외 열이 실제로 있는지
  2. 사건 경계: 복리식 100x[(1+a)^(h/12)-1] 과 문서 4장 수치의 일치
  3. 보완 셀 수(V003·V057·V059, 17시도·서울구)와 설명서 수치의 일치
  4. 시차 보정: V060 분기 실적치가 분기 종료 다음 달부터, V072 가 1개월 늦게 들어가는지
  5. CSI 지역 유형 매핑이 17개 시도를 정확히 한 번씩 덮는지
  6. 기준 지역: 변환 후 입력으로 최초 훈련기간(2016.01~2017.12)에서 재계산한 집합 = 설정의 집합
  7. 교란 시험: 마스킹 대상 셀을 난수로 바꿔도 변환·분해 결과가 같은지
  8. 관측률: 최초 훈련기간 입력 열별 관측률 (70% 미만 열 표시)
출력: analysis/output/check_settings_<settings_version>.csv (항목, 결과, 상세; v9.0 → check_settings_v9.0.csv, v8 설정이면 check_settings.csv)
"""

import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from rmpi import data as D  # noqa: E402
from rmpi import settings as S  # noqa: E402

rows = []


def put(item, ok, detail=""):
    rows.append({"항목": item, "결과": "통과" if ok else "불일치", "상세": detail})


def main():
    s = S.load()
    put("설정 파일", True, f"{s['_path']} sha256 {s['_sha']}, version {s['meta']['settings_version']}")
    ml, ref, raw = D.load_sheets(s, "sido")

    # 1. 열 이름
    cols = set(ml.columns)
    missing = [v["col"] for k, v in s["variables"].items() if k != "CSI" and v["col"] not in cols]
    put("1 설정 열 존재(17시도 2차)", not missing, f"변수 {len(s['variables'])}개, 없음: {missing}")
    missing_ex = [c for c in s["excluded_columns"] if c not in cols]
    put("1 제외 열 존재", not missing_ex, f"제외 {len(s['excluded_columns'])}열, 없음: {missing_ex}")
    used = {v["col"] for k, v in s["variables"].items() if k != "CSI"} | set(s["csi_columns"].values()) | set(s.get("rt_columns", {}).values())
    others = sorted(c for c in cols if c not in used and c not in s["excluded_columns"]
                    and c not in D.ID_COLS + ["P", s["inputs"]["official_flag"]] and not c.startswith("Y_"))
    put("1 설정에 없는 설명변수 열", not others, f"{others}")

    # 2. 사건 경계
    ev = s["events"]
    bad = []
    for h in s["timing"]["horizons"]:
        dn = 100 * ((1 + ev["down_annual"]) ** (h / 12) - 1)
        up = 100 * ((1 + ev["up_annual"]) ** (h / 12) - 1)
        if round(dn, 3) != ev["expected_thresholds"]["down"][h] or round(up, 3) != ev["expected_thresholds"]["up"][h]:
            bad.append((h, round(dn, 3), round(up, 3)))
    put("2 사건 경계(복리식)", not bad, "급락 " + "/".join(f"{100 * ((1 + ev['down_annual']) ** (h / 12) - 1):.3f}" for h in (1, 3, 6))
        + ", 급등 " + "/".join(f"{100 * ((1 + ev['up_annual']) ** (h / 12) - 1):.3f}" for h in (1, 3, 6)) + (f" 불일치 {bad}" if bad else ""))

    # 3. 보완 셀 수
    for panel in ("sido", "gu"):
        m, r, w = (ml, ref, raw) if panel == "sido" else D.load_sheets(s, panel)
        _, log = D.future_mask(s, m, r, w)
        exp = s["masking"]["expected_cells"][panel]
        got = dict(zip(log["변수"], log["보완셀(기준기간)"]))
        put(f"3 보완 셀 수({panel})", got == exp, f"{got} (설명서 {exp})")

    # 4. 시차 보정
    m2 = D.lag_adjust(s, D.future_mask(s, ml, ref, raw)[0])
    v60 = D.colname(s, "V060")
    seoul0 = ml[ml.region == "서울"].set_index("P")[v60]
    seoul1 = m2[m2.region == "서울"].set_index("P")[v60]
    q4 = pd.Period("2024-10", "M")
    ok60 = (seoul0.loc[q4] == -42) and (seoul1.loc[q4 + 3] == -42) and (seoul1.loc[q4 + 2] == seoul0.loc[q4 - 1])
    put("4 V060 시차 보정", bool(ok60), f"2024Q4 실적 -42: 기존 {q4}부터 -> 보정 {q4 + 3}부터 (보정 전 {q4 + 2} 값 {seoul1.loc[q4 + 2]})")
    v72 = D.colname(s, "V072")
    s0 = ml[ml.region == "서울"].set_index("P")[v72]
    s1 = m2[m2.region == "서울"].set_index("P")[v72]
    p = pd.Period("2025-12", "M")
    put("4 V072 시차 보정", bool(s0.loc[p] == 93.9 and s1.loc[p] == 93.6 and s1.loc[p + 1] == 93.9),
        f"2024년 값 93.9: 기존 {p}부터 -> 보정 {p + 1}부터 ({p} 행은 {s1.loc[p]})")

    # 5. CSI 매핑
    cover = sum((s["csi_region_type"][k] for k in s["csi_region_type"]), [])
    put("5 CSI 지역 유형 매핑", sorted(cover) == sorted(s["inputs"]["regions"]) and len(cover) == 17,
        f"{ {k: len(v) for k, v in s['csi_region_type'].items()} }")

    # 6. 기준 지역 재계산
    X, spec = D.transform(s, m2, "sido")
    calc = D.reference_regions(s, X, spec)
    conf = D.configured_reference_regions(s, spec)
    diff = {k: (len(calc[k]), sorted(set(conf[k]) ^ set(calc[k]))) for k in conf if set(conf[k]) != set(calc[k])}
    put("6 기준 지역 집합", not diff, "불일치 " + str(diff) if diff else
        "설정과 일치. 17개 미만: " + str({k: [g for g in s['inputs']['regions'] if g not in v] for k, v in calc.items() if len(v) < 17}))
    lo, hi = S.per(s["timing"]["official_first_decision"]), S.per(s["timing"]["first_train_end"])
    sub = X[(X.P >= lo) & (X.P <= hi)]
    empty = [k for k, v in calc.items() if not v]
    put("6 기준 지역이 없는 입력", not empty, str(empty))
    allmiss = {k: int((sub.pivot(index="P", columns="region", values=k).notna().sum(axis=1) == 0).sum())
               for k in calc}
    allmiss = {k: v for k, v in allmiss.items() if v}
    put("6 최초 훈련기간에 모든 지역이 함께 빠진 달", True, str(allmiss) if allmiss else "없음")

    # 7. 교란 시험
    rng = np.random.default_rng(s["meta"]["random_seed"])
    mlp, _ = D.future_mask(s, ml, ref, raw, perturb_rng=rng)
    Xp, _ = D.transform(s, D.lag_adjust(s, mlp), "sido")
    same = all(np.allclose(X[c].values, Xp[c].values, equal_nan=True) for c in spec["입력"])
    put("7 교란 시험(마스킹 셀 난수 교란 후 파생 입력 불변)", same, f"입력 {len(spec)}열 비교")
    C, Dd = D.decompose(s, X, spec, conf)
    Cp, Dp = D.decompose(s, Xp, spec, conf)
    put("7 교란 시험(분해 결과 불변)", np.allclose(C.values.astype(float), Cp.values.astype(float), equal_nan=True)
        and np.allclose(Dd.drop(columns=["region", "P"]).values.astype(float), Dp.drop(columns=["region", "P"]).values.astype(float), equal_nan=True))

    # 8. 관측률 (최초 훈련기간, 분해 후 입력 기준)
    Csub = C.loc[lo:hi]
    Dsub = Dd[(Dd.P >= lo) & (Dd.P <= hi)]
    low = {}
    for c in C.columns:
        r = Csub[c].notna().mean()
        if r < s["rmpi"]["obs_rate_min"]:
            low[f"공통 {c}"] = round(r, 2)
    for c in Dsub.columns.drop(["region", "P"]):
        r = Dsub[c].notna().mean()
        if r < s["rmpi"]["obs_rate_min"]:
            low[f"지역 {c}"] = round(r, 2)
    put("8 최초 훈련기간 관측률 70% 미만 입력", True, str(low) if low else "없음")
    put("8 입력 열 수", True, f"공통 블록 {C.shape[1]}열(지역 변수 기준 지역 평균 포함), 지역 블록 {Dsub.shape[1] - 2}열, 가격 추세 {int((spec.block == 'price_trend').sum())}열")

    # ---------------- v9 추가 점검 (설정에 extra_inputs 가 있을 때만)
    if "extra_inputs" in s:
        import math
        for panel in ("sido", "gu"):
            mlx = m2 if panel == "sido" else D.lag_adjust(s, D.future_mask(s, *D.load_sheets(s, panel))[0])
            Xr, spr = D.rt_inputs(s, mlx, panel)
            first = S.per(s["masking"]["rt_break_first_row"])
            n_mask = s["masking"]["rt_mask_rows"]["sum3_d12"]
            win = Xr[(Xr.P >= first) & (Xr.P < first + n_mask)]
            cells = int(win[[c for c in spr["입력"]]].isna().sum().sum())
            exp_cells = s["masking"]["rt_expected_mask_cells"][panel]
            after = Xr[Xr.P == first + n_mask][[c for c in spr["입력"]]].isna().sum().sum()
            put(f"9 실거래 경계 결측 셀({panel})", cells == exp_cells and after == 0,
                f"{first}~{first + n_mask - 1} 결측 {cells}셀 (설정 {exp_cells}), 다음 달 결측 {int(after)}")
            asof = S.per(s["inputs"]["data_asof"]); k = s["masking"]["rt_truncate_last_months"]
            tail = Xr[Xr.P >= asof - (k - 1)][[c for c in spr["입력"]]]
            put(f"9 실거래 최신 {k}개월 결측(data_asof {asof}, {panel})", bool(tail.isna().all().all()), f"행 {len(tail)}")
            lo_, hi_ = S.per(s["timing"]["official_first_decision"]), S.per(s["timing"]["first_train_end"])
            obs = Xr[(Xr.P >= lo_) & (Xr.P <= hi_)][[c for c in spr["입력"]]].notna().mean().round(3).to_dict()
            put(f"9 실거래 입력 최초 훈련기간 관측률({panel})", all(v >= s["rmpi"]["obs_rate_min"] for v in obs.values()), str(obs))
        # 10 in_common 플래그와 추세 제외 목록
        Xs, sps = D.transform(s, m2, "sido")
        off = set(sps.loc[~sps["in_common"], "입력"])
        trend = set(s["rmpi"]["trend_exclusion"]["applied"])
        put("10 S1 추세 제외 입력이 공통 블록에서 빠짐", trend <= off, f"추세 제외 {sorted(trend)}; 공통 제외 전체 {sorted(off)}")
        put("10 별도 입력은 공통 블록에 없음", all(x in off for x in sps.loc[sps.block == "extra", "입력"]), "")
        # 11 중첩 격자: 1단계 25점(0.25 로그 간격) + 2단계 ±0.5/0.125 → 새 점 4개
        g = s["models"]["ridge_national"]["nested"]
        g1 = np.logspace(math.log10(g["stage1"]["min"]), math.log10(g["stage1"]["max"]), g["stage1"]["n"])
        step1 = round(float(np.log10(g1[1]) - np.log10(g1[0])), 6)
        offs = np.arange(-g["stage2"]["half_width_log10"], g["stage2"]["half_width_log10"] + 1e-9, g["stage2"]["step_log10"])
        new = [o for o in offs if not any(abs((o / step1) - round(o / step1)) < 1e-9 for _ in [0])]
        put("11 중첩 격자 규칙", step1 == 0.25 and len(new) == 4 and g["rule"] == "cv_min" and g["min_folds"] == 4,
            f"1단계 간격 {step1}, 2단계 새 점 {len(new)}개, 규칙 {g['rule']}, 폴드 하한 {g['min_folds']}, fallback {g['fallback_alpha']}")
        # 12 보정 구조·경보 설정 존재
        c = s["models"]["correction"]
        put("12 보정 구조 설정", c["baseline"] == "mom1" and set(c["models"]) == {"M0", "M1", "M2", "M3"} and "M1_slope" in c["sensitivities"],
            f"baseline {c['baseline']}, 모형 {sorted(c['models'])}, 민감도 {sorted(c['sensitivities'])}")
        a = s["events"]["alert_comparison"]
        put("12 경보 비교 설정", a["target_rates_x_event_rate"] == [1, 2] and "동월지역쌍_AUC" in a["report"], str(a["target_rates_x_event_rate"]))
        put("12 산출물 폴더 분리", s["output_dir"] != "rmpi/output", s["output_dir"])

    out = pd.DataFrame(rows)
    os.makedirs(os.path.join(BASE, "analysis", "output"), exist_ok=True)
    ver = str(s["meta"]["settings_version"])
    fname = "check_settings.csv" if ver.startswith("v8") else f"check_settings_{ver.split('-')[0]}.csv"
    out.to_csv(os.path.join(BASE, "analysis", "output", fname), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 160)
    print(out.to_string(index=False))
    n_bad = int((out["결과"] == "불일치").sum())
    print(f"\n불일치 {n_bad}건")
    return n_bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
