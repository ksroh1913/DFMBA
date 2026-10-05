# -*- coding: utf-8 -*-
"""8차 대용계열 확장 실험의 구현 결함 확인 (9차 설계안 5장, 8차 보고서 정오표의 근거; 정오표 원문은 rmpi/output_v9/참고_8차/README.md).
재실행 주의: 8차 설정의 입력(데이터취합_전처리_20261004.xlsx)은 저장소에서 지웠다. analysis/output/plan_v9_extension_check*.csv 는 동결 결과이며,
다시 돌리려면 git 이력(커밋 de16c0d)에서 복원한다.

계획(8차 5장 확장 실험)은 공식 구간 훈련자료로 고른 편입 변수를 확장 모형에도 유지한다고 했으나, engine.national_run 은
옛 체계 행을 합친 훈련표 전체를 RMPI.fit 에 넘기고, RMPI.fit 은 그 표 전체에서 관측률(70% 필터)·표준화 척도·부호를 다시 계산한다.
그래서 2012~2015 년에 결측이 많은 변수가 확장 모형에서만 빠지고 부호·척도도 달라진다. 첫 평가 시점(2018-01)에서 주 모형과
확장 모형(w=1.0)의 구성명세를 나란히 비교해 저장한다. 8차 설정(20261004 입력) 그대로.

출력: analysis/output/plan_v9_extension_check.csv (h, 블록, 입력, 유지·관측률·적용부호·표준편차 의 주/확장 값), 화면 요약
"""

import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from rmpi import settings as S, data as D, targets as T, frames as F, engine as E  # noqa: E402

OUT = os.path.join(BASE, "analysis", "output")


def main():
    s = S.load("config/plan_v8_settings.yaml")   # 8차 설정 전용 (plan_v9 에는 extension 항목이 없음)
    b = D.build(s, "sido")
    spec, ml = b["spec"], b["ml"]
    pt = T.regional(s, ml, "sido")
    nat = T.national(s)
    nf = F.national_frame(s, b, nat).join(T.gbar_frame(pt, s), how="left")
    origins = pd.period_range(S.per(s["timing"]["eval_start"]), S.per(s["timing"]["eval_start"]) + 1, freq="M")
    rows, summary = [], []
    cols = ["유지", "관측률", "적용부호", "표준편차"]
    for h in s["timing"]["horizons"]:
        p0, c0 = E.national_run(s, nf, spec, h, origins, info="B")
        ef = T.extension_frame(s, h, "main")
        p1, c1 = E.national_run(s, nf, spec, h, origins, info="B", extension=ef, ext_weight=1.0)
        a, bb = c0[0], c1[0]
        m = a[["블록", "입력"] + cols].merge(bb[["블록", "입력"] + cols], on=["블록", "입력"], suffixes=("_주", "_확장"))
        m.insert(0, "h", h)
        rows.append(m)
        kept0, kept1 = int(a["유지"].sum()), int(bb["유지"].sum())
        both = m[m["유지_주"] & m["유지_확장"]]
        flipped = int((both["적용부호_주"] != both["적용부호_확장"]).sum())
        obs_diff = int((abs(m["관측률_주"] - m["관측률_확장"]) >= 0.1).sum())
        n_old = int(p1["훈련월_옛체계"].iloc[0]) if len(p1) else 0
        summary.append({"h": h, "시점": str(origins[0]), "입력수": len(m), "유지_주": kept0, "유지_확장": kept1,
                        "둘다유지중_부호다름": flipped, "관측률차0.1이상": obs_diff, "훈련행_주": int(p0["훈련행수"].iloc[0]),
                        "훈련행_확장": int(p1["훈련행수"].iloc[0]), "그중_옛체계": n_old})
        print(f"h={h} {origins[0]}: 유지 주 {kept0}/{len(m)} vs 확장 {kept1}/{len(m)}; 둘 다 유지 중 부호 다름 {flipped}; "
              f"관측률 차 0.1 이상 {obs_diff}; 훈련행 주 {summary[-1]['훈련행_주']} vs 확장 {summary[-1]['훈련행_확장']} (옛 체계 {n_old})", flush=True)
    res = pd.concat(rows, ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, "plan_v9_extension_check.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(os.path.join(OUT, "plan_v9_extension_check_요약.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    sys.exit(main())
