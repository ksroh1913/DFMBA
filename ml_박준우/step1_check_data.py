# -*- coding: utf-8 -*-
"""
[1단계] 전처리본 17시도 2차 시트의 변수 시계열 점검 (2015.06 이후)

출력: output/01_변수시계열점검.csv
  열마다 기간, 결측률, 결측이 있는 지역·기간, 전국 공통 여부, 사용 여부와 제외 사유
"""
import os

import pandas as pd

from spec import CSI, EXCLUDED, HH, OUT, POP, SPEC, START, YCOL, load


def main():
    os.makedirs(OUT, exist_ok=True)
    raw, meta, fname = load()
    s = raw[raw.month >= START]
    used = {col: name for name, col, _, _ in SPEC}
    used.update({c: "주택가격전망CSI_수준" for c in CSI.values()})
    rows = []
    for c in raw.columns[4:]:
        v = s[c]
        miss = s[v.isna()].groupby("region").month.agg(["min", "max", "count"])
        note = "; ".join(f"{r} {a}~{b}({n}개월)" for r, (a, b, n) in miss.iterrows()) if 0 < len(miss) <= 4 \
            else (f"{len(miss)}개 지역" if len(miss) else "")
        m = meta.loc[c] if c in meta.index else {}
        if c == YCOL:
            use, why = "목표변수 + 모멘텀", ""
        elif c == "Y_평가사용가능":
            use, why = "표본 선택(=1인 행)", ""
        elif c in used:
            use, why = f"사용 → {used[c]}", ""
            if c in (HH, POP):
                use += " (다른 변수의 분모로도 사용)"
        else:
            use, why = "제외", EXCLUDED.get(c, "")
        rows.append({
            "컬럼명": c, "Master ID": m.get("Master ID", ""), "동인": m.get("동인", ""), "주기": m.get("주기", ""),
            "17시도 패널 내 값": "전국 공통" if (s.groupby("month")[c].nunique(dropna=True) <= 1).all() else "지역별",
            "첫 관측월(전체)": raw.loc[raw[c].notna(), "month"].min(),
            "마지막 관측월": raw.loc[raw[c].notna(), "month"].max(),
            "2015.06 이후 결측률": round(v.isna().mean(), 3), "결측 위치": note,
            "공표 시차 규칙": m.get("2차 시트 시차 규칙", ""), "사용 여부": use, "제외 사유": why,
        })
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "01_변수시계열점검.csv"), index=False, encoding="utf-8-sig")
    print(f"입력: {fname} / {len(s)}행 ({s.month.min()}~{s.month.max()}, {s.region.nunique()}개 시도)")
    print(f"열 {len(t)}개: 사용 {t['사용 여부'].str.startswith('사용').sum()}개, 제외 {(t['사용 여부'] == '제외').sum()}개")
    assert (t.loc[t["사용 여부"] == "제외", "제외 사유"] != "").all(), "제외 사유가 없는 열이 있음"


if __name__ == "__main__":
    main()
