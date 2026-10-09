# -*- coding: utf-8 -*-
"""
모형 입력표(xlsx, 수식)와 같은 규칙으로 '값'을 계산해 학습용 CSV 를 만든다 — model_sheet.py 의 수식 열과 1:1 대응.

출력: analysis/output/모형입력표_10차_values.csv        (17개 시도 × 결정월; region, 결정월, 구간, 타깃 8열, 설명변수 변환 열)
      analysis/output/모형입력표_10차_values_서울구.csv  (서울 25개 구 보조 패널, 같은 구조)
규칙(model_sheet.py 와 동일):
  타깃  R_last = R(t−1), past_k = 100·(R_last_t / R_last_{t−k} − 1), G_h = 100·(R_last_{t+h} / R_last_t − 1), r_h = G_h − 같은 결정월 패널 평균(모든 지역 있을 때만)
  설명  결정 로그 '결정' 행의 변환을 raw 에 적용(variable_card.candidates), 공표시차만큼 뒤로 밀어 '결정월에 아는 값'. 저빈도는 기간 끝 달 배치 + 시차 + 다음 공표까지 유지(계단).
        전국 단일 계열은 모든 지역에 복제, 구 raw 가 없는 변수는 서울 값 복제(결정 로그 지역단위에 '서울 값 복제'라고 적은 것만)
사용: PYTHONUTF8=1 python analysis/model_values.py [--check]   # --check: xlsx 의 값 열(계단·raw)과 대조
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import variable_card as vc  # noqa: E402
from model_sheet import ANALYSIS_START, BUFFER_START, _stepfill_monthly  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_SIDO = os.path.join(BASE, "analysis", "output", "모형입력표_10차_values.csv")
OUT_GU = os.path.join(BASE, "analysis", "output", "모형입력표_10차_values_서울구.csv")


def target_values(R, regions, months):
    """R: 달력월 × 지역 지수 → 결정월 × 지역 타깃 8열 (dict of DataFrame)"""
    C = R.reindex(columns=regions).shift(1).reindex(months)          # 결정월 t 에 아는 마지막 지수 R(t−1)
    out = {"R_last": C}
    for k in (1, 3, 6):
        out[f"past{k}"] = 100 * (C / C.shift(k) - 1)
    for h in (3, 6):
        G = 100 * (C.shift(-h) / C - 1)
        out[f"G{h}"] = G
        out[f"r{h}"] = G.sub(G.mean(axis=1).where(G.notna().all(axis=1)), axis=0)
    return out


def explanatory_values(dec, ctx, level, months):
    """결정 로그의 '결정' 행마다 (열 이름, 결정월 × 지역 값) — model_sheet._write_x 와 같은 분기"""
    regions = list(vc.REGIONS) if level == "sido" else list(vc.GU_LIST)
    if level == "gu":                                                  # 천세대당·천명당 분모는 구 세대수·인구 (model_sheet 의 helper 와 같음)
        ctx = dict(ctx, hh=ctx.get("hh_gu", ctx["hh"]), pop=ctx.get("pop_gu", ctx["pop"]))
    cols, notes = {}, []
    for _, row in dec.iterrows():
        vid = row["ID"]
        name = re.sub(r"\s+", "", str(row["변수명"]))[:20]
        files = vc.find_raw_files(vid)
        if not files:
            notes.append(f"{vid}: raw 없음"); continue
        parsed = [vc.parse_raw_file(f, vid) for f in files]
        replicated = False
        der = vc.derived_raw(vid, parsed, level) or (vc.derived_raw(vid, parsed, "sido") if level == "gu" else None)
        if der is not None:
            wide, dlabel, _ = der
            name = f"{name}_{dlabel}"
            if level == "gu" and not any(c in regions for c in wide.columns):
                replicated = True
        else:
            items_sido = vc.select_items(vid, parsed, "sido")
            force_rep = level == "gu" and "서울 값 복제" in str(row.get("지역단위", ""))
            items = items_sido if (level == "sido" or force_rep) else vc.select_items(vid, parsed, "gu")
            if force_rep and items_sido:
                replicated = True
            if not items:
                if level == "gu" and items_sido:
                    items, replicated = items_sido, True
                else:
                    notes.append(f"{vid}: 항목 없음"); continue
            wide = items[list(items)[0]]
        kind = wide.attrs.get("kind", "M")
        key = str(row["변환"]).strip()
        cands = vc.candidates(vid, wide, kind, vc.vtype_of(vid), ctx)
        if key not in cands:
            key = key.split(" ")[0]
        if key not in cands:
            notes.append(f"{vid} ({'시도' if level == 'sido' else '서울 구'} 패널): 변환 '{row['변환']}' 을 {'월' if kind == 'M' else '저빈도'} 표에 만들 수 없어 열 생략"); continue
        lag = int(row["공표시차_개월"]) if str(row["공표시차_개월"]).lstrip("-").isdigit() else 0
        lag += int(row["추가시차"]) if str(row["추가시차"]).lstrip("-").isdigit() else 0
        lag = max(lag, 0)
        tw = cands[key]
        tw = _stepfill_monthly(tw, kind, months) if kind != "M" else tw
        tw = tw.reindex(months).shift(lag)
        nat_only = list(wide.columns) == ["전국"]
        src = "전국" if nat_only else ("서울" if replicated else None)
        if src is not None and src in tw.columns:
            vals = pd.DataFrame({r: tw[src] for r in regions})
        else:
            vals = tw.reindex(columns=regions)
        cols[f"{vid}_{name}_{key}"] = vals
    return cols, notes


def build(level, ctx, dec):
    regions = list(vc.REGIONS) if level == "sido" else list(vc.GU_LIST)
    R = ctx["targets"]["R_panel"] if level == "sido" else ctx["targets_gu"]["R_panel"]
    last = ctx["targets"]["R_panel"].dropna(how="all").index.max() + 1
    months = pd.period_range(BUFFER_START, last, freq="M")
    tcols = target_values(R, regions, months)
    xcols, notes = explanatory_values(dec, ctx, level, months)
    idx = pd.MultiIndex.from_product([regions, months], names=["region", "결정월"])
    df = pd.DataFrame(index=idx)
    df["구간"] = ["분석" if m >= ANALYSIS_START else "버퍼" for _, m in idx]
    for k, v in {**tcols, **xcols}.items():
        df[k] = v.reindex(columns=regions).T.stack(dropna=False).reindex(idx).values
    df = df.reset_index()
    df["결정월"] = df["결정월"].astype(str)
    return df, notes


def main():
    ctx = vc.load_context()
    log = vc.log_load()
    dec = log[(log["상태"] == "결정") & (log["ID"] != "V001")]
    df, notes = build("sido", ctx, dec)
    df.to_csv(OUT_SIDO, index=False, encoding="utf-8-sig")
    print(f"{os.path.relpath(OUT_SIDO, BASE)}: {df.shape[0]}행 × {df.shape[1]}열 (타깃 8 + 설명변수 {df.shape[1] - 11})")
    for n in notes:
        print("  [주의]", n)
    if ctx.get("targets_gu"):
        dg, notes_g = build("gu", ctx, dec)
        dg.to_csv(OUT_GU, index=False, encoding="utf-8-sig")
        print(f"{os.path.relpath(OUT_GU, BASE)}: {dg.shape[0]}행 × {dg.shape[1]}열")
        for n in notes_g:
            print("  [주의]", n)
    if "--check" in sys.argv:
        check(df)


def check(df):
    """xlsx 의 값 열(원값·계단 열은 수식이 아니라 값)과 대조 — 설명변수 시트의 '(계단)' 열은 시차 반영 전이라 열 이름이 다르므로 raw 열만 비교"""
    from openpyxl import load_workbook
    wb = load_workbook(os.path.join(BASE, "analysis", "output", "모형입력표_10차.xlsx"), read_only=True)
    ws = wb["타깃"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    sheet = pd.DataFrame(rows, columns=[c.value for c in next(wb["타깃"].iter_rows(min_row=1, max_row=1))])
    m = df.merge(sheet[["region", "결정월", "R_last"]], on=["region", "결정월"], suffixes=("", "_xlsx"))
    diff = (m["R_last"] - m["R_last_xlsx"]).abs()
    print(f"[대조] 타깃 R_last: 비교 {int(diff.notna().sum())}셀, 최대 차이 {float(diff.max()):.6f}")


if __name__ == "__main__":
    main()
