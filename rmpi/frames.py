# -*- coding: utf-8 -*-
"""모형 입력표 조립 (연구계획 8차 확정본 1장 정보군 표).

national_frame: 결정월 단위. A = 전국 월세 과거 1·3·6개월 변화 + 전국 매매·전세 6개월 변화율. 공통 블록 = C(기준 지역 평균 + 전국 공통).
panel_frame:    시도×결정월. A_지역 = 시도 과거 1·6개월 상대 월세 변화 + 매매·전세 6개월 변화율 월내 편차 (RQ2),
                A_시도 = 시도 과거 1·3·6개월 월세 변화 + 해당 시도 매매·전세 6개월 변화율 (RQ3·결합).
                지역 블록 = D(월내 편차), 공통 블록 = C(월별로 병합).
부호 결정용 열: __ysign_common = G_전국(t,h), __ysign_regional = r(i,t,h). RMPI 변환기가 쓰고 출력에서 뺀다.
"""

import pandas as pd

SIGN_C, SIGN_R = "__ysign_common", "__ysign_regional"
A_NAT = ["past1", "past3", "past6", "V002|6개월%", "V026|6개월%"]
A_REL = ["rel_past1", "rel_past6", "V002|6개월%|월내편차", "V026|6개월%|월내편차"]
A_REG = ["past1", "past3", "past6", "V002|6개월%", "V026|6개월%"]


def national_frame(s, b, nat):
    """b: data.build 결과, nat: targets.national 결과(결정월 index)."""
    C = b["C"]
    f = nat.join(C, how="left")
    f.index.name = "P"
    return f


def panel_frame(s, b, pt, nat):
    """pt: targets.regional 결과(패널), nat: 전국 표. 반환: 패널 DataFrame(region, P, ...)"""
    X, C, D, A = b["X"], b["C"], b["D"], b["A_price"]
    f = pt.merge(A, on=["region", "P"], how="left")
    # 상대 월세 변화 (같은 달 17개 시도 평균 대비)
    for k in (1, 6):
        m = f.groupby("P")[f"past{k}"].transform("mean")
        full = f.groupby("P")[f"past{k}"].transform("count") == f["region"].nunique()
        f[f"rel_past{k}"] = (f[f"past{k}"] - m).where(full)
    dcols = [c for c in D.columns if c not in ("region", "P")]
    f = f.merge(D[["region", "P"] + dcols].rename(columns={c: f"D|{c}" for c in dcols}), on=["region", "P"], how="left")
    Cm = C.copy()
    Cm.columns = [f"C|{c}" for c in Cm.columns]
    f = f.merge(Cm, left_on="P", right_index=True, how="left")
    gn = nat[[f"G{h}" for h in s["timing"]["horizons"]]].rename(columns=lambda c: f"Gnat{c[1:]}")
    f = f.merge(gn, left_on="P", right_index=True, how="left")
    return f


def regional_cols(f):
    return [c for c in f.columns if c.startswith("D|")]


def common_cols(f):
    return [c for c in f.columns if c.startswith("C|")]


def spec_for(spec, prefix=""):
    """입력 명세의 '입력' 이름에 접두사를 붙인 사본 (패널표의 'C|'·'D|' 열 이름에 맞춤)"""
    sp = spec.copy()
    sp["입력"] = prefix + sp["입력"]
    return sp
