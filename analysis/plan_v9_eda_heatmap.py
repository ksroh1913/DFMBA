# -*- coding: utf-8 -*-
"""[탐색 보조] 변수 간 Spearman 상관 행렬(부호 포함)을 탐색 창(결정월 <= 2017-12)에서 계산해 저장하고 히트맵 그림을 만든다.
plan_v9_eda.py 와 같은 입력(RMPI 구성 입력; 가격 추세·실거래 후보 제외), 쌍마다 겹치는 관측 24개월 이상.
출력: analysis/output/plan_v9_eda_상관행렬_공통.csv, _지역.csv, _겹침_공통.csv; rmpi/output_v9/fig0_5_상관행렬_공통.png"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from rmpi import data as Dm, settings as S  # noqa: E402
from rmpi import viz as V  # noqa: E402

TRAIN_END = pd.Period("2017-12", "M")
MIN_OVERLAP = 24


def short(col):
    return re.sub(r"^V\d+[a-c]?_", "", str(col))


def main():
    s = S.load()
    out = os.path.join(BASE, "analysis", "output")
    b = Dm.build(s, "sido")
    spec, C, D = b["spec"], b["C"], b["D"]
    sp = spec[spec["block"].isin(["regional", "common"])].copy()
    sp["이름"] = sp["원열"].map(short)
    cols = [c for c in sp["입력"] if c in C.columns]
    Ct = C.loc[C.index <= TRAIN_END, cols]
    rho = Ct.corr(method="spearman", min_periods=MIN_OVERLAP)
    nov = pd.DataFrame({a: {bb: int((Ct[a].notna() & Ct[bb].notna()).sum()) for bb in cols} for a in cols}).loc[cols, cols]
    rho.to_csv(os.path.join(out, "plan_v9_eda_상관행렬_공통.csv"), encoding="utf-8-sig")
    nov.to_csv(os.path.join(out, "plan_v9_eda_겹침_공통.csv"), encoding="utf-8-sig")
    dcols = [c for c in sp.loc[sp["block"] == "regional", "입력"] if c in D.columns]
    Dt = D[D["P"] <= TRAIN_END]
    rho_r = Dt[dcols].corr(method="spearman", min_periods=MIN_OVERLAP)
    rho_r.to_csv(os.path.join(out, "plan_v9_eda_상관행렬_지역.csv"), encoding="utf-8-sig")
    # 그림 (공통 블록): 동인 순서, 같은 변수의 두 형태는 빈칸
    V.setup()
    order = sp.set_index("입력").loc[cols]
    ids = order["ID"].tolist()
    M = rho.loc[cols, cols].values.copy()
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j and ids[i] == ids[j]:
                M[i, j] = np.nan
    fig, ax = plt.subplots(figsize=(13, 12))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", [V.SERIES[0], "#f0efec", "#e34948"])
    im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1)
    labels = [f"{r['ID']} {r['이름']} ({c.split('|')[1]})" for c, (_, r) in zip(cols, order.iterrows())]
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(labels, fontsize=6.5)
    ax.grid(False)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    bounds = order["동인"].values
    for k in range(1, len(cols)):
        if bounds[k] != bounds[k - 1]:
            ax.axhline(k - 0.5, color=V.INK2, lw=0.8); ax.axvline(k - 0.5, color=V.INK2, lw=0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("Spearman ρ (탐색 창 2017.12 이전, 겹침 24개월 이상)")
    ax.set_title("전국(공통) 블록 입력 간 상관 (동인 ①~⑥ 순; 같은 변수의 두 형태는 빈칸)", loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, s["output_dir"], "fig0_5_상관행렬_공통.png"), dpi=150)
    print("wrote matrices and figure:", rho.shape, rho_r.shape)


if __name__ == "__main__":
    main()
