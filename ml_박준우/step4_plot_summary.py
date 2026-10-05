# -*- coding: utf-8 -*-
"""
[4단계] 성과 요약 도표

입력: output/03_성과_요약.csv, output/03_성과_연도별RMSE.csv
출력: output/04_성과요약_도표.png
  왼쪽  모형별 기준모형 대비 RMSE 비율 (17개 시도 전체 채점 / 세종 제외 채점)
  오른쪽 시험연도별 기준모형 대비 RMSE 비율
  두 그림 모두 1보다 작으면 기준모형보다 정확하다는 뜻이다
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from spec import OUT

BASELINE = "기준: 모멘텀+지역FE"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a3a2", "#c23b3a"])


def main():
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK})
    S = pd.read_csv(os.path.join(OUT, "03_성과_요약.csv")).set_index("모형")
    Y = pd.read_csv(os.path.join(OUT, "03_성과_연도별RMSE.csv"), index_col=0)
    base_rmse = S.loc[BASELINE, "RMSE"]
    order = [m for m in S.index if m != BASELINE]            # 요약표가 이미 RMSE 순
    full = S.loc[order, "기준 대비 RMSE 비율"]
    ex = S.loc[order, "기준 대비 RMSE 비율(세종 제외 채점)"]
    year = Y[order].div(Y[BASELINE], axis=0).T               # 모형 x 연도
    pos = range(len(order))

    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6.6), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.42},
                               facecolor=SURFACE)
    fig.suptitle("17개 시도 전체로는 어떤 확장 모형도 기준모형을 넘지 못했다", x=0.012, ha="left", fontsize=15,
                 fontweight="bold")
    fig.text(0.012, 0.915, f"표본외 2021~2025년, 6개월 누적 월세 변화율. 기준모형(가격 모멘텀 + 지역 고정효과) RMSE = "
             f"{base_rmse:.3f}%p. 비율이 1보다 작으면 기준모형보다 정확", fontsize=10, color=MUTED)

    # 왼쪽: 전체 채점과 세종 제외 채점을 잇는 점 도표
    a.set_facecolor(SURFACE)
    a.hlines(pos, full, ex, color=GRID, lw=2, zorder=1)
    a.scatter(full, pos, s=70, color=BLUE, zorder=3, label="17개 시도 전체 채점", edgecolor=SURFACE, lw=1.5)
    a.scatter(ex, pos, s=70, color=ORANGE, marker="D", zorder=3, label="세종 제외 채점", edgecolor=SURFACE, lw=1.5)
    a.axvline(1, color=INK, lw=1, zorder=2)
    a.text(1, -0.85, "기준모형 = 1", ha="center", va="bottom", fontsize=9, color=MUTED)
    for i, m in enumerate(order):
        a.text(max(full[m], ex[m]) + 0.012, i, f"{full[m]:.2f} / {ex[m]:.2f}", va="center", fontsize=9, color=MUTED)
    a.set_yticks(list(pos), order)
    a.set_ylim(len(order) - 0.5, -0.9)
    a.set_xlim(0.78, 1.36)
    a.set_xlabel("기준모형 대비 RMSE 비율 (오른쪽 숫자: 전체 / 세종 제외)")
    a.set_title("모형별 (5개 연도 합산)", loc="left", fontsize=11, color=INK, pad=24)
    a.grid(axis="x", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    a.tick_params(length=0)
    for s in a.spines.values():
        s.set_visible(False)
    a.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=9, borderaxespad=0.2,
             handletextpad=0.2, columnspacing=1.2)

    # 오른쪽: 연도별 비율
    b.set_facecolor(SURFACE)
    b.imshow(year.values, cmap=DIVERGING, norm=TwoSlopeNorm(vmin=0.6, vcenter=1, vmax=1.6), aspect="auto")
    for i in range(year.shape[0]):
        for j in range(year.shape[1]):
            v = year.values[i, j]
            b.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                   color="white" if (v < 0.75 or v > 1.45) else INK)
    b.set_xticks(range(year.shape[1]), [f"{y}\n(기준 {Y.loc[y, BASELINE]:.2f})" for y in year.columns], fontsize=9)
    b.set_yticks(list(pos), order)
    b.set_xticks([x - 0.5 for x in range(1, year.shape[1])], minor=True)
    b.set_yticks([y - 0.5 for y in range(1, year.shape[0])], minor=True)
    b.grid(which="minor", color=SURFACE, lw=2)
    b.tick_params(which="both", length=0)
    for s in b.spines.values():
        s.set_visible(False)
    b.set_title("시험연도별 (파랑 = 기준보다 정확, 빨강 = 부정확)", loc="left", fontsize=11, color=INK, pad=24)
    b.set_xlabel("시험연도 (괄호: 그 해 기준모형 RMSE, %p)")

    fig.subplots_adjust(left=0.165, right=0.985, top=0.80, bottom=0.11)
    fig.savefig(os.path.join(OUT, "04_성과요약_도표.png"), dpi=140, facecolor=SURFACE)
    print("저장: output/04_성과요약_도표.png")


if __name__ == "__main__":
    main()
