# -*- coding: utf-8 -*-
"""그림 공통 스타일. 범주 색은 고정 순서(1 파랑, 2 주황, 3 청록)로만 쓰고 순환하지 않는다.
마크는 가늘게(선 2px), 격자는 실선 헤어라인, 글자는 글자색(계열 색을 글자에 쓰지 않음)."""

import glob

import matplotlib
import matplotlib.font_manager as fm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]      # 범주 1·2·3 (validate_palette.js 통과: 인접·전체 쌍)
GRAY = "#898781"                                # 맥락 계열·축 글자
WASH = "#f0efec"                                # 국면 음영
GRID = "#e1e0d9"
BASE = "#c3c2b7"
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"


def setup():
    for p in glob.glob("/usr/share/fonts/truetype/nanum/NanumGothic*.ttf"):
        try:
            fm.fontManager.addfont(p)
        except Exception:
            pass
    names = {f.name for f in fm.fontManager.ttflist}
    fam = "NanumGothic" if "NanumGothic" in names else ("Unifont" if "Unifont" in names else "DejaVu Sans")
    plt.rcParams.update({
        "font.family": fam, "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": GRAY, "ytick.color": GRAY, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "text.color": INK, "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 2, "lines.solid_joinstyle": "round", "lines.solid_capstyle": "round",
        "figure.dpi": 110, "savefig.dpi": 170, "savefig.bbox": "tight",
    })
    return fam


SHORT = {"2012~2014 하락": "12~14 하락", "2016~2019 보합·지방 하락": "16~19 보합·지방 하락", "2020~2021 상승": "20~21 상승",
         "2022 하반기~2023 하락": "22下~23 하락", "2024~2026 재상승": "24~26 재상승"}


def regime_bands(ax, regimes, label=True, pos="top"):
    """국면 음영과 짧은 라벨(위 또는 아래, 이웃끼리 높이를 엇갈리게). regimes: [(이름, 시작 Period, 끝 Period)]"""
    ymin, ymax = ax.get_ylim()
    for i, (name, a, b) in enumerate(regimes):
        x0, x1 = a.to_timestamp(), (b + 1).to_timestamp()
        ax.axvspan(x0, x1, color=WASH, zorder=0, lw=0)
        if label:
            y = ymax if pos == "top" else ymin
            dy = (-3 if i % 2 == 0 else -13) if pos == "top" else (3 if i % 2 == 0 else 13)
            ax.annotate(SHORT.get(name, name), (x0 + (x1 - x0) / 2, y), xytext=(0, dy), textcoords="offset points",
                        ha="center", va="top" if pos == "top" else "bottom", fontsize=7.2, color=INK2, zorder=5)
    ax.set_ylim(ymin, ymax)


def end_label(ax, x, y, text, color=INK2, dx=4):
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", va="center", ha="left",
                fontsize=8.5, color=color)


def bar_cap_labels(ax, bars, fmt="{:.1f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INK2)
