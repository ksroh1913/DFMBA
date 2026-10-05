# -*- coding: utf-8 -*-
"""[9차 그림] 급락 경보 사례 그림 (h=6, 훈련 사건비율 1배 컷오프): 지역 × 결정월 격자에 적중·오경보·놓침·기존 급락 상태를 표시한다.
입력: rmpi/output_v9/v9_s1_예측값_패널.csv, v9_s3_예측값_패널.csv. 출력: fig6_4_경보사례_h6.png.
RMPI_SETTINGS=config/plan_v9_settings.yaml 로 실행."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from rmpi import settings as S  # noqa: E402
from rmpi import viz as V  # noqa: E402

TARGETS = {
    "down": dict(models=[("panel_down_A", "A 가격 추세"), ("panel_down_B", "B 추세 + RMPI"), ("panel_down_B+RT", "B + 실거래")], state="기존급락상태",
                 files=("v9_s1_예측값_패널.csv", "v9_s3_예측값_패널.csv"), out="fig6_4_경보사례_h6.png", word="급락", state_word="기존 급락 상태",
                 rule="급락 = 향후 6개월 변화율 < -1.005% (연율 -2%). 기존 급락 상태 = 결정월 직전 6개월 변화율이 이미 임계 아래.", src="v9_s1_예측값_패널.csv, v9_s3_예측값_패널.csv"),
    "up": dict(models=[("panel_up_A", "A 가격 추세"), ("panel_up_B", "B 추세 + RMPI"), ("panel_up_B+RT", "B + 실거래")], state="기존급등상태",
               files=("v9_s5_급등_예측값_패널.csv",), out="fig6_5_급등경보사례_h6.png", word="급등", state_word="기존 급등 상태",
               rule="급등 = 향후 6개월 변화율이 +1.489% 이상 (연율 +3%). 기존 급등 상태 = 결정월 직전 6개월 변화율이 이미 경계 이상. 학습 양성 30행 이상인 결정월부터 평가.", src="v9_s5_급등_예측값_패널.csv"),
}
CODES = {"없음": 0, "기존 상태": 1, "놓침": 2, "오경보": 3, "적중": 4}
COLORS = [V.SURFACE, V.BASE, V.SERIES[0], V.SERIES[1], V.SERIES[2]]


def grid(df, h=6, k=1, state="기존급락상태"):
    d = df[df["h"] == h].copy()
    d["P"] = pd.PeriodIndex(d["P"], freq="M")
    alert = d["yhat"] >= d[f"컷오프_{k}x"]
    code = np.select([d[state] == 1, alert & (d["y"] == 1), alert, d["y"] == 1],
                     [CODES["기존 상태"], CODES["적중"], CODES["오경보"], CODES["놓침"]], CODES["없음"])
    d["code"] = code
    g = d.pivot_table(index="region", columns="P", values="code", aggfunc="first")
    return g, d


def main(target="down"):
    s = S.load()
    out = os.path.join(BASE, s["output_dir"])
    V.setup()
    cfg = TARGETS[target]
    parts = [pd.read_csv(os.path.join(out, f)) for f in cfg["files"] if os.path.exists(os.path.join(out, f))]
    allp = pd.concat(parts, ignore_index=True)
    allp = allp[allp["타깃"] == target]
    if "평가대상" in allp.columns:
        allp = allp[allp["평가대상"]]
    h, k = 6, 1
    fig, axes = plt.subplots(len(cfg["models"]), 1, figsize=(12.5, 9.6), sharex=True)
    cmap = ListedColormap(COLORS)
    for ax, (m, title) in zip(axes, cfg["models"]):
        g, d = grid(allp[allp["모형"] == m], h, k, cfg["state"])
        g = g.reindex(sorted(g.index))
        ax.pcolormesh(np.arange(g.shape[1] + 1), np.arange(g.shape[0] + 1), g.values, cmap=cmap, vmin=-0.5, vmax=4.5, edgecolors=V.GRID, linewidth=0.3)
        ax.set_yticks(np.arange(g.shape[0]) + 0.5)
        ax.set_yticklabels(g.index, fontsize=8)
        ax.invert_yaxis()
        nz = d[d[cfg["state"]] == 0]
        hit = int(((nz["code"] == CODES["적중"])).sum()); fa = int((nz["code"] == CODES["오경보"]).sum()); miss = int((nz["code"] == CODES["놓침"]).sum())
        prec = hit / max(hit + fa, 1); rec = hit / max(hit + miss, 1)
        ax.set_title(f"{title}   적중 {hit} · 오경보 {fa} · 놓침 {miss}   (적중률 {prec:.2f}, 포착률 {rec:.2f})", fontsize=10, loc="left", color=V.INK)
        for sp in ax.spines.values():
            sp.set_color(V.GRID)
        cols = list(g.columns)
        ticks = [i for i, p in enumerate(cols) if p.month == 1]
        ax.set_xticks(np.array(ticks) + 0.5)
        ax.set_xticklabels([str(cols[i].year) for i in ticks], fontsize=8.5)
        ax.tick_params(length=0)
    handles = [Patch(facecolor=COLORS[CODES[n if n != cfg["state_word"] else "기존 상태"]], edgecolor=V.GRID, label=n) for n in ("적중", "오경보", "놓침", cfg["state_word"])]
    axes[0].legend(handles=handles, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.08), frameon=False, fontsize=8.5)
    p0 = str(pd.PeriodIndex(allp.loc[allp["h"] == h, "P"].astype(str), freq="M").min()).replace("-", ".")
    fig.suptitle(f"{cfg['word']} 경보 사례 (h={h}, 결정월 {p0}~2025.12, 컷오프 = 훈련 사건비율 {k}배 분위, 결정월에 {cfg['word']} 상태가 아닌 지역)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=0.995)
    fig.text(0.01, 0.005, cfg["rule"] + " 출처: " + cfg["src"], fontsize=7.5, color=V.INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    path = os.path.join(out, cfg["out"])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="down", choices=list(TARGETS))
    main(ap.parse_args().target)
