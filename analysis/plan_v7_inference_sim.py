# -*- coding: utf-8 -*-
"""
※ 분석 자료를 쓰지 않는 모의실험. 연구계획 8차 확정본 1장의 판정 규칙(두 t구간 결합)을 고른 근거를 계산한다.

원래 이 이름의 스크립트는 저장소에 없어서, 계획서 1·12장과 부록 B의 설정 설명(손실 차이의 1개월 자기상관 0.6~0.9, 96개월,
8개 연평균 t구간·4개 2년 평균 t구간, 블록 길이 6·12·18개월의 순환 블록 부트스트랩, 고정 난수)으로 2026-10-05 에 새로 만들었다.
원문 수치(연평균 구간만 3.0~9.5%, 두 구간 결합 1.4~4.0%, 부트스트랩 양측 기각률 12.6~36.5% 대 t구간 6.6~18.2%)와
완전히 같을 수는 없다. 계획서 1장 수치를 쓸 때는 이 스크립트의 결과로 바꿔 적어야 한다.

설정
  n = 96 (결정연도 8개 x 12개월; 2년 묶음 4개). 손실 차이 d_t 는 평균 0(귀무) 또는 음수(대립)인 정상 과정.
  과정:  AR(1) φ ∈ {0.6, 0.7, 0.8, 0.9}, t(5) 혁신
         겹침 구조: MA(h−1) 이동합 + AR(0.5)  (h=3, 6; 목표 기간이 겹치는 손실 차이를 흉내)
  규칙(모두 95% 구간):
    R1  연평균 8개의 t구간(자유도 7) 상한 < 0                       ← 한쪽만 보므로 명목 2.5%
    R2  R1 이고 2년 묶음 4개의 t구간(자유도 3) 상한 < 0             ← 계획의 ‘개선’
    R3  R1 이고 8개 연도 중 5개 이상 연평균 < 0                      ← 6차안에서 검토했던 조건
    T2  연평균 t구간이 0 을 포함하지 않음(양측)                       ← 부트스트랩 양측과 비교용
    B6·B12·B18  순환 블록 부트스트랩(블록 6·12·18개월, 999회) 95% 백분위 구간: 양측(0 미포함)과 한쪽(상한 < 0)
  대립: 평균을 −0.25σ_d, −0.5σ_d 로 옮긴다(σ_d 는 d 의 무조건 표준편차). 검정력으로 보고.
  반복: t구간 규칙 10,000회, 부트스트랩 2,000회(모의실험 반복마다 재표집 999회. 실제 평가는 2,000회를 쓰지만 2.5·97.5 백분위에는 차이가 없다).
        난수 고정(20261004). 몬테카를로 표준오차: t구간 규칙 0.2~0.4%p, 부트스트랩 0.8~1.1%p.
  참고: R3(연평균 상한<0 이고 5/8 연도 우세)는 모든 칸에서 R1 과 같다. 연평균 구간 상한이 0 아래이면 8개 연도 중 5개 이상이 음수인 경우뿐이어서,
        6차안의 ‘연도 과반’ 조건은 R1 에 아무것도 더하지 않는다(판정 규칙에서 뺀 이유).
  판정 비율을 좌우하는 것은 1개월 자기상관이 아니라 이웃한 연평균 사이의 상관이다(AR(1) φ 0.6~0.9 에서 0.09~0.47). 겹침 구조 과정은
        1개월 자기상관이 더 높아도(0.84·0.93) 연평균 간 상관이 낮아(0.08·0.13) 거짓 개선 비율이 AR(1) 범위의 아래쪽에 온다.
        독립 재계산(2026-10-05, 다른 난수·4만 회)에서 12개 대표 칸이 모두 몬테카를로 오차 안에서 일치했다.

출력: analysis/output/plan_v7_inference_sim.csv  (과정, φ, 실측 lag1 자기상관, 효과, 규칙, 비율, 반복수)
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import lfilter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "analysis", "output")
SEED = 20261004
N, YEARS = 96, 8
R_T, R_B, B_BOOT = 10_000, 2_000, 999
BLOCKS = (6, 12, 18)
Q7, Q3 = stats.t.ppf(0.975, YEARS - 1), stats.t.ppf(0.975, YEARS // 2 - 1)


def simulate(kind, phi, R, rng, burn=200):
    """R x N 의 d_t (평균 0, 분산 1 로 정규화). kind: 'ar' 또는 'ma3','ma6'(겹침 구조)."""
    e = rng.standard_t(5, size=(R, N + burn))
    if kind.startswith("ma"):
        h = int(kind[2:])
        e = np.cumsum(e, axis=1)
        e = np.concatenate([e[:, :h], e[:, h:] - e[:, :-h]], axis=1)   # h개 혁신의 이동합
    x = lfilter([1.0], [1.0, -phi], e, axis=1)[:, burn:]
    return x / x.std()        # 무조건 표준편차(모든 경로 합산)로만 나눈다. 경로별로 평균을 빼면 d̄=0 이 강제되어 검정이 무의미해진다


def lag1(x):
    a, b = x[:, 1:] - x.mean(axis=1, keepdims=True), x[:, :-1] - x.mean(axis=1, keepdims=True)
    return float(np.mean((a * b).sum(axis=1) / (b ** 2).sum(axis=1)))


def t_rules(d):
    """R x N -> dict of bool arrays"""
    R = d.shape[0]
    ann = d.reshape(R, YEARS, 12).mean(axis=2)
    bi = d.reshape(R, YEARS // 2, 24).mean(axis=2)
    m = ann.mean(axis=1)
    se_a = ann.std(axis=1, ddof=1) / np.sqrt(YEARS)
    se_b = bi.std(axis=1, ddof=1) / np.sqrt(YEARS // 2)
    up_a, lo_a = m + Q7 * se_a, m - Q7 * se_a
    up_b = m + Q3 * se_b
    r1 = up_a < 0
    return {"R1 연평균 상한<0": r1, "R2 두 구간 상한<0 (계획 ‘개선’)": r1 & (up_b < 0),
            "R3 R1 & 5/8 연도 우세": r1 & ((ann < 0).sum(axis=1) >= 5),
            "T2 연평균 구간 0 미포함(양측)": (up_a < 0) | (lo_a > 0), "5/8 연도 우세(단독)": (ann < 0).sum(axis=1) >= 5}


def boot_rules(d, rng):
    """R x N -> dict of bool arrays (순환 블록 부트스트랩 백분위 구간)"""
    R = d.shape[0]
    out = {}
    for L in BLOCKS:
        nb = int(np.ceil(N / L))
        two, one = np.zeros(R, bool), np.zeros(R, bool)
        for r in range(R):
            starts = rng.integers(0, N, size=(B_BOOT, nb))
            idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B_BOOT, -1)[:, :N] % N
            means = d[r][idx].mean(axis=1)
            lo, hi = np.quantile(means, [0.025, 0.975])
            two[r], one[r] = (hi < 0) | (lo > 0), hi < 0
        out[f"B{L} 블록 {L} 양측 0 미포함"] = two
        out[f"B{L} 블록 {L} 상한<0"] = one
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rows = []
    procs = [("AR(1)", "ar", p) for p in (0.6, 0.7, 0.8, 0.9)] + [("MA(2)+AR(0.5)", "ma3", 0.5), ("MA(5)+AR(0.5)", "ma6", 0.5)]
    for name, kind, phi in procs:
        for eff in (0.0, -0.25, -0.5):
            d = simulate(kind, phi, R_T, rng) + eff
            ac = lag1(d)
            for rule, hit in t_rules(d).items():
                rows.append({"과정": name, "φ": phi, "lag1 자기상관(실측)": round(ac, 2), "효과(σ)": eff, "규칙": rule,
                             "비율": float(hit.mean()), "반복수": R_T})
            if eff in (0.0, -0.5):
                db = d[:R_B]
                for rule, hit in boot_rules(db, rng).items():
                    rows.append({"과정": name, "φ": phi, "lag1 자기상관(실측)": round(ac, 2), "효과(σ)": eff, "규칙": rule,
                                 "비율": float(hit.mean()), "반복수": R_B})
            print(f"{name} φ={phi} 효과={eff}: {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, "plan_v7_inference_sim.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    null = res[res["효과(σ)"] == 0].pivot_table(index="규칙", columns=["과정", "φ"], values="비율")
    print("\n귀무(차이 없음)에서 각 규칙이 ‘개선’ 또는 기각으로 판정하는 비율\n", (100 * null).round(1).to_string())
    alt = res[res["효과(σ)"] < 0].pivot_table(index=["효과(σ)", "규칙"], columns=["과정", "φ"], values="비율")
    print("\n대립(평균 −0.25σ, −0.5σ)에서의 판정 비율(검정력)\n", (100 * alt).round(1).to_string())
    # 계획서 1장 문장에 대응하는 범위
    ar_null = res[(res["효과(σ)"] == 0) & (res["과정"] == "AR(1)")]
    r1 = ar_null[ar_null["규칙"].str.startswith("R1")]["비율"]
    r2 = ar_null[ar_null["규칙"].str.startswith("R2")]["비율"]
    t2 = ar_null[ar_null["규칙"].str.startswith("T2")]["비율"]
    bt = ar_null[ar_null["규칙"].str.startswith("B") & ar_null["규칙"].str.contains("양측")]["비율"]
    print(f"\nAR(1) φ 0.6~0.9, 차이 없음: 연평균 구간만 {100 * r1.min():.1f}~{100 * r1.max():.1f}%, 두 구간 결합 {100 * r2.min():.1f}~{100 * r2.max():.1f}%, "
          f"연평균 t구간 양측 {100 * t2.min():.1f}~{100 * t2.max():.1f}%, 블록 부트스트랩 양측 {100 * bt.min():.1f}~{100 * bt.max():.1f}%")
    print(f"총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
