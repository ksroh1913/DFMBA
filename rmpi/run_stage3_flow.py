# -*- coding: utf-8 -*-
"""
[3단계 흐름 진단 · 분석 1] 단위별 월세 흐름, 매매·전세와의 차이, 공통 변동과 지역 변동의 비중 (연구계획 8차 확정본 3장).

이 장의 수치는 전체 기간 원자료로 계산한 기술 통계이며 예측 성능이 아니다. 6개월 변화는 이웃한 달끼리 겹치므로
상관에 유의성 검정을 붙이지 않는다. 수치는 analysis/plan_v5_numbers.py 와 같은 정의로 다시 계산해 대조한다.

출력 (rmpi/output/)
  fig3_1_단위별흐름.png      전국·수도권·지방권 지수(2015.06=100)와 12개월 변화율, 옛 수도권 대용계열(회색), 접합점, 국면
  fig3_2_17시도흐름.png      시도별 12개월 변화율(파랑)과 전국(회색) 소형 다중 그림
  fig3_3_서울구흐름.png      서울 12개월 변화율과 25개 구의 범위, 최근 상·하위 구
  fig3_4_매매전세비교.png    전국 월세·매매·전세 12개월 변화율, 매매·전세 선행 상관
  fig3_5_분산분해.png        타깃 분산 중 월 공통 비중, 월내 Spearman (17개 시도 vs 서울 25개 구)
  stage3_국면표.csv, stage3_연말변화표.csv, stage3_수치대조.csv (이 스크립트의 값과 plan_v5_numbers.csv 의 값)
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from rmpi import data as D  # noqa: E402
from rmpi import settings as S  # noqa: E402
from rmpi import targets as T  # noqa: E402
from rmpi import viz as V  # noqa: E402

plt = V.plt
REGIMES = [("2012~2014 하락", "2012-01", "2014-12"), ("2016~2019 보합·지방 하락", "2016-01", "2019-12"),
           ("2020~2021 상승", "2020-01", "2021-12"), ("2022 하반기~2023 하락", "2022-07", "2023-12"),
           ("2024~2026 재상승", "2024-01", "2026-08")]
REGIMES = [(n, pd.Period(a, "M"), pd.Period(b, "M")) for n, a, b in REGIMES]
DEV_END = pd.Period("2020-12", "M")


def ts(s_):
    return s_.index.to_timestamp()


def yoy(x):
    return 100 * (x / x.shift(12) - 1)


def month_share(g_wide):
    """월 고정효과의 제곱합 / 총제곱합. g_wide: P x region 의 G"""
    long = g_wide.stack(future_stack=True).dropna()
    mu = long.mean()
    tot = ((long - mu) ** 2).sum()
    mfe = long.groupby(level=0).transform("mean")
    return float(((mfe - mu) ** 2).sum() / tot)


def within_spearman(past_wide, g_wide):
    rs = []
    for p in g_wide.index:
        a, b = past_wide.loc[p], g_wide.loc[p]
        m = a.notna() & b.notna()
        if m.sum() >= 5:
            rs.append(spearmanr(a[m], b[m]).statistic)
    return float(np.nanmean(rs))


def main():
    s = S.load()
    out = os.path.join(BASE, s["output_dir"])
    os.makedirs(out, exist_ok=True)
    fam = V.setup()
    print("font:", fam)
    ref_csv = pd.read_csv(os.path.join(BASE, "analysis", "output", "plan_v5_numbers.csv"))
    cmp_rows = []

    def _nums(v):
        try:
            return [float(x) for x in str(v).split("/")]
        except ValueError:
            return None

    def compare(item_sub, mine):
        hit = ref_csv[ref_csv["항목"].str.contains(item_sub, regex=False)]
        ref = hit["값"].iloc[0] if len(hit) else "(없음)"
        a, b = _nums(mine), _nums(ref)
        same = (a is not None and b is not None and len(a) == len(b) and all(abs(x - y) < 0.0051 for x, y in zip(a, b))) or str(mine) == str(ref)
        cmp_rows.append({"항목": item_sub, "이 스크립트": mine, "plan_v5_numbers": ref, "일치": bool(same) if len(hit) else None})

    # ------------------------------------------------------------ 자료
    raw_rent = S.path(s, "raw_rent")
    nat = {n: T.rone(raw_rent, n) for n in ("전국", "수도권", "지방권")}
    sale, jeon = T.rone(S.path(s, "raw_sale"), "전국"), T.rone(S.path(s, "raw_jeonse"), "전국")
    J = S.per(s["timing"]["splice_month"])
    proxy = T.proxy_level(s, "main")
    old_cap = proxy[proxy.index < J]
    ml, _, _ = D.load_sheets(s, "sido")
    pt = T.regional(s, ml, "sido")
    ml_gu, _, _ = D.load_sheets(s, "gu")
    pg = T.regional(s, ml_gu, "gu")
    R_sido = ml.assign(R=ml[s["inputs"]["target"]["sido"]].where(ml[s["inputs"]["official_flag"]] == 1)).pivot(index="P", columns="region", values="R")
    R_gu = ml_gu.assign(R=ml_gu[s["inputs"]["target"]["gu"]].where(ml_gu[s["inputs"]["official_flag"]] == 1)).pivot(index="P", columns="region", values="R")

    # ------------------------------------------------------------ 그림 1 단위별 흐름
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    ax = axes[0]
    base = {n: 100 * v / v.loc[J] for n, v in nat.items()}
    old_b = 100 * old_cap / nat["전국"].loc[J]        # 2015.06 수준에 맞춘 대용계열 (이미 전국 수준으로 연결됨)
    ax.plot(ts(old_b), old_b.values, color=V.GRAY, lw=1.6)
    ax.annotate("옛 수도권 지수(대용계열, 2015.06 수준에 연결)", (ts(old_b)[0], old_b.iloc[0]), xytext=(0, -14),
                textcoords="offset points", ha="left", va="top", fontsize=8, color=V.INK2)
    for i, n in enumerate(("전국", "수도권", "지방권")):
        ax.plot(ts(base[n]), base[n].values, color=V.SERIES[i], label=n)
        V.end_label(ax, ts(base[n])[-1], base[n].iloc[-1], f"{n} {base[n].iloc[-1]:.1f}")
    ax.axvline(J.to_timestamp(), color=V.BASE, lw=0.8)
    ax.text(J.to_timestamp(), ax.get_ylim()[1], " 2015.06 접합", fontsize=7.5, color=V.INK2, va="top")
    ax.set_title("아파트 월세통합가격지수 (2015.06=100)")
    ax.legend(loc="upper left", ncol=3)
    V.regime_bands(ax, REGIMES, label=False)
    ax = axes[1]
    for i, n in enumerate(("전국", "수도권", "지방권")):
        y = yoy(nat[n]).dropna()
        ax.plot(ts(y), y.values, color=V.SERIES[i], label=n)
        V.end_label(ax, ts(y)[-1], y.iloc[-1], f"{y.iloc[-1]:+.1f}%")
    yo = yoy(old_cap).dropna()
    ax.plot(ts(yo), yo.values, color=V.GRAY, lw=1.6)
    ax.axhline(0, color=V.BASE, lw=0.8)
    ax.axvline(J.to_timestamp(), color=V.BASE, lw=0.8)
    ax.set_title("12개월 변화율 (%)")
    ax.legend(loc="lower left", ncol=3)
    ax.set_ylim(-5, 11)
    V.regime_bands(ax, REGIMES, label=True)
    fig.text(0.01, -0.01, "자료: 한국부동산원 R-ONE 월세통합가격지수(2015.06~), 옛 월세가격지수(수도권 집계, 2010.06~2015.06, 2015.06 수준에 연결). 음영은 국면 구분.",
             fontsize=7.5, color=V.INK2)
    fig.savefig(os.path.join(out, "fig3_1_단위별흐름.png"))
    plt.close(fig)

    # ------------------------------------------------------------ 그림 2 17개 시도
    regs = s["inputs"]["regions"]
    nat_y = yoy(nat["전국"]).dropna()
    fig, axes = plt.subplots(3, 6, figsize=(13, 6.4), sharex=True, sharey=True)
    for k, ax in enumerate(axes.ravel()):
        if k >= 17:
            ax.axis("off")
            continue
        r = regs[k]
        y = yoy(R_sido[r]).dropna()
        ax.plot(ts(nat_y), nat_y.values, color=V.GRAY, lw=1.3)
        ax.plot(ts(y), y.values, color=V.SERIES[0], lw=1.8)
        ax.axhline(0, color=V.BASE, lw=0.8)
        ax.set_title(f"{r}  {y.iloc[-1]:+.1f}%", fontsize=9.5)
        ax.tick_params(labelsize=7.5)
    axes[0, 0].set_ylim(-12, 18)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=V.SERIES[0], lw=2, label="해당 시도"), Line2D([], [], color=V.GRAY, lw=1.3, label="전국")],
               loc="lower right", bbox_to_anchor=(0.98, 0.08), ncol=2)
    yo_all = yoy(R_sido)
    clipped = [f"{r} {yo_all[r].max():+.0f}%" for r in regs if yo_all[r].max() > 18 or yo_all[r].min() < -12]
    fig.suptitle("17개 시도 월세통합가격지수 12개월 변화율 (%)   제목의 수치는 2026.08" + (f"   범위 밖: {', '.join(clipped)}" if clipped else ""),
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_2_17시도흐름.png"))
    plt.close(fig)

    # ------------------------------------------------------------ 그림 3 서울 25개 구
    gu_y = yoy(R_gu).dropna(how="all")
    seoul_y = yoy(R_sido["서울"]).dropna()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.fill_between(ts(gu_y), gu_y.min(axis=1).values, gu_y.max(axis=1).values, color=V.SERIES[0], alpha=0.10, lw=0,
                    label="25개 구의 범위")
    ax.plot(ts(seoul_y), seoul_y.values, color=V.SERIES[0], label="서울")
    last = gu_y.iloc[-1].sort_values()
    for name, val in ((last.index[-1], last.iloc[-1]), (last.index[0], last.iloc[0])):
        V.end_label(ax, ts(gu_y)[-1], val, f"{name} {val:+.1f}%")
    V.end_label(ax, ts(seoul_y)[-1], seoul_y.iloc[-1], f"서울 {seoul_y.iloc[-1]:+.1f}%")
    ax.axhline(0, color=V.BASE, lw=0.8)
    ax.set_title("서울 25개 구 월세통합가격지수 12개월 변화율 (%)  공식 지수 구간 2016.06~")
    ax.legend(loc="upper left")
    ax.set_xlim(pd.Timestamp("2016-01-01"), pd.Timestamp("2027-03-01"))
    V.regime_bands(ax, [r for r in REGIMES if r[1] >= pd.Period("2016-01", "M")], label=True)
    fig.savefig(os.path.join(out, "fig3_3_서울구흐름.png"))
    plt.close(fig)

    # ------------------------------------------------------------ 그림 4 매매·전세 비교
    series = {"월세": nat["전국"], "매매": sale, "전세": jeon}
    natdf = pd.DataFrame(series).loc[J:]                              # 공식 구간으로 맞춘 뒤 변화율 계산 (plan_v5 와 같은 표본)
    g6 = 100 * (natdf.shift(-5) / natdf.shift(1) - 1)
    c = g6.dropna().corr()
    for a_, b_ in (("월세", "매매"), ("월세", "전세"), ("매매", "전세")):
        compare(f"6개월 변화 상관 {a_}-{b_}", f"{c.loc[a_, b_]:.2f}")
    fut = 100 * (natdf["월세"].shift(-6) / natdf["월세"] - 1)          # 향후 6개월 월세 변화
    past = {k: 100 * (natdf[k] / natdf[k].shift(6) - 1) for k in ("매매", "전세")}
    leads = {}
    for k, pv in past.items():
        leads[k] = [fut.corr(pv.shift(L)) for L in (0, 6, 12)]
    for k in ("매매", "전세"):
        compare(f"{k} 최근6개월→월세 향후6개월", "/".join(f"{x:.2f}" for x in leads[k]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.4, 1]})
    ax = axes[0]
    ends = []
    for i, (k, v) in enumerate(series.items()):
        y = yoy(v).loc[pd.Period("2016-01", "M"):].dropna()
        ax.plot(ts(y), y.values, color=V.SERIES[i], label=k)
        ends.append(f"{k} {y.iloc[-1]:+.1f}%")
    ax.axhline(0, color=V.BASE, lw=0.8)
    ax.set_title("전국 월세·매매·전세 가격지수 12개월 변화율 (%)   2026.08: " + ", ".join(ends))
    ax.legend(loc="upper left", ncol=3)
    ax.set_ylim(-14, 18)
    V.regime_bands(ax, [r for r in REGIMES if r[1] >= pd.Period("2016-01", "M")], label=True, pos="bottom")
    ax = axes[1]
    xs = np.arange(3)
    w = 0.3
    b1 = ax.bar(xs - w / 2 - 0.01, leads["매매"], width=w, color=V.SERIES[1], label="매매")
    b2 = ax.bar(xs + w / 2 + 0.01, leads["전세"], width=w, color=V.SERIES[2], label="전세")
    V.bar_cap_labels(ax, list(b1) + list(b2), "{:.2f}")
    ax.set_xticks(xs, ["동시", "6개월 앞", "12개월 앞"])
    ax.axhline(0, color=V.BASE, lw=0.8)
    ax.set_ylim(-0.4, 0.8)
    ax.set_title("매매·전세 6개월 변화와\n향후 6개월 월세 변화의 상관")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_4_매매전세비교.png"))
    plt.close(fig)

    # 연말 전년 대비 표
    rows = []
    for y_ in range(2016, 2026):
        p = pd.Period(f"{y_}-12", "M")
        rows.append({"연도": y_, **{k: round(float(yoy(v).loc[p]), 1) for k, v in series.items()}})
    ye = pd.DataFrame(rows)
    ye.to_csv(os.path.join(out, "stage3_연말변화표.csv"), index=False, encoding="utf-8-sig")
    for y_ in (2018, 2022, 2025):
        r_ = ye.set_index("연도").loc[y_]
        compare(f"{y_} 연말 전년비", f"{r_['월세']:+.1f}/{r_['매매']:+.1f}/{r_['전세']:+.1f}")

    # ------------------------------------------------------------ 그림 5 분산 분해
    lo = S.per(s["timing"]["official_first_decision"])
    stats = {}
    for name, p_ in (("17개 시도", pt), ("서울 25개 구", pg)):
        for h in s["timing"]["horizons"]:
            full = p_[(p_.P >= lo) & p_[f"G{h}"].notna()]
            dev = full[full.P + (h - 1) <= DEV_END]
            gw_f = full.pivot(index="P", columns="region", values=f"G{h}")
            gw_d = dev.pivot(index="P", columns="region", values=f"G{h}")
            pw_d = dev.pivot(index="P", columns="region", values=f"past{h}")
            stats[(name, h, "share_full")] = month_share(gw_f)
            stats[(name, h, "share_dev")] = month_share(gw_d)
            stats[(name, h, "spearman_dev")] = within_spearman(pw_d, gw_d)
            if h == 6:
                stats[(name, h, "sd_within_dev")] = float(gw_d.std(axis=1).mean())    # 월별 지역 간 표준편차의 평균 (plan_v5 정의)
    for name, pn in (("17개 시도", "17시도"), ("서울 25개 구", "서울25구")):
        for h in (1, 3, 6):
            compare(f"{pn} h={h} 월 공통 비중(전체)", f"{stats[(name, h, 'share_full')]:.3f}")
            compare(f"{pn} h={h} 월 공통 비중(개발)", f"{stats[(name, h, 'share_dev')]:.3f}")
            compare(f"{pn} h={h} 과거 상대변화→다음 상대순위", f"{stats[(name, h, 'spearman_dev')]:.2f}")
        compare(f"{pn} h=6 월내 지역 간 표준편차", f"{stats[(name, 6, 'sd_within_dev')]:.2f}")
    pd.DataFrame([{"패널": k[0], "h": k[1], "항목": k[2], "값": round(v, 4)} for k, v in stats.items()]).to_csv(
        os.path.join(out, "stage3_분산분해.csv"), index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    hs = s["timing"]["horizons"]
    xs = np.arange(len(hs))
    for ax, key, title, ylim in ((axes[0], "share_full", "타깃 분산 중 월 공통 비중 (%), 전체 기간", (0, 100)),
                                 (axes[1], "share_dev", "같은 비중 (%), 개발 성격 구간(정답 2020.12까지)", (0, 100)),
                                 (axes[2], "spearman_dev", "과거 상대 변화 → 다음 상대 순위, 월내 Spearman (개발)", (0, 1))):
        a = [stats[("17개 시도", h, key)] * (100 if "share" in key else 1) for h in hs]
        b = [stats[("서울 25개 구", h, key)] * (100 if "share" in key else 1) for h in hs]
        b1 = ax.bar(xs - w / 2 - 0.01, a, width=w, color=V.SERIES[0], label="17개 시도")
        b2 = ax.bar(xs + w / 2 + 0.01, b, width=w, color=V.SERIES[1], label="서울 25개 구")
        V.bar_cap_labels(ax, list(b1) + list(b2), "{:.1f}" if "share" in key else "{:.2f}")
        ax.set_xticks(xs, [f"h={h}" for h in hs])
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=9.5)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_5_분산분해.png"))
    plt.close(fig)

    # ------------------------------------------------------------ 국면 표
    rows = []
    for name, a, b in REGIMES:
        rec = {"국면": name, "시작": str(a), "끝": str(b)}
        for n in ("전국", "수도권", "지방권"):
            ser = nat[n] if b > J else None
            if ser is not None and a >= J:
                rec[f"{n} 누적변화(%)"] = round(100 * (ser.loc[b] / ser.loc[a - 1] - 1), 1)
        if b < J:
            rec["옛 수도권 누적변화(%)"] = round(100 * (old_cap.loc[b] / old_cap.loc[a - 1] - 1), 1)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(os.path.join(out, "stage3_국면표.csv"), index=False, encoding="utf-8-sig")

    # 권역 상관(plan_v5 와 대조)
    agg = pd.DataFrame({k: nat[k] for k in ("전국", "수도권", "지방권")})
    g6a = 100 * (agg.shift(-5) / agg.shift(1) - 1)
    for lo_, hi_ in ((J, pd.Period("2026-08", "M")), (J, pd.Period("2020-12", "M")), (pd.Period("2021-01", "M"), pd.Period("2026-08", "M"))):
        x = g6a.loc[lo_:hi_].dropna()
        compare(f"{lo_.year}{lo_.month:02d}~{hi_.year}{hi_.month:02d}",
                f"{x['전국'].corr(x['수도권']):.2f}/{x['전국'].corr(x['지방권']):.2f}/{x['수도권'].corr(x['지방권']):.2f}")
    pd.DataFrame(cmp_rows).to_csv(os.path.join(out, "stage3_수치대조.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print(pd.DataFrame(cmp_rows).to_string(index=False))
    print(pd.DataFrame([{"패널": k[0], "h": k[1], "항목": k[2], "값": round(v, 3)} for k, v in stats.items()]).pivot(index=["패널", "항목"], columns="h", values="값"))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
