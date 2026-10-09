# -*- coding: utf-8 -*-
"""
변수 통합 소개 카드 — 최종 학습 데이터셋(analysis/output/모형입력표_10차_values.csv)의 타깃(G3·G6)과 설명변수 시계열, 변수 간 상관을 한 html 로.

출력: analysis/output/변수카드/00_변수통합소개_10차.html
구성: ① 데이터셋 개요·변수 목록(동인별)  ② 타깃 G3·G6 시계열  ③ 설명변수 시계열(동인별, 변수당 1장)
      ④ 타깃과의 상관(풀링·시도 평균)  ⑤ 설명변수 간 상관 히트맵 + 강한 상관 쌍  ⑥ 읽는 법·주의
시계열 색: 서울 = 파랑(강조), 시도 평균 = 검정, 나머지 시도 = 회색. 상관 = 파랑(+)·회색(0)·빨강(−).
사용: PYTHONUTF8=1 python analysis/dataset_card.py   (먼저 analysis/model_values.py 로 값 CSV 를 만들어 둘 것)
"""
import base64
import datetime as dt
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import variable_card as vc  # noqa: E402  (글꼴·결정 로그 로더 재사용)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALUES = os.path.join(BASE, "analysis", "output", "모형입력표_10차_values.csv")
OUT = os.path.join(BASE, "analysis", "output", "변수카드", "00_변수통합소개_10차.html")

# 팔레트 (dataviz 기준 팔레트: 표면 #fcfcfb, 잉크 #0b0b0b/#52514e/#898781, 격자 #e1e0d9, 강조 파랑 #2a78d6, 발산 파랑–회색–빨강)
C = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
         blue="#2a78d6", red="#e34948", mid="#f0efec", buffer="#f0efec")
DIVERGING = LinearSegmentedColormap.from_list("bgr", [C["blue"], C["mid"], C["red"]])
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
                     "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"], "xtick.color": C["muted"], "ytick.color": C["muted"],
                     "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": C["ink"], "legend.fontsize": 8, "legend.frameon": False})

ANALYSIS_START = pd.Period("2016-01", "M")
DRIVER_ORDER = "①②③④⑤⑥"


def img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=C["surface"])
    plt.close(fig)
    return "<img src='data:image/png;base64," + base64.b64encode(buf.getvalue()).decode() + "'/>"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ------------------------------------------------------------ 데이터
def load():
    df = pd.read_csv(VALUES, encoding="utf-8-sig")
    df["결정월"] = pd.PeriodIndex(df["결정월"], freq="M")
    log = vc.log_load()
    dec = log[(log["상태"] == "결정") & (log["ID"] != "V001")].copy()
    meta = {}
    for col in df.columns[11:]:
        vid = col.split("_")[0]
        r = dec[dec["ID"] == vid]
        if len(r):
            r = r.iloc[0]
            meta[col] = dict(ID=vid, 변수명=r["변수명"], 동인=r["동인"], 역할=r["역할"], 변환=r["변환"], 공표시차=r["공표시차_개월"], 지역단위=r["지역단위"],
                             드라이버=next((i for i, ch in enumerate(DRIVER_ORDER) if str(r["동인"]).startswith(ch)), 9))
    return df, meta


def wide_of(df, col):
    return df.pivot(index="결정월", columns="region", values=col)


def is_national(w):
    sub = w.dropna(how="all")
    return bool(len(sub)) and bool((sub.nunique(axis=1) <= 1).all())


# ------------------------------------------------------------ 시계열 차트
def plot_series(w, title, subtitle="", unit=""):
    fig, ax = plt.subplots(figsize=(10.5, 2.7))
    xs = w.index.to_timestamp()
    nat = is_national(w)
    ax.axvspan(pd.Timestamp("2014-01-01"), pd.Timestamp("2015-12-31"), color=C["buffer"], zorder=0)
    ax.text(pd.Timestamp("2014-02-01"), 0.97, "버퍼(12개월 변화 계산용)", transform=ax.get_xaxis_transform(), va="top", fontsize=7.5, color=C["muted"])
    if nat:
        s = w.iloc[:, 0] if w.shape[1] == 1 else w.bfill(axis=1).iloc[:, 0]
        ax.plot(xs, s.values, lw=2, color=C["ink"], label="전국 공통(모든 시도 같은 값)")
        vals = s.dropna().values
    else:
        regs = [c for c in w.columns if c != "서울"]
        for c in regs:
            ax.plot(xs, w[c].values, lw=0.7, color=C["muted"], alpha=0.55, label="기타 시도" if c == regs[0] else None)
        mean = w.mean(axis=1)
        ax.plot(xs, mean.values, lw=2, color=C["ink"], label="시도 평균")
        if "서울" in w.columns:
            ax.plot(xs, w["서울"].values, lw=2, color=C["blue"], label="서울")
            last = w["서울"].last_valid_index()
            if last is not None:
                ax.annotate("서울", (last.to_timestamp(), w["서울"][last]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=C["blue"], va="center")
        vals = w.values[np.isfinite(w.values)]
    if len(vals) > 20:                                           # 극단값이 축을 눌러 버리면 0.5~99.5 백분위로 자른다
        lo, hi = np.percentile(vals, [0.5, 99.5])
        full_lo, full_hi = vals.min(), vals.max()
        if (full_hi - full_lo) > 2.5 * (hi - lo) and hi > lo:
            pad = 0.06 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
            subtitle = (subtitle + " · " if subtitle else "") + "세로축은 0.5~99.5 백분위로 자름(극단값 생략)"
    ax.axhline(0, color=C["axis"], lw=0.8, zorder=1)
    ax.set_title(title, loc="left", pad=20 if subtitle else 8)                 # 부제가 있으면 제목을 위로 올려 겹치지 않게
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=8, color=C["ink2"], va="bottom")
    if unit:
        ax.set_ylabel(unit, fontsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)        # 범례는 그림 아래 (데이터를 가리지 않게)
    ax.set_xlim(pd.Timestamp("2014-01-01"), xs.max() + pd.Timedelta(days=60))
    return img(fig)


# ------------------------------------------------------------ 상관
def correlations(df, meta, cols):
    a = df[df["결정월"] >= ANALYSIS_START]
    rows = []
    for col in cols:
        d = {"열": col, "ID": meta[col]["ID"], "드라이버": meta[col]["드라이버"]}
        for t in ("G3", "G6"):
            d[f"풀링_{t}"] = a[col].corr(a[t], method="spearman")
            m = a.groupby("결정월")[[col, t]].mean()
            d[f"시도평균_{t}"] = m[col].corr(m[t], method="spearman")
        rows.append(d)
    return pd.DataFrame(rows)


def plot_target_corr(cr, meta):
    cr = cr.sort_values("풀링_G6")
    labels = [f"{meta[c]['ID']} {meta[c]['변수명'][:14]} · {meta[c]['변환'][:16]}" for c in cr["열"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.26 * len(cr) + 1.2), sharey=True)
    y = np.arange(len(cr))
    for ax, t in zip(axes, ("G3", "G6")):
        v = cr[f"풀링_{t}"].values
        ax.barh(y, v, color=[C["red"] if x >= 0 else C["blue"] for x in v], height=0.62)   # 히트맵과 같은 부호 색: 빨강 = 양, 파랑 = 음
        ax.scatter(cr[f"시도평균_{t}"].values, y, s=14, color=C["ink"], zorder=3, label="시도 평균 계열끼리")
        ax.axvline(0, color=C["axis"], lw=0.8)
        ax.set_xlim(-1, 1)
        ax.set_title(f"{t} 와의 Spearman ρ (막대 = 시도×월 풀링, 점 = 시도 평균 계열)", loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7.5)
    axes[1].legend(loc="lower right")
    fig.text(0.5, 0.005, "빨강 = 양(+), 파랑 = 음(−). 분석 구간(2016-01~)의 결정월 기준 값. 전국 공통 변수는 풀링과 시도 평균이 같음", ha="center", fontsize=8, color=C["ink2"])
    return img(fig)


def plot_heatmap(df, meta, cols):
    a = df[df["결정월"] >= ANALYSIS_START]
    order = sorted(cols, key=lambda c: (meta[c]["드라이버"], meta[c]["ID"]))
    corr = a[order].corr(method="spearman")
    n = len(order)
    fig, ax = plt.subplots(figsize=(0.26 * n + 2.2, 0.26 * n + 1.6))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ids = [meta[c]["ID"] for c in order]
    ax.set_xticks(range(n)); ax.set_xticklabels(ids, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(ids, fontsize=7)
    ax.grid(False)
    bounds = [i for i in range(1, n) if meta[order[i]]["드라이버"] != meta[order[i - 1]]["드라이버"]]
    for b in bounds:
        ax.axhline(b - 0.5, color=C["ink"], lw=0.8); ax.axvline(b - 0.5, color=C["ink"], lw=0.8)
    for i in range(n):
        for j in range(n):
            if i != j and abs(corr.values[i, j]) >= 0.6:
                ax.text(j, i, f"{corr.values[i, j]:.1f}".replace("0.", "."), ha="center", va="center", fontsize=7, color=C["ink"])
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Spearman ρ (시도×월 풀링, 분석 구간)", fontsize=8)
    ax.set_title("설명변수 간 상관 — 동인 순서(굵은 선 = 동인 경계), |ρ| ≥ 0.6 인 칸만 숫자 표기", loc="left")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            r = corr.values[i, j]
            if abs(r) >= 0.6:
                pairs.append({"변수 A": f"{ids[i]} {meta[order[i]]['변수명'][:18]} ({meta[order[i]]['변환']})", "변수 B": f"{ids[j]} {meta[order[j]]['변수명'][:18]} ({meta[order[j]]['변환']})", "ρ": round(float(r), 2)})
    pairs = pd.DataFrame(pairs).sort_values("ρ", key=lambda s: -s.abs()) if pairs else pd.DataFrame(columns=["변수 A", "변수 B", "ρ"])
    return img(fig), pairs


# ------------------------------------------------------------ html
def build():
    df, meta = load()
    cols = [c for c in df.columns[11:] if c in meta]
    a = df[df["결정월"] >= ANALYSIS_START]
    cr = correlations(df, meta, cols)
    crm = cr.set_index("열")
    # ① 개요·목록
    n_reg, m0, m1 = df["region"].nunique(), df["결정월"].min(), df["결정월"].max()
    parts = []
    parts.append(f"""<h2>① 데이터셋 개요</h2>
<table class='kv'><tr><th>파일</th><td>{esc(os.path.relpath(VALUES, BASE))} (수식 원본: analysis/output/모형입력표_10차.xlsx)</td></tr>
<tr><th>구조</th><td>{n_reg}개 시도 × 결정월 {m0}~{m1} = {len(df):,}행. 결정월 t = 월말, 그 시점에 공표된 값만 사용(공표시차 반영). 2016-01 이전은 12개월 변화 계산용 버퍼</td></tr>
<tr><th>타깃</th><td>아파트 월세통합가격지수 R(raw V001). R_last = R(t−1). <b>G3 = 100·[R(t+2)/R(t−1) − 1]</b>, <b>G6 = 100·[R(t+5)/R(t−1) − 1]</b> (앞으로 3·6개월 변화율, %). r3·r6 = 같은 결정월 17개 시도 평균 대비 편차</td></tr>
<tr><th>설명변수</th><td>{len(cols)}열 (변수 52개 중 흡수 4·종료 1 제외, 사용자 추가 V078 포함). 변수당 변환 1개, 역할 = 취약성(느린 구조) / 트리거(빠른 신호) / 가격추세</td></tr>
<tr><th>결정 근거</th><td>docs/전처리결정로그_10차.csv (변수마다 변환·시차·결측·정상성·비고), 변수별 탐색 카드 analysis/output/변수카드/</td></tr></table>""")
    tbl = []
    for c in sorted(cols, key=lambda c: (meta[c]["드라이버"], meta[c]["ID"])):
        m = meta[c]
        w = wide_of(df, c)
        tbl.append({"동인": m["동인"][:14], "ID": m["ID"], "변수명": m["변수명"], "변환(열)": m["변환"], "역할": m["역할"], "공표시차": m["공표시차"],
                    "지역": "전국 공통" if is_national(w) else "시도별", "분석구간 결측률": f"{a[c].isna().mean():.1%}",
                    "ρ_G6 풀링": f"{crm.at[c, '풀링_G6']:+.2f}", "ρ_G6 시도평균": f"{crm.at[c, '시도평균_G6']:+.2f}"})
    parts.append("<h2>② 변수 목록 (동인별)</h2>" + pd.DataFrame(tbl).to_html(index=False, escape=True))
    # ③ 타깃
    html_t = ""
    for t, lab in (("G3", "앞으로 3개월 변화율 G3 (%)"), ("G6", "앞으로 6개월 변화율 G6 (%)")):
        w = wide_of(df, t)
        html_t += plot_series(w, f"타깃 {lab}", "17개 시도(회색) · 서울(파랑) · 시도 평균(검정). 정답은 결정월 t 로부터 3·6개월 뒤에 확정", "%")
    parts.append("<h2>③ 타깃 시계열</h2>" + html_t)
    # ④ 설명변수
    cur = None
    html_x = ""
    for c in sorted(cols, key=lambda c: (meta[c]["드라이버"], meta[c]["ID"])):
        m = meta[c]
        if m["동인"] != cur:
            cur = m["동인"]
            html_x += f"<h3>{esc(cur)}</h3>"
        w = wide_of(df, c)
        sub = f"{m['역할']} · 공표시차 {m['공표시차']}개월 · ρ_G6 풀링 {crm.at[c, '풀링_G6']:+.2f}, 시도평균 {crm.at[c, '시도평균_G6']:+.2f}"
        html_x += plot_series(w, f"{m['ID']} {m['변수명']} — {m['변환']}", sub)
    parts.append("<h2>④ 설명변수 시계열 (결정월에 아는 값, 공표시차 반영)</h2>" + html_x)
    # ⑤ 상관
    hm, pairs = plot_heatmap(df, meta, cols)
    parts.append("<h2>⑤ 타깃과의 상관</h2>" + plot_target_corr(cr, meta) +
                 "<h2>⑥ 설명변수 간 상관</h2>" + hm + "<p class='note'><b>|ρ| ≥ 0.6 인 쌍</b> (중복 묶음 판단용 — 10차 결정: 전부 유지하고 학습 단계에서 묶음별 비교)</p>" + pairs.to_html(index=False, escape=True))
    parts.append("""<h2>⑦ 읽는 법·주의</h2><ul>
<li>모든 값은 <b>결정월에 알 수 있는 값</b>이다: 월 변수는 공표시차만큼 뒤로 민 값, 연·분기 변수는 기간 끝 달에 두고 공표시차를 더한 뒤 다음 공표까지 유지한 계단(그래서 계단 모양).</li>
<li>'전국 공통' 변수(금리·M2·CPI·KOSPI 등)는 17개 시도에 같은 값이 들어가 지역 간 차이를 설명하지 못하고 시점 효과와 겹친다.</li>
<li>V024 서울 구 신축허가는 서울 행에만 값이 있어 결측률이 높다(설계대로). V077 임대주택 공급은 공공+민간 사업승인 기준으로 표가 바뀐 연도를 이은 계열이다.</li>
<li>상관은 탐색용 Spearman(순위) 상관이며 분석 구간(2016-01~) 전체를 쓴다. 변수 선택은 학습 단계에서 변수 중요도·묶음별 비교로 정한다(10차 결정).</li>
<li>이 카드는 정적 그림이라 마우스 오버 값은 없다. 숫자는 values CSV 에서 확인한다.</li></ul>""")
    css = ("<style>body{font-family:'Malgun Gothic',system-ui,sans-serif;font-size:13px;color:#0b0b0b;background:#f9f9f7;margin:24px;max-width:1240px}"
           "h1{font-size:21px}h2{font-size:16px;border-bottom:1px solid #c3c2b7;margin-top:32px;padding-bottom:4px}h3{font-size:14px;color:#52514e;margin-top:22px}"
           "table{border-collapse:collapse;font-size:12px;margin:8px 0}th,td{border:1px solid #e1e0d9;padding:3px 7px;text-align:left;vertical-align:top}th{background:#f0efec}"
           "table.kv th{width:90px}img{display:block;margin:6px 0 14px 0;max-width:100%}p.note{color:#52514e}</style>")
    html = (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>변수 통합 소개 10차</title>{css}</head><body>"
            f"<h1>10차 모형 입력 변수 통합 소개 — 타깃·설명변수 시계열과 상관</h1><p class='note'>생성 {dt.date.today().isoformat()} · 자료 {esc(os.path.relpath(VALUES, BASE))}</p>"
            + "".join(parts) + "</body></html>")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"{os.path.relpath(OUT, BASE)} ({os.path.getsize(OUT) / 1e6:.1f} MB): 설명변수 {len(cols)}개, 강한 상관 쌍 {len(pairs)}개")
    return pairs


if __name__ == "__main__":
    build()
