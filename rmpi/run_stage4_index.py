# -*- coding: utf-8 -*-
"""
[4단계 지수 구축] 공통·지역 RMPI 와 구성 명세, 품질 점검 (연구계획 8차 확정본 4장, 11장 4단계).

최초 훈련기간(결정월 2016.01 ~ 2018.01−h)으로 RMPI 변환기를 적합해 구성 명세를 쓰고, 같은 변환기로 전체 기간 지수를 만든다
(그림용. 평가에서는 재적합 때마다 그 시점 훈련자료로 다시 적합한다). 예측 성능은 계산하지 않는다.

출력 (설정 output_dir, 9차는 rmpi/output_v9/)
  stage4_구성명세.csv        h별·블록별 구성변수의 유지·제외 이유·관측률·부호(근거)·표준편차·가중치, 동인 지수 유지 여부
  stage4_안정성.csv          재적합 시점(2018.01·2020.01·2022.01·2024.01)별 부호·유지 여부 변화
  stage4_상관중복.csv        동인·하위 묶음 안 구성변수 상관(평균·최대)
  stage4_하나빼기.csv        구성변수를 하나 뺐을 때 동인 지수와의 상관(정보 손실 점검)
  stage4_PCA대조.csv         동인별 첫 주성분과 동일가중 지수의 상관, 첫 주성분 설명 비중
  stage4_월공통비중.csv      지역 변환 입력(분해 전)의 월 공통 비중
  stage4_지수_공통.csv, stage4_지수_지역.csv   전체 기간 지수 시계열(최초 훈련기간 적합 기준)
  fig4_1_공통RMPI.png, fig4_2_지역RMPI.png
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from rmpi import data as D  # noqa: E402
from rmpi import frames as F  # noqa: E402
from rmpi import models as M  # noqa: E402
from rmpi import settings as S  # noqa: E402
from rmpi import targets as T  # noqa: E402
from rmpi import viz as V  # noqa: E402
from rmpi.index import DRIVER_MARK, RMPI  # noqa: E402

plt = V.plt


def fit_common(s, spec, nf, h, T0):
    lo = S.per(s["timing"]["official_first_decision"])
    tr = nf[(nf.index >= lo) & (nf.index <= T0 - h)].copy()
    tr[F.SIGN_C] = tr[f"G{h}"]
    ccols = [c for c in spec.loc[spec.block.isin(["regional", "common"]), "입력"] if c in nf.columns]
    ct = M.features(s, spec, F.A_NAT, common_cols=ccols)
    ct.fit(tr, tr[f"G{h}"])
    return ct, tr


def fit_regional(s, spec, pf, h, T0):
    lo = S.per(s["timing"]["official_first_decision"])
    tr = pf[(pf.P >= lo) & (pf.P <= T0 - h)].copy()
    tr[F.SIGN_R] = tr[f"r{h}"]
    tr[F.SIGN_C] = tr[f"Gnat{h}"]
    dcols = F.regional_cols(pf)
    ct = M.features(s, spec, F.A_REL, regional_cols=dcols, prefix_d="D|")
    ct.fit(tr, tr[f"r{h}"])
    return ct, tr


def rmpi_step(ct, name):
    for n, tr, _ in ct.transformers_:
        if n == name:
            return tr
    return None


def main():
    s = S.load()
    out = os.path.join(BASE, s["output_dir"])
    os.makedirs(out, exist_ok=True)
    V.setup()
    b = D.build(s, "sido")
    spec = b["spec"]
    ml = b["ml"]
    pt = T.regional(s, ml, "sido")
    nat = T.national(s)
    nf = F.national_frame(s, b, nat)
    pf = F.panel_frame(s, b, pt, nat)
    T0 = S.per(s["timing"]["eval_start"])
    lo = S.per(s["timing"]["official_first_decision"])
    hs = s["timing"]["horizons"]

    comps, series_c, series_r = [], {}, {}
    fitted = {}
    for h in hs:
        ct_c, tr_c = fit_common(s, spec, nf, h, T0)
        ct_r, tr_r = fit_regional(s, spec, pf, h, T0)
        fitted[h] = (ct_c, ct_r)
        for ct, blk in ((ct_c, "공통"), (ct_r, "지역")):
            comp = M.compositions(ct)
            comp.insert(0, "h", h)
            comp.insert(1, "훈련기간", f"{lo}~{T0 - h}")
            comps.append(comp)
        full_c = nf.copy()
        full_c[F.SIGN_C] = full_c[f"G{h}"]
        Zc = ct_c.transform(full_c)
        series_c[h] = Zc[[c for c in Zc.columns if c.startswith("RMPI공통|") and not c.endswith("결측")]]
        full_r = pf.copy()
        full_r[F.SIGN_R] = full_r[f"r{h}"]
        full_r[F.SIGN_C] = full_r[f"Gnat{h}"]
        Zr = ct_r.transform(full_r)
        zr = Zr[[c for c in Zr.columns if c.startswith("RMPI지역|") and not c.endswith("결측")]].copy()
        zr.insert(0, "P", pf["P"].values)
        zr.insert(0, "region", pf["region"].values)
        series_r[h] = zr
    comp_all = pd.concat(comps, ignore_index=True)
    comp_all.to_csv(os.path.join(out, "stage4_구성명세.csv"), index=False, encoding="utf-8-sig")
    pd.concat([v.assign(h=h) for h, v in series_c.items()]).rename_axis("P").reset_index().to_csv(
        os.path.join(out, "stage4_지수_공통.csv"), index=False, encoding="utf-8-sig")
    pd.concat([v.assign(h=h) for h, v in series_r.items()]).to_csv(os.path.join(out, "stage4_지수_지역.csv"), index=False, encoding="utf-8-sig")

    # ---------------- 안정성: 재적합 시점별 부호·유지
    rows = []
    for h in (3,):
        for Tx in ("2018-01", "2020-01", "2022-01", "2024-01"):
            Tp = S.per(Tx)
            ct_c, _ = fit_common(s, spec, nf, h, Tp)
            ct_r, _ = fit_regional(s, spec, pf, h, Tp)
            for ct, blk in ((ct_c, "공통"), (ct_r, "지역")):
                step = rmpi_step(ct, f"RMPI{blk}")
                for c in list(step.keep_) + list(step.dropped_):
                    rows.append({"h": h, "재적합시점": Tx, "블록": blk, "입력": c, "유지": c in step.keep_,
                                 "적용부호": step.sign_.get(c, np.nan), "부호근거": step.sign_source_.get(c, step.dropped_.get(c, "")),
                                 "표준편차": step.sd_.get(c, np.nan)})
                for d in list(step.drivers_) + list(step.dropped_drivers_):
                    rows.append({"h": h, "재적합시점": Tx, "블록": blk, "입력": f"[동인 {d}]", "유지": d in step.drivers_,
                                 "부호근거": step.dropped_drivers_.get(d, "")})
    stab = pd.DataFrame(rows)
    stab.to_csv(os.path.join(out, "stage4_안정성.csv"), index=False, encoding="utf-8-sig")
    piv = stab[~stab["입력"].str.startswith("[")].pivot_table(index=["블록", "입력"], columns="재적합시점", values="적용부호", aggfunc="first")
    flips = piv[(piv.nunique(axis=1) > 1)]

    # ---------------- 상관·중복, 하나 빼기, PCA (h=3, 최초 훈련기간)
    h = 3
    ct_c, tr_c = fit_common(s, spec, nf, h, T0)
    ct_r, tr_r = fit_regional(s, spec, pf, h, T0)
    corr_rows, loo_rows, pca_rows = [], [], []
    for ct, blk, tr in ((ct_c, "공통", tr_c), (ct_r, "지역", tr_r)):
        step = rmpi_step(ct, f"RMPI{blk}")
        Z = (tr[step.keep_] - step.mu_) / step.sd_ * step.sign_
        idx_full = step._indices(tr)
        def _flat(groups):   # v9: 묶음 -> 개념 -> 구성변수 (8차: 묶음 -> 구성변수) 모두 지원
            return {g: (sum(v.values(), []) if isinstance(v, dict) else list(v)) for g, v in groups.items()}
        for d, groups_raw in step.struct_.items():
            groups = _flat(groups_raw)
            dm = DRIVER_MARK[d]
            cols = sum(groups.values(), [])
            if len(cols) >= 2:
                cm = Z[cols].corr().abs()
                tri = cm.where(np.triu(np.ones(cm.shape), 1).astype(bool)).stack()
                corr_rows.append({"블록": blk, "동인": dm, "구성변수수": len(cols), "묶음수": len(groups),
                                  "평균|상관|": round(float(tri.mean()), 2), "최대|상관|": round(float(tri.max()), 2),
                                  "최대쌍": " ~ ".join(tri.idxmax())})
                for g, gcols in groups.items():
                    if len(gcols) >= 2:
                        cg = Z[gcols].corr().abs()
                        tg = cg.where(np.triu(np.ones(cg.shape), 1).astype(bool)).stack()
                        corr_rows.append({"블록": blk, "동인": dm, "구성변수수": len(gcols), "묶음수": f"묶음 {g}",
                                          "평균|상관|": round(float(tg.mean()), 2), "최대|상관|": round(float(tg.max()), 2),
                                          "최대쌍": " ~ ".join(tg.idxmax())})
                # PCA
                Zd = Z[cols].dropna()
                if len(Zd) >= 12:
                    p = PCA(n_components=1).fit(Zd)
                    pc = pd.Series(p.transform(Zd)[:, 0], index=Zd.index)
                    r_ = pc.corr(idx_full.loc[Zd.index, dm])
                    pca_rows.append({"블록": blk, "동인": dm, "첫주성분 설명비중": round(float(p.explained_variance_ratio_[0]), 2),
                                     "첫주성분-동일가중지수 상관": round(float(abs(r_)), 2), "표본": len(Zd)})
            # 하나 빼기
            for c in cols:
                sub = RMPI(step.spec[step.spec["입력"] != c], step.block, step.prefix, step.obs_rate_min,
                           step.group_min_share, step.driver_min_share)
                sub.keep_ = [k for k in step.keep_ if k != c]
                sub.mu_, sub.sd_, sub.sign_ = step.mu_.drop(c), step.sd_.drop(c), step.sign_.drop(c)
                def _drop(v):
                    if isinstance(v, dict):
                        o = {cp: [k for k in ks if k != c] for cp, ks in v.items()}
                        return {cp: ks for cp, ks in o.items() if ks}
                    return [k for k in v if k != c]
                sub.struct_ = {dd: {g: _drop(gc) for g, gc in gr.items() if _drop(gc)} for dd, gr in step.struct_.items()}
                sub.struct_ = {dd: gr for dd, gr in sub.struct_.items() if gr}
                if d not in sub.struct_:
                    loo_rows.append({"블록": blk, "동인": dm, "뺀 변수": c, "지수 상관": np.nan, "비고": "유일한 구성변수"})
                    continue
                alt = sub._indices(tr)[dm]
                loo_rows.append({"블록": blk, "동인": dm, "뺀 변수": c, "지수 상관": round(float(alt.corr(idx_full[dm])), 3), "비고": ""})
    pd.DataFrame(corr_rows).to_csv(os.path.join(out, "stage4_상관중복.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(loo_rows).to_csv(os.path.join(out, "stage4_하나빼기.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(pca_rows).to_csv(os.path.join(out, "stage4_PCA대조.csv"), index=False, encoding="utf-8-sig")

    # ---------------- 지역 변환 입력의 월 공통 비중 (분해 전, 최초 훈련기간)
    X = b["X"]
    sub = X[(X.P >= lo) & (X.P <= T0 - 1)]
    mc = []
    for c in spec.loc[spec.block == "regional", "입력"]:
        v = sub[["P", c]].dropna()
        if len(v) < 30:
            continue
        mu = v[c].mean()
        tot = ((v[c] - mu) ** 2).sum()
        mfe = v.groupby("P")[c].transform("mean")
        mc.append({"입력": c, "월 공통 비중": round(float(((mfe - mu) ** 2).sum() / tot), 3)})
    pd.DataFrame(mc).to_csv(os.path.join(out, "stage4_월공통비중.csv"), index=False, encoding="utf-8-sig")

    # ---------------- 그림
    end = S.per(s["timing"]["posthoc_label_end"])
    sc = series_c[3]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6), sharex=True)
    for ax, c in zip(axes.ravel(), sc.columns):
        y = sc[c].loc[lo:end]
        ax.plot(y.index.to_timestamp(), y.values, color=V.SERIES[0])
        ax.axhline(0, color=V.BASE, lw=0.8)
        ax.axvspan(lo.to_timestamp(), (T0 - 3 + 1).to_timestamp(), color=V.WASH, zorder=0, lw=0)
        ax.set_title(c.replace("RMPI공통|", "공통 ") + " " + s["driver_names"][int([k for k, v in DRIVER_MARK.items() if v == c[-1]][0])][2:], fontsize=9.5)
    for ax in axes.ravel()[len(sc.columns):]:
        ax.axis("off")
    fig.suptitle("공통 RMPI 하위지수 (h=3, 최초 훈련기간 2016.01~2017.10 적합 기준, 표준화 단위). 음영은 적합에 쓴 기간",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_1_공통RMPI.png"))
    plt.close(fig)

    sr = series_r[3]
    cols = [c for c in sr.columns if c.startswith("RMPI지역|")]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6), sharex=True)
    for ax, c in zip(axes.ravel(), cols):
        w = sr.pivot(index="P", columns="region", values=c).loc[lo:end]
        x = w.index.to_timestamp()
        ax.fill_between(x, w.min(axis=1).values, w.max(axis=1).values, color=V.SERIES[0], alpha=0.10, lw=0, label="17개 시도 범위")
        ax.plot(x, w["서울"].values, color=V.SERIES[0], label="서울")
        ax.plot(x, w["세종"].values, color=V.SERIES[1], label="세종")
        ax.axhline(0, color=V.BASE, lw=0.8)
        ax.axvspan(lo.to_timestamp(), (T0 - 3 + 1).to_timestamp(), color=V.WASH, zorder=0, lw=0)
        ax.set_title(c.replace("RMPI지역|", "지역 ") + " " + s["driver_names"][int([k for k, v in DRIVER_MARK.items() if v == c[-1]][0])][2:], fontsize=9.5)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("지역 RMPI 하위지수 (h=3, 월내 편차, 최초 훈련기간 적합 기준). 서울·세종과 17개 시도 범위", x=0.01, ha="left",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_2_지역RMPI.png"))
    plt.close(fig)

    # ---------------- 출력 요약
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.max_rows", 200)
    c3 = comp_all[comp_all.h == 3]
    print(c3[["블록", "동인", "하위묶음", "입력", "유지", "제외이유", "관측률", "예상부호", "적용부호", "부호근거", "동인지수유지", "동인제외이유"]].to_string(index=False))
    print("\n부호가 재적합 시점에 따라 바뀐 구성변수(h=3):")
    print(flips.to_string() if len(flips) else "없음")
    print("\n상관·중복:\n" + pd.DataFrame(corr_rows).to_string(index=False))
    print("\nPCA 대조:\n" + pd.DataFrame(pca_rows).to_string(index=False))
    print("\n하나 빼기(상관 0.9 미만만):\n" + pd.DataFrame(loo_rows).query("`지수 상관` < 0.9").to_string(index=False))
    S.manifest(s, {"stage": 4}).to_csv(os.path.join(out, "stage4_manifest.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
