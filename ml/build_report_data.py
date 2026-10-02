# -*- coding: utf-8 -*-
"""ml/output 의 실험 결과 CSV -> 시각화 페이지용 JSON (ml/output/report_data.json)"""
import json
import os

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def r(x, n=3):
    return None if pd.isna(x) else round(float(x), n)


def main():
    summ = pd.read_csv(os.path.join(OUT, "결과_요약.csv"))
    fold = pd.read_csv(os.path.join(OUT, "결과_fold별.csv"))
    th = pd.read_csv(os.path.join(OUT, "임계값.csv"))
    feat = pd.read_csv(os.path.join(OUT, "SHAP_변수별.csv"))
    grp = pd.read_csv(os.path.join(OUT, "SHAP_동인별_행.csv"))
    pred = pd.read_csv(os.path.join(OUT, "예측값.csv"))
    used = pd.read_csv(os.path.join(OUT, "사용변수.csv"))
    excl = pd.read_csv(os.path.join(OUT, "제외변수.csv"))

    data = {"summary": [], "fold": [], "thresholds": th.to_dict("records"), "shapGroup": [], "shapTop": [],
            "timeline": {}, "used": used.to_dict("records"),
            "excluded": {row["패널"]: (row["제외변수"] if isinstance(row["제외변수"], str) else "").split("; ")
                         for _, row in excl.iterrows()}}
    for _, x in summ.iterrows():
        data["summary"].append({"panel": x["패널"], "task": x["과제"], "set": x["변수"], "model": x["모형"],
                                "span": x["시험구간"], "auc": r(x["AUC"]), "acc": r(x.get("정확도")),
                                "aucLo": r(x["AUC_연도최저"]), "aucHi": r(x["AUC_연도최고"]), "pooled": r(x["통합AUC"]),
                                "ll": r(x["로그손실"]), "prauc": r(x.get("PR-AUC")), "rate": r(x.get("사건비율")),
                                "n": int(x["시험행수"])})
    for _, x in fold.iterrows():
        rate = x.get("사건비율")
        if x["과제"] == "방향" and isinstance(x.get("정답비율(상승/보합/하락)"), str):
            rate = float(x["정답비율(상승/보합/하락)"].split("/")[0])   # 방향 과제는 상승 비율
        data["fold"].append({"panel": x["패널"], "task": x["과제"], "set": x["변수"], "model": x["모형"],
                             "year": int(x["시험연도"]), "auc": r(x["AUC"]), "rate": r(rate),
                             "prauc": r(x.get("PR-AUC"))})

    # 동인별 SHAP 비중 (시험연도 전체, 행별 |합|의 평균)
    meta = ["패널", "모형", "과제", "시험연도", "region", "month"]
    gcols = [c for c in grp.columns if c not in meta]
    for (p, m, t), g in grp.groupby(["패널", "모형", "과제"]):
        imp = g[gcols].abs().mean()
        tot = imp.sum()
        for k, v in imp.sort_values(ascending=False).items():
            if pd.notna(v) and v > 0:
                data["shapGroup"].append({"panel": p, "model": m, "task": t, "driver": k, "share": r(v / tot, 4)})
    drv = dict(zip(used["변수"], used["동인"]))
    for (p, m, t), g in feat.groupby(["패널", "모형", "과제"]):
        a = g.groupby("변수")[["평균|SHAP|", "평균SHAP"]].mean().sort_values("평균|SHAP|", ascending=False)
        a = a[~a.index.str.startswith("지역_")].head(15)
        tot = g.groupby("변수")["평균|SHAP|"].mean().sum()
        for v, row in a.iterrows():
            data["shapTop"].append({"panel": p, "model": m, "task": t, "var": v, "driver": drv.get(v, ""),
                                    "share": r(row["평균|SHAP|"] / tot, 4), "mean": r(row["평균SHAP"], 4)})

    # 지역별 표본외 경보 확률과 실제 사건 + 동인별 기여 (ET·M3)
    pe = pred[(pred["모형"] == "ET") & (pred["변수"] == "M3")]
    for (p, t), g in pe.groupby(["패널", "과제"]):
        key = f"{p}|{t}"
        data["timeline"][key] = {}
        prob = "p_상승" if t == "방향" else "p_1"
        gg = grp[(grp["패널"] == p) & (grp["모형"] == "ET") & (grp["과제"] == t)]
        for reg, s in g.groupby("region"):
            s = s.sort_values("month")
            c = gg[gg["region"] == reg].set_index("month").reindex(s["month"])
            actual = (s["정답"] == "상승").astype(int) if t == "방향" else s["정답"].astype(int)
            data["timeline"][key][reg] = {
                "month": s["month"].astype(int).tolist(),
                "p": [r(v) for v in s[prob]],
                "y": actual.tolist(),
                "contrib": {d: [r(v, 4) for v in c[d].values] for d in gcols if d in c.columns and c[d].abs().sum() > 0},
            }
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("NaN", "null").replace("</", "<\\/")
    with open(os.path.join(OUT, "report_data.json"), "w", encoding="utf-8") as f:
        f.write(js)

    # 페이지 조립: 템플릿에 데이터와 머리말 정보를 넣는다
    n = summ[(summ["시험구간"] == "2021~2025") & (summ["모형"] == "기준(학습빈도)") & (summ["과제"] == "방향")]
    meta = ["입력 데이터취합_전처리 2차 시트 (실거래 제외)",
            "표본 결정월 2016.01~2026.03",
            "시험 2021~2025 (설계안 2021~2023)"]
    meta += [f"{r['패널']} 시험 {int(r['시험행수']):,}행" for _, r in n.iterrows()]
    tpl = open(os.path.join(os.path.dirname(OUT), "report_template.html"), encoding="utf-8").read()
    html = tpl.replace("__DATA__", js).replace("__META__", json.dumps(meta, ensure_ascii=False))
    dst = os.path.join(OUT, "월세조기경보_1차실험.html")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(html)
    print("ok", len(js) // 1024, "KB ->", dst)


if __name__ == "__main__":
    main()
