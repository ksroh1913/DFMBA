# -*- coding: utf-8 -*-
"""
카드(html) 안에 base64 로 박힌 그림을 PNG 파일로 꺼낸다 — GitHub 에서 html 은 렌더링되지 않으므로 개요 문서(docs/10차_개요.md)가 바로 보여 줄 수 있게.
출력: analysis/output/그림_10차/<카드>_<번호>_<절>.png  (절 이름은 그림 앞에 가장 가까운 h2/h3 제목)
사용: PYTHONUTF8=1 python analysis/export_card_figures.py
"""
import base64
import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARDS = {
    "모형결과": os.path.join(BASE, "analysis", "output", "모형결과_10차", "모형결과카드_10차.html"),
    "변수소개": os.path.join(BASE, "analysis", "output", "변수카드", "00_변수통합소개_10차.html"),
}
OUT = os.path.join(BASE, "analysis", "output", "그림_10차")


def slug(s, n=28):
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"[\\/:*?\"<>|\s]+", "_", s).strip("_")
    return s[:n]


def main():
    os.makedirs(OUT, exist_ok=True)
    for old in os.listdir(OUT):
        if old.endswith(".png"):
            os.remove(os.path.join(OUT, old))
    index = []
    for card, path in CARDS.items():
        if not os.path.exists(path):
            continue
        html = open(path, encoding="utf-8").read()
        k = 0
        for m in re.finditer(r"data:image/png;base64,([A-Za-z0-9+/=]+)", html):
            before = html[:m.start()]
            h2 = re.findall(r"<h2[^>]*>(.*?)</h2>", before, re.S)
            h3 = re.findall(r"<h3[^>]*>(.*?)</h3>", before, re.S)
            sec = slug(h2[-1], 22) if h2 else "그림"
            sub = slug(h3[-1], 16) if h3 and before.rfind("<h3") > before.rfind("<h2") else ""
            k += 1
            name = f"{card}_{k:02d}_{sec}" + (f"_{sub}" if sub else "") + ".png"
            with open(os.path.join(OUT, name), "wb") as f:
                f.write(base64.b64decode(m.group(1)))
            index.append(name)
    with open(os.path.join(OUT, "목록.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(index))
    print(f"{os.path.relpath(OUT, BASE)}: {len(index)}장")


if __name__ == "__main__":
    main()
