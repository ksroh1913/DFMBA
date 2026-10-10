# -*- coding: utf-8 -*-
"""카드(html) 공통 스타일과 조립 도우미 — 결과 카드·변수 소개 카드가 같은 모양을 갖게 한다."""
import re

CSS = """<style>
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--accent:#1c5cab}
*{box-sizing:border-box}
body{font-family:'Malgun Gothic',system-ui,-apple-system,'Segoe UI',sans-serif;font-size:13.5px;line-height:1.65;color:var(--ink);background:var(--page);margin:0 auto;padding:28px 20px 60px;max-width:1240px}
header.top h1{font-size:22px;margin:0 0 6px;letter-spacing:-0.01em}header.top p{margin:0 0 4px;color:var(--ink2)}
nav.toc{display:flex;flex-wrap:wrap;gap:6px 8px;font-size:12.5px;margin:12px 0 22px}
nav.toc a{color:var(--ink2);text-decoration:none;border:1px solid var(--grid);border-radius:999px;padding:3px 11px;background:var(--surface)}nav.toc a:hover{border-color:var(--axis);color:var(--ink)}
section.card{background:var(--surface);border:1px solid var(--grid);border-radius:12px;padding:18px 22px 14px;margin:0 0 18px;overflow-x:auto}
h2{font-size:16.5px;margin:0 0 10px;padding-bottom:8px;border-bottom:1px solid var(--grid)}h3{font-size:14px;color:var(--ink2);margin:18px 0 8px}h4{font-size:13.5px;margin:14px 0 6px}
table{border-collapse:collapse;font-size:12px;margin:8px 0;max-width:100%}
th,td{border:1px solid var(--grid);padding:4px 8px;text-align:right;vertical-align:top;font-variant-numeric:tabular-nums}
th{background:#f0efec;text-align:left;font-weight:600}td:first-child{text-align:left}tr:nth-child(even) td{background:#f7f7f4}
table.kv th{width:100px}table.kv td,table.left td{text-align:left}
img{display:block;margin:8px 0 16px;max-width:100%;border:1px solid var(--grid);border-radius:8px;background:#fff}
p.note{color:var(--ink2);margin:6px 0 10px}details{margin:8px 0}summary{cursor:pointer;color:var(--ink2)}
.tiles{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0 14px}
.tile{flex:1 1 230px;border:1px solid var(--grid);border-radius:10px;padding:12px 14px;background:#fff}
.tile .lab{font-size:12px;color:var(--ink2)}.tile .num{font-size:26px;font-weight:700;margin:2px 0;letter-spacing:-0.02em}.tile .sub{font-size:12px;color:var(--ink2)}.tile .good{color:var(--accent)}
div.analysis h4{margin:14px 0 6px}ul{padding-left:20px}li{margin:3px 0}
.caveat{border:1px solid #e6c48f;background:#fff7e8;border-radius:10px;padding:10px 14px;margin:12px 0}.caveat .t{display:block;font-weight:700;margin-bottom:4px}.caveat ul{margin:4px 0}
a{color:var(--accent)}
</style>"""


def wrap_sections(body):
    """<h2> 마다 한 섹션 카드로 감싸고, 맨 위에 목차를 단다"""
    titles = re.findall(r"<h2>(.*?)</h2>", body, re.S)
    for k, t in enumerate(titles):
        body = body.replace(f"<h2>{t}</h2>", f"<h2 id='s{k}'>{t}</h2>", 1)
    body = re.sub(r"<h2 id=", "</section><section class='card'><h2 id=", body)
    body = body[len("</section>"):] + "</section>" if body.startswith("</section>") else body + "</section>"
    toc = "<nav class='toc'>" + "".join(f"<a href='#s{k}'>{re.sub(r'<[^>]+>', '', t)[:34]}</a>" for k, t in enumerate(titles)) + "</nav>"
    return toc + body


def page(title, subtitle, body, lang="ko"):
    return (f"<!doctype html><html lang='{lang}'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>{title}</title>{CSS}</head><body>"
            f"<header class='top'><h1>{title}</h1><p>{subtitle}</p></header>" + wrap_sections(body) + "</body></html>")
