#!/usr/bin/env python3
"""Collect monthly Seoul district statistics without persisting API keys."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

KOSIS_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
RONE_LIST_URL = "https://www.reb.or.kr/r-one/openapi/SttsApiTbl.do"
RONE_DATA_URL = "https://www.reb.or.kr/r-one/openapi/SttsApiTblData.do"
DISTRICTS = (
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구",
    "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구",
    "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구",
    "서초구", "강남구", "송파구", "강동구",
)


class CollectionError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self._secrets: list[str] = []

    def register_secret(self, value: str) -> None:
        self._secrets.append(value)

    def redact(self, message: str) -> str:
        for secret in self._secrets:
            message = message.replace(secret, "[REDACTED]")
            message = message.replace(urllib.parse.quote(secret, safe=""), "[REDACTED]")
        return message

    def get_json(self, url: str, params: dict[str, str | int]) -> object:
        request_url = f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            request_url,
            headers={"Accept": "application/json", "User-Agent": "seoul-monthly-collector/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except (OSError, urllib.error.URLError) as exc:
            raise CollectionError(self.redact(f"API 연결 실패 ({url}): {exc}")) from exc
        try:
            return json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            preview = self.redact(raw[:200].decode("utf-8", "replace"))
            raise CollectionError(f"JSON이 아닌 API 응답: {preview}") from exc


def require_key(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise CollectionError(f"환경 변수 {name}가 설정되지 않았습니다.")
    return value


def rows_from_response(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("row", "rows", "data", "list"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    for value in payload.values():
        if isinstance(value, dict):
            found = rows_from_response(value)
            if found:
                return found
    return []


def api_error(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    result = payload.get("RESULT") or payload.get("result")
    if isinstance(result, dict):
        code = str(result.get("CODE") or result.get("code") or "")
        message = str(result.get("MESSAGE") or result.get("message") or "")
        if code and code not in {"INFO-000", "00", "0", "200"}:
            return f"{code}: {message}"
    return None


def district_of(text: object) -> str | None:
    normalized = re.sub(r"\s+", "", str(text or ""))
    return next((name for name in DISTRICTS if name in normalized), None)


def month_of(value: object) -> str | None:
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) >= 6 and 1 <= int(digits[4:6]) <= 12:
        return digits[:6]
    return None


def first_value(row: dict[str, object], names: tuple[str, ...]) -> object:
    upper = {str(k).upper(): v for k, v in row.items()}
    return next((upper[n] for n in names if n in upper), "")


def normalize_kosis(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for row in rows:
        area = " ".join(str(v) for k, v in row.items()
                        if str(k).upper().endswith("_NM") or str(k) == "행정구역별")
        if "서울" not in area:
            continue
        district = district_of(area)
        month = month_of(first_value(row, ("PRD_DE", "PRD_DE_NM", "시점")))
        if district and month:
            output.append({
                "month": month,
                "district": district,
                "population": first_value(row, ("DT", "값")),
                "unit": first_value(row, ("UNIT_NM", "단위")) or "명",
                "source_table": first_value(row, ("TBL_ID",)) or "DT_1B040A3",
            })
    return deduplicate(output, "KOSIS")


def normalize_rone(rows: list[dict[str, object]], table_id: str) -> list[dict[str, object]]:
    output = []
    for row in rows:
        # R-ONE can split 서울/자치구 over separate classification-name columns.
        area = " ".join(str(v) for k, v in row.items()
                        if "NM" in str(k).upper() or str(k) == "지역명")
        if "서울" not in area:
            continue
        district = district_of(area)
        month = month_of(first_value(row, ("WRTTIME_IDTFR_ID", "WRTTIME", "PRD_DE", "시점")))
        value = first_value(row, ("DTA_VAL", "DT", "VALUE", "값"))
        if district and month and str(value).strip() not in {"", "-"}:
            output.append({
                "month": month,
                "district": district,
                "apartment_sale_price_index": value,
                "unit": first_value(row, ("UI_NM", "UNIT_NM", "단위")) or "지수",
                "source_table": table_id,
            })
    return deduplicate(output, "R-ONE")


def deduplicate(rows: list[dict[str, object]], source: str) -> list[dict[str, object]]:
    result: dict[tuple[object, object], dict[str, object]] = {}
    for row in rows:
        key = (row["month"], row["district"])
        if key in result:
            raise CollectionError(f"{source} 중복 지역/월: {key[0]} {key[1]}")
        result[key] = row
    return sorted(result.values(), key=lambda x: (str(x["month"]), DISTRICTS.index(str(x["district"]))))


def collect_kosis(client: ApiClient, key: str, start: str, end: str) -> list[dict[str, object]]:
    all_rows: list[dict[str, object]] = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        year_start = max(start, f"{year}01")
        year_end = min(end, f"{year}12")
        payload = client.get_json(KOSIS_URL, {
            "method": "getList", "apiKey": key, "format": "json", "jsonVD": "Y",
            "orgId": "101", "tblId": "DT_1B040A3", "itmId": "T20",
            "objL1": "ALL", "prdSe": "M", "startPrdDe": year_start,
            "endPrdDe": year_end,
        })
        if isinstance(payload, dict) and payload.get("err"):
            raise CollectionError(f"KOSIS 오류: {payload.get('err')} {payload.get('errMsg', '')}")
        all_rows.extend(rows_from_response(payload))
    normalized = normalize_kosis(all_rows)
    if not normalized:
        raise CollectionError("KOSIS에서 서울 자치구 월별 자료를 찾지 못했습니다.")
    return normalized


def find_rone_table(client: ApiClient, key: str) -> str:
    matches: list[tuple[str, str]] = []
    for page in range(1, 101):
        payload = client.get_json(RONE_LIST_URL, {
            "KEY": key, "Type": "json", "pIndex": page, "pSize": 100,
        })
        error = api_error(payload)
        if error:
            raise CollectionError(f"R-ONE 오류: {error}")
        rows = rows_from_response(payload)
        if not rows:
            break
        for row in rows:
            name = str(first_value(row, ("STATBL_NM", "TBL_NM", "통계표명")))
            table_id = str(first_value(row, ("STATBL_ID", "TBL_ID", "통계표ID")))
            if table_id and all(word in name for word in ("아파트", "매매", "가격지수")):
                matches.append((table_id, name))
        if len(rows) < 100:
            break
    if not matches:
        raise CollectionError("R-ONE 월별 아파트 매매가격지수 통계표를 찾지 못했습니다. --rone-table-id를 지정하세요.")
    # Prefer the most specific national/district table; stable ordering is reproducible.
    matches.sort(key=lambda item: ("월" not in item[1], "지역" not in item[1], item[1], item[0]))
    return matches[0][0]


def collect_rone(client: ApiClient, key: str, table_id: str, start: str, end: str) -> list[dict[str, object]]:
    all_rows: list[dict[str, object]] = []
    for page in range(1, 1001):
        payload = client.get_json(RONE_DATA_URL, {
            "KEY": key, "Type": "json", "pIndex": page, "pSize": 1000,
            "STATBL_ID": table_id,
        })
        error = api_error(payload)
        if error:
            raise CollectionError(f"R-ONE 오류: {error}")
        rows = rows_from_response(payload)
        all_rows.extend(rows)
        if len(rows) < 1000:
            break
    normalized = normalize_rone(all_rows, table_id)
    normalized = [r for r in normalized if start <= str(r["month"]) <= end]
    if not normalized:
        raise CollectionError("R-ONE에서 서울 자치구 월별 자료를 찾지 못했습니다.")
    return normalized


def validate_coverage(rows: list[dict[str, object]], source: str) -> None:
    months: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        months[str(row["month"])].add(str(row["district"]))
    incomplete = [(month, sorted(set(DISTRICTS) - areas))
                  for month, areas in sorted(months.items()) if areas != set(DISTRICTS)]
    if incomplete or not months:
        month, missing = incomplete[-1] if incomplete else ("없음", list(DISTRICTS))
        raise CollectionError(
            f"{source} 자치구 자료가 불완전합니다. 월={month}, 누락={','.join(missing)}"
        )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def excel_col(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def sheet_xml(rows: list[list[object]]) -> str:
    lines = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>']
    for row_index, row in enumerate(rows, 1):
        cells = []
        for col_index, value in enumerate(row, 1):
            ref = f"{excel_col(col_index)}{row_index}"
            text = escape(str(value if value is not None else ""))
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        lines.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    lines.append('</sheetData></worksheet>')
    return "".join(lines)


def write_xlsx(path: Path, sheets: list[tuple[str, list[list[object]]]]) -> None:
    content_types = ['<?xml version="1.0" encoding="UTF-8"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>']
    for i in range(1, len(sheets) + 1):
        content_types.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
    content_types.append('</Types>')
    workbook_sheets = "".join(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i, (name, _) in enumerate(sheets, 1))
    workbook = ('<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{workbook_sheets}</sheets></workbook>')
    rels = ['<?xml version="1.0" encoding="UTF-8"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    rels.extend(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(sheets) + 1))
    rels.append('</Relationships>')
    root_rels = ('<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "".join(content_types))
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", "".join(rels))
        for i, (_, rows) in enumerate(sheets, 1):
            archive.writestr(f"xl/worksheets/sheet{i}.xml", sheet_xml(rows))


def tabular(rows: list[dict[str, object]]) -> list[list[object]]:
    headers = list(rows[0])
    return [headers] + [[row[h] for h in headers] for row in rows]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-month", default="200801", help="YYYYMM (default: 200801)")
    parser.add_argument("--end-month", default=dt.date.today().strftime("%Y%m"), help="YYYYMM")
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--rone-table-id")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not re.fullmatch(r"\d{6}", args.start_month) or not re.fullmatch(r"\d{6}", args.end_month):
        raise CollectionError("월은 YYYYMM 형식이어야 합니다.")
    kosis_key, rone_key = require_key("KOSIS_API_KEY"), require_key("RONE_API_KEY")
    client = ApiClient()
    client.register_secret(kosis_key)
    client.register_secret(rone_key)

    # These first requests both authenticate and return data; no key is printed.
    kosis_rows = collect_kosis(client, kosis_key, args.start_month, args.end_month)
    table_id = args.rone_table_id or find_rone_table(client, rone_key)
    rone_rows = collect_rone(client, rone_key, table_id, args.start_month, args.end_month)
    validate_coverage(kosis_rows, "KOSIS")
    validate_coverage(rone_rows, "R-ONE")
    print(f"연결 확인 완료: KOSIS {len(kosis_rows):,}행, R-ONE {len(rone_rows):,}행")
    if args.check_only:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir) as temp_name:
        temp = Path(temp_name)
        files = {
            "kosis_seoul_population_monthly.csv": temp / "kosis.csv",
            "rone_seoul_apartment_price_index_monthly.csv": temp / "rone.csv",
            "seoul_monthly_statistics.xlsx": temp / "combined.xlsx",
        }
        write_csv(files["kosis_seoul_population_monthly.csv"], kosis_rows)
        write_csv(files["rone_seoul_apartment_price_index_monthly.csv"], rone_rows)
        metadata = [["source", "table_id", "latest_month", "rows"],
                    ["KOSIS", "DT_1B040A3", max(str(r["month"]) for r in kosis_rows), len(kosis_rows)],
                    ["R-ONE", table_id, max(str(r["month"]) for r in rone_rows), len(rone_rows)]]
        write_xlsx(files["seoul_monthly_statistics.xlsx"], [
            ("KOSIS_인구", tabular(kosis_rows)), ("RONE_가격지수", tabular(rone_rows)),
            ("수집정보", metadata),
        ])
        for name, source in files.items():
            os.replace(source, args.output_dir / name)
    print(f"저장 완료: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CollectionError as error:
        print(f"오류: {error}", file=sys.stderr)
        raise SystemExit(1)
