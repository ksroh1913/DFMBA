# 서울 자치구 월별 통계 수집기

KOSIS의 **주민등록인구**와 R-ONE의 **아파트 매매가격지수**를 서울 25개
자치구 단위로 수집해 CSV와 XLSX로 저장하는 표준 라이브러리 기반 Python
프로그램입니다. API 키는 환경 변수에서만 읽으며 URL, 로그, 산출물에 기록하지
않습니다.

## 실행

Python 3.11 이상에서 별도 패키지 설치 없이 실행할 수 있습니다.

```bash
export KOSIS_API_KEY='...'
export RONE_API_KEY='...'
python scripts/collect_seoul_monthly.py --output-dir data/output
```

기본 산출물은 다음과 같습니다.

* `kosis_seoul_population_monthly.csv`
* `rone_seoul_apartment_price_index_monthly.csv`
* `seoul_monthly_statistics.xlsx` (`KOSIS_인구`, `RONE_가격지수`, `수집정보` 시트)

KOSIS는 `DT_1B040A3`(행정구역/성별 주민등록인구)의 총인구 항목을 월별로
요청합니다. R-ONE 통계표 ID는 API의 통계표 목록에서 이름에 `아파트`, `매매`,
`가격지수`가 모두 포함된 월별 표를 자동으로 고릅니다. 운영 환경에서 목록 검색이
제한되거나 다른 표를 써야 한다면 `--rone-table-id`로 명시할 수 있습니다.

```bash
python scripts/collect_seoul_monthly.py --rone-table-id '통계표_ID'
```

`--start-month YYYYMM`으로 시작 월을 바꿀 수 있습니다(기본 `200801`). 종료 월은
실행 시점의 최신 월이며, 아직 공표되지 않은 월은 응답에 없으므로 자동으로 제외됩니다.
수집기는 정확히 25개 자치구만 남기며, 지역/월 중복 또는 자치구 누락이 있으면 실패하여
불완전한 파일을 만들지 않습니다. 기존 파일은 모든 검증이 끝난 뒤 원자적으로
교체합니다.

## 보안 및 연결 확인

두 API의 실제 자료 응답을 받아 연결과 인증을 함께 확인합니다. `--check-only`는 전체
응답과 25개 자치구 완전성을 검증하되 파일은 생성하지 않습니다.
키 자체는 예외 메시지에서도 `[REDACTED]`로 치환됩니다. `.gitignore`는 `.env`와
생성된 자료 디렉터리를 제외합니다.

```bash
python scripts/collect_seoul_monthly.py --check-only
```

## 테스트

```bash
python -m unittest discover -s tests -v
```
