# DFMBA — 서울 자치구 아파트 월세 리스크 데이터 파이프라인

데이터사전 `docs/260928_데이터사전_취합_수정.xlsx`의 **Variable_Master 시트 52개 변수**만
공공 API로 수집하고 전처리한다.

```
수집(collect/) → raw/ 저장 → 전처리(preprocess/) → processed/ → 병합(merge/) → 데이터취합_YYYYMMDD.xlsx
```

## 실행

```bash
# 1. 수집 (재실행 시 신규·최근 구간만 받음)
python collect/collect_kosis.py          # --full : 전 기간 재수집
python collect/collect_rone.py
python collect/collect_ecos.py
python collect/collect_molit.py          # 실거래 건별. apt_rent / apt_trade 지정 가능

# 2. 전처리 (raw만 읽으므로 API 호출 없음)
for f in preprocess/prep_*.py; do python "$f"; done

# 3. 데이터사전 변수가 모두 나오는지 점검
python check_variables.py

# 4. 병합: 17시도_월별 / 서울25구_월별 + 분기·반기·연간 시트 (로데이터 취합본, 빈칸 그대로)
python merge/build_panel.py              # 2011-01 ~ 최신월 (시작월 인자로 변경 가능)

# 5. 결측 보완 + 가용시점 정렬 (별도 파일, 로데이터 취합본은 그대로 둠)
python collect/collect_supplement.py     # 보완용 원자료(사전 변수 아님) -> raw/7_보완용_사전외/
python impute/build_preprocessed.py      # -> 데이터취합_전처리_YYYYMMDD.xlsx
```

전처리본의 규칙: 원본 값은 바꾸지 않고 빈칸만 채우며, 채운 셀마다 옆 `<컬럼>__꼬리표`에
방법을 남긴다. `2차_공표시점반영(ML용)` 시트는 `1차_결측보완` 결과를 공표 시점만큼 밀어, 각 행의 월말에 실제로 알 수 있던 값만 담은 ML 입력용이다
(공표 시차는 `variables.py`의 `RELEASE`, `RELEASE_MONTHLY`). 목표변수는 `Y_`로 시작하며,
서울 구 `Y_V001_학습용`의 2015.06 이전 값은 학습 전용이다(평가는 `Y_평가사용가능=1`인 행만).

R-ONE은 페이지를 넘겨 받을 때 행이 빠질 수 있어, 가끔 `python collect/collect_rone.py --verify`로
연도별 건수를 대조한다(건수만 조회하므로 1~2분).

API 키는 `.env`에 둔다(저장소에 올리지 않음): `KOSIS_API_KEY`, `RONE_API_KEY`,
`ECOS_API_KEY`, `DATA_GO_KR_API_KEY`. 데이터사전 변수는 모두 이 4개 키로 수집한다.

## 구성

| 경로 | 내용 |
|---|---|
| `variables.py` | **Master ID ↔ API 표 ↔ 전처리 ↔ processed 통계명 대응표.** 병합 단계는 이것만 보고 변수를 찾는다 |
| `check_variables.py` | 대응표의 모든 변수가 processed에 있는지, 기간·지역 커버리지 출력 |
| `collect/` | 기관별 수집기 4개 (KOSIS·R-ONE·ECOS·국토부 실거래). 표마다 Master ID 주석 |
| `preprocess/` | 데이터 특성별 전처리. 출력은 `광역지자체, 자치구, 연월(또는 기간), 통계명, 항목명, 값, 단위` 형식 |
| `raw/` | API 응답 원본 (git 제외, 수집기로 재생성). 데이터사전 1차 동인별 폴더(`1_임차수요압력` … `6_거시경기금융시장여건`, `7_보완용_사전외`)에 `<Master ID>_<데이터명>_<원 표 ID>.csv` 로 저장. 실거래는 `V006_아파트전월세실거래/`, `V007_아파트매매실거래/` 하위에 구별 파일 |
| `raw_layout.py` | raw 파일 위치표. 코드 안의 `raw:kosis/DT_1C96` 같은 표기는 (출처, 원래 이름) 키이고 실제 위치는 이 표가 정한다 |
| `processed/` | 데이터사전 변수만 |
| `analysis/` | 데이터사전 밖의 분석 (실거래 기반 월세 품질조정 지수, 유형별 집계 등). 결과는 `analysis/output/` |
| `archive/` | 대체된 구버전 코드·산출물 (v1 통합 스크립트 등). 파이프라인에서 쓰지 않음 |

## 데이터 처리 원칙

- 월별이 아닌 자료(분기·반기·연간)는 기준기간 그대로 둔다. 공표 전 월에 채우면 미래정보가 섞인다.
- 기준시점이나 조사 방식이 다른 계열은 이어 붙이지 않고 통계명을 나눈다
  (월세가격지수 구/신, 수급동향 구/현행, 가계동향 구/신계열, 임대주택 표별).
- 광주·전남은 2026-07부터 통합코드로만 공표된다. 구 단위 표는 광주 5개 구 합계로 광주를,
  통합값에서 뺀 나머지로 전남을 복원한다(저량·순이동만. 전입·전출은 복원 불가).
- 실거래 raw는 `_firstSeen`/`_lastSeen`으로 취소(응답에서 사라진 건)를 추적하고,
  전처리는 월별 최신 확인분만 쓴다.
