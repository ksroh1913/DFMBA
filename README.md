# DFMBA — 아파트 월세 흐름 진단과 급변 예측

> **10차(2026-10, 재구축) 안내 → [`docs/10차_개요.md`](docs/10차_개요.md)**
> 변수 52개를 raw 원자료부터 다시 전처리하고(결정 로그), 학습 데이터·롤링 모형·기여도 분석을 새로 만든 라운드. 산출물 위치·핵심 결과·재현 방법이 그 문서에 있다. 아래 내용은 9차(RMPI) 기준이다.

17개 시도(보조: 서울 25개 구) 아파트 월세통합가격지수의 h = 1·3·6개월 변화율과 급락·급등 사건을,
가격 추세·시장심리·거시 변수를 묶은 RMPI(Rental Market Pressure Index)와 실거래 정보로 예측할 수 있는지 전진(walk-forward) 평가한 연구다.

이 저장소에는 **최종 계획, 최종 결과, 결과 해설 페이지, 그리고 그것을 다시 만드는 데 필요한 자료와 코드만** 둔다.
이전 라운드(1~8차 계획·결과, 이전 날짜의 데이터 파일, 1차 ML 실험, 구버전 수집 코드)는 저장소에서 지웠고 git 이력(커밋 `de16c0d` 이전)에 남아 있다.

## 최종 산출물

| 구분 | 파일 |
|---|---|
| 최종 계획 | `docs/연구계획_9차_보완설계안.md` (9차 후속 탐색 분석 설계, 확정) · 그 바탕인 `docs/연구계획_8차_확정본.md` |
| 설정(동결) | `config/plan_v9_settings.yaml` (v9.0) · 그 부모 `config/plan_v8_settings.yaml` |
| 최종 결과 | `docs/연구결과_9차_후속분석_보고.md` (수치는 모두 `rmpi/output_v9/` 의 CSV 에서 생성) |
| 결과 해설 페이지 | `docs/월세예측_결과해설.html` (자료를 안에 넣은 단일 파일; 아티팩트로 게시한 원본) |
| 결과 산출물 | `rmpi/output_v9/` (v9_s1~s7 표, fig0_5·fig3~fig6 그림, `참고_수정전/` 적합 주기 민감도, `참고_8차/` 9차 보고·설계안이 가리키는 8차 산출물 동결 사본과 8차 보고서 정오표) |
| 변수 탐색 | `analysis/output/plan_v9_eda_*.csv` (탐색 창 변수 탐색), `plan_v9_alpha_check.csv`·`plan_v9_data_audit.csv`·`plan_v9_extension_check*.csv` (9차 설계안 근거, 8차 설정·20261004 입력의 동결 결과), `check_settings_v9.0.csv` |
| 최종 입력 자료 | `데이터취합_20261005.xlsx` (로데이터 취합본) · `데이터취합_전처리_20261005.xlsx` (결측 보완 + 공표시점 반영, 모형 입력) |

## 재현 순서

```bash
# 0. 설정 점검 (열 이름·사건 경계·보완 셀·시차 보정을 자료와 대조. 성능 계산 없음)
RMPI_SETTINGS=config/plan_v9_settings.yaml python analysis/check_settings.py

# 1. 변수 탐색 (결정월 <= 2017.12 창에서만)
RMPI_SETTINGS=config/plan_v9_settings.yaml python analysis/plan_v9_eda.py
RMPI_SETTINGS=config/plan_v9_settings.yaml python analysis/plan_v9_eda_heatmap.py

# 2~5. 자료 진단 → 흐름 그림 → RMPI 구축 → 파이프라인 점검 (출력 rmpi/output_v9/)
for s in run_stage2_data run_stage3_flow run_stage4_index run_stage5_checks; do
  RMPI_SETTINGS=config/plan_v9_settings.yaml python rmpi/$s.py; done

# 6. 평가 ①~④ (전진 평가, 오래 걸림) 와 사후 추가(개별 변수 모형 C, 급등 경보)
RMPI_SETTINGS=config/plan_v9_settings.yaml python rmpi/run_v9_stage6.py
RMPI_SETTINGS=config/plan_v9_settings.yaml python rmpi/run_v9_c.py
RMPI_SETTINGS=config/plan_v9_settings.yaml python rmpi/run_v9_up.py

# 7. 해석·보조 분석 (A 동인 제거, B 실거래 수준형, C 서울 구, D TH V11 행 비교)
RMPI_SETTINGS=config/plan_v9_settings.yaml python rmpi/run_v9_stage7.py            # --parts A,B,C,D

# 그림과 보고서
RMPI_SETTINGS=config/plan_v9_settings.yaml python rmpi/run_v9_fig.py
RMPI_SETTINGS=config/plan_v9_settings.yaml python rmpi/run_v9_fig.py --target up
RMPI_SETTINGS=config/plan_v9_settings.yaml python rmpi/run_v9_fig_index.py
RMPI_SETTINGS=config/plan_v9_settings.yaml python rmpi/run_v9_report.py            # -> docs/연구결과_9차_후속분석_보고.md
```

`RMPI_SETTINGS` 를 주지 않으면 `rmpi/settings.py` 는 `config/plan_v9_settings.yaml` 을 읽는다. 9차 설계안의 근거 스크립트 가운데
`analysis/plan_v9_alpha_check.py`, `plan_v9_data_audit.py`, `plan_v9_extension_check.py` 는 8차 설정과 20261004 입력으로 돈 것이라
(스크립트 안에 고정) 결과 CSV 만 남겼다. 다시 돌리려면 `git show de16c0d:데이터취합_전처리_20261004.xlsx > 데이터취합_전처리_20261004.xlsx`
(병합본도 같은 방법)로 복원한다. `plan_v9_eda.py` 도 8차 설정(V044 포함)에 20261005 입력으로 돈 것이며 스크립트가 그렇게 고정돼 있다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `docs/` | 최종 계획·결과·해설 페이지, 데이터사전 `260928_데이터사전_취합_수정.xlsx` |
| `config/` | 동결된 설정 (`plan_v9_settings.yaml`, 부모 `plan_v8_settings.yaml`) |
| `rmpi/` | 연구 파이프라인. 라이브러리(`settings` `data` `targets` `frames` `index` `models` `engine` `evaluate` `baselines` `split` `viz`)와 단계 실행기(`run_stage2~5`, `run_v9_*`) |
| `rmpi/output_v9/` | 9차 산출물 전부 |
| `analysis/` | 설정 점검(`check_settings.py`)과 9차 설계 근거 스크립트(`plan_v9_*.py`). 결과는 `analysis/output/` |
| `variables.py` | **Master ID ↔ API 표 ↔ 전처리 ↔ processed 통계명 대응표.** 병합 단계는 이것만 보고 변수를 찾는다 |
| `check_variables.py` | 대응표의 모든 변수가 processed 에 있는지, 기간·지역 커버리지 출력 |
| `collect/` | 기관별 수집기 (KOSIS·R-ONE·ECOS·국토부 실거래, 보완용). 표마다 Master ID 주석 |
| `preprocess/` | 데이터 특성별 전처리. 출력은 `광역지자체, 자치구, 연월(또는 기간), 통계명, 항목명, 값, 단위` 형식 |
| `merge/`, `impute/` | 병합(`build_panel.py` → `데이터취합_YYYYMMDD.xlsx`)과 결측 보완·공표시점 반영(`build_preprocessed.py` → `데이터취합_전처리_YYYYMMDD.xlsx`) |
| `raw/` | API 응답 원본. 데이터사전 1차 동인별 폴더(`1_임차수요압력` … `6_거시경기금융시장여건`, `7_보완용_사전외`)에 `<Master ID>_<데이터명>_<원 표 ID>.csv`. 실거래 건별 자료(`V006_*/`, `V007_*/`)는 수 GB 라 git 제외(`collect/collect_molit.py` 로 재수집) |
| `raw_layout.py` | raw 파일 위치표. 코드 안의 `raw:kosis/DT_1C96` 같은 표기는 (출처, 원래 이름) 키이고 실제 위치는 이 표가 정한다 |
| `processed/` | 데이터사전 변수만 (전처리 출력) |

## 자료 만들기

데이터사전 `docs/260928_데이터사전_취합_수정.xlsx` 의 Variable_Master 시트 변수만 공공 API 로 수집하고 전처리한다.

```
수집(collect/) → raw/ → 전처리(preprocess/) → processed/ → 병합(merge/) → 데이터취합_YYYYMMDD.xlsx → 보완(impute/) → 데이터취합_전처리_YYYYMMDD.xlsx
```

```bash
python collect/collect_kosis.py          # --full : 전 기간 재수집
python collect/collect_rone.py
python collect/collect_ecos.py
python collect/collect_molit.py          # 실거래 건별. apt_rent / apt_trade 지정 가능
python collect/collect_supplement.py     # 보완용 원자료(사전 변수 아님) -> raw/7_보완용_사전외/
for f in preprocess/prep_*.py; do python "$f"; done
python check_variables.py
python merge/build_panel.py              # 2011-01 ~ 최신월
python impute/build_preprocessed.py
```

전처리본의 규칙: 원본 값은 바꾸지 않고 빈칸만 채우며, 채운 셀마다 옆 `<컬럼>__꼬리표` 에 방법을 남긴다.
`2차_공표시점반영(ML용)` 시트는 `1차_결측보완` 결과를 공표 시점만큼 밀어, 각 행의 월말에 실제로 알 수 있던 값만 담은 모형 입력용이다
(공표 시차는 `variables.py` 의 `RELEASE`, `RELEASE_MONTHLY`). 목표변수는 `Y_` 로 시작하며,
서울 구 `Y_V001_학습용` 의 2015.06 이전 값은 학습 전용이다(평가는 `Y_평가사용가능=1` 인 행만).

R-ONE 은 페이지를 넘겨 받을 때 행이 빠질 수 있어, 가끔 `python collect/collect_rone.py --verify` 로 연도별 건수를 대조한다.

API 키는 `.env` 에 둔다(저장소에 올리지 않음): `KOSIS_API_KEY`, `RONE_API_KEY`, `ECOS_API_KEY`, `DATA_GO_KR_API_KEY`.

## 데이터 처리 원칙

- 월별이 아닌 자료(분기·반기·연간)는 기준기간 그대로 둔다. 공표 전 월에 채우면 미래정보가 섞인다.
- 기준시점이나 조사 방식이 다른 계열은 이어 붙이지 않고 통계명을 나눈다
  (월세가격지수 구/신, 수급동향 구/현행, 가계동향 구/신계열, 임대주택 표별).
- 광주·전남은 2026-07부터 통합코드로만 공표된다. 구 단위 표는 광주 5개 구 합계로 광주를,
  통합값에서 뺀 나머지로 전남을 복원한다(저량·순이동만. 전입·전출은 복원 불가).
- 실거래 raw 는 `_firstSeen`/`_lastSeen` 으로 취소(응답에서 사라진 건)를 추적하고, 전처리는 월별 최신 확인분만 쓴다.
