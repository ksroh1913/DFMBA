# -*- coding: utf-8 -*-
"""연구계획 8차 확정본(docs/연구계획_8차_확정본.md)과 9차 보완설계안(docs/연구계획_9차_보완설계안.md)의 실행 코드.
단계 실행기: run_stage2_data·3_flow·4_index·5_checks (8차 공통), run_v9_stage6·stage7·c·up·fig·fig_index·report (9차).

모듈
  settings   설정 파일(기본 config/plan_v9_settings.yaml; 8차는 plan_v8_settings.yaml) 읽기와 해시
  data       ① 보완 셀 마스킹 → ② 공표 시차 보정 → ③ 변수 변환 → ④ 기준 지역 분해
  targets    타깃 G·Ḡ·r, 전국 계열, 대용계열 접합, 사건 정의
  index      ⑤~⑨ RMPI 변환기 (훈련자료로만 적합)
  split      전진 날짜 분할
  baselines  단순 기준과 선택형 단순기준
  models     사전 지정 모형과 보조 모형
  evaluate   손실, 두 t구간 판정, 블록 부트스트랩, 사건 지표
"""
