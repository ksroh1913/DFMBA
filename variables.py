# -*- coding: utf-8 -*-
"""
데이터사전(docs/260928_데이터사전_취합_수정.xlsx) Variable_Master 시트의 최종 변수 52개
↔ 수집·전처리 코드 대응표. 이 목록 밖의 변수는 수집하지 않는다.
(같은 파일의 Selection_Log/Source_Inventory 77행은 선별 전 원본 목록이라 기준이 아니다)

변수 하나가 어느 API의 어느 표에서 오고(raw), 어느 전처리 스크립트를 거쳐
어느 processed 파일의 어떤 통계명으로 나오는지를 한 곳에 적는다.
병합 단계는 이 표만 보고 변수를 찾아가면 된다.

  status
    수집     raw 수집과 전처리가 모두 연결됨
    중복     같은 원자료를 다른 조원이 따로 적은 행. same_as 변수로 대신한다

  freq      M 월 / Q 분기 / H 반기 / Y 연
  level     시도 / 서울구 / 시도+서울구 / 전국   (값이 존재하는 지역 단위)
  stat      processed 파일의 '통계명' 값. items 가 있으면 '항목명'까지 거른다.

검증: python check_variables.py
"""

P = "processed/"

# 자주 쓰는 processed 파일
PRICE = P + "부동산원_가격지수_아파트.csv"
SUPPLY_DEMAND = P + "부동산원_수급동향_아파트.csv"
TRANSACTION = P + "부동산원_아파트거래현황.csv"
POP = P + "주민등록_인구세대이동_월별.csv"
HOUSING = P + "주택건설실적_월별.csv"
ANNUAL = P + "연간_소득생산공급.csv"
ECOS_M = P + "ECOS_거시경제_전국.csv"
ECOS_Q = P + "ECOS_거시경제_전국_분기.csv"
RTMS = P + "서울_아파트_실거래_월별건수.csv"


def V(vid, name, freq, level, source, table, prep, processed, stat, items=None,
      status="수집", same_as=None, note=""):
    return dict(id=vid, name=name, freq=freq, level=level, source=source, table=table,
                prep=prep, processed=processed, stat=stat, items=items or [],
                status=status, same_as=same_as, note=note)


VARIABLES = [
    # ------------------------------------------------------------------ 가격·임대시장
    V("V001", "아파트 월세통합가격지수", "M", "시도+서울구", "R-ONE", "A_2024_00054",
      "prep_rone_price_index", PRICE, "월세통합가격지수_아파트",
      note="종속변수 후보. 동일월 설명변수 사용 금지(leakage)"),
    V("V002", "아파트 매매가격지수", "M", "시도+서울구", "R-ONE", "A_2024_00045",
      "prep_rone_price_index", PRICE, "매매가격지수_아파트"),
    V("V003", "아파트 전세가율", "M", "시도+서울구", "R-ONE", "A_2024_00072",
      "prep_rone_jeonse_ratio", P + "부동산원_전세가율_아파트.csv", "전세가율_아파트"),
    V("V004", "아파트 전월세전환율", "M", "시도+서울구", "R-ONE", "A_2024_00156",
      "prep_rone_conv_rate", P + "부동산원_전월세전환율_아파트.csv", "지역별전월세전환율_아파트"),
    V("V005", "아파트 월세 수급동향", "M", "시도", "R-ONE", "A_2024_00078",
      "prep_rone_supply_demand", SUPPLY_DEMAND, "월세수급동향_아파트"),
    V("V006", "아파트 전월세 실거래 원본", "M", "서울구", "국토부 RTMS", "RTMSDataSvcAptRent",
      "prep_rtms_counts", RTMS, "전월세실거래_건수",
      note="건별 원본은 raw/4_임대시장수급전환구조/V006_아파트전월세실거래/. processed는 신고건수(공식 거래량 아님). 신규·갱신 건수는 계약구분 기록률이 낮아 취합하지 않음"),
    V("V007", "아파트 매매 실거래 원본", "M", "서울구", "국토부 RTMS", "RTMSDataSvcAptTradeDev",
      "prep_rtms_counts", RTMS, "매매실거래_건수",
      note="건별 원본은 raw/3_금융여건상대가격/V007_아파트매매실거래/. 해제 건은 제외하고 셈(해제 건수는 취합 안 함)"),
    # ------------------------------------------------------------------ 임차수요
    V("V009", "주민등록 인구", "M", "시도+서울구", "KOSIS", "101/DT_1B040A3",
      "prep_kosis_population", POP, "주민등록인구"),
    V("V010", "주민등록 세대수", "M", "시도+서울구", "KOSIS", "101/DT_1B040B3",
      "prep_kosis_population", POP, "주민등록세대수"),
    V("V011", "20–29세·30–39세 인구", "M", "시도+서울구", "KOSIS", "101/DT_1B04006",
      "prep_kosis_population", POP, "주민등록인구_연령",
      items=["20-29세", "30-39세", "20-39세", "20-39세 비중"]),
    V("V012", "순이동자수", "M", "시도+서울구", "KOSIS", "101/DT_1B26001_A01",
      "prep_kosis_population", POP, "인구이동", items=["순이동"]),
    V("V013", "고용률·취업자", "M", "시도", "KOSIS", "101/DT_1DA7004S",
      "prep_kosis_employment_monthly", P + "경제활동인구_시도_월별.csv", "경제활동인구_시도월별",
      items=["고용률", "취업자"], note="시도 월별. 서울 구는 반기만 있어 V013H로 별도"),
    V("V013H", "고용률·취업자 (서울 구 반기)", "H", "서울구", "KOSIS", "101/DT_1ES3A01S",
      "prep_kosis_employment_half", P + "경제활동인구_서울구_반기.csv", "지역별고용조사_서울구반기",
      items=["고용률", "취업자"], note="2021H1~. 4월 조사 8월 공표, 10월 조사 익년 2월 공표"),
    V("V015", "1인당 GRDP", "Y", "시도", "KOSIS", "101/DT_1C96",
      "prep_kosis_annual", ANNUAL, "지역소득_1인당", items=["1인당 지역내총생산"]),
    V("V017", "1인당 가계총처분가능소득", "Y", "시도", "KOSIS", "101/DT_1C96",
      "prep_kosis_annual", ANNUAL, "지역소득_1인당", items=["1인당 가계총처분가능소득"]),
    # ------------------------------------------------------------------ 공급·재고
    V("V021", "아파트 인허가 호수", "M", "시도", "KOSIS", "116/DT_MLTM_1948",
      "prep_kosis_housing_supply", HOUSING, "주택건설실적_아파트", items=["인허가"],
      note="원자료 연초누계 -> 월 차분"),
    V("V022", "아파트 착공 호수", "M", "시도", "KOSIS", "116/DT_MLTM_5387",
      "prep_kosis_housing_supply", HOUSING, "주택건설실적_아파트", items=["착공"]),
    V("V023", "아파트 준공 호수", "M", "시도", "KOSIS", "116/DT_MLTM_5373",
      "prep_kosis_housing_supply", HOUSING, "주택건설실적_아파트", items=["준공"]),
    V("V024", "서울 구 주거용 신축허가", "Y", "서울구", "KOSIS(서울)", "201/DT_201004_O090005",
      "prep_kosis_annual", ANNUAL, "서울구_주거용신축허가"),
    # ------------------------------------------------------------------ 노경수 사전
    V("V026", "아파트 전세가격지수", "M", "시도+서울구", "R-ONE", "A_2024_00050",
      "prep_rone_price_index", PRICE, "전세가격지수_아파트"),
    V("V031", "아파트 매매수급동향지수", "M", "시도", "R-ONE", "A_2024_00076",
      "prep_rone_supply_demand", SUPPLY_DEMAND, "매매수급동향_아파트"),
    V("V032", "아파트 전세수급동향지수", "M", "시도", "R-ONE", "A_2024_00077",
      "prep_rone_supply_demand", SUPPLY_DEMAND, "전세수급동향_아파트"),
    V("V033", "아파트 월세수급동향지수", "M", "시도", "R-ONE", "A_2024_00078", "", "", "",
      status="중복", same_as="V005"),
    V("V034", "구(舊) 수급동향 통합지수", "M", "시도", "R-ONE", "T242093131851096",
      "prep_rone_supply_old", P + "부동산원_수급동향구_통합.csv", "수급동향(구)_통합",
      note="2010.06~2015.06, 8개 지역. 현행 수급동향과 이어붙이기 금지"),
    V("V035", "행정구역별 아파트 거래현황(동·호수, 면적)", "M", "시도+서울구", "R-ONE",
      "A_2024_00549", "prep_rone_transaction", TRANSACTION, "행정구역별_아파트거래현황"),
    V("V036", "행정구역별 아파트 매매거래현황(동·호수, 면적)", "M", "시도+서울구", "R-ONE",
      "A_2024_00554", "prep_rone_transaction", TRANSACTION, "행정구역별_아파트매매거래현황"),
    V("V037", "주택시장 소비심리지수", "M", "시도", "R-ONE", "T232543129897499",
      "prep_rone_sentiment", P + "부동산원_주택시장소비심리지수.csv", "주택시장_소비심리지수"),
    V("V038", "미분양주택현황", "M", "시도+서울구", "R-ONE", "T237973129847263",
      "prep_rone_unsold", P + "부동산원_미분양주택현황.csv", "미분양주택현황"),
    V("V043", "한국은행 기준금리", "M", "전국", "ECOS", "722Y001/0101000",
      "prep_ecos_macro", ECOS_M, "한국은행_기준금리"),
    V("V044", "CD(91일) 시장금리", "M", "전국", "ECOS", "721Y001/2010000",
      "prep_ecos_macro", ECOS_M, "시장금리_CD91일"),
    V("V045", "국고채 3년 시장금리", "M", "전국", "ECOS", "721Y001/5020000",
      "prep_ecos_macro", ECOS_M, "시장금리_국고채3년"),
    V("V046", "M2 통화량(평잔, 원계열)", "M", "전국", "ECOS", "161Y006/BBHA00",
      "prep_ecos_macro", ECOS_M, "M2_통화량"),
    V("V047", "가계신용 총액", "Q", "전국", "ECOS", "151Y001/1000000",
      "prep_ecos_macro", ECOS_Q, "가계신용_총액"),
    V("V048", "소비자물가지수 총지수", "M", "전국", "ECOS", "901Y009/0",
      "prep_ecos_macro", ECOS_M, "소비자물가지수"),
    V("V049", "시간당 명목임금지수(비농전산업)", "Q", "전국", "ECOS", "901Y102/A10001",
      "prep_ecos_macro", ECOS_Q, "시간당명목임금지수"),
    V("V050", "지역별 주택관련대출(예금은행)", "M", "시도", "ECOS", "151Y003/11110A0",
      "prep_ecos_regional_loan", P + "ECOS_지역별_주택관련대출.csv", "지역별_주택관련대출_예금은행"),
    # ------------------------------------------------------------------ 박준우 사전
    V("V056", "은행대출금리_주택담보대출(신규취급액)", "M", "전국", "ECOS", "121Y006/BECBLA0302",
      "prep_ecos_macro", ECOS_M, "예금은행대출금리_주택담보"),
    V("V057", "은행대출금리_전세자금(신규취급액)", "M", "전국", "ECOS", "121Y006/BECBLA03041",
      "prep_ecos_macro", ECOS_M, "예금은행대출금리_전세자금"),
    V("V059", "비은행대출금리_주택담보대출", "M", "전국", "ECOS", "121Y007/BEDBMC41",
      "prep_ecos_macro", ECOS_M, "비은행대출금리_주택담보"),
    V("V060", "대출행태서베이(대출태도)_은행_가계(주택)", "Q", "전국", "ECOS", "514Y001/AA04",
      "prep_ecos_macro", ECOS_Q, "대출태도_국내은행_가계주택"),
    V("V061", "소비자동향조사_주택가격전망CSI_서울", "M", "전국", "ECOS", "511Y002/FMFB·F0001",
      "prep_ecos_macro", ECOS_M, "주택가격전망CSI_서울", note="권역값(서울)"),
    V("V062", "소비자동향조사_주택가격전망CSI_6대광역시", "M", "전국", "ECOS", "511Y002/FMFB·F0002",
      "prep_ecos_macro", ECOS_M, "주택가격전망CSI_6대광역시", note="권역값(6대 광역시)"),
    V("V063", "소비자동향조사_주택가격전망CSI_기타도시", "M", "전국", "ECOS", "511Y002/FMFB·F0003",
      "prep_ecos_macro", ECOS_M, "주택가격전망CSI_기타도시", note="권역값(기타 도시)"),
    V("V064", "KOSPI_종가", "M", "전국", "ECOS", "901Y014/1070000",
      "prep_ecos_macro", ECOS_M, "KOSPI_종가"),
    V("V066", "사업체노동력조사 - 산업/규모별 임금 및 근로시간", "M", "전국", "ECOS",
      "901Y052·901Y148/A·I72BI·1 + KOSIS 118/DT_118N_MON041", "prep_ecos_macro", ECOS_M, "임금총액_사업체노동력",
      note="전체산업·전규모(1인이상) 전체임금총액. 2011~2019 KOSIS(산업분류 9차) + 2020~ ECOS(10·11차)를 이어 붙임"),
    V("V067", "선행지수 순환변동치", "M", "전국", "ECOS", "901Y067/I16E",
      "prep_ecos_macro", ECOS_M, "선행지수순환변동치"),
    V("V069", "예금취급기관 가계대출(용도별) 전체대출/주택관련대출", "M", "전국", "ECOS",
      "151Y005/1110000·11100A0", "prep_ecos_macro", ECOS_M,
      ["예금취급기관_가계대출", "예금취급기관_주택관련대출"]),
    V("V070", "수출입 동향 - 수출", "M", "전국", "ECOS", "901Y118/T002",
      "prep_ecos_macro", ECOS_M, "수출금액", note="사전 원출처 e-나라지표. 같은 통관 수출을 ECOS로"),
    V("V071", "주택금융공사 및 주택도시기금의 정책대출", "M", "전국", "ECOS", "151Y005/1120093",
      "prep_ecos_macro", ECOS_M, "정책대출_주금공_주택도시기금"),
    V("V072", "서울시 및 구별 가구 및 주택 수", "Y", "시도", "KOSIS", "116/DT_MLTM_2100",
      "prep_kosis_annual", ANNUAL, "주택보급률_국토부",
      items=["보급률(다가구 구분거처 반영)"],
      note="사전 원출처(인구주택총조사 가구·주택 수)를 국토부 공식 주택보급률로 대체. 시도만 공표(서울 구 시계열 없음). "
           "등록센서스 가구 기준, 다가구 구분거처를 주택수에 반영"),
    V("V074", "가구당 월평균 가계수지 (도시, 2인이상) - 소득", "Q", "전국", "KOSIS",
      "101/DT_1L9I002", "prep_kosis_household_survey", P + "가계동향_소득_분기.csv",
      "가계동향_도시2인이상_소득", note="1990Q1~2019Q4. 사전 수록기간대로 구계열만"),
    V("V076", "임대주택공급현황", "Y", "시도", "KOSIS", "116/DT_MLTM_6827·DT_MLTM_7174",
      "prep_kosis_annual", ANNUAL, "임대주택공급현황",
      note="2020~ 민간(6827), 2024~ 공공+민간(7174). 표마다 범위가 달라 항목을 나눠 둠"),
    V("V077", "임대주택건설공급현황", "Y", "시도", "KOSIS", "116/DT_MLTM_5560",
      "prep_kosis_annual", ANNUAL, "임대주택건설공급현황", note="2012~2019, 총계"),
]

# ============================================================================
# 공표 시차 (공식 근거로 검증, 2026-09-30)
#   lag  = 기준기간 마지막 달로부터 몇 개월 뒤 달에 공표되는가 (그 달 말에는 알 수 있음)
#   day  = 그 달의 공표일(공식 자료에 일자가 있을 때만). None 이면 '월'까지만 공식 확인
#   근거는 모두 공식 자료다: KOSIS 통계설명서(공표시기 항목), 한국은행 ECOS 통계메타DB
#   공표일정·한국은행 통계공표일정, 작성기관 공식 페이지(국토연구원 발간물 목록, 국토부
#   통계누리 메타, 관세청), KOSIS DB 수록기간·최종수정일, 법령. 언론 기사는 쓰지 않는다.
#   verdict: 일치(사전과 같음) / 보완(사전에 시기 없음) / 수정(사전과 다름) / 종료통계
#   known_delay: 공표는 됐으나 우리가 받는 수집원(R-ONE·ECOS) 게시가 늦음
# ============================================================================
_KOSIS_EXPL = "https://kosis.kr/statHtml/statHtml.do?orgId={org}&tblId={tbl}"
_BOK_META = "ECOS 통계메타DB '통화금융통계' 공표일정 (ecos.bok.or.kr, StatisticMeta API)"
_BOK_CAL = "https://www.bok.or.kr/portal/stats/statsPublictSchdul/listKnd.do?menuNo=200776"


def _kx(org, tbl, text):
    return dict(evidence=f"KOSIS 통계설명서 공표시기: '{text}'", source=_KOSIS_EXPL.format(org=org, tbl=tbl))


RELEASE = {
    "V047": dict(lag=2, dict="분기 종료 후 약 1.5개월", verdict="일치",
                 evidence="한국은행 통계메타 '가계신용: 해당분기 종료후 60일이내(잠정치)'. 2026년 공표일정 2.20·5.19·8.19·11.18",
                 source=_BOK_META + " / " + _BOK_CAL),
    "V049": dict(lag=3, dict="고정 공표일 미특정", verdict="보완",
                 evidence="KOSIS 통계설명서(노동생산성지수) 공표주기 '분기', 공표시기 '작성기준 분기 분기'. "
                          "KOSIS DB 수록 최신 2026 2/4 (2026-09-30 조회) -> 분기 종료 후 다음 분기 중 공표",
                 source=_KOSIS_EXPL.format(org="344", tbl="DT_344N_1D8B_BB")),
    "V060": dict(lag=-2, dict="분기 첫 월", verdict="일치",
                 evidence="KOSIS 통계설명서(금융기관대출행태조사) '조사기준 분기 익월, 매 분기가 종료된 다음 달(1·4·7·10월)에 공표'. "
                          "공표월에 다음 분기 전망치가 나오므로 ECOS 기간표시(전망 분기) 기준 첫 달",
                 source=_KOSIS_EXPL.format(org="301", tbl="DT_514Y001")),
    "V074": dict(lag=2, ended=True, dict="익익월말", verdict="일치(월 기준)",
                 evidence="KOSIS 통계설명서에 공표시기 기재 없음 -> KOSIS DB 최종수정일: 2019Q1~Q4 = 2019-05-16, 08-16, 11-15, 2020-02-13",
                 source="KOSIS DT_1L9I002 LST_CHN_DE"),
    "V013H": dict(lag=2, dict="상반기 8월, 하반기 익년 2월", verdict="일치",
                  evidence="KOSIS 통계설명서(지역별고용조사) '상반기는 당해년도에, 하반기는 익년도에 공표'. "
                           "월은 KOSIS DB 최종수정일: 2024H1 08-02, 2024H2 2025-02-04, 2025H1 08-19, 2025H2 2026-02-12, 2026H1 08-10",
                  source=_KOSIS_EXPL.format(org="101", tbl="DT_1ES3A01S")),
    "V015": dict(lag=12, dict="공표시차가 길어 별도 관리", verdict="보완", **_kx("101", "DT_1C96", "작성기준 년도 익년 12월")),
    "V017": dict(lag=12, dict="공표시차가 길어 별도 관리", verdict="보완", **_kx("101", "DT_1C96", "작성기준 년도 익년 12월")),
    "V024": dict(lag=12, dict="실제 공표일 별도 확인", verdict="보완",
                 **_kx("201", "DT_201004_O090005", "작성기준 년도 익년 12월 (서울특별시기본통계)")),
    "V072": dict(lag=12, dict="익년 8월", verdict="수정",
                 **_kx("116", "DT_MLTM_2100", "작성기준 년도 익년 12월 (국토부 주택보급률. 사전 원출처 총조사에서 대체)")),
    "V076": dict(lag=12, dict="익년 10월", verdict="수정",
                 evidence="국토교통 통계누리 임대주택통계 메타: 공표주기 '매년', 공표시기 '익년 12월'",
                 source="https://stat.molit.go.kr/portal/cate/statMetaView.do?hRsId=37"),
    "V077": dict(lag=12, ended=True, dict="익년 10월", verdict="수정",
                 evidence="국토교통 통계누리 임대주택통계 메타: 공표주기 '매년', 공표시기 '익년 12월' (2012~2019 표도 같은 통계)",
                 source="https://stat.molit.go.kr/portal/cate/statMetaView.do?hRsId=37"),
}

_REB = dict(dict="기준월 익월 15일", verdict="일치(월 기준)",
            **_kx("408", "DT_30404_N0006", "조사기준 월 익월 (전국주택가격동향조사)"))
_MOLIT_SUPPLY = dict(dict="기준월 익월 말", verdict="일치(월 기준)",
                     **_kx("116", "DT_MLTM_1948", "작성기준 월 익월(잠정치), 작성기준 년 익년 9월(확정치) (주택건설실적통계)"))
_RATE = dict(dict="익월 말", verdict="일치",
             evidence="한국은행 통계메타 '금융기관 가중평균금리: 익월말'. 2026년 공표일정 8월분 9.30",
             source=_BOK_META + " / " + _BOK_CAL)
_HHLOAN = dict(dict="익익월", verdict="일치",
               evidence="한국은행 통계메타 '예금취급기관 가계대출: 익익월 14일이내'",
               source=_BOK_META)
_MARKET = dict(dict="즉시(일별 관측)", verdict="일치",
               evidence="시장 관측치(월말값·월평균)라 정의상 해당 월 말에 확정. 공표 시차 없음", source="-")

RELEASE_MONTHLY = {
    **{v: dict(lag=1, day=None, **_REB) for v in ("V001", "V002", "V003", "V005", "V026", "V031", "V032")},
    "V004": dict(lag=2, day=None, dict="익익월 15일 (가격지수보다 1개월 추가 시차)", verdict="일치",
                 evidence="KOSIS 통계설명서 '조사기준 월 익월'은 조사 전체 기준. 전월세전환율 표(408/DT_30404_N0010)의 "
                          "KOSIS 수록 최신은 2026.07로 가격동향 표(2026.08)보다 1개월 늦음 (2026-09-30 조회)",
                 source=_KOSIS_EXPL.format(org="408", tbl="DT_30404_N0010")),
    "V034": dict(lag=1, day=None, ended=True, dict="현재 신규 공표 없음 (2015.06 종료)", verdict="종료통계",
                 evidence="종료된 (구)월세가격동향조사. 현행 전국주택가격동향조사 시차 준용", source="-"),
    **{v: dict(lag=1, day=31, dict="기준월 익월 말일경", verdict="일치",
               **_kx("408", "DT_408_2006_S0064", "작성기준 월 익월 말일경 (부동산거래현황)")) for v in ("V035", "V036")},
    "V037": dict(lag=1, day=19, known_delay=True, dict="익월 중순", verdict="일치",
                 evidence="국토연구원 공식 발간물 목록: 2025.11~2026.08분 공표일 익월 15~19일 (8월분 2026-09-16). "
                          "단 수집원 R-ONE 게시는 더 늦음(9/29 기준 7월분까지)",
                 source="https://www.krihs.re.kr/menu.es?mid=a10109000000"),
    **{v: dict(lag=1, day=None, **_MOLIT_SUPPLY) for v in ("V021", "V022", "V023")},
    "V038": dict(lag=1, day=31, dict="기준월 익월 말", verdict="일치",
                 **_kx("116", "DT_MLTM_2082", "작성기준 월 익월 말 (미분양주택현황보고)")),
    "V006": dict(lag=1, day=31, kind="수시", dict="API 일 1회 갱신, 계약 후 30일 이내 신고", verdict="보완",
                 evidence="부동산 거래신고 등에 관한 법률 제6조의2: 주택 임대차 계약 체결일부터 30일 이내 신고 -> "
                          "기준월 자료는 익월 말에 대부분 채워짐",
                 source="https://www.law.go.kr/법령/부동산거래신고등에관한법률"),
    "V007": dict(lag=1, day=31, kind="수시", dict="API 수시 갱신, 신고·정정 시차", verdict="보완",
                 evidence="같은 법 제3조: 매매 계약 체결일부터 30일 이내 신고. 해제는 계약 후 최대 11개월까지 추가(실측, 5개월 내 98.9%)",
                 source="https://www.law.go.kr/법령/부동산거래신고등에관한법률"),
    **{v: dict(lag=1, day=1, dict="기준월 익월 1일 전후", verdict="일치",
               **_kx("101", "DT_1B040A3", "작성기준 월 익월 1일 (주민등록인구현황)")) for v in ("V009", "V010", "V011")},
    "V012": dict(lag=1, day=None, dict="통상 익월 말 전후", verdict="일치(월 기준)",
                 **_kx("101", "DT_1B26001_A01", "작성기준 월 익월 (국내인구이동통계)")),
    "V013": dict(lag=1, day=None, dict="시도 월별: 익월 중순", verdict="일치(월 기준)",
                 **_kx("101", "DT_1DA7004S", "조사기준 월 익월 (경제활동인구조사)")),
    "V043": dict(lag=0, day=31, **_MARKET), "V044": dict(lag=0, day=31, **_MARKET),
    "V045": dict(lag=0, day=31, **_MARKET), "V064": dict(lag=0, day=31, **_MARKET),
    "V046": dict(lag=2, day=15, dict="익익월 중순", verdict="일치",
                 evidence="한국은행 통계메타 '통화 및 유동성: 익익월 14일이내'. 2026년 공표일정 7월분 9.15",
                 source=_BOK_META + " / " + _BOK_CAL),
    "V048": dict(lag=1, day=None, dict="익월 초 (첫 주)", verdict="일치(월 기준)",
                 **_kx("101", "DT_1J22003", "조사기준 월 익월 (소비자물가조사)")),
    "V050": dict(lag=2, day=None, dict="기준월 후 약 2개월", verdict="일치",
                 evidence="한국은행 통계메타 '지역별여수신: 익익월 중순경'", source=_BOK_META),
    **{v: dict(lag=1, day=31, **_RATE) for v in ("V056", "V057", "V059")},
    **{v: dict(lag=0, day=28, dict="월말", verdict="일치",
               evidence="한국은행 통계메타(소비자동향조사) 공표일정 '매월'. 2026년 공표일정 당월 22~28일(9월분 9.23)",
               source=_BOK_CAL) for v in ("V061", "V062", "V063")},
    "V066": dict(lag=2, day=None, dict="3개월 후", verdict="수정",
                 evidence="KOSIS 통계설명서(사업체노동력조사) '조사기준 월 익월'. 임금은 조사기준월의 전월분이라 "
                          "임금 기준월로는 익익월: KOSIS 임금표(118/DT_118N_MON054) 수록 최신 2026.06 (2026-09-30 조회)",
                 source=_KOSIS_EXPL.format(org="118", tbl="DT_118N_MON051")),
    "V067": dict(lag=1, day=None, dict="익월 말", verdict="일치(월 기준)",
                 **_kx("101", "DT_1C8015", "작성기준 월 익월 (경기종합지수)")),
    "V069": dict(lag=2, day=14, **_HHLOAN),
    "V070": dict(lag=1, day=15, known_delay=True, dict="익월 초", verdict="일치(월 기준)",
                 evidence="관세청 공식 안내: 확정치는 매월 15일에 전월 실적 공표(잠정치는 익월 1일). "
                          "우리가 받는 ECOS 통관 확정치는 게시가 더 늦음(9/29 기준 7월분까지)",
                 source="https://www.customs.go.kr/kcs/na/ntt/selectNttInfo.do?mi=10600&bbsId=2280&nttSn=10141577"),
    "V071": dict(lag=2, day=14, **_HHLOAN),
}

# 데이터사전 Variable_Master 끝의 '추가검토' 행. 구체적인 통계가 정해지지 않아 수집하지 않음.
PENDING = [
    "세금 관련 데이터 (보유세 등)",
    "월세 비중 (전세 대비? 거래 대비?)  -> V006 건수로 산출 가능",
    "정책 변화 더미변수",
    "거시건전성 정책 지수",
]

BY_ID = {v["id"]: v for v in VARIABLES}

DICT_XLSX = "docs/260928_데이터사전_취합_수정.xlsx"


def drivers():
    """데이터사전 Variable_Master 의 1차·2차 동인 -> {Master ID: (1차, 2차)}. V013H 는 V013 것을 쓴다.
    병합 파일에는 1차 동인만 표시한다."""
    import os
    import pandas as pd
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DICT_XLSX)
    vm = pd.read_excel(path, sheet_name="Variable_Master")
    vm = vm[vm["Master ID"].notna()]
    out = {r["Master ID"]: (r["1차 동인"] if pd.notna(r["1차 동인"]) else "",
                            r["2차 동인"] if pd.notna(r["2차 동인"]) else "") for _, r in vm.iterrows()}
    out["V013H"] = out.get("V013", ("", ""))
    return out
