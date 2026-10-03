# -*- coding: utf-8 -*-
"""
국토부 실거래 수집을 하루 단위로 차례대로 돌린다 (전월세와 매매를 동시에 돌리지 않음).

  매일: 전월세를 일일 한도까지 받고 -> 이어서 매매를 일일 한도까지 받고 -> 다음 날 0시 10분까지 대기
  두 유형 모두 끝까지 받으면 종료한다.

사용법: python collect/run_molit_daily.py [--delay 0.5]
"""
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
KINDS = ["apt_rent", "apt_trade"]


def wait_next_day():
    now = datetime.now()
    resume = (now + timedelta(days=1)).replace(hour=0, minute=10, second=0, microsecond=0)
    print(f"[{now:%m-%d %H:%M}] 오늘 분량 끝. {resume:%m-%d %H:%M}까지 대기", flush=True)
    # 한 번에 길게 자면 절전·타이머 지연으로 시각을 놓칠 수 있어 1분마다 실제 시각을 확인한다
    while datetime.now() < resume:
        time.sleep(60)


def main():
    extra = sys.argv[1:]
    pending = list(KINDS)
    while pending:
        for kind in list(pending):
            print(f"\n[{datetime.now():%m-%d %H:%M}] {kind} 시작", flush=True)
            # --wait-quota 없이 실행: 한도에 걸리면 저장 후 종료(코드 1), 다 받으면 코드 0
            rc = subprocess.call([sys.executable, "-u", os.path.join(HERE, "collect_molit.py"), kind] + extra, cwd=BASE)
            if rc == 0:
                print(f"[{datetime.now():%m-%d %H:%M}] {kind} 전부 받음", flush=True)
                pending.remove(kind)
            else:
                print(f"[{datetime.now():%m-%d %H:%M}] {kind} 오늘 한도 도달 또는 오류(코드 {rc})", flush=True)
        if pending:
            wait_next_day()
    print("모든 유형 수집 완료", flush=True)


if __name__ == "__main__":
    main()
