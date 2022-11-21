import re
import time
from glob import glob
from datetime import datetime

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler


if __name__ == "__main__":

    scheduler = BackgroundScheduler()

    @scheduler.scheduled_job("cron", hour="*/6", minute="0", timezone="Asia/Seoul")
    def create_summary():
        print("===" * 5, str(datetime.now()).split(".")[0], "===" * 5)
        logs = sorted(glob("./logs/.*"))
        dict_logs = {}

        for idx, log in enumerate(logs):
            ymd = re.search(r"\d{8}", log).group()
            if ymd in dict_logs:
                dict_logs[ymd].append(idx)
            else:
                dict_logs[ymd] = [idx]

        dict_reqs = {}

        for key, value in dict_logs.items():
            print(f"Parsing log . . . ({key[:4]}/{key[4:6]}/{key[6:8]})")
            lines = []
            for idx_i in value:
                with open(logs[idx_i], "r") as f:
                    lines += f.readlines()

            dict_req = {}
            pattern_l = re.compile(r":\s{2,}Search (success|fail)")
            pattern_s = re.compile(r":\s{2,}Search success")  # 성공
            pattern_f_1 = re.compile(r":\s{2,}Search fail \(Field title empty\)")  # 쿼리 중 타이틀 필드가 없는 경우
            pattern_f_2 = re.compile(r":\s{2,}Search fail \(Konansearch result empty\)")  # Konansearch 결과가 없는 경우
            pattern_f_3 = re.compile(r":\s{2,}Search fail \(Konansearch result over 100\)")  # Konansearch 결과가 100개 이상인 경우
            pattern_f_4 = re.compile(r":\s{2,}Search fail \(Error occurred during Konansearch\)")  # Konansearch 에러 (소켓 에러)
            pattern_f_5 = re.compile(r":\s{2,}Search fail \(Error occurred during compute similarity\)")  # 유사도 계산 중 에러
            patterns_f = [pattern_f_1, pattern_f_2, pattern_f_3, pattern_f_4, pattern_f_5]

            for line in lines:
                splited_line = line.split("\t")
                dt = splited_line[0]  # datetime
                lc = splited_line[2]  # log content
                if pattern_l.search(lc):
                    if not dt in dict_req:
                        dict_req[dt] = {
                            "success": 0,
                            "fail_1": 0,
                            "fail_2": 0,
                            "fail_3": 0,
                            "fail_4": 0,
                            "fail_5": 0,
                            "total": 0,
                        }
                    if pattern_s.search(lc):  # success
                        dict_req[dt]["success"] += 1
                    else:  # fail
                        for idx_j, pattern_f in enumerate(patterns_f):
                            if pattern_f.search(lc):
                                dict_req[dt][f"fail_{idx_j+1}"] += 1
                                break
                    dict_req[dt]["total"] += 1

            dict_reqs[key] = dict_req

        print("Saving summary.csv . . .")

        list_summary = []

        for ymd, dict_req in dict_reqs.items():
            dict_cnt = {"success": 0, "fail_1": 0, "fail_2": 0, "fail_3": 0, "fail_4": 0, "fail_5": 0, "total": 0}
            if len(dict_req) > 0:
                for value in dict_req.values():
                    for status, cnt in value.items():
                        dict_cnt[status] += cnt
                list_summary.append(
                    {
                        "date": ymd,
                        "success": dict_cnt["success"],
                        "fail_1": dict_cnt["fail_1"],  # 쿼리 중 타이틀 필드가 없는 경우
                        "fail_2": dict_cnt["fail_2"],  # Konansearch 결과가 없는 경우
                        "fail_3": dict_cnt["fail_3"],  # Konansearch 결과가 100개 이상인 경우
                        "fail_4": dict_cnt["fail_4"],  # Konansearch 에러 (소켓 에러)
                        "fail_5": dict_cnt["fail_5"],  # 유사도 계산 중 에러
                        "total": dict_cnt["total"],
                        "requests per second": round(dict_cnt["total"] / len(dict_req), 2),
                    }
                )
        df = pd.DataFrame(list_summary)
        df.to_csv("api_summary.csv", index=False)

    scheduler.start()
    print("Start background scheduler . . .")

    while True:
        time.sleep(3)
