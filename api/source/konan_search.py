import sys
import time
from pathlib import Path
from typing import Dict, List

import jpype
from logger_main import logger


class KonanSearch:
    def __init__(self, jar_path: str):
        if jar_path is None or not Path(jar_path).exists():
            logger.error(f"Invalid path (konansearch): {jar_path}")
            sys.exit(1)
        jpype.startJVM(jpype.getDefaultJVMPath(), f"-Djava.class.path={jar_path}", convertStrings=True)
        self.jpkg = jpype.JPackage("com.konantech.konansearch")

    def search_with_engine(self, query: Dict, max_record: int) -> List[Dict]:
        # 검색 쿼리 만들기
        where_clause = []
        for k, v in query.items():
            if v:
                where_clause.append(f"{k.upper()}='{v}'")
        where_clause = " allwordthruindex AND ".join(where_clause)
        where_clause += " allwordthruindex"

        for cnt in range(1, 11):  # 소켓 에러 방지용
            try:
                ks = self.jpkg.KSEARCH()
                hc = ks.CreateHandle()
                ks.SetOption(hc, ks.OPTION_SOCKET_ASYNC_REQUEST, 1)
                ks.Search(
                    hc,
                    "192.168.20.31:7577",  # server address (java.lang.String)
                    "abbrnew_utf8_con_all",  # scenario name (java.lang.String)
                    where_clause,  # where clause (java.lang.String)
                    "",  # sorting clause (java.lang.String)
                    "",  # highlight text (java.lang.String)
                    "",  # log info (java.lang.String)
                    0,  # start offset (int)
                    max_record,  # record count (int)
                    1,  # language (int)
                    4,  # charset (int)
                )

                list_data = []
                column_size = ks.GetResult_ColumnSize(hc)
                field_data = jpype.JString[column_size]
                for i in range(max_record):
                    for j in range(column_size):
                        if j > 7:
                            break
                        ks.GetResult_Row(hc, field_data, i)
                        if j == 0:
                            rec_key = field_data[j]
                        elif j == 1:
                            title = field_data[j]
                        elif j == 2:
                            author = field_data[j]
                        elif j == 3:
                            publisher = field_data[j]
                        elif j == 4:
                            publisher_year = field_data[j]
                        elif j == 7:
                            isbn = field_data[j]

                    if not rec_key:  # rec_key가 None인 경우
                        continue
                    elif rec_key in [data["rec_key"] for data in list_data]:  # 중복 데이터인 경우
                        continue

                    list_data.append(
                        {
                            "rec_key": rec_key,
                            "title": title,
                            "author": author,
                            "publisher": publisher,
                            "publisher_year": publisher_year,
                            "isbn": isbn,
                        }
                    )

                logger.info(
                    f"Qeury: {query}, "
                    f"Total result count: {ks.GetResult_TotalCount(hc):,}, "
                    f"Search time: {ks.GetResult_SearchTime(hc)} ms"
                )

                return list_data

            except:
                logger.warning(f"Qeury: {query}, Socket error occurred during Konansearch (retry: {cnt})")
                time.sleep(0.1 * cnt)
                continue
        
        return None
