import os
from typing import List

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from konan_search import KonanSearch
from pydantic import BaseModel
from similarity_module import SimilarityModule
from util import create_content

app = FastAPI()
sm = SimilarityModule(device_num=os.getenv("DEVICE_NUM"), model_path=os.getenv("MODEL_PATH"))
ks = KonanSearch(jar_path="konansearch.4.2.jar")


class Query(BaseModel):
    title: str = ""
    author: str = ""
    publisher: str = ""
    publisher_year: str = ""
    isbn: str = ""


class Candidate(BaseModel):
    title: str = ""
    author: str = ""
    publisher: str = ""
    publisher_year: str = ""
    isbn: str = ""
    rec_key: str = ""


class QueryAndCandidates(BaseModel):
    query: Query
    candidates: List[Candidate]


@app.get("/")
def check():
    content = {
        "status_code": status.HTTP_200_OK,
        "message": "API running",
    }
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


@app.post("/search")
def search(qac: QueryAndCandidates):
    qac = qac.dict()
    query = qac["query"]
    candidates = qac["candidates"]

    if not query["title"]:  # 쿼리 중 타이틀 필드가 없는 경우
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=create_content(
                query=query,
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                message="Search fail (Field title empty)",
            ),
        )

    try:  # 정상 작동
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=create_content(
                query=query,
                status_code=status.HTTP_200_OK,
                message="Search success",
                data=sm.compute_with_model(query, candidates, sim_threshold=float(os.getenv("SIM_THRESHOLD"))),
            ),
        )

    except:  # 유사도 계산 중 에러
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_content(
                query=query,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Search fail (Error occurred during compute similarity)",
            ),
        )


@app.post("/search-with-engine")
def search_with_engine(query: Query):
    query = query.dict()

    if not query["title"]:  # 쿼리 중 타이틀 필드가 없는 경우
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=create_content(
                query=query,
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                message="Search fail (Field title empty)",
            ),
        )

    candidates = ks.search_with_engine(query, max_record=200)

    if candidates is None:  # Konansearch 에러
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_content(
                query=query,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Search fail (Error occurred during Konansearch)",
            ),
        )

    elif len(candidates) == 0:  # Konansearch 결과가 없는 경우
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=create_content(
                query=query,
                status_code=status.HTTP_200_OK,
                message="Search fail (Konansearch result empty)",
            ),
        )

    elif len(candidates) > 99:  # Konansearch 결과가 100개 이상인 경우
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=create_content(
                query=query,
                status_code=status.HTTP_200_OK,
                message="Search fail (Konansearch result over 100)",
            ),
        )

    try:  # 정상 작동
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=create_content(
                query=query,
                status_code=status.HTTP_200_OK,
                message="Search success",
                data=sm.compute_with_model(query, candidates, sim_threshold=float(os.getenv("SIM_THRESHOLD"))),
            ),
        )

    except:  # 유사도 계산 중 에러
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=create_content(
                query=query,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Search fail (Error occurred during compute similarity)",
            ),
        )
