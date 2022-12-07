from typing import List

from pydantic import BaseModel


class Query(BaseModel):
    title: str = ""
    author: str = ""
    publisher: str = ""
    publisher_year: str = ""
    isbn: str = ""
    rec_key: str = ""


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
