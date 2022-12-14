import sys

from fastapi import FastAPI
from router import router
from util import check_envs

app = FastAPI()
app.include_router(router)

@app.on_event("startup")
def startup_event():
    if not check_envs():
        sys.exit(1)
