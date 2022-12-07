import sys

from fastapi import FastAPI
from router import router
from util import check_envs

if not check_envs():
    sys.exit(0)

app = FastAPI()
app.include_router(router)
