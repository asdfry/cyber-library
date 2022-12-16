import sys
from util import check_envs

if not check_envs():
    sys.exit(1)

from fastapi import FastAPI
from router import router

app = FastAPI()
app.include_router(router)
