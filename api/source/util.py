import os
import re
import traceback
from pathlib import Path
from typing import Dict, List

from logger_main import logger


def create_content(query: Dict, status_code: int, message: str, data: List[Dict] = []) -> Dict:
    content = {"status_code": status_code, "message": message, "data": data}

    if status_code == 200:
        if "success" in message:
            logger.info(f"{message}, Query: {query}")
        elif "fail" in message:
            logger.warning(f"{message}, Query: {query}")

    elif status_code == 422:
        logger.error(f"{message}, Query: {query}")

    elif status_code == 500:
        traceback_content = re.sub(r"\s{2,}|\n", " ", traceback.format_exc()).strip()
        logger.error(f"{message}, Query: {query}, Exception: {traceback_content}")

    return content


def check_envs():
    env_var_num = {"DEVICE_NUM": os.getenv("DEVICE_NUM"), "SIM_THRESHOLD": os.getenv("SIM_THRESHOLD")}
    env_var_path = {"JAR_PATH": os.getenv("JAR_PATH"), "MODEL_PATH": os.getenv("MODEL_PATH")}

    for key, value in env_var_num.items():
        if not value:
            logger.error(f"Invalid environment variable ({key}): {value}")
            return False

    for key, value in env_var_path.items():
        if not value or not Path(value).exists():
            logger.error(f"Invalid environment variable ({key}): {value}")
            return False

    return True
