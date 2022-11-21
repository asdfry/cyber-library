import re
import traceback
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
