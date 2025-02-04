from functools import wraps
from fastapi import HTTPException
from trainable_entity_extractor.config import config_logger


def catch_exceptions(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception:
            config_logger.error("Error see traceback", exc_info=1)
            raise HTTPException(status_code=422, detail="An error has occurred. Check graylog for more info")

    return wrapper
