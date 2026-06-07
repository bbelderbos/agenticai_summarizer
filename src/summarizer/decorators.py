import logging
from functools import wraps
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def retry_on_validation_error(max_attempts: int):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except ValidationError:
                    if attempt == max_attempts:
                        logger.exception(
                            "Tool input failed validation after %d attempts",
                            max_attempts,
                        )
                        raise
                    logger.warning(
                        "Malformed tool input on attempt %d/%d, retrying",
                        attempt,
                        max_attempts,
                    )

        return wrapper

    return decorator
