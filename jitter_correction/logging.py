from loguru import logger
import sys

def enable(level="INFO"):
    logger.add(
        sys.stderr,
        level=level,
        format="<lvl>{level:<8}</lvl> ({name}:{line}): <lvl>{message}</lvl>"
    )

def disable():
    logger.remove()

disable()
