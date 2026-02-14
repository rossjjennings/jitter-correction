from loguru import logger
import sys

def enable(level="INFO"):
    logger.add(
        sys.stderr,
        level=level,
        format="<lvl>{level:<8}</lvl> ({name}:{line}): {message}"
    )

def disable():
    logger.remove()

disable()
