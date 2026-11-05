import logging
from logging.handlers import RotatingFileHandler

from app.core.config import BASE_DIR, Settings


def configure_logging(settings: Settings) -> logging.Logger:
    directory = settings.log_dir
    if not directory.is_absolute():
        directory = BASE_DIR / directory
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sms_guard")
    logger.setLevel(settings.log_level.upper())
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    # Five rotated backups, each at most 1 MiB, plus the active app.log.
    handler = RotatingFileHandler(
        directory / "app.log", maxBytes=1024 * 1024, backupCount=5, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    return logger
