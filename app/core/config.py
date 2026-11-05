from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SMS_GUARD_", env_file=BASE_DIR / ".env", env_file_encoding="utf-8"
    )

    app_name: str = "SMS Guard"
    debug: bool = False
    database_url: str = f"sqlite:///{(BASE_DIR / 'sms_guard.db').as_posix()}"
    log_dir: Path = BASE_DIR / "logs"
    log_level: str = "INFO"
    secret_key: str = ""
