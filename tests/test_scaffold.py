import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from sqlalchemy import text

from app.core.config import Settings


def test_health_reports_liveness_and_unavailable_models(client):
    response = client.get("/api/v1/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models": {"sms": False, "url": False}}


def test_home_and_local_bootstrap(client):
    response = client.get("/login/")
    assert response.status_code == 200
    assert 'lang="ru"' in response.text
    assert "Вход в SMS Guard" in response.text
    assert "viewport" in response.text
    css = client.get("/static/vendor/bootstrap.min.css")
    assert css.status_code == 200
    assert "Bootstrap v5.3.8" in " ".join(css.text.split())


def test_openapi_contains_protected_product_api(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/api/v1/check/sms/" in response.json()["paths"]
    assert client.post("/api/v1/check/sms/", json={"text": "test"}).status_code in {401, 403}


def test_sqlite_connection_and_foreign_keys(client):
    with client.app.state.engine.connect() as connection:
        assert connection.scalar(text("SELECT 1")) == 1
        assert connection.scalar(text("PRAGMA foreign_keys")) == 1


def test_logging_rotation_and_model_warning(client):
    handlers = logging.getLogger("sms_guard").handlers
    handler = next(item for item in handlers if isinstance(item, RotatingFileHandler))
    handler.flush()
    assert handler.maxBytes == 1024 * 1024
    assert handler.backupCount == 5
    content = Path(handler.baseFilename).read_text(encoding="utf-8")
    assert "WARNING" in content
    assert "анализ недоступен" in content


def test_settings_read_environment(monkeypatch):
    monkeypatch.setenv("SMS_GUARD_DEBUG", "false")
    monkeypatch.setenv("SMS_GUARD_APP_NAME", "Environment Guard")
    settings = Settings(_env_file=None)
    assert settings.debug is False
    assert settings.app_name == "Environment Guard"


def test_template_escapes_configured_name(tmp_path):
    from fastapi.testclient import TestClient

    from app.main import create_app

    settings = Settings(_env_file=None, app_name="<script>alert(1)</script>", log_dir=tmp_path)
    with TestClient(create_app(settings)) as test_client:
        html = test_client.get("/login/").text
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
