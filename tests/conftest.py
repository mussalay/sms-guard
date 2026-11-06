import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        _env_file=None,
        database_url="sqlite:///:memory:",
        log_dir=tmp_path / "logs",
        artifact_dir=tmp_path / "no-models",
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client
