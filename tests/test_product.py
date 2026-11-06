import json

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import Settings
from app.core.security import hash_password, verify_password
from app.db.models import Check, User
from app.main import create_app
from ml.data.load_sms import ROOT
from ml.inference import ModelStore
from ml.risk import compute_risk, compute_url_risk
from ml.sms_experiment import choose_threshold, finalize, metrics
from ml.url_features import FEATURE_NAMES, extract_url_features, model_features, validate_url


@pytest.fixture
def product(tmp_path):
    with TestClient(
        create_app(
            Settings(_env_file=None, database_url="sqlite:///:memory:", log_dir=tmp_path / "logs")
        )
    ) as client:
        yield client


def account(client, username="alice"):
    client.get("/login/")
    client.headers["x-csrf-token"] = client.cookies["sg_csrf"]
    credentials = {
        "username": username,
        "email": username + "@example.test",
        "password": "Password123!",
    }
    assert client.post("/api/v1/auth/register/", json=credentials).status_code == 201
    assert client.post("/api/v1/auth/login/", json=credentials).status_code == 200
    return credentials


def test_passwords_are_salted_and_verified():
    one, two = hash_password("Password123"), hash_password("Password123")
    assert one != two and "Password123" not in one
    assert verify_password("Password123", one)
    assert not verify_password("wrong", one)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/a",
        "example.org/path",
        "http://192.0.2.1/x",
        "http://[2001:db8::1]/",
        "http://example.test:8080/a",
        "http://user@example.test/login",
        "http://xn--exmple-cua.test",
        "https://раypal.test",
        "http://bit.ly/x",
        "http://example.test/a//b",
        "https://example.test?q=1&x=2",
        "https://goggle.test/verify",
    ],
)
def test_url_features_offline_and_fixed_order(url, monkeypatch):
    import socket

    def blocked(*a, **k):
        raise AssertionError("No DNS allowed")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    features = extract_url_features(url)
    assert list(features) == FEATURE_NAMES and len(features) == 22
    assert len(model_features(url)) == 22


@pytest.mark.parametrize(
    "url",
    [
        "",
        "ftp://example.test",
        "javascript:alert(1)",
        "http:///",
        "http://a b.test",
        "http://example.test:99999",
        "x" * 2049,
    ],
)
def test_invalid_urls_fail_cleanly(url):
    with pytest.raises(ValueError):
        validate_url(url)


def test_reserved_domain_typo_flags_and_frozen_model_features():
    features = extract_url_features("http://goggle.test/login")
    assert features["brand_similarity"] == 1 and features["subdomains"] == 0
    # Model was evaluated with v1 PSL semantics; rule improvements do not silently alter it.
    assert model_features("http://goggle.test/login")[FEATURE_NAMES.index("brand_similarity")] == 0
    encoded = "раypal.test".encode("idna").decode("ascii")
    flags = extract_url_features("https://" + encoded)
    assert flags["punycode"] and flags["mixed_scripts"] and flags["brand_similarity"]


@pytest.mark.parametrize("score", [0, 1, 10, 20, 34, 35, 50, 69, 70, 85, 99, 100])
def test_risk_boundaries(score, tmp_path):
    config = yaml.safe_load((ROOT / "risk_config.yaml").read_text())
    config["weights"] = {"sms": 1, "url": 0, "se": 0}
    config["levels"] = {"medium": 35, "high": 70}
    path = tmp_path / "risk.yaml"
    path.write_text(yaml.safe_dump(config))
    result = compute_risk(score / 100, None, [], [], config_path=path)
    assert result["risk_score"] == score
    assert result["risk_level"] == ("HIGH" if score >= 70 else "MEDIUM" if score >= 35 else "LOW")
    assert result["explanation"] and result["recommendation"]


def test_risk_overrides_and_missing_sms():
    assert compute_risk(None, None, ["CODE_REQUEST"], [])["risk_score"] >= 70
    assert compute_risk(0.9, 0.95, [], [], t2=0.8)["risk_score"] >= 85
    assert compute_risk(None, 0.1, [], [{"ip_host": 1}])["risk_score"] >= 60
    assert compute_risk(None, None, [], [])["risk_score"] == 0


def test_risk_reasons_ranked_by_weighted_contribution():
    result = compute_risk(0.51, 0.99, [], [])
    assert result["explanation"][0].startswith("URL-модель")


@pytest.mark.parametrize(
    ("probability", "category"),
    [(0.05, "SAFE"), (0.2, "SUSPICIOUS"), (0.3, "DANGEROUS"), (0.99, "DANGEROUS")],
)
def test_standalone_url_score_matches_classifier_bands(probability, category):
    result = compute_url_risk(probability, {}, {"t1": 0.15, "t2": 0.28})
    assert result["category"] == category
    assert result["risk_score"] >= 70 if category == "DANGEROUS" else result["risk_score"] < 70


def test_metrics_and_test_guard(tmp_path):
    threshold, scores = choose_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert scores["f2"] == 1 and 0.2 < threshold <= 0.8
    assert metrics([0, 1], [0.1, 0.9], 0.5)["confusion_matrix"] == [[1, 0], [0, 1]]
    # Real finalization must refuse a second evaluation before loading heldout texts.
    if (ROOT / "docs/reports/07_test_claim.json").exists():
        with pytest.raises(FileExistsError):
            finalize()


def test_model_reload_and_russian_rejection():
    if not (ROOT / "ml/artifacts/url_model.joblib").exists():
        pytest.skip("Local trained artifacts not available")
    one, two = ModelStore(ROOT / "ml/artifacts"), ModelStore(ROOT / "ml/artifacts")
    assert one.predict_sms("See you at the library tomorrow") == two.predict_sms(
        "See you at the library tomorrow"
    )
    assert one.predict_url("https://example.com/about") == pytest.approx(
        two.predict_url("https://example.com/about"), abs=1e-12, rel=0
    )
    with pytest.raises(ValueError):
        one.predict_sms("Назовите код из SMS", language="ru")


def test_auth_csrf_logout_and_owner_protection(product):
    assert product.get("/history/", follow_redirects=False).status_code == 303
    assert product.post("/api/v1/auth/register/", json={}).status_code == 403
    account(product)
    result = product.post("/api/v1/check/sms/", json={"text": "Hello, see you tomorrow"}).json()
    check_id = result["id"]
    assert product.post("/api/v1/auth/logout/", json={}).status_code == 204
    assert product.get(f"/api/v1/checks/{check_id}/").status_code == 401
    account(product, "bob")
    assert product.get(f"/api/v1/checks/{check_id}/").status_code == 404
    assert product.delete(f"/api/v1/checks/{check_id}/").status_code == 404
    assert product.get("/api/v1/checks/").json()["total"] == 0


@pytest.mark.parametrize(
    "payload", [{"text": ""}, {"text": " "}, {"text": "x" * 1001}, {"text": 123}, [], "string"]
)
def test_api_invalid_inputs_no_500(product, payload):
    account(product)
    assert product.post("/api/v1/check/sms/", json=payload).status_code == 400


def test_privacy_idempotency_russian_and_profile(product):
    account(product)
    text = "Срочно назовите код из SMS. Переведите 5000 руб на карту 1234567890123456"
    first = product.post(
        "/api/v1/check/sms/", json={"text": text}, headers={"Idempotency-Key": "one"}
    )
    assert first.status_code == 200, first.text
    result = first.json()
    from datetime import datetime

    datetime.fromisoformat(result["created_at"].replace("Z", "+00:00"))
    assert result["p_sms"] is None and result["risk_score"] >= 70
    assert "1234567890123456" not in result["input"]
    second = product.post(
        "/api/v1/check/sms/", json={"text": text}, headers={"Idempotency-Key": "one"}
    )
    assert result["id"] == second.json()["id"]
    with product.app.state.session_factory() as db:
        stored = db.scalar(select(Check))
        assert stored.input_text is None
    assert product.post("/api/v1/profile/", json={"store_sms_text": True}).status_code == 200
    third = product.post("/api/v1/check/sms/", json={"text": text}).json()
    assert third["input"] == text
    assert product.get("/api/v1/checks/?type=URL").json()["total"] == 0
    assert product.get("/api/v1/checks/?date_from=bad").status_code == 400
    assert product.get("/api/v1/stats/summary/").json()["total"] == 2
    assert product.delete("/api/v1/history/").status_code == 204
    assert product.get("/api/v1/checks/").json()["total"] == 0


def test_cookie_flags_and_login_pause(product):
    credentials = account(product)
    response = product.post("/api/v1/auth/login/", json=credentials)
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    for _ in range(5):
        assert (
            product.post(
                "/api/v1/auth/login/", json={"username": "alice", "password": "wrong"}
            ).status_code
            == 401
        )
    assert product.post("/api/v1/auth/login/", json=credentials).status_code == 429


def test_rate_limit_and_xss_isolation(product):
    account(product)
    for _ in range(30):
        response = product.post("/api/v1/check/sms/", json={"text": "<script>alert(1)</script>"})
        assert response.status_code == 200, response.text
    assert product.post("/api/v1/check/sms/", json={"text": "Hello"}).status_code == 429
    html = product.get("/history/").text
    assert "<script>alert(1)</script>" not in html
    source = (ROOT / "static/js/app.js").read_text()
    assert "innerHTML" not in source and "textContent" in source


def test_admin_aggregates_do_not_expose_other_texts(product):
    account(product)
    product.post("/api/v1/check/sms/", json={"text": "private sample"})
    with product.app.state.session_factory() as db:
        user = db.scalar(select(User).where(User.username == "alice"))
        user.is_staff = True
        db.commit()
    response = product.get("/api/v1/stats/summary/?scope=all").json()
    assert response["total"] == 1 and response["recent"] == []
    assert "private sample" not in json.dumps(response)


def test_missing_models_return_503(client):
    account(client)
    assert client.post("/api/v1/check/sms/", json={"text": "Hello"}).status_code == 503


def test_sql_injection_profile_password_and_session_revocation(product):
    account(product)
    assert (
        product.post(
            "/api/v1/auth/login/", json={"username": "' OR 1=1 --", "password": "anything"}
        ).status_code
        == 401
    )
    assert (
        product.post(
            "/api/v1/auth/register/",
            json={"username": "bad' OR 1=1", "email": "a@example.test", "password": "Password123"},
        ).status_code
        == 400
    )
    assert (
        product.post(
            "/api/v1/profile/", json={"password": "NewPassword123", "current_password": "wrong"}
        ).status_code
        == 400
    )
    assert (
        product.post(
            "/api/v1/profile/",
            json={"password": "NewPassword123", "current_password": "Password123!"},
        ).status_code
        == 200
    )
    assert product.get("/api/v1/profile/").status_code == 401
    assert (
        product.post(
            "/api/v1/auth/login/", json={"username": "alice", "password": "NewPassword123"}
        ).status_code
        == 200
    )


def test_sms_multiple_urls_and_standalone_url(product):
    account(product)
    response = product.post(
        "/api/v1/check/sms/",
        json={"text": "See http://a.test http://b.test http://c.test http://d.test"},
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["urls"]) == 3
    response = product.post("/api/v1/check/url/", json={"url": "https://www.example.com/about"})
    assert response.status_code == 200
    assert response.json()["p_sms"] is None
    assert response.json()["urls"][0]["domain"] == "www.example.com"
    assert (
        product.post("/api/v1/check/url/", json={"url": "javascript:alert(1)"}).status_code == 400
    )


def test_binary_input_https_cookie_and_pagination(product):
    credentials = account(product)
    assert product.post("/api/v1/check/sms/", content=b"\xff\x00").status_code == 400
    response = product.post("https://testserver/api/v1/auth/login/", json=credentials)
    assert "secure" in response.headers["set-cookie"].lower()
    # The Secure cookie is deliberately not returned over plain HTTP.
    assert product.get("http://testserver/api/v1/profile/").status_code == 401


def test_url_split_groups_are_disjoint():
    path = ROOT / "data/processed/url_split.json"
    if not path.exists():
        pytest.skip("URL preparation artifacts unavailable")
    manifest = json.loads(path.read_text())
    groups = manifest["groups"]
    train, val, test = [
        {groups[i] for i in manifest["positions"][name]} for name in ("train", "val", "test")
    ]
    assert not train & val and not train & test and not val & test


def test_migrations_create_schema_from_zero(tmp_path, monkeypatch):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    url = "sqlite:///" + (tmp_path / "fresh.db").as_posix()
    monkeypatch.setenv("SMS_GUARD_DATABASE_URL", url)
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    engine = create_engine(url)
    try:
        assert {
            "users",
            "user_profile",
            "auth_sessions",
            "checks",
            "url_results",
            "detected_features",
            "model_versions",
        } <= set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_no_consent_masks_sensitive_sms_url_parameters(product):
    account(product)
    response = product.post(
        "/api/v1/check/sms/",
        json={"text": "Check http://example.test/?phone=12345678901&card=1234567890123456"},
    )
    assert response.status_code == 200, response.text
    check_id = response.json()["id"]
    stored = product.get(f"/api/v1/checks/{check_id}/").json()
    assert "1234567890123456" not in json.dumps(stored)


def test_startup_refreshes_model_metadata_after_training_finalization(tmp_path):
    from app.db.models import ModelVersion

    settings = Settings(
        _env_file=None,
        database_url="sqlite:///" + (tmp_path / "persistent.db").as_posix(),
        log_dir=tmp_path / "logs",
    )
    with TestClient(create_app(settings)) as client:
        with client.app.state.session_factory() as session:
            version = session.scalar(select(ModelVersion).where(ModelVersion.model_type == "sms"))
            version.metrics_json = {}
            session.commit()
    with TestClient(create_app(settings)) as client:
        with client.app.state.session_factory() as session:
            version = session.scalar(select(ModelVersion).where(ModelVersion.model_type == "sms"))
            assert version.metrics_json["precision"] >= 0.9 and version.is_active
