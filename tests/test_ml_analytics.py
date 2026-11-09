import copy
import hashlib
import json

import pytest

from app import ml_analytics


def login(client):
    client.get("/login/")
    client.headers["x-csrf-token"] = client.cookies["sg_csrf"]
    credentials = {
        "username": "analytics_demo",
        "email": "analytics@example.test",
        "password": "Analytics123!",
    }
    assert client.post("/api/v1/auth/register/", json=credentials).status_code == 201
    assert client.post("/api/v1/auth/login/", json=credentials).status_code == 200


def test_reports_match_saved_metrics_and_keep_partitions_separate():
    report = ml_analytics.load_analytics()
    sms = json.loads(ml_analytics.REPORT_FILES["sms_test"].read_text())
    url = json.loads(ml_analytics.REPORT_FILES["url"].read_text())
    assert report["sms"]["test"]["f1"] == sms["metrics"]["f1"]
    assert report["url"]["test"]["confusion_matrix"] == url["test"]["confusion_matrix"]
    assert report["sms"]["test"] != report["sms"]["validation"]
    assert len(report["sms"]["comparison"]) == 3
    assert len(report["url"]["comparison"]) == 4
    assert report["sms"]["dataset"]["splits"]["train"]["rows"] == 3594
    assert report["url"]["dataset"]["splits"]["test"]["classes"] == {
        "0": 4293,
        "1": 2226,
    }
    assert report["url"]["thresholds"] == url["thresholds"]
    assert "inputs" not in json.dumps(report)
    assert "text_raw" not in json.dumps(report)


def test_analytics_requires_login_and_does_not_change_dashboard(client):
    assert client.get("/ml-analytics/", follow_redirects=False).status_code == 303
    assert client.get("/api/v1/ml/analytics/").status_code == 401
    assert client.get("/api/v1/ml/figures/sms/").status_code == 401
    login(client)
    before = client.get("/api/v1/stats/summary/").json()
    page = client.get("/ml-analytics/")
    assert page.status_code == 200
    for text in ("Dataset Overview", "Model Information", "0.936170", "0.642469"):
        assert text in page.text
    assert "не достигла целевых" in page.text and "Казахский" in page.text
    assert "ml_analytics.js" in page.text and "sms-matrix" in page.text
    assert "Только validation" in page.text or "только validation" in page.text
    assert client.get("/api/v1/ml/analytics/").headers["cache-control"] == "no-store"
    assert client.post("/api/v1/ml/analytics/", json={}).status_code == 405
    assert client.get("/api/v1/stats/summary/").json() == before
    dashboard = client.get("/").text
    assert 'id="categories-chart"' in dashboard and 'id="period"' in dashboard
    assert 'href="/ml-analytics/"' in dashboard


def test_fixed_figure_allowlist_and_no_path_parameters(client):
    login(client)
    image = client.get("/api/v1/ml/figures/sms/")
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert image.content.startswith(b"\x89PNG")
    assert client.get("/api/v1/ml/figures/unknown/").status_code == 404
    result = client.get("/api/v1/ml/analytics/?path=.env&file=../../.env")
    assert result.status_code == 200
    assert "secret_key" not in result.text.lower()


def test_missing_report_is_explicit_503_not_fabricated_zeroes(client, monkeypatch, tmp_path):
    login(client)
    monkeypatch.setitem(ml_analytics.REPORT_FILES, "sms_test", tmp_path / "absent.json")
    api = client.get("/api/v1/ml/analytics/")
    assert api.status_code == 503
    assert api.json()["error"]["code"] == "ML_REPORT_UNAVAILABLE"
    page = client.get("/ml-analytics/")
    assert page.status_code == 503 and "не подменяются" in page.text
    assert str(tmp_path) not in api.text


@pytest.mark.parametrize("corruption", ["nan", "matrix", "mismatch", "threshold"])
def test_corrupt_or_mixed_experiments_are_rejected(monkeypatch, corruption):
    sources = {key: ml_analytics._read(key) for key in ml_analytics.REPORT_FILES}
    sources = copy.deepcopy(sources)
    if corruption == "nan":
        sources["sms_selection"]["comparison"][0]["validation"]["f1"] = float("nan")
    elif corruption == "matrix":
        sources["sms_selection"]["comparison"][0]["validation"]["confusion_matrix"] = [[1]]
    elif corruption == "mismatch":
        sources["sms_meta"]["model_sha256"] = "different-artifact"
    else:
        sources["url"]["thresholds"]["t1"] = 1
        sources["url_meta"]["thresholds"]["t1"] = 1
    monkeypatch.setattr(ml_analytics, "_read", lambda key: sources[key])
    with pytest.raises(ml_analytics.ReportUnavailable):
        ml_analytics.load_analytics()


def test_page_escapes_saved_report_strings(client, monkeypatch):
    login(client)
    original = ml_analytics._read

    def altered(key):
        result = original(key)
        if key == "sms_meta":
            result["source"]["source_page"] = "<script>alert(1)</script>"
        return result

    monkeypatch.setattr(ml_analytics, "_read", altered)
    response = client.get("/ml-analytics/")
    assert response.status_code == 200
    assert "<script>alert(1)</script>" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text


def test_reports_are_read_only_and_never_invoke_models(client, monkeypatch):
    from ml.inference import ModelStore

    login(client)
    before = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in ml_analytics.REPORT_FILES.items()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("Analytics must not train or predict")

    monkeypatch.setattr(ModelStore, "predict_sms", forbidden)
    monkeypatch.setattr(ModelStore, "predict_url", forbidden)
    monkeypatch.setattr("ml.sms_experiment.fit_candidates", forbidden)
    assert client.get("/api/v1/ml/analytics/").status_code == 200
    assert client.get("/ml-analytics/").status_code == 200
    assert before == {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in ml_analytics.REPORT_FILES.items()
    }
