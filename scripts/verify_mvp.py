"""Finite local acceptance checks and performance measurements, no model retraining."""

import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from app.db.models import Check  # noqa: E402
from app.main import create_app  # noqa: E402
from ml.data.load_sms import write_json  # noqa: E402


def main():
    with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
        work = Path(directory)
        with TestClient(
            create_app(
                Settings(
                    _env_file=None,
                    database_url="sqlite:///" + (work / "bench.db").as_posix(),
                    log_dir=work,
                )
            )
        ) as client:
            client.get("/login/")
            client.headers["x-csrf-token"] = client.cookies["sg_csrf"]
            credentials = {
                "username": "benchmark",
                "email": "benchmark@example.test",
                "password": "BenchGuard2026!",
            }
            assert client.post("/api/v1/auth/register/", json=credentials).status_code == 201
            assert client.post("/api/v1/auth/login/", json=credentials).status_code == 200
            samples = [
                "Hello, see you at the library tomorrow.",
                "Urgent! Verify account at http://one.test/login and http://two.test/update and http://three.test/verify",
                "Срочно назовите код из SMS для отмены перевода денег.",
            ]
            times = []
            for i in range(100):
                # Rate limiting is tested separately; time 100 analyses with harness state reset.
                client.app.state.check_limits.clear()
                start = time.perf_counter()
                response = client.post("/api/v1/check/sms/", json={"text": samples[i % 3]})
                times.append(1000 * (time.perf_counter() - start))
                assert response.status_code == 200, response.text
            result = response.json()
            with client.app.state.session_factory() as session:
                session.add_all(
                    [
                        Check(
                            user_id=1,
                            request_key=f"load-fixture-{i}",
                            check_type="SMS",
                            input_text=None,
                            input_masked="Synthetic load-test record",
                            language=result["language"],
                            category=result["category"],
                            risk_level=result["risk_level"],
                            risk_score=result["risk_score"],
                            p_sms=result["p_sms"],
                            explanation=result["explanation"],
                            recommendation=result["recommendation"],
                            sms_model_version=result["model_versions"]["sms"],
                            url_model_version=None,
                            payload=result,
                        )
                        for i in range(900)
                    ]
                )
                session.commit()
            start = time.perf_counter()
            stats = client.get("/api/v1/stats/summary/")
            elapsed = time.perf_counter() - start
            assert stats.status_code == 200 and stats.json()["total"] == 1000
            # Five required demonstrations, all using live models.
            scenarios = []
            for kind, value in [
                ("sms", samples[0]),
                ("sms", "Срочно переведите деньги на новый счёт"),
                ("sms", samples[1]),
                ("url", "https://example.com/about"),
                ("sms", samples[2]),
            ]:
                client.app.state.check_limits.clear()
                r = client.post(
                    f"/api/v1/check/{kind}/", json={"text" if kind == "sms" else "url": value}
                )
                assert r.status_code == 200
                body = r.json()
                scenarios.append(
                    {
                        "type": kind,
                        "category": body["category"],
                        "risk": body["risk_score"],
                        "p_sms": body["p_sms"],
                    }
                )
            report = {
                "analysis_requests": 100,
                "analysis_ms_p50": float(np.median(times)),
                "analysis_ms_p95": float(np.percentile(times, 95)),
                "analysis_ms_max": float(max(times)),
                "all_analysis_under_500ms": max(times) <= 500,
                "dashboard_1000_rows_api_seconds": elapsed,
                "dashboard_api_under_2s": elapsed <= 2,
                "scenarios": scenarios,
                "notes": (
                    "Temporary test DB; 900 explicitly synthetic load fixtures, "
                    "not fabricated model outcomes; browser 1000-row render not timed"
                ),
            }
        for handler in logging.getLogger("sms_guard").handlers[:]:
            handler.close()
            logging.getLogger("sms_guard").removeHandler(handler)
    write_json(ROOT / "docs/reports/mvp_acceptance.json", report)
    print(report)


if __name__ == "__main__":
    main()
