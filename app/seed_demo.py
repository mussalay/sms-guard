"""Local, explicitly labelled demonstration account; results are computed by live models."""

import os

from fastapi.testclient import TestClient

from app.main import create_app

SAMPLES = [
    "Hello! See you at the library tomorrow.",
    "Your order has been shipped and will arrive on Monday.",
    "URGENT! Your account is blocked. Verify at http://bank-check.test/login",
    "Срочно переведите деньги. Назовите секретный код из SMS.",
    "Встречаемся завтра вечером у библиотеки.",
    "Congratulations! You won a prize. Claim at https://prize-claim.example",
    "Никому не сообщайте коды из SMS, даже сотрудникам банка.",
    "Мама, срочно переведи деньги на новый счёт. Не звони.",
    "Ваш заказ готов к получению в пункте выдачи.",
    "Срочно оплатите доставку на http://delivery-pay.test/verify",
]


def seed_demo():
    password = os.environ.get("SMS_GUARD_DEMO_PASSWORD", "DemoGuard2026!")
    with TestClient(create_app()) as client:
        client.get("/login/")
        client.headers["x-csrf-token"] = client.cookies["sg_csrf"]
        client.post(
            "/api/v1/auth/register/",
            json={"username": "demo", "email": "demo@example.test", "password": password},
        )
        response = client.post(
            "/api/v1/auth/login/", json={"username": "demo", "password": password}
        )
        response.raise_for_status()
        total = client.get("/api/v1/checks/").json()["total"]
        for i in range(max(0, 30 - total)):
            kind = "url" if i % 5 == 0 else "sms"
            value = "https://www.example.com/about" if kind == "url" else SAMPLES[i % len(SAMPLES)]
            result = client.post(
                f"/api/v1/check/{kind}/",
                json={"url" if kind == "url" else "text": value},
                headers={"Idempotency-Key": f"demo-{i + total}"},
            )
            result.raise_for_status()
        print(
            "Демо-пользователь demo: 30 результатов настоящих моделей. "
            "Пароль — из SMS_GUARD_DEMO_PASSWORD или README."
        )


if __name__ == "__main__":
    seed_demo()
