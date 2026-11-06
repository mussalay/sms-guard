"""Real local browser checks and five required screen artifacts; no external navigation."""

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "docs/screenshots"
    folder.mkdir(exist_ok=True)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            headless=True,
        )
        context = browser.new_context(viewport={"width": 1366, "height": 900})
        context.route(
            "**/*",
            lambda route: (
                route.continue_()
                if route.request.url.startswith("http://127.0.0.1:8080/")
                else route.abort()
            ),
        )
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8080/login/")
        page.screenshot(path=str(folder / "01_login.png"), full_page=True)
        page.locator('[name="username"]').fill("demo")
        page.locator('[name="password"]').fill("DemoGuard2026!")
        page.locator("#auth-form button").click()
        page.wait_for_url("http://127.0.0.1:8080/")
        page.wait_for_selector("#counters .card")
        page.screenshot(path=str(folder / "02_dashboard.png"), full_page=True)
        widths = []
        for width in (360, 768, 1366):
            page.set_viewport_size({"width": width, "height": 900})
            page.wait_for_timeout(250)
            overflow = page.evaluate("document.documentElement.scrollWidth > innerWidth")
            widths.append({"width": width, "horizontal_overflow": overflow})
            assert not overflow, f"Horizontal overflow at {width}"
        page.goto("http://127.0.0.1:8080/check/sms/")
        page.locator("#check-input").fill("Срочно назовите код из SMS для отмены перевода денег.")
        page.locator("#check-submit").click()
        page.wait_for_selector("#result h2")
        assert "Invalid Date" not in page.locator("#result").inner_text()
        assert (
            page.locator("#result").get_by_text("Русский ML не применяется", exact=False).count()
            == 1
        )
        page.screenshot(path=str(folder / "03_sms_result.png"), full_page=True)
        page.goto("http://127.0.0.1:8080/check/url/")
        page.locator("#check-input").fill("http://example.test:8080/login?verify=1")
        page.locator("#check-submit").click()
        page.wait_for_selector("#result h2")
        assert page.locator("#result a").count() == 0
        page.screenshot(path=str(folder / "04_url_result.png"), full_page=True)
        page.goto("http://127.0.0.1:8080/history/")
        page.wait_for_selector("#history-table table")
        page.screenshot(path=str(folder / "05_history.png"), full_page=True)
        page.goto("http://127.0.0.1:8080/profile/")
        page.wait_for_function(
            "document.querySelector('#profile-info').textContent.includes('demo')"
        )
        assert not page.locator("[name=store_sms_text]").is_checked()
        page.locator("#logout").click()
        page.wait_for_url("http://127.0.0.1:8080/login/")
        page.goto("http://127.0.0.1:8080/history/")
        page.wait_for_url("http://127.0.0.1:8080/login/")
        browser.close()
    result = {
        "browser": "Microsoft Edge headless",
        "screenshots": 5,
        "widths": widths,
        "page_errors": errors,
        "external_navigation_blocked": True,
        "logout_protection": True,
    }
    (ROOT / "docs/reports/browser_smoke.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    assert not errors, errors
    print(result)


if __name__ == "__main__":
    main()
