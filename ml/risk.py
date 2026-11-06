"""Configurable risk calculation, including absence of an English SMS probability."""

from pathlib import Path

import yaml

from ml.data.load_sms import ROOT

DISCLAIMER = "Оценка носит вероятностный характер и не гарантирует безопасность"


def compute_risk(
    p_sms,
    p_url_max,
    se_groups,
    url_flags,
    *,
    t2=0.7,
    config_path=ROOT / "risk_config.yaml",
    configuration=None,
):
    config = configuration or yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    weights, overrides = config["weights"], config["overrides"]
    numerator, denominator = 0.0, 0.0
    for name, value in (("sms", p_sms), ("url", p_url_max), ("se", min(len(se_groups) / 3, 1))):
        if value is not None:
            if not 0 <= value <= 1:
                raise ValueError("Вероятность вне диапазона 0–1")
            numerator += weights[name] * value
            denominator += weights[name]
    score = round(100 * numerator / denominator) if denominator else 0
    base_score = score
    contributions = {}
    reasons = []
    if p_sms is not None and p_sms > 0.5:
        reasons.append("SMS-модель обнаружила признаки спама")
        contributions[reasons[-1]] = weights["sms"] * p_sms / denominator
    if p_url_max is not None and p_url_max > 0.5:
        reasons.append("URL-модель обнаружила признаки опасной ссылки")
        contributions[reasons[-1]] = weights["url"] * p_url_max / denominator
    titles = {
        "URGENCY": "Срочность",
        "MONEY_TRANSFER": "Запрос перевода денег",
        "BANK_CARD": "Упоминание банка/карты",
        "ACCOUNT_BLOCK": "Угроза блокировки",
        "PRIZE": "Выигрыш или приз",
        "CODE_REQUEST": "Запрос секретного кода/пароля",
    }
    reasons += [titles.get(group, group) for group in se_groups]
    for group in se_groups:
        contributions[titles.get(group, group)] = (
            weights["se"] * min(len(se_groups) / 3, 1) / max(1, len(se_groups)) / denominator
        )
    if "CODE_REQUEST" in se_groups:
        score = max(score, overrides["code_request"])
        title = titles["CODE_REQUEST"]
        contributions[title] = max(contributions[title], (score - base_score) / 100)
    if (
        p_sms is not None
        and p_url_max is not None
        and p_sms >= overrides["sms_probability"]
        and p_url_max >= t2
    ):
        score = max(score, overrides["both_models"])
    if any(
        f.get("ip_host") or (f.get("punycode") and f.get("brand_similarity")) for f in url_flags
    ):
        score = max(score, overrides["url_structural"])
        reasons.insert(0, "Структурные признаки маскировки URL")
        contributions[reasons[0]] = max(0, (score - base_score) / 100)
    reasons.sort(key=lambda reason: contributions.get(reason, 0), reverse=True)
    return _risk_result(score, reasons, config)


def _risk_result(score, reasons, config):
    score = max(0, min(100, score))
    level = (
        "HIGH"
        if score >= config["levels"]["high"]
        else "MEDIUM"
        if score >= config["levels"]["medium"]
        else "LOW"
    )
    category = {"LOW": "SAFE", "MEDIUM": "SUSPICIOUS", "HIGH": "FRAUDULENT"}[level]
    recommendation = {
        "HIGH": (
            "Не переходите по ссылкам и не сообщайте коды. "
            "Свяжитесь с организацией по официальному номеру."
        ),
        "MEDIUM": "Проверьте отправителя через официальный канал. Не вводите персональные данные.",
        "LOW": (
            "Явных признаков недостаточно. Всё равно проверяйте отправителя "
            "и не сообщайте секретные коды."
        ),
    }[level]
    return {
        "risk_score": score,
        "risk_level": level,
        "category": category,
        "explanation": (reasons or ["Выраженных признаков угрозы не обнаружено"])[:6],
        "recommendation": recommendation,
        "disclaimer": DISCLAIMER,
    }


def compute_url_risk(probability, flags, thresholds, *, config_path=ROOT / "risk_config.yaml"):
    """Standalone URL: map its validated probability bands to matching risk-score bands.

    There is no text/SE signal to dilute a standalone URL score. Structural overrides
    can raise severity, but the summary must never be safer than its URL classifier.
    """
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    base = compute_risk(None, probability, [], [flags], t2=thresholds["t2"], configuration=config)
    medium, high = config["levels"]["medium"], config["levels"]["high"]
    t1, t2 = thresholds["t1"], thresholds["t2"]
    if probability < t1:
        lo, hi, start, end = 0, t1, 0, medium - 1
    elif probability < t2:
        lo, hi, start, end = t1, t2, medium, high - 1
    else:
        lo, hi, start, end = t2, 1, high, 100
    mapped = start + round((probability - lo) / max(hi - lo, 1e-12) * (end - start))
    reasons = base["explanation"]
    if probability >= t1:
        reasons = [r for r in reasons if r != "Выраженных признаков угрозы не обнаружено"]
        reasons.insert(0, "URL-модель обнаружила признаки угрозы по настроенным порогам")
    result = _risk_result(max(mapped, base["risk_score"]), reasons, config)
    result["category"] = {"LOW": "SAFE", "MEDIUM": "SUSPICIOUS", "HIGH": "DANGEROUS"}[
        result["risk_level"]
    ]
    return result
