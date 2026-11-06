"""Combined local analysis; Russian uses rules and URL only."""

import re
from dataclasses import asdict, dataclass

from app.analysis.social import matched_groups
from ml.inference import ModelStore
from ml.preprocessing import URL
from ml.risk import compute_risk, compute_url_risk
from ml.url_features import explain_url, extract_url_features, validate_url


@dataclass
class AnalysisResult:
    risk_score: int
    risk_level: str
    category: str
    explanation: list[str]
    recommendation: str
    disclaimer: str
    type: str
    language: str
    p_sms: float | None
    p_url_max: float | None
    urls: list[dict]
    features: list[str]
    language_warning: str | None
    model_versions: dict

    @property
    def payload(self):
        return asdict(self)


def analyze_url(url: str, store: ModelStore) -> dict:
    normalized = validate_url(url)
    features = extract_url_features(normalized)
    probability = store.predict_url(normalized)
    thresholds = store.url_meta["thresholds"]
    category = (
        "DANGEROUS"
        if probability >= thresholds["t2"]
        else "SUSPICIOUS"
        if probability >= thresholds["t1"]
        else "SAFE"
    )
    from urllib.parse import urlsplit

    return {
        "url": url,
        "domain": urlsplit(normalized).hostname,
        "p_url": probability,
        "category": category,
        "features": features,
        "explanation": explain_url(features),
    }


def analyze(text: str, check_type: str, store: ModelStore) -> AnalysisResult:
    if check_type == "URL":
        urls = [analyze_url(text, store)]
        language, p_sms, groups = "other", None, []
    else:
        if not text.strip() or len(text) > 1000:
            raise ValueError("SMS должно содержать от 1 до 1000 символов")
        language = (
            "ru"
            if re.search(r"[а-яё]", text, re.I)
            else "en"
            if re.search("[a-z]", text, re.I)
            else "other"
        )
        groups = matched_groups(text)
        p_sms = store.predict_sms(text)["probability"] if language == "en" else None
        urls = []
        for match in list(URL.finditer(text))[:3]:
            candidate = match.group().rstrip(".,!?;:)]}")
            try:
                urls.append(analyze_url(candidate, store))
            except ValueError:
                continue
    p_url = max((u["p_url"] for u in urls), default=None)
    risk = compute_risk(
        p_sms,
        p_url,
        groups,
        [u["features"] for u in urls],
        t2=store.url_meta.get("thresholds", {}).get("t2", 0.7),
    )
    if check_type == "URL":
        risk = compute_url_risk(p_url, urls[0]["features"], store.url_meta["thresholds"])
    return AnalysisResult(
        **{
            **risk,
            "type": check_type,
            "language": language,
            "p_sms": p_sms,
            "p_url_max": p_url,
            "urls": urls,
            "features": groups,
            "language_warning": "Русский ML не применяется: оценка основана на правилах и URL"
            if language == "ru"
            else "Язык не поддерживается SMS-моделью"
            if language == "other" and check_type == "SMS"
            else None,
            "model_versions": {
                "sms": store.sms_meta.get("version") if p_sms is not None else None,
                "url": store.url_meta.get("version") if urls else None,
            },
        }
    )


def mask_text(text: str) -> str:
    from ml.preprocessing import MONEY, PHONE

    text = MONEY.sub("***", text)
    text = PHONE.sub(
        lambda m: "***" if sum(c.isdigit() for c in m.group()) >= 7 else m.group(), text
    )
    return re.sub(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)", "***", text)
