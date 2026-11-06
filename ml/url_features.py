"""Offline URL parsing only. No requests, DNS, redirects or live reputation lookups."""

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit

import tldextract
from rapidfuzz.distance import Levenshtein

PSL = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
SHORTENERS = set(
    "bit.ly tinyurl.com goo.gl t.co is.gd cutt.ly ow.ly clck.ru rb.gy buff.ly "
    "rebrand.ly shorturl.at tiny.cc lnkd.in s.id bl.ink v.gd".split()
)
BRANDS = (
    "google.com microsoft.com apple.com amazon.com paypal.com facebook.com instagram.com "
    "whatsapp.com telegram.org netflix.com github.com dropbox.com yahoo.com outlook.com "
    "sberbank.ru tinkoff.ru vtb.ru kaspi.kz halykbank.kz gosuslugi.ru ebay.com linkedin.com"
)
BRANDS = BRANDS.split()
WORDS = (
    "login verify secure account update bank confirm password bonus free prize "
    "wallet payment support"
).split()
FEATURE_NAMES = [
    "url_length",
    "host_length",
    "path_length",
    "dots",
    "subdomains",
    "host_hyphens",
    "digits",
    "host_digit_ratio",
    "specials",
    "ip_host",
    "https",
    "unusual_port",
    "at_sign",
    "double_slash_path",
    "shortener",
    "suspicious_words",
    "punycode",
    "mixed_scripts",
    "brand_similarity",
    "host_entropy",
    "flagged_tld",
    "query_parameters",
]


def validate_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise ValueError("Введите URL длиной от 1 до 2048 символов")
    value = value.strip()
    if re.search(r"[\s\x00-\x1f\x7f]", value):
        raise ValueError("URL не должен содержать пробелы или управляющие символы")
    candidate = value if "://" in value else "http://" + value
    try:
        parsed = urlsplit(candidate)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError()
        host = parsed.hostname
        _ = parsed.port
        try:
            ipaddress.ip_address(host)
        except ValueError:
            ascii_host = host.encode("idna").decode("ascii")
            if len(ascii_host) > 253 or "." not in ascii_host:
                raise ValueError()
            if any(
                not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
                for label in ascii_host.rstrip(".").split(".")
            ):
                raise ValueError()
    except (ValueError, UnicodeError):
        raise ValueError("Некорректный URL: проверьте домен, схему и порт") from None
    return candidate


def registrable_domain(value: str) -> str:
    host = urlsplit(validate_url(value)).hostname.lower()
    extracted = PSL(host)
    return extracted.top_domain_under_public_suffix or host


def _model_feature_contract_v1(value: str) -> dict:
    value = validate_url(value)
    parsed = urlsplit(value)
    host = parsed.hostname.lower()
    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0
    extracted = PSL(host)
    try:
        unicode_host = host.encode("ascii").decode("idna")
    except (UnicodeError, UnicodeEncodeError):
        unicode_host = host
    domain = extracted.top_domain_under_public_suffix or host
    label = extracted.domain or host.split(".")[0]
    similarity = any(
        domain != brand and Levenshtein.distance(label, brand.split(".")[0]) <= 2
        for brand in BRANDS
    )
    frequencies = Counter(host)
    values = [
        len(value),
        len(host),
        len(parsed.path),
        value.count("."),
        len(extracted.subdomain.split(".")) if extracted.subdomain and not is_ip else 0,
        host.count("-"),
        sum(c.isdigit() for c in value),
        sum(c.isdigit() for c in host) / max(len(host), 1),
        sum(value.count(c) for c in "@~%=&_?"),
        is_ip,
        int(parsed.scheme == "https"),
        int(parsed.port is not None and parsed.port not in {80, 443}),
        int("@" in value),
        int("//" in parsed.path),
        int(host in SHORTENERS),
        sum(word in unquote(value).lower() for word in WORDS),
        int("xn--" in host),
        int(bool(re.search("[а-яё]", unicode_host) and re.search("[a-z]", unicode_host))),
        int(similarity),
        -sum((n / len(host)) * math.log2(n / len(host)) for n in frequencies.values()),
        int(extracted.suffix in {"xyz", "top", "click", "zip"}),
        len(parse_qsl(parsed.query)),
    ]
    return dict(zip(FEATURE_NAMES, values, strict=True))


def extract_url_features(value: str) -> dict:
    features = _model_feature_contract_v1(value)
    host = urlsplit(validate_url(value)).hostname.lower()
    extracted = PSL(host)
    label = extracted.domain
    domain = extracted.top_domain_under_public_suffix or host
    if host.endswith((".test", ".example")):
        # Reserved suffixes are absent from the public PSL but still have clear label boundaries.
        labels = host.split(".")
        features["subdomains"] = max(0, len(labels) - 2)
        label = labels[-2]
    try:
        label = label.encode("ascii").decode("idna")
    except UnicodeError:
        pass
    features["brand_similarity"] = int(
        any(
            domain != brand and Levenshtein.distance(label, brand.split(".")[0]) <= 2
            for brand in BRANDS
        )
    )
    return features


def model_url(value: str) -> str:
    """Uniform scheme/www treatment for BOTH train and inference: counter source artifacts."""
    parsed = urlsplit(validate_url(value))
    host = parsed.hostname.lower().removeprefix("www.")
    if ":" in host:
        host = "[" + host + "]"
    if parsed.port is not None:
        host += ":" + str(parsed.port)
    if parsed.username is not None:
        host = parsed.username + "@" + host
    return urlunsplit(("http", host, parsed.path, parsed.query, parsed.fragment))


def model_features(value: str) -> list[float]:
    # Freeze classifier features. Reserved-domain fixes are rule/display only.
    return list(_model_feature_contract_v1(model_url(value)).values())


def explain_url(features: dict) -> list[str]:
    titles = {
        "ip_host": "IP-адрес вместо домена",
        "unusual_port": "Нестандартный порт",
        "at_sign": "Символ @ в URL",
        "double_slash_path": "Двойной слеш в пути",
        "shortener": "Сокращённая ссылка",
        "suspicious_words": "Подозрительные слова в адресе",
        "punycode": "Домен Punycode",
        "mixed_scripts": "Смешение алфавитов",
        "brand_similarity": "Сходство с доменом известного сервиса",
        "flagged_tld": "Доменная зона из эвристического списка (не доказательство угрозы)",
    }
    return [title for name, title in titles.items() if features.get(name)]
