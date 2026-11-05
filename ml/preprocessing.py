"""Pure shared SMS normalization; no file downloads or corpus fitting at inference."""

import re
import unicodedata
from pathlib import Path

import pandas as pd
from nltk.tokenize import RegexpTokenizer

from ml.data.load_sms import ROOT, sha256, write_json

EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.UNICODE)
URL = re.compile(
    r"https?://[^\s<>]+|www\.[^\s<>]+|"
    r"(?<![\w@./])(?:[a-zа-яё0-9](?:[a-zа-яё0-9-]*[a-zа-яё0-9])?\.)+"
    r"(?:com|org|net|edu|gov|info|biz|ru|рф|io|co|uk|me|ly|test|example)"
    r"(?::\d+)?(?:[/\?#][^\s<>]*)?(?![\w.-])",
    re.IGNORECASE,
)
AMOUNT = r"\d+(?:[\s,.]\d+)*"
CURRENCY = r"(?:руб(?:лей|ля|ль)?\.?|тг|usd|gbp|eur|rub)"
MONEY = re.compile(
    rf"[$£€₸]\s*{AMOUNT}|(?<!\w){AMOUNT}\s*(?:[$£€₸]|{CURRENCY}(?!\w))|"
    rf"(?<!\w){CURRENCY}\s*{AMOUNT}(?!\w)",
    re.IGNORECASE,
)
PHONE = re.compile(r"(?<!\w)\+?\d(?:[\d ()-]*\d)?(?!\w)")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
TOKENS = RegexpTokenizer(r"<[a-z]+>|[^\W\d_]+(?:'[^\W\d_]+)?|[!?]|[^\w\s]", flags=re.UNICODE)


def load_stopwords(language: str = "en") -> frozenset[str]:
    if language not in {"en", "ru"}:
        raise ValueError("Допустимы только en/ru")
    path = Path(__file__).parent / "resources" / f"stopwords_{language}.txt"
    return frozenset(path.read_text(encoding="utf-8").splitlines())


def _replace_url(match: re.Match) -> str:
    value = match.group()
    # Do not swallow sentence pressure/punctuation along with a URL.
    suffix = ""
    while value and value[-1] in ".,!?;:)]}":
        suffix = value[-1] + suffix
        value = value[:-1]
    return " <url> " + suffix


def normalization_steps(text: str) -> dict[str, str]:
    """Named intermediate steps used by normalize_sms and the data-cleaning audit."""
    if not isinstance(text, str):
        raise TypeError("Текст SMS должен быть строкой")
    steps = {}
    text = unicodedata.normalize("NFKC", text).lower()
    steps["nfkc_lower"] = text
    text = "".join(" " if unicodedata.category(char).startswith("C") else char for char in text)
    steps["controls"] = text
    text = EMAIL.sub(" <email> ", text)
    steps["email"] = text
    text = URL.sub(_replace_url, text)
    steps["url"] = text
    text = MONEY.sub(" <money> ", text)
    steps["money"] = text
    text = PHONE.sub(
        lambda match: (
            " <phone> "
            if 7 <= sum(char.isdigit() for char in match.group()) <= 15
            else match.group()
        ),
        text,
    )
    steps["phone"] = text
    text = NUMBER.sub(" <num> ", text)
    steps["number"] = text
    text = re.sub(r"!+", "!", text)
    text = re.sub(r"\?+", "?", text)
    steps["pressure"] = text
    text = " ".join(text.split())
    steps["whitespace"] = text
    return steps


def normalize_sms(text: str, *, remove_stopwords: bool = False, language: str = "en") -> str:
    """NFKC/lower → entities → pressure punctuation → whitespace; preserves Unicode."""
    cleaned = normalization_steps(text)["whitespace"]
    if remove_stopwords:
        excluded = load_stopwords(language)
        cleaned = " ".join(token for token in TOKENS.tokenize(cleaned) if token not in excluded)
    return cleaned


def clean_sms(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove empty, exact and normalized duplicates BEFORE splitting.

    Conflicting-label normalized groups are excluded entirely rather than silently
    keeping an arbitrary target. Surviving IDs always reference original UCI lines.
    """
    required = {"id", "text", "label"}
    if not required <= set(frame.columns) or not frame.id.is_unique:
        raise ValueError("Нужны уникальные id и столбцы text/label")
    if frame.label.isna().any() or not set(frame.label) <= {0, 1}:
        raise ValueError("Метки должны быть 0/1")
    work = frame.rename(columns={"text": "text_raw"}).copy()
    if work.text_raw.dropna().map(lambda value: not isinstance(value, str)).any():
        raise TypeError("Нечисловой текст или пропуск ожидается в text")
    report = {"input_rows": len(work), "steps": [], "deduplication": []}

    def remove(mask, step):
        nonlocal work
        removed = work.loc[mask]
        report["deduplication"].append(
            {
                "step": step,
                "removed": len(removed),
                "by_class": {str(k): int((removed.label == k).sum()) for k in (0, 1)},
                "removed_ids": removed.id.tolist(),
            }
        )
        work = work.loc[~mask].copy()

    remove(work.text_raw.isna() | work.text_raw.fillna("").str.strip().eq(""), "empty_raw")
    # Count affected records for each transformation, not the number of regex matches.
    all_steps = work.text_raw.map(normalization_steps)
    previous = work.text_raw
    for name in (
        "nfkc_lower",
        "controls",
        "email",
        "url",
        "money",
        "phone",
        "number",
        "pressure",
        "whitespace",
    ):
        values = all_steps.map(lambda steps: steps[name])
        report["steps"].append({"step": name, "affected_records": int((values != previous).sum())})
        previous = values
    work["text_clean"] = previous
    remove(work.text_clean.eq(""), "empty_clean")
    conflicting = work.groupby("text_clean").label.nunique()
    conflicting = conflicting[conflicting > 1].index
    remove(work.text_clean.isin(conflicting), "conflicting_label_groups")
    remove(work.text_raw.duplicated(keep="first"), "exact_duplicates")
    remove(work.text_clean.duplicated(keep="first"), "normalized_duplicates")
    work = work[["id", "text_raw", "text_clean", "label"]].reset_index(drop=True)
    report["output_rows"] = len(work)
    report["class_counts"] = {str(k): int((work.label == k).sum()) for k in (0, 1)}
    report["conflicting_groups"] = len(conflicting)
    return work, report


def main():
    source = ROOT / "data/processed/sms_raw.csv"
    frame = pd.read_csv(source, keep_default_na=False)
    cleaned, report = clean_sms(frame)
    target = source.with_name("sms_clean.csv")
    cleaned.to_csv(target, index=False, encoding="utf-8")
    report["source_sha256"] = sha256(source.read_bytes())
    report["clean_sha256"] = sha256(target.read_bytes())
    write_json(ROOT / "docs/reports/04_preprocessing.json", report)
    print(f"UCI: {len(frame)} -> {len(cleaned)}; report: docs/reports/04_preprocessing.json")


if __name__ == "__main__":
    main()
