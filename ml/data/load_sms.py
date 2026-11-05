"""Official UCI acquisition; no links found inside SMS are ever opened."""

import argparse
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
RAW_DIR = ROOT / "data/raw"
SOURCE_RECORD = ROOT / "docs/datasets/uci_source.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download_uci(raw_dir: Path = RAW_DIR, record_path: Path = SOURCE_RECORD) -> Path:
    """Cache immutable archive; verify against recorded SHA256 on subsequent runs."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "sms_spam_collection.zip"
    if not archive.exists():
        # Only this constant official endpoint, never arbitrary input URLs.
        with urlopen(SOURCE_URL, timeout=30) as response:
            payload = response.read(2_000_001)
        if len(payload) > 2_000_000:
            raise ValueError("Архив UCI превышает допустимый размер")
        with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
            if not {"SMSSpamCollection", "readme"} <= set(zipped.namelist()):
                raise ValueError("Неожиданный состав архива UCI")
        archive.write_bytes(payload)
    digest = sha256(archive.read_bytes())
    if record_path.exists():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        if previous["archive_sha256"] != digest:
            raise ValueError("SHA256 архива не совпадает с паспортом; источник мог измениться")
    else:
        write_json(
            record_path,
            {
                "source_url": SOURCE_URL,
                "source_page": SOURCE_PAGE,
                "doi": "10.24432/C5CC84",
                "citation": "Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. UCI.",
                "license": "CC BY 4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "license_checked_on": "2026-11-06",
                "downloaded_at": datetime.fromtimestamp(
                    archive.stat().st_mtime, ZoneInfo("Asia/Qyzylorda")
                ).isoformat(),
                "archive_sha256": digest,
            },
        )
    with zipfile.ZipFile(archive) as zipped:
        # Explicit member names avoid Zip Slip and preserve original bytes.
        for name in ("SMSSpamCollection", "readme"):
            payload = zipped.read(name)
            target = raw_dir / name
            if target.exists() and target.read_bytes() != payload:
                raise ValueError(f"Исходный файл {name} изменён; перезапись запрещена")
            if not target.exists():
                target.write_bytes(payload)
    return raw_dir / "SMSSpamCollection"


def load_sms(path: Path = RAW_DIR / "SMSSpamCollection") -> pd.DataFrame:
    """Strict UTF-8, one label and text separated by the FIRST tab; stable line IDs."""
    content = Path(path).read_bytes().decode("utf-8", errors="strict")
    records = []
    for number, line in enumerate(content.splitlines(), 1):
        label, separator, text = line.partition("\t")
        if not separator or label not in {"ham", "spam"}:
            raise ValueError(f"Некорректная строка UCI: {number}")
        records.append({"id": number, "text": text, "label": int(label == "spam")})
    if not records:
        raise ValueError("Пустой набор UCI")
    return pd.DataFrame(records).astype({"id": "int64", "label": "int64"})


def redact_example(text: str) -> str:
    """Public-corpus examples in a document, not raw SMS in application logs."""
    text = re.sub(r"(?:https?://|www\.)\S+", "[URL]", text, flags=re.I)
    text = re.sub(r"[\w.+-]+@[\w.-]+", "[EMAIL]", text)
    return re.sub(r"\d", "*", text)


def prepare_raw(raw_path: Path, output_dir: Path, report_path: Path) -> dict:
    frame = load_sms(raw_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "sms_raw.csv", index=False, encoding="utf-8")
    report = {
        "rows": len(frame),
        "encoding": "utf-8 (strict decode)",
        "file_sha256": sha256(raw_path.read_bytes()),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "class_counts": {
            str(k): int(v) for k, v in frame.label.value_counts().sort_index().items()
        },
        "missing_text": int(frame.text.isna().sum()),
        "empty_text": int(frame.text.str.strip().eq("").sum()),
        "exact_duplicate_texts": int(frame.text.duplicated().sum()),
        "examples_redacted": {
            str(label): [redact_example(t) for t in frame.loc[frame.label == label, "text"].head(5)]
            for label in (0, 1)
        },
    }
    write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Загрузка оригинального UCI SMS Spam Collection")
    parser.add_argument(
        "--download", action="store_true", help="Получить/проверить официальный архив"
    )
    args = parser.parse_args()
    raw = download_uci() if args.download else RAW_DIR / "SMSSpamCollection"
    report = prepare_raw(raw, ROOT / "data/processed", ROOT / "docs/reports/03_sms_loading.json")
    # No original message content in console/application logs.
    print(json.dumps({k: v for k, v in report.items() if k != "examples_redacted"}, indent=2))


if __name__ == "__main__":
    main()
