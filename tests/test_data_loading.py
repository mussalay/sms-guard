import json
import re

import pandas as pd
import pytest

from ml.data.custom_ru import build_custom_ru
from ml.data.load_sms import ROOT, download_uci, load_sms, prepare_raw


def test_loader_preserves_first_tab_and_stable_ids(tmp_path):
    path = tmp_path / "sms"
    path.write_text("ham\tHello\tfriend\nspam\tPrize £5\n", encoding="utf-8")
    frame = load_sms(path)
    assert frame.to_dict("list") == {
        "id": [1, 2],
        "text": ["Hello\tfriend", "Prize £5"],
        "label": [0, 1],
    }


@pytest.mark.parametrize("content", [b"", b"ham No tab", b"other\tMessage", b"ham\t\xff"])
def test_loader_rejects_invalid_input(tmp_path, content):
    path = tmp_path / "sms"
    path.write_bytes(content)
    with pytest.raises((ValueError, UnicodeDecodeError)):
        load_sms(path)


def test_prepare_raw_reports_missing_and_duplicates(tmp_path):
    path = tmp_path / "sms"
    path.write_text("ham\tHello\nham\tHello\nspam\t\n", encoding="utf-8")
    report = prepare_raw(path, tmp_path / "out", tmp_path / "report.json")
    assert report["rows"] == 3
    assert report["exact_duplicate_texts"] == 1
    assert report["empty_text"] == 1
    assert len(pd.read_csv(tmp_path / "out/sms_raw.csv")) == 3


def test_archive_checksum_mismatch_is_not_overwritten(tmp_path):
    archive = tmp_path / "sms_spam_collection.zip"
    archive.write_bytes(b"changed")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"archive_sha256": "not-correct"}))
    with pytest.raises(ValueError, match="SHA256"):
        download_uci(tmp_path, manifest)
    assert archive.read_bytes() == b"changed"


def test_custom_ru_is_reproducible_balanced_and_test_only(tmp_path):
    path = tmp_path / "custom.csv"
    build_custom_ru(path)
    assert path.read_bytes() == (ROOT / "data/raw/custom_ru.csv").read_bytes()
    frame = pd.read_csv(path)
    assert len(frame) >= 100
    assert frame.label.value_counts().to_dict() == {0: 50, 1: 50}
    assert frame.text.is_unique and frame.id.is_unique
    assert set(frame.language) == {"ru"}
    assert set(frame.split) == {"test_only"}
    assert frame.synthetic.all()
    assert frame.annotation_author.notna().all()
    assert not frame.text.str.contains(r"\d{7,}|[\w.+-]+@[\w.-]+", regex=True).any()
    hosts = re.findall(r"https?://([^/\s]+)", " ".join(frame.text))
    assert all(host.endswith((".test", ".example")) for host in hosts)
