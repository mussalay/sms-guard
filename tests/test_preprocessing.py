import pandas as pd
import pytest

from ml.preprocessing import clean_sms, normalize_sms


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        (" \t\n ", ""),
        ("HeLLo WORLD", "hello world"),
        ("ПРИВЕТ Ёжик", "привет ёжик"),
        ("http://bank.test/login?q=123", "<url>"),
        ("www.example.com/a", "<url>"),
        ("bit.ly/prize", "<url>"),
        ("https://пример.рф/путь!!!", "<url> !"),
        ("Mail Test.Name+tag@example.com", "mail <email>"),
        ("Call +1 (202) 555-0100 now", "call <phone> now"),
        ("Звоните 020 7946 0958", "звоните <phone>"),
        ("£1,000 or $100.50 or 2 500 руб", "<money> or <money> or <money>"),
        ("50 ₸ и 100 тг", "<money> и <money>"),
        ("Заказ 1234 готов в 18:30", "заказ <num> готов в <num> : <num>"),
        ("FREE!!! Really???", "free! really?"),
        ("ＨＥＬＬＯ　１２３", "hello <num>"),
        ("a\x00b\u200bc", "a b c"),
        ("Спасибо 😊 👍", "спасибо 😊 👍"),
        ("<url> <email> <phone> <money> <num>", "<url> <email> <phone> <money> <num>"),
        ("CODE 123456", "code <num>"),
        ("1234567890123456", "<num>"),
    ],
)
def test_entity_boundaries_and_unicode(raw, expected):
    assert normalize_sms(raw) == expected


@pytest.mark.parametrize("value", [None, 1, b"binary", float("nan")])
def test_non_string_input_rejected(value):
    with pytest.raises(TypeError):
        normalize_sms(value)


def test_normalization_is_idempotent_and_does_not_mutate():
    raw = "PAY $200 to Example@Example.com at http://example.test?q=5!!!"
    once = normalize_sms(raw)
    assert normalize_sms(once) == once
    assert raw.startswith("PAY $200")
    assert once.count("<money>") == 1
    assert once.count("<email>") == 1
    assert once.count("<url>") == 1


def test_optional_local_stopwords_preserve_entity_tokens():
    assert normalize_sms("this is a FREE prize at https://example.test", remove_stopwords=True) == (
        "free prize <url>"
    )
    with pytest.raises(ValueError):
        normalize_sms("hello", remove_stopwords=True, language="other")


def test_empty_duplicates_and_conflicts_are_audited_before_split():
    frame = pd.DataFrame(
        {
            "id": range(1, 9),
            "text": [None, " ", "Hi", "Hi", "HI", "Prize 1", "Prize 2", "\x00"],
            "label": [0, 0, 0, 0, 0, 0, 1, 0],
        }
    )
    original = frame.copy(deep=True)
    cleaned, report = clean_sms(frame)
    pd.testing.assert_frame_equal(frame, original)
    assert cleaned.id.tolist() == [3]
    assert cleaned.text_raw.tolist() == ["Hi"]
    assert cleaned.text_clean.tolist() == ["hi"]
    stats = {row["step"]: row for row in report["deduplication"]}
    assert stats["empty_raw"]["removed"] == 2
    assert stats["empty_clean"]["removed"] == 1
    assert stats["conflicting_label_groups"]["by_class"] == {"0": 1, "1": 1}
    assert stats["exact_duplicates"]["removed"] == 1
    assert stats["normalized_duplicates"]["removed"] == 1
    assert sum(row["removed"] for row in stats.values()) + report["output_rows"] == len(frame)


def test_whole_sms_processing_has_no_network(monkeypatch):
    import socket
    import urllib.request

    def forbidden(*args, **kwargs):
        raise AssertionError("Network is forbidden during normalization")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    assert normalize_sms("https://never-open.test") == "<url>"
