import pandas as pd
import pytest

from ml.eda import content_tokens, observations, summarize


def test_eda_counts_word_boundaries_and_indicator_denominators():
    data = pd.DataFrame(
        {
            "text_raw": ["Hello friend", "Second text", "Win money!", "Claim prize"],
            "text_clean": ["hello friend", "second text", "win <money>! <url>", "claim prize"],
            "label": [0, 0, 1, 1],
        }
    )
    _, report = summarize(data)
    assert report["rows"] == 4
    assert report["classes"]["0"]["percentage"] == 50
    assert report["classes"]["1"]["indicator_percentages"]["url"] == 50
    assert report["classes"]["0"]["indicator_percentages"]["url"] == 0
    bigrams = dict(report["classes"]["0"]["top20_bigrams"])
    assert "friend second" not in bigrams
    assert bigrams["hello friend"] == 1
    assert len(observations(report)) >= 5
    assert "the" not in content_tokens("the prize")


def test_eda_rejects_empty_corpus():
    with pytest.raises(ValueError):
        summarize(pd.DataFrame())
