import pandas as pd
import pytest

from ml.eda import content_tokens, observations, plot_eda, summarize


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


def test_eda_exports_all_required_analyses_as_png(tmp_path):
    from PIL import Image

    data = pd.DataFrame(
        {
            "text_raw": ["Hello", "Win £10!"],
            "text_clean": ["hello", "win <money>!"],
            "label": [0, 1],
        }
    )
    features, report = summarize(data)
    figures = plot_eda(features, report, tmp_path)
    assert len(figures) == 5
    for path in figures:
        with Image.open(path) as image:
            assert image.format == "PNG" and image.width >= 1000 and image.height >= 700
