import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone

from ml.data.split_sms import split_sms
from ml.features import ManualSmsFeatures, build_feature_pipeline
from ml.prepare_features import prepare_features
from ml.rules import GROUPS, matched_groups


def sample_frame():
    return pd.DataFrame(
        {
            "id": range(1, 101),
            "text_clean": [f"unique message {i}" for i in range(100)],
            "label": [0] * 80 + [1] * 20,
        }
    )


def test_split_is_stratified_deterministic_and_disjoint():
    frame = sample_frame()
    split = split_sms(frame)
    assert split == split_sms(frame)
    assert [len(split[name]) for name in ("train", "val", "test")] == [70, 15, 15]
    assert set(split["train"]).isdisjoint(split["val"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["val"]).isdisjoint(split["test"])
    assert set().union(*map(set, split.values())) == set(range(100))
    assert [frame.iloc[indexes].label.sum() for indexes in split.values()] == [14, 3, 3]


def test_split_rejects_duplicates_and_bad_targets():
    frame = sample_frame()
    frame.loc[1, "text_clean"] = frame.loc[0, "text_clean"]
    with pytest.raises(ValueError, match="дубликаты"):
        split_sms(frame)
    frame = sample_frame()
    frame["label"] = 0
    with pytest.raises(ValueError):
        split_sms(frame)


def test_six_manual_features_use_original_case_and_are_nonnegative():
    matrix = ManualSmsFeatures().fit_transform(
        ["URGENT! Pay $50 at https://example.test", "123", ""]
    )
    assert sparse.issparse(matrix) and matrix.shape == (3, 6)
    row = matrix.toarray()[0]
    assert row[0] == len("URGENT! Pay $50 at https://example.test")
    assert 0 < row[1] <= 1
    assert row[2:].tolist() == [1, 1, 1, 1]
    assert not matrix.toarray()[2].any()
    assert (matrix.data >= 0).all()


def test_pipeline_fit_only_train_no_heldout_vocabulary_or_idf_changes(tmp_path):
    train = ["alpha beta ordinary", "alpha gamma ordinary", "beta gamma notice", "delta notice"]
    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(train)
    word = dict(pipeline.named_steps["features"].transformer_list)["word"].named_steps["tfidf"]
    assert word.ngram_range == (1, 2) and word.min_df == 2
    assert word.max_df == 0.95 and word.sublinear_tf
    vocabulary = word.vocabulary_.copy()
    idf = word.idf_.copy()
    heldout = pipeline.transform(["exclusiveheldoutword fr33 b@nk"])
    assert "exclusiveheldoutword" not in word.vocabulary_
    assert vocabulary == word.vocabulary_
    np.testing.assert_array_equal(idf, word.idf_)
    assert sparse.issparse(X) and heldout.shape[1] == X.shape[1]
    assert len(pipeline.get_feature_names_out()) == X.shape[1]
    char = dict(pipeline.named_steps["features"].transformer_list)["char"].named_steps["tfidf"]
    assert char.analyzer == "char_wb" and char.ngram_range == (3, 5)
    path = tmp_path / "features.joblib"
    joblib.dump(pipeline, path)
    assert (joblib.load(path).transform(train) != X).nnz == 0
    assert clone(pipeline).get_params()["features__word__normalize__remove_stopwords"] is False


def test_entities_remain_word_tokens_and_rules_are_only_group_counts():
    pipeline = build_feature_pipeline(include_char=False)
    pipeline.fit(
        ["offer https://a.test now", "offer https://b.test today", "hello friend", "hello world"]
    )
    names = pipeline.get_feature_names_out()
    assert "word__<url>" in names
    assert len(GROUPS) == 6
    assert "CODE_REQUEST" in matched_groups("Назовите код из SMS")
    assert matched_groups("Встречаемся завтра") == []
    assert "BANK_CARD" not in matched_groups("riverbank")


def test_feature_artifacts_reproduce_and_never_read_russian_test(tmp_path, monkeypatch):
    from sklearn.feature_extraction.text import TfidfVectorizer

    from ml.data.load_sms import sha256, write_json
    from ml.preprocessing import normalize_sms

    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    raw_text = [
        f"{'ordinary hello' if i < 80 else 'claim prize'} word{chr(97 + i // 26)}{chr(97 + i % 26)}"
        for i in range(100)
    ]
    frame = pd.DataFrame(
        {
            "id": range(1, 101),
            "text_raw": raw_text,
            "text_clean": [normalize_sms(t) for t in raw_text],
            "label": [0] * 80 + [1] * 20,
        }
    )
    source = processed / "sms_clean.csv"
    frame.to_csv(source, index=False)
    write_json(
        tmp_path / "docs/reports/04_preprocessing.json",
        {"clean_sha256": sha256(source.read_bytes())},
    )
    russian = tmp_path / "data/raw/custom_ru.csv"
    russian.parent.mkdir()
    russian.write_bytes(b"INVALID UTF8: \xff")  # Deliberately unreadable, must never be opened.
    original = TfidfVectorizer.fit_transform
    fitted_on = []

    def record_fit(self, documents, y=None):
        documents = list(documents)
        fitted_on.append(documents)
        return original(self, documents, y=y)

    monkeypatch.setattr(TfidfVectorizer, "fit_transform", record_fit)
    result = prepare_features(tmp_path)
    assert len(fitted_on) == 2  # word and char branches, each on train only
    positions = split_sms(frame)
    expected = frame.iloc[positions["train"]].text_clean.tolist()
    assert fitted_on == [expected, expected]
    again = prepare_features(tmp_path)
    assert result == again
    assert result["pipeline_reload_equal"] is True and result["classifier_trained"] is False
    assert result["matrix_shapes"]["train"][0] == 70
    assert result["feature_counts"]["manual"] == 6
    assert len(result["top15_train_tfidf_by_class"]["1"]) == 15
    for name, indexes in positions.items():
        matrix = sparse.load_npz(processed / f"X_{name}.npz")
        assert matrix.shape[0] == len(indexes)
        np.testing.assert_array_equal(
            np.load(processed / f"y_{name}.npy"), frame.iloc[indexes].label
        )
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="изменён"):
        prepare_features(tmp_path)
