"""Task-6 feature pipeline only; this module does not train a classifier."""

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils.validation import check_is_fitted

from ml.preprocessing import normalize_sms
from ml.rules import matched_groups

MANUAL_NAMES = np.array(
    [
        "length_raw",
        "uppercase_fraction",
        "exclamation_count",
        "has_url",
        "has_money",
        "se_group_count",
    ],
    dtype=object,
)


class SmsNormalizer(TransformerMixin, BaseEstimator):
    def __init__(self, remove_stopwords=False):
        self.remove_stopwords = remove_stopwords

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, "is_fitted_")
        return [
            normalize_sms(text, remove_stopwords=self.remove_stopwords, language="en") for text in X
        ]

    def get_feature_names_out(self, input_features=None):
        return np.array(["normalized_text"], dtype=object)


class ManualSmsFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, "is_fitted_")
        rows = []
        for text in X:
            cleaned = normalize_sms(text)
            letters = sum(character.isalpha() for character in text)
            upper = sum(character.isupper() for character in text)
            rows.append(
                [
                    len(text),
                    upper / letters if letters else 0.0,
                    text.count("!"),
                    int("<url>" in cleaned),
                    int("<money>" in cleaned),
                    len(matched_groups(text)),
                ]
            )
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float64).reshape(-1, len(MANUAL_NAMES)))

    def get_feature_names_out(self, input_features=None):
        return MANUAL_NAMES.copy()


def build_feature_pipeline(*, include_char=True, remove_stopwords=False) -> Pipeline:
    """Raw SMS → word/char TF-IDF + six raw/manual features, as a serializable Pipeline."""
    word = Pipeline(
        [
            ("normalize", SmsNormalizer(remove_stopwords=remove_stopwords)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    token_pattern=r"(?u)<[a-z]+>|\b\w\w+\b",
                ),
            ),
        ]
    )
    transformers = [("word", word)]
    if include_char:
        transformers.append(
            (
                "char",
                Pipeline(
                    [
                        ("normalize", SmsNormalizer(remove_stopwords=remove_stopwords)),
                        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
                    ]
                ),
            )
        )
    transformers.append(("manual", ManualSmsFeatures()))
    return Pipeline([("features", FeatureUnion(transformers))])
