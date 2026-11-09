"""Read-only presentation of fixed experiment reports; no estimators or data loaders."""

import json
import math

from app.core.config import BASE_DIR

REPORT_FILES = {
    "sms_selection": BASE_DIR / "docs/reports/07_selection.json",
    "sms_test": BASE_DIR / "docs/reports/07_final_test.json",
    "sms_meta": BASE_DIR / "ml/artifacts/sms_model_meta.json",
    "sms_features": BASE_DIR / "docs/reports/06_features.json",
    "sms_raw": BASE_DIR / "docs/reports/03_sms_loading.json",
    "sms_clean": BASE_DIR / "docs/reports/04_preprocessing.json",
    "url": BASE_DIR / "docs/reports/url_experiment.json",
    "url_meta": BASE_DIR / "ml/artifacts/url_model_meta.json",
}
FIGURES = {
    "sms": BASE_DIR / "docs/figures/sms_roc_pr.png",
    "url": BASE_DIR / "docs/figures/url_roc_pr.png",
}
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "f2", "roc_auc", "pr_auc")


class ReportUnavailable(ValueError):
    """A missing or inconsistent source must never turn into invented metrics."""


def _invalid_constant(value):
    raise ReportUnavailable()


def _read(key):
    path = REPORT_FILES[key]
    if path.stat().st_size > 2_000_000:
        raise ReportUnavailable()
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_invalid_constant)
    if not isinstance(value, dict):
        raise ReportUnavailable()
    return value


def _metrics(value):
    result = {}
    for name in METRIC_NAMES:
        score = value[name]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ReportUnavailable()
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ReportUnavailable()
        result[name] = score
    matrix = value["confusion_matrix"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
        or any(type(n) is not int or n < 0 for row in matrix for n in row)
    ):
        raise ReportUnavailable()
    return {**result, "confusion_matrix": matrix}


def _threshold(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
        raise ReportUnavailable()
    return value


def _information(meta):
    # Deliberately exclude input paths, arbitrary metadata and all corpus examples.
    return {
        key: meta.get(key)
        for key in ("version", "trained_at", "model_sha256", "versions", "seed", "fit_scope")
    }


def load_analytics():
    """Only fixed JSON sources. No arguments, fit, predict, joblib, raw SMS or URL reads."""
    try:
        data = {key: _read(key) for key in REPORT_FILES}
        selection, final, sm = (data[k] for k in ("sms_selection", "sms_test", "sms_meta"))
        url, um = data["url"], data["url_meta"]
        selected = selection["selected"]
        if not (
            selected["algorithm"] == final["algorithm"] == sm["algorithm"]
            and selected["parameter"] == sm["parameter"]
            and selection["model_sha256"] == final["model_sha256"] == sm["model_sha256"]
            and final["metrics"] == sm["test_metrics"]
            and selected["validation"] == sm["validation_metrics"]
            and selected["validation"]["threshold"] == sm["threshold"]
            and url["selected"] == um["selected"]
            and url["test"] == um["test"]
            and url["thresholds"] == um["thresholds"]
        ):
            raise ReportUnavailable()
        sms_counts = data["sms_features"]["counts"]
        sms_comparison = [
            {
                "algorithm": row["algorithm"],
                "parameters": {
                    "alpha" if row["algorithm"] == "MultinomialNB" else "C": row["parameter"]
                },
                "validation": _metrics(row["validation"]),
                "threshold": _threshold(row["validation"]["threshold"]),
                "fit_seconds": row["fit_seconds"],
            }
            for row in selection["comparison"]
        ]
        url_comparison = [
            {
                "algorithm": row["name"],
                "validation": _metrics(row["validation"]),
                "threshold": _threshold(row["validation"]["threshold"]),
                "fit_seconds": row["fit_seconds"],
            }
            for row in url["comparison"]
        ]
        winner = next(row for row in url_comparison if row["algorithm"] == url["selected"])
        sms_test, url_test = _metrics(final["metrics"]), _metrics(url["test"])
        # This checks report consistency, not another evaluation or prediction.
        if (
            sum(map(sum, sms_test["confusion_matrix"])) != sms_counts["test"]["rows"]
            or sum(map(sum, url_test["confusion_matrix"])) != url["counts"]["test"]["rows"]
        ):
            raise ReportUnavailable()
        t1, t2 = (_threshold(url["thresholds"][key]) for key in ("t1", "t2"))
        if t1 >= t2:
            raise ReportUnavailable()
        if (
            winner["threshold"] != t2
            or sm["feature_count"] != data["sms_features"]["matrix_shapes"]["train"][1]
        ):
            raise ReportUnavailable()
        return {
            "sms": {
                "targets_met": all(sms_test[k] >= 0.9 for k in ("precision", "recall", "f1")),
                "algorithm": sm["algorithm"],
                "parameters": {"C": sm["parameter"]},
                "thresholds": {"threshold": _threshold(sm["threshold"])},
                "dataset": {
                    "name": "UCI SMS Spam Collection",
                    "source": sm["source"]["source_page"],
                    "license": sm["source"]["license"],
                    "raw_rows": data["sms_raw"]["rows"],
                    "clean_rows": data["sms_clean"]["output_rows"],
                    "feature_count": sm["feature_count"],
                    "splits": sms_counts,
                    "labels": ["ham", "spam"],
                    "split_note": (
                        "Стратификация 70/15/15, seed=42. Дубликаты удалены до разбиения."
                    ),
                },
                "validation": _metrics(selected["validation"]),
                "test": sms_test,
                "comparison": sms_comparison,
                "information": {**_information(sm), "seed": selection["seed"]},
                "selection_note": (
                    "Выбор по максимуму validation F2 при Precision ≥ 0.85: "
                    f"{sm['algorithm']}, C={sm['parameter']}. "
                    "Test не участвовал в подборе."
                ),
                "limitations": (
                    "Только английский spam, не все виды fraud. Нет полного CV-подбора и ablation; "
                    "сохранена оценённая train-only модель без train+val refit. "
                    "В metadata SMS указан C; остальные параметры не восстанавливаются из кода."
                ),
            },
            "url": {
                "targets_met": (
                    url_test["precision"] >= 0.85
                    and url_test["recall"] >= 0.85
                    and url_test["roc_auc"] >= 0.9
                ),
                "algorithm": um.get("algorithm", url["selected"]),
                "configuration": url["selected"],
                "parameters": um.get("parameters", {}),
                "thresholds": {"t1": t1, "t2": t2},
                "dataset": {
                    "name": "PhiUSIIL Phishing URL Dataset",
                    "source": url["source"],
                    "license": url["license"],
                    "raw_rows": url["original_rows"],
                    "clean_rows": url["clean_rows_before_sampling"],
                    "sample_rows": url["sample_rows"],
                    "feature_count": len(url["features"]),
                    "splits": {
                        key: {
                            "rows": counts["rows"],
                            "classes": {
                                "0": counts["rows"] - counts["phishing"],
                                "1": counts["phishing"],
                            },
                        }
                        for key, counts in url["counts"].items()
                    },
                    "labels": ["legitimate", "phishing"],
                    "split_note": (
                        "GroupShuffleSplit по регистрируемым доменам, seed=42. "
                        "Группы не пересекаются; доли строк отличаются от 70/15/15."
                    ),
                },
                "validation": winner["validation"],
                "test": url_test,
                "comparison": url_comparison,
                "information": _information(um),
                "selection_note": (
                    f"{url['selected']} выбран по validation F2 при Precision ≥ 0.85 среди LR/RF. "
                    "Подбор t1/t2 выполнен на validation; test оценён однократно."
                ),
                "limitations": (
                    "Цели test не достигнуты. Возможны source/path bias и сдвиг доменных групп; "
                    "это гипотезы. Нет полного CV-подбора. "
                    "Только строковые признаки, без открытия сайтов."
                ),
            },
        }
    except (OSError, ValueError, KeyError, TypeError, StopIteration, OverflowError):
        raise ReportUnavailable(
            "Сохранённые ML-отчёты отсутствуют, повреждены или не согласованы"
        ) from None
