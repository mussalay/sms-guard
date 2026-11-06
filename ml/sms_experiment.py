"""Task 7: fixed train/validation selection. Holdout evaluation is a separate command."""

import argparse
import copy
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from ml.data.load_sms import ROOT, sha256, write_json
from ml.features import build_feature_pipeline

GRID = {
    "MultinomialNB": [0.1, 0.5, 1.0],
    "LogisticRegression": [0.1, 1.0, 10.0],
    "CalibratedLinearSVM": [0.1, 1.0, 10.0],
}
PROBES = [
    "Hello! See you at the library tomorrow.",
    "URGENT! Claim your cash prize now at https://reward.test/login",
]


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Qyzylorda")).isoformat()


def array_hash(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def metrics(y, probabilities, threshold: float) -> dict:
    y = np.asarray(y)
    probabilities = np.asarray(probabilities)
    if set(y) != {0, 1} or not np.isfinite(probabilities).all():
        raise ValueError("Нужны оба класса и конечные вероятности")
    if probabilities.shape != y.shape or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Некорректные вероятности")
    predicted = (probabilities >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=[0, 1], zero_division=0
    )
    pr_precision, pr_recall, _ = precision_recall_curve(y, probabilities)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision[1]),
        "recall": float(recall[1]),
        "f1": float(f1[1]),
        "f2": float(fbeta_score(y, predicted, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(auc(pr_recall, pr_precision)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "macro": {
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "f1": float(f1.mean()),
        },
        "by_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(("ham", "spam"))
        },
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "threshold": float(threshold),
    }


def choose_threshold(y, probabilities, min_precision=0.85) -> tuple[float, dict]:
    """F2 first, then recall/precision; deterministic threshold tie-break, validation only."""
    candidates = []
    for threshold in np.unique(np.r_[0.0, probabilities, 1.0]):
        predicted = np.asarray(probabilities) >= threshold
        precision = precision_score(y, predicted, zero_division=0)
        if precision >= min_precision:
            candidates.append(
                (
                    float(fbeta_score(y, predicted, beta=2, zero_division=0)),
                    float(recall_score(y, predicted)),
                    float(precision),
                    float(threshold),
                )
            )
    if not candidates:
        raise ValueError("На validation нет порога с Precision >= 0.85")
    _, _, _, threshold = max(candidates)
    return threshold, metrics(y, probabilities, threshold)


def make_model(algorithm: str, value: float):
    features = build_feature_pipeline().steps
    if algorithm == "MultinomialNB":
        return Pipeline(features + [("classifier", MultinomialNB(alpha=value))])
    if algorithm == "LogisticRegression":
        return Pipeline(
            features
            + [
                ("scale", MaxAbsScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=value,
                        class_weight="balanced",
                        max_iter=1000,
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        )
    if algorithm == "CalibratedLinearSVM":
        inner = Pipeline(
            features
            + [
                ("scale", MaxAbsScaler()),
                (
                    "classifier",
                    LinearSVC(
                        C=value,
                        class_weight="balanced",
                        dual=False,
                        max_iter=10000,
                        random_state=42,
                    ),
                ),
            ]
        )
        return CalibratedClassifierCV(
            inner,
            method="sigmoid",
            ensemble=False,
            n_jobs=1,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
        )
    raise ValueError("Неизвестный алгоритм")


def input_fingerprint(root: Path) -> dict:
    paths = [
        "data/processed/sms_clean.csv",
        "data/processed/sms_split.json",
        "data/processed/X_train.npz",
        "data/processed/X_val.npz",
        "ml/artifacts/sms_feature_pipeline.joblib",
        "ml/features.py",
        "ml/preprocessing.py",
        "ml/rules.py",
    ]
    return {path: sha256((root / path).read_bytes()) for path in paths}


def load_partition(root: Path, name: str):
    if name not in {"train", "val", "test"}:
        raise ValueError("Неизвестная выборка")
    split = json.loads((root / "data/processed/sms_split.json").read_text(encoding="utf-8"))
    source = root / "data/processed/sms_clean.csv"
    if sha256(source.read_bytes()) != split["clean_sha256"]:
        raise ValueError("Корпус не совпадает с сохранённым split")
    frame = pd.read_csv(source, keep_default_na=False)
    sets = [set(split["positions"][part]) for part in ("train", "val", "test")]
    if (
        sets[0] & sets[1]
        or sets[0] & sets[2]
        or sets[1] & sets[2]
        or set.union(*sets) != set(range(len(frame)))
        or not frame.text_clean.is_unique
    ):
        raise ValueError("Пересечение/повторы/неполное покрытие split")
    part = frame.iloc[split["positions"][name]].copy()
    if part.id.tolist() != split["source_ids"][name]:
        raise ValueError("Исходные id изменились")
    return part


def fit_candidates(root: Path, grid=None):
    """Only train/val are loaded. No heldout matrix, Russian corpus, or test metrics."""
    train, val = load_partition(root, "train"), load_partition(root, "val")
    feature_model = joblib.load(root / "ml/artifacts/sms_feature_pipeline.joblib")
    X_train = sparse.load_npz(root / "data/processed/X_train.npz")
    X_val = sparse.load_npz(root / "data/processed/X_val.npz")
    report6 = json.loads((root / "docs/reports/06_features.json").read_text(encoding="utf-8"))
    for name, matrix, data in (("train", X_train, train), ("val", X_val, val)):
        if (
            sha256((root / f"data/processed/X_{name}.npz").read_bytes())
            != report6["matrix_sha256"][name]
        ):
            raise ValueError("Матрица признаков изменена")
        if (
            matrix.shape[0] != len(data)
            or not np.isfinite(matrix.data).all()
            or (matrix.data < 0).any()
        ):
            raise ValueError("Нужна конечная неотрицательная sparse-матрица")
        difference = feature_model.transform(data.text_raw.tolist()) - matrix
        if difference.nnz and np.max(np.abs(difference.data)) > 1e-12:
            raise ValueError("Признаки не соответствуют исходному pipeline")
    rows, models, predictions = [], {}, {}
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for algorithm, values in (grid or GRID).items():
            for value in values:
                print(f"train/validation: {algorithm}, parameter={value}", flush=True)
                model = make_model(algorithm, value)
                start = time.perf_counter()
                if algorithm == "CalibratedLinearSVM":
                    # Complete feature pipeline is inside calibration CV: fit on each fold only.
                    model.fit(train.text_raw.tolist(), train.label.to_numpy())
                else:
                    # Reuse the immutable task-6 fit, never refit it on validation/test.
                    fitted_features = copy.deepcopy(feature_model.named_steps["features"])
                    model.steps[0] = ("features", fitted_features)
                    if "scale" in model.named_steps:
                        scaled = model.named_steps["scale"].fit_transform(X_train)
                    else:
                        scaled = X_train
                    model.named_steps["classifier"].fit(scaled, train.label.to_numpy())
                elapsed = time.perf_counter() - start
                probabilities = model.predict_proba(val.text_raw.tolist())[:, 1]
                threshold, scores = choose_threshold(val.label.to_numpy(), probabilities)
                row = {
                    "algorithm": algorithm,
                    "parameter": value,
                    "validation": scores,
                    "fit_seconds": elapsed,
                    "validation_probabilities_sha256": array_hash(probabilities),
                }
                rows.append(row)
                key = (algorithm, value)
                models[key], predictions[key] = model, probabilities
    return rows, models, predictions


def row_key(row):
    score = row["validation"]
    return score["f2"], score["recall"], score["precision"], -row["parameter"], row["algorithm"]


def select(root: Path = ROOT):
    reports, artifacts = root / "docs/reports", root / "ml/artifacts"
    if (reports / "07_test_claim.json").exists():
        raise RuntimeError("Test уже зарезервирован/оценён; новый подбор запрещён")
    before = input_fingerprint(root)
    rows, models, probabilities = fit_candidates(root)
    selected = max(rows, key=row_key)
    comparison = [max([r for r in rows if r["algorithm"] == name], key=row_key) for name in GRID]
    key = (selected["algorithm"], selected["parameter"])
    joblib.dump(models[key], artifacts / "sms_model.joblib")
    # Task-7 baseline explicitly remains NB(alpha=1), independently of winning model.
    baseline_key = ("MultinomialNB", 1.0)
    joblib.dump(models[baseline_key], artifacts / "sms_baseline.joblib")
    selection = {
        "created_at": now(),
        "version": "1.0.0",
        "seed": 42,
        "protocol": "max validation F2 with Precision >= .85; train-only calibration",
        "grid": GRID,
        "candidates": rows,
        "comparison": comparison,
        "selected": selected,
        "inputs": before,
        "model_sha256": sha256((artifacts / "sms_model.joblib").read_bytes()),
        "baseline_sha256": sha256((artifacts / "sms_baseline.joblib").read_bytes()),
        "test_used": False,
        "russian_corpus_used": False,
        "probability_hash": array_hash(probabilities[key]),
    }
    if before != input_fingerprint(root):
        raise AssertionError("Входные артефакты изменились во время обучения")
    for row in comparison:
        name, value = row["algorithm"], row["parameter"]
        np.save(root / f"data/processed/07_val_{name}.npy", probabilities[(name, value)])
    write_json(reports / "07_selection.json", selection)
    metadata = {
        "version": "1.0.0",
        "trained_at": selection["created_at"],
        "algorithm": selected["algorithm"],
        "parameter": selected["parameter"],
        "threshold": selected["validation"]["threshold"],
        "languages": ["en"],
        "labels": {"0": "ham", "1": "spam"},
        "feature_count": 59924,
        "validation_metrics": selected["validation"],
        "test_metrics": None,
        "model_sha256": selection["model_sha256"],
        "inputs": before,
        "source": json.loads((root / "docs/datasets/uci_source.json").read_text(encoding="utf-8")),
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "fit_scope": "train only, no train+val refit",
        "limitations": [
            "English spam detection, not guaranteed fraud detection",
            "Russian ML unsupported; kk excluded by DEV-01",
        ],
    }
    write_json(artifacts / "sms_model_meta.json", metadata)
    write_json(
        artifacts / "sms_baseline_meta.json",
        {
            "algorithm": "MultinomialNB",
            "alpha": 1.0,
            "threshold": 0.5,
            "model_sha256": selection["baseline_sha256"],
            "version": "1.0.0",
            "validation_metrics": metrics(
                load_partition(root, "val").label.to_numpy(), probabilities[baseline_key], 0.5
            ),
        },
    )
    return selection


def verify_reproduction(root: Path = ROOT):
    selection = json.loads((root / "docs/reports/07_selection.json").read_text(encoding="utf-8"))
    if input_fingerprint(root) != selection["inputs"]:
        raise ValueError("Входные файлы отличаются от selection")
    rows, _, _ = fit_candidates(root)
    winner = max(rows, key=row_key)
    equal = (winner["algorithm"], winner["parameter"], winner["validation"]) == (
        selection["selected"]["algorithm"],
        selection["selected"]["parameter"],
        selection["selected"]["validation"],
    )
    for old, new in zip(selection["candidates"], rows, strict=True):
        equal = (
            equal
            and old["validation_probabilities_sha256"] == new["validation_probabilities_sha256"]
        )
    if not equal:
        raise AssertionError("Train/validation воспроизведение отличается")
    result = {
        "verified_at": now(),
        "identical_candidate_probabilities": True,
        "identical_selection_and_threshold": True,
        "test_accessed": False,
    }
    write_json(root / "docs/reports/07_reproduction.json", result)
    return result


def finalize(root: Path = ROOT):
    reports, artifacts = root / "docs/reports", root / "ml/artifacts"
    selection = json.loads((reports / "07_selection.json").read_text(encoding="utf-8"))
    if input_fingerprint(root) != selection["inputs"]:
        raise ValueError("Входные файлы изменились после выбора модели")
    model_path = artifacts / "sms_model.joblib"
    if sha256(model_path.read_bytes()) != selection["model_sha256"]:
        raise ValueError("Модель изменена после selection")
    # Exclusive durable reservation BEFORE even loading the test partition.
    with (reports / "07_test_claim.json").open("x", encoding="utf-8") as claim:
        json.dump(
            {
                "claimed_at": now(),
                "model_sha256": selection["model_sha256"],
                "threshold": selection["selected"]["validation"]["threshold"],
            },
            claim,
            indent=2,
        )
    test = load_partition(root, "test")
    model = joblib.load(model_path)
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        probabilities = model.predict_proba(test.text_raw.tolist())[:, 1]  # ONE final test pass.
        elapsed = time.perf_counter() - start
    threshold = selection["selected"]["validation"]["threshold"]
    scores = metrics(test.label.to_numpy(), probabilities, threshold)
    result = {
        "evaluated_at": now(),
        "model_sha256": selection["model_sha256"],
        "algorithm": selection["selected"]["algorithm"],
        "metrics": scores,
        "rows": len(test),
        "test_probability_sha256": array_hash(probabilities),
        "batch_prediction_seconds": elapsed,
        "batch_ms_per_sms": elapsed * 1000 / len(test),
        "targets_met": all(scores[name] >= 0.90 for name in ("precision", "recall", "f1")),
        "no_refit_or_test_tuning": True,
    }
    write_json(reports / "07_final_test.json", result)
    np.save(root / "data/processed/07_test_probabilities.npy", probabilities)
    metadata_path = artifacts / "sms_model_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["test_metrics"] = scores
    metadata["test_evaluated_at"] = result["evaluated_at"]
    write_json(metadata_path, metadata)
    write_json(reports / "07_model_metadata.json", metadata)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["select", "verify", "finalize"])
    action = parser.parse_args().action
    result = {"select": select, "verify": verify_reproduction, "finalize": finalize}[action]()
    print(json.dumps(result.get("selected", result), ensure_ascii=False, indent=2))
