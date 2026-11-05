"""Reproduce task 6 from cleaned UCI only; no classifier or test metrics."""

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse

from ml.data.load_sms import ROOT, sha256, write_json
from ml.data.split_sms import split_sms
from ml.features import build_feature_pipeline


def top_tfidf(pipeline, matrix, labels: np.ndarray) -> dict:
    # Manual magnitudes must not be mixed with TF-IDF rankings.
    union = pipeline.named_steps["features"]
    names = union.get_feature_names_out()
    indexes = np.array([i for i, name in enumerate(names) if not name.startswith("manual__")])
    result = {}
    for label in (0, 1):
        means = np.asarray(matrix[labels == label][:, indexes].mean(axis=0)).ravel()
        # Stable tie-breaking by feature name.
        ordered = sorted(range(len(indexes)), key=lambda i: (-means[i], names[indexes[i]]))[:15]
        result[str(label)] = [
            {"feature": str(names[indexes[i]]), "mean_tfidf": float(means[i])} for i in ordered
        ]
    return result


def prepare_features(root: Path = ROOT) -> dict:
    processed = root / "data/processed"
    source = processed / "sms_clean.csv"
    audit = json.loads((root / "docs/reports/04_preprocessing.json").read_text(encoding="utf-8"))
    digest = sha256(source.read_bytes())
    if digest != audit["clean_sha256"]:
        raise ValueError("Очищенный корпус изменён: сначала повторите очистку и аудит")
    frame = pd.read_csv(source, keep_default_na=False)
    split = split_sms(frame)
    manifest = {
        "dataset": "UCI SMS Spam Collection (English)",
        "clean_sha256": digest,
        "seed": 42,
        "method": "two stratified train_test_split: holdout=0.30, then half",
        "positions": split,
        "source_ids": {name: frame.iloc[indexes].id.tolist() for name, indexes in split.items()},
        "counts": {
            name: {
                "rows": len(indexes),
                "classes": {str(k): int((frame.iloc[indexes].label == k).sum()) for k in (0, 1)},
            }
            for name, indexes in split.items()
        },
        "excluded": "custom_ru.csv: test-only, never read by this preparation",
    }
    write_json(processed / "sms_split.json", manifest)
    pipeline = build_feature_pipeline()
    # The ONLY fitting call. Labels are not supplied to a classifier (none exists).
    train = frame.iloc[split["train"]]
    matrices = {"train": pipeline.fit_transform(train.text_raw.tolist())}
    for name in ("val", "test"):
        matrices[name] = pipeline.transform(frame.iloc[split[name]].text_raw.tolist())
    for name, matrix in matrices.items():
        sparse.save_npz(processed / f"X_{name}.npz", matrix.tocsr())
        np.save(processed / f"y_{name}.npy", frame.iloc[split[name]].label.to_numpy())
    names = pipeline.get_feature_names_out().tolist()
    write_json(processed / "sms_feature_names.json", {"features": names})
    artifacts = root / "ml/artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    pipeline_path = artifacts / "sms_feature_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    reloaded = joblib.load(pipeline_path)
    reference = pipeline.transform(train.text_raw.iloc[:5].tolist())
    restored = reloaded.transform(train.text_raw.iloc[:5].tolist())
    if (reference != restored).nnz:
        raise AssertionError("Pipeline изменился после сохранения")
    report = {
        "dataset_sha256": digest,
        "seed": 42,
        "counts": manifest["counts"],
        "matrix_shapes": {name: list(matrix.shape) for name, matrix in matrices.items()},
        "matrix_nnz": {name: matrix.nnz for name, matrix in matrices.items()},
        "matrix_sha256": {
            name: sha256((processed / f"X_{name}.npz").read_bytes()) for name in matrices
        },
        "feature_counts": {
            prefix: sum(name.startswith(prefix + "__") for name in names)
            for prefix in ("word", "char", "manual")
        },
        "top15_train_tfidf_by_class": top_tfidf(
            pipeline, matrices["train"], train.label.to_numpy()
        ),
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "pipeline_reload_equal": True,
        "classifier_trained": False,
        "fit_scope": "train only; val/test transform only; Russian corpus excluded",
    }
    write_json(root / "docs/reports/06_features.json", report)
    lines = [
        "# Задание 6 — фактические размеры и признаки",
        "",
        "Fit только на train; val/test только transform. Классификатор не обучался.",
        "",
        "| Часть | Строк | ham | spam | Размер матрицы |",
        "|---|---:|---:|---:|---|",
    ]
    for name, counts in manifest["counts"].items():
        lines.append(
            f"| {name} | {counts['rows']} | {counts['classes']['0']} | "
            f"{counts['classes']['1']} | {report['matrix_shapes'][name]} |"
        )
    lines.extend(["", f"Число признаков: {report['feature_counts']}", ""])
    for label, values in report["top15_train_tfidf_by_class"].items():
        lines.extend(
            [
                f"## Топ-15 по среднему TF-IDF на train, класс {label}",
                "",
                "| Признак | Средний TF-IDF |",
                "|---|---:|",
            ]
        )
        lines.extend(f"| `{value['feature']}` | {value['mean_tfidf']:.8f} |" for value in values)
        lines.append("")
    (root / "docs/06_feature_results.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = prepare_features()
    print(
        json.dumps(
            {key: result[key] for key in ("counts", "matrix_shapes", "feature_counts")}, indent=2
        )
    )
