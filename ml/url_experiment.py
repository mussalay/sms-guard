"""Small offline PhiUSIIL experiment with registrable-domain group holdouts."""

import json
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.data.load_sms import ROOT, sha256, write_json
from ml.sms_experiment import choose_threshold, metrics, now
from ml.url_features import FEATURE_NAMES, model_features, model_url, registrable_domain


def run(root: Path = ROOT):
    report_path = root / "docs/reports/url_experiment.json"
    if report_path.exists() or (root / "docs/reports/url_test_claim.json").exists():
        raise RuntimeError("URL test уже оценён: повторный подбор запрещён")
    archive = root / "data/raw/phiusiil.zip"
    with zipfile.ZipFile(archive) as zipped:
        name = next(n for n in zipped.namelist() if n.endswith(".csv"))
        with zipped.open(name) as source:
            frame = pd.read_csv(source, usecols=["URL", "label"])
    original = len(frame)
    frame = frame.dropna().copy()
    frame["label"] = 1 - frame.label.astype(int)  # UCI: 1 legitimate, 0 phishing
    missing, normalized = 0, []
    for url in frame.URL:
        try:
            normalized.append(model_url(url))
        except ValueError:
            normalized.append(None)
            missing += 1
    frame["canonical"] = normalized
    frame = frame.dropna(subset=["canonical"])
    conflicts = frame.groupby("canonical").label.nunique()
    frame = frame[~frame.canonical.isin(conflicts[conflicts > 1].index)]
    dedup_before = len(frame)
    frame = frame.drop_duplicates("canonical")
    duplicates_removed = dedup_before - len(frame)
    clean_rows_before_sampling = len(frame)
    if len(frame) > 50000:
        frame, _ = train_test_split(frame, train_size=50000, random_state=42, stratify=frame.label)
    frame = frame.reset_index(drop=True)
    groups = frame.canonical.map(registrable_domain)
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train, holdout = next(splitter.split(frame, frame.label, groups))
    val_local, test_local = next(
        GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42).split(
            frame.iloc[holdout], frame.iloc[holdout].label, groups.iloc[holdout]
        )
    )
    val, test = holdout[val_local], holdout[test_local]
    parts = {"train": train, "val": val, "test": test}
    assert not (set(groups.iloc[train]) & set(groups.iloc[test]))
    assert not (set(groups.iloc[val]) & set(groups.iloc[test]))
    print(f"URL: {len(frame)} rows, extracting 22 offline features", flush=True)
    X = np.asarray([model_features(url) for url in frame.canonical], dtype=float)
    y = frame.label.to_numpy()
    features = pd.DataFrame(X, columns=FEATURE_NAMES)
    features["label"] = y
    features.to_parquet(root / "data/processed/url_features.parquet", index=False)
    write_json(
        root / "data/processed/url_split.json",
        {
            "seed": 42,
            "positions": {k: v.tolist() for k, v in parts.items()},
            "groups": groups.tolist(),
            "archive_sha256": sha256(archive.read_bytes()),
        },
    )
    configs = [
        (
            "LR_C1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1, max_iter=1000, class_weight="balanced", random_state=42),
            ),
        ),
        (
            "LR_C10",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=10, max_iter=1000, class_weight="balanced", random_state=42),
            ),
        ),
        (
            "RF_depth12",
            RandomForestClassifier(
                n_estimators=150, max_depth=12, class_weight="balanced", n_jobs=2, random_state=42
            ),
        ),
        (
            "RF_depth24",
            RandomForestClassifier(
                n_estimators=150, max_depth=24, class_weight="balanced", n_jobs=2, random_state=42
            ),
        ),
    ]
    rows, models = [], {}
    for name, model in configs:
        print("URL train/validation: " + name, flush=True)
        start = time.perf_counter()
        model.fit(X[train], y[train])
        p = model.predict_proba(X[val])[:, 1]
        threshold, scores = choose_threshold(y[val], p)
        rows.append(
            {"name": name, "validation": scores, "fit_seconds": time.perf_counter() - start}
        )
        models[name] = model
    selected = max(rows, key=lambda r: (r["validation"]["f2"], r["validation"]["precision"]))
    model = models[selected["name"]]
    t2 = selected["validation"]["threshold"]
    # t1 chosen using validation recall >=.95; never test-tuned.
    pval = model.predict_proba(X[val])[:, 1]
    t1 = min(t2 * 0.5, float(np.quantile(pval[y[val] == 1], 0.05)))
    joblib.dump(model, root / "ml/artifacts/url_model.joblib")
    with (root / "docs/reports/url_test_claim.json").open("x", encoding="utf-8") as target:
        json.dump({"at": now(), "selected": selected["name"], "t1": t1, "t2": t2}, target)
    start = time.perf_counter()
    ptest = model.predict_proba(X[test])[:, 1]  # one final holdout evaluation
    test_metrics = metrics(y[test], ptest, t2)
    result = {
        "source": "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset",
        "license": "CC BY 4.0",
        "downloaded_on": "2026-11-06",
        "archive_sha256": sha256(archive.read_bytes()),
        "original_rows": original,
        "invalid": missing,
        "duplicates_removed": duplicates_removed,
        "clean_rows_before_sampling": clean_rows_before_sampling,
        "sample_rows": len(frame),
        "counts": {k: {"rows": len(v), "phishing": int(y[v].sum())} for k, v in parts.items()},
        "domain_disjoint": True,
        "normalization": "scheme=http, strip www for BOTH classes and inference",
        "features": FEATURE_NAMES,
        "comparison": rows,
        "selected": selected["name"],
        "thresholds": {"t1": t1, "t2": t2},
        "test": test_metrics,
        "prediction_ms_per_url": 1000 * (time.perf_counter() - start) / len(test),
        "trained_at": now(),
        "version": "1.0.0",
        "eda_means_by_class": features.groupby("label")[FEATURE_NAMES[:6]].mean().to_dict(),
        "limitations": [
            "URL strings only; no webpage/DNS data",
            "Remaining source/path distribution bias possible",
            "flagged TLD list is an unvalidated heuristic",
        ],
    }
    write_json(report_path, result)
    write_json(root / "ml/artifacts/url_model_meta.json", result)
    print(json.dumps({"selected": result["selected"], "test": test_metrics}, indent=2))


if __name__ == "__main__":
    run()
