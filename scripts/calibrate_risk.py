"""Small risk weight/level search on existing English validation only, never custom_ru/test."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.analysis.social import matched_groups  # noqa: E402
from ml.data.load_sms import write_json  # noqa: E402
from ml.inference import ModelStore  # noqa: E402
from ml.preprocessing import URL  # noqa: E402
from ml.risk import compute_risk  # noqa: E402
from ml.sms_experiment import load_partition  # noqa: E402
from ml.url_features import extract_url_features  # noqa: E402


def main():
    frame = load_partition(ROOT, "val")
    selection = json.loads((ROOT / "docs/reports/07_selection.json").read_text(encoding="utf-8"))
    p_sms = np.load(ROOT / f"data/processed/07_val_{selection['selected']['algorithm']}.npy")
    store = ModelStore(ROOT / "ml/artifacts")
    samples = []
    for text in frame.text_raw:
        probabilities, flags = [], []
        for match in list(URL.finditer(text))[:3]:
            value = match.group().rstrip(".,!?;:)]}")
            try:
                flags.append(extract_url_features(value))
                probabilities.append(store.predict_url(value))
            except ValueError:
                pass
        samples.append((max(probabilities, default=None), matched_groups(text), flags))
    base = yaml.safe_load((ROOT / "risk_config.yaml").read_text())
    y = frame.label.to_numpy()
    candidates = []
    for weights in (
        {"sms": 0.5, "url": 0.3, "se": 0.2},
        {"sms": 0.7, "url": 0.2, "se": 0.1},
        {"sms": 0.6, "url": 0.3, "se": 0.1},
    ):
        config = copy.deepcopy(base)
        config["weights"] = weights
        scores = np.array(
            [
                compute_risk(
                    float(p), u, g, f, t2=store.url_meta["thresholds"]["t2"], configuration=config
                )["risk_score"]
                for p, (u, g, f) in zip(p_sms, samples, strict=True)
            ]
        )
        false_high = float(np.mean(scores[y == 0] >= 70))
        levels = [n for n in range(1, 36) if np.mean(scores[y == 1] >= n) >= 0.95]
        medium = max(levels) if levels else 1
        candidates.append(
            {
                "weights": weights,
                "medium": medium,
                "high": 70,
                "recall_medium_plus": float(np.mean(scores[y == 1] >= medium)),
                "false_high": false_high,
            }
        )
    eligible = [
        r for r in candidates if r["false_high"] <= 0.03 and r["recall_medium_plus"] >= 0.95
    ]
    chosen = max(eligible or candidates, key=lambda r: (r["medium"], -r["false_high"]))
    base["weights"] = chosen["weights"]
    base["levels"] = {"medium": chosen["medium"], "high": 70}
    (ROOT / "risk_config.yaml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    write_json(
        ROOT / "docs/reports/risk_calibration.json",
        {
            "scope": "English validation only; ru/test excluded",
            "candidates": candidates,
            "selected": chosen,
            "goals_met": bool(eligible),
            "note": (
                "Medium boundary calibrated from initial35; high70 fixed; "
                "Russian quality not calibrated"
            ),
        },
    )
    print(chosen)


if __name__ == "__main__":
    main()
