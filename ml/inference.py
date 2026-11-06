"""Load trusted local models once per process; no network at inference."""

import json
import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from ml.data.load_sms import ROOT, sha256
from ml.url_features import model_features


class ModelStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.sms = self.url = None
        self.sms_meta = self.url_meta = {}
        for kind in ("sms", "url"):
            model = self.directory / f"{kind}_model.joblib"
            meta = self.directory / f"{kind}_model_meta.json"
            if model.exists() and meta.exists():
                metadata = json.loads(meta.read_text(encoding="utf-8"))
                if (
                    metadata.get("model_sha256")
                    and sha256(model.read_bytes()) != metadata["model_sha256"]
                ):
                    raise ValueError("Контрольная сумма модели не совпадает")
                setattr(self, kind, joblib.load(model))
                setattr(self, kind + "_meta", metadata)

    def predict_sms(self, text: str, language="en") -> dict:
        if not isinstance(text, str) or not text.strip() or len(text) > 1000:
            raise ValueError("Текст должен содержать от 1 до 1000 символов")
        if language != "en" or re.search(r"[А-Яа-яЁё]", text):
            raise ValueError("Русский ML не поддерживается; используйте правила и URL-анализ")
        if self.sms is None:
            raise RuntimeError("SMS-модель недоступна")
        probability = float(self.sms.predict_proba([text])[0, 1])
        return {"label": int(probability >= self.sms_meta["threshold"]), "probability": probability}

    def predict_url(self, url: str) -> float:
        if self.url is None:
            raise RuntimeError("URL-модель недоступна")
        return float(self.url.predict_proba(np.asarray([model_features(url)]))[0, 1])


@lru_cache(maxsize=4)
def get_store(directory=ROOT / "ml/artifacts"):
    return ModelStore(Path(directory))


def predict_sms(text: str, *, language="en") -> dict:
    return get_store().predict_sms(text, language)
