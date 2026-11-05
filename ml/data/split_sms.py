"""Deterministic pre-classifier split, with source IDs and positions recorded."""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_sms(frame: pd.DataFrame, random_state: int = 42) -> dict[str, list[int]]:
    if not {"id", "text_clean", "label"} <= set(frame.columns):
        raise ValueError("Нужен очищенный UCI с id/text_clean/label")
    if not frame.id.is_unique or not frame.text_clean.is_unique:
        raise ValueError("Удалите дубликаты до разбиения")
    if frame.text_clean.isna().any() or frame.text_clean.str.strip().eq("").any():
        raise ValueError("Пустой текст до разбиения")
    if set(frame.label) != {0, 1}:
        raise ValueError("Нужны оба класса 0/1")
    positions = list(range(len(frame)))
    train, remainder = train_test_split(
        positions,
        test_size=0.30,
        random_state=random_state,
        stratify=frame.label,
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=random_state,
        stratify=frame.iloc[remainder].label,
    )
    return {"train": sorted(train), "val": sorted(validation), "test": sorted(test)}
