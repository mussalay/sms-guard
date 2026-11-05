"""Descriptive EDA only: no fitting, model selection or access to the Russian test set."""

from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ml.preprocessing import load_stopwords

TOKENIZER = RegexpTokenizer(r"<[a-z]+>|[a-z]+(?:'[a-z]+)?")
LABELS = {0: "ham (обычные)", 1: "spam (спам)"}
COLORS = ["#276b8e", "#b34034"]


def content_tokens(text: str) -> list[str]:
    excluded = load_stopwords("en")
    return [word for word in TOKENIZER.tokenize(text) if word not in excluded]


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if frame.empty or not {"text_raw", "text_clean", "label"} <= set(frame.columns):
        raise ValueError("Нужен непустой очищенный SMS-корпус")
    work = frame.copy()
    work["chars_raw"] = work.text_raw.str.len()
    work["words_raw"] = work.text_raw.str.split().str.len()
    work["chars_clean"] = work.text_clean.str.len()
    work["words_clean"] = work.text_clean.str.split().str.len()
    for name in ("url", "phone", "money"):
        work["has_" + name] = work.text_clean.str.contains("<" + name + ">", regex=False)
    work["has_pressure"] = work.text_clean.str.contains("!", regex=False)
    result = {"rows": len(work), "scope": "all_cleaned_UCI_before_training_no_parameter_selection"}
    classes = {}
    for label in (0, 1):
        subset = work.loc[work.label == label]
        documents = subset.text_clean.map(content_tokens)
        words = Counter(word for tokens in documents for word in tokens)
        bigrams = Counter(
            " ".join(pair) for tokens in documents for pair in zip(tokens, tokens[1:])
        )
        classes[str(label)] = {
            "count": len(subset),
            "percentage": 100 * len(subset) / len(work),
            "mean_chars_raw": float(subset.chars_raw.mean()),
            "median_chars_raw": float(subset.chars_raw.median()),
            "mean_words_raw": float(subset.words_raw.mean()),
            "mean_chars_clean": float(subset.chars_clean.mean()),
            "mean_words_clean": float(subset.words_clean.mean()),
            "indicator_percentages": {
                name: float(100 * subset["has_" + name].mean())
                for name in ("url", "phone", "money", "pressure")
            },
            "top20_words": words.most_common(20),
            "top20_bigrams": bigrams.most_common(20),
        }
    result["classes"] = classes
    return work, result


def plot_eda(frame: pd.DataFrame, summary: dict, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        fig.tight_layout()
        path = directory / name
        fig.savefig(path, dpi=150, metadata={"Software": "SMS Guard EDA"})
        plt.close(fig)
        paths.append(path)

    fig, axis = plt.subplots(figsize=(8, 5))
    counts = [summary["classes"][str(k)]["count"] for k in (0, 1)]
    bars = axis.bar([LABELS[k] for k in (0, 1)], counts, color=COLORS)
    axis.bar_label(bars, labels=[f"{n} ({100 * n / len(frame):.2f}%)" for n in counts], padding=5)
    axis.set(xlabel="Класс", ylabel="Число сообщений", title="Баланс классов после очистки UCI")
    axis.set_ylim(0, max(counts) * 1.15)
    save(fig, "05_class_balance.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, column, label in zip(
        axes,
        ("chars_raw", "words_raw"),
        ("Символов в исходном SMS", "Слов в исходном SMS"),
        strict=True,
    ):
        axis.boxplot(
            [frame.loc[frame.label == k, column] for k in (0, 1)],
            tick_labels=[LABELS[k] for k in (0, 1)],
        )
        axis.set(xlabel="Класс", ylabel=label, title=f"Распределение длины: {label.lower()}")
    save(fig, "05_message_lengths.png")

    for key, filename, title in (
        ("top20_words", "05_top_words.png", "Топ-20 слов без стоп-слов"),
        ("top20_bigrams", "05_top_bigrams.png", "Топ-20 биграмм без стоп-слов"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(15, 9))
        for k, axis in enumerate(axes):
            values = summary["classes"][str(k)][key]
            axis.barh(
                [word for word, _ in values][::-1], [n for _, n in values][::-1], color=COLORS[k]
            )
            axis.set(
                xlabel="Число вхождений", ylabel="Слово / биграмма", title=f"{title}: {LABELS[k]}"
            )
        save(fig, filename)

    fig, axis = plt.subplots(figsize=(10, 5))
    indicators = ("url", "phone", "money", "pressure")
    positions = np.arange(len(indicators))
    for k in (0, 1):
        rates = summary["classes"][str(k)]["indicator_percentages"]
        bars = axis.bar(
            positions + (k - 0.5) * 0.35,
            [rates[x] for x in indicators],
            width=0.35,
            label=LABELS[k],
            color=COLORS[k],
        )
        axis.bar_label(bars, fmt="%.1f", padding=3)
    axis.set_xticks(positions, ["<url>", "<phone>", "<money>", "!"])
    axis.set(xlabel="Признак", ylabel="Доля сообщений, %", title="Признаки сущностей и давления")
    axis.set_ylim(0, 105)
    axis.legend(title="Класс")
    save(fig, "05_entity_rates.png")
    return paths


def observations(summary: dict) -> list[str]:
    ham, spam = summary["classes"]["0"], summary["classes"]["1"]
    facts = [
        f"После очистки {summary['rows']} SMS: ham={ham['count']} ({ham['percentage']:.2f}%), "
        f"spam={spam['count']} ({spam['percentage']:.2f}%).",
        f"Средняя длина исходного текста: ham {ham['mean_chars_raw']:.2f} символа, "
        f"spam {spam['mean_chars_raw']:.2f}; медианы {ham['median_chars_raw']:.0f} "
        f"и {spam['median_chars_raw']:.0f} соответственно.",
        f"Среднее число слов: ham {ham['mean_words_raw']:.2f}, spam {spam['mean_words_raw']:.2f}.",
    ]
    for name, title in (
        ("url", "URL"),
        ("phone", "Телефоны"),
        ("money", "Суммы"),
        ("pressure", "Восклицательный знак"),
    ):
        facts.append(
            f"{title}: ham {ham['indicator_percentages'][name]:.2f}%, "
            f"spam {spam['indicator_percentages'][name]:.2f}% сообщений."
        )
    facts.append(
        "Это описательные различия данного корпуса, не метрики классификатора и не причинные связи."
    )
    return facts
