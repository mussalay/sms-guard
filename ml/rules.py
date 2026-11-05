"""Small en/ru phrase dictionary for the task-6 numeric group-count feature.

Not a fraud classifier: no probabilities, final verdicts or risk overrides.
The full explanation service/dictionary expansion belongs to task 12.
"""

import re
import unicodedata

GROUPS = {
    "URGENCY": (
        "urgent",
        "immediately",
        "act now",
        "срочно",
        "немедленно",
        "сегодня последний день",
    ),
    "MONEY_TRANSFER": (
        "transfer money",
        "send money",
        "payment",
        "переведите",
        "переведи",
        "оплатите",
    ),
    "BANK_CARD": ("bank", "card", "account number", "банк", "банка", "карта", "карты"),
    "ACCOUNT_BLOCK": (
        "blocked",
        "suspended",
        "account closed",
        "заблокирована",
        "блокировка",
        "отключим",
    ),
    "PRIZE": ("prize", "winner", "won", "выиграли", "выигрыш", "приз"),
    "CODE_REQUEST": (
        "send code",
        "tell us your password",
        "share your code",
        "назовите код",
        "сообщите код",
        "отправьте код",
        "назовите пароль",
    ),
}
PATTERNS = {
    name: re.compile(r"(?<!\w)(?:" + "|".join(re.escape(value) for value in phrases) + r")(?!\w)")
    for name, phrases in GROUPS.items()
}


def matched_groups(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return [name for name, pattern in PATTERNS.items() if pattern.search(normalized)]
