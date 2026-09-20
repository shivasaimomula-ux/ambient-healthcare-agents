"""Deterministic checks on responder output before it can be spoken.

Any failure means the base question is used instead, so a misbehaving LLM can only ever make the
agent sound plainer, never unsafe.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

BLOCKED_TERMS_PATH = Path(__file__).resolve().parent.parent / "data_files" / "blocked_terms.txt"

MAX_WORDS = 45
_MARKDOWN = re.compile(r"[*#`_>|\[\]]|^\s*[-•]\s", re.MULTILINE)
_EMOJI = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f2ff]")
_DIGITS = re.compile(r"\d+")
_CONTENT_WORD = re.compile(r"[a-z]{3,}")
_STOPWORDS = frozenset(
    [
        "the",
        "and",
        "are",
        "you",
        "your",
        "for",
        "any",
        "this",
        "that",
        "with",
        "have",
        "has",
        "been",
        "how",
        "what",
        "who",
        "does",
        "did",
        "can",
        "could",
        "would",
        "there",
        "them",
        "they",
        "their",
        "right",
        "now",
        "like",
        "also",
        "say",
        "want",
        "about",
        "into",
        "from",
        "its",
        "it's",
    ]
)
MIN_QUESTION_OVERLAP = 0.5


@lru_cache
def blocked_patterns() -> tuple[re.Pattern, ...]:
    lines = BLOCKED_TERMS_PATH.read_text().splitlines()
    return tuple(
        re.compile(line.strip(), re.IGNORECASE) for line in lines if line.strip() and not line.startswith("#")
    )


def blocked_terms_sha256() -> str:
    return hashlib.sha256(BLOCKED_TERMS_PATH.read_bytes()).hexdigest()


def _content_words(text: str) -> set[str]:
    return {w for w in _CONTENT_WORD.findall(text.lower()) if w not in _STOPWORDS}


def lint_response(text: str, user_text: str = "", base_question: str | None = None) -> list[str]:
    """Return a list of problems; empty means the text may be spoken.

    With `base_question`, at least half of its content words must survive the rephrasing, so the
    responder cannot quietly change what is being asked.
    """
    problems: list[str] = []
    if not text.strip():
        return ["empty"]
    if text.count("?") != 1:
        problems.append("must_contain_exactly_one_question")
    if len(text.split()) > MAX_WORDS:
        problems.append("too_long")
    if _MARKDOWN.search(text):
        problems.append("markdown")
    if _EMOJI.search(text):
        problems.append("emoji")
    if any(ch in text for ch in '"“”:'):
        problems.append("quote_or_colon")
    if base_question:
        expected = _content_words(base_question)
        if expected and len(expected & _content_words(text)) / len(expected) < MIN_QUESTION_OVERLAP:
            problems.append("question_meaning_changed")
    for pattern in blocked_patterns():
        if pattern.search(text):
            problems.append(f"blocked_term:{pattern.pattern}")
    user_numbers = set(_DIGITS.findall(user_text))
    stray = [n for n in _DIGITS.findall(text) if n not in user_numbers]
    if stray:
        problems.append("number_not_from_user")
    return problems
