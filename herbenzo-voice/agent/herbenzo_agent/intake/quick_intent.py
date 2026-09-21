"""Deterministic yes/no detection for confirmation prompts (consent, readback, restart).

Plain "yes"/"no" answers are the most common turns and must never fail because an LLM timed out.
Only messages made entirely of confirmation words are handled here; anything else goes to the LLM.
"""

from __future__ import annotations

import re

from herbenzo_agent.intake.models import Intent

_WORD = re.compile(r"[a-z']+")

_YES_CORE = {
    "yes",
    "yeah",
    "yep",
    "yup",
    "sure",
    "ok",
    "okay",
    "correct",
    "right",
    "fine",
    "haan",
    "han",
    "ji",
    "absolutely",
    "definitely",
}
_YES_FILLER = {
    "go",
    "ahead",
    "that's",
    "thats",
    "that",
    "is",
    "all",
    "it's",
    "its",
    "please",
    "continue",
    "sounds",
    "good",
    "of",
    "course",
    "thank",
    "thanks",
    "you",
    "everything",
    "perfect",
    "do",
    "can",
    "we",
}
_NO_CORE = {"no", "nope", "nah", "wrong", "incorrect", "nahi", "not"}
_NO_FILLER = {
    "that's",
    "thats",
    "that",
    "is",
    "isn't",
    "isnt",
    "it's",
    "its",
    "quite",
    "right",
    "correct",
    "thanks",
    "thank",
    "you",
    "i",
    "don't",
    "dont",
    "want",
    "to",
    "all",
    "sorry",
}


def quick_confirmation(text: str) -> Intent | None:
    words = _WORD.findall(text.lower().replace("’", "'"))
    if not words or len(words) > 8:
        return None
    vocabulary = set(words)
    if vocabulary & _NO_CORE and vocabulary <= (_NO_CORE | _NO_FILLER):
        return Intent.confirm_no
    if vocabulary & _YES_CORE and vocabulary <= (_YES_CORE | _YES_FILLER):
        return Intent.confirm_yes
    return None
