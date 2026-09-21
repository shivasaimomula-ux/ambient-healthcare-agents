"""Emergency (red-flag) screening.

Layer 1 is deterministic rules from `data_files/red_flags.yaml` and runs on every user turn
before guardrails or any LLM. Layer 2 merges codes suggested by the model, restricted to the
lexicon. Either layer firing escalates the session.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from herbenzo_agent.contracts.symptom_spec import RedFlag

LEXICON_PATH = Path(__file__).resolve().parent.parent / "data_files" / "red_flags.yaml"

NEGATION_CUES = frozenset(
    {
        "no",
        "not",
        "never",
        "without",
        "dont",
        "don't",
        "doesnt",
        "doesn't",
        "didnt",
        "didn't",
        "denies",
        "nor",
    }
)
NEGATION_WINDOW = 3
_WORD = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class FlagRule:
    code: str
    negation_aware: bool
    patterns: tuple[re.Pattern, ...]


@dataclass(frozen=True)
class Lexicon:
    version: str
    sha256: str
    rules: tuple[FlagRule, ...]
    trigger_keywords: tuple[str, ...]

    @property
    def codes(self) -> frozenset[str]:
        return frozenset(r.code for r in self.rules)


@lru_cache
def load_lexicon(path: Path = LEXICON_PATH) -> Lexicon:
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    rules = tuple(
        FlagRule(
            code=code,
            negation_aware=bool(spec.get("negation_aware", True)),
            patterns=tuple(re.compile(p, re.IGNORECASE) for p in spec["patterns"]),
        )
        for code, spec in data["flags"].items()
    )
    return Lexicon(
        version=str(data["version"]),
        sha256=hashlib.sha256(raw).hexdigest(),
        rules=rules,
        trigger_keywords=tuple(k.lower() for k in data.get("model_trigger_keywords", [])),
    )


_CLAUSE_BREAK = re.compile(r"[,.;:!?]|\bbut\b|\bexcept\b", re.IGNORECASE)


def _is_negated(text: str, start: int) -> bool:
    # Only look inside the current clause: "No, but I have chest pain" is not negated.
    clause = _CLAUSE_BREAK.split(text[:start])[-1]
    preceding = _WORD.findall(clause.lower())[-NEGATION_WINDOW:]
    return any(w in NEGATION_CUES for w in preceding)


def screen_text(text: str, turn_id: str, lexicon: Lexicon | None = None) -> list[RedFlag]:
    """Rule layer: return at most one flag per code for this turn."""
    lexicon = lexicon or load_lexicon()
    flags: list[RedFlag] = []
    for rule in lexicon.rules:
        for pattern in rule.patterns:
            match = next(
                (
                    m
                    for m in pattern.finditer(text)
                    if not (rule.negation_aware and _is_negated(text, m.start()))
                ),
                None,
            )
            if match:
                flags.append(
                    RedFlag(
                        code=rule.code, source="rule", evidence_turn_id=turn_id, matched_text=match.group(0)
                    )
                )
                break
    return flags


def needs_model_check(text: str, lexicon: Lexicon | None = None) -> bool:
    """Only spend an LLM call when the turn mentions symptom words the rules might have missed."""
    lexicon = lexicon or load_lexicon()
    lowered = text.lower()
    return any(k in lowered for k in lexicon.trigger_keywords)


def merge_model_codes(
    rule_flags: list[RedFlag], model_codes: list[str], text: str, turn_id: str, lexicon: Lexicon | None = None
) -> list[RedFlag]:
    lexicon = lexicon or load_lexicon()
    seen = {f.code for f in rule_flags}
    merged = list(rule_flags)
    for code in model_codes:
        code = code.strip().upper()
        if code in lexicon.codes and code not in seen:
            merged.append(
                RedFlag(code=code, source="model", evidence_turn_id=turn_id, matched_text=text[:200])
            )
            seen.add(code)
    return merged
