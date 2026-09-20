"""Hashes and model IDs recorded in every SymptomSpec's provenance (feeds the Reproducibility Log)."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from herbenzo_agent import __version__
from herbenzo_agent.contracts.symptom_spec import Provenance
from herbenzo_agent.intake import lint, red_flags, scripts

PACKAGE_DIR = Path(__file__).resolve().parent.parent


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def directory_sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


@lru_cache
def prompt_hashes() -> dict[str, str]:
    prompts = PACKAGE_DIR / "prompts"
    return {
        "extractor_system": _sha((prompts / "extractor_system.md").read_bytes()),
        "responder_system": _sha((prompts / "responder_system.md").read_bytes()),
        "scripts": scripts.scripts_sha256(),
        "red_flags": red_flags.load_lexicon().sha256,
        "blocked_terms": lint.blocked_terms_sha256(),
    }


@lru_cache
def graph_version() -> str:
    sources = [
        "intake/slots.py",
        "intake/planner.py",
        "intake/models.py",
        "intake/readback.py",
        "intake/graph.py",
    ]
    digest = hashlib.sha256()
    for rel in sources:
        path = PACKAGE_DIR / rel
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def build_provenance(
    *,
    session_id: str,
    llm_models: dict[str, str],
    guardrails_config_sha256: str,
    asr_model: str | None = None,
    tts_model: str | None = None,
) -> Provenance:
    return Provenance(
        agent_version=__version__,
        graph_version=graph_version(),
        llm_models=llm_models,
        prompt_sha256=prompt_hashes(),
        guardrails_config_sha256=guardrails_config_sha256,
        asr_model=asr_model,
        tts_model=tts_model,
        transcript_ref=f"turns:{session_id}",
    )
