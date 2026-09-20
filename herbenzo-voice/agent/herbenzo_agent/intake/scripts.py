"""Every fixed utterance the agent speaks. None of these are LLM-generated.

Consent, escalation, readback, closing and deflection wording must be identical every time, so
it can be reviewed (clinician/legal) and so its hash can be recorded in each spec's provenance.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml

from herbenzo_agent.intake.models import IntakeSession

JURISDICTIONS_PATH = Path(__file__).resolve().parent.parent / "data_files" / "jurisdictions.yaml"

Channel = Literal["voice", "chat"]

SCRIPTS: dict[str, str] = {
    "greeting_consent": (
        "Namaste, and welcome to Herbenzo. I'm an automated assistant. I'll ask a few short questions "
        "about how you're feeling, so our evidence team can review natural medicine options for you. "
        "I can't diagnose or give medical advice. If this is an emergency, please call {emergency} now. "
        "Your answers are stored securely and used only for this review. Is it okay to continue?"
    ),
    "consent_reask": "Sorry, I didn't catch that. Is it okay for me to ask you a few questions about your health?",
    "consent_refused": "That's completely fine. No information has been saved. Take care.",
    "escalation": (
        "What you're describing needs urgent medical attention. Please stop here and call {emergency}, "
        "or {ambulance} for an ambulance, or go to the nearest emergency department now. "
        "I'm ending this session so you can get help."
    ),
    "escalation_self_harm": (
        "I'm really sorry you're going through this, and I'm glad you told me. Please call {emergency} now "
        "if you might act on these thoughts. You can also call {mental_health_name} on {mental_health}, "
        "any time, free. I'm ending this session so you can reach someone who can help."
    ),
    "minor_out_of_scope": (
        "Thank you. I'm not able to continue an intake for someone under {min_age}. "
        "Please see a qualified practitioner together with a parent or guardian. Take care."
    ),
    "deflect_advice": (
        "I can't give medical advice or suggest what to take, but I'll make sure your details reach our review team."
    ),
    "deflect_off_topic": "I can only help with your health intake today.",
    "deflect_unsafe": "I can't help with that, but I'm happy to continue with your health intake.",
    "correction_prompt": "No problem. What should I change?",
    "confirm_restart": "Do you want to start over? Everything you've told me so far will be cleared.",
    "restart_resume": "Okay, let's carry on.",
    "stopped": "Okay, I've stopped here. Nothing has been sent for review. Take care.",
    "closing_eligible": (
        "Thank you. Your details are saved. Our evidence review takes a few minutes, and the results will appear "
        "on your Herbenzo results page. Please don't change any of your current medicines without your doctor. "
        "Take care."
    ),
    "closing_not_eligible": (
        "Thank you. Your details are saved, but I don't have enough information for an automated review, "
        "so a practitioner will need to look at this. Take care."
    ),
    "session_closed": "This session has ended. Please start a new session if you need anything else.",
    "empty_input": "Sorry, I didn't catch that. Could you say it again?",
    "system_retry": "Sorry, I missed that. Could you say it again, please?",
}


@lru_cache
def load_jurisdictions() -> dict[str, Any]:
    return yaml.safe_load(JURISDICTIONS_PATH.read_text())


def _contacts(jurisdiction: str, channel: Channel) -> dict[str, str]:
    j = load_jurisdictions().get(jurisdiction) or load_jurisdictions()["IN"]
    form = "spoken" if channel == "voice" else "display"
    return {
        "emergency": j["emergency"][form],
        "ambulance": j["ambulance"][form],
        "mental_health": j["mental_health"][form],
        "mental_health_name": j["mental_health"]["name"],
    }


def render(key: str, session: IntakeSession, **params: Any) -> str:
    return SCRIPTS[key].format(**_contacts(session.jurisdiction, session.channel), **params)


def scripts_sha256() -> str:
    payload = json.dumps(SCRIPTS, sort_keys=True) + JURISDICTIONS_PATH.read_text()
    return hashlib.sha256(payload.encode()).hexdigest()
