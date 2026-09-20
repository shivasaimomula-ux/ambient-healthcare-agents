"""Write the SymptomSpec JSON Schema to herbenzo-voice/contracts/.

Run: uv run python -m herbenzo_agent.contracts.export_schema
"""

from __future__ import annotations

import json
from pathlib import Path

from herbenzo_agent.contracts.symptom_spec import SPEC_VERSION, SymptomSpec

CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "contracts"
SCHEMA_PATH = CONTRACTS_DIR / f"symptom_spec.v{SPEC_VERSION.split('.')[0]}.schema.json"


def build_schema() -> dict:
    schema = SymptomSpec.model_json_schema()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://herbenzo.com/contracts/symptom_spec/{SPEC_VERSION}",
        "x-spec-version": SPEC_VERSION,
        **schema,
    }


def render_schema() -> str:
    return json.dumps(build_schema(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(render_schema())
    print(f"wrote {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
