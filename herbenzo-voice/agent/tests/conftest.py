import os

import pytest

# Tests must never depend on the developer's real .env values.
os.environ.setdefault("ENV", "test")


def pytest_collection_modifyitems(config, items):
    if os.getenv("HERBENZO_LIVE_TESTS") == "1":
        return
    skip_live = pytest.mark.skip(reason="live test; set HERBENZO_LIVE_TESTS=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
