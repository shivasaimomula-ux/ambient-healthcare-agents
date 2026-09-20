import pytest

from herbenzo_agent.settings import MissingConfigError, Settings


def test_defaults_use_current_nvidia_model():
    s = Settings(_env_file=None)
    assert s.llm_extractor_model == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert s.min_adult_age == 18
    assert s.pipeline_mode is True
    assert s.recommender_url == "http://localhost:8000"


def test_require_reports_missing_names():
    s = Settings(_env_file=None, internal_api_token=None)
    with pytest.raises(MissingConfigError, match="INTERNAL_API_TOKEN"):
        s.require("internal_api_token")


def test_prod_refuses_content_logging():
    s = Settings(_env_file=None, env="prod", log_content=True, internal_api_token="x")
    with pytest.raises(MissingConfigError, match="LOG_CONTENT"):
        s.require("internal_api_token")
