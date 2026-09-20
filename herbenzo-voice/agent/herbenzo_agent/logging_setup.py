"""Logging that never writes patient content unless LOG_CONTENT=true (refused in production).

Application log statements only carry ids, statuses, counts and timings. The remaining leak paths are
exception messages (a validation error or provider error can echo user text or prompts) and chatty
third-party loggers; both are handled here.
"""

from __future__ import annotations

import logging
import sys
import traceback

REDACTED = "<redacted: set LOG_CONTENT=true in dev to see exception messages>"
QUIET_LIBRARIES = (
    "httpx",
    "httpcore",
    "openai",
    "langchain",
    "langchain_core",
    "langgraph",
    "aiosqlite",
    "urllib3",
)
_HANDLER_NAME = "herbenzo-redacting"


class RedactingFormatter(logging.Formatter):
    def __init__(self, log_content: bool):
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")
        self.log_content = log_content

    def format(self, record: logging.LogRecord) -> str:
        # Another handler may already have cached an unredacted traceback on the record.
        record.exc_text = None
        return super().format(record)

    def formatException(self, ei) -> str:  # noqa: N802 (logging API name)
        if self.log_content:
            return super().formatException(ei)
        exc_type, exc, tb = ei
        frames = "".join(traceback.format_tb(tb))
        chain = []
        cause = exc.__cause__ or exc.__context__ if exc else None
        while cause is not None and len(chain) < 5:
            chain.append(type(cause).__name__)
            cause = cause.__cause__ or cause.__context__
        caused_by = f" (caused by {' <- '.join(chain)})" if chain else ""
        return f"Traceback (most recent call last):\n{frames}{exc_type.__name__}: {REDACTED}{caused_by}"


def configure_logging(level: str = "INFO", log_content: bool = False) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        if handler.get_name() == _HANDLER_NAME:
            root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(RedactingFormatter(log_content))
    root.addHandler(handler)
    if not log_content:
        for name in QUIET_LIBRARIES:
            logging.getLogger(name).setLevel(logging.WARNING)
