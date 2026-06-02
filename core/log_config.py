# core/log_config.py
# Centralised logging setup. Keeps stdout output (so `docker compose logs`
# still works) and adds a bounded RotatingFileHandler that writes to
# `logs/odysseus.log` — bind-mounted in the default docker-compose so the
# admin "App Logs" viewer has something to tail without operators having to
# shell into the container.

import logging
import os
from logging.handlers import RotatingFileHandler

# Defaults are conservative: 5 MiB per file * 5 backups = 25 MiB ceiling, so
# an idle install can't silently fill the disk. Operators can override via
# env vars if they want more retention.
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 5
DEFAULT_LOG_FILENAME = "odysseus.log"


def resolve_log_dir() -> str:
    """Where rotating logs land. Honours ODYSSEUS_LOG_DIR for tests/operators
    who want a custom location; otherwise the repo's logs/ folder (which the
    docker-compose mounts to ./logs on the host)."""
    override = os.environ.get("ODYSSEUS_LOG_DIR")
    if override:
        return override
    # Sit at repo root next to core/, src/, etc. — same convention as
    # core.constants.BASE_DIR but without importing the heavier module
    # (this file is invoked very early during app startup).
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "logs")


def configure_logging(
    level: int = logging.INFO,
    log_dir: str | None = None,
    filename: str = DEFAULT_LOG_FILENAME,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> str | None:
    """Install a StreamHandler + RotatingFileHandler on the root logger.

    Idempotent: safe to call from a reload context or a test that re-imports
    app.py. Returns the absolute path of the log file when file logging is
    active, or None if the file handler couldn't be attached (e.g. read-only
    filesystem) — callers can still rely on stdout in that case.
    """
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    # Always (re)attach a stream handler if one isn't present. `basicConfig`
    # used to do this, but only on a virgin root logger — re-importing under
    # uvicorn --reload would otherwise leave us silent.
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        for h in root.handlers
    ):
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        stream.setLevel(level)
        root.addHandler(stream)

    log_dir = log_dir or resolve_log_dir()
    log_path = os.path.join(log_dir, filename)

    # If a RotatingFileHandler already points at this exact path, leave it
    # alone — avoids piling up duplicate handlers across reloads.
    for h in root.handlers:
        if isinstance(h, RotatingFileHandler):
            existing = getattr(h, "baseFilename", None)
            if existing and os.path.abspath(existing) == os.path.abspath(log_path):
                return os.path.abspath(log_path)

    try:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root.addHandler(file_handler)
        return os.path.abspath(log_path)
    except OSError as e:
        # Read-only volume, missing perms, etc. — keep stdout logging working
        # so the operator still sees output, and warn loudly once.
        root.warning("Could not enable file logging at %s: %s", log_path, e)
        return None


def get_log_path() -> str:
    """Path the file handler writes to. Doesn't require configure_logging()
    to have been called — admin routes use this to locate the file even if
    the handler failed to attach (so they can report `file_size: 0` instead
    of crashing)."""
    return os.path.join(resolve_log_dir(), DEFAULT_LOG_FILENAME)
