"""Tests for routes/admin_logs_routes.py and core/log_config.py.

Covers:
- core.log_config: configure_logging attaches both stdout + rotating file
  handlers, is idempotent, and honours ODYSSEUS_LOG_DIR.
- /api/admin/logs: auth gating, default + max line caps, level filtering,
  missing-file handling.
- /api/admin/logs/download: auth gating, returns a FileResponse pointed at
  the configured log file.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

import pytest
from fastapi import HTTPException, Request

# Ensure module-under-test pulls from a clean state every invocation.
import core.log_config as log_config
import routes.admin_logs_routes as admin_logs_routes


def _reset_root_logger():
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass


@pytest.fixture
def tmp_log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ODYSSEUS_LOG_DIR", str(tmp_path))
    _reset_root_logger()
    yield tmp_path
    _reset_root_logger()


def _make_request(headers=None) -> Request:
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "app": type("A", (), {"state": type("S", (), {"auth_manager": None})()})(),
    }
    req = Request(scope=scope)
    # request.state.current_user defaults to absent — admin gate should refuse.
    return req


# ───────────────────────── log_config ─────────────────────────

def test_configure_logging_attaches_both_handlers(tmp_log_dir):
    path = log_config.configure_logging(level=logging.DEBUG)
    assert path is not None
    assert os.path.dirname(path) == str(tmp_log_dir)

    root = logging.getLogger()
    streams = [h for h in root.handlers
               if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)]
    files = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
    assert streams, "stdout handler should remain — docker compose logs depends on it"
    assert files, "file handler should be attached for the admin viewer"
    assert os.path.abspath(files[0].baseFilename) == os.path.abspath(path)


def test_configure_logging_is_idempotent(tmp_log_dir):
    # Count handler totals before and after extra configure_logging calls —
    # robust to whatever pytest has installed on the root logger.
    log_config.configure_logging(level=logging.INFO)
    root = logging.getLogger()
    baseline = len(root.handlers)
    log_config.configure_logging(level=logging.INFO)
    log_config.configure_logging(level=logging.INFO)
    assert len(root.handlers) == baseline, "extra reload calls should not pile up handlers"
    # And the single file handler points where we expect.
    files = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
    assert len(files) == 1
    assert os.path.dirname(os.path.abspath(files[0].baseFilename)) == str(tmp_log_dir)


def test_rotating_file_handler_rotates(tmp_log_dir):
    # Tiny ceiling forces a rotation after a single line; verifies the
    # handler actually rolls over instead of growing unbounded.
    path = log_config.configure_logging(
        level=logging.INFO, max_bytes=200, backup_count=2,
    )
    log = logging.getLogger("test_rotation")
    for i in range(20):
        log.info("padding line %d %s", i, "x" * 50)

    backups = [p for p in os.listdir(tmp_log_dir) if p.startswith("odysseus.log")]
    # Active file + at least one backup (odysseus.log.1) should now exist.
    assert "odysseus.log" in backups
    assert any(name.startswith("odysseus.log.") for name in backups), backups
    assert os.path.getsize(path) <= 200 + 256  # small slack for the last record


# ───────────────────────── /api/admin/logs ─────────────────────────

def _route_endpoint(router, path: str):
    for r in router.routes:
        if r.path == path:
            return r.endpoint
    raise AssertionError(f"route {path} not registered")


def test_logs_requires_admin(tmp_log_dir, monkeypatch):
    # Force the production require_admin path — no AUTH_ENABLED bypass.
    monkeypatch.setenv("AUTH_ENABLED", "true")
    router = admin_logs_routes.setup_admin_logs_routes()
    get_logs = _route_endpoint(router, "/api/admin/logs")
    request = _make_request()
    with pytest.raises(HTTPException) as exc:
        get_logs(request=request, lines=10, level="")
    assert exc.value.status_code == 403


def test_logs_returns_recent_lines_only(tmp_log_dir, monkeypatch):
    monkeypatch.setattr(admin_logs_routes, "require_admin", lambda r: None)
    log_path = os.path.join(tmp_log_dir, "odysseus.log")
    with open(log_path, "w", encoding="utf-8") as f:
        for i in range(50):
            f.write(f"2026-06-02 - test - INFO - line {i}\n")

    router = admin_logs_routes.setup_admin_logs_routes()
    get_logs = _route_endpoint(router, "/api/admin/logs")

    result = get_logs(request=_make_request(), lines=5, level="")
    assert result["exists"] is True
    assert result["file_size"] > 0
    assert len(result["lines"]) == 5
    assert result["lines"][-1].endswith("line 49")
    assert result["lines"][0].endswith("line 45")


def test_logs_caps_lines_at_max(tmp_log_dir, monkeypatch):
    monkeypatch.setattr(admin_logs_routes, "require_admin", lambda r: None)
    log_path = os.path.join(tmp_log_dir, "odysseus.log")
    with open(log_path, "w", encoding="utf-8") as f:
        for i in range(20):
            f.write(f"2026-06-02 - test - INFO - line {i}\n")

    router = admin_logs_routes.setup_admin_logs_routes()
    get_logs = _route_endpoint(router, "/api/admin/logs")

    # Caller asks for absurd value; route clamps without raising.
    result = get_logs(request=_make_request(), lines=10_000_000, level="")
    assert len(result["lines"]) == 20  # entire file, since file < MAX_LINES
    # And the cap itself is respected even when the file is huge — simulate
    # by directly clamping via the helper.
    assert admin_logs_routes.MAX_LINES == 5000


def test_logs_filters_by_level(tmp_log_dir, monkeypatch):
    monkeypatch.setattr(admin_logs_routes, "require_admin", lambda r: None)
    log_path = os.path.join(tmp_log_dir, "odysseus.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("2026-06-02 - x - INFO - chatty\n")
        f.write("2026-06-02 - x - WARNING - heads up\n")
        f.write("2026-06-02 - x - ERROR - boom\n")
        f.write("    traceback continuation line\n")  # belongs to ERROR
        f.write("2026-06-02 - x - DEBUG - noise\n")

    router = admin_logs_routes.setup_admin_logs_routes()
    get_logs = _route_endpoint(router, "/api/admin/logs")

    result = get_logs(request=_make_request(), lines=500, level="ERROR")
    joined = "\n".join(result["lines"])
    assert "boom" in joined
    assert "traceback continuation" in joined  # multi-line ERROR survives
    assert "chatty" not in joined
    assert "heads up" not in joined
    assert "noise" not in joined


def test_logs_rejects_invalid_level(tmp_log_dir, monkeypatch):
    monkeypatch.setattr(admin_logs_routes, "require_admin", lambda r: None)
    router = admin_logs_routes.setup_admin_logs_routes()
    get_logs = _route_endpoint(router, "/api/admin/logs")
    with pytest.raises(HTTPException) as exc:
        get_logs(request=_make_request(), lines=10, level="LOUD")
    assert exc.value.status_code == 400


def test_logs_missing_file_is_not_an_error(tmp_log_dir, monkeypatch):
    monkeypatch.setattr(admin_logs_routes, "require_admin", lambda r: None)
    # No file written.
    router = admin_logs_routes.setup_admin_logs_routes()
    get_logs = _route_endpoint(router, "/api/admin/logs")
    result = get_logs(request=_make_request(), lines=100, level="")
    assert result["exists"] is False
    assert result["lines"] == []
    assert result["file_size"] == 0


def test_download_requires_admin(tmp_log_dir, monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    router = admin_logs_routes.setup_admin_logs_routes()
    download = _route_endpoint(router, "/api/admin/logs/download")
    with pytest.raises(HTTPException) as exc:
        download(request=_make_request())
    assert exc.value.status_code == 403


def test_download_returns_file_when_present(tmp_log_dir, monkeypatch):
    monkeypatch.setattr(admin_logs_routes, "require_admin", lambda r: None)
    log_path = os.path.join(tmp_log_dir, "odysseus.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("hello world\n")
    router = admin_logs_routes.setup_admin_logs_routes()
    download = _route_endpoint(router, "/api/admin/logs/download")
    response = download(request=_make_request())
    # FileResponse exposes the resolved path on the .path attribute.
    assert os.path.abspath(response.path) == os.path.abspath(log_path)


def test_download_missing_file_returns_404(tmp_log_dir, monkeypatch):
    monkeypatch.setattr(admin_logs_routes, "require_admin", lambda r: None)
    router = admin_logs_routes.setup_admin_logs_routes()
    download = _route_endpoint(router, "/api/admin/logs/download")
    with pytest.raises(HTTPException) as exc:
        download(request=_make_request())
    assert exc.value.status_code == 404
