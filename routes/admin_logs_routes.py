"""Admin app-logs viewer.

Surfaces the last N lines of logs/odysseus.log (the rotating file handler
set up in core.log_config) plus a download endpoint for offline analysis.
The route is intentionally narrow: read-only, admin-only, single file. The
goal is letting operators debug a Docker install without `docker compose
logs` — not building a full log browser.

URL shape:
    GET /api/admin/logs?lines=500&level=ERROR
    GET /api/admin/logs/download
"""

import logging
import os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from core.log_config import get_log_path
from core.middleware import require_admin

logger = logging.getLogger(__name__)

# Hard ceiling on `?lines=`. 5k lines * ~200 bytes ≈ 1 MiB of JSON — large
# but still bounded so a stray query can't force the server to slurp a 25 MiB
# file into memory and ship it over a single response.
MAX_LINES = 5000
DEFAULT_LINES = 500

ALLOWED_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

# Read in chunks from the tail so we don't load the whole file into memory
# when somebody only wants the last few hundred lines.
_TAIL_CHUNK = 8192


def _tail_lines(path: str, n: int) -> list[str]:
    """Read the last `n` lines of `path` without loading the whole file.

    Returns lines in chronological order (oldest first). If the file is
    smaller than the buffer we still avoid a full slurp via seek. Tolerates
    missing files (returns [])."""
    if n <= 0 or not os.path.isfile(path):
        return []

    try:
        size = os.path.getsize(path)
        if size == 0:
            return []
        with open(path, "rb") as f:
            buf = bytearray()
            pos = size
            newlines = 0
            # Read backwards in chunks until we've seen n+1 newlines (the
            # extra one anchors the start of the first kept line) or hit BOF.
            while pos > 0 and newlines <= n:
                read_size = min(_TAIL_CHUNK, pos)
                pos -= read_size
                f.seek(pos)
                chunk = f.read(read_size)
                buf[0:0] = chunk
                newlines = buf.count(b"\n")
            text = buf.decode("utf-8", errors="replace")
    except OSError as e:
        logger.warning("admin-logs tail failed for %s: %s", path, e)
        return []

    lines = text.splitlines()
    return lines[-n:] if len(lines) > n else lines


def _filter_by_level(lines: list[str], level: str) -> list[str]:
    """Keep only lines whose level token is at-or-above `level`.

    Matches the format installed by core.log_config: "TS - name - LEVEL - msg".
    Lines that don't parse (e.g. multi-line tracebacks continuing a record)
    are kept attached to whichever level we last saw so tracebacks aren't
    silently dropped from an ERROR filter."""
    level = level.upper()
    if level not in ALLOWED_LEVELS:
        return lines
    threshold = logging.getLevelName(level)
    if not isinstance(threshold, int):
        return lines

    keep: list[str] = []
    current_keep = False
    for line in lines:
        parts = line.split(" - ", 3)
        if len(parts) >= 3 and parts[2].strip() in ALLOWED_LEVELS:
            line_level = logging.getLevelName(parts[2].strip())
            current_keep = isinstance(line_level, int) and line_level >= threshold
        # else: continuation line — fall through and reuse the previous decision.
        if current_keep:
            keep.append(line)
    return keep


def setup_admin_logs_routes() -> APIRouter:
    router = APIRouter(prefix="/api/admin")

    @router.get("/logs")
    def get_logs(request: Request, lines: int = DEFAULT_LINES, level: str = ""):
        require_admin(request)

        # Clamp first, then read — keeps the tail cheap even if the caller
        # passed something absurd.
        try:
            n = int(lines)
        except (TypeError, ValueError):
            raise HTTPException(400, "lines must be an integer")
        if n < 1:
            n = 1
        if n > MAX_LINES:
            n = MAX_LINES

        level = (level or "").strip().upper()
        if level and level not in ALLOWED_LEVELS:
            raise HTTPException(
                400,
                f"level must be one of {sorted(ALLOWED_LEVELS)} or empty",
            )

        path = get_log_path()
        file_size = 0
        exists = os.path.isfile(path)
        if exists:
            try:
                file_size = os.path.getsize(path)
            except OSError:
                file_size = 0

        out_lines = _tail_lines(path, n)
        if level:
            out_lines = _filter_by_level(out_lines, level)

        return {
            "lines": out_lines,
            "truncated": exists and file_size > 0 and len(out_lines) >= n,
            "file_size": file_size,
            "path": os.path.basename(path),
            "exists": exists,
        }

    @router.get("/logs/download")
    def download_logs(request: Request):
        require_admin(request)
        path = get_log_path()
        if not os.path.isfile(path):
            raise HTTPException(404, "Log file not found")
        return FileResponse(
            path,
            media_type="text/plain",
            filename=os.path.basename(path),
        )

    return router
