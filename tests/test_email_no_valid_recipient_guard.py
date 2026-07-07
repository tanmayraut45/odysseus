"""Regression: blank or comma-only recipient fields must not reach SMTP.

After normalisation, _envelope_recipients() can return [] when all To/Cc/Bcc
fields are empty or contain only separators.  Without a guard _send_smtp_message()
would be called with an empty list, causing an SMTP protocol error or silent drop
depending on the MTA.

The guard raises HTTP 422 in both the immediate-send handler (/send) and the
scheduled-send handler (/schedule) before any SMTP connection is attempted.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from fastapi import HTTPException, BackgroundTasks

import routes.email_routes as email_routes
import routes.email_helpers as email_helpers


def _route_endpoint(router, path: str, method: str):
    method = method.upper()
    for route in router.routes:
        if route.path == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"route not found: {method} {path}")


_STUB_CFG = {
    "smtp_host": "smtp.test",
    "smtp_port": 587,
    "from_address": "sender@test.com",
    "smtp_user": "u",
    "smtp_password": "p",
    "account_id": "acct-1",
}


async def test_send_raises_422_when_no_valid_recipients(monkeypatch):
    """Blank/comma-only To/Cc/Bcc must not reach SMTP on the /send path."""
    smtp_mock = MagicMock()
    monkeypatch.setattr(email_routes, "_send_smtp_message", smtp_mock)
    monkeypatch.setattr(email_routes, "_resolve_send_config", MagicMock(return_value=_STUB_CFG))

    router = email_routes.setup_email_routes()
    send_email = _route_endpoint(router, "/api/email/send", "POST")

    req = email_helpers.SendEmailRequest(to=",,,", subject="Hi", body="hello")
    try:
        result = await send_email(req, BackgroundTasks(), owner="alice")
        # Guard must raise before returning any success
        assert False, f"Expected HTTPException 422 but got: {result}"
    except HTTPException as exc:
        assert exc.status_code == 422
    smtp_mock.assert_not_called()


async def test_send_proceeds_when_recipient_is_valid(monkeypatch):
    """A valid To address must not raise 422 — delivery is queued."""
    smtp_mock = MagicMock()
    monkeypatch.setattr(email_routes, "_send_smtp_message", smtp_mock)
    monkeypatch.setattr(email_routes, "_resolve_send_config", MagicMock(return_value=_STUB_CFG))

    router = email_routes.setup_email_routes()
    send_email = _route_endpoint(router, "/api/email/send", "POST")

    req = email_helpers.SendEmailRequest(to="recipient@test.com", subject="Hi", body="hello")
    result = await send_email(req, BackgroundTasks(), owner="alice")
    # Must not 422 — delivery is queued as a background task
    assert result.get("success") is True
    assert result.get("queued") is True


async def test_schedule_raises_422_when_no_valid_recipients(tmp_path, monkeypatch):
    """Blank/comma-only recipients must be rejected before being written to the scheduled queue."""
    db_path = tmp_path / "scheduled_emails.db"
    monkeypatch.setattr(email_helpers, "SCHEDULED_DB", db_path)
    monkeypatch.setattr(email_routes, "SCHEDULED_DB", db_path)
    email_helpers._init_scheduled_db()

    smtp_mock = MagicMock()
    monkeypatch.setattr(email_routes, "_send_smtp_message", smtp_mock)

    router = email_routes.setup_email_routes()
    schedule_email = _route_endpoint(router, "/api/email/schedule", "POST")

    send_at = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    try:
        result = await schedule_email(
            {"to": "", "cc": ",,", "bcc": "", "subject": "Hi", "body": "hello", "send_at": send_at},
            owner="alice",
        )
        # Guard must raise before returning success
        assert False, f"Expected HTTPException 422 but got: {result}"
    except HTTPException as exc:
        assert exc.status_code == 422
    smtp_mock.assert_not_called()
