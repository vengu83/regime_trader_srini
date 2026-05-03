"""Email / webhook alerts for critical trading events."""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.text import MIMEText

import requests

from config import settings

logger = logging.getLogger(__name__)


class AlertManager:
    def send(self, subject: str, body: str) -> None:
        self._webhook(subject, body)
        self._email(subject, body)

    def _webhook(self, subject: str, body: str) -> None:
        if not settings.ALERT_WEBHOOK:
            return
        try:
            requests.post(
                settings.ALERT_WEBHOOK,
                json={"text": f"*{subject}*\n{body}"},
                timeout=5,
            )
        except Exception as exc:
            logger.warning("Webhook alert failed: %s", exc)

    def _email(self, subject: str, body: str) -> None:
        smtp_host = os.getenv("ALERT_SMTP_HOST", "")
        if not smtp_host:
            return  # SMTP not configured — skip silently

        smtp_port = int(os.getenv("ALERT_SMTP_PORT", "587"))
        smtp_user = os.getenv("ALERT_SMTP_USER", "")
        smtp_pass = os.getenv("ALERT_SMTP_PASSWORD", "")
        to_addr   = os.getenv("ALERT_EMAIL_TO", settings.ALERT_EMAIL)
        from_addr = smtp_user or to_addr

        if not to_addr:
            return

        try:
            msg = MIMEText(body)
            msg["Subject"] = f"[RegimeTrader] {subject}"
            msg["From"]    = from_addr
            msg["To"]      = to_addr
            with smtplib.SMTP(smtp_host, smtp_port) as s:
                s.ehlo()
                s.starttls()
                if smtp_user and smtp_pass:
                    s.login(smtp_user, smtp_pass)
                s.send_message(msg)
        except Exception as exc:
            logger.warning("Email alert failed: %s", exc)
