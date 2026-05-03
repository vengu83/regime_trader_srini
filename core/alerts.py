"""
Notifications for critical trading events.

Channels (all optional — skip silently if not configured):
  - Telegram Bot API   → TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
  - Gmail / SMTP       → ALERT_SMTP_* env vars
  - Generic webhook    → ALERT_WEBHOOK (Slack, Discord, etc.)

Only send alerts when a trade actually executes or a circuit breaker fires —
never for informational / dry-run events.
"""

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
        """Fire all configured notification channels."""
        self._telegram(subject, body)
        self._email(subject, body)
        self._webhook(subject, body)

    # ── Telegram ──────────────────────────────────────────────────────────────

    def _telegram(self, subject: str, body: str) -> None:
        token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return

        text = f"*[RegimeTrader] {subject}*\n{body}"
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            resp = requests.post(
                url,
                json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
                timeout=10,
            )
            resp.raise_for_status()
            logger.debug("Telegram alert sent: %s", subject)
        except Exception as exc:
            logger.warning("Telegram alert failed: %s", exc)

    # ── Gmail / SMTP ──────────────────────────────────────────────────────────

    def _email(self, subject: str, body: str) -> None:
        smtp_host = os.getenv("ALERT_SMTP_HOST", "")
        if not smtp_host:
            return

        smtp_port = int(os.getenv("ALERT_SMTP_PORT", "587"))
        smtp_user = os.getenv("ALERT_SMTP_USER", "")
        smtp_pass = os.getenv("ALERT_SMTP_PASSWORD", "")
        to_addr   = os.getenv("ALERT_EMAIL_TO", settings.ALERT_EMAIL)
        from_addr = smtp_user or to_addr

        if not to_addr:
            return

        try:
            msg            = MIMEText(body)
            msg["Subject"] = f"[RegimeTrader] {subject}"
            msg["From"]    = from_addr
            msg["To"]      = to_addr
            with smtplib.SMTP(smtp_host, smtp_port) as s:
                s.ehlo()
                s.starttls()
                if smtp_user and smtp_pass:
                    s.login(smtp_user, smtp_pass)
                s.send_message(msg)
            logger.debug("Email alert sent to %s: %s", to_addr, subject)
        except Exception as exc:
            logger.warning("Email alert failed: %s", exc)

    # ── Generic webhook (Slack / Discord) ─────────────────────────────────────

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
