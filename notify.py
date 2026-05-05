"""Telegram notification helper — token and chat ID from environment variables."""
import os
import sys
import json
import urllib.request

TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send(msg: str) -> None:
    if not TOKEN or not CHAT_ID:
        print(f"[notify] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping.", flush=True)
        return
    url  = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = json.dumps({"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}).encode()
    req  = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10)


if __name__ == "__main__":
    send(sys.argv[1])
