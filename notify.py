"""Telegram notification helper — called by cloud routines."""
import sys
import json
import urllib.request

TOKEN   = "8724633123:AAHJBSxmM2qSbTpueGumTwTK1zTBKgih-KM"
CHAT_ID = "5423504643"


def send(msg: str) -> None:
    url  = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = json.dumps({"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}).encode()
    req  = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10)


if __name__ == "__main__":
    send(sys.argv[1])
