"""
Perplexity-powered pre-market research provider.

Queries Perplexity's sonar-pro model (real-time web search) for:
  - Macro context: Fed stance, scheduled economic releases, futures
  - Per-ticker news: material events for each ticker in the universe
  - Regime alignment: whether the news supports or conflicts with the HMM regime

Results are written to memory/market_research.md for the Claude routines to read.

Usage:
    from core.research_provider import PerplexityResearchProvider
    rp = PerplexityResearchProvider()
    rp.run(regime_map={"SPY": "bull", "QQQ": "bull", "TLT": "bear", "GLD": "neutral"})
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional

import requests
from dotenv import load_dotenv

from config import settings
from core.memory_writer import MEMORY_DIR, _write

load_dotenv()
logger = logging.getLogger(__name__)

_PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"


class PerplexityResearchProvider:
    """Calls Perplexity sonar-pro and writes pre-market research to memory/market_research.md."""

    def __init__(self):
        self._api_key = os.getenv("PERPLEXITY_API_KEY", "")
        if not self._api_key or self._api_key == "your_perplexity_api_key_here":
            logger.warning(
                "PERPLEXITY_API_KEY not set — research will be skipped. "
                "Add it to .env to enable pre-market research."
            )

    @property
    def _available(self) -> bool:
        return bool(self._api_key) and self._api_key != "your_perplexity_api_key_here"

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, regime_map: Optional[Dict[str, str]] = None) -> bool:
        """
        Run full pre-market research pipeline and write memory/market_research.md.
        Returns True on success, False if API key missing or request failed.
        """
        if not self._available:
            logger.warning("Perplexity API key not configured — skipping research.")
            return False

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        tickers = settings.TICKERS

        logger.info("Running pre-market research via Perplexity (%s)…", settings.PERPLEXITY_MODEL)

        macro   = self._query_macro(today)
        news    = self._query_ticker_news(tickers, today)
        risks   = self._query_risk_flags(today)
        alignment = self._assess_regime_alignment(regime_map or {}, macro, news)

        content = self._format_markdown(today, macro, news, risks, alignment, regime_map or {})
        _write("market_research.md", content)
        logger.info("Pre-market research written to memory/market_research.md")
        return True

    # ── Perplexity queries ────────────────────────────────────────────────────

    def _query_macro(self, date: str) -> str:
        prompt = (
            f"Today is {date}. Give a concise pre-market brief (4–6 bullets) covering:\n"
            "1. S&P 500 and Nasdaq futures direction right now (pre-market)\n"
            "2. Any scheduled US economic data releases today (CPI, NFP, FOMC, PPI, jobless claims, etc.)\n"
            "3. Fed commentary or Fed speaker events today\n"
            "4. Overnight international market moves (Asia, Europe)\n"
            "Keep each bullet to one sentence. Use facts, not forecasts."
        )
        return self._call(prompt)

    def _query_ticker_news(self, tickers: list, date: str) -> Dict[str, str]:
        results = {}
        for ticker in tickers:
            prompt = (
                f"Today is {date}. In 2–3 sentences, summarise the most material recent news "
                f"for {ticker} ETF that could affect its price today. "
                "Focus on ETF-specific news, sector moves, or macro drivers. "
                "If nothing notable, say 'No material news.'"
            )
            results[ticker] = self._call(prompt)
        return results

    def _query_risk_flags(self, date: str) -> str:
        prompt = (
            f"Today is {date}. List any scheduled high-volatility events or risk flags "
            "for US equity markets today or this week (e.g. FOMC meetings, CPI releases, "
            "earnings from major index constituents, geopolitical events). "
            "3–5 bullets, one sentence each. If quiet, say 'No major risk events scheduled.'"
        )
        return self._call(prompt)

    def _assess_regime_alignment(
        self,
        regime_map: Dict[str, str],
        macro: str,
        news: Dict[str, str],
    ) -> str:
        if not regime_map:
            return "No current regime data available — run the bot to generate regime_insights.md."

        regime_str = "\n".join(f"  - {t}: {r}" for t, r in regime_map.items())
        prompt = (
            "Based on the following market context, assess whether today's news supports or "
            "conflicts with the detected HMM trading regimes.\n\n"
            f"Current regimes:\n{regime_str}\n\n"
            f"Macro context:\n{macro}\n\n"
            "For each ticker, state in one sentence: ALIGNED, CONFLICT, or CAUTION — "
            "with a brief reason. Then give an overall risk posture recommendation "
            "(e.g. 'proceed as normal', 'reduce sizes', 'flag for manual review')."
        )
        return self._call(prompt)

    # ── HTTP ──────────────────────────────────────────────────────────────────

    def _call(self, user_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.PERPLEXITY_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a quantitative trading research assistant. "
                        "Provide concise, factual market information. "
                        "No investment advice — only factual summaries."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": settings.PERPLEXITY_MAX_TOKENS,
            "temperature": settings.PERPLEXITY_TEMPERATURE,
        }
        try:
            resp = requests.post(_PERPLEXITY_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as exc:
            logger.error("Perplexity HTTP error: %s", exc)
            return f"[Research unavailable — HTTP {exc.response.status_code}]"
        except Exception as exc:
            logger.error("Perplexity request failed: %s", exc)
            return "[Research unavailable — request failed]"

    # ── Formatting ────────────────────────────────────────────────────────────

    def _format_markdown(
        self,
        date: str,
        macro: str,
        news: Dict[str, str],
        risks: str,
        alignment: str,
        regime_map: Dict[str, str],
    ) -> str:
        ticker_sections = "\n".join(
            f"- **{t}** (regime: `{regime_map.get(t, 'unknown')}`): {summary}"
            for t, summary in news.items()
        )

        return f"""# Market Research
_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} via Perplexity {settings.PERPLEXITY_MODEL}_

## {date} Pre-Market Research

### Macro Context
{macro}

### Ticker News
{ticker_sections}

### Risk Flags
{risks}

### Regime Alignment Assessment
{alignment}
"""
