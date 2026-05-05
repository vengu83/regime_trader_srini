"""
Standalone pre-market / end-of-day research runner.

Calls Perplexity and writes memory/market_research.md without starting
the full trading bot (no Alpaca connection, no HMM training, no SQLite).

Usage:
    python scripts/research.py
"""

from __future__ import annotations

import logging
import os
import sys

# Add repo root to path so `core` and `config` packages are importable
# regardless of the working directory the script is called from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure log and memory dirs exist before any imports that might write to them
os.makedirs("logs", exist_ok=True)
os.makedirs("memory", exist_ok=True)

logging.basicConfig(
    level="INFO",
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("research")

from core.research_provider import PerplexityResearchProvider  # noqa: E402

logger.info("Starting pre-market Perplexity research …")
rp = PerplexityResearchProvider()
success = rp.run()

if success:
    logger.info("Research complete — memory/market_research.md updated.")
else:
    logger.warning("Research skipped (PERPLEXITY_API_KEY not set or request failed).")

# Always exit 0 — research is best-effort; a missing API key should not
# fail the whole workflow and prevent the memory commit step from running.
sys.exit(0)
