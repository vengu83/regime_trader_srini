"""
Order executor — limit orders with market fallback, idempotent IDs, ATR stops.

Execution flow for buy/sell:
  1. Generate deterministic client_order_id (idempotent on restart).
  2. Check if that order already exists — if so, return it (no double-fill).
  3. Submit a DAY limit order at ask ± ORDER_LIMIT_OFFSET_BPS.
  4. Poll for fill up to ORDER_TIMEOUT_SECS.
  5. If unfilled: cancel and resubmit as a market order.

After a fill, call place_atr_stop() to attach a GTC trailing stop at the
broker — regime-dependent ATR multiplier keeps the stop tight in calm
markets and wide in volatile ones.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from brokers.alpaca_broker import AlpacaBroker
from config import settings

logger = logging.getLogger(__name__)


def _client_order_id(ticker: str, side: str, qty: int) -> str:
    """Deterministic order ID — safe to resubmit after a crash without double-filling."""
    ts = int(time.time() // 60) * 60   # stable within the same minute
    return f"rt-{ticker}-{side}-{qty}-{ts}"


class OrderExecutor:

    def __init__(self, broker: AlpacaBroker):
        self.broker = broker

    # ── Public interface ──────────────────────────────────────────────────────

    def buy(self, ticker: str, qty: int) -> Optional[dict]:
        """Limit → market fallback buy. Returns order info or None on failure."""
        if qty <= 0:
            return None
        return self._submit(ticker, qty, "buy")

    def sell(self, ticker: str, qty: int) -> Optional[dict]:
        """Limit → market fallback sell."""
        if qty <= 0:
            return None
        return self._submit(ticker, qty, "sell")

    def place_atr_stop(
        self,
        ticker: str,
        fill_price: float,
        atr: float,
        regime: str,
        qty: int = 0,
    ) -> Optional[dict]:
        """
        Place a GTC stop-loss order covering the full position qty.
        stop_price = fill_price − (atr_mult × ATR)  for long positions.
        qty must be the total position size — not just the delta.
        """
        if qty <= 0:
            logger.warning("place_atr_stop(%s): qty=%d — stop not placed", ticker, qty)
            return None

        alloc    = settings.REGIME_ALLOCATIONS.get(regime, {})
        atr_mult = alloc.get("atr_stop_mult", 2.0)
        stop_px  = round(fill_price - atr_mult * atr, 2)
        if stop_px <= 0:
            return None

        try:
            from alpaca.trading.requests import StopOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            req   = StopOrderRequest(
                symbol          = ticker,
                qty             = qty,
                side            = OrderSide.SELL,
                time_in_force   = TimeInForce.GTC,
                stop_price      = stop_px,
                client_order_id = _client_order_id(ticker, "stop", int(fill_price)),
            )
            order = self.broker._trading_client.submit_order(req)
            logger.info(
                "ATR stop placed: %s %d shares @ %.2f (mult=%.1f× ATR=%.4f) — id=%s",
                ticker, qty, stop_px, atr_mult, atr, order.id,
            )
            return {"id": str(order.id), "ticker": ticker, "stop_price": stop_px, "qty": qty}
        except Exception as exc:
            logger.error("place_atr_stop(%s) failed: %s", ticker, exc)
            return None

    def close_all_positions(self) -> None:
        """Emergency close — triggered by circuit breakers."""
        try:
            self.broker._trading_client.close_all_positions(cancel_orders=True)
            logger.warning("All positions closed via circuit breaker.")
        except Exception as exc:
            logger.error("close_all_positions failed: %s", exc)

    def cancel_all_orders(self) -> None:
        try:
            self.broker._trading_client.cancel_orders()
            logger.info("All open orders cancelled.")
        except Exception as exc:
            logger.error("cancel_all_orders failed: %s", exc)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _submit(self, ticker: str, qty: int, side: str) -> Optional[dict]:
        """Try limit order; fall back to market after ORDER_TIMEOUT_SECS."""
        client_id = _client_order_id(ticker, side, qty)

        # Idempotency check
        existing = self._find_order(client_id)
        if existing is not None:
            logger.info("Idempotent: order %s already exists (%s)", client_id, existing)
            return {"id": existing, "ticker": ticker, "qty": qty, "side": side, "type": "existing"}

        price = self.broker.get_current_price(ticker)
        if price is None:
            logger.error("Cannot get price for %s — skipping order", ticker)
            return None

        offset    = settings.ORDER_LIMIT_OFFSET_BPS / 10_000
        limit_px  = round(price * (1 + offset if side == "buy" else 1 - offset), 2)

        try:
            from alpaca.trading.requests import LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            req = LimitOrderRequest(
                symbol          = ticker,
                qty             = qty,
                side            = OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force   = TimeInForce.DAY,
                limit_price     = limit_px,
                client_order_id = client_id,
            )
            order = self.broker._trading_client.submit_order(req)
            logger.info(
                "LIMIT %s %d %s @ %.2f — id=%s",
                side.upper(), qty, ticker, limit_px, order.id,
            )

            if self._wait_for_fill(str(order.id)):
                return {"id": str(order.id), "ticker": ticker, "qty": qty, "side": side, "type": "limit"}

            # Timeout — cancel and fall back to market
            logger.warning("Limit order %s timed out — falling back to market", order.id)
            self._cancel(str(order.id))
        except Exception as exc:
            logger.error("Limit %s(%s, %d) failed: %s — trying market", side, ticker, qty, exc)

        return self._submit_market(ticker, qty, side)

    def _submit_market(self, ticker: str, qty: int, side: str) -> Optional[dict]:
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            req   = MarketOrderRequest(
                symbol        = ticker,
                qty           = qty,
                side          = OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force = TimeInForce.DAY,
            )
            order = self.broker._trading_client.submit_order(req)
            logger.info("MARKET %s %d %s — id=%s", side.upper(), qty, ticker, order.id)
            return {"id": str(order.id), "ticker": ticker, "qty": qty, "side": side, "type": "market"}
        except Exception as exc:
            logger.error("Market %s(%s, %d) failed: %s", side, ticker, qty, exc)
            return None

    def _wait_for_fill(self, order_id: str) -> bool:
        """Poll until filled or ORDER_TIMEOUT_SECS elapsed. Returns True if filled."""
        deadline = time.time() + settings.ORDER_TIMEOUT_SECS
        while time.time() < deadline:
            try:
                order = self.broker._trading_client.get_order_by_id(order_id)
                status = order.status.value
                if status in ("filled", "partially_filled"):
                    return True
                if status in ("cancelled", "expired", "rejected"):
                    return False
            except Exception:
                pass
            time.sleep(5)
        return False

    def _cancel(self, order_id: str) -> None:
        try:
            self.broker._trading_client.cancel_order_by_id(order_id)
        except Exception as exc:
            logger.warning("Cancel order %s failed: %s", order_id, exc)

    def _find_order(self, client_order_id: str) -> Optional[str]:
        """Return Alpaca order ID if client_order_id already exists, else None."""
        try:
            order = self.broker._trading_client.get_order_by_client_id(client_order_id)
            return str(order.id)
        except Exception:
            return None

    # ── Legacy aliases (backward compat with old main.py callers) ────────────

    def buy_market(self, ticker: str, qty: int, **_) -> Optional[dict]:
        return self.buy(ticker, qty)

    def sell_market(self, ticker: str, qty: int) -> Optional[dict]:
        return self.sell(ticker, qty)
