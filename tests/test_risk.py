"""Tests for the risk management layer."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.risk_manager import RiskManager
from config import settings


class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager(initial_equity=100_000.0)

    def test_ok_status_when_no_losses(self):
        status = self.rm.check_all(100_000.0, {})
        assert not status.circuit_breaker_active

    def test_daily_loss_halve_triggered(self):
        # Simulate -2.5% daily loss
        from datetime import date
        today = date.today()
        self.rm._day_start_equity[today] = 100_000.0
        status = self.rm.check_all(97_400.0, {})
        assert any("half" in m.lower() or "halve" in m.lower() for m in status.messages)

    def test_daily_loss_close_all_triggers_cb(self):
        from datetime import date
        today = date.today()
        self.rm._day_start_equity[today] = 100_000.0
        status = self.rm.check_all(96_500.0, {})  # -3.5%
        assert status.circuit_breaker_active

    def test_peak_drawdown_writes_lock_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "LOCK_FILE_PATH", str(tmp_path / "test.lock"))
        self.rm.peak_equity = 100_000.0
        status = self.rm.check_all(89_000.0, {})  # -11% from peak
        assert status.circuit_breaker_active
        assert os.path.exists(settings.LOCK_FILE_PATH)

    def test_lock_file_blocks_trading(self, tmp_path, monkeypatch):
        lock = tmp_path / "test.lock"
        lock.write_text("halted")
        monkeypatch.setattr(settings, "LOCK_FILE_PATH", str(lock))
        status = self.rm.check_all(100_000.0, {})
        assert status.circuit_breaker_active
        assert status.lock_file_exists

    def test_size_position_approved(self):
        from core.risk_manager import RiskStatus
        rs = RiskStatus()
        result = self.rm.size_position(
            equity=100_000.0, price=150.0,
            signal_equity_pct=0.50, risk_status=rs,
            existing_positions={}, ticker="SPY",
        )
        assert result.approved
        assert result.shares > 0

    def test_size_position_blocked_by_cb(self):
        from core.risk_manager import RiskStatus
        rs = RiskStatus(circuit_breaker_active=True)
        result = self.rm.size_position(
            equity=100_000.0, price=150.0,
            signal_equity_pct=0.50, risk_status=rs,
            existing_positions={}, ticker="SPY",
        )
        assert not result.approved

    def test_correlation_guard_blocks_duplicate(self):
        from core.risk_manager import RiskStatus
        rs = RiskStatus()
        # SPY and QQQ are marked as correlated
        existing = {"SPY": {"notional": 50_000}}
        result = self.rm.size_position(
            equity=100_000.0, price=400.0,
            signal_equity_pct=0.30, risk_status=rs,
            existing_positions=existing, ticker="QQQ",
        )
        assert not result.approved

    def test_size_multiplier_normal(self):
        from core.risk_manager import RiskStatus
        rs = RiskStatus()
        assert self.rm.get_size_multiplier(rs) == 1.0

    def test_size_multiplier_halved_on_daily_loss(self):
        from datetime import date
        from core.risk_manager import RiskStatus
        rs = RiskStatus(daily_pnl_pct=-0.025)
        assert self.rm.get_size_multiplier(rs) == 0.5
