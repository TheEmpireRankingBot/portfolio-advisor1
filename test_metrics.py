"""Unit tests for the pure-maths metrics, using synthetic price series."""

import numpy as np
import pandas as pd

from advisor import metrics


def _series(values):
    idx = pd.bdate_range(end=pd.Timestamp("2024-01-01"), periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_hhi_extremes():
    # Everything in one basket -> 1.0
    assert metrics.hhi([1.0]) == 1.0
    # Four equal baskets -> 0.25
    assert abs(metrics.hhi([0.25, 0.25, 0.25, 0.25]) - 0.25) < 1e-9


def test_flat_series_has_zero_volatility():
    flat = _series([100.0] * 50)
    assert metrics.annualized_volatility(flat) == 0.0


def test_volatility_is_positive_for_moving_series():
    rng = np.random.default_rng(0)
    prices = _series(100 * np.cumprod(1 + rng.normal(0, 0.01, 200)))
    assert metrics.annualized_volatility(prices) > 0


def test_beta_of_identical_series_is_one():
    rng = np.random.default_rng(1)
    market = _series(100 * np.cumprod(1 + rng.normal(0, 0.01, 200)))
    assert abs(metrics.beta(market, market) - 1.0) < 1e-6


def test_beta_of_double_moves_is_two():
    rng = np.random.default_rng(2)
    m_ret = rng.normal(0, 0.01, 200)
    market = _series(100 * np.cumprod(1 + m_ret))
    asset = _series(100 * np.cumprod(1 + 2 * m_ret))  # moves exactly twice as hard
    assert abs(metrics.beta(asset, market) - 2.0) < 0.05


def test_max_drawdown_is_negative_after_a_crash():
    prices = _series([100, 120, 60, 80])  # peak 120 -> trough 60 = -50%
    assert abs(metrics.max_drawdown(prices) - (-0.5)) < 1e-9


def test_build_positions_and_summary():
    holdings = [
        {"ticker": "AAA", "shares": 10, "avg_price": 10.0, "asset_type": "stock"},
        {"ticker": "BBB", "shares": 5, "avg_price": 20.0, "asset_type": "etf"},
    ]
    quotes = {
        "AAA": {"name": "AAA", "price": 15.0, "sector": "Tech", "asset_type": "stock", "day_change_pct": 0},
        "BBB": {"name": "BBB", "price": 20.0, "sector": "ETF", "asset_type": "etf", "day_change_pct": 0},
    }
    positions = metrics.build_positions(holdings, quotes)
    # AAA value 150, BBB value 100, total 250
    summary = metrics.portfolio_summary(positions)
    assert abs(summary["total_value"] - 250.0) < 1e-9
    assert abs(summary["total_cost"] - 200.0) < 1e-9  # 100 + 100
    aaa = next(p for p in positions if p["ticker"] == "AAA")
    assert abs(aaa["weight"] - 0.6) < 1e-9
    assert abs(aaa["pnl"] - 50.0) < 1e-9
