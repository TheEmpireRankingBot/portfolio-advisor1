"""Portfolio maths.

Everything here is deliberately small and pure so it can be unit-tested with
synthetic price series. No network calls live in this module — callers pass in
the prices / quotes they already fetched.

Key ideas (explained for a beginner in the Learn tab):
    * volatility -- how much a price bounces around; our proxy for "riskiness".
    * beta       -- how much a holding moves when the whole market moves.
    * drawdown   -- the worst peak-to-trough drop; how much pain you'd have felt.
    * HHI        -- a concentration score; high means "lots of eggs in one basket".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(prices: pd.Series) -> pd.Series:
    """Percentage change from one day to the next."""
    return prices.pct_change().dropna()


def annualized_volatility(prices: pd.Series) -> float:
    """Standard deviation of daily returns, scaled to a yearly figure."""
    r = daily_returns(prices)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def beta(prices: pd.Series, market_prices: pd.Series) -> float:
    """Sensitivity of a holding to the overall market (1.0 = moves with market)."""
    a = daily_returns(prices)
    m = daily_returns(market_prices)
    joined = pd.concat([a, m], axis=1, join="inner").dropna()
    if len(joined) < 2:
        return 1.0
    cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1])
    market_var = cov[1, 1]
    if market_var == 0:
        return 1.0
    return float(cov[0, 1] / market_var)


def max_drawdown(prices: pd.Series) -> float:
    """Largest peak-to-trough decline, as a negative fraction (e.g. -0.35)."""
    if prices.empty:
        return 0.0
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    return float(drawdown.min())


def hhi(weights: list[float]) -> float:
    """Herfindahl-Hirschman Index: sum of squared weights (0=spread, 1=all-in-one)."""
    return float(sum(w**2 for w in weights))


# --------------------------------------------------------------------------- #
# Portfolio-level aggregation
# --------------------------------------------------------------------------- #
def build_positions(holdings: list[dict], quotes: dict) -> list[dict]:
    """Combine holdings with live quotes into per-position rows with P/L & weight."""
    positions = []
    for h in holdings:
        ticker = h["ticker"].upper()
        q = quotes.get(ticker, {})
        price = q.get("price", h.get("avg_price", 0.0))
        shares = float(h["shares"])
        avg = float(h["avg_price"])
        market_value = price * shares
        cost_basis = avg * shares
        pnl = market_value - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis else 0.0
        positions.append(
            {
                "ticker": ticker,
                "name": q.get("name", ticker),
                "shares": shares,
                "avg_price": avg,
                "price": price,
                "day_change_pct": q.get("day_change_pct", 0.0),
                "market_value": market_value,
                "cost_basis": cost_basis,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "sector": q.get("sector", "Unknown"),
                "asset_type": q.get("asset_type", h.get("asset_type", "stock")),
            }
        )

    total = sum(p["market_value"] for p in positions) or 1.0
    for p in positions:
        p["weight"] = p["market_value"] / total
    return positions


def portfolio_summary(positions: list[dict]) -> dict:
    """Totals across all positions."""
    total_value = sum(p["market_value"] for p in positions)
    total_cost = sum(p["cost_basis"] for p in positions)
    total_pnl = total_value - total_cost
    return {
        "total_value": total_value,
        "total_cost": total_cost,
        "total_pnl": total_pnl,
        "total_pnl_pct": (total_pnl / total_cost * 100) if total_cost else 0.0,
        "num_holdings": len(positions),
    }


def allocation_by(positions: list[dict], key: str) -> dict:
    """Sum of weights grouped by a field such as 'sector' or 'asset_type'."""
    out: dict[str, float] = {}
    for p in positions:
        out[p[key]] = out.get(p[key], 0.0) + p["weight"]
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def weighted_metric(positions: list[dict], per_ticker: dict, default: float = 0.0) -> float:
    """Portfolio weight-average of a per-ticker metric (e.g. volatility, beta)."""
    return float(
        sum(p["weight"] * per_ticker.get(p["ticker"], default) for p in positions)
    )
