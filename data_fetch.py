"""Market data access.

Tries to fetch *live* data with yfinance. If the network is unavailable (for
example inside a sandbox with a host allowlist, or simply offline), it falls
back to the bundled ``sample_data/`` files and flips ``DEMO_MODE`` on so the UI
can tell the user they're looking at illustrative data rather than real quotes.

The public functions all return plain Python / pandas objects so the rest of
the app never has to know where the numbers came from.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

_SAMPLE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "sample_data")

# Flipped to True the first time a live fetch fails. The UI reads this to show
# a "demo mode" banner. It starts False and we discover the truth lazily.
# Set PORTFOLIO_ADVISOR_DEMO=1 to force demo mode up front (instant, no network)
# — handy offline or in a sandbox where market hosts are blocked.
DEMO_MODE = os.environ.get("PORTFOLIO_ADVISOR_DEMO", "").lower() in ("1", "true", "yes")

MARKET_TICKER = "SPY"  # used as the market benchmark for beta


def _load_sample(name: str) -> dict:
    with open(os.path.join(_SAMPLE_DIR, f"{name}.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _go_demo() -> None:
    global DEMO_MODE
    DEMO_MODE = True


def _try_import_yf():
    """Import yfinance lazily so the app still loads if it isn't installed."""
    try:
        import yfinance as yf  # noqa: WPS433 (intentional local import)

        return yf
    except Exception:  # pragma: no cover - environment dependent
        return None


# --------------------------------------------------------------------------- #
# Quotes
# --------------------------------------------------------------------------- #
def get_quote(ticker: str) -> dict:
    """Return ``{name, price, day_change_pct, sector, asset_type}`` for a ticker."""
    ticker = ticker.upper().strip()
    yf = _try_import_yf()
    if yf is not None and not DEMO_MODE:
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            prev = info.get("previousClose")
            if price:
                day_change = ((price - prev) / prev * 100) if prev else 0.0
                return {
                    "name": info.get("shortName") or ticker,
                    "price": float(price),
                    "day_change_pct": float(day_change),
                    "sector": info.get("sector") or _infer_sector(info, ticker),
                    "asset_type": _infer_asset_type(info, ticker),
                }
        except Exception:
            _go_demo()

    # ---- demo fallback ----
    _go_demo()
    quotes = _load_sample("quotes")
    q = quotes.get(ticker)
    if q is None:
        # Unknown ticker in demo mode: synthesise a neutral placeholder.
        return {
            "name": ticker,
            "price": 100.0,
            "day_change_pct": 0.0,
            "sector": "Unknown",
            "asset_type": "stock",
        }
    return {
        "name": q["name"],
        "price": q["price"],
        "day_change_pct": q["day_change_pct"],
        "sector": q["sector"],
        "asset_type": q["asset_type"],
    }


def _infer_asset_type(info: dict, ticker: str) -> str:
    if ticker.endswith("-USD"):
        return "crypto"
    qt = (info.get("quoteType") or "").upper()
    if qt == "ETF":
        return "etf"
    if qt == "CRYPTOCURRENCY":
        return "crypto"
    return "stock"


def _infer_sector(info: dict, ticker: str) -> str:
    if ticker.endswith("-USD"):
        return "Cryptocurrency"
    return info.get("sector") or "Unknown"


# --------------------------------------------------------------------------- #
# Price history (for volatility / beta / drawdown)
# --------------------------------------------------------------------------- #
def get_history(ticker: str, period: str = "1y") -> pd.Series:
    """Daily closing prices as a pandas Series indexed by date."""
    ticker = ticker.upper().strip()
    yf = _try_import_yf()
    if yf is not None and not DEMO_MODE:
        try:
            df = yf.Ticker(ticker).history(period=period)
            if not df.empty:
                return df["Close"].dropna()
        except Exception:
            _go_demo()

    _go_demo()
    return _synthetic_history(ticker)


_DEMO_DAYS = 252
_MARKET_ANNUAL_VOL = 0.15


@lru_cache(maxsize=1)
def _demo_market_returns() -> np.ndarray:
    """A single shared 'market' daily-return series for demo mode.

    Every synthetic ticker is built from this series so that beta is recoverable
    (a holding with sample beta 1.75 really does move ~1.75x the market here).
    """
    rng = np.random.default_rng(20240101)
    daily_vol = _MARKET_ANNUAL_VOL / np.sqrt(252)
    return rng.normal(loc=0.0004, scale=daily_vol, size=_DEMO_DAYS)


@lru_cache(maxsize=64)
def _synthetic_history(ticker: str, days: int = _DEMO_DAYS) -> pd.Series:
    """Deterministic price series for demo mode, correlated to the market.

    Returns are modelled as ``r = beta * market + idiosyncratic noise`` with the
    noise sized so the holding's total volatility matches the bundled sample
    metadata. This keeps demo-mode volatility AND beta realistic.
    """
    quotes = _load_sample("quotes")
    q = quotes.get(ticker, {"price": 100.0, "annual_vol": 0.30, "beta": 1.0})
    end_price = float(q.get("price", 100.0))
    annual_vol = float(q.get("annual_vol", 0.30))
    beta = float(q.get("beta", 1.0))

    market = _demo_market_returns()[:days]
    market_daily_vol = _MARKET_ANNUAL_VOL / np.sqrt(252)
    target_daily_var = (annual_vol / np.sqrt(252)) ** 2
    # var(r) = beta^2 * var_market + var_noise  ->  solve for the noise variance.
    noise_var = max(target_daily_var - (beta**2) * (market_daily_vol**2), 1e-8)

    seed = abs(hash(ticker)) % (2**32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_var), size=days)
    daily_ret = 0.0004 + beta * market + noise

    path = np.cumprod(1 + daily_ret)
    path = path / path[-1] * end_price  # end exactly at the quoted price
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.Series(path, index=idx, name=ticker)


def get_market_history(period: str = "1y") -> pd.Series:
    """Benchmark (S&P 500) history used as the 'market' when computing beta."""
    return get_history(MARKET_TICKER, period=period)


# --------------------------------------------------------------------------- #
# News
# --------------------------------------------------------------------------- #
def get_news(ticker: str, limit: int = 6) -> list[dict]:
    """Recent headlines: ``[{title, publisher, link, published}, ...]``."""
    ticker = ticker.upper().strip()
    yf = _try_import_yf()
    if yf is not None and not DEMO_MODE:
        try:
            raw = yf.Ticker(ticker).news or []
            items = []
            for n in raw[:limit]:
                content = n.get("content", n)  # yfinance schema varies by version
                title = content.get("title") or n.get("title")
                if not title:
                    continue
                pub = (
                    content.get("provider", {}).get("displayName")
                    if isinstance(content.get("provider"), dict)
                    else n.get("publisher")
                )
                link = (
                    content.get("canonicalUrl", {}).get("url")
                    if isinstance(content.get("canonicalUrl"), dict)
                    else n.get("link")
                )
                items.append(
                    {
                        "title": title,
                        "publisher": pub or "",
                        "link": link or "",
                        "published": content.get("pubDate") or n.get("providerPublishTime") or "",
                    }
                )
            if items:
                return items
        except Exception:
            _go_demo()

    _go_demo()
    news = _load_sample("news")
    return news.get(ticker, [])[:limit]
