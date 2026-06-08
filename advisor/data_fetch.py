"""Market data access (cloud-friendly).

The original version relied on Yahoo Finance, which blocks requests coming from
cloud servers (so a deployed app fell back to demo data). This version uses
sources that work from the cloud:

    * Quotes  -> Finnhub (needs a free FINNHUB_API_KEY) for name/price/%/sector.
                 Falls back to Stooq (keyless) for stocks and for crypto.
    * History -> Stooq daily CSV (keyless) for volatility / beta / drawdown.
    * News    -> Finnhub company-news (needs the key).

If everything is unreachable (e.g. offline), it falls back to the bundled
``sample_data/`` and flips ``DEMO_MODE`` so the UI shows a clear banner. Set your
key on Streamlit Cloud under *Settings -> Secrets* as:  FINNHUB_API_KEY="..."
(Streamlit also exposes secrets as environment variables, which is what we read.)
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

_SAMPLE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "sample_data")

FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY", "").strip()
FINNHUB = "https://finnhub.io/api/v1"
STOOQ = "https://stooq.com"
MARKET_TICKER = "SPY"  # benchmark used for beta
TIMEOUT = 10

# True only when the user-visible PRICES are demo data (drives the UI banner).
# History/news falling back stay silent so we don't cry wolf when prices are live.
DEMO_MODE = False

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (compatible; PortfolioAdvisor/1.0)"})


def _load_sample(name: str) -> dict:
    with open(os.path.join(_SAMPLE_DIR, f"{name}.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _go_demo() -> None:
    global DEMO_MODE
    DEMO_MODE = True


# --------------------------------------------------------------------------- #
# Symbol helpers
# --------------------------------------------------------------------------- #
def _is_crypto(ticker: str) -> bool:
    return ticker.upper().endswith("-USD")


def _stooq_symbol(ticker: str) -> str:
    """Map a ticker to Stooq's symbol format.

    AAPL -> aapl.us | VOO -> voo.us | BTC-USD -> btcusd | D05.SI -> d05.sg
    """
    t = ticker.upper().strip()
    if t.endswith("-USD"):
        return t[:-4].lower() + "usd"
    if "." in t:
        base, suffix = t.split(".", 1)
        suffix = {"SI": "sg", "L": "uk", "HK": "hk"}.get(suffix, suffix.lower())
        return f"{base.lower()}.{suffix}"
    return t.lower() + ".us"


# --------------------------------------------------------------------------- #
# Quotes
# --------------------------------------------------------------------------- #
def get_quote(ticker: str) -> dict:
    """Return ``{name, price, day_change_pct, sector[, asset_type]}``.

    asset_type is only set for crypto; for everything else it's omitted so the
    holding's own asset_type (stock/etf) wins downstream.
    """
    t = ticker.upper().strip()

    # Crypto: Finnhub free doesn't cover BTC-USD cleanly, so use Stooq.
    if _is_crypto(t):
        q = _stooq_quote(t)
        if q:
            return q
    elif FINNHUB_KEY:
        q = _finnhub_quote(t)
        if q:
            return q
    else:
        # No key: still try to show live prices via keyless Stooq.
        q = _stooq_quote(t)
        if q:
            return q

    _go_demo()
    return _demo_quote(t)


def _finnhub_quote(ticker: str) -> dict | None:
    try:
        r = _SESSION.get(
            f"{FINNHUB}/quote",
            params={"symbol": ticker, "token": FINNHUB_KEY},
            timeout=TIMEOUT,
        )
        data = r.json()
        price = data.get("c")
        if not price:  # 0 or None -> unknown symbol / no data
            return None
        name, sector = ticker, "Unknown"
        try:
            p = _SESSION.get(
                f"{FINNHUB}/stock/profile2",
                params={"symbol": ticker, "token": FINNHUB_KEY},
                timeout=TIMEOUT,
            ).json()
            name = p.get("name") or ticker
            sector = p.get("finnhubIndustry") or "Unknown"
        except Exception:
            pass
        return {
            "name": name,
            "price": float(price),
            "day_change_pct": float(data.get("dp") or 0.0),
            "sector": sector,
        }
    except Exception:
        return None


def _stooq_quote(ticker: str) -> dict | None:
    try:
        sym = _stooq_symbol(ticker)
        r = _SESSION.get(
            f"{STOOQ}/q/l/",
            params={"s": sym, "f": "sd2t2ohlcvn", "h": "", "e": "csv"},
            timeout=TIMEOUT,
        )
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return None
        row = df.iloc[0]
        close = row.get("Close")
        if close in (None, "N/D") or pd.isna(close):
            return None
        close = float(close)
        open_ = row.get("Open")
        day = 0.0
        try:
            if open_ not in (None, "N/D") and not pd.isna(open_) and float(open_):
                day = (close - float(open_)) / float(open_) * 100
        except Exception:
            day = 0.0
        crypto = _is_crypto(ticker)
        res = {
            "name": str(row.get("Name") or ticker),
            "price": close,
            "day_change_pct": day,
            "sector": "Cryptocurrency" if crypto else "Unknown",
        }
        if crypto:
            res["asset_type"] = "crypto"
        return res
    except Exception:
        return None


def _demo_quote(ticker: str) -> dict:
    q = _load_sample("quotes").get(ticker)
    if q is None:
        return {"name": ticker, "price": 100.0, "day_change_pct": 0.0, "sector": "Unknown"}
    return {
        "name": q["name"],
        "price": q["price"],
        "day_change_pct": q["day_change_pct"],
        "sector": q["sector"],
        "asset_type": q["asset_type"],
    }


# --------------------------------------------------------------------------- #
# Price history (for volatility / beta / drawdown)
# --------------------------------------------------------------------------- #
def get_history(ticker: str, period: str = "1y") -> pd.Series:
    """Daily closing prices. Uses Stooq; falls back to a synthetic series.

    A history miss does NOT flip DEMO_MODE — prices can still be live even if a
    single symbol's history is unavailable.
    """
    s = _stooq_history(ticker)
    if s is not None and len(s) > 5:
        return s
    return _synthetic_history(ticker.upper().strip())


def _stooq_history(ticker: str) -> pd.Series | None:
    try:
        sym = _stooq_symbol(ticker)
        r = _SESSION.get(f"{STOOQ}/q/d/l/", params={"s": sym, "i": "d"}, timeout=TIMEOUT)
        df = pd.read_csv(io.StringIO(r.text))
        if "Close" not in df.columns or df.empty:
            return None
        s = pd.Series(
            df["Close"].astype(float).values,
            index=pd.to_datetime(df["Date"]),
            name=ticker.upper(),
        ).dropna()
        return s.tail(252)  # ~1 trading year
    except Exception:
        return None


def get_market_history(period: str = "1y") -> pd.Series:
    return get_history(MARKET_TICKER, period=period)


_DEMO_DAYS = 252
_MARKET_ANNUAL_VOL = 0.15


@lru_cache(maxsize=1)
def _demo_market_returns() -> np.ndarray:
    rng = np.random.default_rng(20240101)
    daily_vol = _MARKET_ANNUAL_VOL / np.sqrt(252)
    return rng.normal(loc=0.0004, scale=daily_vol, size=_DEMO_DAYS)


@lru_cache(maxsize=64)
def _synthetic_history(ticker: str, days: int = _DEMO_DAYS) -> pd.Series:
    """Deterministic fallback series, correlated to the market so beta is sane."""
    quotes = _load_sample("quotes")
    q = quotes.get(ticker, {"price": 100.0, "annual_vol": 0.30, "beta": 1.0})
    end_price = float(q.get("price", 100.0))
    annual_vol = float(q.get("annual_vol", 0.30))
    beta = float(q.get("beta", 1.0))

    market = _demo_market_returns()[:days]
    market_daily_vol = _MARKET_ANNUAL_VOL / np.sqrt(252)
    target_daily_var = (annual_vol / np.sqrt(252)) ** 2
    noise_var = max(target_daily_var - (beta**2) * (market_daily_vol**2), 1e-8)

    seed = abs(hash(ticker)) % (2**32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_var), size=days)
    daily_ret = 0.0004 + beta * market + noise

    path = np.cumprod(1 + daily_ret)
    path = path / path[-1] * end_price
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.Series(path, index=idx, name=ticker)


# --------------------------------------------------------------------------- #
# News
# --------------------------------------------------------------------------- #
def get_news(ticker: str, limit: int = 6) -> list[dict]:
    """Recent headlines: ``[{title, publisher, link, published}, ...]``.

    Uses Finnhub company-news (needs the key, stocks/ETFs only). On any miss it
    quietly returns bundled sample headlines without flipping DEMO_MODE.
    """
    t = ticker.upper().strip()
    if FINNHUB_KEY and not _is_crypto(t):
        try:
            to = datetime.utcnow().date()
            frm = to - timedelta(days=14)
            r = _SESSION.get(
                f"{FINNHUB}/company-news",
                params={"symbol": t, "from": frm.isoformat(), "to": to.isoformat(), "token": FINNHUB_KEY},
                timeout=TIMEOUT,
            )
            arr = r.json()
            items = []
            for n in arr[:limit]:
                title = n.get("headline")
                if not title:
                    continue
                ts = n.get("datetime")
                items.append(
                    {
                        "title": title,
                        "publisher": n.get("source", ""),
                        "link": n.get("url", ""),
                        "published": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else "",
                    }
                )
            if items:
                return items
        except Exception:
            pass
    return _load_sample("news").get(t, [])[:limit]
