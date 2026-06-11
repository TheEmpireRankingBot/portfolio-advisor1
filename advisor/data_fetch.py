"""Market data access (cloud-friendly).

Sources (all chosen to work from cloud servers like Streamlit Cloud):

    * Stock/ETF quotes  -> Finnhub (free FINNHUB_API_KEY) with Stooq keyless fallback.
    * Crypto quotes     -> CoinGecko (keyless) with Stooq fallback.
    * Price history     -> Stooq daily CSV (keyless); CoinGecko for crypto.
    * Fundamentals      -> Finnhub /stock/metric (stocks), CoinGecko /coins (crypto).
    * Company news      -> Finnhub company-news; optional Alpha Vantage
                           (ALPHAVANTAGE_API_KEY) which also covers crypto news.
    * Market news       -> Finnhub general news.
    * Macro series      -> FRED fredgraph CSV (keyless).

If a source is unreachable the functions degrade gracefully: bundled
``sample_data/`` for quotes/news, a deterministic synthetic series for history.
``DEMO_MODE`` flips only when user-visible PRICES had to fall back.

Keys go in Streamlit *Settings -> Secrets* (exposed as env vars):
    FINNHUB_API_KEY = "..."        # quotes/news/fundamentals for stocks & ETFs
    ALPHAVANTAGE_API_KEY = "..."   # optional: richer news incl. crypto
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
ALPHAVANTAGE_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "").strip()
FINNHUB = "https://finnhub.io/api/v1"
STOOQ = "https://stooq.com"
COINGECKO = "https://api.coingecko.com/api/v3"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
MARKET_TICKER = "SPY"  # benchmark used for beta
TIMEOUT = 10

# True only when user-visible PRICES are demo data. History/news falling back
# stay silent so we don't cry wolf while prices are live.
DEMO_MODE = False

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (compatible; PortfolioAdvisor/1.0)"})

# Symbol -> CoinGecko id for the majors; anything else resolves via /search.
CRYPTO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple",
    "DOGE": "dogecoin", "ADA": "cardano", "BNB": "binancecoin", "LTC": "litecoin",
    "DOT": "polkadot", "LINK": "chainlink", "AVAX": "avalanche-2",
    "MATIC": "matic-network", "TRX": "tron", "SHIB": "shiba-inu",
    "UNI": "uniswap", "XLM": "stellar", "NEAR": "near", "ATOM": "cosmos",
}


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
    t = ticker.upper()
    return t.endswith("-USD") or t in CRYPTO_IDS


def _crypto_base(ticker: str) -> str:
    """BTC-USD -> BTC, BTC -> BTC."""
    t = ticker.upper()
    return t[:-4] if t.endswith("-USD") else t


@lru_cache(maxsize=64)
def _coingecko_id(ticker: str) -> str | None:
    base = _crypto_base(ticker)
    if base in CRYPTO_IDS:
        return CRYPTO_IDS[base]
    try:  # resolve unknown symbols via CoinGecko search
        r = _SESSION.get(f"{COINGECKO}/search", params={"query": base}, timeout=TIMEOUT)
        for coin in r.json().get("coins", []):
            if coin.get("symbol", "").upper() == base:
                return coin.get("id")
    except Exception:
        pass
    return None


def _stooq_symbol(ticker: str) -> str:
    """AAPL -> aapl.us | VOO -> voo.us | BTC-USD -> btcusd | D05.SI -> d05.sg"""
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

    asset_type is only set for crypto; otherwise the holding's own type wins.
    """
    t = ticker.upper().strip()

    if _is_crypto(t):
        q = _coingecko_quote(t) or _stooq_quote(t)
        if q:
            return q
    elif FINNHUB_KEY:
        q = _finnhub_quote(t) or _stooq_quote(t)
        if q:
            return q
    else:
        q = _stooq_quote(t)
        if q:
            return q

    _go_demo()
    return _demo_quote(t if t.endswith("-USD") or not _is_crypto(t) else t + "-USD")


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


def _coingecko_quote(ticker: str) -> dict | None:
    cid = _coingecko_id(ticker)
    if not cid:
        return None
    try:
        r = _SESSION.get(
            f"{COINGECKO}/simple/price",
            params={
                "ids": cid,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true",
            },
            timeout=TIMEOUT,
        )
        d = r.json().get(cid) or {}
        price = d.get("usd")
        if not price:
            return None
        return {
            "name": cid.replace("-", " ").title(),
            "price": float(price),
            "day_change_pct": float(d.get("usd_24h_change") or 0.0),
            "sector": "Cryptocurrency",
            "asset_type": "crypto",
            "market_cap": d.get("usd_market_cap"),
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
# Price history (for volatility / beta / drawdown / charts)
# --------------------------------------------------------------------------- #
def get_history(ticker: str, period: str = "1y") -> pd.Series:
    """Daily closing prices (~1y). A miss does NOT flip DEMO_MODE."""
    t = ticker.upper().strip()
    if _is_crypto(t):
        s = get_crypto_history(t) or _stooq_history(t)
    else:
        s = _stooq_history(t)
    if s is not None and len(s) > 5:
        return s
    return _synthetic_history(t)


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
        return s.tail(252)
    except Exception:
        return None


def get_crypto_history(ticker: str, days: int = 365) -> pd.Series | None:
    """Daily USD closes from CoinGecko (auto-daily granularity for days > 90)."""
    cid = _coingecko_id(ticker)
    if not cid:
        return None
    try:
        r = _SESSION.get(
            f"{COINGECKO}/coins/{cid}/market_chart",
            params={"vs_currency": "usd", "days": days},
            timeout=TIMEOUT,
        )
        prices = r.json().get("prices") or []
        if len(prices) < 5:
            return None
        idx = pd.to_datetime([p[0] for p in prices], unit="ms").normalize()
        s = pd.Series([p[1] for p in prices], index=idx, name=ticker.upper())
        return s.groupby(s.index).last()  # collapse intraday points to daily
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
# Fundamentals (terminal `F` function)
# --------------------------------------------------------------------------- #
def get_fundamentals(ticker: str) -> dict:
    """Key metrics for a symbol. Stocks via Finnhub; crypto via CoinGecko.

    Returns a dict whose values may be None when a metric isn't available
    (e.g. no API key). ``kind`` is "stock" or "crypto".
    """
    t = ticker.upper().strip()
    if _is_crypto(t):
        return _crypto_fundamentals(t)

    out: dict = {"kind": "stock", "pe": None, "eps": None, "market_cap": None,
                 "high_52w": None, "low_52w": None, "div_yield": None, "beta": None}
    if FINNHUB_KEY:
        try:
            r = _SESSION.get(
                f"{FINNHUB}/stock/metric",
                params={"symbol": t, "metric": "all", "token": FINNHUB_KEY},
                timeout=TIMEOUT,
            )
            m = r.json().get("metric") or {}
            out["pe"] = m.get("peTTM") or m.get("peBasicExclExtraTTM")
            out["eps"] = m.get("epsTTM") or m.get("epsBasicExclExtraItemsTTM")
            mc = m.get("marketCapitalization")
            out["market_cap"] = mc * 1e6 if mc else None  # Finnhub reports millions
            out["high_52w"] = m.get("52WeekHigh")
            out["low_52w"] = m.get("52WeekLow")
            out["div_yield"] = m.get("dividendYieldIndicatedAnnual")
            out["beta"] = m.get("beta")
        except Exception:
            pass
    if out["high_52w"] is None or out["low_52w"] is None:
        # Keyless fallback: derive the 52-week range from price history.
        hist = get_history(t)
        if len(hist):
            out["high_52w"] = out["high_52w"] or float(hist.max())
            out["low_52w"] = out["low_52w"] or float(hist.min())
    return out


def _crypto_fundamentals(ticker: str) -> dict:
    out: dict = {"kind": "crypto", "market_cap": None, "rank": None, "volume_24h": None,
                 "ath": None, "from_ath_pct": None, "high_24h": None, "low_24h": None}
    cid = _coingecko_id(ticker)
    if not cid:
        return out
    try:
        r = _SESSION.get(
            f"{COINGECKO}/coins/{cid}",
            params={
                "localization": "false", "tickers": "false", "market_data": "true",
                "community_data": "false", "developer_data": "false",
            },
            timeout=TIMEOUT,
        )
        md = r.json().get("market_data") or {}

        def usd(field):
            v = md.get(field)
            return v.get("usd") if isinstance(v, dict) else v

        out["market_cap"] = usd("market_cap")
        out["rank"] = md.get("market_cap_rank")
        out["volume_24h"] = usd("total_volume")
        out["ath"] = usd("ath")
        out["from_ath_pct"] = usd("ath_change_percentage")
        out["high_24h"] = usd("high_24h")
        out["low_24h"] = usd("low_24h")
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# Macro series (FRED, keyless)
# --------------------------------------------------------------------------- #
def get_macro_series(fred_id: str) -> pd.Series | None:
    """Full history of a FRED series via the keyless fredgraph CSV endpoint."""
    try:
        r = _SESSION.get(FRED_CSV, params={"id": fred_id.upper()}, timeout=TIMEOUT)
        df = pd.read_csv(io.StringIO(r.text))
        if df.shape[1] < 2 or df.empty:
            return None
        s = pd.Series(
            pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
            index=pd.to_datetime(df.iloc[:, 0], errors="coerce"),
            name=fred_id.upper(),
        ).dropna()
        return s if len(s) else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# News
# --------------------------------------------------------------------------- #
def get_news(ticker: str, limit: int = 6) -> list[dict]:
    """Recent headlines for one symbol: ``[{title, publisher, link, published}]``.

    Finnhub (stocks) -> Alpha Vantage if keyed (also covers crypto) -> samples.
    """
    t = ticker.upper().strip()
    if FINNHUB_KEY and not _is_crypto(t):
        items = _finnhub_company_news(t, limit)
        if items:
            return items
    if ALPHAVANTAGE_KEY:
        items = _alpha_vantage_news(t, limit)
        if items:
            return items
    key = t if t in _load_sample("news") else (t + "-USD" if _is_crypto(t) else t)
    return _load_sample("news").get(key, [])[:limit]


def _finnhub_company_news(ticker: str, limit: int) -> list[dict]:
    try:
        to = datetime.utcnow().date()
        frm = to - timedelta(days=14)
        r = _SESSION.get(
            f"{FINNHUB}/company-news",
            params={"symbol": ticker, "from": frm.isoformat(), "to": to.isoformat(),
                    "token": FINNHUB_KEY},
            timeout=TIMEOUT,
        )
        items = []
        for n in r.json()[:limit]:
            title = n.get("headline")
            if not title:
                continue
            ts = n.get("datetime")
            items.append({
                "title": title,
                "publisher": n.get("source", ""),
                "link": n.get("url", ""),
                "published": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else "",
            })
        return items
    except Exception:
        return []


def _alpha_vantage_news(ticker: str, limit: int) -> list[dict]:
    try:
        av_symbol = f"CRYPTO:{_crypto_base(ticker)}" if _is_crypto(ticker) else ticker
        r = _SESSION.get(
            "https://www.alphavantage.co/query",
            params={"function": "NEWS_SENTIMENT", "tickers": av_symbol,
                    "limit": limit, "apikey": ALPHAVANTAGE_KEY},
            timeout=TIMEOUT,
        )
        items = []
        for n in (r.json().get("feed") or [])[:limit]:
            title = n.get("title")
            if not title:
                continue
            ts = n.get("time_published", "")  # e.g. 20260608T101500
            published = f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else ""
            items.append({
                "title": title,
                "publisher": n.get("source", ""),
                "link": n.get("url", ""),
                "published": published,
            })
        return items
    except Exception:
        return []


def get_market_news(limit: int = 8) -> list[dict]:
    """General market headlines (Finnhub). Falls back to bundled samples."""
    if FINNHUB_KEY:
        try:
            r = _SESSION.get(
                f"{FINNHUB}/news",
                params={"category": "general", "token": FINNHUB_KEY},
                timeout=TIMEOUT,
            )
            items = []
            for n in r.json()[:limit]:
                title = n.get("headline")
                if not title:
                    continue
                ts = n.get("datetime")
                items.append({
                    "title": title,
                    "publisher": n.get("source", ""),
                    "link": n.get("url", ""),
                    "published": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else "",
                })
            if items:
                return items
        except Exception:
            pass
    flat: list[dict] = []
    for arr in _load_sample("news").values():
        flat.extend(arr)
    flat.sort(key=lambda n: n.get("published", ""), reverse=True)
    return flat[:limit]
