"""Recent headlines per holding, tagged good / bad / neutral.

Sentiment uses VADER, a lightweight rule-based model that runs entirely offline
(no API key, no network) — perfect for tagging short news headlines. We keep the
thresholds slightly wider than VADER's defaults so only clearly positive or
clearly negative headlines get a colour; everything else stays neutral.
"""

from __future__ import annotations

from functools import lru_cache

from . import data_fetch


@lru_cache(maxsize=1)
def _analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def classify_sentiment(text: str) -> tuple[str, float]:
    """Return ('good' | 'bad' | 'neutral', compound_score) for a headline."""
    score = _analyzer().polarity_scores(text or "")["compound"]
    if score >= 0.20:
        return "good", score
    if score <= -0.20:
        return "bad", score
    return "neutral", score


def get_holding_news(ticker: str, limit: int = 6) -> list[dict]:
    """Headlines for one ticker, each annotated with sentiment + emoji."""
    items = data_fetch.get_news(ticker, limit=limit)
    out = []
    for item in items:
        label, score = classify_sentiment(item.get("title", ""))
        emoji = {"good": "🟢", "bad": "🔴", "neutral": "⚪"}[label]
        out.append({**item, "sentiment": label, "sentiment_score": score, "emoji": emoji})
    return out


def get_portfolio_news(tickers: list[str], limit_per: int = 4) -> dict[str, list[dict]]:
    """Map of ticker -> annotated headlines for the whole portfolio."""
    return {t: get_holding_news(t, limit=limit_per) for t in tickers}


def sentiment_tally(news_by_ticker: dict[str, list[dict]]) -> dict[str, int]:
    """Count good / bad / neutral across the whole portfolio for a quick read."""
    tally = {"good": 0, "bad": 0, "neutral": 0}
    for items in news_by_ticker.values():
        for n in items:
            tally[n["sentiment"]] += 1
    return tally
