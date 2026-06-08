"""Beginner education: a glossary of tooltips and a set of short lessons.

The glossary entries are reused as Streamlit tooltips across the app (e.g. next
to the word "Beta" on the Risk page), so a beginner can hover any jargon term
and get a plain-English definition right where they meet it.
"""

from __future__ import annotations

# term -> one-line plain-English definition (used as hover tooltips)
GLOSSARY: dict[str, str] = {
    "Diversification": "Spreading money across many investments so no single one can sink you.",
    "Concentration": "The opposite of diversification — too much riding on one position.",
    "Volatility": "How much a price bounces around. Higher = bigger swings = more risk.",
    "Beta": "How much a holding moves vs the whole market. 1.0 = moves with it; 2.0 = twice as much.",
    "Max drawdown": "The worst peak-to-trough drop — how much pain you'd have felt at the lowest point.",
    "HHI": "A concentration score from 0 to 1. Higher means more eggs in one basket.",
    "Cost basis": "The total amount you originally paid for a holding.",
    "Unrealized P/L": "Paper profit or loss on what you still hold — not locked in until you sell.",
    "Position size": "How big one holding is relative to your whole portfolio.",
    "Sector": "The industry a company belongs to, e.g. Technology or Healthcare.",
    "ETF": "A fund you buy like a stock that instantly holds many companies — easy diversification.",
    "Dollar-cost averaging": "Investing a fixed amount on a schedule, so you buy more when prices are low.",
    "Risk tolerance": "How much loss you can stomach — financially and emotionally — without panicking.",
}


# Short lessons shown on the Learn tab. Kept punchy on purpose.
LESSONS: list[dict] = [
    {
        "title": "Investing vs. gambling",
        "body": (
            "The line isn't the asset — it's the **process**. Gambling is an "
            "outsized, undiversified bet placed on hope, with no plan for being "
            "wrong. Calculated risk is a position you've sized deliberately, "
            "diversified sensibly, and can hold through a downturn. This app's "
            "**Discipline Score** measures exactly that process — not whether you "
            "got lucky."
        ),
    },
    {
        "title": "Why diversification is the only 'free lunch'",
        "body": (
            "Owning many uncorrelated investments lowers your risk *without* "
            "necessarily lowering your expected return — that's why it's called the "
            "only free lunch in finance. The simplest version: a low-cost index "
            "ETF that holds hundreds of companies in one click."
        ),
    },
    {
        "title": "Position sizing: how big is too big?",
        "body": (
            "A common guideline for beginners is to keep any single stock under "
            "~10-20% of your portfolio, and highly speculative bets (like crypto or "
            "tiny companies) to ~5-10% — money you could afford to lose entirely. "
            "Sizing, not stock-picking, is what keeps one mistake from being fatal."
        ),
    },
    {
        "title": "Volatility is the price of admission",
        "body": (
            "Higher returns usually come with bigger swings. The danger isn't the "
            "swing itself — it's selling in a panic at the bottom. Match your risk "
            "to your time horizon: money you need soon shouldn't be in volatile "
            "assets at all."
        ),
    },
    {
        "title": "Time in the market beats timing the market",
        "body": (
            "Trying to jump in and out perfectly is where most beginners lose. "
            "Investing steadily over time (dollar-cost averaging) removes the "
            "pressure to be right about the exact day — and tends to win out."
        ),
    },
]


def tip(term: str) -> str:
    """Look up a tooltip definition (empty string if the term isn't known)."""
    return GLOSSARY.get(term, "")
