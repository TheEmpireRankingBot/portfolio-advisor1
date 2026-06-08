"""Portfolio Advisor — a beginner-friendly trading portfolio tracker and coach.

Modules:
    data_fetch  -- live market data via yfinance, with an offline demo fallback
    metrics     -- returns, volatility, beta, drawdown, concentration (HHI)
    risk_score  -- the 0-100 "discipline score" + red flags (gambling vs calculated risk)
    weaknesses  -- plain-English weakness findings that teach as they diagnose
    news        -- recent headlines per holding, tagged good / bad / neutral
    education   -- glossary + lessons + tooltips for beginners
    ai_coach    -- optional natural-language coaching via the Claude API
"""

__version__ = "0.1.0"
