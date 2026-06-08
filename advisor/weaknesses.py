"""Weakness diagnosis that teaches while it diagnoses.

Where ``risk_score`` produces a single number, this module produces a richer,
categorised health-check of the portfolio: diversification, concentration,
sector balance, volatility, and the stock-vs-fund mix. Each finding is labelled
good / warning / bad and written so a beginner learns the concept from the
explanation itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import metrics


@dataclass
class Finding:
    category: str
    status: str  # "good" | "warning" | "bad"
    headline: str
    detail: str  # plain-English explanation that teaches the concept


def analyze(positions: list[dict], per_ticker_vol: dict, per_ticker_beta: dict) -> list[Finding]:
    findings: list[Finding] = []
    if not positions:
        return findings

    num = len(positions)
    weights = [p["weight"] for p in positions]
    hhi = metrics.hhi(weights)
    largest = max(positions, key=lambda p: p["weight"])

    # Diversification -------------------------------------------------------
    if num >= 8:
        findings.append(
            Finding(
                "Diversification",
                "good",
                f"Well spread across {num} holdings",
                "Diversification means not relying on any single investment. With "
                f"{num} holdings, a stumble by one company won't sink your portfolio.",
            )
        )
    elif num >= 5:
        findings.append(
            Finding(
                "Diversification",
                "warning",
                f"Moderately diversified ({num} holdings)",
                "You've spread your money across several names, which is good. Adding "
                "a few more — ideally in different industries — would cushion you "
                "further against any one of them dropping.",
            )
        )
    else:
        findings.append(
            Finding(
                "Diversification",
                "bad",
                f"Under-diversified ({num} holdings)",
                "With only a handful of holdings, your fortunes hang on a few "
                "companies. A single broad index ETF instantly spreads your money "
                "across hundreds of firms — the easiest diversification fix there is.",
            )
        )

    # Concentration ---------------------------------------------------------
    if largest["weight"] > 0.30:
        findings.append(
            Finding(
                "Concentration",
                "bad",
                f"{largest['ticker']} is {largest['weight']:.0%} of the portfolio",
                "Concentration is the flip side of diversification. When one "
                "position is this large, your results are really just that one "
                "stock's results. Trimming it back to ~10-20% reduces single-stock "
                "risk.",
            )
        )
    elif largest["weight"] > 0.20:
        findings.append(
            Finding(
                "Concentration",
                "warning",
                f"{largest['ticker']} is your biggest position ({largest['weight']:.0%})",
                "Not alarming, but worth watching. If you keep adding to a winner it "
                "can quietly grow into an outsized risk.",
            )
        )
    else:
        findings.append(
            Finding(
                "Concentration",
                "good",
                f"Balanced position sizes (largest is {largest['weight']:.0%})",
                "No single holding dominates, so no single surprise can wreck your "
                "year. The concentration index (HHI) is "
                f"{hhi:.2f} — lower is more spread out.",
            )
        )

    # Sector balance --------------------------------------------------------
    sector_alloc = metrics.allocation_by(positions, "sector")
    top_sector, top_w = next(iter(sector_alloc.items()))
    if top_w > 0.50:
        findings.append(
            Finding(
                "Sector balance",
                "bad",
                f"{top_w:.0%} concentrated in {top_sector}",
                "Stocks in the same sector tend to rise and fall together. Being "
                f"this heavy in {top_sector} means a sector-wide downturn hits almost "
                "everything you own at once. Spreading across unrelated sectors "
                "smooths this out.",
            )
        )
    elif top_w > 0.35:
        findings.append(
            Finding(
                "Sector balance",
                "warning",
                f"Tilted toward {top_sector} ({top_w:.0%})",
                "A sector tilt is fine if it's intentional — just make sure it "
                "reflects a view you actually hold, not an accident of what you "
                "happened to buy.",
            )
        )
    else:
        findings.append(
            Finding(
                "Sector balance",
                "good",
                "Reasonable spread across sectors",
                f"Your largest sector is {top_sector} at {top_w:.0%}. Spreading "
                "across industries means they won't all slump together.",
            )
        )

    # Volatility ------------------------------------------------------------
    port_vol = metrics.weighted_metric(positions, per_ticker_vol)
    if port_vol > 0.45:
        findings.append(
            Finding(
                "Volatility",
                "bad",
                f"High volatility (~{port_vol:.0%}/yr)",
                "Volatility measures how violently your portfolio's value swings. "
                "Very high volatility is emotionally hard to hold — it's what makes "
                "people panic-sell at the bottom. Steadier assets calm the ride.",
            )
        )
    elif port_vol > 0.30:
        findings.append(
            Finding(
                "Volatility",
                "warning",
                f"Above-average volatility (~{port_vol:.0%}/yr)",
                "Expect some bumpy stretches. That's acceptable if your time horizon "
                "is long and you won't be forced to sell during a dip.",
            )
        )
    else:
        findings.append(
            Finding(
                "Volatility",
                "good",
                f"Manageable volatility (~{port_vol:.0%}/yr)",
                "Your portfolio shouldn't swing so hard that it tempts panic "
                "decisions — that makes it much easier to stay invested.",
            )
        )

    # Stock vs fund mix -----------------------------------------------------
    fund_w = sum(p["weight"] for p in positions if p["asset_type"] == "etf")
    if fund_w < 0.10 and num < 8:
        findings.append(
            Finding(
                "Core stability",
                "warning",
                "Almost entirely individual stocks",
                "Individual stocks are higher-risk than diversified funds. A low-cost "
                "index ETF as your 'core' gives instant diversification, letting you "
                "take focused bets with the rest — a common, sensible structure.",
            )
        )
    elif fund_w >= 0.30:
        findings.append(
            Finding(
                "Core stability",
                "good",
                f"Solid fund core ({fund_w:.0%} in ETFs)",
                "A meaningful ETF allocation gives your portfolio a stable, "
                "diversified backbone — a hallmark of calculated, not reckless, "
                "investing.",
            )
        )

    return findings
