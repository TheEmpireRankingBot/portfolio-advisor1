"""The signature feature: a transparent 0-100 "discipline score".

This answers the user's real question -- *"am I investing, or just gambling?"* --
without pretending to be a crystal ball. It starts every portfolio at a perfect
100 and subtracts points for habits that push a portfolio from *calculated risk*
toward *gambling*: betting too big on one name, owning too few things, piling
into a single sector, or chasing very volatile assets.

Crucially it is **transparent**: every deduction comes back with the points
lost, *why* it matters, and the *disciplined alternative*. Nothing is a black
box, which is exactly what a beginner needs in order to learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Flag:
    title: str
    points: int  # points deducted (positive number)
    severity: str  # "high" | "medium" | "low"
    why: str  # plain-English reason this hurts you
    fix: str  # what a disciplined investor would do


@dataclass
class DisciplineResult:
    score: int
    label: str
    verdict: str  # one-line gambling-vs-calculated summary
    flags: list[Flag] = field(default_factory=list)
    positives: list[str] = field(default_factory=list)


def _label_and_verdict(score: int) -> tuple[str, str]:
    if score >= 80:
        return (
            "Calculated risk-taker",
            "This looks like investing, not gambling. Your risks are mostly "
            "deliberate and spread out — keep it up.",
        )
    if score >= 60:
        return (
            "Mostly calculated — a few gaps",
            "You're investing thoughtfully overall, but a couple of habits below "
            "are nudging you toward unnecessary risk.",
        )
    if score >= 40:
        return (
            "Mixed — leaning risky",
            "There's a real strategy here, but several big risk flags mean a bad "
            "week could hurt a lot more than it should.",
        )
    return (
        "Leaning toward gambling",
        "Right now this behaves more like a bet than an investment plan. The good "
        "news: every flag below is fixable, and small changes move the needle fast.",
    )


def compute_discipline_score(
    positions: list[dict],
    weighted_volatility: float,
    weighted_beta: float,
    portfolio_hhi: float,
) -> DisciplineResult:
    """Score the portfolio and return the full, explained breakdown.

    Parameters mirror what ``metrics.py`` produces so the UI can compute once
    and pass the results straight in.
    """
    flags: list[Flag] = []
    positives: list[str] = []
    score = 100

    num = len(positions)
    weights = {p["ticker"]: p["weight"] for p in positions}
    largest_ticker = max(weights, key=weights.get) if weights else None
    largest_w = weights.get(largest_ticker, 0.0) if largest_ticker else 0.0

    # 1) Single-position concentration ------------------------------------- #
    if largest_w > 0.40:
        d = 25
        score -= d
        flags.append(
            Flag(
                f"Over-concentrated in {largest_ticker}",
                d,
                "high",
                f"{largest_w:.0%} of your money is riding on one position. If that "
                "single name drops 30%, your whole portfolio takes a serious hit.",
                "Trim oversized winners and cap any single position around 10-20% "
                "of the portfolio.",
            )
        )
    elif largest_w > 0.25:
        d = 12
        score -= d
        flags.append(
            Flag(
                f"Large single position in {largest_ticker}",
                d,
                "medium",
                f"{largest_w:.0%} in one holding is on the heavy side — your results "
                "will swing mostly with this one name.",
                "Consider keeping individual positions under ~20% so no single "
                "stock can dominate your outcome.",
            )
        )
    else:
        positives.append("No single position dominates the portfolio — good balance.")

    # 2) Too few holdings -------------------------------------------------- #
    if num < 3:
        d = 20
        score -= d
        flags.append(
            Flag(
                "Very few holdings",
                d,
                "high",
                f"With only {num} holding(s), you have almost no diversification — "
                "you're exposed to the fate of just one or two companies.",
                "Build toward at least 8-15 positions, or use a broad index ETF to "
                "instantly own hundreds of companies.",
            )
        )
    elif num < 5:
        d = 10
        score -= d
        flags.append(
            Flag(
                "Thin diversification",
                d,
                "medium",
                f"{num} holdings is a start, but one bad name still moves the whole "
                "portfolio a lot.",
                "Add a few more positions across different sectors, or anchor the "
                "portfolio with a low-cost index ETF.",
            )
        )
    else:
        positives.append(f"You hold {num} positions — a reasonable spread.")

    # 3) Sector concentration --------------------------------------------- #
    sector_w: dict[str, float] = {}
    for p in positions:
        sector_w[p["sector"]] = sector_w.get(p["sector"], 0.0) + p["weight"]
    if sector_w:
        top_sector = max(sector_w, key=sector_w.get)
        top_sector_w = sector_w[top_sector]
        if top_sector_w > 0.60:
            d = 15
            score -= d
            flags.append(
                Flag(
                    f"Heavy bet on one sector ({top_sector})",
                    d,
                    "high",
                    f"{top_sector_w:.0%} of the portfolio is in {top_sector}. When "
                    "that sector falls out of favour, everything drops together.",
                    "Spread across unrelated sectors (e.g. tech, healthcare, "
                    "financials, consumer staples) so they don't all sink at once.",
                )
            )
        elif top_sector_w > 0.40:
            d = 7
            score -= d
            flags.append(
                Flag(
                    f"Tilted toward {top_sector}",
                    d,
                    "low",
                    f"{top_sector_w:.0%} in {top_sector} is a meaningful tilt — fine "
                    "if intentional, risky if accidental.",
                    "Check that this concentration is a deliberate view, not just "
                    "where you happened to buy.",
                )
            )

    # 4) Chasing volatility ------------------------------------------------ #
    if weighted_volatility > 0.50:
        d = 15
        score -= d
        flags.append(
            Flag(
                "Very high overall volatility",
                d,
                "high",
                f"Your blended volatility is ~{weighted_volatility:.0%} a year. Wild "
                "swings tempt panic-selling at the worst moments — a classic gambling "
                "trap.",
                "Mix in steadier assets (broad ETFs, large stable companies) to calm "
                "the ride so you can actually stick to your plan.",
            )
        )
    elif weighted_volatility > 0.35:
        d = 8
        score -= d
        flags.append(
            Flag(
                "Elevated volatility",
                d,
                "medium",
                f"Blended volatility of ~{weighted_volatility:.0%} means noticeably "
                "bumpy returns.",
                "That can be fine if you won't need the money soon — just size "
                "positions so a rough patch won't force your hand.",
            )
        )

    # 5) High market sensitivity (beta) ----------------------------------- #
    if weighted_beta > 1.5:
        d = 10
        score -= d
        flags.append(
            Flag(
                "Amplifies market moves",
                d,
                "medium",
                f"A portfolio beta near {weighted_beta:.1f} means you fall (and rise) "
                "much harder than the market — great in a boom, brutal in a bust.",
                "Add some low-beta holdings if you want to dampen the swings.",
            )
        )

    # 6) Crypto / speculative weight -------------------------------------- #
    crypto_w = sum(p["weight"] for p in positions if p["asset_type"] == "crypto")
    if crypto_w > 0.30:
        d = 15
        score -= d
        flags.append(
            Flag(
                "Large speculative (crypto) allocation",
                d,
                "high",
                f"{crypto_w:.0%} in crypto is a big bet on a very volatile, "
                "sentiment-driven asset. Sizing it this large is closer to gambling "
                "than investing.",
                "A common rule of thumb is to cap highly speculative assets at "
                "5-10% — money you could afford to lose entirely.",
            )
        )
    elif crypto_w > 0.10:
        d = 6
        score -= d
        flags.append(
            Flag(
                "Notable speculative allocation",
                d,
                "low",
                f"{crypto_w:.0%} in crypto is meaningful for such a volatile asset.",
                "Make sure this is genuinely money you can afford to lose, and keep "
                "it as a satellite — not the core of your portfolio.",
            )
        )

    # 7) Raw concentration index ------------------------------------------ #
    if portfolio_hhi > 0.40:
        d = 10
        score -= d
        flags.append(
            Flag(
                "High concentration index (HHI)",
                d,
                "medium",
                f"Your concentration score is {portfolio_hhi:.2f} (1.0 = everything "
                "in one asset). The lower this is, the more your eggs are spread "
                "across baskets.",
                "Adding more, similarly-sized positions naturally pushes this down.",
            )
        )

    score = max(0, min(100, score))
    label, verdict = _label_and_verdict(score)
    return DisciplineResult(
        score=score, label=label, verdict=verdict, flags=flags, positives=positives
    )
