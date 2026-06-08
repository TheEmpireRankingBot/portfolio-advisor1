"""Optional natural-language coaching via the Claude API.

This is a *bonus* layer on top of the rules engine. If an ``ANTHROPIC_API_KEY``
is available (and the network allows it), we ask Claude to turn the numbers the
rules engine already computed into warm, mentor-style coaching for a beginner.

If there's no key, the SDK isn't installed, or the call fails for any reason, we
fall back to a deterministic rules-based summary so the feature *always* returns
something useful. The model never invents data — it only reframes the metrics we
hand it.
"""

from __future__ import annotations

import os

MODEL = "claude-sonnet-4-6"

_SYSTEM = (
    "You are a friendly, plain-spoken senior investment mentor coaching a "
    "BEGINNER investor. Be encouraging but honest. Explain any jargon in one "
    "short clause. Never invent numbers — only use the metrics provided. Do not "
    "give specific buy/sell price targets or guarantees. Keep it to ~150-220 "
    "words, structured as: a one-line read on whether this is calculated risk or "
    "gambling, then 2-4 concrete, prioritised suggestions, then one encouraging "
    "sentence. End by reminding them this is educational, not financial advice."
)


def is_available() -> bool:
    """True if we *could* call Claude (key present and SDK importable)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401

        return True
    except Exception:
        return False


def _rules_fallback(context: dict) -> str:
    """Deterministic coaching text when the API isn't available."""
    disc = context["discipline"]
    lines = [
        f"**Read:** {disc['label']} — {disc['verdict']}",
        "",
        "**What I'd focus on first:**",
    ]
    if disc["flags"]:
        for f in disc["flags"][:4]:
            lines.append(f"- **{f['title']}** — {f['fix']}")
    else:
        lines.append("- Nothing urgent — keep contributing steadily and stay diversified.")
    lines += [
        "",
        "You're already doing the most important thing: looking at your portfolio "
        "honestly instead of ignoring it. Small, consistent adjustments compound.",
        "",
        "_Educational use only — this is not financial advice._",
    ]
    return "\n".join(lines)


def _build_prompt(context: dict) -> str:
    disc = context["discipline"]
    summary = context["summary"]
    flags = "; ".join(f"{f['title']} (-{f['points']})" for f in disc["flags"]) or "none"
    holdings = ", ".join(
        f"{p['ticker']} {p['weight']:.0%}" for p in context["positions"]
    )
    return (
        "Here is the investor's portfolio analysis (already computed):\n"
        f"- Total value: {summary['total_value']:.0f} {context.get('currency','USD')}\n"
        f"- Total profit/loss: {summary['total_pnl_pct']:.1f}%\n"
        f"- Number of holdings: {summary['num_holdings']}\n"
        f"- Holdings & weights: {holdings}\n"
        f"- Blended volatility: {context['volatility']:.0%}/yr; "
        f"blended beta: {context['beta']:.2f}\n"
        f"- Discipline score: {disc['score']}/100 ({disc['label']})\n"
        f"- Risk flags: {flags}\n\n"
        "Coach this beginner based ONLY on the above."
    )


def generate_coaching(context: dict) -> tuple[str, str]:
    """Return (coaching_text, source) where source is 'ai' or 'rules'."""
    if not is_available():
        return _rules_fallback(context), "rules"
    try:
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=MODEL,
            max_tokens=600,
            system=_SYSTEM,
            messages=[{"role": "user", "content": _build_prompt(context)}],
        )
        text = "".join(block.text for block in msg.content if block.type == "text")
        return text.strip() or _rules_fallback(context), "ai"
    except Exception:
        return _rules_fallback(context), "rules"
