"""Portfolio Advisor — Streamlit app.

Run locally for LIVE data:
    pip install -r requirements.txt
    streamlit run app.py

Inside a restricted sandbox (or offline) it automatically switches to DEMO mode
with illustrative sample data so every page still works.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from advisor import ai_coach, data_fetch, education, metrics, news, risk_score, terminal, weaknesses

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PORTFOLIO_PATH = os.path.join(DATA_DIR, "portfolio.json")
EXAMPLE_PATH = os.path.join(os.path.dirname(__file__), "portfolio.example.json")

st.set_page_config(page_title="Portfolio Advisor", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- #
# Portfolio storage
# --------------------------------------------------------------------------- #
def load_portfolio() -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(PORTFOLIO_PATH):
        with open(EXAMPLE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        save_portfolio(data)
        return data
    with open(PORTFOLIO_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_portfolio(data: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PORTFOLIO_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# --------------------------------------------------------------------------- #
# Cached analysis (recomputed when holdings change)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Fetching market data…")
def analyse(holdings_key: str) -> dict:
    """Heavy lifting: fetch quotes/history, compute every metric once."""
    holdings = json.loads(holdings_key)["holdings"]
    quotes = {h["ticker"].upper(): data_fetch.get_quote(h["ticker"]) for h in holdings}
    positions = metrics.build_positions(holdings, quotes)
    summary = metrics.portfolio_summary(positions)

    market_hist = data_fetch.get_market_history()
    per_vol, per_beta = {}, {}
    for p in positions:
        hist = data_fetch.get_history(p["ticker"])
        per_vol[p["ticker"]] = metrics.annualized_volatility(hist)
        per_beta[p["ticker"]] = metrics.beta(hist, market_hist)

    w_vol = metrics.weighted_metric(positions, per_vol)
    w_beta = metrics.weighted_metric(positions, per_beta)
    p_hhi = metrics.hhi([p["weight"] for p in positions])

    disc = risk_score.compute_discipline_score(positions, w_vol, w_beta, p_hhi)
    return {
        "positions": positions,
        "summary": summary,
        "per_vol": per_vol,
        "per_beta": per_beta,
        "w_vol": w_vol,
        "w_beta": w_beta,
        "hhi": p_hhi,
        "discipline": disc,
        "demo_mode": data_fetch.DEMO_MODE,
    }


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def money(x: float, cur: str = "USD") -> str:
    return f"{x:,.2f} {cur}"


def demo_banner(demo: bool) -> None:
    # Banner intentionally disabled — live data is the normal case, and a single
    # symbol falling back shouldn't shout at the user. Kept as a no-op so the
    # existing call sites don't need to change.
    return


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_dashboard(a: dict, cur: str) -> None:
    st.header("📊 Dashboard")
    demo_banner(a["demo_mode"])
    s = a["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total value", money(s["total_value"], cur))
    c2.metric("Cost basis", money(s["total_cost"], cur), help=education.tip("Cost basis"))
    c3.metric(
        "Unrealized P/L",
        money(s["total_pnl"], cur),
        f"{s['total_pnl_pct']:.1f}%",
        help=education.tip("Unrealized P/L"),
    )
    c4.metric("Holdings", s["num_holdings"])

    left, right = st.columns(2)
    df = pd.DataFrame(a["positions"])
    with left:
        st.subheader("Allocation by holding")
        fig = px.pie(df, names="ticker", values="market_value", hole=0.4)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Allocation by sector")
        sec = metrics.allocation_by(a["positions"], "sector")
        secdf = pd.DataFrame({"sector": list(sec), "weight": list(sec.values())})
        fig2 = px.pie(secdf, names="sector", values="weight", hole=0.4)
        st.plotly_chart(fig2, width="stretch")

    st.subheader("Holdings")
    show = df[
        ["ticker", "name", "shares", "avg_price", "price", "day_change_pct",
         "market_value", "pnl", "pnl_pct", "weight", "sector"]
    ].copy()
    show["weight"] = (show["weight"] * 100).round(1)
    show["pnl_pct"] = show["pnl_pct"].round(1)
    show["day_change_pct"] = show["day_change_pct"].round(2)
    st.dataframe(
        show.rename(
            columns={
                "avg_price": "avg cost",
                "price": "last",
                "day_change_pct": "day %",
                "market_value": "value",
                "pnl_pct": "P/L %",
                "weight": "weight %",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def page_holdings(portfolio: dict) -> None:
    st.header("💼 My Holdings")
    st.caption("Add, edit, or remove positions. Changes are saved locally to `data/portfolio.json`.")

    holdings = portfolio["holdings"]
    if holdings:
        edited = st.data_editor(
            pd.DataFrame(holdings),
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", required=True),
                "shares": st.column_config.NumberColumn("Shares", min_value=0.0, format="%.4f"),
                "avg_price": st.column_config.NumberColumn("Avg buy price", min_value=0.0),
                "buy_date": st.column_config.TextColumn("Buy date"),
                "asset_type": st.column_config.SelectboxColumn(
                    "Type", options=["stock", "etf", "crypto"]
                ),
            },
        )
        if st.button("💾 Save changes", type="primary"):
            cleaned = [
                {
                    "ticker": str(r["ticker"]).upper().strip(),
                    "shares": float(r["shares"] or 0),
                    "avg_price": float(r["avg_price"] or 0),
                    "buy_date": str(r.get("buy_date") or ""),
                    "asset_type": r.get("asset_type") or "stock",
                }
                for _, r in edited.iterrows()
                if str(r.get("ticker") or "").strip()
            ]
            portfolio["holdings"] = cleaned
            save_portfolio(portfolio)
            st.cache_data.clear()
            st.success("Saved! Other tabs now reflect your updated portfolio.")
            st.rerun()

    st.divider()
    st.subheader("Quick add")
    with st.form("add", clear_on_submit=True):
        cols = st.columns(5)
        ticker = cols[0].text_input("Ticker", placeholder="AAPL")
        shares = cols[1].number_input("Shares", min_value=0.0, step=1.0)
        avg = cols[2].number_input("Avg buy price", min_value=0.0, step=1.0)
        bdate = cols[3].text_input("Buy date", placeholder="2024-01-15")
        atype = cols[4].selectbox("Type", ["stock", "etf", "crypto"])
        if st.form_submit_button("➕ Add holding", type="primary") and ticker.strip():
            portfolio["holdings"].append(
                {
                    "ticker": ticker.upper().strip(),
                    "shares": float(shares),
                    "avg_price": float(avg),
                    "buy_date": bdate.strip(),
                    "asset_type": atype,
                }
            )
            save_portfolio(portfolio)
            st.cache_data.clear()
            st.success(f"Added {ticker.upper().strip()}.")
            st.rerun()


def page_risk(a: dict) -> None:
    st.header("🎲 Risk & Discipline")
    demo_banner(a["demo_mode"])
    disc = a["discipline"]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Discipline score", f"{disc.score}/100")
        st.progress(disc.score / 100)
        st.markdown(f"### {disc.label}")
    with c2:
        st.info(disc.verdict)
        b1, b2, b3 = st.columns(3)
        b1.metric("Volatility", f"{a['w_vol']:.0%}/yr", help=education.tip("Volatility"))
        b2.metric("Beta", f"{a['w_beta']:.2f}", help=education.tip("Beta"))
        b3.metric("Concentration (HHI)", f"{a['hhi']:.2f}", help=education.tip("HHI"))

    st.caption(
        "The score starts at 100 and loses points for habits that turn investing "
        "into gambling. Every deduction is shown below — nothing is hidden."
    )

    if disc.flags:
        st.subheader("🚩 Red flags")
        sev_emoji = {"high": "🔴", "medium": "🟠", "low": "🟡"}
        for f in disc.flags:
            with st.expander(f"{sev_emoji[f.severity]} {f.title}  (−{f.points} pts)"):
                st.markdown(f"**Why this matters:** {f.why}")
                st.markdown(f"**What a disciplined investor does:** {f.fix}")
    else:
        st.success("No red flags — your risk-taking looks deliberate and well-spread.")

    if disc.positives:
        st.subheader("✅ What you're doing well")
        for p in disc.positives:
            st.markdown(f"- {p}")


def page_weaknesses(a: dict) -> None:
    st.header("🔍 Weaknesses & Health Check")
    demo_banner(a["demo_mode"])
    findings = weaknesses.analyze(a["positions"], a["per_vol"], a["per_beta"])
    color = {"good": "✅", "warning": "⚠️", "bad": "❌"}
    for f in findings:
        with st.container(border=True):
            st.markdown(f"{color[f.status]} **{f.category} — {f.headline}**")
            st.caption(f.detail)


def page_news(a: dict) -> None:
    st.header("📰 Recent News")
    demo_banner(a["demo_mode"])
    tickers = [p["ticker"] for p in a["positions"]]
    news_by = news.get_portfolio_news(tickers, limit_per=4)
    tally = news.sentiment_tally(news_by)
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Positive", tally["good"])
    c2.metric("🔴 Negative", tally["bad"])
    c3.metric("⚪ Neutral", tally["neutral"])
    st.caption("Headlines are auto-tagged by sentiment. Read them yourself — a tag is a hint, not a verdict.")

    for ticker, items in news_by.items():
        if not items:
            continue
        st.subheader(ticker)
        for n in items:
            link = n.get("link")
            title = f"{n['emoji']} {n['title']}"
            if link:
                st.markdown(f"- [{title}]({link})  \n  <small>{n.get('publisher','')} · {n.get('published','')}</small>", unsafe_allow_html=True)
            else:
                st.markdown(f"- {title}")


def page_coach(a: dict, cur: str) -> None:
    st.header("🧑‍🏫 AI Coach")
    disc = a["discipline"]
    context = {
        "summary": a["summary"],
        "positions": a["positions"],
        "discipline": {
            "score": disc.score,
            "label": disc.label,
            "verdict": disc.verdict,
            "flags": [{"title": f.title, "points": f.points, "fix": f.fix} for f in disc.flags],
        },
        "volatility": a["w_vol"],
        "beta": a["w_beta"],
        "currency": cur,
    }
    if ai_coach.is_available():
        st.caption("Personalised coaching generated from your metrics by Claude.")
    else:
        st.caption(
            "💡 Set `ANTHROPIC_API_KEY` (see `.env.example`) to unlock natural-language "
            "coaching. For now, here's the transparent rules-based summary:"
        )
    if st.button("✨ Generate coaching", type="primary"):
        with st.spinner("Thinking…"):
            text, source = ai_coach.generate_coaching(context)
        st.markdown(text)
        st.caption(f"_Source: {'Claude AI' if source == 'ai' else 'rules engine'}_")


def page_learn() -> None:
    st.header("🎓 Learn")
    st.caption("Short, beginner-friendly lessons. Master these and you're already ahead of most.")
    for lesson in education.LESSONS:
        with st.expander(f"📘 {lesson['title']}"):
            st.markdown(lesson["body"])
    st.divider()
    st.subheader("Glossary")
    for term, definition in education.GLOSSARY.items():
        st.markdown(f"**{term}** — {definition}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    portfolio = load_portfolio()
    cur = portfolio.get("base_currency", "USD")

    st.sidebar.title("📈 Portfolio Advisor")
    st.sidebar.caption("Track. Diagnose. Learn.")
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Terminal", "My Holdings", "Risk & Discipline", "Weaknesses", "News", "AI Coach", "Learn"],
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "⚠️ Educational tool only — **not financial advice**. Do your own research."
    )

    if page == "Terminal":
        terminal.render(portfolio)
        return
    if page == "My Holdings":
        page_holdings(portfolio)
        return
    if page == "Learn":
        page_learn()
        return

    if not portfolio["holdings"]:
        st.info("No holdings yet — add some on the **My Holdings** tab to begin.")
        return

    a = analyse(json.dumps(portfolio, sort_keys=True))
    if page == "Dashboard":
        page_dashboard(a, cur)
    elif page == "Risk & Discipline":
        page_risk(a)
    elif page == "Weaknesses":
        page_weaknesses(a)
    elif page == "News":
        page_news(a)
    elif page == "AI Coach":
        page_coach(a, cur)


if __name__ == "__main__":
    main()
