"""A personal Bloomberg-style terminal page.

One command line drives everything, in the spirit of a real terminal:

    AAPL            quote (neon green, big)
    AAPL H [days]   historical line chart (default 60 days)
    AAPL N          latest news headlines, sentiment-tagged
    AAPL F          key financials (P/E, market cap, 52-week range, ...)
    BTC / ETH ...   crypto quotes via CoinGecko (H/N/F work too)
    MACRO CPI       macro-economics charts via FRED (GDP, UNRATE, 10Y, ...)
    NEWS            general market headlines
    PORT            your portfolio as a terminal blotter
    HELP            command reference

``parse`` is a pure function (unit tested); rendering is Streamlit + raw HTML
with a dark retro aesthetic: black panels, neon green (#00ff41) data, amber
(#ffb000) macro accents, monospace everywhere.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import data_fetch, metrics, news

GREEN = "#00ff41"
AMBER = "#ffb000"
RED = "#ff4444"
BG = "#050905"

# --------------------------------------------------------------------------- #
# Command grammar
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Command:
    kind: str  # quote|chart|news|fundamentals|macro|market_news|port|help|none|error
    symbol: str = ""
    arg: int = 0
    error: str = ""


# Friendly alias -> (FRED series id, human title, transform)
FRED_SERIES: dict[str, tuple[str, str, str | None]] = {
    "CPI": ("CPIAUCSL", "US Consumer Price Index (1982-84=100)", None),
    "INFLATION": ("CPIAUCSL", "US inflation rate, % year-over-year", "yoy12"),
    "GDP": ("GDP", "US Gross Domestic Product ($B, quarterly)", None),
    "UNEMPLOYMENT": ("UNRATE", "US unemployment rate (%)", None),
    "UNRATE": ("UNRATE", "US unemployment rate (%)", None),
    "FEDFUNDS": ("FEDFUNDS", "Effective Federal Funds Rate (%)", None),
    "RATES": ("FEDFUNDS", "Effective Federal Funds Rate (%)", None),
    "10Y": ("DGS10", "US 10-year Treasury yield (%)", None),
    "2Y": ("DGS2", "US 2-year Treasury yield (%)", None),
    "YIELDCURVE": ("T10Y2Y", "10y minus 2y Treasury spread (%) — negative = inverted", None),
    "MORTGAGE": ("MORTGAGE30US", "30-year fixed mortgage rate (%)", None),
    "M2": ("M2SL", "M2 money supply ($B)", None),
    "VIX": ("VIXCLS", "CBOE Volatility Index (fear gauge)", None),
}

_TICKER_RE = re.compile(r"[A-Z0-9.\-]{1,12}")


def parse(raw: str) -> Command:
    """Parse a terminal command. Pure function — see tests/test_terminal.py."""
    tokens = raw.strip().upper().split()
    if not tokens:
        return Command("none")
    head = tokens[0]

    if head in ("HELP", "?"):
        return Command("help")
    if head in ("PORT", "PORTFOLIO"):
        return Command("port")
    if head == "NEWS" and len(tokens) == 1:
        return Command("market_news")
    if head in ("MACRO", "FED", "ECON"):
        if len(tokens) < 2:
            return Command("error", error="Usage: MACRO <SERIES>  e.g. MACRO CPI — type HELP for the list")
        years = 10
        if len(tokens) >= 3 and tokens[2].isdigit():
            years = max(1, min(int(tokens[2]), 50))
        return Command("macro", symbol=tokens[1], arg=years)

    sym = head
    if not _TICKER_RE.fullmatch(sym):
        return Command("error", error=f"'{sym}' doesn't look like a ticker — type HELP")
    if len(tokens) == 1:
        return Command("quote", symbol=sym)

    fn = tokens[1]
    if fn in ("H", "HIST", "CHART"):
        days = 60
        if len(tokens) >= 3 and tokens[2].isdigit():
            days = max(10, min(int(tokens[2]), 252))
        return Command("chart", symbol=sym, arg=days)
    if fn in ("N", "NEWS"):
        return Command("news", symbol=sym)
    if fn in ("F", "FUND", "FA"):
        return Command("fundamentals", symbol=sym)
    return Command("error", error=f"Unknown function '{fn}' — use H (chart), N (news), F (financials), or HELP")


def _norm_symbol(sym: str) -> str:
    """BTC -> BTC-USD so the data layer treats bare crypto symbols as crypto."""
    if sym in data_fetch.CRYPTO_IDS and not sym.endswith("-USD"):
        return sym + "-USD"
    return sym


# --------------------------------------------------------------------------- #
# Cached data wrappers (so repeat commands don't re-hit the APIs)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=120, show_spinner=False)
def _quote(sym: str) -> dict:
    return data_fetch.get_quote(sym)


@st.cache_data(ttl=600, show_spinner=False)
def _history(sym: str) -> pd.Series:
    return data_fetch.get_history(sym)


@st.cache_data(ttl=600, show_spinner=False)
def _fundamentals(sym: str) -> dict:
    return data_fetch.get_fundamentals(sym)


@st.cache_data(ttl=600, show_spinner=False)
def _news_for(sym: str) -> list[dict]:
    return data_fetch.get_news(sym, limit=8)


@st.cache_data(ttl=600, show_spinner=False)
def _market_news() -> list[dict]:
    return data_fetch.get_market_news(limit=8)


@st.cache_data(ttl=3600, show_spinner=False)
def _macro(fred_id: str) -> pd.Series | None:
    return data_fetch.get_macro_series(fred_id)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _fmt_big(x) -> str:
    if x in (None, "") or pd.isna(x):
        return "—"
    x = float(x)
    for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(x) >= div:
            return f"{x / div:,.2f}{unit}"
    return f"{x:,.0f}"


def _fmt(x, nd: int = 2) -> str:
    if x in (None, "") or pd.isna(x):
        return "—"
    return f"{float(x):,.{nd}f}"


def _esc(s) -> str:
    return html.escape(str(s or ""))


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
_CSS = f"""
<style>
.term-box {{
  background: {BG}; border: 1px solid {GREEN}44; border-radius: 6px;
  padding: 14px 18px; margin-bottom: 10px;
  font-family: "Courier New", ui-monospace, monospace; color: {GREEN};
}}
.term-big {{ font-size: 2.0rem; font-weight: 700; text-shadow: 0 0 10px {GREEN}66; }}
.term-mid {{ font-size: 1.15rem; }}
.term-dim {{ color: #58b06f; font-size: 0.85rem; }}
.term-amber {{ color: {AMBER}; }}
.term-red {{ color: {RED}; }}
.term-table td {{ padding: 3px 18px 3px 0; font-family: inherit; color: {GREEN}; }}
.term-table td.lbl {{ color: #58b06f; }}
.term-news {{ margin: 4px 0; line-height: 1.45; }}
.term-news a {{ color: {GREEN}; text-decoration: none; }}
.term-news a:hover {{ text-decoration: underline; }}
[data-testid="stTextInput"] input {{
  background: #000 !important; color: {GREEN} !important;
  font-family: "Courier New", monospace !important;
  border: 1px solid {GREEN} !important; caret-color: {GREEN};
}}
[data-testid="stForm"] {{ border: 1px solid {GREEN}44; background: {BG}; }}
</style>
"""


def _box(inner_html: str) -> None:
    st.markdown(f'<div class="term-box">{inner_html}</div>', unsafe_allow_html=True)


def _fig(series: pd.Series, title: str, color: str = GREEN) -> go.Figure:
    fig = go.Figure(
        go.Scatter(x=series.index, y=series.values, mode="lines",
                   line=dict(color=color, width=2))
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=BG, height=420,
        title=dict(text=title, font=dict(family="Courier New", color=color, size=16)),
        font=dict(family="Courier New", color="#7CFC9A"),
        margin=dict(l=40, r=20, t=50, b=30),
        xaxis=dict(gridcolor="#10331a"), yaxis=dict(gridcolor="#10331a"),
    )
    return fig


def _render_quote(sym: str) -> None:
    q = _quote(_norm_symbol(sym))
    chg = q.get("day_change_pct", 0.0)
    arrow, cls = ("▲", "") if chg >= 0 else ("▼", "term-red")
    _box(
        f'<div class="term-big">{_esc(sym)} <span class="term-dim term-mid">'
        f'{_esc(q.get("name", sym)).upper()}</span></div>'
        f'<div class="term-big">{_fmt(q.get("price"))} '
        f'<span class="term-mid {cls}">{arrow} {chg:+.2f}%</span></div>'
        f'<div class="term-dim">{_esc(q.get("sector", ""))}'
        f'{" · MCAP " + _fmt_big(q.get("market_cap")) if q.get("market_cap") else ""}</div>'
    )


def _render_chart(sym: str, days: int) -> None:
    s = _history(_norm_symbol(sym)).iloc[-days:]
    if s.empty:
        _box(f'<span class="term-red">No history available for {_esc(sym)}.</span>')
        return
    first, last = float(s.iloc[0]), float(s.iloc[-1])
    pct = (last - first) / first * 100 if first else 0.0
    color = GREEN if pct >= 0 else AMBER
    st.plotly_chart(
        _fig(s, f"{sym}  ·  {days}D  ·  {pct:+.1f}%", color=color),
        width="stretch",
    )
    _box(
        f'<span class="term-dim">{days}-day range '
        f'{_fmt(s.min())} – {_fmt(s.max())} · last {_fmt(last)}</span>'
    )


def _render_news(sym: str) -> None:
    items = _news_for(_norm_symbol(sym))
    if not items:
        _box(f'<span class="term-dim">No recent headlines for {_esc(sym)}.</span>')
        return
    lines = [f'<div class="term-mid term-amber">{_esc(sym)} — LATEST HEADLINES</div>']
    for n in items:
        label, _ = news.classify_sentiment(n.get("title", ""))
        dot = {"good": GREEN, "bad": RED, "neutral": "#9aa"}[label]
        title = _esc(n.get("title"))
        link = n.get("link") or ""
        body = f'<a href="{_esc(link)}" target="_blank">{title}</a>' if link else title
        meta = " · ".join(x for x in (_esc(n.get("publisher")), _esc(n.get("published"))) if x)
        lines.append(
            f'<div class="term-news"><span style="color:{dot}">●</span> {body}'
            f'<br><span class="term-dim">&nbsp;&nbsp;{meta}</span></div>'
        )
    lines.append('<div class="term-dim">● green = positive tone, red = negative — a hint, not a verdict.</div>')
    _box("".join(lines))


def _render_fundamentals(sym: str) -> None:
    q = _quote(_norm_symbol(sym))
    f = _fundamentals(_norm_symbol(sym))
    rows: list[tuple[str, str]]
    if f.get("kind") == "crypto":
        rows = [
            ("MARKET CAP", _fmt_big(f.get("market_cap"))),
            ("RANK", f"#{f['rank']}" if f.get("rank") else "—"),
            ("24H VOLUME", _fmt_big(f.get("volume_24h"))),
            ("24H RANGE", f"{_fmt(f.get('low_24h'))} – {_fmt(f.get('high_24h'))}"),
            ("ALL-TIME HIGH", _fmt(f.get("ath"))),
            ("FROM ATH", f"{_fmt(f.get('from_ath_pct'))}%"),
        ]
        teach = ("Crypto has no earnings, so there's no P/E — value rests on adoption "
                 "and scarcity. 'From ATH' shows how far below the peak it trades.")
    else:
        rows = [
            ("P/E (TTM)", _fmt(f.get("pe"), 1)),
            ("EPS (TTM)", _fmt(f.get("eps"))),
            ("MARKET CAP", _fmt_big(f.get("market_cap"))),
            ("52W HIGH", _fmt(f.get("high_52w"))),
            ("52W LOW", _fmt(f.get("low_52w"))),
            ("DIV YIELD", f"{_fmt(f.get('div_yield'))}%"),
            ("BETA", _fmt(f.get("beta"))),
        ]
        teach = ("P/E = price per $1 of yearly earnings (high can mean expensive OR "
                 "fast-growing). Beta = how hard it moves vs the market (1 = with it). "
                 "52W range shows where today's price sits in the past year.")
        if not data_fetch.FINNHUB_KEY:
            teach += " Some metrics need FINNHUB_API_KEY in Secrets."
    table = "".join(
        f'<tr><td class="lbl">{lbl}</td><td>{val}</td></tr>' for lbl, val in rows
    )
    _box(
        f'<div class="term-mid term-amber">{_esc(sym)} — {_esc(q.get("name", sym)).upper()} · KEY FINANCIALS</div>'
        f'<table class="term-table">{table}</table>'
        f'<div class="term-dim">{teach}</div>'
    )


def _render_macro(alias: str, years: int) -> None:
    fred_id, title, transform = FRED_SERIES.get(alias, (alias, alias, None))
    s = _macro(fred_id)
    if s is None or s.empty:
        _box(
            f'<span class="term-red">Couldn\'t fetch FRED series "{_esc(fred_id)}".</span> '
            f'<span class="term-dim">Try an alias from HELP (CPI, GDP, UNRATE, 10Y, VIX, '
            f'YIELDCURVE…) or any raw FRED series id.</span>'
        )
        return
    if transform == "yoy12":
        s = (s.pct_change(12) * 100).dropna()
    cutoff = s.index.max() - pd.DateOffset(years=years)
    s = s[s.index >= cutoff]
    last_date = s.index[-1].strftime("%b %Y")
    st.plotly_chart(_fig(s, f"{title}  ·  {years}Y", color=AMBER), width="stretch")
    _box(
        f'<span class="term-amber term-mid">LATEST: {_fmt(s.iloc[-1])}</span> '
        f'<span class="term-dim">({last_date}) · source: FRED, St. Louis Fed · series {fred_id}</span>'
    )


def _render_market_news() -> None:
    items = _market_news()
    lines = ['<div class="term-mid term-amber">MARKET WIRE — TOP HEADLINES</div>']
    for n in items:
        label, _ = news.classify_sentiment(n.get("title", ""))
        dot = {"good": GREEN, "bad": RED, "neutral": "#9aa"}[label]
        title = _esc(n.get("title"))
        link = n.get("link") or ""
        body = f'<a href="{_esc(link)}" target="_blank">{title}</a>' if link else title
        meta = " · ".join(x for x in (_esc(n.get("publisher")), _esc(n.get("published"))) if x)
        lines.append(
            f'<div class="term-news"><span style="color:{dot}">●</span> {body}'
            f'<br><span class="term-dim">&nbsp;&nbsp;{meta}</span></div>'
        )
    _box("".join(lines))


def _render_port(portfolio: dict) -> None:
    holdings = portfolio.get("holdings", [])
    if not holdings:
        _box('<span class="term-dim">No holdings yet — add some on the My Holdings tab, '
             'then run PORT again.</span>')
        return
    quotes = {h["ticker"].upper(): _quote(h["ticker"].upper()) for h in holdings}
    positions = metrics.build_positions(holdings, quotes)
    total = sum(p["market_value"] for p in positions)
    rows = "".join(
        f'<tr><td>{_esc(p["ticker"])}</td><td>{_fmt(p["shares"], 4).rstrip("0").rstrip(".")}</td>'
        f'<td>{_fmt(p["price"])}</td>'
        f'<td class="{"" if p["day_change_pct"] >= 0 else "term-red"}">{p["day_change_pct"]:+.2f}%</td>'
        f'<td>{_fmt(p["market_value"])}</td><td>{p["weight"]:.1%}</td>'
        f'<td class="{"" if p["pnl"] >= 0 else "term-red"}">{p["pnl"]:+,.0f}</td></tr>'
        for p in positions
    )
    _box(
        '<div class="term-mid term-amber">PORTFOLIO BLOTTER</div>'
        '<table class="term-table"><tr>'
        '<td class="lbl">SYM</td><td class="lbl">QTY</td><td class="lbl">LAST</td>'
        '<td class="lbl">DAY%</td><td class="lbl">VALUE</td><td class="lbl">WT%</td>'
        '<td class="lbl">P/L</td></tr>'
        f"{rows}</table>"
        f'<div class="term-mid">TOTAL {_fmt(total)} {_esc(portfolio.get("base_currency", "USD"))}</div>'
        '<div class="term-dim">Full analysis lives on the Risk &amp; Discipline and Weaknesses tabs.</div>'
    )


def _render_help() -> None:
    fh = "✓ set" if data_fetch.FINNHUB_KEY else "✗ not set"
    av = "✓ set" if data_fetch.ALPHAVANTAGE_KEY else "✗ not set (optional)"
    aliases = ", ".join(sorted(FRED_SERIES))
    _box(
        '<div class="term-mid term-amber">COMMAND REFERENCE</div>'
        '<table class="term-table">'
        '<tr><td class="lbl">AAPL</td><td>live quote</td></tr>'
        '<tr><td class="lbl">AAPL H <span style="opacity:.6">[days]</span></td><td>price chart (default 60 days, e.g. AAPL H 120)</td></tr>'
        '<tr><td class="lbl">AAPL N</td><td>latest headlines, sentiment-tagged</td></tr>'
        '<tr><td class="lbl">AAPL F</td><td>key financials: P/E, market cap, 52-week range…</td></tr>'
        '<tr><td class="lbl">BTC / ETH / SOL</td><td>crypto — all functions work (CoinGecko)</td></tr>'
        '<tr><td class="lbl">MACRO CPI <span style="opacity:.6">[years]</span></td><td>macro chart (FRED)</td></tr>'
        '<tr><td class="lbl">NEWS</td><td>general market headlines</td></tr>'
        '<tr><td class="lbl">PORT</td><td>your portfolio blotter</td></tr>'
        "</table>"
        f'<div class="term-dim">MACRO aliases: {aliases} — or any raw FRED series id.<br>'
        f"Data: Finnhub [{fh}] · Alpha Vantage [{av}] · CoinGecko &amp; FRED need no key.</div>"
    )


# --------------------------------------------------------------------------- #
# Page entry point
# --------------------------------------------------------------------------- #
def render(portfolio: dict) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        f'<div class="term-box"><span class="term-big">⬛ TERMINAL</span> '
        f'<span class="term-dim">personal market terminal — type HELP for commands</span></div>',
        unsafe_allow_html=True,
    )

    with st.form("terminal_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        cmd_text = col1.text_input(
            "command", placeholder="AAPL H · BTC · MACRO CPI · NEWS · PORT · HELP",
            label_visibility="collapsed",
        )
        submitted = col2.form_submit_button("RUN ▸", type="primary")

    if submitted and cmd_text.strip():
        st.session_state["term_last_cmd"] = cmd_text.strip()
        log = st.session_state.setdefault("term_log", [])
        if not log or log[-1] != cmd_text.strip().upper():
            log.append(cmd_text.strip().upper())
        del st.session_state["term_log"][:-30]  # keep the log bounded

    last = st.session_state.get("term_last_cmd", "")
    if not last:
        _render_help()
        return

    cmd = parse(last)
    if cmd.kind == "error":
        _box(f'<span class="term-red">ERR: {_esc(cmd.error)}</span>')
    elif cmd.kind == "help":
        _render_help()
    elif cmd.kind == "quote":
        _render_quote(cmd.symbol)
    elif cmd.kind == "chart":
        _render_chart(cmd.symbol, cmd.arg)
    elif cmd.kind == "news":
        _render_news(cmd.symbol)
    elif cmd.kind == "fundamentals":
        _render_fundamentals(cmd.symbol)
    elif cmd.kind == "macro":
        _render_macro(cmd.symbol, cmd.arg)
    elif cmd.kind == "market_news":
        _render_market_news()
    elif cmd.kind == "port":
        _render_port(portfolio)

    log = st.session_state.get("term_log", [])
    if log:
        st.markdown(
            '<div class="term-dim">'
            + " &nbsp;·&nbsp; ".join(f"▸ {_esc(c)}" for c in reversed(log[-8:]))
            + "</div>",
            unsafe_allow_html=True,
        )
