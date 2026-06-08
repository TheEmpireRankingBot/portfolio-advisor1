"""Unit tests for the discipline score / red-flag engine."""

from advisor import metrics, risk_score


def _positions(weights_by_ticker, sector="Technology", asset_type="stock"):
    """Build minimal position dicts that carry just what the scorer reads."""
    return [
        {
            "ticker": t,
            "weight": w,
            "sector": sector,
            "asset_type": asset_type,
        }
        for t, w in weights_by_ticker.items()
    ]


def test_well_diversified_portfolio_scores_high():
    pos = _positions({f"T{i}": 0.1 for i in range(10)})
    # spread sectors so no single sector dominates
    for i, p in enumerate(pos):
        p["sector"] = ["Tech", "Health", "Finance", "Energy", "Staples"][i % 5]
    hhi = metrics.hhi([p["weight"] for p in pos])
    res = risk_score.compute_discipline_score(pos, weighted_volatility=0.18, weighted_beta=1.0, portfolio_hhi=hhi)
    assert res.score >= 80
    assert res.label == "Calculated risk-taker"


def test_all_in_one_stock_is_flagged_and_low():
    pos = _positions({"YOLO": 1.0})
    res = risk_score.compute_discipline_score(pos, weighted_volatility=0.7, weighted_beta=2.2, portfolio_hhi=1.0)
    assert res.score < 40
    assert res.label == "Leaning toward gambling"
    titles = " ".join(f.title for f in res.flags)
    assert "Over-concentrated" in titles
    assert "Very few holdings" in titles


def test_heavy_crypto_allocation_flagged():
    pos = _positions({"BTC-USD": 0.5, "ETH-USD": 0.2, "AAA": 0.3})
    pos[0]["asset_type"] = "crypto"
    pos[1]["asset_type"] = "crypto"
    res = risk_score.compute_discipline_score(pos, weighted_volatility=0.6, weighted_beta=1.6, portfolio_hhi=0.38)
    assert any("speculative" in f.title.lower() for f in res.flags)


def test_score_is_bounded():
    pos = _positions({"YOLO": 1.0})
    res = risk_score.compute_discipline_score(pos, weighted_volatility=2.0, weighted_beta=5.0, portfolio_hhi=1.0)
    assert 0 <= res.score <= 100


def test_flags_carry_explanations():
    pos = _positions({"YOLO": 1.0})
    res = risk_score.compute_discipline_score(pos, weighted_volatility=0.7, weighted_beta=2.0, portfolio_hhi=1.0)
    for f in res.flags:
        assert f.why and f.fix  # every flag teaches AND prescribes
