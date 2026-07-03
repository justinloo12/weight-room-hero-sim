"""Pure betting-math helpers.

Shared by dashboard.py and check_results.py, and unit-tested in tests/.
Everything in here is deterministic and has no I/O or third-party deps.
"""


def american_to_implied(odds):
    """Convert American odds to implied probability as a 0-1 fraction.

    +250 -> 100 / 350 ~= 0.2857
    -150 -> 150 / 250  = 0.60
    """
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def implied_to_american(implied_pct):
    """Convert implied probability (in percent) back to American odds.

    Probability is clamped to [1%, 99%] before conversion.
    """
    p = max(0.01, min(0.99, implied_pct / 100.0))
    if p >= 0.5:
        return -round(p / (1 - p) * 100)
    return round((1 - p) / p * 100)


def edge_percentage(model_prob_pct, book_implied_pct):
    """Edge in percentage points: model probability minus book implied probability.

    Both inputs are percentages (e.g. 12.5 for 12.5%). Rounded to 2 decimals.
    """
    return round(model_prob_pct - book_implied_pct, 2)


def expected_roi(model_prob_pct, american_odds):
    """Expected return per $1 staked given a model win probability (percent)
    and American odds. Returns -1.0 when inputs are unusable."""
    p = (model_prob_pct or 0.0) / 100.0
    if p <= 0 or american_odds is None:
        return -1.0
    if american_odds > 0:
        payout = american_odds / 100.0
    else:
        payout = 100.0 / abs(american_odds)
    return p * payout - (1.0 - p)


def kelly_fraction(model_prob_pct, american_odds):
    """Full-Kelly optimal fraction of bankroll for a binary bet.

    f* = (b*p - q) / b, where b is the net decimal payout from American odds,
    p is the model win probability, q = 1 - p. Never negative; returns 0.0
    for degenerate inputs (p <= 0, p >= 1, missing odds, non-positive payout).
    """
    p = (model_prob_pct or 0.0) / 100.0
    if p <= 0 or p >= 1 or american_odds is None:
        return 0.0
    b = (american_odds / 100.0) if american_odds > 0 else (100.0 / abs(american_odds))
    if b <= 0:
        return 0.0
    q = 1.0 - p
    return max(0.0, (b * p - q) / b)


def pnl_from_odds(american_odds, stake):
    """Profit (excluding returned stake) on a WINNING bet given American odds.

    +300 with $100 stake -> $300 profit; -150 with $100 stake -> $66.67 profit.
    """
    if american_odds >= 100:
        return stake * (american_odds / 100)
    return stake * (100 / abs(american_odds))
