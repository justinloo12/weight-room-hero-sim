"""
check_results.py
----------------
Looks up whether each pick in picks_history.csv actually hit a home run.
Uses the free MLB Stats API (no key required).
Fills in the `result` (HR / No HR) and `pnl` columns, then writes the file back.

Assumes flat $100 bet on every tracked pick by default, unless a `stake`
column is present in picks_history.csv.
  HR  →  pnl = stake × (american_odds / 100)        for positive odds
          pnl = stake × (100 / abs(american_odds))   for negative odds
  No HR → pnl = -stake
"""
 
import csv
import pathlib
import requests
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
 
ET       = ZoneInfo("America/New_York")
BET_SIZE = 100         # dollars per pick
HISTORY  = pathlib.Path("picks_history.csv")
MLB_GAME_LOG = "https://statsapi.mlb.com/api/v1/people/{batter_id}/stats?stats=gameLog&season={season}&gameType=R&language=en"
RESULT_CHECK_CUTOFF = "2026-06-01"
 
# ── helpers ──────────────────────────────────────────────────────────────────
 
def pnl_from_odds(american_odds: float, stake: float) -> float:
    """Return profit on a winning bet given American odds and stake."""
    if american_odds >= 100:
        return stake * (american_odds / 100)
    else:
        return stake * (100 / abs(american_odds))
 
 
def fetch_hr_dates(batter_id: str, season: int) -> set:
    """Return a set of 'YYYY-MM-DD' strings on which this batter hit a HR."""
    if not batter_id:
        return set()
    url = MLB_GAME_LOG.format(batter_id=batter_id, season=season)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  API error for batter {batter_id}: {e}")
        return set()

    hr_dates = set()
    for group in data.get("stats", []):
        for split in group.get("splits", []):
            date_str = split.get("date", "")
            stat     = split.get("stat", {})
            if stat.get("homeRuns", 0) > 0:
                hr_dates.add(date_str)
    return hr_dates


def row_date(row: dict) -> str:
    """Return the pick date, supporting both legacy and renamed CSV headers."""
    return (row.get("date") or row.get("game_date") or "").strip()
 
 
# ── main ─────────────────────────────────────────────────────────────────────
 
def main():
    if not HISTORY.exists():
        print("picks_history.csv not found — nothing to check.")
        return
 
    with open(HISTORY, newline="") as f:
        rows = list(csv.DictReader(f))
 
    if not rows:
        print("picks_history.csv is empty.")
        return

    fieldnames = list(rows[0].keys()) if rows else []
    if "date" not in fieldnames and "game_date" not in fieldnames:
        print(f"Missing expected date column. Found columns: {fieldnames}")
        return
 
    today_et = datetime.now(ET).strftime("%Y-%m-%d")
 
    # Only check rows that have no result yet, are before today, and are recent
    # enough to avoid re-resolving older backfilled history.
    needs_check = [
        r for r in rows
        if (
            not r.get("result")
            and row_date(r)
            and RESULT_CHECK_CUTOFF <= row_date(r) < today_et
        )
    ]
 
    if not needs_check:
        print("No pending picks to resolve.")
    else:
        print(f"Resolving {len(needs_check)} picks...")
 
    # Cache game-log lookups per (batter_id, season) to minimise API calls
    hr_date_cache: dict[tuple, set] = {}
    updated = 0
 
    for row in needs_check:
        date_str  = row_date(row)
        if not date_str:
            print(f"  Skipping row with no date: {row}")
            continue
        batter_id = row.get("batter_id", "")
        season    = int(date_str[:4]) if date_str else datetime.now().year

        cache_key = (batter_id, season)
        if cache_key not in hr_date_cache:
            print(f"  Fetching game log — {row['player']} (id={batter_id})")
            hr_date_cache[cache_key] = fetch_hr_dates(batter_id, season)
            time.sleep(0.3)   # be polite to the MLB API

        hit_hr = date_str in hr_date_cache[cache_key]
        row["result"] = "HR" if hit_hr else "No HR"
 
        try:
            odds = float(row["book_odds"]) if row.get("book_odds") else None
        except ValueError:
            odds = None
        try:
            stake = float(row.get("stake", BET_SIZE) or BET_SIZE)
        except ValueError:
            stake = BET_SIZE

        if hit_hr and odds is not None:
            row["pnl"] = f'{pnl_from_odds(odds, stake):.2f}'
        elif hit_hr:
            row["pnl"] = ""         # hit but no odds recorded
        else:
            row["pnl"] = f'{-stake:.2f}'
 
        updated += 1
 
    # Write back (preserve all original rows)
    if updated > 0 or True:
        with open(HISTORY, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Updated {updated} rows in picks_history.csv")
 
    # ── Summary stats ──────────────────────────────────────────────────────
    resolved = [r for r in rows if r.get("result")]
    if not resolved:
        print("No resolved picks yet — check back after games are played.")
        return
 
    hits     = sum(1 for r in resolved if r["result"] == "HR")
    total    = len(resolved)
    hit_rate = hits / total * 100
 
    pnls = []
    for r in resolved:
        try:
            pnls.append(float(r["pnl"]))
        except (ValueError, TypeError):
            pass
 
    stakes = []
    for r in resolved:
        try:
            stakes.append(float(r.get("stake", BET_SIZE) or BET_SIZE))
        except (ValueError, TypeError):
            stakes.append(BET_SIZE)

    total_bet    = sum(stakes)
    total_profit = sum(pnls)
    roi          = (total_profit / total_bet * 100) if total_bet else 0
 
    print(f"\n{'='*40}")
    print(f"  Picks tracked : {total}")
    print(f"  HR hit rate   : {hits}/{total} ({hit_rate:.1f}%)")
    print(f"  Total wagered : ${total_bet:.0f}")
    print(f"  Net profit    : ${total_profit:+.2f}")
    print(f"  ROI           : {roi:+.1f}%")
    print(f"{'='*40}\n")
 
 
if __name__ == "__main__":
    main()
