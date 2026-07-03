import csv
import os
import pathlib
import requests
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
HISTORY = pathlib.Path("picks_history.csv")
WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
MLB_GAME_LOG = "https://statsapi.mlb.com/api/v1/people/{batter_id}/stats?stats=gameLog&season={season}&gameType=R&language=en"


def fetch_hr_dates(batter_id: str, season: int) -> set[str]:
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
            if split.get("stat", {}).get("homeRuns", 0) > 0:
                hr_dates.add(split.get("date", ""))
    return hr_dates


def send_discord_alert(row: dict):
    if not WEBHOOK_URL:
        print("DISCORD_WEBHOOK_URL is not set.")
        return False

    odds = row.get("book_odds", "")
    edge = row.get("edge", "")
    payload = {
        "content": (
            f"🚨 **{row.get('player', 'Unknown')}** just homered.\n"
            f"- Pick list: `{row.get('pick_type', 'tracked')}` #{row.get('rank', '?')}\n"
            f"- Matchup: `{row.get('game', '')}` vs `{row.get('pitcher', '')}`\n"
            f"- Odds: `{odds}` | Edge: `{edge}%`\n"
            f"- Date: `{row.get('date', '')}`"
        )
    }
    try:
        resp = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"  Discord webhook error: {e}")
        return False


def main():
    if not HISTORY.exists():
        print("picks_history.csv not found.")
        return

    with open(HISTORY, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("picks_history.csv is empty.")
        return

    fieldnames = list(rows[0].keys())
    if "alert_sent" not in fieldnames:
        fieldnames.append("alert_sent")
        for row in rows:
            row["alert_sent"] = row.get("alert_sent", "")

    today = datetime.now(ET).strftime("%Y-%m-%d")
    pending = [
        r for r in rows
        if r.get("date") == today
        and r.get("batter_id")
        and not r.get("alert_sent")
    ]

    if not pending:
        print("No pending same-day picks to alert on.")
        return

    print(f"Checking {len(pending)} tracked picks for live HR alerts...")
    hr_date_cache = {}
    sent_count = 0

    for row in pending:
        batter_id = row.get("batter_id", "")
        season = int(today[:4])
        cache_key = (batter_id, season)
        if cache_key not in hr_date_cache:
            hr_date_cache[cache_key] = fetch_hr_dates(batter_id, season)
            time.sleep(0.25)

        if today in hr_date_cache[cache_key]:
            if send_discord_alert(row):
                row["alert_sent"] = datetime.now(ET).strftime("%Y-%m-%d %I:%M %p ET")
                sent_count += 1

    with open(HISTORY, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Sent {sent_count} Discord alerts.")


if __name__ == "__main__":
    main()
