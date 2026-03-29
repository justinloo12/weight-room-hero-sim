from datetime import datetime
from time import sleep

import pandas as pd
from pandas.errors import ParserError
from pybaseball import statcast


SEASON_2025_START = "2025-03-27"
SEASON_2025_END = "2025-10-05"
SEASON_2026_START = "2026-03-26"
OUTPUT_PATH = "homerun_data_all.csv"


def pull_statcast_range(start_dt: str, end_dt: str, label: str, retries: int = 3) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        try:
            print(f"Pulling {label} (attempt {attempt}/{retries})...")
            df = statcast(start_dt=start_dt, end_dt=end_dt)
            df["data_source"] = label
            print(f"  {label}: {len(df):,} rows")
            return df
        except ParserError as exc:
            print(f"  Parser error while pulling {label}: {exc}")
        except Exception as exc:
            print(f"  Error while pulling {label}: {exc}")

        if attempt < retries:
            wait_s = 5 * attempt
            print(f"  Retrying in {wait_s} seconds...")
            sleep(wait_s)

    raise RuntimeError(f"Failed to pull {label} after {retries} attempts")


def main() -> None:
    season_2025 = pull_statcast_range(
        start_dt=SEASON_2025_START,
        end_dt=SEASON_2025_END,
        label="regular_season_2025",
    )

    today = datetime.now().strftime("%Y-%m-%d")
    season_2026 = pull_statcast_range(
        start_dt=SEASON_2026_START,
        end_dt=today,
        label="regular_season_2026",
    )

    all_data = pd.concat([season_2025, season_2026], ignore_index=True)
    all_data.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDone! {len(all_data):,} total rows saved to {OUTPUT_PATH}")
    print(f"Columns: {list(all_data.columns)}")
    print("\nBreakdown:")
    print(all_data["data_source"].value_counts())


if __name__ == "__main__":
    main()
