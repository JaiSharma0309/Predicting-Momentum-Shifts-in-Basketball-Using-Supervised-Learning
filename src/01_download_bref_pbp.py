"""Download raw Basketball Reference play-by-play tables.

This script pulls play-by-play HTML tables for targeted dates and stores each
game as a raw CSV file under ``data/raw``.

Author: Jai Sharma
"""

import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

REQUEST_DELAY = 5  # ⭐ safe → 12 requests/min
TARGET_NEW_GAMES = 50


# ================================
# DOWNLOAD ONE GAME
# ================================
def download_bref_pbp(game_code):
    """Download and persist one Basketball Reference play-by-play table.

    @param game_code: Basketball Reference game identifier such as
        ``202503190LAL``.
    @return: ``True`` when a new file is downloaded and saved, otherwise
        ``False`` if the file already exists or no table is found.
    """
    outfile = RAW_DIR / f"pbp_{game_code}.csv"

    if outfile.exists():
        print(f"Already exists → {outfile.name}")
        return False

    url = f"https://www.basketball-reference.com/boxscores/pbp/{game_code}.html"

    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    # The play-by-play rows live in the table with id="pbp".
    table = soup.find("table", {"id": "pbp"})

    if table is None:
        print(f"❌ No table for {game_code}")
        return False

    df = pd.read_html(StringIO(str(table)))[0]
    df.to_csv(outfile, index=False)

    print(f"✅ Saved → {outfile.name}")
    time.sleep(REQUEST_DELAY)
    return True


# ================================
# GET GAME CODES FOR A DATE
# ================================
def get_game_codes_for_date(year, month, day):
    """Collect unique Basketball Reference game codes for a calendar date.

    @param year: Four-digit year to scan.
    @param month: Numeric month value.
    @param day: Numeric day-of-month value.
    @return: A list of game code strings discovered on the date landing page.
    """
    url = f"https://www.basketball-reference.com/boxscores/?month={month}&day={day}&year={year}"

    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    codes = []

    for a in soup.select('a[href^="/boxscores/"]'):
        href = a.get("href", "")

        if href.endswith(".html"):
            code = href.split("/")[-1].replace(".html", "")

            # Valid game codes are 12-character boxscore identifiers.
            if len(code) == 12 and code not in codes:
                codes.append(code)

    return codes


# ================================
# MAIN
# ================================
def main():
    """Scan a date window and download a capped number of unseen games.

    @return: ``None``.
    """

    year = 2025
    month = 3

    existing_games = {
        f.stem.replace("pbp_", "")
        for f in RAW_DIR.glob("pbp_*.csv")
    }

    print("Already have:", len(existing_games), "games")

    downloaded = 0

    # Walk dates in order so the downloader can resume naturally between runs.
    for day in range(18, 32):

        print(f"\nScanning {year}-{month:02d}-{day:02d}")

        try:
            codes = get_game_codes_for_date(year, month, day)
        except:
            continue

        for code in codes:

            if code in existing_games:
                continue

            success = download_bref_pbp(code)

            if success:
                downloaded += 1

            if downloaded >= TARGET_NEW_GAMES:
                print("\n Downloaded 50 new games — stopping.")
                return


if __name__ == "__main__":
    main()
