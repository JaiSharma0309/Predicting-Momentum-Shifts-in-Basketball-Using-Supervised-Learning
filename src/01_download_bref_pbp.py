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
TARGET_NEW_GAMES = 20


# ================================
# DOWNLOAD ONE GAME
# ================================
def download_bref_pbp(game_code):
    outfile = RAW_DIR / f"pbp_{game_code}.csv"

    if outfile.exists():
        print(f"Already exists → {outfile.name}")
        return False

    url = f"https://www.basketball-reference.com/boxscores/pbp/{game_code}.html"

    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
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
    url = f"https://www.basketball-reference.com/boxscores/?month={month}&day={day}&year={year}"

    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    codes = []

    for a in soup.select('a[href^="/boxscores/"]'):
        href = a.get("href", "")

        if href.endswith(".html"):
            code = href.split("/")[-1].replace(".html", "")

            if len(code) == 12 and code not in codes:
                codes.append(code)

    return codes


# ================================
# MAIN
# ================================
def main():

    year = 2023
    month = 2

    existing_games = {
        f.stem.replace("pbp_", "")
        for f in RAW_DIR.glob("pbp_*.csv")
    }

    print("Already have:", len(existing_games), "games")

    downloaded = 0

    # Scan dates progressively
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
                print("\n Downloaded 20 new games — stopping.")
                return


if __name__ == "__main__":
    main()
