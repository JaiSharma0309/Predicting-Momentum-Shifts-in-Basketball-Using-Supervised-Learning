import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

def download_bref_pbp(game_code, outfile):
    url = f"https://www.basketball-reference.com/boxscores/pbp/{game_code}.html"

    headers = {
        "User-Agent": "bMozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    }

    r = requests.get(url, headers=headers)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"id": "pbp"})

    if table is None:
        raise ValueError("Could not find play-by-play table.")

    df = pd.read_html(StringIO(str(table)))[0]

    # ====== REPRODUCIBLE PART ======
    os.makedirs("data/raw", exist_ok=True)
    out = f"data/raw/{outfile}"
    df.to_csv(out, index=False)

    print("Saved to:", out)
    print("Shape:", df.shape)


def get_game_codes_for_date(year: int, month: int, day: int, limit: int | None = None):
    url = f"https://www.basketball-reference.com/boxscores/?month={month}&day={day}&year={year}"
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, headers=headers)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    codes = []
    # Boxscore links look like: /boxscores/202210190UTA.html
    for a in soup.select('a[href^="/boxscores/"]'):
        href = a.get("href", "")
        if href.endswith(".html") and len(href.split("/")[-1]) == len("202210190UTA.html"):
            code = href.split("/")[-1].replace(".html", "")
            if code not in codes:
                codes.append(code)

        if limit is not None and len(codes) >= limit:
            break

    return codes


def main():
    # Pick a date with multiple games
    year, month, day = 2022, 10, 21

    codes = get_game_codes_for_date(year, month, day, limit=5)
    print("Found game codes:", codes)

    for game_code in codes:
        outfile = f"pbp_{game_code}.csv"
        try:
            print(f"\nDownloading {game_code} → {outfile}")
            download_bref_pbp(game_code, outfile)
        except Exception as e:
            print(f"Failed for {game_code}: {e}")


if __name__ == "__main__":
    main()
