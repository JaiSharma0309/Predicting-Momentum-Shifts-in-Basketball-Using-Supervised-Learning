import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

def download_bref_pbp(game_code, outfile):
    url = f"https://www.basketball-reference.com/boxscores/pbp/{game_code}.html"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    }

    r = requests.get(url, headers=headers)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"id": "pbp"})

    if table is None:
        raise ValueError("Could not find play-by-play table.")

    df = pd.read_html(str(table))[0]

    # ====== REPRODUCIBLE PART ======
    os.makedirs("data/raw", exist_ok=True)
    out = f"data/raw/{outfile}"
    df.to_csv(out, index=False)

    print("Saved to:", out)
    print("Shape:", df.shape)

def main():
    game_code = "202210190UTA"
    outfile = "pbp_0022200012.csv"
    download_bref_pbp(game_code, outfile)

if __name__ == "__main__":
    main()
