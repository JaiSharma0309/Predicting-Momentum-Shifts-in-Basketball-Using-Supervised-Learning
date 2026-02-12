import re
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)

END_Q_RE = re.compile(r"End of \d+(st|nd|rd|th) quarter", re.I)
START_Q_RE = re.compile(r"Start of \d+(st|nd|rd|th) quarter", re.I)
Q_CELL_RE = re.compile(r"^\s*(1st|2nd|3rd|4th)\s+Q\s*$", re.I)

def is_quarter_marker_row(row: pd.Series) -> str | None:
    """
    Return '2nd Q' / '3rd Q' / '4th Q' if the row is a quarter marker row
    (i.e., all non-null cells are exactly that quarter), else None.
    """
    vals = [str(x).strip() for x in row.tolist() if str(x) != "nan"]
    if not vals:
        return None

    if all(Q_CELL_RE.match(v) for v in vals) and len(set(v.lower() for v in vals)) == 1:
        return vals[0]
    return None

def is_repeated_table_header_row(df: pd.DataFrame, i: int) -> bool:
    """
    These rows look like: Time | Away | ... | Score | ... | Home
    The first cell is literally 'Time'.
    """
    first_col = df.columns[0]
    v0 = str(df.at[i, first_col]).strip().lower()
    return v0 == "time"

def clean_one_game(f: Path):
    print(f"\n=== Cleaning {f.name} ===")

    # NEW: extract game_id from filename like pbp_202210190UTA.csv
    game_id = f.stem.replace("pbp_", "")

    df = pd.read_csv(f)
    print("Raw shape:", df.shape)

    current_q = "1st Q"
    quarter_out = []
    drop = []

    for i, row in df.iterrows():
        joined = " ".join(row.astype(str).tolist())

        if is_repeated_table_header_row(df, i):
            quarter_out.append(current_q)
            drop.append(True)
            continue

        if END_Q_RE.search(joined):
            quarter_out.append(current_q)
            drop.append(True)
            continue

        if START_Q_RE.search(joined):
            quarter_out.append(current_q)
            drop.append(True)
            continue

        qmark = is_quarter_marker_row(row)
        if qmark is not None:
            current_q = qmark
            quarter_out.append(current_q)
            drop.append(True)
            continue

        quarter_out.append(current_q)
        drop.append(False)

    # Add quarter column
    df.insert(0, "quarter", quarter_out)

    # Keep only real play rows
    df_clean = df.loc[~pd.Series(drop, index=df.index)].reset_index(drop=True)

    # ====== STANDARDIZED COLUMN CLEANING ======

    # Drop columns that are basically empty
    empty_cols = [
        c for c in df_clean.columns
        if c != "quarter" and df_clean[c].isna().mean() > 0.7
    ]
    if empty_cols:
        print("Dropping mostly-empty columns:", empty_cols)
    df_clean = df_clean.drop(columns=empty_cols)

    # Expect 5 columns left: quarter, time, away, score, home
    if len(df_clean.columns) == 5:
        df_clean.columns = ["quarter", "time", "away_team", "score", "home_team"]
    else:
        print("Unexpected column count:", df_clean.columns.tolist())

    # ====== CLEAN SCORE COLUMN ======
    score_pattern = re.compile(r"^\d+\s*[–-]\s*\d+$")

    df_clean["score"] = df_clean["score"].astype(str)
    df_clean["score"] = df_clean["score"].apply(
        lambda x: x if score_pattern.match(x.strip()) else ""
    )

    score_split = (
        df_clean["score"]
        .str.replace("–", "-", regex=False)
        .str.split("-", expand=True)
    )

    df_clean["away_score"] = pd.to_numeric(score_split[0], errors="coerce")
    df_clean["home_score"] = pd.to_numeric(score_split[1], errors="coerce")

    df_clean = df_clean.drop(columns=["score"])

    # Reorder columns
    df_clean = df_clean[
        ["quarter", "time", "away_team", "away_score", "home_score", "home_team"]
    ]

    df_clean["away_team"] = df_clean["away_team"].replace("", pd.NA)
    df_clean["home_team"] = df_clean["home_team"].replace("", pd.NA)

    # ====== ADD TIME IN SECONDS ======
    t = df_clean["time"].astype(str).str.strip()

    mins = pd.to_numeric(t.str.split(":", expand=True)[0], errors="coerce")
    secs = pd.to_numeric(t.str.split(":", expand=True)[1], errors="coerce")

    df_clean["time_sec"] = mins * 60 + secs

    # ✅ NEW: add game identifier (critical for modeling)
    df_clean["game_id"] = game_id

    # ✅ NEW: reorder columns so game_id comes first
    df_clean = df_clean[
        [
            "game_id",
            "quarter",
            "time",
            "away_team",
            "away_score",
            "home_score",
            "home_team",
            "time_sec",
        ]
    ]

    out_path = OUT_DIR / f"{f.stem}_q.csv"
    df_clean.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Clean shape:", df_clean.shape)
    print("Quarter counts:\n", df_clean["quarter"].value_counts())


def main():
    files = sorted(RAW_DIR.glob("pbp_*.csv"))
    print("Found raw files:", [f.name for f in files])

    for f in files:
        try:
            clean_one_game(f)
        except Exception as e:
            print(f"FAILED: {f.name} — {e}")

if __name__ == "__main__":
    main()
