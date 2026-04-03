"""Combine cleaned play-by-play files into a unified modeling corpus.

Author: Jai Sharma
"""

import pandas as pd
from pathlib import Path

CLEAN_DIR = Path("data/clean")
CORPUS_DIR = Path("data/corpus")
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

REGULATION_PERIODS = 4
QUARTER_LENGTH = 12 * 60
OT_LENGTH = 5 * 60

def load_all_clean_games():
    """Load every cleaned game file and append a stable source row index.

    @return: Concatenated DataFrame containing all cleaned games.
    """
    files = sorted(CLEAN_DIR.glob("pbp_*_q.csv"))
    print("Loading cleaned files:")
    for f in files:
        print(" -", f.name)

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_row_id"] = range(len(df))
        dfs.append(df)

    corpus = pd.concat(dfs, ignore_index=True)
    print("\nCombined corpus shape:", corpus.shape)
    return corpus

def sort_within_games(df: pd.DataFrame) -> pd.DataFrame:
    """Sort events chronologically within each game.

    @param df: Corpus DataFrame containing at least game, period, clock, and
        source row columns.
    @return: Sorted DataFrame with reset index.
    """
    df = df.sort_values(
        ["game_id", "period_num", "time_sec", "source_row_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return df

def forward_fill_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill scoreboard state within each game.

    @param df: Corpus DataFrame with score columns.
    @return: DataFrame with ``away_score`` and ``home_score`` forward-filled
        by game.
    """
    df[["away_score", "home_score"]] = (
        df
        .groupby("game_id")[["away_score", "home_score"]]
        .ffill()
    )

    return df

def add_score_differential(df: pd.DataFrame) -> pd.DataFrame:
    """Create score differential from the away-team perspective.

    @param df: Corpus DataFrame with away and home score columns.
    @return: DataFrame with a ``score_diff`` column added.
    """
    df["score_diff"] = df["away_score"] - df["home_score"]
    return df

def add_play_index(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a sequential play identifier within each game.

    @param df: Corpus DataFrame.
    @return: DataFrame with a ``play_id`` column added.
    """
    df["play_id"] = (
        df
        .groupby("game_id")
        .cumcount()
    )

    return df

def add_corpus_text(df: pd.DataFrame) -> pd.DataFrame:
    """Build a lightweight text field from away/home play descriptions.

    @param df: Corpus DataFrame.
    @return: DataFrame with a combined ``play_text`` column added.
    """
    df["play_text"] = (
        df["away_team"].fillna("") + " " + df["home_team"].fillna("")
    ).str.strip()

    return df

def add_absolute_game_time(df: pd.DataFrame) -> pd.DataFrame:
    """Convert period-relative clock values into elapsed game seconds.

    @param df: Corpus DataFrame with period numbers and remaining clock values.
    @return: DataFrame with absolute elapsed-time columns added.
    """
    regulation_elapsed = (df["period_num"] - 1) * QUARTER_LENGTH
    overtime_elapsed = (
        (REGULATION_PERIODS * QUARTER_LENGTH)
        + (df["period_num"] - REGULATION_PERIODS - 1) * OT_LENGTH
    )

    # Regulation and overtime use different clock lengths, so period length
    # must be computed row by row before converting to elapsed game time.
    df["period_length_sec"] = df["period_num"].apply(
        lambda period_num: QUARTER_LENGTH if period_num <= REGULATION_PERIODS else OT_LENGTH
    )
    df["period_start_elapsed"] = regulation_elapsed.where(
        df["period_num"] <= REGULATION_PERIODS,
        overtime_elapsed,
    )
    df["game_seconds_elapsed"] = (
        df["period_start_elapsed"] + (df["period_length_sec"] - df["time_sec"])
    )

    return df

def main():
    """Build and save the combined corpus parquet file.

    @return: ``None``.
    """
    corpus = load_all_clean_games()
    corpus = sort_within_games(corpus)
    corpus = forward_fill_scores(corpus)
    corpus = add_absolute_game_time(corpus)
    corpus = add_score_differential(corpus)
    corpus = add_play_index(corpus)
    corpus = add_corpus_text(corpus)
    
    corpus = corpus[
        [
            "game_id",
            "play_id",
            "period_num",
            "quarter",
            "time",
            "time_sec",
            "game_seconds_elapsed",
            "play_text",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
            "score_diff",
        ]
    ]

    out_path = CORPUS_DIR / "pbp_all_games.parquet"
    corpus.to_parquet(out_path, index=False)

    print("\nSaved corpus to:", out_path)
    print("Final corpus shape:", corpus.shape)

if __name__ == "__main__":
    main()
