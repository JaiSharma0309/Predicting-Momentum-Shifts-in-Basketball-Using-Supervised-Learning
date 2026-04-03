import pandas as pd
from pathlib import Path

CLEAN_DIR = Path("data/clean")
CORPUS_DIR = Path("data/corpus")
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

REGULATION_PERIODS = 4
QUARTER_LENGTH = 12 * 60
OT_LENGTH = 5 * 60

def load_all_clean_games():
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
    df = df.sort_values(
        ["game_id", "period_num", "time_sec", "source_row_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return df

def forward_fill_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill scores within each game only.
    This creates a valid 'current score state' at every play.
    """
    df[["away_score", "home_score"]] = (
        df
        .groupby("game_id")[["away_score", "home_score"]]
        .ffill()
    )

    return df

def add_score_differential(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic derived feature: score differential from the away team's perspective.
    """
    df["score_diff"] = df["away_score"] - df["home_score"]
    return df

def add_play_index(df: pd.DataFrame) -> pd.DataFrame:
    df["play_id"] = (
        df
        .groupby("game_id")
        .cumcount()
    )

    return df

def add_corpus_text(df: pd.DataFrame) -> pd.DataFrame:
    df["play_text"] = (
        df["away_team"].fillna("") + " " + df["home_team"].fillna("")
    ).str.strip()

    return df

def add_absolute_game_time(df: pd.DataFrame) -> pd.DataFrame:
    regulation_elapsed = (df["period_num"] - 1) * QUARTER_LENGTH
    overtime_elapsed = (
        (REGULATION_PERIODS * QUARTER_LENGTH)
        + (df["period_num"] - REGULATION_PERIODS - 1) * OT_LENGTH
    )

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
