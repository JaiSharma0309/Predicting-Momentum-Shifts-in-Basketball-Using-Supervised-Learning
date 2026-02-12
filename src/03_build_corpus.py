import pandas as pd
from pathlib import Path

CLEAN_DIR = Path("data/clean")
CORPUS_DIR = Path("data/corpus")
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

def load_all_clean_games():
    """
    Load all cleaned play-by-play files into a single dataframe.
    """
    files = sorted(CLEAN_DIR.glob("pbp_*_q.csv"))
    print("Loading cleaned files:")
    for f in files:
        print(" -", f.name)

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    corpus = pd.concat(dfs, ignore_index=True)
    print("\nCombined corpus shape:", corpus.shape)
    return corpus

def sort_within_games(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure plays are in correct temporal order within each game.
    We sort by game_id, then quarter, then time_sec descending
    (since time_sec is time remaining).
    """
    df = df.sort_values(
        ["game_id", "quarter", "time_sec"],
        ascending=[True, True, False]
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
    """
    Add a play index within each game (0, 1, 2, ...).
    Useful for windowing and labels later.
    """
    df["play_id"] = (
        df
        .groupby("game_id")
        .cumcount()
    )

    return df

def add_corpus_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a single text field per play for corpus analysis.
    This combines away and home play descriptions into one string.
    """
    df["play_text"] = (
        df["away_team"].fillna("") + " " + df["home_team"].fillna("")
    ).str.strip()

    return df

def main():
    # STEP 1: load all cleaned games
    corpus = load_all_clean_games()

    # STEP 2: ensure correct ordering
    corpus = sort_within_games(corpus)

    # STEP 3: forward-fill scores within each game
    corpus = forward_fill_scores(corpus)

    # STEP 4: add simple derived features
    corpus = add_score_differential(corpus)
    corpus = add_play_index(corpus)

    # STEP 5: add corpus text (NLP-friendly column)
    corpus = add_corpus_text(corpus)

    # STEP 6: final column order (clean and readable)
    corpus = corpus[
        [
            "game_id",
            "play_id",
            "quarter",
            "time",
            "time_sec",
            "play_text",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
            "score_diff",
        ]
    ]

    # Save as Parquet (efficient for large datasets)
    out_path = CORPUS_DIR / "pbp_all_games.parquet"
    corpus.to_parquet(out_path, index=False)

    print("\nSaved corpus to:", out_path)
    print("Final corpus shape:", corpus.shape)

if __name__ == "__main__":
    main()

import pandas as pd

corpus = pd.read_parquet("data/corpus/pbp_all_games.parquet")
print(corpus.head())
