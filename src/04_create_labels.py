# 04_create_labels.py

import numpy as np
import pandas as pd
from pathlib import Path

CORPUS_DIR = Path("data/corpus")
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_CORPUS_FILES = [
    CORPUS_DIR / "pbp.parquet",
    CORPUS_DIR / "pbp_corpus.parquet",
    CORPUS_DIR / "pbp_all_games.parquet",
    CORPUS_DIR / "pbp_merged.parquet",
]

OUT_PATH = CORPUS_DIR / "pbp_labeled.parquet"

HORIZON_SECONDS = 120  # 🔥 momentum over next 120 seconds


def find_corpus_path() -> Path:
    for p in CANDIDATE_CORPUS_FILES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find a corpus parquet file in data/corpus. "
        "Tried: " + ", ".join(str(p) for p in CANDIDATE_CORPUS_FILES)
    )


def compute_time_based_momentum(game_df: pd.DataFrame) -> pd.DataFrame:
    game_df = game_df.sort_values("time_sec").copy()

    times = game_df["time_sec"].values
    scores = game_df["score_diff"].values

    future_scores = []

    for i in range(len(game_df)):
        target_time = times[i] - HORIZON_SECONDS

        # index of the last event at or before the horizon
        future_idx = np.searchsorted(times, target_time, side="right") - 1

        if future_idx >= 0:
            future_scores.append(scores[future_idx])
        else:
            future_scores.append(np.nan)

    game_df["future_score_diff"] = future_scores
    game_df["momentum_next_10"] = (
        game_df["future_score_diff"] - game_df["score_diff"]
    )

    return game_df


def main():
    corpus_path = find_corpus_path()
    print("Loading corpus:", corpus_path)

    df = pd.read_parquet(corpus_path)

    required = {"game_id", "play_id", "score_diff", "time_sec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Corpus is missing required columns: {sorted(missing)}")

    df = df.sort_values(["game_id", "play_id"]).reset_index(drop=True)

    # remove old label columns if they exist
    for col in [
        "score_diff_future_10",
        "momentum_next_10",
        "future_score_diff",
    ]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 🔥 compute time-based momentum
    df = (
        df.groupby("game_id", group_keys=False)
        .apply(compute_time_based_momentum)
        .reset_index(drop=True)
    )

    df.to_parquet(OUT_PATH, index=False)

    print("Saved labeled dataset to:", OUT_PATH)
    print("Shape:", df.shape)
    print("\nMomentum_next_10 summary:")
    print(df["momentum_next_10"].describe())


if __name__ == "__main__":
    main()
