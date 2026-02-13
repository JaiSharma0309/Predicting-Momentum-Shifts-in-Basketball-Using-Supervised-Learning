# 05_build_features.py

import pandas as pd
import numpy as np
from pathlib import Path

CORPUS_PATH = Path("data/corpus/pbp_labeled.parquet")
OUT_DIR = Path("data/model")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def classify_momentum(x):
    # Keep your same thresholding style, just applied to momentum_next_10 now.
    if x > 1:
        return 1
    elif x < -1:
        return -1
    else:
        return 0


def main():
    print("Loading labeled corpus...")
    df = pd.read_parquet(CORPUS_PATH)
    print("Original shape:", df.shape)

    # Need the new label columns to exist
    required = {"game_id", "play_id", "quarter", "time_sec", "score_diff", "away_team", "home_team", "momentum_next_10"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"pbp_labeled.parquet is missing required columns: {sorted(missing)}")

    # Drop rows where future momentum is undefined (end of game)
    df = df.dropna(subset=["momentum_next_10"])

    # ====== LABEL ======
    df["momentum_class"] = df["momentum_next_10"].apply(classify_momentum)

    # Sort for sequential features
    df = df.sort_values(["game_id", "play_id"]).reset_index(drop=True)

    # =====================================================
    # BASIC STATE FEATURES
    # =====================================================
    df["quarter_num"] = df["quarter"].astype(str).str.extract(r"(\d)").astype(float)
    df["abs_score_diff"] = df["score_diff"].abs()
    df["is_away_leading"] = (df["score_diff"] > 0).astype(int)

    # =====================================================
    # SCORING FEATURES
    # =====================================================
    df["last_score_change"] = (
        df.groupby("game_id")["score_diff"].diff().fillna(0)
    )

    df["points_this_play"] = df["last_score_change"]

    # Who scored?
    df["away_scored"] = (df["points_this_play"] > 0).astype(int)
    df["home_scored"] = (df["points_this_play"] < 0).astype(int)

    # Rolling scoring totals
    df["points_last_3"] = (
        df.groupby("game_id")["points_this_play"]
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["points_last_5"] = (
        df.groupby("game_id")["points_this_play"]
        .rolling(5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # =====================================================
    # SCORING RUN LENGTH
    # =====================================================
    df["scoring_team"] = np.where(
        df["points_this_play"] > 0, "away",
        np.where(df["points_this_play"] < 0, "home", "none")
    )

    df["scoring_run_length"] = 0

    for game_id, group in df.groupby("game_id", sort=False):
        run = 0
        prev_team = None
        for idx in group.index:
            team = df.at[idx, "scoring_team"]

            if team == prev_team and team != "none":
                run += 1
            elif team != "none":
                run = 1
            else:
                run = 0

            df.at[idx, "scoring_run_length"] = run
            prev_team = team if team != "none" else prev_team

    # =====================================================
    # EVENT FEATURES (FROM TEXT)
    # =====================================================
    df["play_text"] = (
        df["away_team"].astype(str).fillna("") + " " +
        df["home_team"].astype(str).fillna("")
    ).str.lower()

    df["is_turnover"] = df["play_text"].str.contains("turnover", na=False).astype(int)
    df["is_foul"] = df["play_text"].str.contains("foul", na=False).astype(int)
    df["is_rebound"] = df["play_text"].str.contains("rebound", na=False).astype(int)
    df["is_timeout"] = df["play_text"].str.contains("timeout", na=False).astype(int)

    # Timeout after run
    df["timeout_after_run"] = (
        (df["is_timeout"] == 1) &
        (df["scoring_run_length"] >= 3)
    ).astype(int)

    # Fill any remaining numeric NaNs safely
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # =====================================================
    # FINAL FEATURE SELECTION
    # =====================================================
    feature_cols = [
        "game_id",
        "play_id",
        "quarter_num",
        "time_sec",
        "score_diff",
        "abs_score_diff",
        "is_away_leading",
        "points_this_play",
        "points_last_3",
        "points_last_5",
        "scoring_run_length",
        "is_turnover",
        "is_foul",
        "is_rebound",
        "is_timeout",
        "timeout_after_run",
    ]

    final_cols = feature_cols + ["momentum_class"]

    df_final = df[final_cols]

    out_path = OUT_DIR / "pbp_features.parquet"
    df_final.to_parquet(out_path, index=False)

    print("Saved feature dataset to:", out_path)
    print("Final shape:", df_final.shape)
    print("\nClass distribution:")
    print(df_final["momentum_class"].value_counts())


if __name__ == "__main__":
    main()
