from pathlib import Path

import numpy as np
import pandas as pd

CORPUS_PATH = Path("data/corpus/pbp_labeled.parquet")
OUT_DIR = Path("data/model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGULATION_LENGTH = 4 * 12 * 60


def build_possession_history_features(game_df: pd.DataFrame) -> pd.DataFrame:
    game_df = game_df.sort_values(["start_game_seconds_elapsed", "possession_id"]).copy()

    game_df["prev_possession_scored"] = game_df["possession_scored"].shift(1).fillna(0)
    game_df["prev_possession_points"] = game_df["points_on_possession"].shift(1).fillna(0)
    game_df["prev_offense_margin"] = game_df["offense_score_margin"].shift(1).fillna(0)
    game_df["prev_num_events"] = game_df["num_events"].shift(1).fillna(0)

    game_df["score_rate_last_3_possessions"] = (
        game_df["possession_scored"]
        .shift(1)
        .rolling(3, min_periods=1)
        .mean()
        .fillna(0)
    )
    game_df["points_last_3_possessions"] = (
        game_df["points_on_possession"]
        .shift(1)
        .rolling(3, min_periods=1)
        .sum()
        .fillna(0)
    )
    game_df["points_last_5_possessions"] = (
        game_df["points_on_possession"]
        .shift(1)
        .rolling(5, min_periods=1)
        .sum()
        .fillna(0)
    )

    same_side_scored = []
    current_streak = 0
    previous_side = None

    for offense_side, scored in zip(game_df["offense_side"], game_df["possession_scored"]):
        if offense_side == previous_side:
            current_streak = current_streak + 1 if scored else 0
        else:
            current_streak = 1 if scored else 0
        same_side_scored.append(current_streak)
        previous_side = offense_side

    game_df["same_side_scoring_streak"] = (
        pd.Series(same_side_scored, index=game_df.index).shift(1).fillna(0)
    )

    return game_df


def build_game_state_features(game_df: pd.DataFrame) -> pd.DataFrame:
    game_df = game_df.sort_values(["start_game_seconds_elapsed", "possession_id"]).copy()

    game_df["abs_offense_margin"] = game_df["offense_score_margin"].abs()
    game_df["offense_leading"] = (game_df["offense_score_margin"] > 0).astype(int)
    game_df["offense_trailing"] = (game_df["offense_score_margin"] < 0).astype(int)
    game_df["offense_tied"] = (game_df["offense_score_margin"] == 0).astype(int)
    game_df["one_possession_game"] = (game_df["abs_offense_margin"] <= 3).astype(int)
    game_df["two_possession_game"] = (game_df["abs_offense_margin"] <= 6).astype(int)
    game_df["seconds_remaining_in_regulation"] = np.clip(
        REGULATION_LENGTH - game_df["start_game_seconds_elapsed"],
        0,
        None,
    )
    game_df["last_5_minutes"] = (game_df["seconds_remaining_in_regulation"] <= 300).astype(int)
    game_df["last_2_minutes"] = (game_df["seconds_remaining_in_regulation"] <= 120).astype(int)
    game_df["in_overtime"] = (game_df["period_num"] > 4).astype(int)

    return game_df


def build_text_context(game_df: pd.DataFrame) -> pd.DataFrame:
    game_df = game_df.sort_values(["start_game_seconds_elapsed", "possession_id"]).copy()
    previous_text = game_df["possession_text"].shift(1).fillna("")
    previous_text_2 = game_df["possession_text"].shift(2).fillna("")
    previous_text_3 = game_df["possession_text"].shift(3).fillna("")

    game_df["recent_possession_text"] = (
        previous_text + " " + previous_text_2 + " " + previous_text_3
    ).str.strip()

    return game_df


def main():
    print("Loading possession dataset...")
    df = pd.read_parquet(CORPUS_PATH)
    print("Original shape:", df.shape)

    df = df.dropna(subset=["offense_score_margin", "points_on_possession"]).copy()
    df = (
        df.groupby("game_id", group_keys=False)
        .apply(build_possession_history_features)
        .reset_index(drop=True)
    )
    df = (
        df.groupby("game_id", group_keys=False)
        .apply(build_game_state_features)
        .reset_index(drop=True)
    )
    df = (
        df.groupby("game_id", group_keys=False)
        .apply(build_text_context)
        .reset_index(drop=True)
    )

    feature_cols = [
        "possession_id",
        "period_num",
        "start_time_sec",
        "start_game_seconds_elapsed",
        "offense_is_home",
        "offense_score_margin",
        "abs_offense_margin",
        "offense_leading",
        "offense_trailing",
        "offense_tied",
        "one_possession_game",
        "two_possession_game",
        "seconds_remaining_in_regulation",
        "last_5_minutes",
        "last_2_minutes",
        "in_overtime",
        "prev_possession_scored",
        "prev_possession_points",
        "prev_offense_margin",
        "prev_num_events",
        "score_rate_last_3_possessions",
        "points_last_3_possessions",
        "points_last_5_possessions",
        "same_side_scoring_streak",
    ]

    df_final = df[
        feature_cols
        + [
            "game_id",
            "home_team_code",
            "offense_side",
            "quarter",
            "start_play_id",
            "end_play_id",
            "start_time",
            "possession_text",
            "recent_possession_text",
            "num_events",
            "end_reason",
            "possession_scored",
            "points_on_possession",
        ]
    ].dropna()

    out_path = OUT_DIR / "pbp_features.parquet"
    df_final.to_parquet(out_path, index=False)

    print("Saved possession feature dataset:", out_path)
    print("Final shape:", df_final.shape)
    print("\nTarget distribution:")
    print(df_final["possession_scored"].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()
