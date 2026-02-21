import pandas as pd
import numpy as np
from pathlib import Path

CORPUS_PATH = Path("data/corpus/pbp_labeled.parquet")
OUT_DIR = Path("data/model")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# BUILD CORE TIME WINDOW FEATURES
# =========================================================

def build_time_window_features(game_df):

    game_df = game_df.sort_values("time_sec").copy()

    score_diff = game_df["score_diff"].values
    time_sec = game_df["time_sec"].values

    score_change = np.diff(score_diff, prepend=score_diff[0])

    team_pts = np.where(score_change < 0, -score_change, 0)
    opp_pts  = np.where(score_change > 0,  score_change, 0)
    events   = (score_change != 0).astype(int)

    team_60 = []
    opp_60 = []
    events_60 = []

    for i in range(len(game_df)):

        t = time_sec[i]
        window_start = t + 60

        mask = (time_sec <= window_start) & (time_sec >= t)

        team_60.append(team_pts[mask].sum())
        opp_60.append(opp_pts[mask].sum())
        events_60.append(events[mask].sum())

    game_df["team_points_last_60s"] = team_60
    game_df["opp_points_last_60s"]  = opp_60
    game_df["net_points_last_60s"]  = game_df["team_points_last_60s"] - game_df["opp_points_last_60s"]
    game_df["events_last_60s"]      = events_60

    return game_df


# =========================================================
# CONTINUOUS CONTEXT FEATURES
# =========================================================

def build_continuous_context_features(game_df):

    game_df = game_df.sort_values("time_sec").copy()

    # dominance
    game_df["scoring_dominance_ratio"] = (
        game_df["team_points_last_60s"] /
        (game_df["opp_points_last_60s"] + 1)
    )

    # pace
    game_df["points_per_event_last_60s"] = (
        (game_df["team_points_last_60s"] + game_df["opp_points_last_60s"]) /
        (game_df["events_last_60s"] + 1)
    )

    # stability / volatility
    game_df["net_points_std_last_5_events"] = (
        game_df["net_points_last_60s"]
        .rolling(5, min_periods=1)
        .std()
        .fillna(0)
    )

    # trailing pressure
    game_df["is_trailing"] = (game_df["score_diff"] < 0).astype(int)

    game_df["trailing_run_strength"] = (
        game_df["is_trailing"] * game_df["net_points_last_60s"]
    )

    return game_df


# =========================================================
# LABEL
# =========================================================

def classify_momentum(x):
    if x >= 3:
        return 1
    elif x <= -3:
        return -1
    else:
        return 0


# =========================================================
# MAIN
# =========================================================

def main():

    print("Loading labeled corpus...")
    df = pd.read_parquet(CORPUS_PATH)
    print("Original shape:", df.shape)

    df = df.dropna(subset=["momentum_next_10"])

    df["momentum_class"] = df["momentum_next_10"].apply(classify_momentum)
    df["momentum_binary"] = (df["momentum_class"] != 0).astype(int)

    df = df.sort_values(["game_id", "play_id"])

    # ---- BUILD FEATURES ----

    df = (
        df.groupby("game_id", group_keys=False)
        .apply(build_time_window_features)
    )

    df = (
        df.groupby("game_id", group_keys=False)
        .apply(build_continuous_context_features)
    )

    # ---- GAME CONTEXT ----

    df["quarter_num"] = df["quarter"].str.extract(r"(\d)").astype(int)
    df["abs_score_diff"] = df["score_diff"].abs()

    QUARTER_LENGTH = 12 * 60

    df["seconds_remaining_in_game"] = (
        (4 - df["quarter_num"]) * QUARTER_LENGTH + df["time_sec"]
    )

    # ---- FINAL FEATURE SET ----

    feature_cols = [

        "score_diff",
        "abs_score_diff",
        "seconds_remaining_in_game",

        "team_points_last_60s",
        "opp_points_last_60s",
        "net_points_last_60s",
        "events_last_60s",

        "scoring_dominance_ratio",
        "points_per_event_last_60s",
        "net_points_std_last_5_events",
        "trailing_run_strength",
    ]

    df_final = df[feature_cols + [
        "game_id",
        "play_id",
        "momentum_binary",
        "momentum_class"
    ]]

    df_final = df_final.dropna()

    out_path = OUT_DIR / "pbp_features.parquet"
    df_final.to_parquet(out_path, index=False)

    print("Saved feature dataset:", out_path)
    print("Final shape:", df_final.shape)


if __name__ == "__main__":
    main()