"""Create possession-level labels from the event-level play-by-play corpus.

Author: Jai Sharma
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

CORPUS_DIR = Path("data/corpus")
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_CORPUS_FILES = [
    CORPUS_DIR / "pbp.parquet",
    CORPUS_DIR / "pbp_corpus.parquet",
    CORPUS_DIR / "pbp_all_games.parquet",
    CORPUS_DIR / "pbp_merged.parquet",
]

OUT_PATH = CORPUS_DIR / "pbp_labeled.parquet"

LAST_FT_RE = re.compile(r"makes .*free throw (\d+) of (\d+)", re.I)


def find_corpus_path() -> Path:
    """Return the first available corpus parquet path from known candidates.

    @return: Existing parquet path under ``data/corpus``.
    @raises FileNotFoundError: Raised when none of the candidate files exist.
    """
    for path in CANDIDATE_CORPUS_FILES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find a corpus parquet file in data/corpus. "
        "Tried: " + ", ".join(str(path) for path in CANDIDATE_CORPUS_FILES)
    )


def opposite_side(side: str | None) -> str | None:
    """Flip ``away`` to ``home`` and vice versa.

    @param side: Possession side label.
    @return: Opposite side label, or ``None`` when the input is unknown.
    """
    if side == "away":
        return "home"
    if side == "home":
        return "away"
    return None


def infer_event_side(row: pd.Series) -> str | None:
    """Infer which team generated a play description on a given event row.

    @param row: Corpus row containing ``away_team`` and ``home_team`` text.
    @return: ``"away"``, ``"home"``, or ``None`` when the side is ambiguous.
    """
    away_has_text = pd.notna(row["away_team"])
    home_has_text = pd.notna(row["home_team"])

    if away_has_text and not home_has_text:
        return "away"
    if home_has_text and not away_has_text:
        return "home"
    return None


def is_made_field_goal(text: str) -> bool:
    """Check whether text represents a made non-free-throw field goal.

    @param text: Lowercased play description text.
    @return: ``True`` when the event is a made field goal, else ``False``.
    """
    return "makes" in text and "free throw" not in text


def is_last_made_free_throw(text: str) -> bool:
    """Check whether text represents the final made free throw in a trip.

    @param text: Lowercased play description text.
    @return: ``True`` when the text matches a made final free throw, else
        ``False``.
    """
    match = LAST_FT_RE.search(text)
    if not match:
        return False
    return int(match.group(1)) == int(match.group(2))


def terminal_event_reason(text: str, event_side: str | None, offense_side: str | None) -> tuple[bool, str | None, str | None]:
    """Determine whether an event ends the current possession.

    @param text: Lowercased event text.
    @param event_side: Team side associated with the event itself.
    @param offense_side: Team side currently believed to be on offense.
    @return: Tuple of ``(is_terminal, reason, next_offense_side)``.
    """
    if offense_side is None:
        return False, None, None

    if "turnover" in text:
        return True, "turnover", opposite_side(offense_side)

    if "defensive rebound" in text and event_side in {"away", "home"} and event_side != offense_side:
        return True, "def_rebound", event_side

    if is_made_field_goal(text):
        return True, "made_fg", opposite_side(offense_side)

    if is_last_made_free_throw(text):
        return True, "made_last_ft", opposite_side(offense_side)

    return False, None, None


def build_game_possessions(game_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one game's events into possession-level records.

    @param game_df: Event-level DataFrame for a single game.
    @return: Possession-level DataFrame with labels and summary attributes.
    """
    game_df = game_df.sort_values(["game_seconds_elapsed", "play_id"]).copy()
    game_df["event_side"] = game_df.apply(infer_event_side, axis=1)
    game_df["score_change"] = game_df["score_diff"].diff().fillna(0)
    game_df["away_points_on_event"] = game_df["score_change"].clip(lower=0)
    game_df["home_points_on_event"] = (-game_df["score_change"]).clip(lower=0)

    possession_ids = []
    offense_sides = []
    end_reasons = []

    current_possession_id = 0
    current_offense_side = None
    previous_period_num = None

    rows = list(game_df.itertuples(index=False))

    for i, row in enumerate(rows):
        # Force a new possession at period boundaries because possession cannot
        # carry across quarter or overtime breaks.
        if previous_period_num is not None and row.period_num != previous_period_num:
            current_possession_id += 1
            current_offense_side = None

        event_side = row.event_side
        text = str(row.play_text).lower()

        if current_offense_side is None and event_side in {"away", "home"}:
            current_offense_side = event_side

        possession_ids.append(current_possession_id if current_offense_side is not None else np.nan)
        offense_sides.append(current_offense_side)

        terminal, reason, next_offense_side = terminal_event_reason(
            text,
            event_side,
            current_offense_side,
        )
        end_reasons.append(reason)

        previous_period_num = row.period_num

        if terminal:
            current_possession_id += 1
            current_offense_side = next_offense_side

    game_df["possession_id"] = possession_ids
    game_df["offense_side"] = offense_sides
    game_df["terminal_reason"] = end_reasons
    game_df = game_df.dropna(subset=["possession_id", "offense_side"]).copy()
    game_df["possession_id"] = game_df["possession_id"].astype(int)

    grouped = game_df.groupby("possession_id", sort=True)

    possession_df = grouped.agg(
        game_id=("game_id", "first"),
        home_team_code=("game_id", lambda s: str(s.iloc[0])[-3:]),
        offense_side=("offense_side", "first"),
        period_num=("period_num", "first"),
        quarter=("quarter", "first"),
        start_play_id=("play_id", "first"),
        end_play_id=("play_id", "last"),
        start_time=("time", "first"),
        start_time_sec=("time_sec", "first"),
        start_game_seconds_elapsed=("game_seconds_elapsed", "first"),
        start_score_diff=("score_diff", "first"),
        num_events=("play_id", "count"),
        possession_text=("play_text", lambda s: " ".join(str(v) for v in s if str(v) != "nan").strip()),
        end_reason=("terminal_reason", lambda s: next((v for v in reversed(s.tolist()) if v is not None), "period_end")),
    ).reset_index()

    offense_is_home = possession_df["offense_side"] == "home"
    possession_points = grouped.apply(
        # Score totals are computed from the offense team's point stream only.
        lambda g: (
            g["home_points_on_event"].sum()
            if g["offense_side"].iloc[0] == "home"
            else g["away_points_on_event"].sum()
        )
    ).reset_index(name="offense_points")

    defense_points = grouped.apply(
        lambda g: (
            g["away_points_on_event"].sum()
            if g["offense_side"].iloc[0] == "home"
            else g["home_points_on_event"].sum()
        )
    ).reset_index(name="defense_points")

    possession_df = possession_df.merge(possession_points, on="possession_id", how="left")
    possession_df = possession_df.merge(defense_points, on="possession_id", how="left")

    possession_df["offense_is_home"] = offense_is_home.astype(int)
    possession_df["offense_score_margin"] = np.where(
        possession_df["offense_is_home"] == 1,
        -possession_df["start_score_diff"],
        possession_df["start_score_diff"],
    )
    possession_df["possession_scored"] = (possession_df["offense_points"] > 0).astype(int)
    possession_df["points_on_possession"] = possession_df["offense_points"]

    return possession_df


def main():
    """Create and save the possession-labeled corpus parquet file.

    @return: ``None``.
    """
    corpus_path = find_corpus_path()
    print("Loading corpus:", corpus_path)

    df = pd.read_parquet(corpus_path)
    required = {"game_id", "play_id", "period_num", "time_sec", "game_seconds_elapsed", "score_diff", "play_text", "away_team", "home_team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Corpus is missing required columns: {sorted(missing)}")

    possession_df = (
        df.groupby("game_id", group_keys=False)
        .apply(build_game_possessions)
        .reset_index(drop=True)
    )

    possession_df.to_parquet(OUT_PATH, index=False)

    print("Saved possession dataset to:", OUT_PATH)
    print("Shape:", possession_df.shape)
    print("\nPossession scored distribution:")
    print(possession_df["possession_scored"].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()
