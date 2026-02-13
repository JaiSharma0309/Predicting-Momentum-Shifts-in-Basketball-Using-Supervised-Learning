# 04_create_labels.py

import pandas as pd
from pathlib import Path

CORPUS_DIR = Path("data/corpus")
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

# We’ll try these in order (so this works even if your corpus filename differs)
CANDIDATE_CORPUS_FILES = [
    CORPUS_DIR / "pbp.parquet",
    CORPUS_DIR / "pbp_corpus.parquet",
    CORPUS_DIR / "pbp_all_games.parquet",
    CORPUS_DIR / "pbp_merged.parquet",
]

OUT_PATH = CORPUS_DIR / "pbp_labeled.parquet"

HORIZON = 10  # momentum_next_10 = score_diff(t+10) - score_diff(t)


def find_corpus_path() -> Path:
    for p in CANDIDATE_CORPUS_FILES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find a corpus parquet file in data/corpus. "
        "Tried: " + ", ".join(str(p) for p in CANDIDATE_CORPUS_FILES)
    )


def main():
    corpus_path = find_corpus_path()
    print("Loading corpus:", corpus_path)

    df = pd.read_parquet(corpus_path)

    required = {"game_id", "play_id", "score_diff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Corpus is missing required columns: {sorted(missing)}")

    df = df.sort_values(["game_id", "play_id"]).reset_index(drop=True)

    # If older label columns exist, remove them so we don’t mix definitions
    for col in ["score_diff_future_5", "momentum_next_5", "score_diff_future_10", "momentum_next_10"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df["score_diff_future_10"] = (
        df.groupby("game_id")["score_diff"].shift(-HORIZON)
    )

    df["momentum_next_10"] = df["score_diff_future_10"] - df["score_diff"]

    df.to_parquet(OUT_PATH, index=False)

    print("Saved labeled dataset to:", OUT_PATH)
    print("Shape:", df.shape)
    print("\nMomentum_next_10 summary:")
    print(df["momentum_next_10"].describe())


if __name__ == "__main__":
    main()
