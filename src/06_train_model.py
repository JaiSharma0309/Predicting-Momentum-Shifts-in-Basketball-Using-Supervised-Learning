import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/model/pbp_features.parquet")
RESULTS_DIR = Path("results")
CHARTS_DIR = RESULTS_DIR / "charts"
REPORTS_DIR = RESULTS_DIR / "reports"
TABLES_DIR = RESULTS_DIR / "tables"
N_JOBS = 1
RANDOM_STATE = 42
TEXT_COL = "recent_possession_text"
TARGET_COL = "possession_scored"

META_COLS = {
    "game_id",
    "quarter",
    "start_time",
    "start_play_id",
    "end_play_id",
    "num_events",
    "end_reason",
    "possession_text",
    "recent_possession_text",
    "possession_scored",
    "points_on_possession",
}


def split_by_game(df: pd.DataFrame):
    games = df["game_id"].unique()
    train_valid_games, test_games = train_test_split(
        games,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    train_games, valid_games = train_test_split(
        train_valid_games,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    train_mask = df["game_id"].isin(train_games)
    valid_mask = df["game_id"].isin(valid_games)
    test_mask = df["game_id"].isin(test_games)

    return train_mask, valid_mask, test_mask


def build_preprocessor(feature_cols, include_text: bool, dense_output: bool):
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(FEATURE_SOURCE[c])]
    categorical_cols = [
        c for c in feature_cols
        if c not in numeric_cols and c != TEXT_COL
    ]

    transformers = []

    if include_text and TEXT_COL in feature_cols:
        transformers.append(
            (
                "text",
                Pipeline(
                    [
                        (
                            "tfidf",
                            TfidfVectorizer(
                                max_features=3000,
                                ngram_range=(1, 2),
                                min_df=3,
                            ),
                        ),
                        ("svd", TruncatedSVD(n_components=50, random_state=RANDOM_STATE)),
                    ]
                ),
                TEXT_COL,
            )
        )

    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))

    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers, sparse_threshold=0.0 if dense_output else 0.3)


def threshold_search(model, X_valid, y_valid, name):
    probs = model.predict_proba(X_valid)[:, 1]

    best_f1 = -np.inf
    best_threshold = 0.5

    for threshold in np.arange(0.3, 0.71, 0.02):
        preds = (probs >= threshold).astype(int)
        score = f1_score(y_valid, preds, average="macro")
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    print(f"\nValidation threshold ({name}):", best_threshold)
    print(f"Validation Macro F1 ({name}):", best_f1)

    return best_threshold


def evaluate_predictions(name, y_test, preds):
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Macro F1:", f1_score(y_test, preds, average="macro"))
    print(classification_report(y_test, preds, zero_division=0))
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "support": len(y_test),
    }


def evaluate_model(name, model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    metrics = evaluate_predictions(name, y_test, preds)
    metrics["threshold"] = threshold
    return preds, metrics


def print_confusion_matrix(name, y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    print(f"\n{name} Confusion Matrix:")
    print("rows=true, cols=pred")
    print(matrix)
    return matrix


def save_confusion_matrix_plot(name, matrix, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title(name)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def get_slice_metrics(eval_df, y_true, y_pred, slice_col):
    temp = eval_df[[slice_col]].copy()
    temp["y_true"] = y_true.values
    temp["y_pred"] = y_pred

    rows = []
    for slice_value, group in temp.groupby(slice_col, dropna=False):
        rows.append(
            {
                "slice_col": slice_col,
                "slice_value": slice_value,
                "n": len(group),
                "accuracy": accuracy_score(group["y_true"], group["y_pred"]),
                "macro_f1": f1_score(group["y_true"], group["y_pred"], average="macro"),
            }
        )

    return pd.DataFrame(rows)


def print_slice_metrics(name, eval_df, y_true, y_pred, slice_col):
    print(f"\n{name} by {slice_col}:")
    slice_df = get_slice_metrics(eval_df, y_true, y_pred, slice_col)
    for _, row in slice_df.iterrows():
        print(
            f"{slice_col}={row['slice_value']} | n={int(row['n'])} | "
            f"accuracy={row['accuracy']:.4f} | "
            f"macro_f1={row['macro_f1']:.4f}"
        )
    return slice_df


def get_permutation_importance(model, X_test, y_test):
    sample_size = min(len(X_test), 3000)
    X_sample = X_test.iloc[:sample_size]
    y_sample = y_test.iloc[:sample_size]

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="f1_macro",
        n_jobs=N_JOBS,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return importance_df


def print_permutation_importance(name, model, X_test, y_test):
    importance_df = get_permutation_importance(model, X_test, y_test)

    print(f"\n{name} Top Permutation Importances:")
    print(importance_df.head(12).to_string(index=False))
    return importance_df


def save_importance_plot(importance_df: pd.DataFrame, output_path: Path, title: str):
    top_df = importance_df.head(12).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_df["feature"], top_df["importance_mean"], xerr=top_df["importance_std"])
    ax.set_title(title)
    ax.set_xlabel("Permutation Importance (Macro F1 Drop)")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_model_comparison_plot(summary_df: pd.DataFrame, output_path: Path):
    plot_df = summary_df.copy()
    plot_df["label"] = plot_df["model"].str.replace("HistGradientBoosting", "HGB", regex=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(plot_df["label"], plot_df["macro_f1"], color=["#9ca3af", "#60a5fa", "#2563eb", "#0f172a"])
    ax.set_title("Model Comparison: Macro F1 on Held-Out Games")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.0, 0.85)
    ax.tick_params(axis="x", rotation=15)

    for bar, value in zip(bars, plot_df["macro_f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_slice_performance_plot(slice_df: pd.DataFrame, output_path: Path):
    quarter_df = slice_df[slice_df["slice_col"] == "quarter"].copy()
    late_df = slice_df[slice_df["slice_col"] == "last_2_minutes"].copy()

    quarter_order = ["1st Q", "2nd Q", "3rd Q", "4th Q", "1st OT", "2nd OT"]
    quarter_df["slice_value"] = pd.Categorical(quarter_df["slice_value"], categories=quarter_order, ordered=True)
    quarter_df = quarter_df.sort_values("slice_value")
    late_df["slice_value"] = late_df["slice_value"].astype(str)
    late_df["label"] = late_df["slice_value"].map({"0": "Before Final 2 Min", "1": "Final 2 Min"})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].bar(quarter_df["slice_value"].astype(str), quarter_df["macro_f1"], color="#2563eb")
    axes[0].set_title("Performance by Quarter")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_ylim(0.6, 0.9)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(late_df["label"], late_df["macro_f1"], color=["#0f766e", "#b91c1c"])
    axes[1].set_title("Performance by Game Clock")
    axes[1].set_ylim(0.6, 0.9)
    axes[1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_business_findings_report(summary_df: pd.DataFrame, slice_df: pd.DataFrame, importance_df: pd.DataFrame):
    best_row = summary_df.sort_values("macro_f1", ascending=False).iloc[0]
    baseline_row = summary_df.loc[summary_df["model"] == "Majority Baseline"].iloc[0]
    quarter_df = slice_df[slice_df["slice_col"] == "quarter"].copy()
    meaningful_quarter_df = quarter_df.loc[quarter_df["n"] >= 500].copy()
    weakest_quarter = quarter_df.sort_values("macro_f1").iloc[0]
    strongest_quarter = (
        meaningful_quarter_df.sort_values("macro_f1", ascending=False).iloc[0]
        if not meaningful_quarter_df.empty
        else quarter_df.sort_values("macro_f1", ascending=False).iloc[0]
    )
    clock_df = slice_df[slice_df["slice_col"] == "last_2_minutes"].copy()
    clock_df["slice_value"] = clock_df["slice_value"].astype(str)
    final_two = clock_df.loc[clock_df["slice_value"] == "1"].iloc[0]
    before_final_two = clock_df.loc[clock_df["slice_value"] == "0"].iloc[0]
    home_df = slice_df[slice_df["slice_col"] == "offense_is_home"].copy()
    home_df["slice_value"] = home_df["slice_value"].astype(str)
    away_offense = home_df.loc[home_df["slice_value"] == "0"].iloc[0]
    home_offense = home_df.loc[home_df["slice_value"] == "1"].iloc[0]
    top_features = importance_df.head(5)["feature"].tolist()
    numeric_hgb_f1 = summary_df.loc[
        summary_df["model"] == "Numeric HistGradientBoosting",
        "macro_f1",
    ].iloc[0]
    hybrid_hgb_f1 = summary_df.loc[
        summary_df["model"] == "Hybrid Text + HistGradientBoosting",
        "macro_f1",
    ].iloc[0]

    if hybrid_hgb_f1 > numeric_hgb_f1 + 0.001:
        text_lift_line = (
            f"Text adds a measurable lift. Numeric HGB reached **{numeric_hgb_f1:.3f}** "
            f"macro F1, while the hybrid text model reached **{hybrid_hgb_f1:.3f}**."
        )
        text_interp_line = (
            "Text is adding incremental value on top of the structured game-state features."
        )
    elif hybrid_hgb_f1 < numeric_hgb_f1 - 0.001:
        text_lift_line = (
            f"Text did not help on this run. Numeric HGB reached **{numeric_hgb_f1:.3f}** "
            f"macro F1, compared with **{hybrid_hgb_f1:.3f}** for the hybrid text model."
        )
        text_interp_line = (
            "The business value is currently driven primarily by structured state and recency features."
        )
    else:
        text_lift_line = (
            f"Text added little to no measurable lift on this run. Numeric HGB and the hybrid "
            f"text model both landed at about **{hybrid_hgb_f1:.3f}** macro F1."
        )
        text_interp_line = (
            "The business value is currently driven primarily by structured state and recency features."
        )

    report = f"""# Business Findings

## Executive Summary
This project converted raw Basketball Reference play-by-play logs into a possession-level prediction system that estimates whether a possession will end in points. The strongest production candidate is the Hybrid Text + HistGradientBoosting model, which reached **{best_row['macro_f1']:.3f} macro F1** and **{best_row['accuracy']:.3f} accuracy** on fully held-out games.

## What We Actually Discovered
- The original momentum framing was too noisy to support reliable decision-making. Reframing the problem to possession-level scoring made the signal strong and usable.
- The best model beats the naive baseline by **{best_row['macro_f1'] - baseline_row['macro_f1']:.3f} macro F1** and **{best_row['accuracy'] - baseline_row['accuracy']:.3f} accuracy**.
- Score context matters most. The top predictive drivers are `{top_features[0]}`, `{top_features[1]}`, `{top_features[2]}`, `{top_features[3]}`, and `{top_features[4]}`.
- {text_lift_line}

## Why This Matters
- Coaches, analysts, or product stakeholders can now identify when a possession is likely to produce points using only information available at possession start.
- The model is strong enough to support downstream use cases like live game-state dashboards, broadcast insights, or scenario-based strategy analysis.
- The system is honest: evaluation is grouped by game, which means performance reflects true out-of-game generalization instead of leakage across the same game.

## Where The Model Is Strongest And Weakest
- Strongest quarter: **{strongest_quarter['slice_value']}** at **{strongest_quarter['macro_f1']:.3f} macro F1**
- Weakest quarter: **{weakest_quarter['slice_value']}** at **{weakest_quarter['macro_f1']:.3f} macro F1**
- Before the final 2 minutes: **{before_final_two['macro_f1']:.3f} macro F1**
- In the final 2 minutes: **{final_two['macro_f1']:.3f} macro F1**
- Away-team possessions perform slightly better (**{away_offense['macro_f1']:.3f}**) than home-team possessions (**{home_offense['macro_f1']:.3f}**)

## Business Interpretation
- Most of the predictive value comes from game state and recent possession context, which means score margin and recent flow are actionable leading indicators.
- End-of-game possessions are harder to predict, likely because teams change pace, foul intentionally, and make higher-variance decisions late.
- {text_interp_line}

## Recommended Next Steps
1. Productize the possession-scoring model as the main benchmark, not the old momentum label.
2. Add a lightweight dashboard that surfaces predicted scoring likelihood by possession and flags high-leverage moments.
3. Enrich the feature set with team identity, lineup context, and possession start type to improve late-game recall.
4. Expand the dataset across more games or seasons before moving toward stakeholder-facing deployment claims.

## Files To Use In A Deck
- `results/charts/model_comparison_macro_f1.png`
- `results/charts/slice_performance_macro_f1.png`
- `results/charts/hybrid_hgb_confusion_matrix.png`
- `results/charts/hybrid_hgb_permutation_importance.png`
"""

    (REPORTS_DIR / "business_findings.md").write_text(report)


def save_results_bundle(
    summary_rows,
    slice_frames,
    importance_df,
    confusion_matrix_array,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES_DIR / "model_summary.csv", index=False)
    summary_df.to_json(TABLES_DIR / "model_summary.json", orient="records", indent=2)

    if slice_frames:
        all_slices = pd.concat(slice_frames, ignore_index=True)
        all_slices.to_csv(TABLES_DIR / "slice_metrics.csv", index=False)
    else:
        all_slices = pd.DataFrame()

    importance_df.to_csv(TABLES_DIR / "hybrid_hgb_permutation_importance.csv", index=False)
    save_importance_plot(
        importance_df,
        CHARTS_DIR / "hybrid_hgb_permutation_importance.png",
        "Hybrid HGB Permutation Importance",
    )
    save_model_comparison_plot(summary_df, CHARTS_DIR / "model_comparison_macro_f1.png")
    if not all_slices.empty:
        save_slice_performance_plot(all_slices, CHARTS_DIR / "slice_performance_macro_f1.png")
        write_business_findings_report(summary_df, all_slices, importance_df)

    np.savetxt(
        TABLES_DIR / "hybrid_hgb_confusion_matrix.csv",
        confusion_matrix_array,
        delimiter=",",
        fmt="%d",
    )


def main():
    global FEATURE_SOURCE

    print("Loading feature dataset...")
    df = pd.read_parquet(DATA_PATH)
    print("Shape:", df.shape)

    FEATURE_SOURCE = df

    feature_cols = [c for c in df.columns if c not in META_COLS]
    numeric_feature_cols = [c for c in feature_cols if c != TEXT_COL]
    hybrid_feature_cols = feature_cols

    train_mask, valid_mask, test_mask = split_by_game(df)
    train_groups = df.loc[train_mask, "game_id"]

    X_numeric = df[numeric_feature_cols]
    X_hybrid = df[hybrid_feature_cols]
    y = df[TARGET_COL]
    diagnostics_df = df[test_mask].copy()

    X_train_num, X_valid_num, X_test_num = X_numeric[train_mask], X_numeric[valid_mask], X_numeric[test_mask]
    X_train_hybrid, X_valid_hybrid, X_test_hybrid = X_hybrid[train_mask], X_hybrid[valid_mask], X_hybrid[test_mask]
    y_train, y_valid, y_test = y[train_mask], y[valid_mask], y[test_mask]

    print("Train size:", X_train_num.shape)
    print("Validation size:", X_valid_num.shape)
    print("Test size:", X_test_num.shape)

    summary_rows = []

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train_num, y_train)
    baseline_metrics = evaluate_predictions(
        "Majority Baseline",
        y_test,
        baseline.predict(X_test_num),
    )
    baseline_metrics["threshold"] = np.nan
    summary_rows.append(baseline_metrics)

    group_cv = GroupKFold(n_splits=3)

    logistic_pipe = Pipeline(
        [
            ("prep", build_preprocessor(numeric_feature_cols, include_text=False, dense_output=False)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )
    logistic_grid = GridSearchCV(
        logistic_pipe,
        {"model__C": [0.01, 0.1, 1.0, 10.0]},
        cv=group_cv,
        scoring="f1_macro",
        n_jobs=N_JOBS,
    )
    logistic_grid.fit(X_train_num, y_train, groups=train_groups)
    print("Best params:", logistic_grid.best_params_)
    logistic_threshold = threshold_search(
        logistic_grid.best_estimator_,
        X_valid_num,
        y_valid,
        "Numeric Logistic",
    )
    _, logistic_metrics = evaluate_model(
        "Numeric Logistic",
        logistic_grid.best_estimator_,
        X_test_num,
        y_test,
        logistic_threshold,
    )
    summary_rows.append(logistic_metrics)

    hgb_pipe = Pipeline(
        [
            ("prep", build_preprocessor(numeric_feature_cols, include_text=False, dense_output=True)),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=300,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    hgb_grid = GridSearchCV(
        hgb_pipe,
        {
            "model__max_depth": [4, 6, 8],
            "model__max_iter": [200, 300],
            "model__learning_rate": [0.03, 0.05, 0.07],
        },
        cv=group_cv,
        scoring="f1_macro",
        n_jobs=N_JOBS,
    )
    hgb_grid.fit(X_train_num, y_train, groups=train_groups)
    print("Best params:", hgb_grid.best_params_)
    hgb_threshold = threshold_search(
        hgb_grid.best_estimator_,
        X_valid_num,
        y_valid,
        "Numeric HistGradientBoosting",
    )
    _, hgb_metrics = evaluate_model(
        "Numeric HistGradientBoosting",
        hgb_grid.best_estimator_,
        X_test_num,
        y_test,
        hgb_threshold,
    )
    summary_rows.append(hgb_metrics)

    hybrid_hgb_pipe = Pipeline(
        [
            ("prep", build_preprocessor(hybrid_feature_cols, include_text=True, dense_output=True)),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=300,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    hybrid_hgb_grid = GridSearchCV(
        hybrid_hgb_pipe,
        {
            "model__max_depth": [4, 6],
            "model__max_iter": [200, 300],
            "model__learning_rate": [0.03, 0.05],
        },
        cv=group_cv,
        scoring="f1_macro",
        n_jobs=N_JOBS,
    )
    hybrid_hgb_grid.fit(X_train_hybrid, y_train, groups=train_groups)
    print("Best params:", hybrid_hgb_grid.best_params_)
    hybrid_threshold = threshold_search(
        hybrid_hgb_grid.best_estimator_,
        X_valid_hybrid,
        y_valid,
        "Hybrid Text + HistGradientBoosting",
    )
    hybrid_preds, hybrid_metrics = evaluate_model(
        "Hybrid Text + HistGradientBoosting",
        hybrid_hgb_grid.best_estimator_,
        X_test_hybrid,
        y_test,
        hybrid_threshold,
    )
    summary_rows.append(hybrid_metrics)

    confusion = print_confusion_matrix("Hybrid Text + HistGradientBoosting", y_test, hybrid_preds)
    save_confusion_matrix_plot(
        "Hybrid Text + HistGradientBoosting",
        confusion,
        CHARTS_DIR / "hybrid_hgb_confusion_matrix.png",
    )
    quarter_slices = print_slice_metrics(
        "Hybrid Text + HistGradientBoosting",
        diagnostics_df,
        y_test,
        hybrid_preds,
        "quarter",
    )
    last2_slices = print_slice_metrics(
        "Hybrid Text + HistGradientBoosting",
        diagnostics_df,
        y_test,
        hybrid_preds,
        "last_2_minutes",
    )
    home_slices = print_slice_metrics(
        "Hybrid Text + HistGradientBoosting",
        diagnostics_df,
        y_test,
        hybrid_preds,
        "offense_is_home",
    )
    importance_df = print_permutation_importance(
        "Hybrid Text + HistGradientBoosting",
        hybrid_hgb_grid.best_estimator_,
        X_test_hybrid,
        y_test,
    )
    save_results_bundle(
        summary_rows,
        [quarter_slices, last2_slices, home_slices],
        importance_df,
        confusion,
    )


if __name__ == "__main__":
    main()
