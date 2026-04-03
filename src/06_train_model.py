import warnings
from pathlib import Path

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


def evaluate_model(name, model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    evaluate_predictions(name, y_test, preds)
    return preds


def print_confusion_matrix(name, y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    print(f"\n{name} Confusion Matrix:")
    print("rows=true, cols=pred")
    print(matrix)


def print_slice_metrics(name, eval_df, y_true, y_pred, slice_col):
    print(f"\n{name} by {slice_col}:")
    temp = eval_df[[slice_col]].copy()
    temp["y_true"] = y_true.values
    temp["y_pred"] = y_pred

    for slice_value, group in temp.groupby(slice_col, dropna=False):
        print(
            f"{slice_col}={slice_value} | n={len(group)} | "
            f"accuracy={accuracy_score(group['y_true'], group['y_pred']):.4f} | "
            f"macro_f1={f1_score(group['y_true'], group['y_pred'], average='macro'):.4f}"
        )


def print_permutation_importance(name, model, X_test, y_test):
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

    print(f"\n{name} Top Permutation Importances:")
    print(importance_df.head(12).to_string(index=False))


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

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train_num, y_train)
    evaluate_predictions(
        "Majority Baseline",
        y_test,
        baseline.predict(X_test_num),
    )

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
    evaluate_model(
        "Numeric Logistic",
        logistic_grid.best_estimator_,
        X_test_num,
        y_test,
        logistic_threshold,
    )

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
    evaluate_model(
        "Numeric HistGradientBoosting",
        hgb_grid.best_estimator_,
        X_test_num,
        y_test,
        hgb_threshold,
    )

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
    hybrid_preds = evaluate_model(
        "Hybrid Text + HistGradientBoosting",
        hybrid_hgb_grid.best_estimator_,
        X_test_hybrid,
        y_test,
        hybrid_threshold,
    )

    print_confusion_matrix("Hybrid Text + HistGradientBoosting", y_test, hybrid_preds)
    print_slice_metrics(
        "Hybrid Text + HistGradientBoosting",
        diagnostics_df,
        y_test,
        hybrid_preds,
        "quarter",
    )
    print_slice_metrics(
        "Hybrid Text + HistGradientBoosting",
        diagnostics_df,
        y_test,
        hybrid_preds,
        "last_2_minutes",
    )
    print_slice_metrics(
        "Hybrid Text + HistGradientBoosting",
        diagnostics_df,
        y_test,
        hybrid_preds,
        "offense_is_home",
    )
    print_permutation_importance(
        "Hybrid Text + HistGradientBoosting",
        hybrid_hgb_grid.best_estimator_,
        X_test_hybrid,
        y_test,
    )


if __name__ == "__main__":
    main()
