import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.metrics import accuracy_score, f1_score, classification_report

# =====================================================
# CONFIG
# =====================================================

DATA_PATH = Path("data/model/pbp_features.parquet")

# =====================================================
# HELPERS
# =====================================================

def threshold_search(model, X_test, y_test, name):

    probs = model.predict_proba(X_test)[:, 1]

    best_f1 = 0
    best_thresh = 0.5

    for t in np.arange(0.3, 0.71, 0.02):
        preds = (probs >= t).astype(int)
        score = f1_score(y_test, preds, average="macro")

        if score > best_f1:
            best_f1 = score
            best_thresh = t

    print(f"\nBest threshold ({name}):", best_thresh)
    print(f"Best Macro F1 ({name}):", best_f1)

    return (probs >= best_thresh).astype(int)


# =====================================================
# MAIN
# =====================================================

def main():

    print("Loading feature dataset...")
    df = pd.read_parquet(DATA_PATH)
    print("Shape:", df.shape)

    y = df["momentum_binary"]

    feature_cols = [
        c for c in df.columns
        if c not in [
            "game_id",
            "play_id",
            "momentum_class",
            "momentum_binary",
        ]
    ]

    X = df[feature_cols]

    # =====================================================
    # SPLIT BY GAME
    # =====================================================

    games = df["game_id"].unique()

    train_games, test_games = train_test_split(
        games, test_size=0.2, random_state=42
    )

    train_mask = df["game_id"].isin(train_games)
    test_mask = df["game_id"].isin(test_games)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    # =====================================================
    # COLUMN GROUPS
    # =====================================================

    binary_cols = [c for c in X.columns if X[c].dropna().isin([0, 1]).all()]
    numeric_cols = [c for c in X.columns if c not in binary_cols]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("bin", "passthrough", binary_cols),
    ])

    # =====================================================
    # LOGISTIC REGRESSION
    # =====================================================

    lr_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    lr_grid = GridSearchCV(
        lr_pipe,
        {"model__C": [0.01, 0.1, 1, 10]},
        cv=3,
        scoring="f1_macro",
        n_jobs=-1
    )

    lr_grid.fit(X_train, y_train)

    best_lr = lr_grid.best_estimator_
    lr_preds = best_lr.predict(X_test)

    print("\n===== LOGISTIC REGRESSION =====")
    print("Best params:", lr_grid.best_params_)
    print("Accuracy:", accuracy_score(y_test, lr_preds))
    print("Macro F1:", f1_score(y_test, lr_preds, average="macro"))
    print(classification_report(y_test, lr_preds, zero_division=0))

    # =====================================================
    # RANDOM FOREST
    # =====================================================

    rf_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            random_state=42,
            class_weight={0: 2, 1: 1}
        ))
    ])

    rf_grid = GridSearchCV(
        rf_pipe,
        {
            "model__n_estimators": [200, 400],
            "model__max_depth": [5, 10, None],
            "model__min_samples_leaf": [1, 3, 5],
        },
        cv=3,
        scoring="f1_macro",
        n_jobs=-1
    )

    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_

    rf_preds = threshold_search(best_rf, X_test, y_test, "RF")

    print("\n===== RANDOM FOREST =====")
    print("Best params:", rf_grid.best_params_)
    print("Accuracy:", accuracy_score(y_test, rf_preds))
    print("Macro F1:", f1_score(y_test, rf_preds, average="macro"))
    print(classification_report(y_test, rf_preds, zero_division=0))

    # =====================================================
    # HIST GRADIENT BOOSTING
    # =====================================================

    hgb_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            class_weight={0: 2, 1: 1},
            random_state=42
        ))
    ])

    hgb_grid = GridSearchCV(
        hgb_pipe,
        {
            "model__max_depth": [4, 6, 8],
            "model__learning_rate": [0.03, 0.05, 0.07],
            "model__max_iter": [200, 300]
        },
        cv=3,
        scoring="f1_macro",
        n_jobs=-1
    )

    hgb_grid.fit(X_train, y_train)
    best_hgb = hgb_grid.best_estimator_

    hgb_preds = threshold_search(best_hgb, X_test, y_test, "HGB")

    print("\n===== HIST GRADIENT BOOSTING =====")
    print("Best params:", hgb_grid.best_params_)
    print("Accuracy:", accuracy_score(y_test, hgb_preds))
    print("Macro F1:", f1_score(y_test, hgb_preds, average="macro"))
    print(classification_report(y_test, hgb_preds, zero_division=0))


# =====================================================
if __name__ == "__main__":
    main()