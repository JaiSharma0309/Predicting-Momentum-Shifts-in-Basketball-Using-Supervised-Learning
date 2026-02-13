import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


DATA_PATH = Path("data/model/pbp_features.parquet")


def main():

    print("Loading feature dataset...")
    df = pd.read_parquet(DATA_PATH)
    print("Shape:", df.shape)

    # ======================
    # FEATURE / TARGET SPLIT
    # ======================

    y = df["momentum_class"]

    feature_cols = [c for c in df.columns 
                    if c not in ["game_id", "play_id", "momentum_class"]]


    X = df[feature_cols]
    y = df["momentum_class"]

    # ======================
    # SPLIT BY GAME (NO LEAKAGE)
    # ======================

    unique_games = df["game_id"].unique()

    train_games, test_games = train_test_split(
        unique_games,
        test_size=0.2,
        random_state=42
    )

    train_mask = df["game_id"].isin(train_games)
    test_mask = df["game_id"].isin(test_games)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    # ======================
    # MODEL 1: LOGISTIC REGRESSION
    # ======================

    log_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)

    print("\nLogistic Regression Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))

    # ======================
    # MODEL 2: RANDOM FOREST
    # ======================

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("\nRandom Forest Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))



if __name__ == "__main__":
    main()
