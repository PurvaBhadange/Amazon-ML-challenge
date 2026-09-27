from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, fbeta_score

from src.features import feature_columns


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
MODELS = ROOT / "models"


def load_data():
    train = pd.read_csv(
        OUTPUT / "training_pairs.tsv",
        sep="\t",
    )

    validation = pd.read_csv(
        OUTPUT / "validation_pairs.tsv",
        sep="\t",
    )

    columns = feature_columns()

    X_train = train[columns]
    y_train = train["label"].astype(int)

    X_valid = validation[columns]
    y_valid = validation["label"].astype(int)

    return (
        train,
        validation,
        X_train,
        y_train,
        X_valid,
        y_valid,
    )


def evaluate_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:

    rows = []

    for threshold in np.arange(
        0.10,
        0.951,
        0.025,
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f05 = fbeta_score(
            y_true,
            predictions,
            beta=0.5,
            zero_division=0,
        )

        rows.append(
            {
                "threshold": round(
                    float(threshold),
                    3,
                ),
                "precision": precision,
                "recall": recall,
                "f0.5": f05,
                "predicted_positive": int(
                    predictions.sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def evaluate_entity_level(
    validation: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:

    work = validation[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "label",
        ]
    ].copy()

    work["probability"] = probabilities

    predictions = []

    for entity_id, group in work.groupby(
        "source1_entity_id",
        sort=False,
    ):

        selected = group[
            group["probability"] >= threshold
        ]

        predicted_ids = set(
            selected[
                "candidate_entity_id"
            ]
        )

        true_ids = set(
            group.loc[
                group["label"] == 1,
                "candidate_entity_id",
            ]
        )

        predictions.append(
            (
                entity_id,
                predicted_ids,
                true_ids,
            )
        )

    entity_scores = []

    for _, predicted, true in predictions:

        if not predicted and not true:
            score = 1.0

        elif not predicted and true:
            score = 0.0

        else:
            tp = len(predicted & true)
            fp = len(predicted - true)
            fn = len(true - predicted)

            precision = (
                tp / (tp + fp)
                if tp + fp
                else 0.0
            )

            recall = (
                tp / (tp + fn)
                if tp + fn
                else 0.0
            )

            if precision == 0 and recall == 0:
                score = 0.0
            else:
                score = (
                    1.25
                    * precision
                    * recall
                    / (
                        0.25 * precision
                        + recall
                    )
                )

        entity_scores.append(score)

    return {
        "macro_f0.5": float(
            np.mean(entity_scores)
        ),
        "entities": len(entity_scores),
    }


def train_logistic(
    X_train,
    y_train,
):

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def train_hist_gradient_boosting(
    X_train,
    y_train,
):

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def evaluate_model(
    name,
    model,
    validation,
    X_valid,
    y_valid,
):

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    threshold_results = evaluate_thresholds(
        y_valid.to_numpy(),
        probabilities,
    )

    best = threshold_results.loc[
        threshold_results["f0.5"].idxmax()
    ]

    entity_results = evaluate_entity_level(
        validation,
        probabilities,
        float(best["threshold"]),
    )

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"Best threshold : "
        f"{best['threshold']:.3f}"
    )

    print(
        f"Pair precision : "
        f"{best['precision']:.6f}"
    )

    print(
        f"Pair recall    : "
        f"{best['recall']:.6f}"
    )

    print(
        f"Pair F0.5      : "
        f"{best['f0.5']:.6f}"
    )

    print(
        f"Predicted positives : "
        f"{int(best['predicted_positive']):,}"
    )

    print(
        f"Entity macro F0.5 : "
        f"{entity_results['macro_f0.5']:.6f}"
    )

    return {
        "model": model,
        "threshold": float(
            best["threshold"]
        ),
        "pair_f0.5": float(
            best["f0.5"]
        ),
        "entity_f0.5": float(
            entity_results["macro_f0.5"]
        ),
        "threshold_results": threshold_results,
    }


def main():

    MODELS.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        train,
        validation,
        X_train,
        y_train,
        X_valid,
        y_valid,
    ) = load_data()

    print(
        f"Training rows   : {len(train):,}"
    )

    print(
        f"Validation rows : {len(validation):,}"
    )

    print(
        f"Features        : {len(feature_columns())}"
    )

    print()
    print("Training Logistic Regression...")

    logistic = train_logistic(
        X_train,
        y_train,
    )

    logistic_result = evaluate_model(
        "LOGISTIC REGRESSION",
        logistic,
        validation,
        X_valid,
        y_valid,
    )

    print()
    print(
        "Training HistGradientBoosting..."
    )

    hist = train_hist_gradient_boosting(
        X_train,
        y_train,
    )

    hist_result = evaluate_model(
        "HISTGRADIENTBOOSTING",
        hist,
        validation,
        X_valid,
        y_valid,
    )

    results = [
        logistic_result,
        hist_result,
    ]

    best = max(
        results,
        key=lambda x: x["entity_f0.5"],
    )

    joblib.dump(
        best["model"],
        MODELS / "matching_model.joblib",
    )

    metadata = pd.DataFrame(
        [
            {
                "model": (
                    "LogisticRegression"
                    if best["model"] is logistic
                    else "HistGradientBoosting"
                ),
                "threshold": best["threshold"],
                "pair_f0.5": best["pair_f0.5"],
                "entity_macro_f0.5": best["entity_f0.5"],
            }
        ]
    )

    metadata.to_csv(
        MODELS / "model_metadata.tsv",
        sep="\t",
        index=False,
    )

    print()
    print("=" * 80)
    print("MODEL TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Selected threshold : "
        f"{best['threshold']:.3f}"
    )

    print(
        f"Entity macro F0.5 : "
        f"{best['entity_f0.5']:.6f}"
    )

    print(
        f"Saved model       : "
        f"{MODELS / 'matching_model.joblib'}"
    )


if __name__ == "__main__":
    main()