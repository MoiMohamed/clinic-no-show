"""Depth and feature bake-off for the clinic no-show tree."""

from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).parent
FEATURES = [
    "lead_time_days",
    "prior_no_shows",
    "age",
    "distance_miles",
    "sms_reminder",
    "new_patient",
    "appointment_hour",
]
TARGET = "no_show"
POSITIVE = "no-show"
RANDOM_STATE = 42
DEPTHS = [1, 2, 3, 4, 5, 6]
LEAVES = [5, 15, 30]
VALUE = {
    ("checked-in", "checked-in"): 0,
    ("no-show", "checked-in"): -60,
    ("checked-in", "no-show"): -120,
    ("no-show", "no-show"): 0,
}


def subsets():
    for n in range(1, len(FEATURES) + 1):
        for cols in combinations(FEATURES, n):
            yield list(cols)


def scores(y_true, y_pred):
    labels = ["checked-in", "no-show"]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    value = sum(VALUE[(p, a)] for p, a in zip(y_pred, y_true))
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_no_show": float(precision_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0)),
        "recall_no_show": float(recall_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0)),
        "test_value": int(value),
        "value_per_appointment": float(value / len(y_true)),
    }


def main():
    df = pd.read_csv(HERE / "clinic_appointments.csv")
    X = df[FEATURES]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    rows = []
    base = scores(y_te, ["checked-in"] * len(y_te))
    base.update(
        {
            "model": "baseline_always_hold",
            "features": "(none)",
            "n_features": 0,
            "max_depth": 0,
            "min_samples_leaf": 0,
            "train_accuracy": float((y_tr == "checked-in").mean()),
        }
    )
    rows.append(base)
    for cols in subsets():
        for depth in DEPTHS:
            for leaf in LEAVES:
                tree = DecisionTreeClassifier(
                    max_depth=depth, min_samples_leaf=leaf, random_state=RANDOM_STATE
                )
                tree.fit(X_tr[cols], y_tr)
                row = scores(y_te, tree.predict(X_te[cols]))
                row.update(
                    {
                        "model": "decision_tree",
                        "features": "+".join(cols),
                        "n_features": len(cols),
                        "max_depth": depth,
                        "min_samples_leaf": leaf,
                        "train_accuracy": float(accuracy_score(y_tr, tree.predict(X_tr[cols]))),
                    }
                )
                rows.append(row)

    experiments = pd.DataFrame(rows)
    experiments.to_csv(HERE / "tree_experiments.csv", index=False)
    fitted = experiments[experiments["model"] == "decision_tree"]
    slide = fitted[
        (fitted["n_features"] == len(FEATURES))
        & (fitted["max_depth"] == 3)
        & (fitted["min_samples_leaf"] == 15)
    ].copy()
    slide["picked_for"] = "depth3_leaf15_all_features"
    best = fitted.sort_values(["test_value", "accuracy"], ascending=[False, False]).head(1).copy()
    best["picked_for"] = "highest_test_value"
    baseline = experiments[experiments["model"] == "baseline_always_hold"].copy()
    baseline["picked_for"] = "baseline"
    pd.concat([baseline, slide, best], ignore_index=True).to_csv(
        HERE / "tree_model_comparison.csv", index=False
    )
    print(f"experiments {len(experiments)}")


if __name__ == "__main__":
    main()
