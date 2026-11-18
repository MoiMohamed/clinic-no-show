"""Clinic no-show decision tree. Reproduces every number used in the slides."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

DATA = Path(__file__).with_name("clinic_appointments.csv")
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
POSITIVE = "no-show"  # class we care about
RANDOM_STATE = 42
MAX_DEPTH = 3

# Cost table: value of (prediction, actual). Prediction is the clinic action.
# show / checked-in -> hold. no-show -> overbook.
VALUE = {
    ("checked-in", "checked-in"): 0,  # hold, patient came
    ("no-show", "checked-in"): -60,  # overbook, patient came
    ("checked-in", "no-show"): -120,  # hold, empty slot
    ("no-show", "no-show"): 0,  # overbook, second patient fills slot
}


def metrics(y_true, y_pred, label):
    # Order: negative class checked-in, positive class no-show
    labels = ["checked-in", "no-show"]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    print(f"\n=== {label} ===")
    print(f"TN (hold, came) {tn}  FP (overbook, came) {fp}")
    print(f"FN (hold, missed) {fn}  TP (overbook, missed) {tp}")
    print(f"accuracy  {accuracy_score(y_true, y_pred):.4f}")
    print(
        f"precision (no-show) {precision_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0):.4f}"
    )
    print(
        f"recall (no-show)    {recall_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0):.4f}"
    )
    return tn, fp, fn, tp


def total_value(y_true, y_pred):
    return sum(VALUE[(p, a)] for p, a in zip(y_pred, y_true))


def main():
    df = pd.read_csv(DATA)
    print(f"rows {len(df)}  columns {list(df.columns)}")
    print(df[TARGET].value_counts())
    print(f"no-show rate {df[TARGET].eq(POSITIVE).mean():.4f}")

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\ntrain {len(X_train)}  no-show rate {y_train.eq(POSITIVE).mean():.4f}")
    print(f"test  {len(X_test)}  no-show rate {y_test.eq(POSITIVE).mean():.4f}")

    tree = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_leaf=15,
        random_state=RANDOM_STATE,
    )
    tree.fit(X_train, y_train)
    print("\n" + export_text(tree, feature_names=FEATURES, decimals=2))
    used = [
        FEATURES[i]
        for i in tree.tree_.feature
        if i >= 0
    ]
    print("splits on:", list(dict.fromkeys(used)))

    pred_train = tree.predict(X_train)
    pred_test = tree.predict(X_test)
    print(f"\nv1 train accuracy {accuracy_score(y_train, pred_train):.4f}")
    print(f"v1 test  accuracy {accuracy_score(y_test, pred_test):.4f}")

    baseline = ["checked-in"] * len(y_test)
    metrics(y_test, baseline, "baseline on test (always hold)")
    metrics(y_test, pred_test, "v1 on test")

    base_value = total_value(y_test, baseline)
    v1_value = total_value(y_test, pred_test)
    n = len(y_test)
    print(f"\nbaseline total ${base_value}  per appointment ${base_value / n:.2f}")
    print(f"v1       total ${v1_value}  per appointment ${v1_value / n:.2f}")
    print(f"gain per appointment ${(v1_value - base_value) / n:.2f}")

    upcoming = pd.DataFrame(
        [
            {
                "id": "D",
                "lead_time_days": 21,
                "prior_no_shows": 1,
                "age": 27,
                "distance_miles": 9.5,
                "sms_reminder": 0,
                "new_patient": 1,
                "appointment_hour": 15,
            },
            {
                "id": "E",
                "lead_time_days": 3,
                "prior_no_shows": 0,
                "age": 61,
                "distance_miles": 2.0,
                "sms_reminder": 1,
                "new_patient": 0,
                "appointment_hour": 9,
            },
        ]
    )
    leaves = tree.apply(upcoming[FEATURES])
    preds = tree.predict(upcoming[FEATURES])
    for i, row in upcoming.iterrows():
        action = "overbook" if preds[i] == POSITIVE else "hold"
        print(f"{row['id']}: leaf {leaves[i]}  predict {preds[i]}  -> {action}")

    # Assumed clinic volume for the yearly translation.
    slots_per_year = 400 * 12  # this file is one month of a similar clinic
    gain = (v1_value - base_value) / n
    print(f"\nassumed volume {slots_per_year} appointments/year")
    print(f"yearly value of switching ${gain * slots_per_year:,.0f}")


if __name__ == "__main__":
    main()
