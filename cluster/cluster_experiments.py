"""k and feature bake-off for gym segments. Same split as gym_segments.py."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
FEATURES = [
    "age",
    "visits_per_month",
    "avg_visit_minutes",
    "classes_per_month",
    "months_member",
    "monthly_extras_spend",
]
RANDOM_STATE = 42
CREDIT_CUTOFF = 20


def subsets():
    for n in range(1, len(FEATURES) + 1):
        for cols in combinations(FEATURES, n):
            yield list(cols)


def credit_net(spend):
    spend = np.asarray(spend, dtype=float)
    return np.where(spend >= CREDIT_CUTOFF, -10.0, 5.0)


def monthly_net(train, all_members, labels_train, labels_all):
    train = train.copy()
    train["segment"] = labels_train
    means = train.groupby("segment")["monthly_extras_spend"].mean()
    credit_segments = set(means.index[means < CREDIT_CUTOFF])
    gets = np.isin(labels_all, list(credit_segments))
    net = float(credit_net(all_members.loc[gets, "monthly_extras_spend"]).sum()) if gets.any() else 0.0
    return net, len(credit_segments), int(gets.sum())


def main():
    df = pd.read_csv(HERE / "gym_members.csv")
    train, test = train_test_split(df, test_size=0.30, random_state=RANDOM_STATE)
    rows = []
    for cols in subsets():
        scaler = StandardScaler()
        Z_tr = scaler.fit_transform(train[cols])
        Z_te = scaler.transform(test[cols])
        for k in range(1, 9):
            km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
            labels_tr = km.fit_predict(Z_tr)
            dist = ((Z_te[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
            labels_all = km.predict(scaler.transform(df[cols]))
            if k == 1:
                net = float(credit_net(df["monthly_extras_spend"]).sum())
                n_credit_seg, n_credit = 1, len(df)
            else:
                net, n_credit_seg, n_credit = monthly_net(train, df, labels_tr, labels_all)
            rows.append(
                {
                    "features": "+".join(cols),
                    "n_features": len(cols),
                    "k": k,
                    "train_spread": float(km.inertia_),
                    "test_spread": float(dist.min(axis=1).sum()),
                    "credit_segments": n_credit_seg,
                    "members_getting_credit": n_credit,
                    "monthly_net": net,
                }
            )

    experiments = pd.DataFrame(rows)
    experiments.to_csv(HERE / "cluster_experiments.csv", index=False)
    full = experiments[experiments["n_features"] == len(FEATURES)].copy()
    base = full[full["k"] == 1].copy()
    v1 = full[full["k"] == 4].copy()
    base["model"] = "baseline_k1"
    v1["model"] = "v1_k4"
    comparison = pd.concat([base, v1], ignore_index=True)
    comparison.to_csv(HERE / "cluster_model_comparison.csv", index=False)
    print(f"experiments {len(experiments)}")


if __name__ == "__main__":
    main()
