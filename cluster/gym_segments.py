"""Gym member k-means segments. Reproduces every number used in the slides."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
DATA = HERE / "gym_members.csv"
FEATURES = [
    "age",
    "visits_per_month",
    "avg_visit_minutes",
    "classes_per_month",
    "months_member",
    "monthly_extras_spend",
]
# member_id is an identifier, not a description of the member.
RANDOM_STATE = 42
K_CHOSEN = 4
CREDIT_SPEND_CUTOFF = 20


def credit_net(spend):
    """Finance's per-member monthly net if this member receives the $10 credit."""
    spend = np.asarray(spend, dtype=float)
    return np.where(spend >= CREDIT_SPEND_CUTOFF, -10.0, 5.0)


def main():
    df = pd.read_csv(DATA)
    print(f"rows {len(df)}  mean extras ${df['monthly_extras_spend'].mean():.2f}")
    print("left out: member_id")

    train, test = train_test_split(df, test_size=0.30, random_state=RANDOM_STATE)
    print(f"train {len(train)}  test {len(test)}")

    scaler = StandardScaler()
    Z_train = scaler.fit_transform(train[FEATURES])
    Z_test = scaler.transform(test[FEATURES])

    rows = []
    models = {}
    for k in range(1, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        km.fit(Z_train)
        train_spread = float(km.inertia_)
        dist = ((Z_test[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
        test_spread = float(dist.min(axis=1).sum())
        models[k] = km
        rows.append((k, train_spread, test_spread))
        print(f"k={k}  train spread {train_spread:,.1f}  test spread {test_spread:,.1f}")

    curve = pd.DataFrame(rows, columns=["k", "train", "test"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve["k"], curve["train"], marker="o", color="black", label="Training members")
    ax.plot(curve["k"], curve["test"], marker="o", color="gray", label="Test members")
    ax.axvline(K_CHOSEN, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Number of segments (k)")
    ax.set_ylabel("Within-group spread")
    ax.set_xticks(range(1, 9))
    ax.legend()
    ax.set_title("Elbow: spread vs number of segments")
    fig.tight_layout()
    fig.savefig(HERE / "elbow.png", dpi=150)
    print("wrote elbow.png")

    base_test = float(curve.loc[curve["k"] == 1, "test"].iloc[0])
    v1_test = float(curve.loc[curve["k"] == K_CHOSEN, "test"].iloc[0])
    print(f"\ntest spread k=1 ${''}{base_test:,.1f}")
    print(f"test spread k={K_CHOSEN} {v1_test:,.1f}")
    print(f"test spread falls {(1 - v1_test / base_test) * 100:.0f}% from k=1 to k={K_CHOSEN}")

    km = models[K_CHOSEN]
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=FEATURES)
    centers.index.name = "segment"
    print("\ncentroids (original units)")
    print(centers.round(1).to_string())

    # Credit goes to segments whose members mostly spend under $20.
    # Decided after seeing centroids: segments with center extras under $20.
    # Names describe the training centroid. Credit goes to segments whose
    # center spends under $20, where finance says the credit still pays.
    names = {
        0: "New and occasional",
        1: "Long-time regulars",
        2: "Solo regulars",
        3: "Class regulars",
    }
    credit_segments = set(centers.index[centers["monthly_extras_spend"] < CREDIT_SPEND_CUTOFF])
    print("segments that get the credit:", [names[i] for i in sorted(credit_segments)])

    def assign(frame):
        Z = scaler.transform(frame[FEATURES])
        return km.predict(Z)

    all_members = df.copy()
    all_members["segment"] = assign(all_members)
    means = all_members.groupby("segment")[FEATURES].mean().round(1)
    counts = all_members.groupby("segment").size()
    table = means.copy()
    table.insert(0, "member_count", counts)
    table["name"] = table.index.map(names)
    table["gets_credit"] = table.index.isin(credit_segments)
    table.to_csv(HERE / "segment_table.csv")
    print("\nall members by segment")
    print(table.to_string())

    baseline_net = float(credit_net(all_members["monthly_extras_spend"]).sum())
    gets_credit = all_members["segment"].isin(credit_segments)
    v1_net = float(credit_net(all_members.loc[gets_credit, "monthly_extras_spend"]).sum())
    print(f"\nbaseline monthly net (credit to all {len(all_members)}) ${baseline_net:,.0f}")
    print(f"v1 monthly net (credit to {int(gets_credit.sum())} members) ${v1_net:,.0f}")
    print(f"difference ${v1_net - baseline_net:,.0f} per month")

    new = pd.DataFrame(
        [
            {
                "age": 30,
                "visits_per_month": 4,
                "avg_visit_minutes": 42,
                "classes_per_month": 1,
                "months_member": 5,
                "monthly_extras_spend": 10,
            }
        ]
    )
    seg = int(assign(new)[0])
    offer = "credit" if seg in credit_segments else "no credit"
    print(f"\nnew member -> segment {seg}, {offer}")
    print(centers.round(1).loc[seg].to_string())


if __name__ == "__main__":
    main()
