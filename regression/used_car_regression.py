"""Used-car sale price regression. Reproduces every number used in the slides."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).parent
SALES = HERE / "used_car_sales.csv"
LOTS = HERE / "auction_lots.csv"
PREP_COST = 1500
RANDOM_STATE = 42

FEATURES = ["Mileage_K", "Age_Years", "Horsepower", "Accident_yes", "Num_Owners"]


def design(df):
    out = df.copy()
    out["Accident_yes"] = (out["Accident"] == "yes").astype(int)
    return out[FEATURES]


def main():
    sales = pd.read_csv(SALES)
    print(f"sales rows {len(sales)}  columns {list(sales.columns)}")

    train, test = train_test_split(sales, test_size=0.30, random_state=RANDOM_STATE)
    print(f"train {len(train)}  test {len(test)}")
    print(f"train mean sale price ${train['Sale_Price'].mean():,.0f}")

    y_train = train["Sale_Price"]
    X_train = sm.add_constant(design(train))
    full = sm.OLS(y_train, X_train).fit()
    print("\n=== full model ===")
    print(full.summary())

    # Num_Owners is the weak feature; compare a model without it.
    slim_cols = ["Mileage_K", "Age_Years", "Horsepower", "Accident_yes"]
    slim = sm.OLS(y_train, sm.add_constant(design(train)[slim_cols])).fit()
    print("\n=== without Num_Owners ===")
    print(slim.summary())
    print(f"\nadj R2 full {full.rsquared_adj:.4f}  without owners {slim.rsquared_adj:.4f}")
    print("hand over the model without Num_Owners: its p-value is 0.87 and adjusted R2 rises")

    model = slim
    baseline = float(y_train.mean())

    def predict(df):
        X = sm.add_constant(design(df), has_constant="add")
        X = X.reindex(columns=model.model.exog_names, fill_value=0)
        return model.predict(X)

    for name, frame in ("train", train), ("test", test):
        actual = frame["Sale_Price"]
        pred = predict(frame)
        base = np.full(len(frame), baseline)
        print(f"\n{name} baseline R2 {r2_score(actual, base):.4f}  RMSE ${mean_squared_error(actual, base) ** 0.5:,.0f}")
        print(f"{name} v1       R2 {r2_score(actual, pred):.4f}  RMSE ${mean_squared_error(actual, pred) ** 0.5:,.0f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    actual = test["Sale_Price"]
    pred = predict(test)
    ax.scatter(actual, pred, color="black", s=18)
    lo = min(actual.min(), pred.min())
    hi = max(actual.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], color="gray", linewidth=1)
    ax.set_xlabel("Actual sale price")
    ax.set_ylabel("Predicted sale price")
    ax.set_title("Test cars: predicted vs actual")
    fig.tight_layout()
    fig.savefig(HERE / "predicted_vs_actual.png", dpi=150)
    print("\nwrote predicted_vs_actual.png")

    lots = pd.read_csv(LOTS)
    lots["v1_pred"] = predict(lots)
    lots["baseline_pred"] = baseline

    def settle(mask, label):
        bought = lots.loc[mask]
        profit = bought["Later_Sale_Price"] - bought["Auction_Price"] - PREP_COST
        losers = int((profit < 0).sum())
        print(
            f"{label}: buy {len(bought)} of {len(lots)}, "
            f"money-losing {losers}, total profit ${profit.sum():,.0f}"
        )

    settle(lots["Auction_Price"] + PREP_COST < lots["baseline_pred"], "baseline")
    settle(lots["Auction_Price"] + PREP_COST < lots["v1_pred"], "v1")


if __name__ == "__main__":
    main()
