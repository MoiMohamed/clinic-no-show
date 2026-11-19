"""Feature and penalty bake-off for used-car prices.

Same 70/30 split as used_car_regression.py (random_state 42).
Writes regression_experiments.csv and regression_model_comparison.csv.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
PREP_COST = 1500
RANDOM_STATE = 42
FEATURES = ["Mileage_K", "Age_Years", "Horsepower", "Accident_yes", "Num_Owners"]

RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100]
LASSO_ALPHAS = [1, 10, 50, 100, 500, 1000]
ENET_ALPHAS = [1, 10, 50, 100, 500]
ENET_L1 = [0.25, 0.5, 0.75]


def design(df):
    out = df.copy()
    out["Accident_yes"] = (out["Accident"] == "yes").astype(int)
    return out[FEATURES]


def subsets():
    for k in range(1, len(FEATURES) + 1):
        for cols in combinations(FEATURES, k):
            yield list(cols)


def auction_stats(pred, lots):
    buy = lots["Auction_Price"].to_numpy() + PREP_COST < np.asarray(pred)
    bought = lots.loc[buy]
    if bought.empty:
        return 0, 0, 0.0
    profit = bought["Later_Sale_Price"] - bought["Auction_Price"] - PREP_COST
    return int(buy.sum()), int((profit < 0).sum()), float(profit.sum())


def fit_predict(kind, alpha, l1_ratio, X_tr, y_tr, X_te, X_lots):
    if kind == "ols":
        model = LinearRegression()
        model.fit(X_tr, y_tr)
        return model.predict(X_tr), model.predict(X_te), model.predict(X_lots)
    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(X_tr)
    Z_te = scaler.transform(X_te)
    Z_lots = scaler.transform(X_lots)
    if kind == "ridge":
        model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    elif kind == "lasso":
        model = Lasso(alpha=alpha, max_iter=20000, random_state=RANDOM_STATE)
    else:
        model = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, random_state=RANDOM_STATE
        )
    model.fit(Z_tr, y_tr)
    return model.predict(Z_tr), model.predict(Z_te), model.predict(Z_lots)


def specs():
    yield "ols", None, None
    for alpha in RIDGE_ALPHAS:
        yield "ridge", alpha, None
    for alpha in LASSO_ALPHAS:
        yield "lasso", alpha, None
    for alpha in ENET_ALPHAS:
        for l1 in ENET_L1:
            yield "elastic_net", alpha, l1


def main():
    sales = pd.read_csv(HERE / "used_car_sales.csv")
    lots = pd.read_csv(HERE / "auction_lots.csv")
    train, test = train_test_split(sales, test_size=0.30, random_state=RANDOM_STATE)
    y_tr = train["Sale_Price"].to_numpy()
    y_te = test["Sale_Price"].to_numpy()
    baseline = float(y_tr.mean())
    X_all_tr = design(train)
    X_all_te = design(test)
    X_all_lots = design(lots)

    base_pred = np.full(len(lots), baseline)
    b_buy, b_lose, b_profit = auction_stats(base_pred, lots)
    rows = [
        {
            "model": "baseline_mean",
            "regularization": "none",
            "alpha": np.nan,
            "l1_ratio": np.nan,
            "features": "(none)",
            "n_features": 0,
            "train_r2": 0.0,
            "train_rmse": float(mean_squared_error(y_tr, np.full(len(y_tr), baseline)) ** 0.5),
            "test_r2": float(r2_score(y_te, np.full(len(y_te), baseline))),
            "test_rmse": float(mean_squared_error(y_te, np.full(len(y_te), baseline)) ** 0.5),
            "auction_bought": b_buy,
            "auction_losers": b_lose,
            "auction_profit": b_profit,
        }
    ]

    for cols in subsets():
        X_tr = X_all_tr[cols]
        X_te = X_all_te[cols]
        X_lots = X_all_lots[cols]
        for kind, alpha, l1 in specs():
            pred_tr, pred_te, pred_lots = fit_predict(kind, alpha, l1, X_tr, y_tr, X_te, X_lots)
            bought, losers, profit = auction_stats(pred_lots, lots)
            rows.append(
                {
                    "model": kind,
                    "regularization": "none" if kind == "ols" else kind,
                    "alpha": alpha if alpha is not None else np.nan,
                    "l1_ratio": l1 if l1 is not None else np.nan,
                    "features": "+".join(cols),
                    "n_features": len(cols),
                    "train_r2": float(r2_score(y_tr, pred_tr)),
                    "train_rmse": float(mean_squared_error(y_tr, pred_tr) ** 0.5),
                    "test_r2": float(r2_score(y_te, pred_te)),
                    "test_rmse": float(mean_squared_error(y_te, pred_te) ** 0.5),
                    "auction_bought": bought,
                    "auction_losers": losers,
                    "auction_profit": profit,
                }
            )

    experiments = pd.DataFrame(rows)
    experiments.to_csv(HERE / "regression_experiments.csv", index=False)

    fitted = experiments[experiments["model"] != "baseline_mean"]
    pieces = [experiments[experiments["model"] == "baseline_mean"]]
    for _, group in fitted.groupby("model", sort=False):
        best_fit = group.sort_values(["test_rmse", "auction_profit"], ascending=[True, False]).head(1).copy()
        best_profit = group.sort_values(["auction_profit", "test_rmse"], ascending=[False, True]).head(1).copy()
        best_fit["picked_for"] = "lowest_test_rmse"
        best_profit["picked_for"] = "highest_auction_profit"
        pieces.append(best_fit)
        if best_profit.index[0] != best_fit.index[0]:
            pieces.append(best_profit)
        else:
            best_fit.loc[:, "picked_for"] = "lowest_test_rmse_and_highest_auction_profit"
    comparison = pd.concat(pieces, ignore_index=True)
    comparison.to_csv(HERE / "regression_model_comparison.csv", index=False)
    print(f"experiments {len(experiments)}")


if __name__ == "__main__":
    main()
