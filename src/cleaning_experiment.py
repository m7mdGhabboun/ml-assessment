"""Measure the effect of each cleaning step (ablation study).

Run:  python src/cleaning_experiment.py

Same model and features every time; only the cleaning changes, so any change
in error is caused by the cleaning step. Split is by time: train Jan-Aug,
evaluate Sep-Oct (mirrors predicting Nov-Dec from Jan-Oct).
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from cleaning import Cleaner, daily_market_index, rate_outlier_mask

DATA = Path(__file__).resolve().parents[1] / "data"
SPLIT = pd.Timestamp("2025-09-01")
BASE_FEATURES = ["log_distance", "equipment", "weight", "market_index",
                 "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "month"]


def features(df: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    X = pd.DataFrame({
        "log_distance": np.log(df["distance"]),
        "equipment": df["equipment"].astype("category"),
        "weight": df["weight"], "market_index": df["market_index"],
        "pickup_lat": df["pickup_lat"], "pickup_lon": df["pickup_lon"],
        "delivery_lat": df["delivery_lat"], "delivery_lon": df["delivery_lon"],
        "month": pd.to_datetime(df["date"]).dt.month,
    })
    for c in extra:
        X[c] = df[c]
    return X


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, extra: list[str]) -> np.ndarray:
    model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                              min_child_samples=40, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.9, random_state=0, verbose=-1)
    # Predict log(rate): errors become relative (a $50 miss on a $300 load
    # matters more than on a $5,000 load) and the skewed target becomes symmetric.
    model.fit(features(train, extra), np.log(train["posted_rate"]))
    return np.exp(model.predict(features(test, extra)))


def score(test: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    y = test["posted_rate"].to_numpy()
    typical = ~rate_outlier_mask(test).to_numpy()
    err = np.abs(pred - y)
    return {
        "MAE all": err.mean(),
        "MAE typical": err[typical].mean(),
        "MAPE typical %": 100 * (err[typical] / y[typical]).mean(),
    }


def main() -> None:
    raw = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    valid = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    tr_raw, te_raw = raw[raw["date"] < SPLIT], raw[raw["date"] >= SPLIT]

    cleaner = Cleaner().fit(tr_raw)                       # learned on train only
    mi_by_date = daily_market_index(raw, valid)
    tr_fix = cleaner.transform(tr_raw, mi_by_date).assign(posted_rate=tr_raw["posted_rate"].values)
    te_fix = cleaner.transform(te_raw, mi_by_date).assign(posted_rate=te_raw["posted_rate"].values)
    tr_clean = tr_fix[~rate_outlier_mask(tr_fix)]

    experiments = {
        "A. raw (no cleaning)":                  (tr_raw, te_raw, []),
        "B. + fix weight sign, impute nulls":   (tr_fix, te_fix, []),
        "C. + drop rate outliers from train":   (tr_clean, te_fix, []),
        # What-if: put quote_signal back (the cleaner drops it) to see what it does
        "D. C + quote_signal (what if?)":       (
            tr_clean.assign(quote_signal=tr_raw.loc[tr_clean.index, "quote_signal"].values),
            te_fix.assign(quote_signal=te_raw["quote_signal"].values),
            ["quote_signal"]),
    }

    rows = {name: score(te, fit_predict(tr, te, extra)) for name, (tr, te, extra) in experiments.items()}
    print(f"Train rows: {len(tr_raw):,} raw / {len(tr_clean):,} after dropping outliers")
    print(f"Eval rows (Sep-Oct): {len(te_raw):,}, of which {int(rate_outlier_mask(te_raw).sum())} are rate outliers")
    print("Cleaning report (eval fold):", cleaner.report)
    print(pd.DataFrame(rows).T.round(2).to_string())

    # Per-month view of D vs C: does quote_signal help or hurt, month by month?
    print("\nMAE typical by eval month:")
    for name in ["C. + drop rate outliers from train", "D. C + quote_signal (what if?)"]:
        tr, te, extra = experiments[name]
        pred = fit_predict(tr, te, extra)
        m = te["date"].dt.month.to_numpy(); ok = ~rate_outlier_mask(te).to_numpy()
        e = np.abs(pred - te["posted_rate"].to_numpy())
        print(f"  {name[:2]}", {mm: float(round(e[(m == mm) & ok].mean(), 1)) for mm in (9, 10)})

    # Sep (copy regime) and Oct (mirror regime) flatter quote_signal. Nov/Dec are in the
    # NOISE regime, like August, so the fair test is: train Jan-Jul, evaluate on August.
    print("\nNoise-regime test (train Jan-Jul, evaluate Aug):")
    tr_a, te_a = raw[raw["date"] < "2025-08-01"], raw[raw["date"].dt.month == 8]
    c = Cleaner().fit(tr_a)
    keep = ["posted_rate", "quote_signal"]
    tr_a = c.transform(tr_a, mi_by_date).assign(**{k: tr_a[k].values for k in keep})
    te_a = c.transform(te_a, mi_by_date).assign(**{k: te_a[k].values for k in keep})
    tr_a = tr_a[~rate_outlier_mask(tr_a)]
    for label, extra in [("without quote_signal", []), ("with quote_signal", ["quote_signal"])]:
        s = score(te_a, fit_predict(tr_a, te_a, extra))
        print(f"  {label:22s}", {k: float(round(v, 2)) for k, v in s.items()})


if __name__ == "__main__":
    main()
