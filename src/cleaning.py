"""Data cleaning, shared by training and prediction.

Design rule: anything learned from data (e.g. the median weight used for
imputation) is learned from the TRAINING rows only and then applied to every
other dataset. Learning it from validation rows would leak information about
the data we are evaluated on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Plausible rate-per-mile band. EDA showed real loads sit between ~$1.60 and
# ~$3.60 per mile, with clear gaps before the injected errors start
# (0.2-0.4x typical below, 2.5-5x typical above).
RPM_LOW, RPM_HIGH = 1.2, 4.0

# quote_signal switches meaning by month and is pure noise in Nov/Dec
# (see report/eda_findings.md), so it is never used as a feature.
DROP_COLUMNS = ["quote_signal"]


@dataclass
class Cleaner:
    """Fit on training data, then transform any dataset the same way."""

    weight_median_by_equipment: dict[str, float] = field(default_factory=dict)
    weight_median_overall: float = np.nan
    report: dict[str, int] = field(default_factory=dict)

    # ---- fitting -----------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "Cleaner":
        weight = train["weight"].abs()  # fit on sign-corrected values
        self.weight_median_by_equipment = weight.groupby(train["equipment"]).median().to_dict()
        self.weight_median_overall = float(weight.median())
        return self

    # ---- transforming ------------------------------------------------------
    def transform(self, df: pd.DataFrame, market_index_by_date: pd.Series) -> pd.DataFrame:
        out = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns]).copy()
        out["date"] = pd.to_datetime(out["date"])

        # 1. Negative weights are sign errors: their absolute values are normal.
        self.report["weight_negative_fixed"] = int((out["weight"] < 0).sum())
        out["weight"] = out["weight"].abs()

        # 2. Missing weight -> median for that equipment type (learned on train).
        #    Keep a flag so the model can learn if "missing" itself means something.
        missing_w = out["weight"].isna()
        self.report["weight_imputed"] = int(missing_w.sum())
        out["weight_missing"] = missing_w.astype(int)
        fill = out["equipment"].map(self.weight_median_by_equipment).fillna(self.weight_median_overall)
        out.loc[missing_w, "weight"] = fill[missing_w]

        # 3. Missing market_index -> that day's mean. market_index is a market-wide
        #    daily series (within-day spread is tiny), so the day mean is a near-exact fill.
        missing_mi = out["market_index"].isna()
        self.report["market_index_imputed"] = int(missing_mi.sum())
        out["market_index_missing"] = missing_mi.astype(int)
        out.loc[missing_mi, "market_index"] = out.loc[missing_mi, "date"].map(market_index_by_date)
        return out


def daily_market_index(*frames: pd.DataFrame) -> pd.Series:
    """Mean market_index per date across all given feature tables.

    Uses only feature columns (never the target), so pooling train and
    validation rows here is not leakage.
    """
    both = pd.concat([f[["date", "market_index"]] for f in frames])
    both["date"] = pd.to_datetime(both["date"])
    return both.groupby("date")["market_index"].mean()


def rate_outlier_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows whose rate per mile is outside the plausible band (needs labels)."""
    rpm = df["posted_rate"] / df["distance"]
    return ~rpm.between(RPM_LOW, RPM_HIGH)
