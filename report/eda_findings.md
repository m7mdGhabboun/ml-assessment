# Phase 2 — EDA findings

Figures: `report/figures/` (regenerate with `python src/eda.py`).

## 1. Target
- `posted_rate`: $57 – $25,533, median $2,031, skew 1.9. log(rate) is near-symmetric (skew −0.5) → model log(rate) or rate-per-mile.
- Rate per mile (rpm) median $2.15; 98.6% of loads fall between $1.60 and $3.60.

## 2. Drivers
- Distance dominates: log(rate) ~ log(distance) + equipment gives R² = 0.994 on clean rows.
- rpm falls with distance (fixed costs spread over more miles): $2.75 (<250 mi) → $1.91 (>2,000 mi).
- Equipment premium (median rpm): Dry Van $2.05, Flatbed $2.22, Reefer $2.31.
- Weight: small positive effect (~$2.02 → $2.21 rpm from light to heavy).
- Day of week: negligible.

## 3. quote_signal — regime-switching, unusable for Nov/Dec
| Months | Relationship to rpm |
|---|---|
| Jan–Mar, Jun, Sep | quote ≈ rpm (r = +0.997) |
| Apr, May, Jul, Oct | quote ≈ 4.15 − rpm (r = −0.99) |
| Aug | no relationship (r ≈ 0) |

Label-free fingerprint: corr(quote, log distance) is −0.8 / +0.8 / 0 in the three regimes. Nov and Dec score +0.01 → noise regime.
**Decision: drop `quote_signal`.** A model trained on it would learn a relationship that does not exist in the prediction period. It is also absent from the December chart inputs.

## 4. market_index — date-level market series
- Varies almost entirely by date (within-day std 0.025); no dependence on city or equipment.
- Smooth trend with step changes (mid-Apr up, late-Jul down, early-Oct up). Validation period: ~0.92–0.95 (soft market).
- Effect: elasticity ≈ 0.14 on log(rate), monotone (≈ −2.5% to +4% across its range).
- Available in validation.csv for every Nov–Dec date → **December chart can use the daily mean market_index from validation.csv.**

## 5. Time
- Training labels: Jan 1 – Oct 31 2025. Predictions: Nov 1 – Dec 31 2025 → forecasting, not interpolation.
- rpm drifts: ~$2.03 (Jan) → ~$2.30 (late Jun) → ~$2.12 (Aug) → ~$2.17 (Oct). Month effects add a little beyond market_index.

## 6. Train vs validation
- distance, weight and equipment mix match closely.
- market_index lower in validation (median 0.92 vs 1.06).
- 736 of 4,214 validation lanes (1,461 rows, 12%) never appear in training → lane-ID features alone won't generalise; use city-level + distance features.

## 7. Data quality
| Issue | Train | Validation | Plan |
|---|---|---|---|
| Implausible rpm (<$1.20 or >$4) | 675 rows (1.4%): low ≈ 0.2–0.4× typical, high ≈ 2.5–5× | n/a (no labels) | drop from training; separated by clear gaps |
| Negative weight | 292 | 145 | sign error (abs values look normal) → take abs |
| Missing weight | 300 | 165 | impute (median by equipment) |
| Missing market_index | 374 | 249 | impute with that date's mean |
| Duplicates, city spelling, coordinates, dates | none found | none found | — |
| Lat/lon | consistent per city but synthetic (not real geography) | | use as relative position only |
