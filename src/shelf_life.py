"""Shelf-life (days remaining) estimation — paper §13 reproduction.

Paper Equation (2), 5-stage:  Days Left = α × (Ripening Index − 5)
Table 4 coefficients (OLS, 95% CI):
    T10  α = −4.390 (R²=0.953)
    T20  α = −2.116 (R²=0.963)
    Tamb α = −1.929 (R²=0.959)   ← the group name in the public metadata is 'Tam'
Paper Equation (4):  Loss = |Estimated Shelf Life − Actual Shelf Life|

Warning: no separate days_left regression CNN is trained (§13). The CNN's 5-stage
   prediction is plugged into the per-storage-group linear formula above to compute
   shelf-life.
   Storage Group is not a CNN input — it is only used to select 'which α to use' (keeps §2.3).

Actual Days Left definition (paper): the number of days from the shooting time until the
   sample first reaches stage 5 (end of shelf life) in the 5-stage index.
  - Handling of samples that never reach stage 5 (right-censored) and of observations past
    stage 5 (negative values) is not explicitly specified in the paper, so it is made
    configurable as an implementation choice (uncertainty noted in §13).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

# pandas is only referenced in type hints here (strings under `from __future__ import annotations`),
# never at runtime — so the serving image, which imports this module, does not need pandas.
if TYPE_CHECKING:
    import pandas as pd

# 5-stage coefficients (paper Table 4). Maps both 'Tam' and 'Tamb'.
ALPHA_5STAGE = {
    "T10": -4.390,
    "T20": -2.116,
    "Tam": -1.929,
    "Tamb": -1.929,
}


def estimate_days_left(ripening_index, storage_group: str) -> float:
    """Paper Eq(2): α × (index − 5). index is 1-5."""
    alpha = ALPHA_5STAGE[storage_group]
    return float(alpha * (np.asarray(ripening_index, dtype=float) - 5))


# ─── Days-to-target (service D-day) ────────────────────────────────────────
# The backend's `days_to_target` is days until the CURRENT stage advances to the
# USER'S chosen `target_stage` — not the paper's fixed days-until-stage-5. It is
# the same linear model, just with the endpoint parameterised:
#     days_to_target = α × (predicted_stage − target_stage)
# With target_stage == 5 this collapses back to estimate_days_left exactly, so the
# service can honour CLAUDE.md §1 (peak = stage 4) by having the caller send
# target_stage=4, without the paper-reproduction code changing meaning.
def estimate_days_to_target(predicted_stage, target_stage, alpha: float,
                            clip_negative: bool = True) -> float:
    """Generalised shelf-life: days for `predicted_stage` to reach `target_stage`.

    alpha is the per-storage days-per-stage slope (negative, e.g. ALPHA_5STAGE[g]
    or alpha_from_temp(t)). If the fruit is already at/past the target the raw
    value is negative; clip_negative maps that to 0.0 ("now").
    """
    d = float(alpha) * (float(predicted_stage) - float(target_stage))
    return max(d, 0.0) if clip_negative else d


# ─── Continuous temperature → α  (finalised — CLAUDE.md §8) ───────────────
# The user provides a continuous temp_celsius but the paper only fits α at the
# two controlled storage temperatures. The re-fit anchors show |α| going from
# 4.003 (10°C) to 1.750 (20°C), ratio Q10 = 4.003/1.750 ≈ 2.287, i.e. ripening
# rate roughly 2.29x per +10°C — the textbook produce-respiration behaviour.
# So we interpolate ln|α| LINEARLY in temperature (log-linear / Arrhenius-like),
# giving a smooth, physically-motivated curve that hits both anchors.
#
# Q10 falls out of the two anchors below automatically — don't hard-code it
# separately. If the anchors are ever re-fit again, change ONLY this tuple.
_TEMP_ANCHORS = ((10.0, 4.003), (20.0, 1.750))   # (°C, |α|), re-fit anchors
TEMP_CLAMP = (10.0, 30.0)   # stay within/just past the data; extrapolation is unvalidated


def alpha_from_temp(temp_celsius: float) -> float:
    """Log-linear (Q10≈2.287) map from temperature to α (negative,
    days-per-stage), matching ALPHA_5STAGE's sign convention.

    Input is clamped to TEMP_CLAMP: outside the two anchor temps the curve is an
    extrapolation the dataset does not back up, so we refuse to run away with it.
    """
    (t_lo, a_lo), (t_hi, a_hi) = _TEMP_ANCHORS
    t = min(max(float(temp_celsius), TEMP_CLAMP[0]), TEMP_CLAMP[1])
    k = (math.log(a_hi) - math.log(a_lo)) / (t_hi - t_lo)   # d(ln|α|)/d°C
    magnitude = a_lo * math.exp(k * (t - t_lo))
    return -magnitude


def shelf_life_loss(estimated, actual) -> np.ndarray:
    """Paper Eq(4): |estimated − actual|."""
    return np.abs(np.asarray(estimated, dtype=float) - np.asarray(actual, dtype=float))


def derive_actual_days_left(df: pd.DataFrame,
                            censored: str = "drop",
                            clip_negative_to_zero: bool = True) -> pd.DataFrame:
    """Derives the actual days_left for each observation.

    end_day = the Day of Experiment on which the sample first reached label 5 in the
    5-stage index.
    actual_days_left = end_day − Day of Experiment.

    censored: how to handle samples that never reached label 5 (right-censored)
      - 'drop'  : actual=NaN → excluded from shelf-life evaluation (default; the resulting
                  selection bias is noted in the report)
      - 'keep'  : leaves actual=NaN as-is (handled by the caller)
    clip_negative_to_zero: clips observations past stage 5 (negative values) to 0
      (paper: 'the final stage is forced to 0').
    """
    if censored not in {"drop", "keep"}:
        raise ValueError("censored must be 'drop' or 'keep'")
    end = (df[df["label"] == 5]
           .groupby(["Storage Group", "Sample"])["Day of Experiment"].min()
           .rename("end_day"))
    out = df.merge(end, on=["Storage Group", "Sample"], how="left")
    out["is_censored"] = out["end_day"].isna()
    out["actual_days_left"] = out["end_day"] - out["Day of Experiment"]
    if clip_negative_to_zero:
        out["actual_days_left"] = out["actual_days_left"].clip(lower=0)
    if censored == "drop":
        out.loc[out["is_censored"], "actual_days_left"] = np.nan
    return out


def evaluate_shelf_life(df_eval: pd.DataFrame, index_col: str) -> dict:
    """Shelf-life estimation loss, overall and per storage group (paper Table 6 method).

    df_eval required columns: 'Storage Group', 'actual_days_left', index_col (the
    predicted/actual stage to use).
    Rows where actual_days_left is NaN (censored/undefined) are excluded.
    Returns: {'overall': {mean,std,n}, 'by_group': {grp: {...}}}
    """
    d = df_eval.dropna(subset=["actual_days_left", index_col]).copy()
    est = np.array([estimate_days_left(r[index_col], r["Storage Group"])
                    for _, r in d.iterrows()])
    loss = shelf_life_loss(est, d["actual_days_left"].values)
    d = d.assign(_loss=loss)

    def agg(sub):
        return {"mean_loss_days": float(sub["_loss"].mean()),
                "std_days": float(sub["_loss"].std(ddof=0)),
                "n": int(len(sub))}

    return {
        "overall": agg(d),
        "by_group": {g: agg(sub) for g, sub in d.groupby("Storage Group")},
    }
