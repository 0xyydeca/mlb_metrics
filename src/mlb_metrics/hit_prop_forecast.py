"""Contract-aligned 1+ hit prop forecasts (research-only).

Separates:
- P(qualify) = P(starting lineup and plate appearance)
- P(Got_Hit | qualify)
- contract Yes payout probability ≈ P(qualify) * P(Got_Hit | qualify)

A hit rate among positive official at-bats is never used as an unconditional
contract payout probability. Missing historical executable quotes stay missing.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlb_metrics import hit_prop_research

CANDIDATE_MARKET_MID = "same_time_market_mid_baseline"
CANDIDATE_BASEBALL = "contract_rule_adjusted_hitter_hit_model"
CANDIDATE_RESIDUAL = "regularized_market_residual_prop_logistic"

FORBIDDEN_POSITIVE_AB_SOURCES = frozenset(
    {
        "positive_ab_hit_rate",
        "ab_conditional_batting_average",
        "hits_per_official_ab",
    }
)


def _clip_prob(value: float | None, *, eps: float = 1e-6) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return min(1.0 - eps, max(eps, x))


def market_mid_from_executable(
    yes_buy_price: float | None,
    no_buy_price: float | None,
) -> float | None:
    """Same-time mid from executable Yes ask and No buy (1-bid).

    Missing either side → None. Never invents a mid from closing prices or
    today's book when historical quotes are absent.
    """
    yes = hit_prop_research._number(yes_buy_price)
    no = hit_prop_research._number(no_buy_price)
    if yes is None or no is None:
        return None
    if not (0.0 < yes < 1.0 and 0.0 < no < 1.0):
        return None
    yes_bid = 1.0 - no
    if yes_bid <= 0 or yes_bid >= 1 or yes_bid > yes:
        # Crossed/locked or invalid reconstruction → missing mid.
        return None
    return _clip_prob(0.5 * (yes_bid + yes))


def contract_yes_probability(
    *,
    p_qualify: float | None,
    p_hit_given_qualify: float | None,
    source: str,
) -> dict[str, Any]:
    """Build contract Yes probability from participation × conditional hit.

    Refuses to return a lone conditional/AB rate as the contract probability.
    """
    if source in FORBIDDEN_POSITIVE_AB_SOURCES:
        return {
            "contract_yes_probability": None,
            "status": "rejected_positive_ab_source",
            "source": source,
            "p_qualify": p_qualify,
            "p_hit_given_qualify": p_hit_given_qualify,
            "note": "Positive-AB hit rates are not contract payout probabilities.",
        }
    pq = _clip_prob(p_qualify)
    ph = _clip_prob(p_hit_given_qualify)
    if pq is None or ph is None:
        return {
            "contract_yes_probability": None,
            "status": "incomplete_components",
            "source": source,
            "p_qualify": pq,
            "p_hit_given_qualify": ph,
            "note": (
                "Both P(start+PA) and P(Got_Hit|qualify) are required; "
                "conditional-only proxies are not silently promoted."
            ),
        }
    return {
        "contract_yes_probability": float(pq * ph),
        "status": "ok",
        "source": source,
        "p_qualify": pq,
        "p_hit_given_qualify": ph,
        "note": "contract_yes ≈ P(qualify) * P(Got_Hit | qualify); PA≠AB.",
    }


def baseball_proxy_components(
    *,
    game_hit_probability: float | None = None,
    p_appear: float | None = None,
    p_hit_given_appear: float | None = None,
    start_rate: float | None = None,
    final_hit_probability: float | None = None,
) -> dict[str, Any]:
    """Map reusable hitter-model outputs to contract components with provenance.

    ``Game_Hit_Probability`` is games-with-a-hit among PA appearances — a
    proxy for P(Got_Hit | appeared), not an unconditional contract payout.
    Opportunity ``Final_Hit_Probability = P_Appear * P_Hit|Appear`` is closer
    but still not identical to start+PA Polymarket rules unless p_qualify is
    aligned to starting-lineup+PA.
    """
    if final_hit_probability is not None and p_appear is None and p_hit_given_appear is None:
        # Decomposed opportunity score alone is still appearance-framed; keep
        # as incomplete unless components or start_rate supplied.
        return contract_yes_probability(
            p_qualify=None,
            p_hit_given_qualify=None,
            source="final_hit_probability_without_components",
        ) | {
            "raw_final_hit_probability": _clip_prob(final_hit_probability),
            "warning": (
                "Final_Hit_Probability is appearance-framed; without explicit "
                "P(qualify) it is not used as the contract Yes probability."
            ),
        }

    ph = p_hit_given_appear
    if ph is None:
        ph = game_hit_probability
    pq = p_appear
    if pq is None:
        pq = start_rate
    source = "opportunity_or_start_rate_x_game_hit_proxy"
    if p_hit_given_appear is not None and (p_appear is not None or start_rate is not None):
        source = "opportunity_components"
    elif game_hit_probability is not None and start_rate is not None:
        source = "start_rate_x_game_hit_probability_proxy"
    return contract_yes_probability(p_qualify=pq, p_hit_given_qualify=ph, source=source)


def binary_label_from_settlement(settlement: dict[str, Any]) -> dict[str, Any]:
    """Map settlement class to binary evaluation label; keep nonbinary distinct."""
    cls = settlement.get("settlement_class")
    if cls == "binary_yes":
        return {"y_binary": 1, "in_binary_eval": True, "reason": cls}
    if cls == "binary_no":
        return {"y_binary": 0, "in_binary_eval": True, "reason": cls}
    return {
        "y_binary": None,
        "in_binary_eval": False,
        "reason": cls or settlement.get("reason") or "nonbinary_or_unknown",
    }


def fit_isotonic_or_platt_on_train(
    y_train: np.ndarray,
    p_train: np.ndarray,
) -> Pipeline | None:
    """Fit a simple calibrated logistic map on training folds only."""
    mask = np.isfinite(y_train) & np.isfinite(p_train)
    y = y_train[mask].astype(float)
    p = p_train[mask].astype(float)
    if len(y) < 30 or len(np.unique(y)) < 2:
        return None
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=500,
                ),
            ),
        ]
    )
    pipe.fit(p.reshape(-1, 1), y)
    return pipe


def apply_calibrator(calibrator: Pipeline | None, p: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return p
    out = calibrator.predict_proba(p.reshape(-1, 1))[:, 1]
    return np.clip(out, 1e-6, 1 - 1e-6)


def fit_market_residual_logistic(
    *,
    y_train: np.ndarray,
    market_mid_train: np.ndarray,
    baseball_train: np.ndarray,
) -> Pipeline | None:
    """Regularized logistic on market mid + baseball contract proxy (train only)."""
    mask = (
        np.isfinite(y_train)
        & np.isfinite(market_mid_train)
        & np.isfinite(baseball_train)
    )
    y = y_train[mask].astype(float)
    x = np.column_stack([market_mid_train[mask], baseball_train[mask]])
    if len(y) < 50 or len(np.unique(y)) < 2:
        return None
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.5,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=800,
                ),
            ),
        ]
    )
    pipe.fit(x, y)
    return pipe


def predict_market_residual(
    model: Pipeline | None,
    market_mid: np.ndarray,
    baseball: np.ndarray,
) -> np.ndarray:
    if model is None:
        return np.full(len(market_mid), np.nan)
    mask = np.isfinite(market_mid) & np.isfinite(baseball)
    out = np.full(len(market_mid), np.nan)
    if not mask.any():
        return out
    x = np.column_stack([market_mid[mask], baseball[mask]])
    out[mask] = np.clip(model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
    return out


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def compare_candidates_on_identical_rows(
    frame: pd.DataFrame,
    *,
    label_col: str = "y_binary",
    market_col: str = "market_mid_probability",
    baseball_col: str = "baseball_contract_probability",
    residual_col: str = "residual_contract_probability",
) -> dict[str, Any]:
    """Score market / baseball / residual on the same binary-settled rows."""
    required = [label_col, market_col]
    for col in required:
        if col not in frame.columns:
            return {"status": "insufficient_data", "reason": f"missing_column:{col}"}
    work = frame.dropna(subset=[label_col, market_col]).copy()
    work = work[work[label_col].isin([0, 1, 0.0, 1.0])]
    if work.empty:
        return {"status": "insufficient_data", "reason": "no_binary_settled_rows_with_market_mid"}

    y = work[label_col].astype(float).to_numpy()
    market = work[market_col].astype(float).to_numpy()
    out: dict[str, Any] = {
        "status": "ok",
        "n_rows": int(len(work)),
        "n_dates": int(work["date"].nunique()) if "date" in work.columns else None,
        "candidates": {
            CANDIDATE_MARKET_MID: {
                "log_loss": _log_loss(y, market),
                "brier": _brier(y, market),
            }
        },
        "note": "Identical opportunities only; missing mids excluded rather than filled.",
    }
    if baseball_col in work.columns:
        bb = work.dropna(subset=[baseball_col])
        if not bb.empty and set(bb.index) == set(work.index):
            p = bb[baseball_col].astype(float).to_numpy()
            out["candidates"][CANDIDATE_BASEBALL] = {
                "log_loss": _log_loss(y, p),
                "brier": _brier(y, p),
            }
        elif not bb.empty:
            out["candidates"][CANDIDATE_BASEBALL] = {
                "status": "skipped_nonidentical_coverage",
                "n_rows_with_value": int(len(bb)),
            }
    if residual_col in work.columns:
        rr = work.dropna(subset=[residual_col])
        if not rr.empty and set(rr.index) == set(work.index):
            p = rr[residual_col].astype(float).to_numpy()
            out["candidates"][CANDIDATE_RESIDUAL] = {
                "log_loss": _log_loss(y, p),
                "brier": _brier(y, p),
            }
        elif not rr.empty:
            out["candidates"][CANDIDATE_RESIDUAL] = {
                "status": "skipped_nonidentical_coverage",
                "n_rows_with_value": int(len(rr)),
            }
    return out


def chronological_train_calibrate_predict(
    frame: pd.DataFrame,
    *,
    date_col: str = "date",
    label_col: str = "y_binary",
    market_col: str = "market_mid_probability",
    baseball_col: str = "baseball_contract_probability",
    min_train_rows: int = 50,
) -> dict[str, Any]:
    """Walk-forward: fit residual on past dates only; never peek at future labels."""
    if date_col not in frame.columns:
        return {"status": "insufficient_data", "reason": "missing_date_column"}
    work = frame.dropna(subset=[date_col, label_col, market_col, baseball_col]).copy()
    work = work[work[label_col].isin([0, 1, 0.0, 1.0])]
    work = work.sort_values(date_col)
    dates = sorted(work[date_col].astype(str).unique().tolist())
    if len(dates) < 3:
        return {
            "status": "insufficient_data",
            "reason": "need_at_least_3_dates_for_walk_forward",
            "n_dates": len(dates),
        }

    preds = []
    for i, d in enumerate(dates):
        if i == 0:
            continue
        train = work[work[date_col].astype(str) < d]
        test = work[work[date_col].astype(str) == d]
        if len(train) < min_train_rows or test.empty:
            continue
        model = fit_market_residual_logistic(
            y_train=train[label_col].to_numpy(dtype=float),
            market_mid_train=train[market_col].to_numpy(dtype=float),
            baseball_train=train[baseball_col].to_numpy(dtype=float),
        )
        cal = fit_isotonic_or_platt_on_train(
            train[label_col].to_numpy(dtype=float),
            train[baseball_col].to_numpy(dtype=float),
        )
        bb_cal = apply_calibrator(cal, test[baseball_col].to_numpy(dtype=float))
        resid = predict_market_residual(
            model,
            test[market_col].to_numpy(dtype=float),
            bb_cal,
        )
        part = test.copy()
        part["baseball_contract_probability_calibrated"] = bb_cal
        part["residual_contract_probability"] = resid
        preds.append(part)

    if not preds:
        return {
            "status": "insufficient_data",
            "reason": "no_walk_forward_blocks_met_min_train_rows",
            "min_train_rows": min_train_rows,
            "n_dates": len(dates),
        }
    scored = pd.concat(preds, ignore_index=True)
    comparison = compare_candidates_on_identical_rows(
        scored,
        label_col=label_col,
        market_col=market_col,
        baseball_col="baseball_contract_probability_calibrated",
        residual_col="residual_contract_probability",
    )
    return {
        "status": comparison.get("status", "ok"),
        "n_dates_scored": int(scored[date_col].nunique()),
        "n_rows_scored": int(len(scored)),
        "comparison": comparison,
        "tuning_scope": "chronological_training_folds_only",
        "holdout_untouched": True,
    }


def attach_forecasts_to_contracts(
    contracts: list[dict[str, Any]] | pd.DataFrame,
    *,
    hitter_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join research contracts to baseball proxies; keep missing quotes missing."""
    frame = pd.DataFrame(contracts)
    if frame.empty:
        return frame
    frame["market_mid_probability"] = [
        market_mid_from_executable(r.get("yes_buy_price"), r.get("no_buy_price"))
        for _, r in frame.iterrows()
    ]
    if hitter_features is None or hitter_features.empty:
        frame["baseball_contract_probability"] = None
        frame["baseball_forecast_status"] = "no_hitter_features"
        frame["baseball_forecast_source"] = None
        return frame

    feats = hitter_features.copy()
    if "key_mlbam" in feats.columns:
        feats["key_mlbam"] = pd.to_numeric(feats["key_mlbam"], errors="coerce")
    if "game_pk" in feats.columns:
        feats["game_pk"] = pd.to_numeric(feats["game_pk"], errors="coerce")
    frame["key_mlbam"] = pd.to_numeric(frame.get("key_mlbam"), errors="coerce")
    frame["game_pk"] = pd.to_numeric(frame.get("game_pk"), errors="coerce")
    merge_keys = [k for k in ("game_pk", "key_mlbam") if k in feats.columns and k in frame.columns]
    if len(merge_keys) < 2:
        frame["baseball_contract_probability"] = None
        frame["baseball_forecast_status"] = "hitter_features_missing_keys"
        return frame
    merged = frame.merge(feats, on=merge_keys, how="left", suffixes=("", "_feat"))
    probs = []
    statuses = []
    sources = []
    for _, row in merged.iterrows():
        proxy = baseball_proxy_components(
            game_hit_probability=row.get("Game_Hit_Probability"),
            p_appear=row.get("P_Appear"),
            p_hit_given_appear=row.get("P_Hit_Given_Appearance"),
            start_rate=row.get("start_rate"),
            final_hit_probability=row.get("Final_Hit_Probability"),
        )
        probs.append(proxy.get("contract_yes_probability"))
        statuses.append(proxy.get("status"))
        sources.append(proxy.get("source"))
    merged["baseball_contract_probability"] = probs
    merged["baseball_forecast_status"] = statuses
    merged["baseball_forecast_source"] = sources
    return merged
